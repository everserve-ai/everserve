# Multi-node: the deployed shape

Status: ADOPTED 2026-10-06 (decisions 0014–0018). Phase 2 design for an engine over several nodes on a neocloud's fleet,
installed by an operator who never reads our code. Written against `docs/RUNTIME.md` section 0
(no structural refactor between phase 1 and this). Decisions
marked **M#** were ruled the same week (0014 to 0018) and are listed at the end with their
rulings; M3 and M5 were reversed by decision 0030. The readiness API this document designed
(section 8) became the operator API's engine, node and pool objects (decisions 0039 and 0042):
`docs/RUNBOOK.md` and `docs/api/openapi.json` are the truth about its routes.

## 0. What phase 1 fixed and what this adds

Phase 1 proved the mechanism on one node: a rank dies or leaves, the engine continues at stock
speed with every session's output identical (`bench/runs/2026-10-06-exact-replay-4gpu/`). Every
piece of it was specified over (node, device) pairs so that this document adds nothing to the
worker's step path. What it adds is around the worker:

| piece | single node (phase 1) | multi-node |
|---|---|---|
| who decides a change | the node daemon | a coordinator per cluster; node daemons deliver |
| what a change is | one rank | one or many ranks at one epoch (a node's ranks together) |
| where the mirror is | the rank's own node, shared memory | the same, replicated to the node that will restore it |
| who the spare is | a GPU on the same node | a device anywhere the topology allows, warm or on demand |
| which collectives | NVLink inside the node | the same inside a domain; fabric collectives across |
| how processes find each other | a directory of files | the cluster's API (Kubernetes), files for drills |
| who the operator talks to | a socket | a chart, a policy document, the operator API |

The rule that keeps the worker unchanged: **a worker sees members, epochs, a local store and a
collective handle. It never sees a node, a fabric or a cloud.** Everything in this document lives
in the controller, the store and the policy, behind interfaces the worker already has.

## 1. The topology model

In phase 1 a member is `(node, device)`. The deployed model has four nouns, and every later story,
including mixed hardware, is a change to what these nouns carry, not to who uses them.

- **Device.** One GPU: an identity (UUID), a *class* (`h100-80`, `a100-40`, `l40s-48`, …), its
  memory and compute figures, and the *domain* it belongs to.
- **Domain.** The set of devices that reach each other at full speed: a node's NVLink now, an
  NVL72 rack later. Inside a domain the peer-to-peer slot collective applies; across domains
  the fabric does.
- **Link.** How two domains reach each other: the fabric kind (InfiniBand, RoCE, Ethernet) and
  its bandwidth and latency class.
- **Position.** One rank of one engine: what it needs (shard bytes, KV bytes at the configured
  capacity, which other positions it exchanges with and how much) and what it is doing now
  (serving on device D, or vacant).

The controller holds a `Topology` built by a provider: the node daemon reports its own devices
from NVML and its fabric from the host (phase 2); the Kubernetes backend reads the same facts from
node labels and the device plugin. `Fit(position, device) -> {ok, degrade, why}` says whether a
device can hold a position and at what expected speed relative to the position's current device.
`Fit` is the only place hardware classes are compared. It is a pure function over the topology,
tested on the CPU with made-up fleets.

**Why this is the mixed-hardware story.** A mismatched spare (an A100 taking an H100's position
until the market has an H100) is `Fit` returning `ok` with a `degrade` the policy accepts, plus a
planned swap-back, which is a notice with nobody dying. Capacity-weighted placement (more heads or
experts on the larger device) is `placement.plan` taking weights from `Fit` instead of placing
evenly. Neither touches the runtime, the store or the adapter. Both are additions to two pure
functions and a policy field.

## 2. Membership: one engine, one epoch, many nodes

**Coordinator.** One process per cluster, Rust, holding a lease in the cluster's API so a second
instance takes over if it dies (**M1**). It owns every engine's membership and is the only writer
of epochs. The node daemons become its agents: they keep the worker links, the local store, local
health and local notice intake exactly as on one node, and they deliver the coordinator's changes and
return the acknowledgements. The `coord/` module in the controller was reserved for this.

**A change is a set.** `Change` carries `repeated Replacement {dead, member}` at one epoch. A rank
lost is a set of one; a node lost is a set of eight; the outcome is chosen once for the set. The
runtime's step guard, the collective swap and the mirror restore already take a member set, so
the worker applies a set the way it applies one: refuse and acknowledge, release beside
the change, swap at the boundary, restore, agree (decision 0025).

**Fencing across daemons.** Every message between coordinator and daemon carries the epoch; a
daemon refuses a change for an epoch it has already passed, and the coordinator refuses an
acknowledgement from an epoch it has already committed. A partition therefore leaves an engine
paused, never split: a worker whose daemon lost the coordinator cannot be told a new epoch by
anyone else. The step guard's existing behaviour, no agreement means no accepted step, is what
makes this safe.

**The executor on every node.** Stock vLLM's headless node shuts every local worker down when one
dies. The adapter's executor runs on every node, in a node role on the headless ones, so a death
on any node is an event there as on the head, each node's executor owns that node's slot
region and watchdogs, and the whole-node outcome is decision 0022's choice rather than the
engine's default. The stall rule's gate on the head is the executor's count of step RPCs in
flight, which another node cannot see; on a node the gate is local: each of its workers receives
every step RPC from the head's queue and answers it, so each member keeps its own step word in
its region row, set when a step RPC is received and cleared when it is answered; a member that
dies inside a step leaves its word set, and the step it cut short stays in flight on its node
until the change clears the word, reset as the head's count is. Without the node role a single
GPU death on a serving node is seen only by the survivors' failed pipeline send, about two
seconds, and stock takes the node's other workers with it. *Built 2026-10-07
(`everserve_vllm.node`, decision 0022's as-built note); run across hosts in
`bench/runs/2026-10-13-run3-drains-smoke` and in every Dynamo proof under `bench/runs`.*

**A lost node.** Nobody on a node sees that node die. The coordinator does: a daemon's link that
ends without a goodbye, or misses its heartbeats a set number of times, loses the node, and the
coordinator drives one change per engine for the node's ranks. The daemon forwards every claimed
change to the engine's clients, so an executor learns a remote loss from the coordinator's word
(decision 0023).

**Transport.** The frame protocol over mTLS TCP between coordinator and daemons, the same frames
the Unix sockets carry on a node. Certificates come from the cluster (Kubernetes secrets or the
operator's CA); the daemon identifies itself by node name. Nothing in the worker changes.

## 3. The store: local by construction, replicated by policy

The phase-1 rule holds: a rank's rows leave the GPU into shared memory on its own node and a
restore is a local copy (decision 0011; 0.2 s for 420 MB). Multi-node adds **replication between
stores**, driven by the commit stream, so that the rows are already local on the node that will
restore them when a node dies.

- Each store forwards every acknowledged commit for a replicated rank to the stores named by the
  policy: the staged rows and the step, over a `Replicator` (`store/src/replication.rs`: a Unix
  socket on one node, TCP under mutual TLS across nodes; RDMA is not built). The receiving store applies it to its own tables with the same write-step
  bookkeeping, so the incremental restore (`since_step`) works on a replica.
- A worker's commit is acknowledged when its own store has it (the fence stays local and fast);
  replication is asynchronous and the replica's lag is a metric. On a node loss the replica may be
  one or two steps behind the lost store, which is the same tail the one-step fence already
  replays exactly (decision 0013). The replica's watermark says where.
- Where to replicate is policy (`replicaNodes`), and the default is the engine's own member nodes
  (decision 0030): a rank's rows sit in its siblings' host memory, over the replica's own fabric,
  and a shared spare node holds no rows until it joins. The per-position shape names its spare
  node instead, so the spare restores locally. Decode traffic is 0.2 MB per step per rank;
  prefill bursts of hundreds of megabytes per rank are what the fabric is for.
- On a node loss the coordinator names, in the change, the stores holding each lost rank's rows;
  the spare restores from one of them, local or over the wire, and replays the tail.

The worker's `CommittedKV` is untouched. The store gains one module and one policy field.

## 4. Collectives across the fabric

Inside a domain nothing changes: the slot collective in its peer-to-peer variant, graph-captured,
reading membership on every replay. Across domains the engine's traffic is pipeline send and
receive, or expert all-to-all, over the fabric through NCCL, and NCCL's handle is baked into any
captured kernel. So:

- Every collective is classified by **scope**: domain or fabric. The `Collective` already
  dispatches by tensor shape; it dispatches by scope too, and a fabric collective is a handle that
  reads the member set at call time and is never captured.
- The first multi-node engines run their fabric collectives outside the captured graphs, which is
  how vLLM's piecewise capture already treats some operations (**M2**). For pipeline send and
  receive that is stock vLLM's own behaviour, so it costs nothing against stock; the launch per
  collective per step is the price only where a captured collective would otherwise have run,
  tensor parallel over the fabric or expert all-to-all, and it is what the phase-2 kernel removes.
- A captured collective group cannot span the fabric until the engine has a fabric path, so a
  replacement must sit in its tensor-parallel group's domain; when it cannot, the whole group
  moves to the spare node in one change (decision 0022, `Engine.fabric_collectives`).
- The endgame is a fabric variant of the slot collective: the same push protocol as the
  peer-to-peer kernel, with the push over the fabric by GPU-initiated RDMA (the mechanism DeepEP
  uses for expert all-to-all). Same `SlotCollective` contract, third variant, captured like the
  others. Phase 2, after the two-node run and before a design partner (decision 0015); it changes
  a kernel, not a caller.
- Shadow groups for cross-node member sets are pre-formed as on one node, since decision 0035
  as the coordinator's cover plan, formed while the engine serves; the handle swap is the same
  pointer swap.

## 5. Spares in the deployed shape

The spare pool (decision 0007) becomes a set of devices with state, anywhere in the topology:

- **Warm:** holding a position's shard, KV allocation, captured graphs and shadow groups. Covers
  faults; one per position it stands for.
- **On demand:** a device that loads a position's shard when a notice arrives, from host memory
  or NVMe in seconds. Covers notices; one device can stand for many positions.
- **Mismatched:** either of the above on a device of another class, allowed by policy with a
  maximum degrade, swapped back by a planned change when a matching device is free.

**A pool belongs to the fleet (0031; built as the operator API's pools, decision 0042).** Spare capacity is described by the
engine shape it serves (model, layout, compile key) and what may join it (device class, GPU count,
label, domain), not by engine and node name: a node that matches a pool and holds no position is
its capacity, so a scheduler refills the pool by starting a node with the daemon on it, and
readiness reports each pool's deficit. Maintenance is cordon, drain, service, uncordon.

**Warm is a property of an attachment.** A spare is warm for one engine's position, never in
general: its pre-formed groups formed and warmed with one collective of every kind the engine
issues on them, and the engine's runtime kernels compiled through vLLM's own kernel warm-up (the
pipeline exchange stubbed on a stage past the first), both before it registers for that engine. A
spare registers for an engine only once it is warm for it, so a spare that stands ready for several
engines (a shared allotment, or a resident spare under decision 0028) warms and registers per
engine, and nothing reads a process-wide flag. A spare standing for any position is warm in its
communicators at standing; its kernels are warmed at the take, which builds the stage's worker and
pays the warm-up, named in the record (0028, amended). Warming kernels ahead of a
take, by pre-attaching a slot to positions, is designed as an opt-in trade and not built. The two-node dry run (2026-10-09,
`bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass9/NOTE.md`) read the first step after a resume at
about 0.6 s on a spare whose kernels compiled in that step and 0.06 s on a warm one; the record's
first-step time is what proves warmth.

**A spare for any position** (decision 0030) is the shared pool's unit: a slot without a rank,
holding the model in host memory and no shard, assigned a lost rank by the coordinator at the
change (`Change.take` names it by node and device), warmed for every position it is eligible for
at its registration, held to its position's compile pair at the join.

A spare node for a node-level loss is a set of warm or on-demand spares for that node's
positions. For a 70B engine at TP4xPP2 a spare node can hold both nodes' shards warm; for a 671B
engine it holds one node's warm and the other's on demand, and the policy says which.

The pool is a controller object with a Prometheus view (`everserve_spares{engine,position,state}`)
and it is what the readiness API below reads.

**Spare processes on another node are the daemon's.** The phase-1 spare was launched, bootstrapped
and joined over the executor's host-local pipes and shared-memory queues, none of which reach
another host; the two-node smoke (2026-10-07) found the head holds no handle to a remote node's
processes at all. So the daemon owns the spare processes on its node: the policy's engine entry
says what to preload and how to start it, and the daemon launches, watches and stops them for
any engine, the way it already registers and delivers to them. The join is an opaque exchange:
when a change gives a spare to an engine, the adapter on the executor's side produces a join
payload and the adapter on the spare's side consumes it, and the coordinator and daemons carry
bytes they never read. For vLLM the bytes are its queue handles; for another engine, whatever its
scheduler needs. One control plane, one wire, one engine-specific thing per engine. The
single-node spare moves under the same mechanism, so there is one. *Landed 2026-10-07 as
decision 0024:* the policy's engine entry places spares (`spares`: node, device, rank) and
names the launch (`spare_launch`: a command with `{engine}`, `{rank}`, `{device}`, `{node}`,
`{socket}` filled in, and its environment); the coordinator sends each daemon its node's
`LaunchSpare`s when it links; the daemon's supervisor runs, watches, restarts with a backoff and
stops them; `Join` is the opaque payload on the wire, routed by the coordinator from the engine's
client to the rank's process and back to the clients. The daemon's record carries the host other
nodes reach it at (`--advertise-host`) for the workers' rendezvous across nodes.

## 6. The outcomes, completed

REPLACE, SHRINK and REBUILD stand (decision 0004). Multi-node needs the fourth:

- **GROW.** A shrunken engine adds ranks when capacity returns: a new member set larger than the
  current one, shadow groups formed for it at the time the capacity appears, units redistributed
  by `placement.plan`, a change at a boundary with nobody dying. It is SHRINK's inverse on the
  same machinery and it is what makes a spot tier that shrinks under a crunch recover afterwards.
  Its precondition is the same as SHRINK's: a model whose units can move (the per-architecture
  port), or REBUILD-style re-formation on stock model code.

The policy's `outcomes_in_order` gains it; the runtime gains a join of more than one rank.

## 7. Notice sources and the degradation order

`notice/` holds one module per source: the clouds' metadata endpoints (AWS, GCP, Azure spot
notices), Kubernetes node conditions and taints, and the neocloud's own reclaim call, a webhook
into the coordinator. All produce the same `Notice` with a deadline. The neocloud reclaiming its
own tier is the common case and the one with the most freedom: it chooses the node and the
warning.

Under a capacity crunch the policy states the order: spares first (REPLACE), then SHRINK with
admission to new sessions closed, then drain at step boundaries, releasing a node only when it is
empty. Sessions end only in the last stage and only if the operator releases before the drain
completes. The order is a policy field, not code.

## 8. The readiness API: what the neocloud's scheduler reads

The coordinator serves, per engine and per node:

- can this node be reclaimed now, and with which outcome;
- the expected pause and the minimum notice for that outcome;
- the replica lag and the spare state behind it;
- a drain of a node by a deadline, which is a notice from the operator.

As built (decisions 0039 and 0042; `docs/RUNBOOK.md`, `docs/api/openapi.json`) these are the
operator API's objects, served on the coordinator's own TLS listener with every request
authenticated, not beside `/metrics`: an engine's status (`GET /v1/engines/{engine}`) carries,
per node holding a member, whether it can be reclaimed, the outcome the policy would take, the
minimum notice and the expected pause; a node's status (`GET /v1/nodes/{node}`) its devices, the
ranks on it and its drain; `GET /v1/nodes/{node}/release` whether the node may be given back now;
and a drain is the node's spec, written by `PUT /v1/nodes/{node}` (there is no `/v1/drain`). A
Kubernetes node condition and a custom resource per engine are not built; the coordinator keeps
its own state across a failover in `EverserveState` objects (decision 0036). This is the interface
the neocloud's scheduler uses to pick a victim node and how long to warn it. It reads state the
coordinator already holds; it is not a second source of truth.

## 9. Installation at a neocloud

Built, as `deploy/kubernetes/everserve` (run on managed Kubernetes without GPUs in
`bench/runs/2026-10-13-k8s-e2e` and under the Dynamo proofs with them):

- The Helm chart: the node daemon and the store as a DaemonSet, the store's budget sized by the
  chart's values; the coordinator as a Deployment of two holding a Kubernetes Lease, the standby
  taking it over; the policy as a ConfigMap; each node issued its certificate by the coordinator
  (decision 0037). The daemons follow the coordinator through its Lease record, and an engine's
  pods find their node's daemon through the pod contract's host directories
  (`everserve.podcontract`); the file discovery backend stays for drills and bare metal.
- The engine's pods carry the adapter as a vLLM plugin. Under NVIDIA Dynamo, `integrations/dynamo`
  renders the instance's pods, the engine's policy entry (its shape only) and the chart's values;
  the policy, placement and state reach every process over its daemon link, so no pod mounts a
  policy file.
- Topology from the daemons' hellos (each node's GPUs from `nvidia-smi`) and the cluster's Nodes
  (labels and cordons, read through the API server); nothing hand-written.
- Observability: every process serves `/metrics` and `/healthz`; a Grafana dashboard and
  alerting rules ship (`deploy/grafana/`); the operator API is the operator's view.
- Prebuilt kernels: the engine image (`deploy/images/engine`) builds them ahead of time against
  its runtime's torch, so a worker never compiles; the wheel carries the sources
  (`everserve/_csrc`) for a host whose torch or CUDA differs.

The operator installs the chart, points the policy at the engines, and sets the notice source.
No code.

A GPU tier that is not on Kubernetes installs the same binaries as systemd units
(`deploy/baremetal/`, *landed 2026-10-06*): one policy file, certificates from the operator's CA
or from `everserve-controller certs`, environment files per role, and a smoke test of the whole
control plane over TLS on one machine that the gate runs. The two-node run uses it.

## 10. The interfaces, named

| interface | lives in | what it hides | mixed hardware touches it? |
|---|---|---|---|
| `Topology` provider | controller | where devices and fabrics come from | yes: device class and figures |
| `Fit` | controller | whether a device can hold a position and how well | yes: the whole story |
| `SparePool` | controller | warm, on demand, mismatched; where spares are | yes: a policy field |
| `Coordinator` ↔ daemon protocol | proto, controller | one epoch over many nodes | no |
| `Replicator` | store | a Unix socket on a node, TCP under mutual TLS across nodes | no |
| `NoticeSource` | controller | each cloud's signal | no |
| `Discovery` | controller | files for drills and bare metal; the Lease record and the pod contract on Kubernetes | no |
| `SlotCollective` variants | core/csrc | peer-to-peer, host, fabric | no |
| `CommittedKV` | core | unchanged since 0011 | no |
| the operator API | controller | what the scheduler sees | no |

Three rows say yes, and all three are controller-side pure functions and policy. That is the claim
that mixed hardware is something we add, not something we rebuild for.

## 11. Order of work and the two-node run

Each a landing or a few, each gated, the CPU-testable ones first:

1. Coordinator and daemon split with the mTLS frame transport and epoch fencing across daemons;
   multi-daemon tests on one machine. Rust. *Landed 2026-10-06* (`coord.proto`, `transport`,
   `lease`, `coord`, `node` as agent; a change delivered across two daemons, a stale epoch
   refused, a report held without a coordinator, a restart resuming at the daemons' epoch).
2. `Change` as a set; the runtime's join of many ranks; the adapter applying a set at a boundary.
   Python, CPU tests, then a GPU drill. *Landed 2026-10-06 on the CPU side*: `RankLost` and
   `Change` carry a set of ranks, the coordinator claims one epoch for the set and delivers to
   every spare that takes one of them, the daemon holds several spares, the runtime's
   `on_change` moves to the survivor set or to the set with each lost rank's spare in its place
   (`replacement(rank)` ids, member sets pre-formed per loss), and a node-loss test drives two
   ranks and two spares through one change. What remains is the adapter's executor joining
   several spare processes at one boundary, which the two-node run's GPU session lands.
   *Its first GPU run (the 2026-10-06 commit-path battery) found the executor still pre-forming a
   replacement set per rank while launching one spare, so the sets naming the other spares
   waited out the rendezvous; fixed the same day: `sets_to_preform` takes the launched spares'
   ranks, and `check_formable` refuses a set nobody can play before any rendezvous.*
3. Store replication with the write-step bookkeeping on the replica; lag metric. Rust, CPU tests
   against two stores on one machine. *Landed 2026-10-06*: `Replicate` names a rank's targets,
   the source reads back what each commit wrote and carries it to every target on its own task
   (reconnecting, a bounded queue, lag and drops as metrics), the replica applies it with the same
   per-slot write steps and refuses a stale step, a second listener (Unix or TLS) takes peers,
   and the transport moved to the wire crate for it. The daemon telling its store the targets,
   from the coordinator's policy, is item 4's work with the spare pool.
4. Topology, `Fit`, the spare pool as an object, the policy fields. Rust, CPU tests. *Landed
   2026-10-06*: the daemon's hello carries its devices (`nvidia-smi`) and its store's replica
   address; the coordinator holds a `Topology`, fits a spare to its position by memory and
   class (`controller/src/topology`), admits a mismatched class only under `allowMismatched`
   within `maxDegradePercent`, reports the pool with each spare's expected speed, and tells
   each member's daemon the replication targets its store takes (since decision 0030 the
   engine's `replicaNodes`, its own member nodes unless named, other than the rank's own), sent
   once per change of targets. The pool's Prometheus view and the readiness API
   are item 7.
5. Fabric-scoped collectives outside the graphs; cross-node shadow groups. Adapter, GPU.
6. Kubernetes discovery and the first notice sources, with fakes in CI. *Notice sources landed
   2026-10-06* (`controller/src/notice`): the daemon watches one source (`aws` spot
   interruptions over IMDSv2, `gcp` preemption, `azure` Scheduled Events, `kubernetes` node
   taints: the autoscaler's deletion time, a cordon as the operator's drain) and reports a
   `NodeNotice`; the coordinator drains the node as a `PUT /v1/nodes/{node}` with `spec.drain`
   does. Each source is a pure
   parser plus one shared HTTP client, tested against fakes; a source reports once and again only
   on an earlier deadline. On Kubernetes the daemons follow the coordinator's Lease record and
   the engine's pods reach their daemon through the pod contract (section 9); a Kubernetes node
   condition and a custom resource per engine are not built.
7. The readiness API. *Landed 2026-10-06, before item 6*, because it reads state the coordinator
   already holds and needs no new dependency, as reads on the coordinator's metrics port;
   *superseded 2026-10-14 by decisions 0039 and 0042*: engines, nodes and pools as objects with a
   spec and a status on the API's own authenticated listener, a drain as a node's spec
   (section 8; `controller/src/readiness` is the engine's status as the API shows it). The
   rules stand: a node is reclaimable when a spare stands for every rank on it or SHRINK is
   allowed, within the policy's concurrent losses; a drain is a notice with reason `OPERATOR` to
   every rank on the node and one change for the set at the deadline; the expected pause is the
   last one measured for that outcome, absent until measured; the pool is the
   `everserve_spares{engine,state}` gauge and the API's pool objects.
7b. The node role: the adapter's executor on the headless nodes, with the per-node in-flight gate
   (section 2). Adapter and CPU tests built 2026-10-07; run across hosts in the drains smoke
   (`bench/runs/2026-10-13-run3-drains-smoke`) and the Dynamo proofs.
8. **The two-node run** (roadmap step 5), as first shaped: two 4xH100 nodes serving a 70B model
   at TP4xPP2, a third 4xH100 node as the spare node, a notice on one serving node, the spare
   node taking its four positions, sessions identical; about $42 per hour for twelve H100s, two
   hours. *Superseded by decision 0030 (2026-10-09)*: the shape became two serving nodes and one
   shared spare node with the weights resident in host memory, nothing preloaded, after the
   two-node smoke found ordinary RunPod pods' network a thousandth of a fabric
   (`bench/runs/PLANNED/p2-two-node/`, its pre-registration and dry runs). The run was not
   rented in either shape; the multi-node proofs that ran are the drains smoke
   (`bench/runs/2026-10-13-run3-drains-smoke`) and the Dynamo proofs
   (`bench/runs/2026-10-14-dynamo-proof` and after).

Items 1 through 4, 6, 7 and 7b landed on 2026-10-06 and 2026-10-07. Of item 5, the pipeline case
runs outside the captured graphs as stock vLLM does and ran across nodes in the Dynamo proofs;
tensor parallel over the fabric and the fabric slot kernel are not built.

## 12. Decisions

Ruled 2026-10-06: M1 a leased process (0014); M2 outside the graphs first, the fabric slot kernel
in phase 2 before a design partner (0015); M3 spare nodes only (0016); M4 Kubernetes first (0017);
M5 three 4x H100 nodes (0018). *Ruled again 2026-10-09 by decision 0030, after the two-node dry
run closed REPLACE (`bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass11/NOTE.md`):* M3 is reversed for the shared-pool shape, a replica's rows replicate
to its own sibling nodes' stores and a shared spare holds no KV until it joins; M5's rental shape
is superseded, two serving nodes and one shared spare node with the weights resident in host
memory, nothing preloaded. The options as they were put:

- **M1. Coordinator placement.** A separate leased process (recommended: one thing to run, maps
  to a Kubernetes Deployment, failover is the lease) versus election among node daemons (fewer
  processes, more protocol).
- **M2. Fabric collectives for the first multi-node engines.** Outside the captured graphs
  (recommended: no new kernel on the path to the two-node run; a launch per fabric collective per
  step is the cost) versus the fabric slot kernel first (phase 3's work on phase 2's path).
- **M3. Replication targets.** To spare nodes only (recommended: bandwidth proportional to spares,
  not to nodes) versus to every node (any node can restore any rank; N times the traffic).
- **M4. Discovery for the pilot.** Kubernetes first (recommended: it is what a neocloud runs; the
  file backend stays for drills) versus files first with Kubernetes after the pilot.
- **M5. The two-node run's shape.** Three 4xH100 nodes, 70B at TP4xPP2 plus a spare node
  (recommended: under $100 and every cross-node piece exercised) versus two 8xH100 nodes
  (closer to the frontier shape, three times the cost, nothing extra exercised).
