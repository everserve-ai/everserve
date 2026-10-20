# everserve-controller

The control plane as a library plus a thin binary. One daemon per node, one coordinator. Rust, because it runs unattended at a
customer and its job is exactly the property the product sells.

| module | owns |
|---|---|
| `node` | the per-node daemon, the coordinator's agent: the worker links, undecided registrations, notice intake, delivery of changes, the spares it launches and supervises, the start-up replay a spare reads (decision 0044), the local store's record and its compile cache |
| `coord` | the coordinator: the one writer of every engine's epoch (decision 0014); the transition machine, cover plans, spare pools, placement, roles (which process serves a rank), capability checks, the audience of each engine's facts, takeover and resume after a failover, leaks, the views the API serves |
| `api` | the operator API (decisions 0039, 0042): authentication and authorization, the objects, routes, the watch feed, the node certificate endpoint |
| `operator` | the API's client, what `get`, `watch`, `cordon`, `drain` and `check` call |
| `readiness` | an engine's status as the API shows it: per node whether it can be reclaimed, with which outcome, the minimum notice and the expected pause |
| `spec` | desired state and the spec store (decision 0036): a file, or `EverserveState` objects on Kubernetes |
| `status` | conditions and the event log of each resource |
| `lease` | the coordinator's lease: a file lock, etcd or a Kubernetes Lease, under a term that fences every other |
| `topology` | devices, domains, positions and `Fit`, the one place hardware classes are compared |
| `notice` | one module per cloud's preemption signal, and the Kubernetes taints |
| `kubernetes` | the API server as the controller reaches it: the client, the cluster's Nodes, the objects it keeps |
| `issuer`, `enroll`, `certs` | a node proves its name to be issued its certificate (decision 0037), the node's side of it, and the `certs` command for a deployment without a CA |
| `check` | `everserve-controller check`: is the control plane ready for the policy's engines |
| `discovery` | the `Discovery` trait and the file backend; the daemon publishes its record |
| `metrics` | what an operator watches; names are stable |
| `cli` | the commands; a signal cancels the role and in-flight changes get a graceful period |
| `config` | every flag with its environment variable, listed in one module |
| `policy` | the policy document: load, validate, choose the outcome for an event |
| `http`, `clock` | the HTTP/1.1 client the controller's own callers share; the process's clock and the rule for it |

The wire schema and the transports are the wire crate's (`wire/`). Speaks to an adapter only over
the signal protocol (`proto/`). Never touches engine state. The drill drivers in `bench/harness`
are clients of this daemon and its API, as an operator's tools are.

## Settings in the policy

An engine entry may set `releaseBoundMs`: how long a step boundary waits for a refused group's release before leaving it behind as leaked (decision 0025), the operator's trade of pause against memory; absent, the adapter's default of 10 s. The store's own settings are flags: `--replication-queue-bytes` (the table budget unless given) and `--replication-reply-timeout-s` (30 unless given).

## Identity

A daemon is the node its certificate names (decision 0026): the coordinator refuses a hello for any other node, and a replica takes a rank's rows only from the node the coordinator named, over a link whose certificate is valid for that node. `everserve-controller certs` issues each node's certificate with the node's name; an operator's own CA must do the same.

## Compile cache

An engine's compile cache is part of each of its positions, as its weights are (decision 0029): produced by one warm-up, staged to every node that may host a position, every rank's subtree included, so no node autotunes kernels by itself. Each process reports the key and digest of the cache it runs with when it registers, after its warm-up. The coordinator holds every process of an engine to one key, the compiled artifact its first process reported, and holds each spare to the pair (key and digest) of the serving rank it stands in for, since on a pipeline engine each stage compiles its own graph and a rank's digest is its own: a spare on another pair is not admitted to the pool, or leaves it when its rank registers with another, both pairs in the log; a serving rank on another key is reported per rank. The engine's status carries the key, each process's pair, and `compile_cache_consistent`, which is true only when every serving rank reported the key and every spare in the pool reported its rank's pair; a process that reported none leaves the engine inconsistent, since a check no process answers is no check (over no pairs it would read vacuously true). A scheduler does not run an exact change on an inconsistent engine. Two launches on one cache match row for row; two autotunes of the same kernel need not, so the engine runs with Inductor's deterministic mode (`inductor_compile_config: {"deterministic": true}`), under which no reduction kernel's configuration is chosen by timing.

## Running the control plane

One coordinator per cluster, holding a lease; one daemon per node, linked to it. On one node both
speak over Unix sockets; across nodes the daemons reach the coordinator over TCP under mutual TLS
(`--tls-cert`, `--tls-key`, `--tls-ca` on both sides; plaintext TCP is refused).

```sh
everserve-controller coord --policy docs/examples/policy-pilot.json \
    --listen unix:/run/everserve/coord.sock --lease /run/everserve/coord.lease \
    --metrics-addr 127.0.0.1:9107
everserve-controller node --coordinator unix:/run/everserve/coord.sock \
    --socket /run/everserve/node.sock --discovery-dir /run/everserve/discovery \
    --mirror-socket /run/everserve/mirror.sock --metrics-addr 127.0.0.1:9108
```

A daemon without a coordinator serves its links and holds every report until one is given; a
second coordinator on the same lease stands by until the first releases it. A daemon heartbeats
its link every `--heartbeat-ms` (500) and says goodbye on an orderly stop; the coordinator takes a
node as lost when its link ends without a goodbye or misses `--node-loss-misses` (3) heartbeats,
and changes the node's ranks as one set (decision 0023). Every flag has an
environment variable, listed in `src/config/mod.rs`. `deploy/baremetal/` is the install: systemd
units with environment files, `everserve-controller certs` for a deployment without a CA, and a
smoke test of three nodes over TLS on one machine.

The daemon's hello carries the node's GPUs (`nvidia-smi`'s listing: index, UUID, product name,
memory) and, with `--mirror-replica host:port`, the address at which this node's store takes
replication. From those the coordinator keeps a `Topology`, admits a spare by its `Fit` to the
position (memory decides whether, class decides the speed; a slower class only under a policy
with `allowMismatched` and within `maxDegradePercent`), and tells each member's daemon which
stores its rank replicates to: the engine's `replicaNodes`, its own member nodes unless the policy
names others, other than the rank's own node (decision 0030), so a rank's rows sit in its siblings'
host memory and a shared spare restores from there. `docs/examples/policy-two-node.json` is the
per-position shape, eight members on two nodes at TP4xPP2 and a third node as the spare node, which
it names as its replica node; `policy-two-node-resident.json` is the shared-spare shape, which
names none. On a REPLACE under that shape the coordinator points the replica holding each lost rank's rows at the spare's node before the change goes out and releases it at the commit, so the carrier's snapshot carries the rows and the spare restores from its own store once it reads the rank held and caught up (decision 0030).

## Spares the daemon owns

A `spares` slot without a `rank` places a spare standing for any position of the engine (decision 0030): it holds the model in host memory, no shard, and takes the rank a change names in `Change.take`; the daemon launches it with `{rank}` empty, the coordinator keeps it in the pool by node and device (`anySpares` in the engine's status), gives it a lost rank at the change when no spare preloaded for that rank exists, and holds it to its position's compile pair at the join rather than at its registration: the take carries the pair the position's serving rank last reported, and the spare refuses the change, naming both, when its own differs after it attaches.

An engine entry in the policy may place its spares (`spares`: node, device, the rank each stands
in for) and name how to start one (`spareLaunch`: a command and environment with `{engine}`,
`{rank}`, `{device}`, `{node}` and `{socket}` filled in). When a daemon links, the coordinator
sends it the spares placed on its node; the daemon launches them, restarts one that exits with a
backoff from 1 s to 30 s, and stops them with `SIGTERM` then `SIGKILL` when it stops. The command
is the adapter's spare entry point; `docs/examples/policy-two-node-managed.json` uses a `sleep`
so the gate can run it. An engine entry without a launch keeps launching its own spares.

The join between an engine's executor and a process that took a position is a `Join` signal
with an opaque payload: the executor's client sends one for the rank, the coordinator routes it
to the daemon holding that rank's process, the process answers with its own, routed back to the
engine's clients. The control plane never reads the bytes (decision 0024).

`--advertise-host` puts the address other nodes reach this node at into the daemon's record, for
the workers' rendezvous across nodes.

## The operator API

The coordinator serves engines, nodes and pools (decision 0042) on its own TLS listener
(`--api-addr`, decision 0039): every request presents an operator's or a viewer's certificate
(`everserve-controller certs --operator <name>`, named in `--api-operators` or `--api-viewers`)
or, on Kubernetes (`--api-kubernetes`), a bearer token the cluster reviews as a verb on a resource
of the `everserve.ai` group. No credential: 401; not allowed: 403. `/metrics` and `/healthz` stay
plain on the metrics port. Every figure is state the coordinator already holds to drive changes.

Each object is `metadata` (`name`, `generation` raised by each spec write, `resourceVersion`
raised by any change of the object as shown, and who wrote its spec and when), `spec` (what is
wanted) and `status` (what the coordinator observes, with conditions). `docs/api/openapi.json`
describes every route and field; `everserve-controller openapi` prints it, and a test fails when
the two differ. `everserve-controller get`, `watch`, `cordon`, `uncordon`, `drain` and `undrain`
call it with an operator's certificate or a token (`EVERSERVE_API`, `EVERSERVE_API_CA` and the
credential in the environment); a write to a node is made at the version it read, never over a
change that landed since (`docs/RUNBOOK.md`). A list is read a page at a time with
`?limit=N&continue=T` (decision 0042), each page at the first page's `resourceVersion`.

| route | what |
|---|---|
| `GET /v1/engines`, `/v1/engines/{engine}` | each engine: its spec is its entry in the policy; its status is its instance state, epoch, change in flight, members (node, class, replication targets, process), spares and spares standing for any position with their processes, lost ranks, the slot-collective choice per node, the compile key, the head, why it is down, the last change's record, the cover plan, and per node whether it can be reclaimed, the outcome the policy would take, why not, the minimum notice and the expected pause |
| `GET /v1/nodes`, `/v1/nodes/{node}` | every node whose daemon is linked, that has a spec, or that holds a member: its spec (`unschedulable`, `drain.deadlineMs`, `reason`) and its status (linked, devices, its store's replica address, each engine's ranks on it with the outcome a loss would take, conditions); what `everserve-controller check --policy --coordinator-api` compares the policy with |
| `GET /v1/nodes/{node}/release` | whether the node may be given back now: `releasable`, and `why` not, each reason in words (a rank, the head or a spare taking a position held there, rows staged onto or from it, a change in flight touching it); with `?wait_ms=N` (at most 25 s) it answers once the node may go or the wait runs out. Granted to viewers and operators as `get nodes/release` |
| `PUT /v1/nodes/{node}` `{"spec": {...}}` | writes the node's spec: cordoned (`unschedulable`), drained by `drain.deadlineMs` (a time, Unix milliseconds; a drained node is cordoned), or neither. A drain's reconcile sends a notice to every rank on the node, reason `OPERATOR`, and sets one change per engine for the set of its ranks at the deadline, unless other drains already move as many of that engine's positions as the policy's concurrent losses allow: the node's `Draining` condition is then `Waiting`. 200 with the node; 409 `DrainUnderway` when the write moves or clears a drain whose notices went out; 422 for a drain of a node holding no member; a spec written as it is held is no write |
| `GET /v1/pools`, `/v1/pools/{pool}` | every pool (decision 0031): its spec (engines, wants, device class, domain) and its status (spares standing, taking a position and warming, the deficit, conditions `Satisfied` and `ShapeAgrees`) |
| `PUT /v1/pools/{pool}` `{"spec": {...}}` | writes the pool's spec, its launcher kept; 422 when an engine is not in the policy or the engines' known shapes differ |
| `GET`, `PUT /v1/pools/{pool}/launcher` `{"command": [...], "env": {}}` | the command a node's daemon runs to launch one of the pool's spares: every node of the pool runs it, so it is a grant of its own (`pools/launcher`) |
| `GET /v1/{engines,nodes,pools}/{name}/events` | the newest 64 events of the object, oldest first: what the controller decided and why (`ChangeStarted`, `ChangeCommitted`, `RefusedRebuild`, `SpareRefused`, `DrainWaiting`, `DrainStarted`, `DrainBlocked`, `DrainDone`, `SpecWritten`, and each condition's change) |
| `POST /v1/engines/{engine}/stacks` `{"rank": 3}`, `{"node", "device"}` or `{"head": true}` | the daemon holding that process asks it for every thread's stack; 202 with the node, pid, time and the stacks (at most 1 MiB) when it answered within 5 s; 404 when no process stands there, 502 when the daemon could not |
| `GET /v1/topology` | every device the daemons reported |

A list carries `metadata.resourceVersion`; `GET /v1/{kind}?watch=true&resourceVersion=N` streams
each change after N as one JSON line, `{"type": "ADDED" | "MODIFIED" | "DELETED", "object"}`, and
a `BOOKMARK` with the current version every 30 s; without a version it starts with every object
`ADDED`. A version older than the 4096 changes kept, or of another lease term, is 410 `Gone`: list
again. A write may carry the `metadata.resourceVersion` it read, and is refused, 409 `Conflict`,
if the object changed since. Every error is `{"code", "reason", "message"}`.

Each engine carries its conditions (decision 0036): `Serving`, `Progressing` (a change in
flight), `Covered` (a loss of any one node would be replaced, not rebuilt), `Degraded` (serving
short of the policy: shrunk, a second head, a spare refused) and `Down`; each node carries
`Draining` (`Scheduled`, `Waiting`, `Blocked` or `NotDraining`) and `Degraded` (`DrainBlocked`).
A condition is `True`, `False` or `Unknown`, with a one-word reason, a message, and
`lastTransitionMs`, which moves only when its status changes; each change is also an event.
Drains of nodes sharing an engine start in deadline order, the earliest first, and a drain
whose change is refused is `Blocked` until it is withdrawn or given another deadline. The order
and the concurrent losses are per engine: one engine's drains never hold another's.

With the policy's `cover` on (decision 0035; `{"cover": {"enabled": true, "maxGroupsPerGpu": 0}}`,
off by default until every process speaks protocol 8), the coordinator plans the groups each
engine pre-forms and delivers the plan to every daemon with a process of it, a new generation
whenever the planned groups change. An engine's status `cover` shows the plan: its generation and epoch,
each set with what it is for and how many of its groups every member reported formed, the losses
it holds no set for and why, groups that failed, and groups of earlier plans still releasing. The
`Covered` condition then follows the plan: `EveryLossCovered`, `Forming` (planned, not yet
formed) or `LossUncovered`, naming the losses. A loss of the head's node, or of any rank on it,
is named beside them and not counted: it restarts the engine (decision 0045). `PlanDelivered` and `GroupFailed`
are events.

A node's spec (decision 0036) is what the operator asked of it, kept in the spec store:
`--spec-store` names a file, written whole and atomically, or `kubernetes://namespace/name` for
`EverserveState` objects the API server keeps (decision 0036), which a restarted coordinator reads and
resumes from (a stored drain resumes when the node's daemon says hello). Unset, specs are kept in
memory and lost with the process. A node notice from a notice source writes the same spec as an
operator's drain, recorded as written by the node and its source.

A node is reclaimable when losing every rank on it keeps the engine serving: a spare stands for
each of them (REPLACE) or the model is placement independent (SHRINK), and the policy's concurrent
losses are not exceeded. The expected pause is the last pause this coordinator measured for that
outcome on that engine, and absent until one was; nothing is estimated. The spare pool is also the
`everserve_spares{engine,state}` gauge.

## Notice sources

The daemon can watch where its node learns it is leaving (`--notice-source`, `docs/MULTINODE.md`
section 7). Each source answers one question, is this node leaving and when, and the daemon
reports the answer as a node notice; the coordinator drains the node exactly as an operator's
`PUT /v1/nodes/{node}` would. The operator's own reclaim call is that write and needs no source.

| source | reads | deadline |
|---|---|---|
| `aws` | IMDSv2 `spot/instance-action`, polled | the action's `time` |
| `gcp` | `instance/preempted` and `instance/maintenance-event`, polled | now plus `--notice-lead-ms` for preemption (set it to the VM's preemption notice: GCP's default is 0 s, 120 s if the VM asked for it); 60 minutes for a host maintenance that terminates the VM |
| `azure` | Scheduled Events, polled, filtered to this VM's name | the earliest `Preempt`/`Terminate`/`Reboot`/`Redeploy` event's `NotBefore`; now for one already started |
| `kubernetes` | this node's taints over the API, bearer token from `--kubernetes-token-file`, read again at every request | `ToBeDeletedByClusterAutoscaler`'s time; a cordon gets the lead with reason `OPERATOR` |

A source reports once, and again only when the deadline moves earlier: a fixed time repeats
equal, and a source without a time of its own computes a later deadline on every poll, which is
the same news. A failing endpoint is retried every `--notice-poll-ms`. `--notice-endpoint`
replaces the cloud's default, which is how the tests run every source against a fake; the TLS
connection to a real Kubernetes API server is not exercised in CI.
