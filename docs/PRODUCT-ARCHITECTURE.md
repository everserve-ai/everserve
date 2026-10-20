# Product architecture: Everserve as a neocloud runs it

Status: target architecture, 2026-10-09, amended 2026-10-12 (0033 to 0036); every status re-read
against the tree on 2026-10-19 (decisions to 0045). Every section says what is built and what is
designed; nothing here describes a design as built. Decisions 0024 to 0045 are the record behind
it, and
`docs/MULTINODE.md` is the engineering detail. This document is what an operator, a scheduler
team or a due-diligence reader starts from.

## What it is

A neocloud serves models on GPUs it sizes for peak and pays for when idle, on capacity that is
reclaimed, fails and goes to maintenance, under software it must upgrade. Without it every one
of those events is a restart: a model replica goes down for minutes, or a whole replica's worth of
GPUs is held aside to avoid it, and the sessions on it are dropped. Everserve makes the engine a
logical object over a placement that changes while it serves: one transition machine moves
positions to spares after a fault (built), moves them off a node before a reclaim, a drain or an
upgrade (partly built), and grows or shrinks the engine's width as load moves (designed, 0033),
with every session continuing from its committed position. The operator's systems decide when
and how much; Everserve decides how, and refuses, with the reason, what it cannot do safely.

## The layers

| Layer | Runs as | Job | Status |
|---|---|---|---|
| Engine adapter | a plugin inside the inference engine (vLLM) | commits each step's KV, swaps communicators at a step boundary, resumes requests exactly, builds a spare's worker at a take | built |
| Store | one per node, beside the engine | holds committed KV rows in host memory, replicates them to sibling nodes with a checksum per row, serves restores | built |
| Node daemon | one per node | registers processes, delivers changes, supervises spares, reads the node's reclaim notices, reports devices | built |
| Coordinator | one leased process per cluster, a standby beside it | membership epochs, the decision of every change, spare pools, readiness, the operator API | built, the lease and the standby included (0014, 0036: a file lock, etcd or a Kubernetes Lease; `controller/src/lease`) |
| Operator API | HTTPS on the coordinator, every caller authenticated | the one surface every integration and the console use | built (0039, 0042) |

Engine-independent parts (coordinator, daemon, store, wire) never import an engine; a second
engine is a second adapter, nothing else (`docs/ARCHITECTURE.md`). In the code these are seven
layers with five versioned contracts between them, and five identities kept apart (the engine,
its positions, the members holding them, the processes, the software identity), so that every
later product is a new transition kind, adapter or surface rather than a change through the
layers (0034).

## The resource model

Four resources, each with a desired state the operator or its systems set and a status Everserve
reports, so every integration is declarative and idempotent.

- **Engine**: one model replica, its members by node and device, its parallel layout, its
  guarantee class (exact or continuous) and its KV protection mode (continuous, on notice, none).
  *Built, in the policy; protection modes designed (0030).*
- **Pool**: spare capacity for one engine shape (model, layout, compile key), what may join it, how
  many spares it wants. *Built (0031, 0042): `GET`/`PUT /v1/pools/{pool}` and its launcher; a
  spare stands for any position of any engine of the pool's shape.*
- **Node**: its devices, its spec (cordoned, draining) and the ranks and spares it holds.
  *Built (0036, 0042); a role derived per node as one word is not reported.*
- **Change**: one transition of one engine (replace, relocate, shrink, grow), with feasibility,
  prepare, execute and observe as distinct steps, and a record of every term it took. *Replace
  built, driven by faults and drains; the transition object designed (0030); relocate, shrink and
  grow designed (0033: the machine's REPLACE path and the engine controller built).*

## How it hooks into the operator's control surfaces

Everserve reads the signals a neocloud already produces and exposes state the systems it already
runs can act on. It does not ask an operator to adopt a new scheduler.

**Kubernetes.** The daemon and store ship as a DaemonSet, the coordinator as a Deployment holding a
Lease, the policy as a ConfigMap, each node's certificate issued by the coordinator
(`docs/MULTINODE.md` section 9; the Helm chart in `deploy/kubernetes` and the bare-metal units in
`deploy/baremetal` both built). Kubernetes' own verbs are the control
surface: a cordoned node, or one tainted for deletion by the cluster autoscaler, is read by the
daemon's Kubernetes notice source and drained to the pool's spares before it goes (*built*).
Engines and pools as custom resources an operator applies like any other are *designed*; the
one custom resource built is `EverserveState`, the coordinator's own state across a failover
(0036). A pool's deficit is in its status (*built*, 0031) and is what a node autoscaler
(Karpenter, the cluster autoscaler, a provider's own) provisions against: Everserve says how many
spares of what shape are missing, the autoscaler brings nodes, and the pool's launcher starts a
spare on each (*built*); the autoscaler reading the deficit is the operator's wiring.

**Clouds' reclaim signals.** AWS spot interruption notices, GCP preemption and maintenance events,
Azure scheduled events and a plain HTTP source for anything else (*built*, `controller/src/notice`).
A notice becomes a drain at its deadline; under the on-notice protection mode the KV starts moving
at the notice (*designed*).

**Bare metal and Slurm.** The same binaries as systemd units with one policy file and the
operator's CA (*built*, `deploy/baremetal/`). A cluster manager reads the pool deficit from the API
and returns nodes by starting the daemon on them; maintenance is the API's cordon and drain.

**Routers and serving frameworks.** Everserve works below a router, not instead of one. A router
(NVIDIA Dynamo, llm-d, a provider's own) keeps choosing replicas; Everserve keeps each replica
alive. Readiness says per engine whether it is serving, paused for a change or degraded, so a
router can steer around a replica for the seconds a change takes (*readiness built; the Dynamo
integration built, `integrations/dynamo`, decision 0041; reporting an instance's state to Dynamo
designed*).

**Observability.** Every process serves Prometheus metrics and health; a Grafana dashboard ships
(*built*). Every change writes a record of its terms, its timing and its outcome (*built*, per
change); the same records as OpenTelemetry spans for the operator's tracing (*designed*).

**The console.** A thin client of the same API: the fleet's engines, pools and nodes with their
roles; each change as a timeline of its terms; the actions the API offers (drain, cordon,
uncordon, ask a process for its stacks). It has no private surface.

## The operator API

| Route | What | Status |
|---|---|---|
| `GET /v1/engines`, `/v1/engines/{engine}` | each engine's spec (its policy entry) and status: members and their processes, spares, per node whether it can be reclaimed and how, conditions | built (0042) |
| `GET /v1/nodes`, `/v1/nodes/{node}`; `PUT /v1/nodes/{node}` | each node's spec (cordon, drain by a deadline) and status (linked, devices, each engine's ranks on it, conditions); a write cordons or drains it, drains waiting their turn under the policy's concurrent losses | built (0036, 0042); the Kubernetes native cordon as the same signal is designed |
| `GET /v1/pools`, `/v1/pools/{pool}`; `PUT /v1/pools/{pool}`, `/launcher` | each pool's spec (its engines, which must share a shape, and what may join) and status (standing, taking, warming, deficit); its launcher a grant of its own | built (0031, 0042) |
| `GET /v1/{kind}?watch=true` | every change after a list's version, one JSON line each | built (0042) |
| `GET /v1/{kind}/{name}/events` | the newest events of an engine, node or pool: what the controller decided and why | built (0036) |
| `GET /v1/nodes/{node}/release` | whether the node may be given back now, and why not | built |
| `POST /v1/engines/{engine}/stacks` | a process's stacks, answered within 5 s | built |
| `GET /v1/topology` | devices | built |
| `POST /v1/changes` with `dry_run` | ask whether a transition is feasible, then execute it | designed (0030) |
| `GET /v1/changes` | every change's record | designed; the records exist per change |

## Operating it: by hand, from a console, across a fleet

Three ways in, one API underneath. None of them has a surface the others lack.

**By hand.** The operator API is HTTPS and JSON on the coordinator, so `curl` with a credential
works against every route (`docs/RUNBOOK.md`). The everyday surface is the CLI over the same
routes, subcommands of the controller binary beside `node`, `coord`, `certs` and `check`: `get`
(engines, nodes or pools, each pool with its deficit), `watch`, `cordon` and `uncordon <node>`,
`drain <node> --in <seconds>` and `undrain` (*built*, `controller/src/config/mod.rs`). A process's
stacks are a `POST` to the API; a command over the change records is *designed*. On Kubernetes
the cluster's own tools reach Everserve already: `kubectl cordon` and the autoscaler's deletion
taint become drains (*built*).

**One cluster.** Each cluster has one coordinator, authoritative for that cluster: its engines,
its pools, its nodes, every change. The console and the CLI talk to it directly. Nothing outside
the cluster is needed for an engine to serve or to recover.

**A fleet.** A neocloud runs several clusters, often in several regions. Federation adds no global
control plane: an engine never spans clusters, and a pool never does either, since a spare must
share a fabric with the positions it takes, so there is no cross-cluster state to agree on. What it
adds is a thin fleet layer:

- a registry of clusters, each with its coordinator's endpoint and the credentials to reach it;
- reads fanned out across them: every engine, every pool's deficit, every change, one view;
- each action routed to the one coordinator that owns its engine or node, which decides it as it
  would for a local caller;
- no membership state of its own, so losing it loses the overview and nothing else: every cluster
  keeps serving and recovering.

The console and the CLI speak the same routes to the fleet layer for fleet-wide work and to a
cluster for local work. *Designed, not built.*

**Identity across the three.** An operator signs in once at the fleet layer (OIDC against the
operator's identity provider), with roles per cluster that separate reading from acting; each
coordinator checks the caller again, so a cluster reached directly is no weaker than one reached
through the fleet. Components keep their mutual TLS between themselves (0026). *The operator API's
authentication and its read/act split are built (0039); the sign-on across clusters is designed
with the fleet layer.*

## What an enterprise buyer checks, and where each stands

| Requirement | Status |
|---|---|
| Control plane survives losing a process | built: the coordinator holds a lease (a file lock, etcd or a Kubernetes Lease), a standby takes it over, resumes from the spec store and rebuilds the rest from the daemons' hellos, and a deposed holder's next write is refused (0014, 0036; `controller/src/lease`, `controller/src/coord/takeover.rs`); a 5,000-node cell's failover measured at 0.44 s (`bench/runs/2026-10-14-cell-s7`) |
| Data plane survives the control plane being away | built: engines keep serving with the coordinator gone; only changes wait |
| Identity between components | built: mutual TLS, a daemon is the node its certificate names, a replica takes rows only from the node the coordinator named (0026) |
| Authentication and authorization on the operator API | built (0039): the API has its own TLS listener; a caller presents an operator's or a viewer's certificate, or on Kubernetes a bearer token the cluster reviews as a verb on a resource of the `everserve.ai` group; reading, acting and a pool's launcher are separate grants |
| Rolling upgrades | built (0040): a build serves the protocol of the one before it, so a fleet upgrades node by node; a peer outside the range is refused with both versions in the log (0027) |
| Data integrity | built: a checksum per KV row from the store's write through replication (0030); the worker's device-to-staging path and the restore's copy to the device are named as not yet covered |
| Exactness stated per guarantee | built: exact class with deterministic kernels and the compile-cache check (0029); continuous class designed |
| Audit of every action | partly: every change has a record and an event; the coordinator logs the identity, verb and resource of every API call, and a spec write records its writer; there is no audit log served as a resource |
| Multi-tenancy | designed only as engines in one coordinator; tenants, quotas and per-tenant views are not modelled |
| One view across clusters and regions | designed: a fleet layer that fans out reads and routes actions to each cluster's coordinator, holding no membership state |
| Supported versions | a check of every vLLM dependency at each pin move is designed (owned by the adapter); the support matrix follows from it |
| Install without compiling | built: kernels and the environment's Python bytecode compiled at install, compile and Triton caches staged by the deploy |
| Every process within reach of the operator | built: the processes that register with their node's daemon (workers, spares) answer readiness, stacks and stop, and a spare that ignores its stop leaves its stacks in its log before it is killed; the engine's head registers on its executor's connection, answers stacks and readiness, and its loss puts the engine down with its cause, as does a change that fails after its delivery (0032). The node's own processes outside an engine do not register |
| The engine's head failing over | **known gap** (0045): the head, the scheduler's process, runs on rank 0's node, and that node's loss restarts the engine; under Dynamo the frontend migrates the broken streams to another instance. Any other node's loss is a replacement with the sessions kept |

## Why it lasts

The mechanism inside one engine will not stay ours. Communicator shrink, elastic expert
parallelism and fault-tolerant ranks are moving into the engines, and NVIDIA's Dynamo is building
an orchestration layer above them, free. What lasts is what no single engine does to itself, and
what a hardware vendor's own stack has no reason to do neutrally.

| Durable part | Why an engine or a vendor stack does not build it | Status |
|---|---|---|
| **State that outlives its engine** (below): sessions' KV as a durable address space, replicated, checksummed, fenced by epoch, with a format descriptor | An engine is one process group; durability across its own replacement, a node, a version or a cluster is outside it. The engines' KV connectors are a seam to plug into, and the open stores beside them (LMCache, Mooncake, NIXL) are caches, best-effort by design | built for one engine and one version; the format descriptor designed (0034) |
| **Transitions engines cannot make on themselves** (across): moving positions between nodes, versions and clusters under one machine with capability checks first | Upgrading or relocating an engine is something done to it from outside; a host or driver upgrade and an engine-version upgrade need a destination prepared and checked before the source lets go | replace built; relocate partly built; upgrade and resize designed (0033, 0034) |
| **The fleet** (above): pools shared across engines and models, lending what a shrink frees to batch or RL and recalling it, placement across clusters | An engine sees only itself. Dynamo does this for NVIDIA's stack; the durable version is neutral across engines (vLLM, SGLang) and hardware | designed (0031, 0033) |
| **Guarantees someone else stands behind**: the exact and continuous classes, the compile hold, row checksums, a record of what each change cost and whether sessions survived | An engine does not audit itself, and an operator selling an SLA needs evidence it can show its customers | built for REPLACE; the change record built |

What the system does not promise matters as much. An upgrade still restarts something; what
Everserve removes is the restart's dependence on the requests in flight. A host or driver upgrade
is an evacuation, built from the relocate path. An engine-version upgrade moves sessions only
where the destination can read the state's format, and is refused otherwise. A model-weights
rollout is not a state move at all: an old model's KV is not the new model's, so sessions finish
on the old weights or recompute their context under the new ones (0034).

The engine-side mechanism is given upstream on purpose. When an engine adds native support, it
should ship the hooks this system drives, so that native support makes Everserve the layer to use
with it rather than a duplicate. Being first to a multi-node resize that keeps sessions is a
position of its own; the parts above are what make it a platform rather than a feature.

## Order

The code's layers first: the coordinator split along 0034's layers, the software identity on
registration, and the target layout on every change, behaviour-preserving, before any new kind
lands. Then the transitions in 0033's order: an evacuation of a node that is actually upgraded and
returned while its sessions continue, the engine's own resize read at the pin, resize by data
width, then by expert width, and a pre-registered resize run across two nodes under load. Before a
partner runs Everserve in production: the CLI, API authentication, the protocol compatibility
window, the coordinator's lease and standby, and pools (0031) are built (0036, 0039, 0040, 0042);
the support matrix remains. The head's loss is a known gap outside this order (0045): closing it
needs an engine core that can attach to running workers through a supported executor API, and a
router that keeps an instance's identity while its worker process changes. Cross-version engine
handoff and lending are separately tested capabilities after those.
The Kubernetes custom resources, router integration, OpenTelemetry and the fleet layer follow the
first partner's own stack rather than lead it.
