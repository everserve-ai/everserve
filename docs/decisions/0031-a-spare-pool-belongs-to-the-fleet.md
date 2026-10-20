# 0031: a spare pool belongs to the fleet

2026-10-09. Status: current, as amended below (2026-10-12); landings 1 to 3 built.

**Context.** The shared spare of decision 0030 stands for any position of one engine, named in that
engine's policy entry on a node named by hand. A neocloud runs many engines, most of them replicas
of a few models, and changes its nodes all the time: reclaims, failures, maintenance, new capacity.
Two things follow. A spare should stand for every engine that could use it, so one spare node
protects a fleet of replicas rather than one. And getting a node is the operator's or its
scheduler's work, not ours (0030: scheduling stays outside); ours is to say what is missing and to
take in whatever arrives without configuration by hand.

**Decision.** Three nouns and one rule.

- *Engines* serve, as today.
- *Pools* are spare capacity for one engine shape: a model, its parallel layout and its compile
  key (0029). Every engine of that shape draws from the pool. A pool states what may join it (a
  device class, a GPU count, optionally a label and a network domain) and how many spares it
  wants, and how it holds them (resident in host memory, the default, or preloaded per position,
  0028).
- *Nodes* have a role that is derived, never assigned: serving while it holds positions, spare
  while it matches a pool and holds none, cordoned while an operator has set it aside.

*The rule:* a node that matches a pool and holds no position is that pool's capacity. A node
newly provisioned, one repaired, and one back from maintenance all join the same way: its daemon
connects, the coordinator matches it, and the daemon launches the pool's standing spares on it.
A node that held positions and comes back never reclaims them; positions move only by a change.

A take is unchanged in kind. When a change finds a lost rank of an engine, the coordinator draws a
standing spare from the engine's pool, preferring one in the lost rank's network domain, and names
the engine and the rank in the take; the spare copies that shard from the model it holds resident,
the staging brings the rows, the compile check holds it to the position's pair at the join.

**What a scheduler or a console uses.**

- `GET /v1/pools`: per pool, spares wanted, standing, taking and warming, the deficit, and the
  engines it protects. A scheduler that keeps the deficit at zero is the whole refill loop.
- `GET /v1/readiness`: per engine, whether its pool can cover the loss of its largest unit (a node)
  now.
- `POST /v1/drain`: built. A node's positions move to the pool's spares at a deadline: a
  maintenance window, a reclaim warning, a node being retired.
- `POST /v1/nodes/{node}/cordon` and `uncordon`: the node takes no positions and stands no spares;
  its daemon stays connected. Maintenance is cordon, drain, service, uncordon. On Kubernetes the
  native cordon is the same signal: the daemon's Kubernetes notice source already reads it and
  drains the node, so an operator uses `kubectl cordon` and this route serves bare metal.

The policy names spare nodes by hand only when an operator wants that (today's `spares` slots
remain the pinned form of a pool with one member).

**Consequences.**

- Communicators scale with how wide a pool is. A standing spare forms and warms one set per position
  it can take, now across engines; a pool therefore bounds which engines a spare pre-forms for (by
  network domain, by default), and a take for an engine outside that bound forms its set at the
  take, the time recorded in the change.
- Host memory bounds how many models a spare node covers: one resident copy per model. A pool for a
  model that does not fit stands its spares loading from local storage at the take, slower and
  still exact, and says so in its readiness.
- The control plane's own state grows by the pools and the derived roles; nothing about a change's
  protocol, the staging or the checks changes.
- The headline measurement does not depend on any of this: one engine and its spare. It is built
  after the multi-GPU run and before a partner runs Everserve in production.

## Amendment (2026-10-12): the pool's spare, the shape, and the first landings

Settled with the processes' half before anything is built.

- **One process, one registration per engine.** A pool spare registers once with each engine of
  its pool, each registration carrying the engine's identity and the compile pair it built for
  that engine's configuration. Everything per engine stays per engine: its admission (0034), its
  cover groups keyed in that engine's ids (0035), the slot region an any-rank take makes. When the
  process takes a position of one engine, it closes its other registrations, only after the
  take's commit is applied (its boundary), never before. Each other engine reads a close as a
  withdrawal from its pool, never a loss, when that node and device already holds a member of
  another engine at a committed epoch; otherwise it is a departure as today. The order is
  decidable: the coordinator records a commit in its view before it sends it to any daemon, and
  the process closes only after applying it.
- **The shape includes the compile key.** Two engines of one model can differ in configuration
  that changes the compile key (maximum length, block count, graph sizes). A pool's engines must
  agree on the identity's engine-compatible fields (model, numerics, code) and on the compile key;
  a pool spec naming engines that disagree is refused when it is written, not discovered at a
  join. The KV layout and the pair are checked at the join, as today.
- **Host memory bounds models, per spare process.** The resident copy holds the whole checkpoint;
  a take builds its rank's shard from it. What a take adds per engine is the KV configuration and
  the compiled graphs, not weights.
- **The cover budget counts across engines.** A pool spare holds every pool engine's cover groups
  at once, so the planner's groups per process is counted over all the engines a process stands
  for, not engine by engine.
- **Launch.** The spare's entrypoint takes `--pool NAME` beside `--engine` (a single-engine spare
  is unchanged), and reads the pool's engines from the policy.

**First landings.**
1. The Pool resource: `PoolSpec` in the spec store (its engines, what may join, how many it
   wants, how it holds them), validated at write (the engines' shapes agree); its status and
   conditions (`Satisfied`, the deficit) and `GET /v1/pools`, read over today's spares: an engine's
   spares count toward the pool naming it.
2. Withdrawal: a spare's registration that closes because its process took another engine's
   position leaves the pool without a loss, and the planner counts groups per process across
   engines.
3. Launches: the coordinator launches a pool's spares on matching nodes that hold no position and
   are not cordoned, until the deficit is zero (with `--pool`).

Landing 1 is built (2026-10-12): `PoolSpec` in the spec store, refused at write for an engine not
in the policy or engines whose known shapes differ; `GET /v1/pools` and `POST /v1/pools/{pool}`;
standing counted per process across the pool's engines, on nodes the selector admits and not
cordoned; `Satisfied` and `ShapeAgrees`, derived again after every change to an engine's view;
`everserve_pool_deficit{pool}`.
Landing 2 is built (2026-10-12): a spare's leaving is `SpareWithdrew` (to the engine and rank it took)
or `SpareDeparted`, decided by whether its node and device already serve another engine at a
committed epoch; the cover planner counts each spare process's groups across engines, an engine
planned within what its spares already hold in the others' plans.
Landing 3 is built (2026-10-12): a pool short of spares launches them, with the command its spec
carries, on devices of idle nodes it admits (connected, not cordoned, holding no position); a
launched spare is `warming` until it stands, or for 15 minutes. The pool's spare process (`--pool`)
is the processes' half.
