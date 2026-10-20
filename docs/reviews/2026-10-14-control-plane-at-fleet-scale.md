# Review: the control plane at a fleet of 10,000 to 500,000 GPUs

*A review note from 2026-10-14: a snapshot, kept for the record and not maintained. The current design is docs/ARCHITECTURE.md and docs/decisions/.*

2026-10-14. The direction given: a neocloud's fleet of 10k, 100k, 500k GPUs; **the engine is the
unit**, its control independent of every other engine's; **spare pools shared per shape**. This
review sets the design that scales without a fleet-wide bound, finds where the code at the time did
not follow it, and orders the work. **No figure here is measured yet**: sizes are arithmetic over
the code's and Kubernetes' stated limits, marked as such; each change lands with a measurement
(S0) before any claim.

## The bound we do not own

Upstream Kubernetes supports clusters of up to 5,000 nodes (its published scalability
thresholds), about 40,000 GPUs at 8 per node. 500,000 GPUs is a dozen or more clusters whatever
Everserve does. The design therefore has three levels, and each level's cost is bounded by its
own size, never the level above.

| Level | Holds | Bounded by | At 500k GPUs (arithmetic) |
|---|---|---|---|
| Engine | its ranks, spares, cover, changes, its object in the store and the API | its own size, 8 to 1,024 GPUs | each engine as at 8 GPUs |
| Cell | one cluster's Everserve: its engines, nodes and pools | the cluster, about 5,000 nodes / 40,000 GPUs | 13 or more cells |
| Fleet | the cells, side by side | nothing: a cell is added, none grows | read-only views above them |

## The design

- **The engine is the unit of control.** All of an engine's state, its lock, the facts it sends
  (policy, placement, instance state) and its object are its own; a change touches its engine
  alone. Its facts go to the daemons with a process of it, nowhere else.
- **A cell's coordinator is sharded by engine.** Several controller replicas each hold leases on
  a share of the cell's engines (one lease per engine or per shard of engines); a replica that
  loses one, its standby takes it. Each engine's spec-store object is already its own (0036,
  sharded). A node's daemon links to the replicas owning an engine with a process on its node: at
  most a few, never all.
- **Pools are shared per shape, inside a cell.** A pool spans one shape's engines (0031); its
  controller is sharded by shape and grants a spare to an engine's controller as a claim. A pool
  never spans cells: a spare cannot take a position in another cluster.
- **Nodes are per node.** Cordon, drain and the node's lifecycle are a node controller's, sharded
  by node; a drain asks each engine on the node, by its owner.
- **The API serves per cell, aggregated above.** A cell serves its engines, nodes and pools,
  paged and watched; a fleet view, where one is wanted, reads the cells and owns nothing.

## Where the code did not follow it on 2026-10-14

- **S1. An engine's facts go to every daemon.** *Built:* `coord/audience.rs` sends them to the
  engine's audience only; a link gets its own engines' facts; a hello refreshes only its engines'
  replication; reconnects are jittered. The synthetic cell found linking, not the change, was the
  cell-wide cost (`bench/runs/2026-10-14-cell-s0`). Before it: `push_placement` and `push_instance_state` send
  to every linked daemon, so a spare's node hears its engine before the spare registered (the
  first Kubernetes run's spare heard none). Cost per change: the cell's node count, not the
  engine's. Fix: track, per engine, the daemons with a process of it, from the ask on, and send to
  those.
- **S2. The API feed recomputes everything.** *First step built:* linear reads and typed
  comparison took one refresh at 5,000 nodes from 323 ms to 56 ms
  (`bench/runs/2026-10-14-cell-s2-reads`); per-object marks remain. Before it: `Feed::refresh` serializes every object every
  250 ms, changed or not. Fix: per-object dirty marks where the coordinator changes an object, a
  full sweep every 30 s as the net.
- **S3. One lock holds every engine.** *Measured, deferred* (`bench/runs/2026-10-14-cell-s3`):
  a burst of 100 changes shares about 0.1 ms per change under the lock, and the store writes of
  different engines run concurrently (a 10 ms store adds 27 ms to a burst, not 2.6 s). Before
  it: `engines: Mutex<BTreeMap<..>>`. Fix, when a burst's shared time matters: a lock per
  engine; fleet-wide reads take each in turn.
- **S4. Lists are not paged.** *Built:* `?limit=N&continue=T`, each page at the first page's
  version; `everserve-controller get` reads 500 at a time (decision 0042, amended).
- **S5. One coordinator per cell holds every engine.** *Not needed for capacity, as measured:*
  one coordinator holds a 5,000-node cell, 2,500 engines, in 263 MB, links it within half a
  second and changes an engine in 1–2 ms (`bench/runs/2026-10-14-cell-s1`). What one coordinator
  per cell still risks is its failover: while a standby takes the lease and the cell's daemons
  link to it, no engine in the cell can change. That time is measured next (S7); sharding within a
  cell is built only if it is too long to bear.
- **S6. A pool is a coordinator's, not a shard's.** Follows S5: while a cell has one
  coordinator, a pool per shape within it is already shared by every engine of the cell.
- **S7. A cell's failover.** *Measured* (`bench/runs/2026-10-14-cell-s7`): every daemon of a
  5,000-node cell relinked to a successor and every engine came back whole within 0.44 s; the
  successor's first change took 352 ms there, an open item. The measurement found a defect first
  (a dropped listener deleted its successor's socket, fixed). As asked before it: A coordinator stops; its successor takes the store
  and the lease, and the cell's daemons, with jittered backoff, link to it and say hello. How
  long until every engine can change again, at 1,250, 2,500 and 5,000 nodes.

## Inside one engine

| Item | On 2026-10-14 | Fix |
|---|---|---|
| Cover plan size | 2,192 groups, about 730 KB, at 64 GPUs (measured 2026-10-13); 280 groups, about 15 KB, with shared groups | shared groups (branch `shared-groups`) before any engine past 2 nodes |
| Groups per process | budget 8; too few for node-loss cover at pipeline depth 4 or more | a default from a measured bytes-per-group |
| Spec-store object | bounded at 1 MiB, under etcd's limit | a 1,024-rank engine's log measured against it |

## Measured since (2026-10-14)

S0 to S4 are built or measured; each figure's record is under `bench/runs/2026-10-14-cell-*`.
On one laptop, one coordinator brings up a 5,000-node, 40,000-GPU cell within half a second. Each
change sends about 600 bytes to its engine's two daemons and commits in 1–2 ms. One API refresh
takes 56 ms, and lists are paged. A 500,000-GPU fleet is about thirteen such cells, none larger.
So the fleet scales by adding cells, which is the design, and nothing measured asks for sharding
inside a cell.

## Order

1. **S0, measure at cell size.** A synthetic cell in one process: N fake daemons, M engines of K
   ranks, pools, cover, losses; it records CPU, memory, a change's report-to-commit time and
   bytes per change at 1,250 / 2,500 / 5,000 nodes. Run in CI as a test with stated bounds.
2. **S1 and S3:** a change's cost becomes its engine's.
3. **S2 and S4:** the API at cell size.
4. **S7:** a cell's failover, measured; S5 and S6 only if it is too long.
5. **Inside the engine:** shared groups and the group budget.
6. **Above cells:** a read-only fleet view, when a customer asks for one.
