# 0016: a rank's rows replicate to the nodes holding its eligible spares, and nowhere else

2026-10-06. Status: reversed by 0030 (2026-10-09) for the shared-pool shape; kept as the record of the
mechanism and its proof. A ruling on `docs/MULTINODE.md` M3.

**Decision.** A rank's rows stay in shared memory on its own node (decision 0011) and are
replicated, store to store, asynchronously, to the one or two nodes holding spares eligible for the
rank's position, as the policy names them. The replica carries the same per-slot write steps so an
incremental restore works on it; its lag is a metric; the coordinator names the replica's node in
the change.

**Why not every node.** Traffic would scale with the node count and prefill bursts would multiply
across the fabric, to buy a flexibility that a spare without a shard cannot use. A third target is
a policy change if a pool needs it.

**Built 2026-10-06.** `store/src/replication.rs`: one task per (rank, target), a queue of
`QUEUE_STEPS`, the replica applying with the source's write steps so `since_step` continues from
the same watermark. Who names the targets, the coordinator through the daemon, lands with the
spare pool (`docs/MULTINODE.md` item 4).
