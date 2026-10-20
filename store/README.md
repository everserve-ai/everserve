# everserve-store

The KV mirror service: the production implementation of `everserve.state.CommittedKV`. Holds
committed KV rows off-GPU per engine, rank and layer, indexed by slot, with a written mask, so a
re-run step overwrites in place and a restore streams exactly the written slots. Keeps the last
committed token per request and keeps it true through eviction and finish.

- `storage`: the tables and their rules: fixed row shape per layer, slot bounds, a memory budget
  checked before any row is written, epoch fencing of stale commits, a checksum per row.
- `shm`: the two memory maps the store makes (a table, a worker's staging file) and why each is
  sound; the only `unsafe` in the workspace.
- `service`: the frame protocol over a Unix socket (`proto/everserve/proto/mirror.proto`); one
  request in, one or more replies out, in order. The socket is the control plane; rows move
  through shared memory (decision 0011).
- `replication`: what one commit wrote for a rank, carried to the stores on other nodes (below).
- `cache`: a rank's compile cache moved between nodes by their stores (decision 0043): the holder
  serves the files its digest covers, each checked against its sha256, and a spare's node pulls
  them into place.
- `metrics`: the Prometheus series, registered through `telemetry/`.

```
everserve-store --socket /run/everserve/mirror.sock --slots 262144 --budget-bytes 64000000000
```

## Replication between stores

A rank's rows replicate to the stores the coordinator names from the policy's `replicaNodes`: the
engine's own member nodes other than the rank's own unless the policy names others (decision 0030,
which reversed 0016's spare nodes only), so a spare restores from a store that holds the rows,
local or over the wire. `--replica-listen` takes peers (`host:port` under mutual TLS with
`--tls-cert`, `--tls-key` and `--tls-ca`, or a Unix path on one node), over the wire crate's
transport; RDMA is not built. A `Replicate` request names a rank's targets and the store carries
every later commit to them on its own task, reconnecting with jittered backoff. No step is dropped
for a slow target: each target's queue is bounded in bytes (`--replication-queue-bytes`, the table
budget unless given) and a full queue holds the source's commit, so the engine slows to what its
link carries rather than leaving a hole; `everserve_store_replica_lag_steps` is the lag per target
and `everserve_store_replication_dropped_total` counts a step a target cannot take at all, over the
frame cap or undecodable. On every connection the carrier asks where the replica stands and sends
the rank as held when the answer is not where it left it, so a replica named late or restarted
empty is filled from the source's tables. The replica applies each step with the source's per-slot
write steps, so an incremental restore from it continues from the same watermark, refuses a step
older than the rank's last, and refuses rows whose checksums do not match. A reply that does not
come within `--replication-reply-timeout-s` (30) is a failed link.

A peer's link carries this protocol and nothing else: a store refuses its workers' and its
daemon's requests from a peer (decision 0026).
