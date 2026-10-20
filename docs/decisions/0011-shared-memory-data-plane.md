# 0011: the mirror's data plane is shared memory; the socket carries control

2026-10-05. Status: current.

**Context.** With the device read gone (decision 0010), the step thread's remaining commit cost was
the fence: 4.4–5.6 ms per step on the four-GPU A40 pod (`bench/runs/2026-10-05-nosync-commit-4gpu/`).
Reproduced locally against the real store at the run's model shape, the sender's work per decode
step is 0.6 ms of Python: copying rows into protobuf messages and pushing them through the socket.
On that pod's CPU, with the engine's own Python competing for the interpreter, it is 5 ms. The fence
waits exactly that long, because the sender can only start when the GPU finishes the forward, and
that is the moment the step thread needs the previous step acknowledged. The restore paid the same
cost in the other direction: 420 MB through protobuf in 1.4–1.7 s, most of the REPLACE pause.

**Decision.** Rows never cross the socket on a node. The store keeps every table as a file under
its data directory on shared memory (`/dev/shm/everserve` on a node). A worker stages a step's rows
in a file of its own under its rank directory, mapped as a tensor and pinned for the GPU, so the
side stream's copy lands in it directly. A commit is a control message naming the staged rows: file,
offset, slot ids, table. The store maps the staging file once and copies the rows into the table in
Rust. A restore is a control message naming the dead rank's tables and their written slot ranges;
the restoring worker maps the tables and copies the written rows to its GPU. The protobuf schema
(`proto/everserve/proto/mirror.proto`) carries only descriptions; `RowBatch` is gone.

**Consequences.**

- The sender's Python work per step is one small message, independent of the step's size. The
  store's work is a memory copy. The fence wait tracks the socket round trip, not the row count.
- A restore moves at memory speed: a host gather of the written rows into pinned memory and one
  copy to the device per table.
- Tables outlive the worker process, so a dead rank's rows are where the spare expects them, and a
  restarted store could reload them later (not built: the written mask lives in the store).
- The store's host memory is the tables, bounded by the budget as before, plus the workers'
  staging files, bounded by `depth + 1` buffers per rank, the largest a prefill step.
- Multi-node (phase 2) replicates tables between stores over the fabric and leaves the worker side
  as it is: the worker still stages locally and names rows to its node's store.
- `memmap2` is a dependency of the store, and `store/src/shm.rs` holds the workspace's only
  `unsafe`, two map calls with the invariants that make them sound written above them.
  `unsafe_code` stays denied everywhere else.
- Decision 0008 holds: no Rust in the worker process; the control plane is still the protobuf
  socket. Only where the bytes live changed.
- The data directory must be on shared memory (`tmpfs`, `/dev/shm`) on the same host as the
  workers and readable by them. A disk-backed file cannot be pinned for the GPU, so a disk-backed
  directory does not work at all; the transport says so when registration fails (found on the
  2026-10-06 kernel session). A pod whose `/dev/shm` is small (containers often give 64 MB) needs
  a larger one: `--shm-size` on the container, or a `tmpfs` mount of its own for the directory.
