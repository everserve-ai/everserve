# Review: the private-internals ledger, counted by the hook that would remove each row

*A review note from 2026-10-07: a snapshot, kept for the record and not maintained. The current design is docs/ARCHITECTURE.md and docs/decisions/.*

2026-10-07. Read from `adapters/vllm/PRIVATE-INTERNALS.md` at d389ef0 (rows A1 to A35, 37 rows
with the sub-rows), then checked against the adapter's imports: two reaches were unlisted, the
spare's FlashInfer autotune replacement and the daemon-launched spare's KV configuration, now rows
A36 and A37, and `adapters/vllm/tests/test_private_internals_ledger.py` fails on any vLLM module
the adapter imports that the ledger does not name. The question: are the reaches into vLLM's private names a growing list, or a
fixed set of missing extension points? Each row is assigned to the one upstream hook that would
remove it. No file under `vllm/` is edited; every row is in the adapter.

| hook | what vLLM would take | rows | count |
|---|---|---|---|
| H1 | **A group object for its collectives.** The distributed layer resolves tensor-parallel and pipeline groups from a fixed world at init and bakes them into every collective and into the captured graphs. A group or communicator object the executor can supply and replace, with the pipeline transport and the handler's broadcast taking it too. | A12, A13, A14, A15, A17, A18, A25, A26, A33, A34, A35 | 11 |
| H2 | **An executor that treats a worker's death as an event and a ring that readers can join and leave.** The multiprocess executor shuts down on any death, its broadcast ring is formed once over the world, and the headless launcher hard-codes its executor class. | A1, A2, A4, A5, A6, A8, A9, A10, A29, A30, A31, A32 | 12 |
| H3 | **The engine core calling the executor between steps on a committed membership, and a request resuming from a committed count.** The scheduler's only recovery is preempt-and-re-add from the block boundary. | A3, A11, A11a, A11b, A20a, A23 | 6 |
| H4 | **The connector handed the slot ids a step wrote and a documented paged layout.** The connector API hands over raw tensors; the slot ids reach it through attention metadata by accident of the runner's call. | A21, A22, A24 | 3 |
| H5 | **A worker that can be built as one rank outside a running world, and an RPC target that knows its worker.** The spare's bootstrap and the worker-side RPC context. | A16, A19, A20, A27, A28, A36, A37 | 7 |

Five hooks cover the thirty-nine rows. The first two cover twenty-three of them and are the
same ask from two sides: membership is a runtime fact, not an init-time constant. The rows arrive
one at a time because each surfaces when a run first exercises its path, not because the set is
open; for a dense model at tensor by pipeline parallel the inventory of membership-sensitive paths
is the collectives, the ring, the scheduler's recovery, the connector's view of a step, and the
spare's bootstrap, and after the pipeline landing (A33 to A35) that inventory is complete. Expert
parallel adds the all-to-all under H1 later.

**The rule for a new row,** from `docs/REVIEW-CHECKLIST.md` and this count: a row must map to one
of the five hooks. A reach that fits none is a design problem in the adapter, not a missing hook.

**The rows to watch,** because they are patches or frame walks rather than overrides: A12 (a
class-level patch of the tensor-parallel all-reduce), A19 (caller frames walked to the worker
process), A28 (a monkeypatch in the spare's process), A34 (a process-local patch of
`torch.distributed.broadcast`). Five of thirty-nine, A36 being the fifth, a process-local replacement of a warm-up function. Each is tolerated by the checklist as a
listed line whose entry names the hook; H1 removes A12 and A34, H5 removes A19 and A28.

**What this means for the upstream proposal.** Decision: the RFC waits until the product is
usable, and the two-node result is that moment. Its shape is now known: two asks, H1 and H2,
with H3 as the one that makes a change invisible to sessions, and H4 and H5 as small API
clarifications. The ledger is the evidence for each, row by row, with the pin's file and line.
