# 0010: the step thread never reads device memory

2026-10-05. Status: current.

**Context.** The asynchronous-commit run (`bench/runs/2026-10-05-async-commit-4gpu/RESULT.md`)
moved the store's round trip onto a sender thread and the overhead did not move: 27.0–29.0 %,
the synchronous port's figure, with zero fence wait and zero back-pressure. The step thread paid
13–17 ms per commit, which is one decode step of GPU time on that host. `commit()` copied the
step's slot tensor to the host before anything else. On the step thread, that read waits for
every kernel queued ahead of it, which is the whole forward pass, and then the gather ran on the
CPU with the GPU idle until the sampler was launched. The synchronous version had the same read
in the same place. Decision 0009's context attributed the cost to the store's acknowledgement; the
run shows that was wrong, and the fence design stands on this decision instead.

**Decision.** No code on the step thread reads device memory while a step's rows are saved or
committed: no `.cpu()`, `.tolist()`, `.item()`, `.numpy()`, no boolean conversion, no boolean-mask
index. Slot ids stay on the device as the gather index; padding ids gather row 0. Each layer's
rows leave the device in `save()`, one gather for every held head slot on the compute stream
behind the layer's own writes and one copy into a pooled pinned buffer on a side stream, so the
copies overlap the layers still computing. `commit()` records one event behind the last copy and
hands the step to the sender thread. The sender waits for the event, reads the slot ids from
pinned memory and describes the rows to the store, padding included, which the store skips. A step's cost on the step thread is kernel
launches and one event record.

**Consequences.** The step thread's commit cost is independent of the forward's length. The
pinned pool holds at most `depth + 1` buffers; a prefill of thousands of tokens takes a buffer of
its own size, so host memory for the pool is bounded by `depth + 1` prefill steps. `PagedKV`
gained `rows_of_heads`. The test `test_save_and_commit_never_read_device_memory_on_the_step_thread`
forbids the method forms of a read on the step thread; the boolean-mask form is kept out by
review (`docs/REVIEW-CHECKLIST.md`). The re-run of the pre-registration judges the result against
a stock baseline on the same host.
