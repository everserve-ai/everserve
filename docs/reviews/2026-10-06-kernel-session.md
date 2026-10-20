# Review: the kernel session landings (6f935a1, c4f2c01, 98cf261)

*A review note from 2026-10-06: a snapshot, kept for the record and not maintained. The current design is docs/ARCHITECTURE.md and docs/decisions/.*

2026-10-06. Reviewed against `docs/REVIEW-CHECKLIST.md` after the three landings reached main
with CI green. The GPU evidence is the landing's own, from its GPU session: byte equality of the gather kernel against the torch
path on every layout, bit equality of both slot variants against the reference sum, a captured graph
replaying across a membership change, a REPLACE recovering with 48/48 tables, and the P2P variant
at 16–18 µs a call against NCCL's 20.2. CI cannot run any of it; the GPU tests skip without CUDA.

## Verdict

Accepted. The three landings are the production shape the decisions asked for: one launch per step
for the gather (0010, 0011), both slot variants in `core/csrc` with the interim deleted (0006), and
aborts that overlap. The follow-ups below are real but none blocks the re-run.

## What was checked

- **Gather kernel** (`core/csrc/gather.cu`, `_GatherKernel` in `mirror.py`). The side stream waits
  on an event recorded at commit time, after the whole forward is queued, so the reads see every
  layer's writes; the slot-reuse argument in `_gather_step`'s docstring holds for vLLM's allocator
  (a slot is rewritten only after its block is freed, which only happens to a finished or preempted
  request whose counts the store is told to drop or lower, behind the same commit). The 16-byte
  path checks alignment of every stride and of the output pointer; the layer base pointers are
  checked on the host. Padding reads row 0 as the torch path does. The step thread still reads
  nothing from the device. The torch path remains for CPU tests and for caches that differ in
  shape, stride or dtype.
- **Slot kernel** (`core/csrc/slot.cu`, `everserve/comm/slotkernel.py`). NCCL's LL protocol: an
  8-byte word carries two bf16 and the 32-bit sequence number; a reader that sees the number sees
  the data. Both variants zero their own receive buffer at the boundary, which in the push variant
  is where every sender's stale words live, so a word from the old epoch cannot match a new call.
  The abort word is read every 1,024 polls from host memory, and an aborted read sums zeros; the
  runtime discards the step anyway. Members are summed in row order in fp32 and rounded to nearest
  even, which is the bit-equality the tests pin.
- **Concurrent aborts** (`groups.py`, `collective_op.py`). Each `ncclCommAbort` runs on its own
  thread, bounded; the change waits for all with one shared bound, and records each duration or
  that it was still running. Correct, and the measured halving matches.

## Follow-ups, in order

Items 1–4 landed the same day (the commit after this review); item 5 is a packaging landing.

1. **Variant selection by device identity, not index.** `choose_variant` maps
   `CUDA_VISIBLE_DEVICES` entries to NVML indices with `int()`. That fails on UUID entries and can
   mis-pair devices when CUDA's default device order differs from NVML's PCI order on a mixed
   host. Resolve through UUIDs on both sides.
2. **The worker report names the gather path.** A failed kernel build falls back to the torch
   path with one log line. The report (`commit.transport`) should carry `gather: kernel|torch` and
   `slot: p2p|host`, so a run that quietly lost its kernels is visible in `results.json`, not only
   in a log nobody reads.
3. **Peer memory after a peer dies.** The push variant writes into every member's buffer until
   the mask changes. CUDA IPC imports hold a reference, so the exporter's death does not free the
   pages under the writers; the REPLACE run on the pod bears this out. Write the invariant above
   `_open` so the next reader does not have to rediscover it.
4. **Sequence wrap.** The 32-bit sequence number wraps after 2^32 calls, about twelve hours at
   decode rates. A stale word matches only if its position was last written exactly 2^32 calls
   ago and never since. The same exposure exists in NCCL's LL. Document it in `slot.cu`.
5. **Prebuilt kernels in the wheel.** `everserve.kernels.load` compiles on first use with
   `cpp_extension.load`, minutes on a cold host. `python -m everserve.kernels` ahead of time is
   documented; a packaging landing should ship the built extension per torch and CUDA version so
   a worker never compiles.

## Housekeeping from the session

- Decision 0011's claim that a disk-backed data directory works was wrong: a disk-backed file
  cannot be pinned. Corrected in the decision; the transport's error already says so.
- 2.3 MB of raw run data entered history in 3569906 and was untracked in 8b22ab7. Not worth a
  rewrite.
