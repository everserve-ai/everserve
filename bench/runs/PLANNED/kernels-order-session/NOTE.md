# Kernel order session, 2026-10-05: spend and what it produced

A build session for decision 0012, not a measurement run. Commands, prices and timestamps are in
`run.log`; the A100 development runs' raw outputs are in `data/` (not in git). The H100 pod's
development outputs were lost: its image had no `rsync` and the copy was attempted after the
processes were stopped; what they showed is in `run.log`.

## Spend

| What | Cost |
|---|---|
| Pod `eakvf8b9bkt3i3`: 5× H100 SXM, secure US, 20:36:42Z–21:34:26Z (57.7 min) at $17.45/h | $16.79 |
| Pod `aj84thxtvfu0yf`: 5× A100 SXM, secure US, 21:36:32Z–22:13:55Z (37.4 min) at $7.95/h, before the re-run on the same pod | $4.95 |
| **Total** (cap $25) | **$21.74** |

No 5× A100 SXM was available for an hour at the start, so the owner chose H100; A100 came back
while the H100 findings were being worked through off the GPU.

## What landed (each gated, pushed, CI green before the next)

| Commit | Landing | Evidence |
|---|---|---|
| `419f9d6` | the spare holds its worker lock for its life | the engine hung on H100: `Process.start` drops its arguments, the lock was collected and its semaphore unlinked before the spare opened it |
| `fff9455` | on Hopper the spare loads the serving ranks' FlashInfer tactics; stub groups answer `barrier` | the spare's warm-up failed on H100 at vLLM's autotune barrier; tuning alone it could also pick another kernel than the rank it replaces |
| `3a8ac5e` | 0012 item 1: the sum in rank order, written by the executor, the spare in place; magic `EVRSLOT2` | GPU test bit-equal across a REPLACE with order-sensitive inputs, both variants (H100 and A100); notice 8/8 tokens with the spare in place |
| `b8000ee` | 0012 item 2: each member's order in device memory, abort alone in host memory; 1 word/thread | `slot_bench.py`: 4.21 µs at 16,384 against custom all-reduce 5.81 and NCCL 16.78; adapter 993 tok/s against 839 |
| `e831fb5` | a notice-length no-fault reference in the harness | notice compared 8/8 against `notice_ref` |
| `8b51e6f` | the first-steps probe | reported per rank in every fault run |

## Found on the way

- **Item 1's drill did not pass.** Session 1 still differs from token 114 after every REPLACE,
  with the order applied correctly on every member, while notice swaps match 8/8. The cause is in
  the fault path (aborted step, replay of the tail), not the order. Decision 0012 is wrong in
  attributing the mismatch to the order alone.
- **Stock vLLM is not deterministic between runs on 5× H100** (two identical stock runs differed;
  NVLS off alone did not change that). vLLM's dispatch on Hopper puts FlashInfer's all-reduce
  first, which its own source says has no fixed reduction order, and enables symmetric-memory
  multimem. The harness now turns these off in every arm (`5ff7744`) and records a determinism
  control. **Unverified on H100**: it needs two stock runs there with the settings on.
- **The A100 engine is deterministic between runs** with and without those settings.
