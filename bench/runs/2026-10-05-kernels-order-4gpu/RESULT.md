# Result: the kernels re-run, rank-order sum and device-resident order, on 5× A100

**Pod and software.**
- Run 2026-10-05T22:13:55Z to 22:33:51Z on RunPod pod `aj84thxtvfu0yf`: 5× A100-SXM4-80GB, NVLink
  (NV12, every pair), secure cloud, US, $7.95/h, driver 595.91.07, no `cuda-compat`. The build
  session ran on the same pod before it (`bench/runs/PLANNED/kernels-order-session/`).
- vLLM at the pin (`6e517b15`), torch 2.13.0+cu130, NCCL 2.29.7. **No file under `vllm/` was
  edited.**
- Every arm, stock included, ran with `VLLM_ALLREDUCE_USE_FLASHINFER=0
  VLLM_ALLREDUCE_USE_SYMM_MEM=0 NCCL_NVLS_ENABLE=0`, as pre-registered.

**Slot variant.** Peer-to-peer, reported by every worker of every adapter run (`paths.slot`).

**Data directory.** `/dev/shm/everserve` (274 GB).

**Tree.** `72729ed`: the pre-registration's commit on landings 1–4 (`3a8ac5e`, `b8000ee`,
`e831fb5`, `8b51e6f`) and the two spare fixes (`419f9d6`, `fff9455`), CI green. `PREREGISTRATION.md`
is unchanged since then.

**Stock ran with vLLM's custom all-reduce on** in all three runs (`custom_all_reduce: true`).

**Records.** `results.json` (every figure per run, the determinism control, the first-steps probe
per rank, SHA-256 of `data/`), `run.log`, `data/` (not in git), `flake/` (every rank's event log
for the REPLACE and notice runs).

19 runs, all pre-registered, in order, on this pod. Nothing was repeated or added afterwards.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | 8/8 sessions token-identical after REPLACE (any mismatch, loss, re-prefill) | **session 1 differs from token 114 in 6/6 runs**; 7/8 match; 8/8 kept; replay from 2,149 | **dead** |
| 2 | 8/8 sessions token-identical after notice (any mismatch, or not checked) | 8/8 in 3/3 against `notice_ref` | **alive** |
| 3 | adapter overhead < 3 % against stock with custom all-reduce on (> 5 %) | −1.33, −1.32, −1.29 % | **alive** |
| 4 | mirror-off overhead < 1 % (> 3 %) | −2.00, −1.93, −1.92 % | **alive** |
| 5 | REPLACE pause < 1.5 s (> 2.5 s) | 0.753–1.279 s | **alive** |
| 6 | notice pause < 0.3 s, swap in all, no fault (no swap, > 1.0 s, fault) | 0.200, 0.249, 0.248 s; swap 3/3; 0 faults | **alive** |
| 7 | store span `sender` p50 < 1 ms every rank (> 3 ms) | 0.37–0.49 ms | **alive** |
| 8 | spare restore < 0.4 s (> 1.0 s) | 0.161–0.190 s | **alive** |
| 9 | restored rows equal the dead rank's (any mismatch) | 0 mismatched tables, 9/9 runs | **alive** |
| 10 | commit within 100 ms of resume, epochs applied | 22.2–32.9 ms REPLACE, 49.6–63.1 ms notice; epoch 1 everywhere | **alive** |

**The control.** The three no-fault runs of each arm (stock, mirror-off, adapter) gave identical
tokens in all 8 sessions, and `notice_ref` equals `adapter_1` over their common 201 tokens. So the
REPLACE mismatch is not the engine varying between runs.

## Overhead (all from this pod, this session)

| Arm | tok/s, steps 10–200 | Mean | Overhead against stock |
|---|---|---|---|
| stock, custom all-reduce on | 981.1, 981.2, 982.2 | 981.5 | — |
| adapter, mirror off | 1001.1, 1000.4, 1000.3 | 1000.6 | −1.95 % |
| adapter | 994.5, 994.4, 994.1 | 994.3 | −1.31 % |

The adapter is now faster than stock, where the last run had it 7.8 % behind. The change is the
slot collective's (decision 0012 item 2): with each member's order in its own device memory and
one word per thread, a decode-size call takes 4.21 µs at 16,384 elements against 5.81 for the
custom all-reduce (`slot_bench.py`, in the pre-registration). The mirror costs 0.6 %.

## Prediction 1: what is known

- Session 1 diverges at token 114 in every REPLACE run, both kill timings; the kill came after token
  101 (between) or 102 (mid). The same token as the last run, before the rank-order sum.
- Every member applied the order `[0, 1, 4, 3]`, the spare's row in rank 2's position
  (`slot_order` in each change).
- The notice runs swap in the same spare, at the same position, and match in all 8 sessions. The
  restored rows equal the dead rank's. So the spare computes as the dead rank did, and the sum
  holds across a swap.
- What a notice does not do and a REPLACE does: abort a step mid-flight, and replay the tail from
  the dead rank's committed count (2,149). The divergence is in that path. Rank order was not its
  cause, or not its only cause; decision 0012's account of it is wrong in that part.
- The first-steps probe shows the survivors' first step after resuming takes 89–95 ms of GPU
  against 15–21 ms for the next, with no compile-counter change and no new allocation on the
  survivors. What that step computes differently is the next question; it is not answered here.

## The REPLACE pause, with each abort

Seconds. Aborts are each survivor's `ncclCommAbort` of the live TP handle and of the EP group's
pynccl handle, run together; the torch group's abort was 0.001–0.002 s in every run.

| Run | Pause | Detect | Change | Abort, live TP (ranks 0/1/3) | Abort, EP (ranks 0/1/3) | Restore | Resume | First token | Daemon change |
|---|---|---|---|---|---|---|---|---|---|
| between 1 | 1.083 | 0.384 | 0.351 | 0.247 / 0.510 / 0.247 | 0.226 / 0.226 / 0.217 | 0.164 | 0.030 | 0.155 | 0.514 |
| between 2 | 1.279 | 0.350 | 0.604 | 0.425 / 0.763 / 0.410 | 0.306 / 0.268 / 0.306 | 0.164 | 0.028 | 0.134 | 0.767 |
| between 3 | 0.753 | 0.357 | 0.057 | 0.228 / 0.227 / 0.229 | 0.229 / 0.211 / 0.201 | 0.177 | 0.025 | 0.138 | 0.233 |
| mid 1 | 1.191 | 0.472 | 0.374 | 0.536 / 0.177 / 0.177 | 0.534 / 0.296 / 0.292 | 0.167 | 0.025 | 0.150 | 0.540 |
| mid 2 | 1.157 | 0.321 | 0.460 | 0.423 / 0.403 / 0.403 | 0.625 / 0.549 / 0.464 | 0.171 | 0.022 | 0.180 | 0.629 |
| mid 3 | 1.026 | 0.328 | 0.362 | 0.298 / 0.519 / 0.286 | 0.283 / 0.286 / 0.299 | 0.161 | 0.033 | 0.138 | 0.523 |

The slowest abort on any survivor still sets the change (0.23–0.76 s).

## Driver pause against the daemon's change

| Kind | Runs | Driver pause (largest gap) | Daemon `everserve_change_seconds` |
|---|---|---|---|
| REPLACE | between 1–3 | 1.083, 1.279, 0.753 | 0.514, 0.767, 0.233 |
| REPLACE | mid 1–3 | 1.191, 1.157, 1.026 | 0.540, 0.629, 0.523 |
| notice | 1–3 | 0.200, 0.249, 0.248 | 0.174, 0.182, 0.192 |

## The first step after a change (instrument, no prediction)

- Every fault run's first step after the change took 89–100 ms of GPU on every rank, against
  11–21 ms for the next step.
- No compile-counter change on any rank. No new allocator segments on the survivors in REPLACE;
  the spare allocated 7–11, and in notice runs every rank allocated 5.
- **The slow first step (about 0.55 s) did not occur in this run.** First tokens after resuming
  were 0.13–0.19 s in every fault run, the first ones included. On this pod it did occur in the
  build session's first REPLACE run (0.72 s) and first notice run (0.66 s), before the probe was in
  the tree. So it is once per pod and configuration, which fits a cache on disk filled on first
  use. The probe was never present when it happened.
- The sample classes overcount "compile": compiled code runs through torch's inductor wrappers,
  whose frames match the class. The compile counters are the reliable signal here.

## Surprises

- **The adapter is faster than stock** with stock's custom all-reduce on, by 1.3 %.
- **Notice pauses fell to 0.20–0.25 s** from 0.199–0.753 in the last run. The first notice run is
  no longer slow, for the reason above.
- **The REPLACE token mismatch survived the fix aimed at it.** See prediction 1.

## Dollars

| What | Cost |
|---|---|
| This run: pod `aj84thxtvfu0yf`, 22:13:55Z–22:33:51Z (19.9 min) at $7.95/h | **$2.64** (cap $10) |
| The build session before it: 5× H100 pod `eakvf8b9bkt3i3` (57.7 min at $17.45/h) and this pod 21:36:32Z–22:13:55Z (37.4 min) | $16.79 + $4.95 = **$21.74** (cap $25) |

The balance went from $17.82 to $15.17 over the run. The weights and `/dev/shm` were deleted, the
pod was terminated, and no pods remain.
