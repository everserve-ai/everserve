# Result: the four-GPU battery on the exact replay (decision 0013), 5× A100

**Pod and software.**
- Battery 2026-10-05T23:54:41Z to 2026-10-06T00:16:45Z on RunPod pod `heippwhi15marw`: 5×
  A100-SXM4-80GB, NVLink (NV12, every pair), secure cloud, US, $7.95/h, driver 580.126.16 with
  `cuda-compat-13-0`. The build session's drills ran on the same pod before it
  (`bench/runs/PLANNED/exact-replay-session/`).
- vLLM at the pin (`6e517b15`). **No file under `vllm/` was edited.**
- Every arm ran with the fixed-order all-reduce settings and recorded its top-20 logprobs.

**Slot variant.** Peer-to-peer, reported by every worker of every adapter run.

**Tree.** The pre-registration's commit on `eb67ad3`: the exact replay (`d740223`) and the
drills' checks (`eb67ad3`), CI green. `PREREGISTRATION.md` is unchanged since its commit.

**Stock** ran with vLLM's custom all-reduce on in all three runs.

**Records.** `results.json` (every figure per run, the exactness checks, the determinism control,
SHA-256 of `data/`), `run.log`, `data/` (not in git), `flake/` (every rank's event log for the
REPLACE and notice runs).

20 runs, all pre-registered, in order, on this pod. Nothing was repeated or added afterwards.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | 8/8 sessions token-identical after REPLACE (any mismatch, loss, re-prefill) | **8/8 in 6/6**; all kept, none re-prefilled | **alive** |
| 2 | 8/8 sessions token-identical after notice (any mismatch, not checked) | **8/8 in 3/3** against `notice_ref` | **alive** |
| 3 | adapter overhead < 3 % (> 5 %) | −0.93, −0.85, −0.90 % | **alive** |
| 4 | mirror-off overhead < 1 % (> 3 %) | −1.81, −1.81, −1.85 % | **alive** |
| 5 | REPLACE pause < 1.5 s (> 2.5 s) | 1.081–1.282 s | **alive** |
| 6 | notice pause < 0.3 s, swap in all, no fault (no swap, > 1.0 s, fault) | 0.392, 0.408, 0.418 s; swap 3/3; 0 faults | **weakened** |
| 7 | store span `sender` p50 < 1 ms (> 3 ms) | 0.42–0.68 ms | **alive** |
| 8 | spare restore < 0.4 s (> 1.0 s) | 0.175–0.220 s | **alive** |
| 9 | restored rows equal the dead rank's (any mismatch) | 0 mismatched tables, 9/9 runs | **alive** |
| 10 | commit within 100 ms of resume, epochs applied | 28.6–45.4 ms REPLACE, 60.2–63.4 ms notice; epoch 1 everywhere | **alive** |

**The control.** The three no-fault runs of each arm agree token for token. `notice_ref` equals
`adapter_1`, tokens and logprobs, over their common 201 tokens.

## Decision 0013's checks, every fault run

| Check | REPLACE between ×3 | REPLACE mid ×3 | notice ×3 |
|---|---|---|---|
| K and V rows below the committed count byte-identical before the change and after the first step back, all 4 ranks | pass | pass | pass |
| first step back: one token per request from exactly the committed count, nothing re-added, FULL graph on every rank | pass | pass | pass |
| logits (top-20 logprobs) equal the no-fault reference's, every token, all 8 sessions | pass | pass | pass |

After every REPLACE and every notice, the output is the output without a change, down to the
logits, and the K and V the replay starts from are the ones that were committed.

## Overhead (all from this pod, this session)

| Arm | tok/s, steps 10–200 | Mean | Overhead against stock |
|---|---|---|---|
| stock, custom all-reduce on | 953.3, 950.2, 952.1 | 951.9 | — |
| adapter, mirror off | 969.1, 969.0, 969.5 | 969.2 | −1.82 % |
| adapter | 960.7, 960.0, 960.4 | 960.4 | −0.89 % |

These rates are about 3 % below the last battery's (982 stock) for every arm alike. This pod's
host differs (driver 580 with the compatibility libraries), and every arm now computes top-20
logprobs.

## The first step back, and the notice pause

- **The slow first step after a change is gone.** The survivors' first step after resuming now
  takes 14–16 ms of GPU (the spare 27 ms), against 89–126 ms with the old replay's eager prefill.
  It is a decode step on the captured graph.
- **REPLACE resumes in one step.** The dead rank's committed count is one position behind the
  sessions, so the first new token comes 0.106–0.126 s after resuming.
- **Notice resumes 27 positions back.** The spare restores during the notice window, before the
  swap, and its committed count (2,448) is 27 positions behind the sessions at the swap (2,475).
  The exact replay recomputes those 27 positions one per decode step, on the same graphs. The first
  new token comes 0.340–0.360 s after resuming, and the largest gap is 0.39–0.42 s, over
  prediction 6's 0.3 s.
- **That is the cost of exactness under the current restore timing, not a regression in the
  replay.** The old replay did the same 27 positions as one eager chunk: faster, and the source of
  the divergence. Restoring the spare's rows up to the swap point instead of the notice point would
  leave one position to replay. That is a change to the restore's timing, not made here.

## The REPLACE pause, with each abort

| Run | Pause | Detect | Change | Abort, live TP (ranks 0/1/3) | Abort, EP (ranks 0/1/3) | Restore | Resume | First token | Daemon change |
|---|---|---|---|---|---|---|---|---|---|
| between 1 | 1.244 | 0.502 | 0.392 | 0.396 / 0.540 / 0.539 | 0.216 / 0.589 / 0.590 | 0.205 | 0.041 | 0.107 | 0.596 |
| between 2 | 1.282 | 0.481 | 0.474 | 0.289 / 0.325 / 0.326 | 0.606 / 0.653 / 0.653 | 0.185 | 0.039 | 0.106 | 0.657 |
| between 3 | 1.082 | 0.421 | 0.322 | 0.253 / 0.250 / 0.254 | 0.517 / 0.514 / 0.512 | 0.200 | 0.033 | 0.108 | 0.521 |
| mid 1 | 1.098 | 0.414 | 0.364 | 0.253 / 0.170 / 0.161 | 0.342 / 0.341 / 0.533 | 0.175 | 0.035 | 0.107 | 0.538 |
| mid 2 | 1.169 | 0.455 | 0.355 | 0.466 / 0.535 / 0.477 | 0.291 / 0.354 / 0.356 | 0.185 | 0.045 | 0.126 | 0.539 |
| mid 3 | 1.081 | 0.400 | 0.320 | 0.332 / 0.333 / 0.324 | 0.411 / 0.404 / 0.534 | 0.220 | 0.029 | 0.111 | 0.539 |

The resume now includes one RPC to every worker re-adding the requests; resume stays 29–45 ms.

## Driver pause against the daemon's change

| Kind | Runs | Driver pause | Daemon `everserve_change_seconds` |
|---|---|---|---|
| REPLACE | between 1–3 | 1.244, 1.282, 1.082 | 0.596, 0.657, 0.521 |
| REPLACE | mid 1–3 | 1.098, 1.169, 1.081 | 0.538, 0.539, 0.539 |
| notice | 1–3 | 0.392, 0.408, 0.418 | 0.221, 0.215, 0.218 |

## Dollars

| What | Cost |
|---|---|
| This battery: 23:54:41Z–00:16:45Z (22.1 min) at $7.95/h | $2.92 |
| The build session's drills and setup on the same pod before it: 23:35:15Z–23:54:41Z | $2.58 |
| **Session total** (owner's cap $8) | **$5.50** |

The balance went from $9.31 to $3.21. That $6.10 includes the previous session's pods' charges
that had not yet posted. The weights and `/dev/shm` were deleted, the pod was terminated, and no
pods remain.
