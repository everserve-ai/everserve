# Result: the asynchronous mirror commit on the four-GPU rig

**Pod and software.**
- Run 2026-10-05T04:09Z to 04:38Z on RunPod pod `zq3b5gp661jn6k`: 5× A40 48 GB, secure cloud, CA,
  $2.45/h (community had no 5× A40).
- Driver 570.195.03 with `cuda-compat-13-0`: an older driver and a different PCIe topology from the
  adapter-port pod.
- vLLM at the pin (`6e517b15`), torch 2.13.0+cu130, NCCL 2.29.7.

**Tree.**
- Commit `1e07ade` (CI green): the asynchronous commit with the one-step fence (decision 0009) and
  `reconcile_on_recover=False` in the adapter.
- `PREREGISTRATION.md` is unchanged since `2d8065e`. **No file under `vllm/` was edited.**

**Records.**
- Every figure per run is in `results.json`, including the daemon's and the store's metrics per
  arm and the commit pipeline per rank.
- Commands, prices and timestamps are in `run.log`.
- Raw outputs, including the `/metrics` text after every run, are in `data/` (not in git, SHA-256
  in `results.json`).
- Every rank's event log is in `flake/`. The no-fault arms write none.

18 runs, all pre-registered, in the pre-registered order. One dev smoke run before them is not
scored. No run was repeated or added after the fact.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | REPLACE pause 1.4–2.0 s in every run (any > 2.5 s) | 1.73, 1.96, 2.37 (between); 2.36, 1.77, 2.35 (mid) | **weakened**: 3 of 6 in the band, none over the kill |
| 2 | sessions continue, 20 tokens equal, replay past the prompt (loss, re-prefill, mismatch) | 8/8 sessions, 8/8 equal, replay from 2,149 tokens, 6/6 runs | **alive** |
| 3 | adapter overhead < 2.0 % (> 4.0 %) | 27.0–29.0 % (stock 466.7 tok/s, adapter 331–337) | **dead** |
| 4 | notice pause < 0.5 s, no fault (any > 1.0 s, or a fault) | 1.023, 0.334, 0.344 s; 0 fault events | **dead**: one run over the kill by 23 ms |
| 5 | commit off the step path: step-thread p50 < 1 ms, back-pressure < 1 % (p50 > 2 ms, or > 5 %) | back-pressure 0 waits on every rank in every run; step-thread p50 13.3–17.1 ms | see below |
| 6 | restored rows equal the dead rank's (any mismatch) | 48/48 tables, 9/9 runs (REPLACE and notice) | **alive** |
| 7 | commit within 100 ms of resuming, epochs applied (none, or mismatch) | 24.9–36.2 ms REPLACE, 60.3–66.4 ms notice; epoch 1 on every survivor and the spare | **alive** |

**Prediction 5 has two clauses, and they disagree.**
- The run instruction judges it as back-pressure waits over steps. That is 0 in every run, so on
  that reading it is **alive**.
- The pre-registered text also kills on a step-thread p50 above 2 ms. The p50 is 13.3–17.1 ms, so
  on that reading it is **dead**.
- Both are recorded; which one stands is the owner's call.
- **The cost is in `commit()` itself, not the fence.** Rank 2's fence waited 0.00–0.01 s in total
  over its 100 steps before the kill. It still paid 13.3–14.4 ms p50 per step on the step thread.

## Overhead, split (prediction 3)

| Arm | tok/s, steps 10–200 | Overhead vs stock |
|---|---|---|
| stock ×3 | 461.8 / 469.4 / 468.9 | n/a |
| adapter, mirror off ×3 (slot kernel, shadow sets, spare, daemon link; no commit) | 423.4 / 422.6 / 421.1 | 9.3–9.8 % |
| adapter ×3 | 331.3 / 333.2 / 337.0 | 27.8–29.0 % |

**Commit share: about 19 points.** The adapter's total, 27.8–29.0 %, is the synchronous port's
27.8 % again. The asynchronous pipeline never filled:
- back-pressure was 0;
- at most 2 steps were in flight;
- about 0.2 MB was sent per decode step per rank (8 tokens × 48 layers × 512 B);
- the fence waits, 3.2–5.3 s per rank per run, are almost all the prefill's 400 MB.

The per-step time is spent before the job is queued.

**Kernel and resident share: 9.3–9.8 % on this host**, against week five's 4.5–4.6 % on the
SE host. This host is slower in the same pieces: detection here was 0.34–0.39 s, against
0.23–0.30 s on the adapter-port pod.

## Pause: the driver's stopwatch and the daemon's change seconds

The two clocks measure different spans:
- **Pause** is the driver's largest inter-token gap, from the last token before the kill to the
  first after it.
- **Daemon change** is `everserve_change_seconds`, from the reported loss to the last
  acknowledgement. That includes the spare's restore, which happens before it acknowledges.

| Run | Pause (driver) | Daemon change | Detect | Change − restore | Restore | Resume | First token |
|---|---|---|---|---|---|---|---|
| replace_between_1 | 1.959 | 1.328 | 0.393 | 0.005 | 1.325 | 0.036 | 0.201 |
| replace_between_2 | 1.733 | 1.099 | 0.391 | 0.006 | 1.094 | 0.033 | 0.210 |
| replace_between_3 | 2.373 | 1.791 | 0.353 | 0.005 | 1.788 | 0.026 | 0.203 |
| replace_mid_1 | 2.359 | 1.759 | 0.338 | 0.312 | 1.448 | 0.025 | 0.224 |
| replace_mid_2 | 1.772 | 1.158 | 0.376 | 0.004 | 1.155 | 0.034 | 0.190 |
| replace_mid_3 | 2.350 | 1.717 | 0.366 | 0.005 | 1.714 | 0.032 | 0.222 |
| notice_1 | 1.023 | 1.643 | n/a | n/a | 1.639 (before the deadline) | 0.061 | 0.977 |
| notice_2 | 0.334 | 1.730 | n/a | n/a | 1.726 (before the deadline) | 0.060 | 0.186 |
| notice_3 | 0.344 | 1.558 | n/a | n/a | 1.554 (before the deadline) | 0.066 | 0.207 |

**The restore is the whole spread in prediction 1.** The spare reads about 420 MB from the store
through Python protobuf, and that took 1.09–1.79 s. On the adapter-port host it took 1.03–1.24 s.
- On a notice the restore runs before the deadline and stays off the pause, which is why the notice
  pause is shorter than the daemon change.
- **notice_1's pause is its first token:** 0.98 s against 0.19–0.21 s in the other two notice
  runs. The replay recomputes each session's tail after its replay point (2,581–2,587 tokens).
  That run's tail cost more; the logs do not show why.

## Metrics

Read after every run; per arm in `results.json`, raw text in `data/`.

| Metric | Value |
|---|---|
| `everserve_changes_total` | 1 per REPLACE and notice run, 0 elsewhere |
| `everserve_store_commits_total` | adapter 996; REPLACE 999–1,002 (the spare's commits after it joins); notice 2,604 |
| `everserve_store_bytes_in_use` | 2.04 GB (200 tokens), 2.20 GB (600 tokens), 0 with the mirror off |

**The daemon labels a change by its result, not its outcome.** It reports
`outcome="committed",result="committed"` on `everserve_changes_total` and `outcome="committed"`
on the histogram, so the series cannot tell REPLACE from a notice.

## Commit pipeline per rank

From the worker report, `commit.transport`; all in `results.json`.

- **Every rank, every run:**
  - `max_in_flight` is 2;
  - `backpressure_waits` is 0;
  - `bytes_sent` is about 443 MB per serving rank for 200 tokens and about 534 MB for 600;
  - `fence_waits_s` is 3.2–5.3 s per serving rank per run, almost all of it the prefill.
- **Rank 2 before the kill:**
  - REPLACE runs: 100 steps, 20.4 MB;
  - notice runs: 6–7 steps, then the replacement.

## Reconciliation

`reconcile_on_recover=False`, as the adapter passed it (`adapters/vllm/everserve_vllm/worker.py`).
It is recorded in `run.log` before the run and in `results.json`.

The boundary applies the change on every survivor together, and the scheduler replays from the
replaced rank's own committed counts. Every REPLACE run replayed from 2,149 tokens, the same point
as the synchronous port. The one-step fence cost no extra replay here.

## Surprises

- **Moving the commit off the step path did not move its cost.** The step thread pays 13–17 ms
  per step with no back-pressure and no fence wait. Whatever `commit()` does before handing the job
  to the sender thread still synchronises with the step. That is the next thing to profile.
- **The host matters.** The kernel and resident share doubled against week five's SE host, and
  the restore and detection are slower too. Comparisons across runs need the same host, or a stock
  baseline on each, which this run has.
- **The daemon's change metric labels** carry the result where the outcome was expected.
- **A dev smoke run gave a 2.80 s pause.** It is not scored, but it is within the scored spread.

## Dollars

| What | Cost |
|---|---|
| Pod `zq3b5gp661jn6k`, 5× A40, 29 min at $2.45/h | $1.18 |

That covers setup, one dev smoke and 18 runs. The balance went from $48.51 to $47.51, a $1.00
debit that lags the final minutes. The weights were deleted, the pod was terminated at 04:38:05Z,
the laptop watchdog was stopped, and no pods remain. This is under the $10 cap and the $15
ceiling.
