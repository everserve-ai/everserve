# Result: the commit that never reads the device, on the four-GPU rig

**Pod and software.**
- Run 2026-10-05T10:56Z to 11:27Z on RunPod pod `1r13xzft9kn58i`: 5× A40 48 GB, secure cloud, SE,
  $2.45/h.
- No 5× A40 in either cloud from 10:40Z; obtained on the tenth polling round, every attempt in
  `run.log`.
- Driver 580.159.04 with `cuda-compat-13-0`; PCIe topology in `data/topology.txt`.
- vLLM at the pin (`6e517b15`), torch 2.13.0+cu130, NCCL 2.29.7.

**Tree.**
- Commit `8e51114` (CI green): decision 0010, and `reconcile_on_recover=False` in the adapter.
- `PREREGISTRATION.md` is unchanged since that commit. **No file under `vllm/` was edited.**

**Records.**
- `results.json`: every figure per run, the metrics per arm, the commit pipeline per rank.
- `run.log`: commands, prices and timestamps.
- `data/`: raw outputs and the `/metrics` text after every run (not in git, SHA-256 in
  `results.json`).
- `flake/`: every rank's event log for the REPLACE runs. The no-fault and notice runs wrote none,
  because no change happened in them.

18 runs, all pre-registered, in the pre-registered order, all on this pod. One dev smoke run
before them is not scored. Nothing was repeated or added afterwards.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | step-thread commit p50 < 0.5 ms on every rank (any > 2 ms) | 5.5–6.6 ms on every rank in every run with the mirror on | **dead** |
| 2 | adapter overhead within 2 points of mirror-off, same pod (> 5 points above) | adapter 13.1–13.9 % (mean 13.5), mirror-off 9.2–10.0 % (mean 9.6): 3.9 points above | **weakened** |
| 3 | REPLACE pause 1.4–2.0 s every run (any > 2.5 s) | 2.03–2.45 s; all six above the band, none over the kill | **weakened** |
| 4 | sessions continue, 20 tokens equal, replay past the prompt | 8/8 sessions, 8/8 equal, replay from 2,149 tokens, 6/6 runs | **alive** |
| 5 | restored rows equal the dead rank's (any mismatch) | 48/48 tables, 6/6 REPLACE runs; the notice runs restored nothing | **alive** (REPLACE) |
| 6 | commit within 100 ms of resuming, epochs applied | 25.6–31.8 ms; epoch 1 on every survivor and the spare | **alive** |
| 7 | notice pause < 0.5 s, no fault (any > 1.0 s, or a fault) | no swap happened | **not tested** |

**Prediction 7 was not tested, because of my protocol error.**
- In all three notice runs the sessions ended before the 10 s deadline, by 0.19, 0.16 and 0.17 s.
  The daemon logged the notice and never drove a change, and the executor saw no fault.
- The pre-registration placed the deadline "near step 385" from a 35 ms decode step. That figure
  was wrong even for the earlier pods: their adapter ran 331–337 tok/s over 8 sessions, about
  24 ms a step. On this pod a step took about 20 ms (400 tok/s), so 600 steps ended right at the
  deadline.
- The next notice arm should size the session from the step time measured on the pod, or put the
  deadline a fixed number of steps out, not a fixed number of seconds.

## Overhead (all from this pod, this session)

| Arm | tok/s, steps 10–200 | Overhead vs this pod's stock |
|---|---|---|
| stock ×3 | 461.4 / 466.5 / 466.0 (mean 464.6) | n/a |
| adapter, mirror off ×3 | 420.8 / 421.7 / 418.0 | 9.2–10.0 % |
| adapter ×3 | 400.1 / 402.3 / 403.9 | 13.1–13.9 % |

- **Commit share: 3.9 points**, down from about 19 on the async-commit pod. That pod was a
  different host, so this is the direction of the change, not a measured cross-pod difference.
- **The rest, about 9.6 %, is the slot kernel and the resident pieces.** The mirror never touches
  them.

## The step thread's commit and the fence

From the worker report: the commit p50 on the step thread and `commit.transport` per rank. All
figures in `results.json`.

| Runs | Step-thread commit p50 (ms), per rank | Fence waits (s), per serving rank | Back-pressure |
|---|---|---|---|
| adapter ×3 | 6.0–6.5 | 5.66–5.89 over 202 steps | 0 |
| REPLACE between ×3 | 5.8–6.6 | 6.1–6.7 over 204 steps; rank 2: 0.55–0.56 over 100 | 0 |
| REPLACE mid ×3 | 5.5–6.3 | 5.4–5.7 over 204 steps; rank 2: 0.44–0.54 over 100 | 0 |
| notice ×3 (no swap) | 6.0–6.5 | 7.9–8.3 over 602 steps | 0 |

- **The device read is gone. The fence is what remains.**
  - The step-thread commit fell from 13–17 ms (async-commit run) to 5.5–6.6 ms.
  - What it now pays is about one fence per step: rank 2's fence waits average 4.4–5.6 ms per step
    over its 100 steps.
  - In 600-step runs the fence total grows with the step count: 7.9–8.3 s over 602 steps, against
    5.4–6.7 s over about 200.
- **So the fence does not have a step of slack.** It waits for the previous step's store
  acknowledgement, and that acknowledgement arrives about one decode step late.
- I did not profile where the sender thread spends that time: waiting for the gather event,
  framing the rows in Python, or the store's round trip. That is the next thing to measure.
- `max_in_flight` was 2 and back-pressure 0 in every run.

## Pause: the driver's stopwatch and the daemon's change seconds

The two clocks measure different spans:
- **Pause** is the driver's largest inter-token gap of any session.
- **Daemon change** is `everserve_change_seconds{cause, outcome}`, from the reported loss to the
  last acknowledgement. That includes the spare's restore.

**REPLACE arm** (`cause="fault",outcome="replace"`):

| Run | Pause (driver) | Daemon change | Detect | Change − restore | Restore | Resume | First token |
|---|---|---|---|---|---|---|---|
| replace_between_1 | 2.163 | 1.583 | 0.343 | 0.005 | 1.581 | 0.026 | 0.211 |
| replace_between_2 | 2.265 | 1.651 | 0.354 | 0.005 | 1.648 | 0.028 | 0.230 |
| replace_between_3 | 2.025 | 1.440 | 0.341 | 0.006 | 1.437 | 0.031 | 0.214 |
| replace_mid_1 | 2.329 | 1.658 | 0.374 | 0.005 | 1.654 | 0.027 | 0.259 |
| replace_mid_2 | 2.448 | 1.748 | 0.347 | 0.006 | 1.744 | 0.028 | 0.314 |
| replace_mid_3 | 2.067 | 1.379 | 0.389 | 0.007 | 1.374 | 0.032 | 0.257 |

**Notice arm** (`cause="notice"`): no series. No change happened in any notice run, so the daemon
recorded none.

**The restore set the pause band again.** The spare reads about 420 MB through the store in
1.37–1.74 s. With detection (0.34–0.39 s) and the first token (0.21–0.31 s), that alone puts every
run above 2.0 s. This decision does not touch the restore; on this pod it is about 0.4 s slower
than on the adapter-port pod.

## Metrics

Read after every run; per arm in `results.json`, raw text in `data/`.

| Metric | Value |
|---|---|
| `everserve_changes_total{cause="fault",outcome="replace"}` | 1 per REPLACE run; no series in the other arms |
| `everserve_store_commits_total` | adapter ≈ 1,000 per run, REPLACE ≈ 1,000 (the spare's commits after it joins), notice ≈ 2,600 |
| `everserve_store_bytes_in_use` | about 2.0 GB at 200 steps, about 2.2 GB at 600, 0 with the mirror off |

The per-run values are in `results.json` under `metrics_per_arm`.

## Reconciliation

`reconcile_on_recover=False`, as the adapter passed it. It is recorded in `run.log` before the run
and in `results.json`. Every REPLACE run replayed from 2,149 tokens.

## Surprises

- **Removing the device read uncovered the fence.** The step thread's commit cost now tracks one
  store acknowledgement per step, where decision 0009 expected it to overlap the next forward.
- **The notice arm was mis-sized a second time,** in the opposite direction from the adapter-port
  run. I sized it from a 35 ms step that no pod has shown; the adapter's step is about 24 ms on
  the earlier pods and about 20 ms here.

## Dollars

| What | Cost |
|---|---|
| Pod `1r13xzft9kn58i`, 5× A40, 31.5 min at $2.45/h | $1.29 |

That covers setup, one dev smoke and 18 runs. The balance went from $47.29 to $46.16 ($1.13), a debit
that lags the final minutes. The weights were deleted, the pod was terminated at
11:27:42Z, the laptop watchdog was stopped, and no pods remain. This is under the $10 cap.
