# Result: the commit-path battery, Qwen2.5 72B on 5× H100 (second attempt)

**Pod and software.**
- RunPod pod `647islznuyhxco`: 5× NVIDIA H100 80GB HBM3 SXM on one host, NVLink NV18 between every
  pair, secure cloud, JP, $17.45/h; 2026-10-06T22:55:46Z to 23:17:06Z.
- Driver **570.133.20** with `cuda-compat-13-0`. The frontier run and the decision 0020 development
  run had 580.126.09 on US hosts; this matters under prediction 6. vLLM at the pin; **no file under
  `vllm/` was edited**.
- Setup, the build and the model download took 6.5 minutes, against 17 on the first attempt.

**Tree.** `91e948b`; its product code is `5fd376d`, as pre-registered. Nothing was copied to the pod
after the tree.

**First attempt.** It ran on `a0b3378` and stopped at `adapter_1`, which failed forming the shadow
groups (`NOTE.md`). Nothing from it is scored here.

**Records.**
- `results.json`: every figure per run, the verdicts, the pod, the spend, and SHA-256 of
  `data/run2/`.
- `run.log`: commands, prices and timestamps for both attempts.
- `data/run2/`: raw outputs (not in git).
- `flake/`: every rank's event log for the fault runs.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | adapter overhead < 3 % (> 5 %) | **1.00 %, 0.95 %** (478.2, 478.5 against stock's 483.0 tok/s) | **alive** |
| 2 | adapter against mirror-off within 2 points (> 4) | 0.71, 0.66 points (mirror-off 481.6 tok/s, 0.29 %) | **alive** |
| 3 | store span `sender` p50 < 2 ms, every rank of every mirror-on run (any > 2 ms) | 0.60–1.17 ms | **alive** |
| 4 | `fence_wait` p50 < 1 ms, every rank of every mirror-on run (any > 3 ms) | 0.0014–0.0017 ms | **alive** |
| 5 | notice pause < 0.4 s with the swap (no swap, > 1.0 s); catch-up check | **0.203 s**, swapped; catch-up passes | **alive** |
| 6 | REPLACE pause 1.5–2.5 s (> 4 s) | **6.793 s**, of which detection 6.007 s | **dead** |
| 7 | sessions continue: 8/8 kept, identical after the fault, from the committed count, no re-prefill | 8/8 identical in the REPLACE and the notice; resumed from 2,149 and 2,480; none re-prefilled | **alive** |
| 8 | restored rows equal, every table (any mismatch) | 160/160 tables, REPLACE and notice | **alive** |
| 9 | commit before resume, epochs matched (a commit after resume) | 23.7 ms REPLACE, 59.1 ms notice; epochs matched | **alive** |

`adapter_2` matched `adapter_1` in every session's tokens. Every run made one commit message per
step.

## The commit path: decisions 0019 and 0020 hold at this shape

**Where it was.** The frontier run was 42–48 % behind stock. Its step thread waited on a one-step
fence, and the commit cost 160 messages per step.

**Now.**
- **Overhead:** 1 %.
- **Fence waits:** 0.44–0.69 s per rank over the whole run, against 8.1–8.3 s in the frontier run
  and about 4 s in development.
- **Store round trip:** `sender` p50 is 0.6–1.2 ms, against 2.1–3.4 ms before.
- **Registration:** each rank prepared its 160 tables in 683–710 ms, where 0020 moved them.

**The prefill's first commit still waits.** It is reported, not scored. Its fence span peaked at
350–760 ms per rank in the 201-token runs, and that peak is most of each rank's total fence wait.
So 0020 moved the page faults to registration, but the first commit still stalls the step thread
once, for about half a second.

## Prediction 6: the 6 s is detection, and the host is the leading suspect

| Run | Pause | Detect | Change | Restore | Resume | First token | Daemon change |
|---|---|---|---|---|---|---|---|
| replace_mid_1 | 6.793 | **6.007** | 0.266 | 0.354 | 0.024 | 0.137 | 0.618 |
| frontier mid 1–3 (US, driver 580) | 1.46–1.82 | 0.57–0.65 | 0.27–0.46 | 0.36–0.41 | 0.02 | 0.16–0.28 | 0.63–0.86 |

**What the record shows.**
- **Without detection, the pause is about 0.8 s,** half the frontier run's best. The terms after
  detection are the frontier run's or better.
- **The kill:** the drill sent SIGKILL to rank 2's worker at 23:12:29.83, and its exit code was -9.
- **Detection:** the executor's monitor waits on the worker's process sentinel. It saw the exit at
  23:12:35.84, 6.0 s later, and nothing was logged in between.
- **Where the time went:** the sentinel fires only once the kernel has torn the process down. So
  the worker took six seconds to finish exiting after the kill.

**What is not established.**
- **The host is the leading suspect.**
  - Every earlier REPLACE on H100 detected in 0.55–0.80 s, on a US host with driver 580.126.09.
    That includes the 0020 development run on `c63daa2`.
  - This host runs 570.133.20.
  - Every run used the same slot path (`p2p`), the same kill and the same model.
  - A driver that is slower to release a dead process's GPU and pinned memory would explain a
    clean six seconds.
- **Code changes are not ruled out.** The worker-side commits since `c63daa2` could also lengthen a
  dead worker's exit:
  - the catch-up;
  - sets of ranks;
  - pre-formed sets per spare;
  - the shadow-memory measure.
- **Nothing separates the two causes yet.** The 3× L40S check on `c23c6eb` detected in 0.36 s, but
  that was a 1.7B model on the host path, so it doesn't settle it.

**What would separate them.** One REPLACE arm on the same tree on a host with driver 580, about
$3. The drill would record when the process leaves the process table, so its exit is timed apart
from the monitor.

## Reported, not scored

- **Shadow sets:** 0.508 GB of device memory per member rank, measured at formation. Rank 2 has
  none; the spare used 0.81 GB in the fault runs. Formation took 1.47 s.
- **Slot collective:** `p2p`, confirmed by the probe on the first adapter start (exact sum on all 5
  devices, 4.67 s). The cache answered on every later start, at 0 s.
- **Notice:** 5.0 s of notice. The second restore copied 40,960 rows for the 256 positions
  committed after the watermark. It ended before the first step back, and the replay was one
  position.
- **Logits:** the notice run's logits matched its reference. Logits are not scored on H100.

## Dollars

| What | Cost |
|---|---|
| First attempt, `p267lcynio2uiu` (IN), 33.6 min | $9.78 |
| This attempt, `647islznuyhxco` (JP), 21.3 min at $17.45/h | **$6.20** (the balance fell $5.11) |
| **Battery total** | **$15.98** of the $30 cap |

The pod is terminated and its watchdog stopped; no pods remain.
