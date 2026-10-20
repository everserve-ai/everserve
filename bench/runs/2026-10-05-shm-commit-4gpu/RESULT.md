# Result: the shared-memory data plane, on 5× A100

**Pod and software.**
- Run 2026-10-05T13:03Z to 13:33Z on RunPod pod `ivwa29yn1yec58`: 5× A100-SXM4-80GB, secure cloud,
  US, $7.95/h.
- 5× A40 had no supply for 25 minutes. The pre-registration was amended for the A100, by the
  owner's decision, before any run (`1418c10`), and the A100's community cloud had no supply.
- Driver 580.126.16 with `cuda-compat-13-0`. The GPUs are linked by NVLink (NV12, every pair) but
  run with `NCCL_P2P_DISABLE=1` as pre-registered, so the slot kernel and NCCL take the host-memory
  path. Topology in `data/topology.txt`.
- vLLM at the pin (`6e517b15`), torch 2.13.0+cu130, NCCL 2.29.7.

**Data directory.** `df -h /dev/shm` was 583 GB, so the store ran with `--data-dir
/dev/shm/everserve` as pre-registered, with no disk fallback.

**Tree.** `1418c10` (CI green): decision 0011, and `reconcile_on_recover=False` in the adapter.
`PREREGISTRATION.md` is unchanged since then. **No file under `vllm/` was edited.**

**Records.**
- `results.json`: every figure per run, including the metrics per arm, the commit pipeline per
  rank, and the survivors' abort timings.
- `run.log`: commands, prices and timestamps.
- `data/`: raw outputs and the `/metrics` text (not in git, SHA-256 in `results.json`).
- `flake/`: every rank's event log for the REPLACE and notice runs.

18 runs, all pre-registered, in order, on this pod. One dev smoke run before them is not scored.
Nothing was repeated or added afterwards.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | step-thread commit p50 < 1 ms every rank (> 3 ms) | 3.35–4.04 ms on every rank in every run with the mirror on | **dead** |
| 2 | fence wait per step < 1 ms average every rank (> 3 ms) | full-length ranks 7.8–13.0 ms; rank 2 over its 100 steps before the kill 3.3–3.9 ms | **dead** |
| 3 | adapter overhead within 2 points of mirror-off, same pod (> 5 points above) | adapter 10.3–11.0 % (mean 10.6), mirror-off 3.6–4.9 % (mean 4.2): 6.4 points | **dead** |
| 4 | spare restore of ~420 MB < 0.4 s (> 1.0 s) | 0.17–0.20 s in every REPLACE and notice run | **alive** |
| 5 | REPLACE pause < 1.2 s on A100 (> 2.5 s) | 1.21–1.71 s | **weakened** |
| 6 | notice: swap in all three, pause < 0.5 s, no fault (no swap, > 1.0 s, a fault) | swap in 3/3; 0.829, 0.270, 0.253 s; 0 fault events | **weakened** |
| 7 | sessions continue, 20 tokens equal, replay past the prompt | 8/8 sessions, 8/8 equal, replay from 2,149 tokens, 6/6 runs | **alive** |
| 8 | restored rows equal the dead rank's (any mismatch) | 48/48 tables, 9/9 runs | **alive** |
| 9 | commit within 100 ms of resuming, epochs applied | 18.1–26.2 ms REPLACE, 46.5–70.7 ms notice; epoch 1 everywhere | **alive** |

## What moved, and what did not

- **The restore is solved.** The 420 MB that took 1.37–1.74 s through the socket on the A40 pods
  now takes 0.17–0.20 s by mapping the dead rank's tables. That is a different GPU and host; the
  pre-registered bound is 0.4 s and the figure is well inside it.
- **The step thread and the fence are not.**
  - The step-thread commit p50 is 3.4–4.0 ms. On the faster GPU that is about a third of a decode
    step: stock runs at 738.8 tok/s, about 10.8 ms per step.
  - Ranks that ran the full 200 steps averaged 7.8–13.0 ms of fence wait per step.
  - Rank 2, which only ran the first 100 steps before its kill, averaged 3.3–3.9 ms.
- **So the fence wait grows over the run, and its mean is far above the commit p50.** Most steps
  wait a few milliseconds and some wait much longer. The per-step distribution is not recorded, so
  this run cannot say which steps those are, or whether it is the store's copy, the staging pool,
  or the socket. That is the next measurement.
- **The 400 MB prefill commit is not the explanation for the 200-step ranks.** Rank 2 committed
  the same prefill and averaged 3.3–3.9 ms.

## Overhead (all from this pod, this session)

| Arm | tok/s, steps 10–200 | Overhead vs this pod's stock |
|---|---|---|
| stock ×3 | 744.5 / 728.0 / 744.0 (mean 738.8) | n/a |
| adapter, mirror off ×3 | 712.0 / 702.9 / 708.9 | 3.6–4.9 % |
| adapter ×3 | 660.7 / 662.6 / 657.9 | 10.3–11.0 % |

The commit's share is 6.4 points. The step is shorter here, so the same milliseconds are more
points than on the A40.

## Pause: the driver's stopwatch and the daemon's change seconds

The two clocks measure different spans:
- **Pause** is the driver's largest inter-token gap.
- **Daemon change** is `everserve_change_seconds{cause, outcome}`, from the reported loss to the
  last acknowledgement, including the spare's restore.

**REPLACE arm** (`cause="fault",outcome="replace"`):

| Run | Pause (driver) | Daemon change | Detect | Change − restore | Restore | Resume | First token |
|---|---|---|---|---|---|---|---|
| replace_between_1 | 1.639 | 1.088 | 0.337 | 0.917 | 0.172 | 0.025 | 0.188 |
| replace_between_2 | 1.230 | 0.682 | 0.347 | 0.483 | 0.200 | 0.018 | 0.184 |
| replace_between_3 | 1.509 | 0.882 | 0.415 | 0.698 | 0.184 | 0.026 | 0.185 |
| replace_mid_1 | 1.709 | 1.113 | 0.366 | 0.924 | 0.190 | 0.022 | 0.201 |
| replace_mid_2 | 1.213 | 0.601 | 0.370 | 0.404 | 0.198 | 0.022 | 0.214 |
| replace_mid_3 | 1.662 | 1.086 | 0.362 | 0.894 | 0.193 | 0.018 | 0.189 |

**Notice arm** (`cause="notice",outcome="replace"`):

| Run | Pause (driver) | Daemon change | Restore (before the deadline) | Resume | First token |
|---|---|---|---|---|---|
| notice_1 | 0.829 | 0.196 | 0.193 | 0.046 | 0.791 |
| notice_2 | 0.270 | 0.195 | 0.193 | 0.071 | 0.210 |
| notice_3 | 0.253 | 0.208 | 0.204 | 0.049 | 0.212 |

**The change segment is now the NCCL aborts.** "Change − restore" was 0.003–0.006 s on every A40
pod. Here it is 0.40–0.92 s. The survivors' recorded abort times (`results.json`, `aborts`) show:
- the live TP handle aborting in 0.07–0.57 s;
- then the EP group's handle in 0.14–0.51 s, one after the other;
- the slowest survivor sets the change.

A notice skips the aborts, and its change is 0.195–0.208 s, all of it the restore. Two follow-ups:
- run the other groups' aborts concurrently with the live one, or leave a refused handle unaborted
  until the boundary;
- find out why `ncclCommAbort` is slow on this NVLink host with P2P disabled.

**notice_1's pause is its first token**, 0.79 s against 0.21 s in the other two. The same happened
on the async-commit pod (0.98 s against 0.19–0.21 s). Both times it was the first notice run, and
each notice run replays a token count no earlier run on that pod had shown. A first compile of
that shape, with later runs reading it from the shared compile cache, would explain it. That is a
hypothesis; the logs do not show the compile.

## Commit pipeline per rank

All per run in `results.json`:
- `max_in_flight` was 2 and `backpressure_waits` 0 in every run.
- Fence wait per step: as above.

## Metrics

| Metric | Value |
|---|---|
| `everserve_changes_total` | `{cause="fault",outcome="replace"}` 1 per REPLACE run; `{cause="notice",outcome="replace"}` 1 per notice run |
| `everserve_change_seconds` | as in the two tables above |
| `everserve_store_commits_total`, `everserve_store_bytes_in_use` | per run in `results.json` under `metrics_per_arm` |

## Reconciliation

`reconcile_on_recover=False`, as the adapter passed it. It is recorded in `run.log` before the run
and in `results.json`. Every REPLACE run replayed from 2,149 tokens, every notice run from
2,447–2,448.

## Surprises

- **The restore went from the largest term to one of the smallest,** and the NCCL aborts took its
  place in the REPLACE pause.
- **The fence wait grows through a run,** and its mean (8–13 ms) is far above the step-thread p50
  (3.4–4.0 ms).
- **The mirror-off overhead is lower here (3.6–4.9 %)** than on any A40 pod (9.2–10.0 %). The slot
  kernel's cost is a host-memory figure and this host is faster at it.
- **The first notice run's first token was slow on both pods** that ran the notice arm. See the
  hypothesis above.

## Dollars

| What | Cost |
|---|---|
| Pod `ivwa29yn1yec58`, 5× A100 SXM 80 GB, 30.1 min at $7.95/h | $3.99 |

That covers setup, one dev smoke and 18 runs; the 25 minutes of A40 polling cost nothing. The
balance went from $45.98 to $41.98. The weights and `/dev/shm/everserve` were deleted, the pod
was terminated at 13:33:17Z, the laptop watchdog was stopped, and no pods remain. This is under
the $10 cap.
