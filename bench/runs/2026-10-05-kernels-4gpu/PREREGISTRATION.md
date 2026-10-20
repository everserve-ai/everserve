# Pre-registration: the kernels on the four-GPU rig, peer-to-peer on

Status: PLANNED 2026-10-05, arms set by the owner. Spend: about 50 minutes on 5× A100 SXM (setup,
kernel build, 18 runs) at $7.95/h, about $6.60; hard cap $10, the laptop watchdog set from the
hourly rate to stop the pod before it. This is the last A100 measurement before the frontier run.

**Tree.** `98cf261`, or a later commit whose diff from it touches only `bench/`, named in `run.log`
with `scripts/preflight.sh` and CI green. That tree has:
- one gather kernel per step for the mirror (`6f935a1`);
- the slot collective in `core/csrc`, both variants (`c4f2c01`);
- concurrent aborts (`98cf261`).

## Why this run exists

The build session (`bench/runs/PLANNED/kernels-build-session/`) landed the three kernels and
measured them in development runs on one pod. This run measures them as pre-registered.

## The pod

- **Hardware.** 5× A100-SXM4-80GB on one NVLink host (GPUs 0–3 serve, GPU 4 the spare), or
  5× H100 on one NVLink host if the A100 is unavailable. The GPU and host are named in `run.log`.
- **Peer-to-peer on.** No `NCCL_P2P_DISABLE`; the topology goes in `data/topology.txt`.
- **Slot variant.** It must be peer-to-peer, as `everserve.comm.slotkernel.choose_variant` picks it
  on such a host. If the pod picks host memory, the run stops before its first arm and the report
  says so.
- **Data directory.** `/dev/shm/everserve`, with `df -h /dev/shm` in `run.log`. Staging files cannot
  be pinned on disk.

**Measured on the build session's pod (5× A100 SXM, P2P on, Qwen3-30B-A3B TP4 + EP, 8 sessions ×
2,048-token prompts, steps 10–200).** Development runs, not verdicts:

| Arm | tok/s | Median decode step |
|---|---|---|
| stock ×3 | 836.5–838.2 | **9.55–9.57 ms** |
| adapter, mirror off ×2 | 904.8–906.9 | 8.81–8.82 ms |
| adapter ×2 | 903.7–904.0 | 8.83–8.84 ms |

The adapter runs faster than stock because the slot collective beats vLLM's NCCL at decode sizes.
In a CUDA graph on that pod (`bench/harness/slot_bench.py`), 16,384 bf16 elements take 16–18 µs
peer-to-peer against 20.2 µs for pynccl, and 2,048 elements take 7.7 µs against 15.8.

## What runs

Same engine, model, prompts, decode lengths and harness as `2026-10-05-shm-commit-4gpu`, with
peer-to-peer on. The arms, three runs each, in this order, all on the one pod in one session:
1. stock, with vLLM's default custom all-reduce where the pod runs it
   (`adapter_port.sh stock_default`):
   - a stock run that fails with it on is repeated with it off;
   - `results.json` records per run which one ran (`custom_all_reduce`);
   - the overhead verdicts are against whatever stock ran. The adapter arms keep it off, since the
     graph-safe op replaces vLLM's TP all-reduce;
2. adapter with the mirror off;
3. adapter;
4. REPLACE between steps;
5. REPLACE mid-step;
6. notice with `--notice-steps 300` and 600 decode steps.

**Recording.**
- Metrics as the last two runs: the daemon on 127.0.0.1:9108 and the store on 127.0.0.1:9109, read
  after every run.
- Per rank: `commit.transport`, and the `fence_wait` and `event_lag` spans from the commit
  timeline.
- `reconcile_on_recover` as the adapter passes it.
- Every rank's event log in `flake/`.

**Every overhead is against stock on the same pod in the same session.** The overhead is negative
when the adapter is faster than stock.

## Predictions (kill conditions in bold)

1. **Adapter overhead** against stock: under 3 %, wall-clock tokens per second over steps 10–200,
   each arm's mean. **Kill: over 5 %.**
2. **Mirror-off overhead** against stock: under 1 %. **Kill: over 3 %.**
3. **REPLACE pause**, the largest inter-token gap of any session: under 1.5 s in every REPLACE
   run, both kill timings. **Kill: any run over 2.5 s.**
   - The term this band rests on is the abort: each survivor's `ncclCommAbort` of the live TP
     handle and of the EP group's handle, run together.
   - Each handle's abort time is recorded per run, and `RESULT.md` gives it its own column beside
     detect, change, restore, resume and first token.
4. **Notice pause**, the largest inter-token gap, with the swap happening in every notice run:
   under 0.3 s. **Kill: no swap in any run, any run over 1.0 s, or a fault event.**
5. **The store span.** The commit timeline's `sender` span, from the step's device copies landing
   to the store's acknowledgement (the sender thread's work and the store's round trip), has a p50
   under 1 ms on every rank in every run with the mirror on. **Kill: any rank's p50 over 3 ms.**
   - `fence_wait` and `event_lag` are reported per rank, beside it, without a verdict.
   - Restated by the owner before the run from `fence_wait`, for the reason below.
6. **The spare's restore** takes under 0.4 s in every REPLACE and notice run. **Kill: over 1.0 s.**
7. **Sessions continue with their KV in place.**
   - Every session reaches its last token.
   - The 20 tokens after the fault equal the adapter no-fault run's.
   - Every replay starts past the prompt.
   - **Kill: any session lost, re-prefilled, or with a mismatch.**
8. **Restore exactness.** The restored rows equal the dead rank's committed rows (digests per layer
   and head slot) in every REPLACE and notice run. **Kill: any mismatched table.**
9. **Commit before resume.** `MembershipCommitted` is heard within 100 ms of the engine resuming,
   with every survivor's and the spare's epoch applied. **Kill: no commit, or an epoch mismatch.**

### Evidence against predictions 3 and 5, stated before the run

**Prediction 3, the REPLACE pause.**
- Three development REPLACE runs on the build pod paused 1.20, 1.29 and 2.28 s.
- With peer-to-peer on, `ncclCommAbort` alone took 0.41–0.76 s per handle. The aborts now run
  together, but the slowest still sets the change: 0.39–0.55 s.
- Detection took 0.40–0.53 s in two runs and 1.36 s in the third. A detection like the third's puts
  a run near the kill.

**Why prediction 5 is on the store span and not the fence wait.**
- In the development adapter runs the step thread's commit p50 was 7.0 ms, and all of it was
  `fence_wait` (p50 6.95 ms). `event_lag` (commit queued to the copies landing) had a p50 of
  15.3 ms, about 1.7 decode steps.
- A high-priority side stream changed neither figure.
- With async scheduling the step thread runs about one step ahead of the GPU. The fence waits for
  the previous step's copies, and those copies only start when that step's forward ends. So
  `fence_wait` measures how far the CPU is ahead, not a cost: throughput with the mirror on equals
  throughput with it off (903.7–904.0 against 904.8–906.9 tok/s).
- The span that isolates the store's own time is `sender` (ready to acknowledged). In those runs it
  had a p50 of 0.51–0.62 ms per rank and a p90 of 1.2–2.3 ms.

## What is recorded

- **`results.json`:** every figure above per run, and per rank `commit.transport` with `fence_wait`
  and `event_lag`; the metrics per arm; the slot variant the pod picked; `reconcile_on_recover`.
- **`RESULT.md`:**
  - the verdict per prediction;
  - the overhead table, all arms from the pod;
  - the pause breakdown with each handle's `ncclCommAbort` time;
  - the driver's pause beside the daemon's change seconds, REPLACE and notice as separate rows;
  - surprises and dollars.
- **`run.log`** and **`data/`** as before.

## What this run does not claim

Frontier shapes, multi-node, or the host-memory variant's cost (peer-to-peer is the variant under
test). It makes no comparison with runs that had peer-to-peer off.
