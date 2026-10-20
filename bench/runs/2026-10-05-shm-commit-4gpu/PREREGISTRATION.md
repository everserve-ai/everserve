# Pre-registration: the shared-memory data plane on the four-GPU rig

Status: PLANNED 2026-10-05; amended before any run, the same day, for the GPU (below). Spend:
about 40 minutes on 5× A100-SXM4-80GB (about $4.60 at $6.95/h community, $5.30 at $7.95/h
secure), inside the standing $10 budget; hard cap $10, the laptop watchdog set from the hourly
rate so the pod stops before it.

**GPU.**
- 5× A40 had no supply in either cloud for 25 minutes (`run.log`).
- Per the owner's decision, this run uses 5× A100-SXM4-80GB at the same cap. If SXM4 is
  unavailable when renting, it uses 5× A100 80GB PCIe, and `run.log` names which.
- Nothing in the verdicts compares across GPUs:
  - overhead is against stock on the same pod;
  - the commit, fence and restore figures are host-side;
  - the notice deadline is in decode steps.
- The REPLACE pause has an absolute band, so it is written for the A100 below.
- The host flags stay as in the A40 runs (`NCCL_P2P_DISABLE=1`, `disable_custom_all_reduce=True`)
  so the slot kernel and NCCL take the same host-memory path. NVLink, if present, is not used.

**Tree.** `3925d3b`, or a later commit whose diff from it touches only `bench/`. It is named in
`run.log` before the first run, with `scripts/preflight.sh` and CI green on it.

**The data directory.**
- The store runs with `--data-dir /dev/shm/everserve`.
- `df -h /dev/shm` on the pod is recorded in `run.log` before anything else. It must hold the
  tables (about 420 MB per rank) plus the staging files, so several GB.
- If the pod gives less, the run either restarts the pod with a larger `--shm-size`, or points
  `MDATA` at local disk. The latter is said in `RESULT.md`: the disk path goes through the page
  cache and is slower.

## Why this run exists

`bench/runs/2026-10-05-nosync-commit-4gpu/RESULT.md`:
- The step thread's commit cost was the fence: 4.4–5.6 ms per step, which is the sender thread's
  Python moving rows through the socket.
- The spare's restore moved 420 MB through the same socket in 1.37–1.74 s, which was most of the
  2.03–2.45 s REPLACE pause.

Decision 0011 takes rows off the socket:
- the store keeps tables as files on shared memory;
- a worker stages a step's rows in a pinned shared-memory file and commits by naming them;
- a restore maps the dead rank's tables and copies the written rows to the device.

This run tests whether both costs are gone.

## What runs

**Engine and pod.**
- Engine: vLLM at the pin in `adapters/vllm/PIN`, no file under `vllm/` edited.
- Model: `Qwen/Qwen3-30B-A3B` bf16 at `/workspace/models/Qwen3-30B-A3B`, TP4 + EP, 8 sessions ×
  2,048-token prompts, greedy, seed 0.
- Pod: RunPod 5× A100 80 GB (GPUs 0–3 serve, GPU 4 the spare), community cloud if available, else
  secure; `NCCL_P2P_DISABLE=1`, `disable_custom_all_reduce=True`.
- The host's driver, topology, location and `/dev/shm` size go in `run.log`.

**Decode length.**
- 200 steps for the stock, mirror-off, adapter and REPLACE arms.
- 600 steps (`--max-model-len 2688`, 1,400 KV blocks) for the notice arm.
  - The notice goes out after step 100 with `--notice-steps 300`: the deadline is 300 decode steps
    out, in seconds from the step time measured on the pod.
  - It therefore falls near step 400, inside the session, whatever the pod's step time.

**Arms**, three runs each, in this order, all on the one pod in one session:
1. stock;
2. adapter with the mirror off (`bench/harness/null_connector.py`);
3. adapter, no fault;
4. REPLACE, SIGKILL of rank 2 between steps after step 100;
5. REPLACE, SIGKILL of rank 2 half a median step into step 101;
6. notice for rank 2 as above; the leaving rank is killed once the daemon's replacement is
   completed.

**Daemons, metrics and harness.**
- The daemon runs with `--metrics-addr 127.0.0.1:9108`, the store with `--metrics-addr
  127.0.0.1:9109`. Both are fresh per run and their `/metrics` text is read after every run.
- `reconcile_on_recover` is whatever the adapter passes at the run's tree. It is recorded in
  `run.log` before the first run and in `results.json`.
- The driver, the restore check and the summariser are those in `bench/harness/` at the run's
  tree.

**Every overhead in this run is against the stock arm measured on the same pod in the same
session**, never against an earlier run.

## Predictions (kill conditions in bold)

1. **Step-thread commit.** The worker report's commit p50 on the step thread is under 1 ms on
   every rank in every adapter, REPLACE and notice run. **Kill: any rank over 3 ms in any run.**
2. **Fence wait per step.** Each rank's `commit.transport.fence_waits_s` divided by its `steps`
   averages under 1 ms.
   - This counts every step, including the first fence after the 400 MB prefill commit.
   - It applies to every rank in every adapter, REPLACE and notice run.
   - **Kill: any rank over 3 ms in any run.**
3. **Steady-state overhead.** The adapter's overhead (arm 3) against stock (arm 1) is within 2
   percentage points of the mirror-off arm's (arm 2), all three from this pod.
   - Wall-clock tokens per second over steps 10–200, each arm's mean of three.
   - **Kill: the adapter more than 5 points above the mirror-off arm.**
4. **The spare's restore** of the dead rank's rows, about 420 MB, takes under 0.4 s in every
   REPLACE run, as the spare reports it. **Kill: any run over 1.0 s.**
5. **REPLACE pause**, as the largest inter-token gap of any session: under 1.2 s in every REPLACE
   run, both kill timings, on the A100. **Kill: any run over 2.5 s.**
   - The band is the sum of: detection (0.34–0.39 s on the A40 pods, a host figure); the change
     (under 10 ms); a restore under 0.4 s (prediction 4); the resume (about 30 ms); and a first
     token after the tail replay of about 0.15 s on the faster GPU.
   - The A40 band of 1.5 s is not carried over.
6. **Notice.**
   - The daemon drives the replacement in all three notice runs.
   - The pause, as the largest inter-token gap, is under 0.5 s in every notice run.
   - There is no `rank_lost` event in the executor and no fault in any runtime log.
   - **Kill: no swap in any run, any run over 1.0 s, or any fault event.**
7. **Sessions continue with their KV in place.**
   - Every session reaches its last token.
   - The 20 tokens after the fault equal the adapter no-fault run's on the same seed.
   - Every session's replay starts past its prompt.
   - **Kill: any session lost, re-prefilled, or with a token mismatch in those 20.**
8. **Restore exactness.** In every REPLACE and notice run, the spare's restored rows equal the
   dead rank's committed rows (digests per layer and head slot, taken at the last step before the
   kill or the notice deadline). **Kill: any mismatched table.**
9. **Commit before resume.** The daemon's `MembershipCommitted` is heard within 100 ms of the
   engine resuming, and every survivor and the spare applied its epoch. **Kill: no commit, or an
   epoch mismatch.**

## What is recorded

- **`results.json`:** every figure above per run, and:
  - per rank, from `commit.transport`: the step thread's commit p50, `fence_waits_s`, `steps`,
    fence wait per step, `backpressure_waits`, `bytes_sent`, `max_in_flight`;
  - per arm, from the metrics: `everserve_change_seconds` and `everserve_changes_total` by `cause`
    and `outcome`, `everserve_store_commits_total`, `everserve_store_bytes_in_use`;
  - `reconcile_on_recover`;
  - the `/dev/shm` size.
- **`RESULT.md`:**
  - the verdict per prediction;
  - the overhead table with all three arms from this pod;
  - the pause breakdown (detect, change, restore, resume, first token);
  - the driver's pause beside the daemon's change seconds, with the REPLACE and notice rows
    separate;
  - the commit pipeline per rank; the data directory used; surprises; dollars.
- **Elsewhere:**
  - every rank's event log in `flake/`;
  - every command, price and timestamp in `run.log`, with the price and running total at each arm;
  - raw outputs and the `/metrics` text in `data/` (not in git, SHA-256 in `results.json`).

Runs added after the fact are labelled post-hoc and reported apart from these verdicts.

## What this run does not claim

Anything about the P2P slot kernel, NVLink hosts, frontier shapes, multi-node replication of the
tables, or a comparison of overhead across pods.
