# Pre-registration: the commit that never reads the device, on the four-GPU rig

Status: PLANNED 2026-10-05. Spend: about 40 minutes on 5× A40-class (about $1.70 at $2.45/h),
inside the standing $10 budget; hard cap $10, laptop watchdog as in the earlier runs.

Tree: `ef2fc73` or a later commit whose diff from it touches only `bench/`, named in `run.log`
before the first run, with `scripts/preflight.sh` green and CI green on it.

## Why this run exists

`bench/runs/2026-10-05-async-commit-4gpu/RESULT.md`:
- the asynchronous commit left the overhead at 27.0–29.0 %;
- there was zero back-pressure and essentially zero fence wait, but the step thread paid 13–17 ms
  per commit.

Decision 0010 traces that to a device read on the step thread inside `commit()`, which waited for
the whole forward pass. It removes every device read from the step thread:
- `save()` gathers each layer's held heads in one kernel and copies them to pinned memory on a
  side stream while later layers compute;
- `commit()` records one event;
- the sender thread reads the slot ids from pinned memory and drops padding there.

The fence and the two-step pipeline are unchanged. This run tests whether the step thread's
commit cost, and with it the commit's share of the overhead, is gone.

## What runs

**Engine and pod.**
- Engine: vLLM at the pin in `adapters/vllm/PIN`, no file under `vllm/` edited.
- Model: `Qwen/Qwen3-30B-A3B` bf16 at `/workspace/models/Qwen3-30B-A3B`, TP4 + EP, 8 sessions ×
  2,048-token prompts, greedy, seed 0.
- Pod: RunPod 5× A40 48 GB (GPUs 0–3 serve, GPU 4 the spare), community cloud if available, else
  secure; `NCCL_P2P_DISABLE=1`, `disable_custom_all_reduce=True`.
- The host's driver, topology and location go in `run.log`.

**Decode length.**
- 200 steps for the stock, mirror-off, adapter and REPLACE arms.
- 600 steps (`--max-model-len 2688`, 1,400 KV blocks) for the notice arm, so the 10 s deadline
  from a notice at step 100 falls inside the session.

**Arms**, three runs each, in this order, all on the one pod in one session:
1. stock;
2. adapter with the mirror off (`bench/harness/null_connector.py`): slot kernel, shadow sets,
   spare and daemon link resident, no commit;
3. adapter, no fault;
4. REPLACE, SIGKILL of rank 2 between steps after step 100;
5. REPLACE, SIGKILL of rank 2 half a median step into step 101;
6. notice for rank 2 after step 100 with a 10 s deadline, the leaving rank killed once the
   daemon's replacement is completed.

**Daemons and harness.**
- The daemon runs with `--metrics-addr 127.0.0.1:9108`, the store with `--metrics-addr
  127.0.0.1:9109`. Both are fresh per run and their `/metrics` text is read after every run,
  before they stop.
- `reconcile_on_recover` is whatever the adapter passes at the run's tree. It is recorded in
  `run.log` before the first run and in `results.json`.
- The driver, the restore check and the summariser are `bench/harness/adapter_drill.py`,
  `restore_check.py` and `summarize_adapter_port.py` at the run's tree; the definitions in the
  summariser's docstring hold.

**Every overhead in this run is against the stock arm measured on the same pod in the same
session.** It is never compared with an earlier run: the async-commit pod was a slower host (its
mirror-off share was 9.3–9.8 %, against week five's 4.5–4.6 % for the kernel).

## Predictions (kill conditions in bold)

1. **The step thread does not wait for the device.** The commit time measured on the step thread
   (the worker report's commit p50) is under 0.5 ms on every rank in every adapter, REPLACE and
   notice run. **Kill: any rank's p50 over 2 ms in any run.**
2. **The commit costs nearly nothing in steady state.** The adapter's overhead (arm 3) against
   stock (arm 1) is within 2 percentage points of the mirror-off arm's (arm 2), all three from
   this pod. Wall-clock tokens per second over steps 10–200, each arm's mean of three. **Kill: the
   adapter's overhead more than 5 points above the mirror-off arm's.**
3. **REPLACE pause**, as the largest inter-token gap of any session: 1.4–2.0 s in every REPLACE
   run, both kill timings. **Kill: any run over 2.5 s.**
   - The spare's restore is reported separately for every run.
   - On the async-commit pod it alone took 1.09–1.79 s and set the whole spread. This change does
     not touch the restore, so a slow host can miss the band on the restore alone, and the report
     will say so where it does.
4. **Sessions continue with their KV in place.** Every session reaches its last token. The 20
   tokens after the fault equal the adapter no-fault run's on the same seed. Every session's
   replay starts past its prompt. **Kill: any session lost, re-prefilled, or with a token
   mismatch in those 20.**
5. **Restore exactness.** In every REPLACE and notice run, the spare's restored rows equal the
   dead rank's committed rows (digests per layer and head slot, taken at the last step before the
   kill). **Kill: any mismatched table.**
6. **Commit before resume.** The daemon's `MembershipCommitted` is heard within 100 ms of the
   engine resuming, and every survivor and the spare applied its epoch. **Kill: no commit, or an
   epoch mismatch.**
7. **Notice pause**, as the largest inter-token gap: under 0.5 s in every notice run, with no
   `rank_lost` event in the executor and no fault in any runtime log. **Kill: any run over 1.0 s,
   or any fault event.**

## What is recorded

- **`results.json`:** every figure above per run, and:
  - per rank, from the worker report's `commit.transport`: the step thread's commit p50,
    `fence_waits_s`, `backpressure_waits`, `steps`, `bytes_sent`, `max_in_flight`;
  - per arm, from the metrics: `everserve_change_seconds` and `everserve_changes_total` by their
    `cause` and `outcome` labels, `everserve_store_commits_total`, `everserve_store_bytes_in_use`;
  - `reconcile_on_recover`.
- **`RESULT.md`:**
  - the verdict per prediction;
  - the overhead table with all three arms from this pod;
  - the pause breakdown (detect, change, restore, resume, first token), with the restore as its
    own column;
  - the driver's stopwatch pause beside the daemon's change seconds, with the REPLACE arm's and
    the notice arm's daemon change seconds as separate rows;
  - the commit pipeline per rank; surprises; dollars.
- **Elsewhere:**
  - every rank's event log in `flake/`;
  - every command, price and timestamp in `run.log`, with the price and running total at each arm;
  - raw outputs and the `/metrics` text in `data/` (not in git, SHA-256 in `results.json`).

Runs added after the fact are labelled post-hoc and reported apart from these verdicts.

## What this run does not claim

- Anything about the restore's speed, the P2P slot kernel, NVLink hosts, frontier shapes or
  multi-node.
- A comparison of overhead across pods.
