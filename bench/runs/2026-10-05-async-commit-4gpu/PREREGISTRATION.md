# Pre-registration: the asynchronous mirror commit on the four-GPU rig

Status: DRAFT 2026-10-05. Spend: about 1.5 GPU-hours on 5× A40-class (about $3.70 at $2.45/h),
inside the standing $10 budget; hard cap $10, laptop watchdog as in the adapter-port run.

Prerequisites, all landed before this file's predictions are committed:
- the asynchronous commit in `MirrorCommittedKV` and the adapter (below), with `docs/CONTRACT.md`
  item 3 amended: the commit leaves the step path;
- the runtime's recover-time reconciliation as wired by its owner (`Runtime.reconcile_on_recover`),
  with its setting under REPLACE recorded here before the run;
- the drill driver's arm with the adapter resident and the connector off (arm 2 below);
- `scripts/preflight.sh` green on the commit the run uses, which is named in `run.log`.

## Why this run exists

`bench/runs/2026-10-05-adapter-port-4gpu/RESULT.md`:
- **Overhead.** The ported adapter costs 27.8 % per step. The synchronous mirror commit takes
  ~17 ms per decode step (p50), about one whole step: its device-to-host copy waits for the step's
  forward and removes async scheduling's overlap.
- **Pause.** After the abort-order fix, the REPLACE pause is 1.55–1.70 s (post-hoc). Its largest
  part is the spare's restore, 1.1 s.
- **Notice.** With sessions long enough to span the deadline, the notice pause is 0.36–0.41 s
  with no fault observed (post-hoc).

This run takes the commit off the step path and re-measures all three against stated numbers,
pre-registered this time.

## What changes

- **The asynchronous commit.**
  - At the end of a step the connector enqueues the step's row gather on a side stream into
    pinned host memory and records an event. No host synchronisation happens on the step path.
  - A sender thread waits for the event and sends the step's `Commit` frames to the store, the
    per-request counts on the last frame as before, so a count is never ahead of its rows.
  - At most two steps are in flight; a third step waits for the oldest (back-pressure, counted).
  - A step interrupted by a membership change is dropped before it is enqueued, as today.
- **What a restore can trail.** The dead rank's in-flight commits die with it, so its own
  committed counts can trail its last completed step by up to the pipeline depth (two steps). The
  scheduler replays that tail from the counts, as it does now for the partial block.
- **Nothing else changes.** Same engine pin, model, adapter, daemon, store and harness as the
  adapter-port run, with the abort-order fix.

## What runs

- Pod: RunPod 5× A40 48 GB (GPUs 0–3 serve, GPU 4 the spare), community cloud if available, else
  secure; `NCCL_P2P_DISABLE=1`, `disable_custom_all_reduce=True`.
- Model: `Qwen/Qwen3-30B-A3B` bf16 at `/workspace/models/Qwen3-30B-A3B`, TP4 + EP, 8 sessions ×
  2,048-token prompts, greedy, seed 0.
- **Decode length.**
  - 200 steps for the stock, overhead and REPLACE arms, comparable with the adapter-port run.
  - 600 steps (`--max-model-len 2688`, 1,400 KV blocks) for the notice arm. At about 35 ms per
    step, the 10 s deadline from a notice at step 100 falls near step 385, well inside the session.
- Arms, three runs each, in this order:
  1. stock;
  2. adapter with the mirror off: the slot kernel, shadow sets and spare resident, no connector
     (attributes the overhead);
  3. adapter, no fault (the overhead arm);
  4. REPLACE, SIGKILL of rank 2 between steps after step 100;
  5. REPLACE, SIGKILL of rank 2 half a median step into step 101;
  6. notice for rank 2 after step 100 with a 10 s deadline; the driver kills the leaving rank
     once the daemon's replacement is completed.
- The drill driver, the restore check and the summariser are `bench/harness/adapter_drill.py`,
  `restore_check.py` and `summarize_adapter_port.py` at the run's commit; the definitions in the
  summariser's docstring hold.

## Predictions (kill conditions in bold)

1. **REPLACE pause**, as the largest inter-token gap of any session: 1.4–2.0 s in every REPLACE
   run, both kill timings. **Kill: any run > 2.5 s.**
2. **Sessions continue with their KV in place.** Every session reaches its last token. The 20
   tokens after the fault equal the adapter no-fault run's on the same seed. Every session's
   replay starts past its prompt. **Kill: any session lost, re-prefilled, or with a token
   mismatch in those 20.**
3. **Steady-state overhead** of the adapter (arm 3) against stock (arm 1), wall-clock tokens per
   second over steps 10–200: under 2.0 %. **Kill: > 4.0 %.**
4. **Notice pause**, as the largest inter-token gap: under 0.5 s in every notice run, with no
   `rank_lost` event in the executor and no fault in any runtime log. **Kill: any run > 1.0 s, or
   any fault event.**
5. **The commit is off the step path.** Mirror commit time measured on the step thread is under
   1 ms p50 on every rank, and back-pressure waits occur in under 1 % of steps. **Kill: p50
   > 2 ms, or back-pressure in > 5 % of steps.**
6. **Restore exactness.** The spare's restored rows equal the dead rank's committed rows (digests
   per layer and head slot, taken at the last step before the kill) in every REPLACE run.
   **Kill: any mismatched table.**
7. **Commit latency.** The daemon's `MembershipCommitted` is heard within 100 ms of the engine
   resuming, and every survivor and the spare applied its epoch. **Kill: no commit, or an epoch
   mismatch.**

### Evidence against prediction 3, stated before the run

- Week five measured the host-memory slot collective alone at 4.5–4.6 % on this hardware
  (`infertap/invent`, `week5-tests/A/PART2-results.md`, arm 1). It is unchanged here: its
  replacement is the `core/csrc` kernel of decision 0006.
- If that cost holds, prediction 3 dies on the kernel, not on the commit, whatever the commit
  does.
- Arm 2 measures the kernel's share directly, and `RESULT.md` will report both shares. The
  prediction stands as written.

## What is recorded

- `results.json`: every figure above per run, labelled by arm.
- `RESULT.md`:
  - the table and the verdict per prediction (alive, dead, weakened);
  - the pause breakdown (detect, change, restore, resume, first token);
  - the overhead split into kernel and commit shares;
  - the commit pipeline's depth and back-pressure counts;
  - the reconciliation setting under REPLACE;
  - surprises and dollars.
- `run.log`: every command, price and timestamp; the price and running total at each arm.
- Raw outputs in `data/` (not in git, SHA-256 in `results.json`).

Runs added after the fact are labelled post-hoc and reported apart from these verdicts.

## What this run does not claim

- Anything about the P2P slot kernel, NVLink hosts, frontier shapes or multi-node.
- A faster restore. The 1.1 s restore through the store is unchanged here, which is why
  prediction 1's band starts at 1.4 s.
