# Result: the adapter port on the four-GPU rig

Run 2026-10-04T23:46Z to 2026-10-05T01:10Z on RunPod pod `t8jgr7fvniodb6`: 5× A40 48 GB, secure
cloud, SE, $2.45/h (community had no 5× A40). GPUs 0–3 serve, GPU 4 holds the spare. vLLM at the pin
(`6e517b15`, wheel `0.30.1rc1.dev606`), torch 2.13.0+cu130, NCCL 2.29.7, driver 580.159.04 with
`cuda-compat-13-0`. `NCCL_P2P_DISABLE=1`, `disable_custom_all_reduce=True`. Qwen3-30B-A3B bf16,
TP4 + EP, 8 sessions × 2,048-token prompts, 200 decode steps, SIGKILL of rank 2 after step 100.
**No file under `vllm/` was edited.** Predictions are unchanged since `6844048`. Every figure per
run is in `results.json`; commands, prices and timestamps are in `run.log`; raw outputs are in
`data/` (not in git, SHA-256 in `results.json`).

**Two groups of runs.**
- **Pre-registered:** 15 runs, exactly as planned.
- **Post-hoc:** 9 more runs, made after a fix to the code (the order in which NCCL handles are
  aborted) and with a longer notice run. They are labelled `posthoc_*` and reported separately.
  They do not replace the pre-registered verdicts.

## Verdict per prediction

| # | Prediction (kill) | Pre-registered | Verdict | Post-hoc |
|---|---|---|---|---|
| 1 | REPLACE pause 2.0–3.5 s (> 5 s) | 10.73–11.07 s | **dead** | 1.55–1.70 s |
| 2 | sessions keep KV; 20 tokens after the fault equal the no-fault run (any re-prefill or loss) | 8/8 sessions, 8/8 equal, none re-prefilled, 6/6 runs | **alive** | same, 6/6 |
| 3 | adapter overhead 4.0–6.0 % (> 8 %) | 27.8–27.9 % | **dead** | n/a |
| 4 | `MembershipCommitted` within 100 ms of resuming, acked epoch (none, or mismatch) | 25–36 ms; epoch 1 on every survivor and the spare | **alive** | 19–31 ms (notice 54–63 ms) |
| 5 | restore bytes equal the dead rank's committed rows (any mismatch) | 48/48 tables equal, 6/6 runs | **alive** | 9/9 runs, including notice |
| 6 | notice: no fault observed, pause < 1 s (a fault event) | no swap happened | **not tested** | 0 fault events, 0.36–0.41 s |

**Prediction 6 was not tested by the pre-registered protocol.**
- At 200 tokens, the sessions finished about 3.5 s after the notice at step 100. The 10 s
  deadline never arrived.
- The post-hoc notice runs generate 600 tokens (max model length 2,688, 1,400 blocks) so the
  deadline falls inside the run.

## Pause breakdown (REPLACE, seconds)

Timed from the driver's kill:
- **detect:** the executor sees the death.
- **change:** the daemon's commit is heard, minus the spare's restore.
- **restore:** the spare reads the dead rank's rows from the mirror.
- **resume:** the engine-core half (ring, spare in place, replay).
- **first token:** every session's first token after resume.

| Runs | Pause | Detect | Change | Restore | Resume | First token |
|---|---|---|---|---|---|---|
| pre-registered, between-step ×3 | 10.85–11.00 | 0.23–0.26 | **9.20–9.44** | 1.03–1.24 | 0.025–0.036 | 0.19–0.25 |
| pre-registered, mid-step ×3 | 10.73–11.07 | 0.25–0.28 | **9.15–9.44** | 1.04–1.10 | 0.025–0.032 | 0.23 |
| post-hoc, between-step ×3 | 1.56–1.70 | 0.23–0.30 | 0.003 | 1.06–1.16 | 0.026–0.031 | 0.20–0.24 |
| post-hoc, mid-step ×3 | 1.55–1.69 | 0.23–0.28 | 0.002–0.003 | 1.07–1.21 | 0.019–0.031 | 0.19–0.22 |
| post-hoc, notice ×3 | 0.36–0.41 | n/a | n/a | 1.50–1.61 (before the deadline, off the pause) | 0.054–0.063 | 0.30–0.32 |

**Why the change segment took 9.2 s in the pre-registered runs.**
- `on_fault` took 10.43 s on every survivor, against 0.0004 s on the spare.
- The abort retired the other groups' NCCL handles before the live TP handle.
- The survivor's main thread was still blocked in the TP all-gather with the dead rank, so each
  of those aborts waited for it until the 10 s bound.
- Fix (`e0cf7b4`): release the live handle first, then retire the rest. The change segment fell
  to 3 ms.

**What remains of the 1.6 s:**
- the spare's restore: 1.1 s for about 420 MB of rows through Python protobuf over the Unix
  socket;
- detection: 0.25 s;
- the first token: 0.2 s, mostly recomputing each session's last partial block.

**The replay point is exact.** The spare restores to the dead rank's own committed counts (2,149
tokens per session: the prompt plus 101 tokens). Each session recomputes only the 5 tokens of its
last partial block, so no session was re-prefilled.

## Mirror and daemon

- **Mirror commit per decode step:** p50 16.7–17.9 ms on every rank in every run (p99 26–33 ms in
  the no-fault arm). The mean of 42 ms includes the one prefill commit of about 5 s (400 MB, split
  into 8 MB frames).
- **Overhead:** the commit is about one whole decode step, which is where prediction 3's 28 % comes
  from.
  - Stock runs at 463–467 tok/s; the adapter at 335 tok/s.
  - Week five's slot kernel alone cost about 4.6 % on this kind of host. The rest is the
    synchronous commit: its device-to-host copy waits for the step's forward, which removes async
    scheduling's overlap.
  - Making the commit asynchronous (gather on a side stream into pinned memory, send from a
    thread; counts only after their rows) is the obvious next step. It changes `docs/CONTRACT.md`
    item 3 ("commit on the step path"), so it is a design decision, not a fix made here.
- **Daemon:**
  - Loss report to `MembershipCommitted`: about 1.1 s, which includes the spare's restore before
    its acknowledgement.
  - Commit to engine resumed: 19–36 ms (REPLACE), 54–63 ms (notice).
  - Every survivor and the spare acknowledged epoch 1; the daemon now rejects a mismatched
    acknowledgement.

## Private-internals delta

`adapters/vllm/PRIVATE-INTERNALS.md`, rows A1–A30. The port touches about 450 lines in the listed
ranges, against the harness's 70 + 110 + 230 for the same mechanisms.

**Removed:** relaxed mode, standby, eager recapture, the torch re-rendezvous, the `execute_model`
wrapper.

**Added on the GPU:**
- A15: the refusing stand-in and the abort order.
- A17: abort under the communicator's own device.
- A19: walk the frames to the `WorkerProc`.

**Upstream hooks:**
- **New:** an engine-core step-loop hook to finish a committed membership change. Today the bench
  driver calls `complete_change`.
- **Still needed:** a graph-safe TP all-reduce registration, and a public "replace this group's
  device communicator".
- **No longer needed for slot ids:** the model runner already hands `attn_metadata` to
  `start_load_kv` under FULL graphs.

## Defects found on the GPUs and fixed during the run

Each fix is its own commit on top of `836416b`. Hashes are those on main after the trailers were dropped and the branch was rebased onto the telemetry commits; `results.json` maps them to the trees the runs executed on.

1. `4230925`, controller: a SIGKILLed worker's socket closes before the loss is reported, and the
   daemon had already unregistered the rank. It now keeps a dropped worker as departed.
2. `e453301`: the boundary RPC took the caller one frame up as the `WorkerProc`. It now walks the
   frames.
3. `3096fae`, bench: vLLM renames its workers, so killing them by name left GPUs held. They are now
   killed by process id.
4. `c5f0ef5`: the executor's monitor stopped after the first death. It now sees every death.
5. `cdd354b`: the NCCL abort ran on a fresh thread, which starts on device 0. It now runs on the
   communicator's device. This was necessary but did not fix the crash.
6. `161d60d`: an aborted handle marked `disabled` sent vLLM's TP all-gather to its torch group,
   which still held the dead rank. The survivors' CUDA context then faulted with an illegal memory
   access. Handles are now swapped for a refusing stand-in first.
7. `e0cf7b4`: the abort order, as described above.

## Surprises

- **The mirror commit, not the slot kernel, dominates the steady cost.** The pre-registration
  attributed the expected 4–6 % to the slot kernel.
- **The abort order decides the pause.** It made the difference between 1.6 s and 11 s.
- **vLLM's silent fallback.** A disabled pynccl handle makes vLLM fall back to the torch group
  rather than fail.
- **The notice arm was mis-sized.** At this step rate, 200 tokens cannot contain a 10 s deadline.
  This was a mistake in the protocol I set, not a finding about the engine.
- **The slow-test flake** (`core/tests/test_runtime_fault.py::test_mid_step_kill_is_bit_identical[0.003-tp]`)
  was caught with its event logs in `flake/`.

## Dollars

| What | Cost |
|---|---|
| Pod `t8jgr7fvniodb6`, 5× A40, 1 h 24 min at $2.45/h | $3.43 |

That covers setup, five dev runs, the two hung attempts, and the 24 runs. The balance went from
$52.03 to $48.62. The pod was terminated after the weights were deleted, the laptop watchdog was
stopped, and no pods remain. This is under the $5 target and the $10 cap.
