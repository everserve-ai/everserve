# Pre-registration: detection at the collective (decision 0021) on the 72B shape

Status: PLANNED 2026-10-06, written before the run. One 5× H100 SXM pod, secure cloud, Qwen2.5 72B
Instruct at TP4, cap $10. The watchdog is set at the estimate plus 12 minutes: 16 + 12 = 28 min,
about $8.10 at $17.45/h.

**Tree.** The landing of decision 0021 on main, named in `run.log` with `scripts/preflight.sh` and
CI green. Before it, the same tree ran a compile and function check on 3× A6000: Qwen3-1.7B at
TP2, the kernel test, one adapter run and one REPLACE (`run.log`, `data/cheap/`).

## Why this run exists

The commit-path battery (`bench/runs/2026-10-06-commit-path-72b/`) measured a REPLACE pause of
6.8 s, of which 6.0 s was the executor waiting on the dead process's sentinel. Decision 0021
detects the death in the slot collective, and checks the late member's process in `/proc`. This
run measures the pause with that detection on the same shape.

## What runs

Qwen2.5 72B Instruct, TP4, 8 sessions × 2,048-token prompts, 200 decode steps, the determinism
settings, top-20 logprobs. In this order, nothing repeated:
1. stock ×1, custom all-reduce on (the base for the overhead);
2. adapter ×1 (the no-fault reference, and the check for false reports);
3. REPLACE mid-step ×1 (rank 2 killed during the step after step 100).

**Recorded,** as the battery did, plus:
- `detected_by` and `detect_s`;
- the dead process's own exit (`exit_s`), from the kill to the process being a zombie or gone;
- the sentinel's sighting (`sentinel_s`) if it comes before the change completes;
- the detection bound and the step period it was measured from (`started.detect`);
- every slot-watch report, and every late member found alive (`slot_watch`);
- the first 16 steps' commit stamps per rank (`first_ms_from_commit`), for the prefill stall.

## Predictions (kill conditions in bold)

1. **REPLACE pause** about 1 s: 0.6–1.2 s. This battery's post-detection part was 0.8 s, and
   detection should add about 0.1 s. **Kill: over 2 s.**
2. **Detection** by the slot collective (`detected_by` = `slot`), under 0.15 s from the kill: the
   bound of 3 step periods (about 50 ms), plus the watchdog's 2 ms poll, plus the executor's 20 ms
   wait slice. **Kill: by the sentinel, or over 0.5 s.**
3. **No false report:** the adapter run has no `rank_lost` event and no slot-watch report. A late
   member found alive (the prefill's stall) is recorded and not reported. **Kill: any report or
   loss in the adapter run.**
4. **Overhead** of the adapter run against stock: under 3 %, as in the battery (the kernel now
   reads the global timer once per call). **Kill: over 5 %.**
5. **Sessions continue:** 8/8 kept, identical to the adapter run after the fault, resumed from the
   committed count, no re-prefill; restored rows equal in every table; epochs matched. **Kill: any
   session lost, re-prefilled or mismatched, any restore mismatch, or a commit after resume.**

**Reported, not scored:**
- `exit_s` against `detect_s`. The battery's host took 6.0 s to tear the process down; another host
  may be faster, and that no longer enters the pause.
- Whether the coordinator folded the sentinel's report.
- The prefill stall's span from `first_ms_from_commit`: in the gather (`event_lag`) or in the store
  round trip (`sender`), per rank.

## What this run does not claim

A hung worker (the hang bound path), two deaths at once, or the host-memory variant at this shape.
The cheap check on 3× A6000 runs the host-memory variant at TP2.
