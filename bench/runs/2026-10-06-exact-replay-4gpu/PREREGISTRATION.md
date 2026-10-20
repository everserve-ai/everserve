# Pre-registration: the four-GPU battery with the exact replay (decision 0013)

Status: PLANNED 2026-10-06, arms and predictions set by the owner. Spend: about 22 minutes (20
runs) on the session's 5× A100 SXM pod at $7.95/h, about $2.90. The owner capped the whole session
(build, drills and this run) at $8; the sequencer checks the spend before every arm and stops
within $0.50 of that.

**Tree.** `eb67ad3`, or a later commit whose diff from it touches only `bench/`, named in
`run.log` with `scripts/preflight.sh` and CI green. On top of the last battery's tree
(`bench/runs/2026-10-05-kernels-order-4gpu/`, `72729ed`) it has:
- the exact replay (decision 0013, `d740223`): after a change every running request stays
  running with its blocks, its computed length is set to its committed count exactly, every rank
  re-adds it from the same state, and the uncommitted tail replays one position per step on the
  captured graphs. Nothing is preempted, evicted or found through the prefix cache;
- the drills' three exactness checks (`eb67ad3`), and top-20 logprobs recorded in every arm.

## Why this run exists

The last battery passed nine predictions and failed one: session 1 diverged after every REPLACE.
The diagnostic session (`bench/runs/2026-10-06-replace-divergence/`) traced it to the replay:
- it restarted at the block boundary, not the committed position;
- it recomputed six positions as an eager prefill chunk;
- that overwrote committed K and V rows on every rank with eager arithmetic.

Decision 0013 replaces that replay. This battery measures the replacement, with prediction 1 back
to eight of eight.

## The pod

As the last battery: 5× A100-SXM4-80GB on one NVLink host (GPUs 0–3 serve, GPU 4 the spare),
peer-to-peer on, the peer-to-peer slot variant reported by every worker, `/dev/shm/everserve` as
the data directory. The host and driver are named in `run.log`.

## What runs

Same engine, model (Qwen3-30B-A3B, TP4 + EP), prompts (8 sessions × 2,048 tokens), decode lengths,
harness and arms as the last battery, in this order, on one pod in one session:
1. stock, custom all-reduce on, ×3;
2. adapter, mirror off, ×3;
3. adapter, ×3;
4. REPLACE between steps, ×3;
5. REPLACE mid-step, ×3;
6. the notice reference (no fault, 601 tokens), ×1;
7. notice (`--notice-steps 300`, 600 decode steps), ×3.

Every arm runs with the fixed-order all-reduce settings and records its top-20 logprobs (both new
since 2026-10-05; stock pays the same for the logprobs as the adapter).

**Recording**, as the last battery, plus per fault run `exactness` (decision 0013):
- `kv_below_committed`: every rank's K and V rows at the committed positions, before the change
  and after the first step back, byte-identical;
- `first_step_back`: the first steps after resuming schedule each request one token at a time
  from exactly its committed count, with no request re-added, and every rank runs the first on the
  FULL graph;
- `logits`: the run's logprobs equal its no-fault reference's for every token.

## Predictions (kill conditions in bold)

1. **Sessions token-identical after REPLACE.** In every REPLACE run, all eight sessions' tokens
   after the fault equal `adapter_1`'s, every session reaches its last token, and every replay
   starts past the prompt. **Kill: any session mismatched, lost or re-prefilled.** If the three
   no-fault adapter runs disagree with each other, the prediction is not tested and the report
   says so.
2. **Sessions token-identical after notice.** In every notice run, all eight sessions' tokens after
   the swap equal `notice_ref`'s, with the same conditions. **Kill: any mismatch, or the check not
   run.**
3. **Adapter overhead** against stock with custom all-reduce on: under 3 %, wall-clock tokens per
   second over steps 10–200, each arm's mean. **Kill: over 5 %.**
4. **Mirror-off overhead** against stock: under 1 %. **Kill: over 3 %.**
5. **REPLACE pause**, the largest inter-token gap of any session: under 1.5 s in every REPLACE
   run. **Kill: any run over 2.5 s.** Each handle's `ncclCommAbort` time has its own column.
6. **Notice pause**, the largest inter-token gap, with the swap in every notice run: under 0.3 s.
   **Kill: no swap, any run over 1.0 s, or a fault event.**
7. **The store span.** `sender` p50 under 1 ms on every rank in every run with the mirror on.
   **Kill: any rank's p50 over 3 ms.**
8. **The spare's restore** under 0.4 s in every REPLACE and notice run. **Kill: over 1.0 s.**
9. **Restore exactness.** The restored rows equal the dead rank's committed rows in every REPLACE
   and notice run. **Kill: any mismatched table.**
10. **Commit before resume.** `MembershipCommitted` within 100 ms of the engine resuming, every
    epoch applied. **Kill: no commit, or an epoch mismatch.**

The three exactness checks are recorded with each fault run and reported beside predictions 1 and
2; they carry no separate verdict.

## What this run does not claim

Multi-node, frontier shapes, Hopper, or the host-memory variant's cost.
