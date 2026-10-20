# Pre-registration: the kernels re-run, rank-order sum and device-resident order

Status: PLANNED 2026-10-05, arms and predictions set by the owner. Spend: about 25 minutes (19 runs) on
the session's 5× A100 SXM pod ($7.95/h), about $3.30; hard cap $10, counted from the first arm:
the sequencer checks the spend before every arm and stops within $0.75 of the cap. The build session that
precedes it on the same pod is accounted separately in `bench/runs/PLANNED/kernels-order-session/`.

**Tree.** `5ff7744`, or a later commit whose diff from it touches only `bench/`, named in
`run.log` with `scripts/preflight.sh` and CI green. That tree has, on top of the last run's
(`bench/runs/2026-10-05-kernels-4gpu/`):
- two fixes the H100 attempt found in the spare: its worker lock is held for its life (`419f9d6`),
  and on Hopper it loads the serving ranks' FlashInfer tactics instead of tuning alone (`fff9455`);
- the slot collective's sum in tensor-parallel rank order, the order written by the executor with
  the spare in the replaced rank's position (decision 0012 item 1, `3a8ac5e`);
- each member's order in its own device memory, copied at the boundary, the abort word alone in
  host memory (decision 0012 item 2, `b8000ee`);
- a notice-length no-fault reference in the harness (`e831fb5`);
- the first-steps probe: what each worker's first two steps after a change did (`8b51e6f`).

## Why this run exists

The last run (`2026-10-05-kernels-4gpu`) failed three predictions. One session in eight produced
different tokens from 13 tokens after every REPLACE, and the adapter ran 7.8 % behind stock with
vLLM's custom all-reduce on. Decision 0012 names the causes: the sum's order changed with the
spare's row, and every call read its membership from host memory. This run measures the fix.

## The pod

- **Hardware.** 5× A100-SXM4-80GB on one NVLink host (GPUs 0–3 serve, GPU 4 the spare), as the
  last run. The host and driver are named in `run.log`. The build session first ran on 5× H100,
  where stock vLLM gave different tokens in two identical runs (see below); it moved to A100.
- **Peer-to-peer on.** No `NCCL_P2P_DISABLE`; the topology goes in `data/topology.txt`.
- **Slot variant.** Peer-to-peer, as `choose_variant` picks on such a host and as each worker's
  report names it (`paths.slot`, `results.json`). If any worker reports host memory, the run stops
  before its first arm and the report says so.
- **Data directory.** `/dev/shm/everserve`, with `df -h /dev/shm` in `run.log`.

**Per-call times on this pod (`bench/harness/slot_bench.py`, 4 ranks, one CUDA graph of 2,000
calls, µs per call).** Measured in the build session, before the run:

| Elements | slot, peer-to-peer, 1 word/thread (the default) | 4 words | 16 words | host memory | NCCL (pynccl) | stock custom all-reduce |
|---|---|---|---|---|---|---|
| 2,048 | **3.99** | 5.52 | 8.56 | 9.22 | 13.06 | 5.32 |
| 4,096 | **3.50** | 4.66 | 7.64 | 9.01 | 14.30 | 5.51 |
| 8,192 | **3.62** | 4.76 | 9.86 | 12.01 | 16.03 | 5.53 |
| 16,384 | **4.21** | 5.20 | 10.04 | 18.73 | 16.78 | 5.81 |
| 32,768 | **5.36** | 6.42 | 10.10 | 31.10 | 17.08 | 6.54 |

The decode step of this model at 8 sessions is 16,384 elements. With the order in device memory the
slot collective is under the custom all-reduce at every decode size; with it in host memory (the
last run's kernel, 4 and 16 words) it was 10 µs at 16,384.

## What runs

Same engine, model (Qwen3-30B-A3B, TP4 + EP), prompts (8 sessions × 2,048 tokens), decode lengths
and harness as `2026-10-05-kernels-4gpu`. The arms, in this order, all on the one pod in one
session:
1. stock, with vLLM's default custom all-reduce on (`adapter_port.sh stock_default`), ×3. A stock
   run that fails with it on is repeated with it off; `results.json` records which ran
   (`custom_all_reduce`), and the overhead verdicts are against whatever stock ran;
2. adapter with the mirror off, ×3;
3. adapter, ×3;
4. REPLACE between steps, ×3;
5. REPLACE mid-step, ×3;
6. the notice reference: adapter, no fault, 601 tokens at `--max-model-len 2688`, ×1
   (`adapter_port.sh notice_ref`);
7. notice with `--notice-steps 300` and 600 decode steps, ×3.

**Fixed-order all-reduces in every arm, stock included** (`adapter_port.sh`):
`VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 NCCL_NVLS_ENABLE=0`. Token equality
is checked against another run, so the engine must give the same tokens twice. On 5× H100 two
identical stock runs differed; vLLM's own source says its FlashInfer all-reduce has no fixed
reduction order, and these are the backends its batch-invariant mode turns off. On A100 the last
run's no-fault runs were identical, and these settings are expected to change nothing there; stock
still runs its custom all-reduce.

**Recording**, as the last run, plus:
- per run, `reference` (the no-fault run its tokens are compared with) and `after_fault_match`;
- per fault run and rank, `first_steps`: the first two steps after the change, their CPU and GPU
  times, compile-counter and allocator changes, and sampled stacks by class (compile, graph
  capture, device wait, queue wait, other). An instrument: no prediction rests on it;
- `determinism`: per arm and length, each no-fault run's sessions against the arm's first run,
  the first token where they differ (the control for predictions 1 and 2);
- every rank's event log in `flake/`.

**Every overhead is against stock on the same pod in the same session.**

### Evidence stated before the run (development runs on this pod, not verdicts)

- **Determinism.** Two no-fault adapter runs gave identical tokens in all 8 sessions.
- **REPLACE, session 1.** Three development REPLACE runs (landing 1 between and mid, landings 1–4
  between) applied the order `[0, 1, 4, 3]` on every member and still gave session 1 different
  tokens from token 114, 13 after the kill; the other seven matched. This is the last run's
  failure, unchanged by the rank-order sum, so prediction 1 is expected to die. Its cause is open:
  it is in the fault path (the aborted step or the replay of the tail), not the spare's arithmetic.
- **Notice.** One development notice run matched `notice_ref` in all 8 sessions, with the spare in
  rank 2's position: the swap itself keeps the tokens.
- **Speed.** One development adapter run on landings 1–4 ran 993.3 tok/s; landing 1 alone ran
  839.2. Stock was not run in development.
- **The first step after a change.** In one REPLACE run the survivors' first step took about 90 ms
  of GPU against 16 ms for the next, with no compile-counter change and no new allocation.

## Predictions (kill conditions in bold)

1. **Sessions token-identical after REPLACE.** In every REPLACE run, all eight sessions' 20 tokens
   after the fault equal `adapter_1`'s, every session reaches its last token, and every replay
   starts past the prompt. **Kill: any session mismatched, lost or re-prefilled.** The verdict
   needs the control: if the three no-fault adapter runs do not agree with each other, the
   prediction is not tested, and the report says so rather than scoring it.
2. **Sessions token-identical after notice.** In every notice run, all eight sessions' 20 tokens
   after the swap equal `notice_ref`'s, with the same conditions. **Kill: any mismatch, or the
   check not run.** Its control is `notice_ref` itself against `adapter_1` over their common
   201 tokens, which the report gives.
3. **Adapter overhead** against stock with custom all-reduce on: under 3 %, wall-clock tokens per
   second over steps 10–200, each arm's mean. **Kill: over 5 %.**
4. **Mirror-off overhead** against stock: under 1 %. **Kill: over 3 %.**
5. **REPLACE pause**, the largest inter-token gap of any session: under 1.5 s in every REPLACE
   run, both kill timings. **Kill: any run over 2.5 s.** Each handle's `ncclCommAbort` time is
   recorded per run and given its own column.
6. **Notice pause**, the largest inter-token gap, with the swap happening in every notice run:
   under 0.3 s. **Kill: no swap in any run, any run over 1.0 s, or a fault event.**
7. **The store span.** The commit timeline's `sender` span has a p50 under 1 ms on every rank in
   every run with the mirror on. **Kill: any rank's p50 over 3 ms.** `fence_wait` and `event_lag`
   are reported beside it without a verdict.
8. **The spare's restore** takes under 0.4 s in every REPLACE and notice run. **Kill: over 1.0 s.**
9. **Restore exactness.** The restored rows equal the dead rank's committed rows in every REPLACE
   and notice run. **Kill: any mismatched table.**
10. **Commit before resume.** `MembershipCommitted` is heard within 100 ms of the engine resuming,
    with every survivor's and the spare's epoch applied. **Kill: no commit, or an epoch mismatch.**

Predictions 5 to 10 are the last run's 3 to 9, carried as they were, on the same GPU.

## What this run does not claim

Multi-node, frontier shapes, Hopper, or the host-memory variant's cost. The first-steps probe explains the first fault run's slow first step only if it shows
something; it is not a prediction.
