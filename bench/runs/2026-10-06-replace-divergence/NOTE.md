# The REPLACE divergence: where it is born, 2026-10-05

A diagnostic session, no product code changed. Four runs on one 5× A100 pod, tree `5f5b116`
(drill instrumentation in `bench/harness/divergence_probe.py`):
- `diag_ref_1` and `diag_ref_2`: no-fault adapter runs, the control;
- `diag_replace`: one REPLACE run, rank 2 killed between steps after token 101;
- `eager_1`: one no-fault adapter run with CUDA graphs disabled.

The three diagnostic runs and `eager_1` recorded every session's top-20 logprobs per token.
- The diagnostic runs also recorded, per rank and session, a digest of the K and V rows at every
  position from 2,040.
- They also recorded every step's batch, from the scheduler and from each worker's model runner.

**The control holds.** `diag_ref_1` and `diag_ref_2` agree in every session's tokens, every logprob,
and every K and V row on every rank.

## The answer in one paragraph

After the fault, the engine replays each session from **2,144**, the start of the KV block that
holds the dead rank's committed count (2,149), not from 2,149 itself.
- Each request comes back as a new request with 2,144 tokens computed. Six positions per request
  (2,144–2,149) run in one 48-token batch, flagged as prefill.
- That batch is too large for a captured graph, so it runs **eager**.
- Eager arithmetic differs from the graphs' (question 3). Positions 2,144–2,148 had been computed
  correctly in decode graphs and committed. They are overwritten **on every rank, the survivors
  included**, with slightly different values.
- From that step on, every session's logits differ from the no-fault run's. Session 1 is the one
  where the difference first changes the argmax, at token 114.

The divergence is born in the first step back. It is in the attention inputs: the recomputed K and
V rows. The logits follow from them.

## 1. The slow first step after a notice swap

**Yes, it occurs after a notice swap too.** In the re-run's three notice runs (probe on,
`bench/runs/2026-10-05-kernels-order-4gpu`), the first step back took 97–126 ms of GPU on every rank
against 11–29 ms for the next. REPLACE showed 89–95 against 15–21. No notice run was repeated here;
those figures are from a pre-registered run on the same tree's probe.

So the slowdown is not unique to the fault path. What both paths share is the replay after the
change. In this REPLACE run the first step back is the 48-token eager prefill described above,
which accounts for a long first step. The notice runs also replayed every session (`n_replayed` 8,
from 2,448). Their batch composition was not recorded, so that the notice's first step is the same
kind of step is inferred, not measured.

## 2. Logits and K/V rows at the first step back

| Quantity | `diag_replace` against `diag_ref_1` |
|---|---|
| tokens | session 1 from token 114; sessions 0, 2–7 identical to the end |
| top-20 logprobs | **all 8 sessions differ from token 102**, the first token produced after resuming (101 were produced before the kill); identical before it |
| K and V rows | **all 8 sessions, all 4 ranks, differ from position 2,144** to the last position recorded (2,191); identical before 2,144 |

- **Where it is born.** The first differing logits are produced by the first step back, and the
  first differing K/V rows are among the positions that step writes. So the divergence is born in
  that step.
- **Which quantity.** The K and V rows at 2,144–2,148 were written by decode steps before the kill.
  In the no-fault run they are never rewritten; here the first step back rewrites them with
  different values. The rows feeding attention changed, and the logits differ because of that.
- **Not the spare.** Ranks 0, 1 and 3 never left; their rows at 2,144–2,148 differ all the same.
- **Not the restore.** The restore check still matches 48/48 tables, and rows below 2,144 are
  identical on every rank.

## 3. Eager against graphs, no fault

| Quantity | `eager_1` against `diag_ref_1` |
|---|---|
| tokens | sessions 1, 2 and 3 differ, from tokens 77, 11 and 14; sessions 0, 4–7 identical |
| top-20 logprobs | all 8 sessions differ from token 0 |

Eager and graph execution compute different values from the very first token. This is the
mechanism: the first step back runs eager because its batch does not fit a captured graph.

## 4. Batch composition: the aborted step and the first step back

From the scheduler (engine) and rank 0's model runner. Ranks 1 and 3 recorded the same steps. The spare (rank 2) recorded none: the probe was installed only in the workers that existed at startup.

| Step | Requests (order) | Query length each | Computed before | Prefill? | CUDA-graph mode | Padded tokens |
|---|---|---|---|---|---|---|
| before the kill, every step | 0–7 | 1 | 2,135 … 2,148 | no | FULL (decode) | 8 |
| scheduled as the kill landed (computed 2,149, 2,150) | 0–7 | 1 | 2,149 / 2,150 | no | FULL | 8 |
| **first step back** | 0–7, re-added as new requests | **6** | **2,144** (prefill length 2,150) | **yes** | **NONE (eager)** | 48 |
| the step after | 0–7 | 1 | 2,150 | no | FULL | 8 |

**What differs:**
- The request order is the same in every step.
- What changes is the query length (6 against 1), the starting position (2,144, the block boundary,
  against the dead rank's 2,149), the flag (prefill against decode), and the path (eager against
  the FULL decode graph).
- In the no-fault run every one of positions 2,144–2,150 was computed by a FULL-graph decode step
  of query length 1.

## What this does not settle

- **Why the replay starts at the block boundary.** It looks like the scheduler re-adding the
  requests through the prefix cache, which only reuses whole blocks. That is the next question, not
  answered here.
- **Notice identity.** The notice arm matched 8/8 in tokens, but its logprobs were never compared.
  Its replay also runs a non-graph step. Its token match over 20 tokens does not show its logits
  are identical.
- **Other steps.** Whether any later step differs in path was not examined; only the steps around
  the change were.

## Pod and spend

| What | Cost |
|---|---|
| Pod `7apjm912t5n0rt`: 5× A100-SXM4-80GB, NVLink NV12, secure cloud, US, driver 580.126.16 (with `cuda-compat-13-0`); 22:57:34Z–23:16:55Z (19.4 min) at $7.95/h | **$2.56** (cap $10) |

The weights and `/dev/shm` were deleted, the pod was terminated and its watchdog stopped; no pods
remain. Commands and timestamps are in `run.log`, every figure above in `results.json` (with SHA-256
of `data/`), and raw outputs in `data/` (not in git).
