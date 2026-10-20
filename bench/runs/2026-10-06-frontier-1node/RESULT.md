# Result: the single-node frontier run, Qwen2.5 72B on 5× H100

**Pod and software.**
- RunPod pod `j7cipj43ecar2r`: 5× NVIDIA H100 80GB HBM3 SXM on one host, NVLink NV18 between every
  pair, secure cloud, US, $17.45/h; 2026-10-06T01:54:31Z to 02:48:37Z.
- Driver 580.126.09 with `cuda-compat-13-0`, NCCL 2.29.7, torch 2.13.0+cu130, vLLM at the pin
  (`0.30.1rc1.dev606+g6e517b15c`). `/dev/shm` 586 GB. **No file under `vllm/` was edited**, and no
  product code changed during the run.
- **Model: Qwen2.5 72B Instruct**, bf16, TP4, dense. Llama 3.3 70B Instruct was refused (gated
  repository, no token). Both the model and the refusal were written into the pre-registration and
  committed (`37e4147`) before the first measured arm.
- **Determinism checked first:** two stock runs with the harness's settings gave identical tokens in
  all 8 sessions.

**Tree.** `37e4147` for the product code throughout. Harness-only commits during the session:
- the drill enables expert parallelism only for a checkpoint with experts;
- the store's memory budget became a setting.

Only `bench/harness` was copied to the pod after the pre-registration.

**Records.**
- `results.json`: every figure per run, the exactness checks, device memory at ready, SHA-256 of
  `data/`.
- `run.log`: commands, prices and timestamps.
- `data/`: raw outputs, the `/metrics` text, the determinism pair, setup and model logs (not in
  git).
- `flake/`: every rank's event log for the fault runs.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | adapter overhead < 3 % (> 5 %) | **47.5 %, 42.4 %** (253.5, 278.5 against 483.1 tok/s) | **dead** |
| 2 | REPLACE pause 1.5–2.5 s (> 4 s) | 1.456, 1.821, 1.697 s | **alive** (one under the band's floor) |
| 3 | sessions continue: 8/8 kept, 20 tokens identical, no re-prefill (any lost or re-prefilled) | 8/8 kept and identical in 3/3; replay from the committed count | **alive** |
| 4 | restored rows equal (any mismatch) | 160/160 tables in 3/3 REPLACE and 2/2 notice | **alive** |
| 5 | notice pause < 0.5 s, swap in both (no swap, > 1.0 s) | 0.527, 0.704 s; swap 2/2 | **weakened** |
| 6 | commit before resume, epochs matched (commit after resume) | 15.9–21.0 ms REPLACE, 53.3 and 156.2 ms notice; epochs matched | **alive** |
| 7 | store span `sender` p50 < 1 ms per rank (> 3 ms) | 2.1–3.4 ms; over 3 ms on at least one rank in 4 of 8 mirror-on runs (adapter_1, replace_mid_1–3) | **dead** |
| 8 | shadow sets: formation < 60 s, < 1 GB per rank (fails, or > 2 GB) | 1.9–2.6 s; adapter minus stock at ready 1.05–1.91 GB per rank (an upper bound) | **weakened** |

## Prediction 1: what costs 45 %

- **The decode step itself is as fast as stock's.** The median step is 16.6 ms in both, and the
  slot collective and the gather kernel ran on every rank (`paths`: `kernel`, `p2p`).
- **The step thread waits on the mirror's one-step fence.** Each step's commit took a median
  27–29 ms from queued to acknowledged. Of that, `event_lag` (queued to the copies landing) was
  23.5–24 ms and `sender` (the store's round trip) 2.5–3.4 ms. That is longer than a 16.6 ms step.
- The fence holds at most one step in flight, so the step thread waited: `fence_waits_s` was
  8.1–8.3 s per rank per run, against 3.2 s for stock's whole decode window.
- **The 30B MoE model on A100 hid the same path.** There `event_lag` was about 15 ms (1.7 steps) and
  `sender` about 0.5 ms. The commit fit inside async scheduling's lead, and throughput equalled
  mirror-off.
- **The store span grows with the tables per step.** This model has 80 layers and 2 KV heads per
  rank: 160 tables per step, against 48 on the 30B. That is also prediction 7's failure. The
  scaling is inferred from the two shapes, not measured on one.
- **The fix belongs to the commit path, not this run:** fewer store operations per step, the gather
  sooner after its forward, or more than one step in flight. It needs its own decision and a build
  session on this shape before an overhead re-run.

## Decision 0013's checks, every fault run

| Check | REPLACE mid ×3 | notice ×2 | 16K REPLACE |
|---|---|---|---|
| committed K/V rows unchanged on all 4 ranks | pass ×3 | pass ×2 | pass |
| first step back: one token per request from the committed count, FULL graph, nothing re-added | pass ×3 | pass ×2 | pass |
| logits equal the no-fault reference's | pass ×1, fail ×2 | fail ×2 | fail |

**The logits check is not measurable on this pod as it stands.**
- `stock_1` and `stock_2` differ in one session's logprobs from token 0 with no change at all, while
  their tokens agree. Logprob values are not bit-reproducible between runs here; tokens are.
- In `replace_mid_2`, `replace_mid_3` and `notice_1` the first differences come before the change,
  at tokens 11–68 against a fault after token 100.
- The H100 determinism check of 2026-10-06 compared tokens, not logprobs, so it could not have shown
  this.

**`notice_2` is the exception.**
- Every session's logits differ from the reference from the swap step on (token 328), by more than
  the last bits (−14.25 against −14.22 in one entry). Before it, they match.
- Its tokens stayed identical. `notice_1`, which swapped at token 276, matches after its swap.
- So something in the second notice's swap changed the arithmetic. The cause is open.

## Post-hoc runs (outside the verdicts)

- **Stock at vLLM's defaults (run 10).**
  - As written, it failed at start: NCCL "unhandled cuda error".
  - Rerun, labelled `posthoc_stock_defaults_nvls0`, with vLLM's all-reduce defaults (FlashInfer and
    symmetric-memory all-reduce on) and only NCCL NVLS off. It ran 482.8 tok/s against 483.1 with
    the determinism settings: **the settings cost nothing measurable (−0.07 %)**.
- **REPLACE mid at 16,384-token prompts (run 11).**
  - Both 16K runs first failed: the harness started the store with an 8 GB budget, and 8.19 GB was
    needed. Rerun with a 120 GB budget (a harness setting).

| 16K REPLACE | Pause | Detect | Change | Restore | Resume | First token | Sessions |
|---|---|---|---|---|---|---|---|
| `posthoc_replace_mid16k` | **4.52 s** | 0.80 | 0.005 | **3.10** | 0.068 | 0.53 | 8/8 kept, 20 tokens identical |

The restore went from 0.36–0.41 s at 2,048-token prompts to 3.10 s at 16,384: roughly linear in the
rows copied. That puts the pause above the 2.5–4 s the pre-registration expected. Its throughput
(253.6 against the 16K reference's 254.6) carries the same commit-path cost.

## The REPLACE pause, with each abort

| Run | Pause | Detect | Change | Abort, live TP (ranks 0/1/3) | Restore | Resume | First token | Daemon change |
|---|---|---|---|---|---|---|---|---|
| mid 1 | 1.456 | 0.569 | 0.265 | 0.619 / 0.618 / 0.620 | 0.364 | 0.016 | 0.235 | 0.628 |
| mid 2 | 1.821 | 0.646 | 0.455 | 0.348 / 0.855 / 0.382 | 0.410 | 0.021 | 0.282 | 0.864 |
| mid 3 | 1.697 | 0.647 | 0.448 | 0.402 / 0.852 / 0.401 | 0.414 | 0.020 | 0.161 | 0.861 |

A dense model has no EP group, so the live TP handle's abort is the only NCCL abort; the torch
group's was 3–4 ms.

## Surprises

- **The overhead is the commit path's**, and the four-GPU runs could not have shown it: their
  model fit the fence.
- **Logprobs are not bit-reproducible between runs on H100 even with the determinism settings;
  tokens are.**
- **The determinism settings are free on H100** at this shape. vLLM's defaults with NVLS on did not
  start in this container.
- **The 16K restore is 3.1 s**, which makes the restore the pause's largest term at long context.
  The incremental restore on main (`c971c69`) is not wired into the spare in this run, as the brief
  said.

## Dollars

| What | Cost |
|---|---|
| Pod `j7cipj43ecar2r`, 01:54:31Z–02:48:37Z (54.1 min) at $17.45/h | **$15.73** (cap $60) |

The balance went from $62.76 to $47.06. The weights and `/dev/shm` were deleted, the pod was
terminated, and no pods remain.
