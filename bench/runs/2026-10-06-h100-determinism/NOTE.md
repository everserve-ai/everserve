# H100 determinism check, 2026-10-05

**Question.** Does stock vLLM give the same tokens twice on H100 with the Hopper all-reduce backends
off, as the harness sets them (`VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
NCCL_NVLS_ENABLE=0`)? And at what throughput cost against vLLM's defaults?

**Answer.** **Yes: identical tokens in all 8 sessions** across two stock runs (custom all-reduce on,
Qwen3-30B-A3B TP4 + EP, 8 sessions × 2,048-token prompts, 201 tokens). Batch-invariant mode was not
needed and not run. **The throughput cost is not measured**: the run with vLLM's defaults failed (see
below).

| Configuration | Tokens, run 1 against run 2 | tok/s, steps 10–200 |
|---|---|---|
| all-reduce settings as the harness sets them | identical, 8/8 sessions | 1425.3, 1423.6 |
| vLLM's defaults | not measured | not measured |

On the earlier 5× H100 pod (`bench/runs/PLANNED/kernels-order-session/run.log`), two stock runs with
the defaults differed in sessions 1 and 3, from tokens 123 and 5. NVLS off alone did not change that.
So it is FlashInfer's all-reduce, symmetric-memory all-reduce, or both; this check does not separate
the two.

**Why the default run failed.** With the defaults, FlashInfer's all-reduce compiles its kernel with
`nvcc` on first use (`trtllm_mnnvl_comm`). This pod was set up for stock runs only, without the
CUDA toolkit. The toolkit install had not finished after 10 minutes, and the pod was stopped at the
cap. A full harness setup includes the toolkit; the frontier pod's stock arm can measure the cost
directly.

**What it means for the frontier pre-registration.** Identity arms on H100 can be written with these
settings in every arm, stock included, with the determinism control as on A100. Whether stock pays
for them in throughput is open until a default-settings stock run sits beside them on one pod.

## Pod and spend

| What | Cost |
|---|---|
| Pod `u1uk92gg94we4h`: 4× H100 80GB HBM3 SXM, NV18 every pair, secure cloud, IN, driver 580.126.09; 22:57:35Z–23:15:43Z (18.1 min) at $13.96/h | **$4.22** (cap $5) |

The pod was terminated and its watchdog stopped; no pods remain. Commands and timestamps are in
`run.log`, figures and hashes in `results.json`, and raw outputs in `data/` (not in git).
