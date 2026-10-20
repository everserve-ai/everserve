# Pre-registration: single-node frontier run (phase 1, step 4)

Status: PLANNED 2026-10-06, budget approved by the owner (cap $60). Supersedes the 2026-10-04 draft, which sized
the run at 8x H100 for two to three runs of three hours ($120–270). This draft is one pod, one
session, about $60 at the cap, for the reasons under "Why this shape".

Prerequisite: `p1-kernels-4gpu` passed, or its failures are understood and not in the path this
run measures.

## What runs

- Pod: 5x H100 80 GB SXM on one NVLink host, peer-to-peer on, `/dev/shm` large enough for the
  tables (checked first; see decision 0011). Community-cloud rate about $2.50–3.00 per GPU-hour,
  so $13–15 per pod-hour. The exact instance, driver, NCCL and torch versions are written here
  before the first run.
- Engine: vLLM at the adapter's pin. Model: a dense 70B-class open checkpoint in bf16 at TP4, with
  the fourth and fifth GPUs as rank 3 and the spare: Llama 3.3 70B Instruct or Qwen2.5 72B
  Instruct, whichever downloads first; the choice is written here before the first run. Weights
  are about 140 GB, so TP4 leaves about 100 GB of KV across the four ranks.
- Adapter and runtime as in the four-GPU runs: shared-memory mirror (0011), one gather kernel per
  step (0010), the peer-to-peer slot collective (0006), concurrent aborts.
- Sessions: 8 concurrent, 2,048-token prompts, 200 decode steps; kill rank 2 at step 100 (REPLACE
  arms) or notice rank 2 after step 100 with the deadline 300 steps out (notice arms).

## Written before the first measured arm (2026-10-06T02:16Z)

- **Model: Qwen2.5 72B Instruct** (`Qwen/Qwen2.5-72B-Instruct`, bf16, 136 GB on disk). Llama 3.3
  70B Instruct was tried first and refused: the repository is gated and the pod has no Hugging
  Face token ("Access denied. This repository requires approval."). Qwen2.5 72B downloaded in
  8 minutes. Its hidden size is 8,192, so an 8-session decode all-reduce is 65,536 elements, the
  slot collective's largest size: 8-session steps run on it, and a 16-request batch (captured, never
  used at 8 sessions) would fall back to NCCL inside its graph.
- **Instance:** RunPod pod `j7cipj43ecar2r`, 5× NVIDIA H100 80GB HBM3 SXM on one host, NVLink NV18
  between every pair, secure cloud, US, $17.45/h. Container disk 250 GB.
- **Driver 580.126.09** with `cuda-compat-13-0`; **NCCL 2.29.7**; **torch 2.13.0+cu130**; vLLM
  `0.30.1rc1.dev606+g6e517b15c` (the pin).
- **`/dev/shm`: 586 GB**, enough for the 16K run's tables (about 11 GB per rank) and staging; run 11
  is not skipped.
- **Determinism on this pod:** two identical stock runs with the harness's settings
  (`VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 NCCL_NVLS_ENABLE=0`, custom
  all-reduce on) gave identical tokens in all 8 sessions.
- **Harness changes for a dense model, bench only:** the drill enables expert parallelism only for a
  checkpoint with experts (the first determinism pair failed at config without it); runs time out
  at 25 minutes instead of 10; every rank's device memory is recorded at engine-ready (for
  prediction 8).
- **References, not scored arms:**
  - Decision 0013's logits assertion needs a no-fault reference of the same length for every fault
    arm.
  - The REPLACE arms use `adapter_1`.
  - The notice arms use `notice_ref`: one no-fault adapter run of 601 tokens, run just before them.
  - The 16K REPLACE uses `posthoc_ref16k`: one no-fault adapter run at 16,384-token prompts, run
    just before it.

## Arms, in order, nothing repeated

1. stock x2 (the baseline on this pod; stock's custom all-reduce enabled if the pod allows, and
   recorded either way);
2. adapter, no fault x2 (the overhead);
3. REPLACE mid-step x3;
4. notice x2.

Nine scored runs. At about 6 minutes each including engine start, with the weight download (about
140 GB, 15–25 minutes on a community pod) and the kernel build, the session is about 2.5 hours:
$35–45, cap $60.

Two more runs after the nine, labelled post-hoc and outside the verdicts:

10. stock x1 at vLLM's default settings, to price the determinism settings against stock itself;
11. REPLACE mid x1 with 8 sessions at 16,384-token prompts (`--prompt-len 16384`,
    `--max-model-len` and blocks raised to fit), one fault at step 100. Long contexts are where a
    lost session costs most, and the restore scales with them: about 10 GB per rank at this length
    against 1.3 GB at 2,048, so the pause is expected nearer 2.5–4 s. Reported as the scaling of
    restore and pause with context, not scored. `/dev/shm` must hold the tables for it (checked
    before the first arm); if it cannot, the run is skipped and the reason written here. No mirror-off arm: the commit's cost was measured at a third of a percent on the kernel
session and the comparison it attributed is settled.

## Why this shape

- TP4 plus a spare is the smallest shape where the model is frontier-class and every mechanism
  (mirror, gather, slot collective, shadow sets, spare, daemon) is exercised as it would be at TP8.
  The rank count changes the arithmetic, not the mechanism.
- Dense rather than MoE: the four-GPU runs used a 30B MoE; a dense 70B is the shape most neocloud
  customers serve and the one a buyer will ask about first.
- Nine arms rather than eighteen: every verdict below is a within-pod comparison or an absolute
  band that two or three runs can speak to; the four-GPU runs showed the spread between repeats
  of one arm is small against the bands.

## Predictions (kill conditions in bold)

1. **Steady-state overhead** of the adapter against stock, same pod: under 3 %. **Kill: over 5 %.**
   Evidence: 903.5 against 906.8 tok/s with the mirror on and off on the kernel session; the
   remaining term is the slot collective against the all-reduce stock uses on this pod.
2. **REPLACE pause**, as the largest inter-token gap of any session: 1.5–2.5 s. **Kill: over 4 s.**
   Terms: detection about 0.35 s; NCCL abort 0.4–0.8 s (the named remainder, decision pending);
   restore 0.3–0.6 s for about 1.3 GB per rank (80 layers x 2 KV heads x 128 x 2 bytes x 2 per
   token, 16k tokens), from 0.17–0.20 s for 420 MB on the A100 run; first token about 0.2 s.
3. **Sessions continue**: 8/8 kept, token-identical for 20 steps after the fault, resumed from the
   committed count with no re-prefill. **Kill: any session lost or re-prefilled.**
4. **Restored rows equal** the dead rank's: every table. **Kill: any mismatch.**
5. **Notice pause** under 0.5 s with the swap in both runs. **Kill: no swap, or over 1.0 s.**
6. **Commit lands before the engine resumes**, epochs matched. **Kill: a commit after resume.**
7. **Store span** (copies landed to acknowledgement, from the timeline) p50 under 1 ms per rank.
   **Kill: over 3 ms.**
8. **Shadow set formation** at startup under 60 s and under 1 GB of device memory per rank.
   **Kill: formation fails or exceeds 2 GB per rank.**

## What is recorded

As the four-GPU runs: `results.json` with every arm's figures, `/metrics` from the daemon and the
store after every arm, change seconds per cause and outcome, `commit.transport` and the timeline
spans per rank, every rank's event log in `flake/`, the pod's topology, versions and `/dev/shm`
size, and the spend.

## What this run does not claim

Multi-node, TP8, MoE at frontier scale, or a pause under 1 s. The abort floor is a design question
for after this run. Step 5 (two nodes, a node-level notice) is its own pre-registration.
