# Pre-registration: the adapter port on the four-GPU rig

Status: PLANNED 2026-10-04. Spend: about 4 GPU-hours on 5x A40-class (roughly $5), inside the
standing $10 budget; no new approval needed.

## Why this run exists

Every core module landed in `everserve-ai/everserve` this week was ported from the harness and
proven on CPU with gloo. The vLLM adapter was deliberately not ported: it is the code that reaches
into engine internals, and it cannot be validated without an engine on a GPU. Phase 1 step 1 ("one
runtime") is not done until the adapter runs the new `Runtime`, `Collective`, `MirrorCommittedKV`
and the daemon link on real GPUs. This run is that validation, on the cheapest hardware that can
do it, before any frontier-scale spend.

## What runs

- Engine: vLLM at the pin in `adapters/vllm/PIN`, no file under `vllm/` edited.
- Model: the small Qwen3-MoE checkpoint the harness used (EP4), 8 sessions, 200 decode steps.
  Checkpoint: `Qwen/Qwen3-30B-A3B` (bf16, Hugging Face), downloaded to
  `/workspace/models/Qwen3-30B-A3B` on the pod, as in the harness's week five.
- Pod: RunPod 5x NVIDIA A40 48 GB (GPUs 0-3 serve, GPU 4 the spare), community cloud at
  $1.75/h if available, else secure cloud at $2.45/h (week five's pod type); `NCCL_P2P_DISABLE=1`,
  `disable_custom_all_reduce=True`. Target $5, hard cap $10, laptop watchdog terminates the pod
  before the cap. Account balance at registration: $52.03 (2026-10-04T22:34Z).
- Adapter: the harness week-five plugin ported onto this repository's core: executor subclass
  (worker death is an event), the graph-safe custom all-reduce op calling `Collective`, the
  connector feeding `MirrorCommittedKV` with the Rust `everserve-store` as the mirror, the
  `DaemonConnection` to the Rust node daemon with the pilot policy document, a pre-loaded spare
  on the fifth GPU, shadow communicators for every replacement set.
- Fault: SIGKILL of rank 2 at step 100, between steps and mid-step; and the notice path, a
  `Notice` 10 s before a planned kill.
- Outcome under test: REPLACE (decision 0004). SHRINK is covered by the CPU suite and is not run.

## Predictions (kill conditions in bold)

1. Pause on a REPLACE fault, measured as the largest inter-token gap: 2.0–3.5 s, matching the
   harness's 2.56 s within the rig's noise. **Kill: > 5 s.**
2. Every session continues with its KV in place: the tokens after the fault equal a no-fault run
   on the same seed for the first 20 steps. **Kill: any session re-prefilled or lost.**
3. Steady-state overhead of the adapter versus stock vLLM at the same pin: 4.0–6.0 % per step
   (the host-memory slot collective, decision 0006, is what the P2P kernel later removes).
   **Kill: > 8 %.**
4. The daemon's `MembershipCommitted` arrives within 100 ms of the engine resuming, with the
   epoch the workers acknowledged. **Kill: no commit, or an epoch mismatch.**
5. The mirror's restore bytes equal the dead rank's committed rows (checked against a host copy
   taken at the last committed step). **Kill: any mismatch.**
6. The notice path: no fault observed by the engine; pause equals the swap alone, under 1 s.
   **Kill: a fault event in the runtime log.**

## What is recorded

`results.json` with every figure above per run, three runs per arm; `RESULT.md`; the runtime
event logs (`rank*.json`) and the daemon log; the private-internals delta the port needed,
written into `adapters/vllm/PRIVATE-INTERNALS.md` in the same landing.

## What this run does not claim

Nothing about frontier shapes, NVLink hosts or multi-node. It proves the port, not the product.
