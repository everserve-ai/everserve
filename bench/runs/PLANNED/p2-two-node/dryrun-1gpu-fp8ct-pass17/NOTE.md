# Battery dry run, pass 17 (2026-10-09 23:28–00:03 UTC): the rental's FP8 scheme exact, the cold term named

The rental checkpoint's FP8 scheme on the cheap shape, on Cort's word: RedHatAI/Qwen3-1.7B-FP8-dynamic
(compressed-tensors, FP8 weights per channel, dynamic per-token activations: the scheme of
RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic, the p3 pick), on three fresh single-GPU L40S VMs on
Nebius eu-north1, the resident shape (node-a rank 0, node-b rank 1 at TP1 x PP2; node-c the shared
spare, the whole model pinned, nothing on its GPU). Deploy pinned to `865a0ba`. The first live run
of the harness's per-start reads (written_since.py), the graceful down and the repeat arm. On Ada
this scheme runs vLLM's CUTLASS FP8 kernels natively (pass 16's "fp8" method fell back to Marlin).

## Result

| arm | sessions identical | restored rows mismatched | backup after the take | store refusals | gap |
|---|---|---|---|---|---|
| repeat (two runs, no fault) | 8/8 agree, all 1,792 tokens | n/a | n/a | 0 | n/a |
| node fault (node-c's first take) | 8/8 | 0 (coverage 1.0) | 1.0 / 1.0 | 0 | 10.73 s |
| single death (node-c's second take) | 8/8 | 0 (coverage 1.0) | 1.0 / 1.0 | 0 | 6.93 s |

The engine repeats itself token for token, so the fault arms' exactness is attributable to the
change. Both fault arms exact; no request re-prefilled.

## The take

| term | first take (cold) | second take | pass 16 ("fp8", Marlin) cold / warm |
|---|---|---|---|
| the take | 9.58 | 5.74 | 22.56 / 5.55 |
| the modules | 4.37 | 1.79 | 4.39 / 1.78 |
| the weights from the host copy | 0.08 | 0.09 | 0.09 / 0.09 |
| the KV configuration | 1.66 | 1.41 | 1.70 / 1.20 |
| the warm-up and graph capture | 3.00 | 2.01 | 16.00 / 2.11 |

Pass 16's 13 s inside `warmup_kernels` does not appear under this scheme. It stays unexplained:
it was read only on vLLM's "fp8" method through the Marlin fallback on Ada, which is not the
rental's path, and nothing here names it. It is open, not closed by this pass.

## What a start writes (written_since.py)

Every start on every node was marked before and read while it ran and once stopped.

- **node-c's first take wrote 1,585 files, 37.9 MB; 1,578 of them (33.7 MB) are Python bytecode**
  under `.venv/lib/python3.12/site-packages/**/__pycache__` (flashinfer, nvidia_cutlass_dsl, z3,
  sympy, torch/_inductor/fx_passes): modules first imported at the take, compiled to bytecode on
  that import because the install (uv) does not compile bytecode. The second take wrote only the
  5 files (4.4 MB) the node's observability agent writes under every read.
- node-b's first start wrote 4,967 files (119 MB: the same bytecode, the kernels' torch-extension
  build, the repo's own `__pycache__`); its first start under a fault arm's control, 300 (18 MB).
- Every stopped read equals its running read: nothing is written at exit.
- Exit kinds: the serving and standing processes stop clean on SIGINT within 20 s; a spare that
  has taken a position does not (killed after the grace on both fault arms). A product defect (a
  drain, a node shutdown and an upgrade all send that signal), to be found and fixed before the
  rental.

The cold take's extra modules time is first imports and their bytecode, an install step: the
install compiles the environment's bytecode (`--compile-bytecode`), so no process compiles it on
a change; the H100 smoke's reads confirm it.

## Cost

VMs 23:28:53 to 00:03:23Z, 98.2 VM-minutes at $1.55/h: $2.54. The user's $20: passes 13 to 17,
$18.91.
