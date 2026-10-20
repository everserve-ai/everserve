# Battery dry run, pass 16 (2026-10-09 22:09–22:37 UTC): FP8 through the resident shape, exact

The p3-resident pre-registration's prerequisite 3, on Cort's word: a small model's FP8 checkpoint
through the resident load, the take, Inductor and graph capture, every fault arm exact, before
any H100 is booked. Three fresh single-GPU L40S VMs on Nebius eu-north1 (node-a rank 0, node-b
rank 1, Qwen3-1.7B-FP8 at TP1 x PP2; node-c holding the whole model pinned with one spare standing
for any position, nothing on its GPU). Block FP8 (128 x 128 weights, dynamic activations); the KV
cache stays bf16. Deploy pinned to `ef411a4` (git archive). Node fault and single death once each;
no stock arm (not an FP8 question).

## Result

| arm | sessions identical | restored rows mismatched | backup after the take | store refusals | gap |
|---|---|---|---|---|---|
| node fault (node-c cold) | 8/8 | 0 (coverage 1.0, 1.81M rows) | 1.0 / 1.0 | 0 | 23.76 s |
| single death | 8/8 | 0 (coverage 1.0, 1.80M rows) | 1.0 / 1.0 | 0 | 5.92 s |

Exact on both arms, no request re-prefilled, every resume at the committed position, replica lag
0 steps. The resident loader took the FP8 shard (507 tensors, 2.65 GB pinned, held in 2.1 s), its
scales included, and vLLM's post-load processing ran on it: the quantization rule of `753c0a6`
works on a real FP8 checkpoint. The take's terms came from the change record (`ef411a4`), not a log.

## The take

| term | node fault (cold) | single death | pass 15 bf16 cold / warm |
|---|---|---|---|
| the take | 22.56 | 5.55 | 9.21 / 5.95 |
| identity and slot | 0.00 | 0.00 | 0.00 / 0.00 |
| the worker built as the rank | 0.24 | 0.23 | 0.23 / 0.22 |
| the modules | 4.39 | 1.78 | 4.36 / 1.77 |
| the weights from the host copy | 0.09 | 0.09 | 0.09 / 0.09 |
| the KV configuration | 1.70 | 1.20 | 1.44 / 1.72 |
| the warm-up and graph capture | 16.00 | 2.11 | 2.96 / 2.02 |

On a warm node FP8 reads like bf16 (5.9 s against 7.1 s). On the cold node the warm-up is 13 s
longer, all of it inside vLLM's own `warmup_kernels` pass on stage 1 (22:24:26.3 to 22:24:38.9 on
node-c; 0.16 s on the same node for single death, 0.16 s on pass 15's cold node). The serving
node-b paid the same 13 s at its first two starts and nothing from its third on. The staged caches
(vLLM's torch_compile_cache and ~/.triton/cache, from node-b, before the spare launched) did not
cover it, so whatever compiles there is cached somewhere neither holds. Not yet named: the nodes are
released and the logs do not say which kernel compiled. It is the FP8 path's term to name before the
rental: read the caches on the spare's node before and after a cold take (the H100 smoke,
prerequisite 5, can carry it), and stage what it finds as 0029 stages the rest.

## What this pass does not show

On the L40S (Ada) vLLM runs FP8 weight-only through Marlin ("Your GPU does not have native support
for FP8 computation"); the H100 runs native FP8 kernels. This pass proves the adapter's FP8 path
(loading, post-processing, the take, the restore, exactness) and not the H100's FP8 kernels or
their determinism, which the H100 smoke reads.

## The pass's own detour

The first launch (record `dryrun-1gpu-fp8-pass16`) stopped at its deploy: a log line in the driver
said "the user's $20" and bash read `$2` under `set -u`. The driver released the three VMs as built
(up 1.6 to 5.3 min each), and the pass ran again from a fixed copy.

## Cost

False start 22:03:30 to 22:08:54Z, 10.5 VM-minutes: $0.27. The pass 22:09:54 to 22:36:52Z, 75.6
VM-minutes: $1.95. At $1.55/h per VM, $2.22 in all. The user's $20: passes 13 to 16, $16.37.
