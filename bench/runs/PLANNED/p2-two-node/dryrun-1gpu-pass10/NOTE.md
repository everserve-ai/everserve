# Battery dry run, pass 10 (2026-10-09 03:43–04:12 UTC): the gap at its floor, on the tree the rental runs on

Three fresh single-GPU L40S VMs on Nebius eu-north1 (`gpu-l40s-a 1gpu-8vcpu-32gb`; node-a
10.0.0.36 rank 0 and the head, node-b 10.0.0.88 rank 1, node-c 10.0.0.1 the spare's node),
Qwen3-1.7B at TP1 x PP2, 8 sessions x 512 prompt, 1792 tokens, blocks 1400, store slots 22400.
Deployed on the user's word ("go for the smoke once everything is landed") at 03:43Z, tree main
`1ba1724` (pass 9's tree plus the spare's warm-up as a property of its attachment: vLLM's kernel
warm-ups run on the spare's pipeline stage alone, before it registers, and it registers for
exactly the position it warmed for). Driver `battery-dry7.sh` (scratch; the `ARMS` value quoted
for its eval this time). Node fault once and single death once on packed depth 2 under
deterministic mode and the staged cache, no bypass. Released at 04:12Z: 0.48 h x 3 x $1.55/h = $2.24.

## Readiness before each kill

One key for the engine (`35fd1f19…`), rank 0 `1e2756a7…`, rank 1 `e6046073…`, the spare for rank 1
admitted with rank 1's pair, `compile_cache_consistent` true, world 2 of 2, pool 1 of 1, on both
trials.

## The trials

| trial | resumed at | coverage | sessions identical | first differing position vs the control | detection | acks | join legs (offers / answers / ring / boundary) | restore | kill→resume | first step after the resume | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| node_fault 1 | 1901 | 1.0 (1.62M rows) | 8/8 | none on both ranks | 10.8 ms | 0.243 s | 23 / 5 / 255 / 0.2 ms | 0.240 s | 0.542 s | 0.155 s | 0.75 s |
| single_death 1 | 2052 | 1.0 (1.84M rows) | 8/8 | none on both ranks | 9.1 ms | 0.275 s | 20 / 5 / 229 / 0.1 ms | 0.272 s | 0.544 s | 0.023 s | 0.63 s |

Row-exact on both ranks against the controls, every request at its committed count, no
re-prefill, the restored rows all matched: the rows claim on the tree with the warm-up in, under
the register check with no bypass.

## The spare's warm-up

The spare on node-c logged "kernel_warmup on pipeline stage 1 alone" and "warmup_kernels on
pipeline stage 1 alone" and attached in 53.6 s on its first launch (warm 15.2 s: the stage's
kernel warm-ups and the graph capture) and 13.9 s on the second (warm 2.0 s, the compile cache
and Triton's cache warm). No error on either; the warm-up's synthetic steps ran through the
stage-alone receive at the runner's own layout.

The first step after the resume: 0.155 s on the node-fault trial against 0.627 s on pass 9's
(the same cold node), 0.023 s on the single-death trial against 0.059 s. On the node-fault trial
vLLM's JIT monitor named exactly one kernel compiled on the spare during inference,
`_topk_log_softmax_kernel` (the sampler's log-probs, `--logprobs 2` on the fault arms); node-b's
serving rank had logged the same kernel compiling during inference in the control, after vLLM's
own warm-up, at its first step with log-probs. That specialization is outside what vLLM's
warm-up covers on any rank; a serving rank pays it once at its first log-probs request and the
spare pays it once at its first step, 0.13 s here. The eight kernels of pass 9 are gone from the
resume. On the single-death trial nothing compiled and the first step cost 0.023 s (the exchange's
first transfer on the new ring 0.011 s).

## The join path's variable term, now located

`ring_ready` (the executor's `wait_until_ready` on the new broadcast queue, then on each spare's
response queue) read 255 and 229 ms on this pass's two joins, 1 and 256 ms on pass 9's: the
0.23–0.26 s is the join path's remaining term, and the restore is not it (the spare restores
during the acknowledgements, from the staged membership, and answers 5 ms after the offer; its
`take_position` is 4 ms; nothing runs between its READY and its own `serve`). The stamps that
split the leg land after this pass: `ring_subscribed` (the broadcast queue's readers) beside
`ring_ready` (the response queues) in `join_legs_s`, and the spare's "subscribed to the ring in
… s (the broadcast queue …, the response queue …)" line. With the leg at 1 ms the gap reads
~0.4 s on this shape (pass 9's node fault: kill→resume 0.391 s).

## Cost

Deployed 03:43Z, released 04:12Z by the driver: 0.48 h x 3 x $1.55/h = $2.24. Passes 5–10 ≈ $28; the
day's dry runs ≈ $49.

## Clocks

Every clock probe of this pass (`clockoffset.py`, UDP 7410, node-a to node-b and node-c at the
start and the end) got no reply: the VPC passed no UDP between the VMs, so `clocks.json` holds no
offset and every cross-node span in this record carries an unmeasured one. Single-clock figures
are unaffected: the gap and the first step after the resume (node-a's steps), kill→resume (the
coordinator's and the executor's stamps, both on node-a). Cross-node: detection (the executor's
rank_lost event on node-a against the kill stamped on node-b), and the spare's stamps against the
executor's. The same holds for passes 5 to 10. The probe runs over TCP from the next pod on.
