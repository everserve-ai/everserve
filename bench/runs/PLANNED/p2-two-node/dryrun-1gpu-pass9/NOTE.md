# Battery dry run, pass 9 (2026-10-09 02:49–03:19 UTC): the clean run, and the gap's last term named

Three fresh single-GPU L40S VMs on Nebius eu-north1 (`gpu-l40s-a 1gpu-8vcpu-32gb`; node-a
10.0.0.75 rank 0 and the head, node-b 10.0.0.33 rank 1, node-c 10.0.0.28 the spare's node),
Qwen3-1.7B at TP1 x PP2, 8 sessions x 512 prompt, 1792 tokens, blocks 1400, store slots 22400.
Deployed on the user's word ("once everything is landed we can go on to the next smoke run",
every landing on main by then) at 02:49Z, tree main `b1a84a4` (the engine's compile key held
to its processes; the key the artifact's hash on every compile path, the digest over the kernel
sources and autotune results; the kill refused on a short pool). Node fault once and single death
once on packed depth 2 under deterministic mode and the staged cache, no bypass. Released at
03:19Z: 0.50 h x 3 x $1.55/h = $2.33.

The driver (`battery-dry6.sh`, scratch) lost the quotes around `ARMS="node_fault single_death"`
inside its `eval` and the battery exited 127 before any arm; it held the VMs and the battery ran
by hand (`p9-hand.sh`) with the same environment, the driver releasing on the hand run's stop
file. (Quote the value for the eval, or pass the battery its environment without one.)

## Readiness before each kill

Both trials read one key for the engine (`35fd1f19…`, the AOT artifact's hash), each serving rank
its own digest (rank 0 `0a4b036a…`, rank 1 `e7b25807…`), the spare for rank 1 admitted with rank
1's pair, `compile_cache_consistent` true, world 2 of 2, pool 1 of 1. The staging's digests
(`cache_digest_b/c.json`) identical before each trial.

## The trials

| trial | resumed at | coverage | sessions identical | first differing position vs the control | detection | acks | join legs (offers / answers / ring / boundary) | restore | kill→resume | first step after the resume | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| node_fault 1 | 1901 | 1.0 (1.70M rows) | 8/8 | none on both ranks | 12.6 ms | 0.250 s | 19 / 5 / 1 / 0.1 ms | 0.247 s | 0.391 s | 0.627 s | 1.07 s |
| single_death 1 | 1836 | 1.0 (1.65M rows) | 8/8 | none on both ranks | 6.4 ms | 0.245 s | 20 / 5 / 256 / 0.1 ms | 0.242 s | 0.541 s | 0.059 s | 0.66 s |

Both row-exact against their controls on both ranks (`first_differing_position` none), every
request at its committed count (`resume_below_committed` 0, `requests_re_prefilled` 0), the
restored rows all matched. The rows claim under the register check with no bypass: the clean
end-to-end run the rental was waiting on.

## The first step after the resume: the spare's Triton kernels

The warm at formation is in (the spare's pre-formed pair group warmed in 0.009–0.010 s on both
trials) and the first step after the node-fault resume still took 0.627 s. vLLM's own JIT
monitor on the spare names it: eight "Triton kernel JIT compilation during inference" warnings
on the spare's first real step, in the resume second, for the V2 runner's helper kernels:
`_prepare_pos_seq_lens_kernel`, `_post_update_kernel`, `_combine_sampled_and_draft_tokens_kernel`
(`v1/worker/gpu/input_batch.py`), `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`
(`block_table.py`), `_apply_write_kernel` (`buffer_utils.py`), `_topk_log_softmax_kernel`,
`_ranks_kernel` (`sample/logprob.py`, the logprobs the fault arms read). node-c's `~/.triton/cache`
gained nine entries at 03:05:56–57 (the resume), node-b's thirty-six were written at 03:05:02–14,
its engine start, under `warmup_kernels` (`v1/worker/gpu/warmup.py`: scheduler-realistic prefill
and decode steps through the worker's `execute_model`), which the spare skips on a pipeline stage
past the first (`spare._skip_pipeline_warmup`: every such step would wait on the stage before, and
a spare has no peer). The single-death trial confirms it: node-c's Triton disk cache warm from the
first trial, the first step after the resume 0.059 s. Pass 7's 0.633 then 0.067 / 0.059 s were
the same term, the disk cache cold then warm; NCCL's lazy setup, warmed since `01578aa`, is off
the list (the spare's first transfer 0.563 s on the node-fault trial was its recv waiting on the
compiling step; 0.045 s on single death).

The landing (the adapter's): the spare runs `warmup_kernels(runner, execute_model, sample_tokens)`
on its stage with the pipeline exchange stubbed (the intermediate tensors made locally as vLLM's
own profiling run makes them on a later stage, the send to the next stage dropped), so its first
step after a join costs what any step costs. Until then a spare's first step after a resume costs
~0.6 s on a node whose Triton cache never compiled these kernels and ~0.06 s on one that has.

**The join path's variable term.** `ring_ready` (the new ring subscribed on the executor's
carrier path) read 1 ms on the node-fault join and 256 ms on the single-death join; the
`boundary_s` of the change record carries it (0.025 against 0.281). The 0.015–0.35 s join-path
spread of pass 7 lives in this leg. What the leg is: the executor's `wait_until_ready` on the new
broadcast queue (every reader subscribed, the spare over ZMQ) and on the spare's response queue
(`executor.py` 565–569); on the single-death join rank 0's boundary record also shows the old
ring's fabric release completing in the same window (`released: ['fabric 0.518s']`, against
`releasing` and not awaited on the node-fault join), the dead rank's node still alive and its
daemon answering. The spare's side of the subscription has no stamp yet (its log reads "took
rank 1's position in 0.004 s" and nothing until the first step); that stamp is the next
instrument, before the leg is named.

**The spare's `shadow_s`** (34–65 s across passes 8 and 9) is the rendezvous wait for the serving
ranks to reach their formation point as the engine loads, not a cost: the spare launched 64 s
before rank 0 formed on the node-fault trial, 26 s before on the single-death one.

## Cost

Deployed 02:49Z, released 03:19Z (the driver on the hand run's stop file): 0.50 h x 3 x $1.55/h = $2.33. Passes 5–9 together ≈ $26; the day's dry runs ≈ $47.

## Clocks

Every clock probe of this pass (`clockoffset.py`, UDP 7410, node-a to node-b and node-c at the
start and the end) got no reply: the VPC passed no UDP between the VMs, so `clocks.json` holds no
offset and every cross-node span in this record carries an unmeasured one. Single-clock figures
are unaffected: the gap and the first step after the resume (node-a's steps), kill→resume (the
coordinator's and the executor's stamps, both on node-a). Cross-node: detection (the executor's
rank_lost event on node-a against the kill stamped on node-b), and the spare's stamps against the
executor's. The same holds for passes 5 to 10. The probe runs over TCP from the next pod on.
