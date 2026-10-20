# Battery dry run, pass 5 (2026-10-08 19:52–20:43 UTC): the coverage fix under node fault x2 and single death x1

Three single-GPU L40S VMs on Nebius eu-north1 (`gpu-l40s-a 1gpu-8vcpu-32gb`), node-a rank 0 and
the head, node-b rank 1, node-c the spare's node, Qwen3-1.7B at TP1 x PP2, 8 sessions x 512
prompt, 1792 tokens, blocks 1400, store slots 22400. Deployed on the user's word at 19:52Z;
arms `node_fault` (2 trials) then `single_death` (1 trial), a fresh control per trial, the
driver `battery-dry2.sh` then `battery-dry2b.sh` (scratch).

Trees: attempt 1 on main `b32e952` (the exchange gate, the packed pipeline route, the drill's
`post_resume`, `re_prefilled` and `restore_coverage` fields) over the peer's `e816775` (a
carrier skips a queued step only when a snapshot that carried rows, or an acknowledgement,
covered it) and `ec6b241` (restore coverage and re-prefill scored in the pre-registration).
Attempt 2 on `b9671b1`: the packed route off by default.

## Attempt 1 (20:04–20:13Z): the control refused on its first step

The control engine's second stage lost its pipeline peer at the first scheduled step:
`irecv_tensor_dict: RuntimeError: a scheduled step whose layout this stage cannot name: the
runner's intermediate tensors or its padding are missing` (node-b's worker, reported lost by
rank 1 at 20:07:23). The packed route (`b32e952`) made the receiving stage refuse inside a step
whose layout it could not name, since a header receive against a packed send hangs both stages;
on this engine it never named one. The driver stopped before its release line (the battery
exits 0 after a stopped arm); the VMs were held.

Fix (`b9671b1`): the packed route is off unless the bootstrap turns it on
(`Bootstrap.pipeline_packed`, `pipeline.PACKED`); the header path carries every step on both
sides, and an unpacked receiver asks for its layout on its first three steps and logs why it
has none. The probe on attempt 2 read, three times on node-b: `the runner's padding raised:
AttributeError: 'GPUModelRunner' object has no attribute '_determine_batch_execution_and_padding'`.
The engine at this pin (`vllm-0.30.1rc1.dev606+g6e517b15c`) runs model runner v2
(`v1/worker/gpu/model_runner.py`), whose padded token count is `max(num_tokens,
batch_desc.num_tokens)` from its own `BatchExecutionDescriptor`; the old runner's private is not
there. The route is rebuilt on v2's descriptor before it is measured.

## Attempt 2 (20:19–20:43Z): three trials

Fabric: slowest link to node-a 34.9 and 36.9 Gbit/s (floor 1 on this VPC shape). Pre-kill
signature on every trial: node-c's daemon `rig: rank 1 replicates here from node-b` before
anything was held, node-b's store `rank 1 replicates to ["10.0.0.78:7402"]`, no `sending the
rank as held` line (nothing held at link time, as `e816775` intends), readiness with rank 1's
`replicates_to` non-empty and rank 0's empty, no refused line.

| trial | kill step | resumed at | restore coverage | re-prefilled | sessions identical | detection | kill→resume | client gap |
|---|---|---|---|---|---|---|---|---|
| node_fault 1 | 100 (+) | 1315 | 1.0 (1,178,240 rows = 112 x 8 x 1315) | 0 | 8/8 x 1792 | 9.9 ms | 0.334 s | 2.44 s, digests inside |
| node_fault 2 | 100 (+) | 1093 | 1.0 (979,328 = 112 x 8 x 1093) | 0 | 7/8; request 4 at generated 732 | 9.9 ms | 0.549 s | 2.29 s, digests inside |
| single_death 1 | 100 (+) | 1307 | 1.0 (1,171,072 = 112 x 8 x 1307) | 0 | 8/8 x 1792 | 0.8 ms | 0.214 s | 0.323 s, digests moved off (`instrument_pause_s` 1.574 at step 1615) |

Every restore copied exactly tables x sum of committed counts; catch-up 0 rows; every rank's
rows below the committed count matched the kill-time snapshot (`kv_below_committed` 0,
`slot_mismatch` empty). The first batch after the resume on both ranks: 1 token per request,
`num_computed_prefill_tokens` = the committed count, `is_prefilling` true for all eight,
`cg_mode FULL`, `decode_graph_eligible` true, padded 8 (v2's prefill-runs-as-decode); the second
batch a decode. Both stores sampled at the prompt slots on the single-death trial (the battery
samples slots 16, 17 and 528 of table 0 on every fault trial from this pass on): node-b's store
written_step 1 at all three (its step 1591), node-c's replica written_step 1 at all three (its
step 1993). Live during the node-fault trial-2 control, node-c's replica read slot 16
written_step 1 with bytes `fdbbf03beb3c…`, the same first bytes as the source's kill-time row
in pass 4 (the same seeded prompt).

Resume parts on all three (from the change record): restore 0.17–0.19 s (the full rows now),
acknowledgements 0.18–0.19 s on the coordinator's clock, boundary 0.016–0.35 s, resume RPC
0.01–0.11 s, join 0.004 s. Throughput on the header path at depth 1 during the fault trials'
adapter runs: 870, 861, 852 tok/s over steps 10–200 (stock 1180 on the exchange pass of the same
day: 27 % under).

## Reading

The store side is closed on three trials: the replica holds the prefill step's rows, the restore
covers every committed position, nothing is re-prefilled. Passes 2–4's exact reads were a
decode-only restore (coverage 831/1343 and 569/1081) that the small model's seeded random
prompts mostly did not expose; pass 4's one flip was one token after the resume, this pass's
one flip 151 tokens after it, with the restored state proven equal to the source's.

The remaining flip is on the adapter's side: the step after a resume runs every request as a
one-token chunked prefill (`prefill_len` = the committed count + 1, because the resume re-adds
each request with all of its tokens as `prefill_token_ids`) where the control runs a decode;
same graph bucket, a different attention path for a length-1 query flagged prefill, different
rounding in the K/V row written at the committed position on both ranks; the sampled token
survives, the row stays, and the first thin margin on request 4 (the same seeded prompt every
trial) flips later. The drill's prompts are seeded (`manual_seed(1234)`), which is why it is
always request 4.

Next landing, one gate: the decode-shaped resume (the runner gets the tokens up to the committed
count as its prefill tokens, `num_computed_tokens` the committed count, and the token at the
committed position in `req_states.last_sampled_tokens`, which a decode step reads through
`combine_sampled_and_draft_tokens` at this pin; a test on the step shape and on the token buffer
against v2's request state; a PRIVATE-INTERNALS row for the buffer); the end-of-run per-position
digests on both arms compared offline (first differing position per request, refused on any
preemption or block-table change); the resumed step's descriptor beside the control's; the digest
move (`ROWS_CHECK_DELAY_STEPS` 20, `instrument_pause_s`); top-2 logprobs on both arms of the fault
trials; the packed route on v2's descriptor, measured at depth 1 and depth = pipeline size, mirror
on and off, against stock, the header path and the route switch going when it measures.

## Cost

Three VMs from 19:53Z; control plane down at 20:43Z (0.84 h x 3 x $1.55/h = $3.92); the VMs held
after that on the user's call, released by hand or by the watchdogs at 22:53Z; the final figure
is appended at the release.
