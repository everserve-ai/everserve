# Battery dry run, pass 11 (2026-10-09 10:15–10:43 UTC): the gap at its floor, read

Three fresh single-GPU L40S VMs on Nebius eu-north1 (`gpu-l40s-a 1gpu-8vcpu-32gb`; node-a
10.0.0.14 rank 0 and the head, node-b 10.0.0.51 rank 1, node-c 10.0.0.10 the spare's node),
Qwen3-1.7B at TP1 x PP2, 8 sessions x 512 prompt, 1792 tokens, blocks 1400, store slots 22400.
Deployed on the user's word ("ok lets do another smoke i need to be 100% sure") at 10:15Z, tree
main `728240d` (pass 10's tree plus: the exchange's wait on a communicator's previous transfer
ends at the refusal, not at the abort; the log-prob kernel warmed at every block up to
`max_logprobs` on every rank; the survivor's boundary stamped with its entry and each release's
start and end, `survivor_lag_s` in the change record). Driver `battery-dry8.sh` (scratch). Node
fault once and single death once on packed depth 2 under deterministic mode and the staged cache,
no bypass. Released at 10:43Z: 0.45 h x 3 x $1.55/h = $2.09.

## Readiness before each kill

One key for the engine (`35fd1f19…`), rank 0 `0bfe1bf4…`, rank 1 `1065d384…`, the spare for rank 1
admitted with rank 1's pair and warm, `compile_cache_consistent` true, world 2 of 2, pool 1 of 1,
node-b reclaimable with outcome `replace`, on both trials.

## The trials

| trial | resumed at | coverage | sessions identical | first differing position vs the control | detection | acks | join legs (offers / answers / subscribed / ring / boundary) | survivor lag | restore | kill→resume | first step after the resume | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| node_fault 1 | 1843 | 1.0 (1.28M rows) | 8/8 | none on both ranks | 15.1 ms | 0.209 s | 15 / 5 / 0.1 / 1.2 / 0.1 ms | 0 ms | 0.206 s | 0.236 s | 0.016 s | 0.31 s |
| single_death 1 | 2037 | 1.0 (1.37M rows) | 8/8 | none on both ranks | 6.2 ms | 0.220 s | 9 / 5 / 0.1 / 0.9 / 0.2 ms | 0 ms | 0.217 s | 0.240 s | 0.016 s | 0.32 s |

Row-exact on both ranks against the controls, every request at its committed count, no
re-prefill, the restored rows all matched, under the register check with no bypass.

## The gap, term by term

Pass 10 read the gap at 0.75 and 0.63 s; this pass reads 0.31 and 0.32 s. The two terms the
previous passes left are gone:

- **The join's `ring_ready` leg**, 229–256 ms on three of pass 9's and 10's four joins, reads
  1.2 and 0.9 ms here. The cause was the exchange's blocking device wait on the previous transfer
  to the dead peer, which ended only when the old communicator's abort freed it; with the wait
  polled against the refusal the survivor's step ends at the refusal. `survivor_lag_s` (the offers
  to the survivor's boundary entry) reads 0 ms on both trials: the survivor was at its boundary
  when the offers went out. The boundary's `release_times` show the fabric and torch releases
  started inside the boundary and still running when the record was taken (`released: []`): the
  abort runs behind the step, as decision 0025 has it.
- **The first step after the resume**, 0.627 s on pass 9 and 0.155 s on pass 10's node fault,
  reads 0.016 s on both trials (a step is 6.6–7.1 ms; the first carries the restored batch's
  first pass). vLLM's JIT monitor reported no kernel compiled during inference on any node in
  either trial, the serving ranks included: both logged "warming the log-prob kernel at [0, 1,
  3, 15] log-probs (max 20)" at start, and the spare logged its `kernel_warmup` and
  `warmup_kernels` on pipeline stage 1 alone.

What remains of kill→resume (0.236 / 0.240 s) is the acknowledgement wait (0.209 / 0.220 s),
inside which the spare restores (0.206 / 0.217 s for 1.28M / 1.37M rows from the staged
membership) and answers; then the boundary (22 / 15 ms) and the resume (5 ms). The restore is
now the floor of the gap on this shape; detection adds 6–15 ms. Nothing in the path waits on a
release or on a kernel.

## The spare

Attached in 55.3 s on its first launch (warm 15.8 s: the stage's kernel warm-ups, the extra
log-prob passes and the graph capture), 27.6 s on the second (warm 15.1 s), 14.2 and 14.3 s on
the third and fourth (warm 2.0 s, the compile cache and Triton's cache warm). It subscribed to the
new ring in 1–2 ms on both joins (the broadcast queue 1 ms, the response queue 1 ms).

## Cost

Deployed 10:15Z, released 10:43Z by the driver: 0.45 h x 3 x $1.55/h = $2.09. Passes 5–11 ≈ $30;
the day's dry runs ≈ $51.

## Clocks

Every clock probe of this pass (`clockoffset.py`, UDP 7410, node-a to node-b and node-c at the
start and the end) got no reply: the VPC passed no UDP between the VMs, so `clocks.json` holds no
offset and every cross-node span in this record carries an unmeasured one. Single-clock figures
are unaffected: the gap and the first step after the resume (node-a's steps), kill→resume (the
coordinator's and the executor's stamps, both on node-a). Cross-node: detection (the executor's
rank_lost event on node-a against the kill stamped on node-b), and the spare's stamps against the
executor's. The same holds for passes 5 to 10. The probe runs over TCP from the next pod on.
