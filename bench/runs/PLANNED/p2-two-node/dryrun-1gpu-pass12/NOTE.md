# Battery dry run, pass 12 (2026-10-09 14:18–16:12 UTC): the resident shape, first run

The shape of decision 0030 on three single-GPU L40S VMs on Nebius eu-north1
(`gpu-l40s-a 1gpu-8vcpu-32gb`): node-a rank 0 and node-b rank 1 serving Qwen3-1.7B at TP1 x PP2,
node-c holding the whole model pinned in host memory (4.06 GB, 2.6 s) with one spare standing for
any position and nothing on its GPU; no `replicaNodes`, so each rank's rows replicate to its
sibling and the coordinator stages the lost rank's rows onto node-c at the change. 8 sessions x
512 prompt, 1792 tokens. Launched on the user's word ("go lets rip"), driver `battery-dry9.sh`,
then the fault arms rerun on the same VMs by hand (`p12-hand3.sh`) after two fixes. Arms: the
stock-restart baseline, node fault, single death, once each.

## Result

| arm | sessions identical | restored rows mismatched | kill→resume | gap |
|---|---|---|---|---|
| stock restart (stock vLLM) | 0/8, all prefilled again | n/a | n/a | 379 s to the first token |
| node fault | 8/8, every token | 0 (coverage 1.0, 1.05M rows) | 10.87 s | 10.95 s |
| single death | 8/8, every token | 0 (coverage 1.0, 1.48M rows) | 11.19 s | 11.28 s |

Readiness before each kill: one spare in the pool standing for any position (`any_spares`), both
nodes `replace`, rank 0 replicating to node-b and rank 1 to node-a, compile cache consistent.

**Stock restart.** Stock vLLM did not fail fast on node-b's loss: the head hung until vLLM's RPC
timeout (`TimeoutError: RPC call to sample_tokens timed out`, 300 s), and the supervisor could
relaunch it only then: 312 s from the kill to the relaunch, the relaunched engine ready in
17.5 s (a 1.7B model with its compile cache warm), its first token 379 s after the kill.

## Where the 11 s goes (node fault; single death within 0.3 s on every term)

| term | s |
|---|---|
| detection | 0.011 |
| the take on the spare: the vLLM worker built as rank 1 and its memory sized (about 3.3), the shard built from the host copy (`shard_copy_s` 1.83), the stage's warm-ups and graph capture (`warm_s` 1.98) | 7.07 |
| the wait for the staged rows | 0.0001 (held and caught up at the first poll) |
| the restore, 1.05M rows | 0.17 |
| the spare's join answer reaching the executor (`answers_back`): the harness's gdb freezing the executor (finding 6) | 3.60 |
| the rest of the boundary, the resume | about 0.03 |

The first step after the resume took 0.017 s; the second 0.54 s (0.62 s on single death), then
steady 7 ms steps. vLLM's JIT monitor named no kernel compiled in inference, so the second step's
term is something it does not watch; open.

## Findings and fixes during the pass

1. **Nebius placement.** Node-c, then node-b on the relaunch, sat in STARTING for 4–5 minutes
   while the other two ran in 40 s; the first attempt's deploy timed out and released all three
   (about $0.40). Not a quota: the project's L40S limit in eu-north1 is 32, 3 in use. A capacity
   delay; `gpu-l40s-d` is the fallback platform.
2. **The take built on the wrong stage (adapter, fixed `4f3c044`).** The standing spare built its
   vLLM worker at start as its placeholder rank 0; the runner fixes its pipeline stage at build,
   so at the take, given rank 1, its profiling run failed on qwen2's
   `assert intermediate_tensors is not None` and the change was never committed. The worker is
   now built at the take, as the rank taken.
3. **Staged rows refused by the fence (coordinator, fixed `5cae0eb`).** The staging expectation
   carried the change's new epoch, which raised node-c's fence before the dead rank's epoch-0
   rows arrived: "refused the rank as held: step 1661 at epoch 0 ... older than the rank's move
   at epoch 1"; the spare waited its 30 s bound and failed the change cleanly. A staging now
   moves no fence.
4. **The spare's first launch exited (adapter, fixed `69ceb21`).** A daemon launches its spares
   at start, before the other nodes' records exist; the spare now waits for them.
5. **Replication after the change is refused both ways (open, store/coordinator).** After the
   commit, node-a's store sending rank 0 to node-c and node-c's store sending rank 1 to node-a
   are both refused by "older than the rank's move at epoch 1", every step: after the take
   neither rank was replicated, so a second loss would not have been exact. node-c's store also
   numbers rank 1's steps from 0 after the take, which may be the spare's commit counter.
6. **The join answer's 3.6 s was the harness.** The answer reached node-a's daemon 0.5 s after
   the offer (the spare's catch-up restore and position), and the executor dequeued it about
   3.1 s later: the battery ran `nodes.sh stacks` 4 s after the kill, and gdb attaching to every
   engine process on node-a froze the executor mid-change (the native-stack file was written at
   16:00:00Z, the answer taken at 16:00:00.4). Pass 11's change finished before the 4 s mark;
   this pass's 7 s take did not. The kill→resume figures above carry it; the harness now takes
   native stacks only for a change that has not committed in 30 s.

## The floor of this shape

On this model, with nothing on the spare's GPU before the change: the shard copy runs at about
1.1 GB/s through vLLM's per-tensor weight loaders (2 GB in 1.8 s) against PCIe gen4's ~20 GB/s
from pinned memory; the worker build and sizing (3.3 s) and the warm-up (2.0 s) are fixed costs
of building a vLLM worker at the take; the join's 3.6 s is the harness's own gdb (finding 6). Removing the defect, sizing
the KV from a recorded configuration and copying the shard in bulk puts the gap near 5 s; building
one worker per eligible stage ahead (no weights, so cheap) and reusing captured graphs over weights
copied into place puts it near 1 s. The preloaded shape's 0.3 s (pass 11) is the bound for a
spare that holds its position before the change.

## Clocks

No offset again: every probe of the TCP client was refused (`[Errno 111] Connection refused`).
The harness starts the probe's server only when none is running, and the UDP server from the
setup was, so nothing listened on TCP; `nodes.sh clocks` now restarts it. Single-clock figures
(gap, kill→resume, the steps) are unaffected; detection and the spare's stamps against the
executor's carry an unmeasured offset, the VMs in one VPC.

## Cost

First attempt 14:18–14:28Z, about $0.40. Second 14:28:58–16:11:57Z, 1.72 h x 3 x $1.55/h = $8.00
(the VMs held through findings 2 and 3 rather than redeployed). Pass 12 about $8.40.
