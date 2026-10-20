# Battery dry run, pass 6 (2026-10-08 21:07–22:25 UTC, then a single-death rerun): the exchange measured, the resume closed, the compile cache found

On the three single-GPU L40S VMs pass 5 held (node-a 10.0.0.14 rank 0 and the head, node-b
10.0.0.9 rank 1, node-c 10.0.0.78 the spare's node), Qwen3-1.7B at TP1 x PP2, 8 sessions x 512
prompt, 1792 tokens; no new rental (the user's word of 19:52Z covered the day's dry run; the
watchdogs re-armed to 23:37Z). Trees: `0cbd1f4` (the decode-shaped resume, the per-position
digests, the packed layout on model runner v2's rule) with `5854fae` (the resume's device copy,
the knobs carried to the spare's launch environment) synced to the nodes at 21:08Z before any
arm. Driver `battery-dry3.sh` (scratch): the throughput arms first, then the fault arms on
whichever route measured.

## The exchange (steps 10–200, same seeded prompts, tok/s per trial)

| arm | route | depth | trial 1 | trial 2 | ops per boundary per side | host waits |
|---|---|---|---|---|---|---|
| stock | vLLM's own | pp | 1196 | | | 0 |
| adapter | header path | 1 | 850 | 877 | 4 (a header, three tensors), 2 device-to-host reads on the receiver | 1 |
| adapter | packed | 1 | 936 | 943 | 1 | 1 (a true wait on the previous transfer) |
| adapter | packed | 2 | 1189 | 1204 | 1 | 1 (on a finished transfer) |
| mirror off | packed | 2 | 1206 | | 1 | 1 |

Packed at depth 2 (= the pipeline size) reads at stock on this shape (within trial noise; 0.6 %
under on trial 1, 0.6 % over on trial 2), with the mirror's cost inside the overlap (1204 on
against 1206 off). The decision on the numbers: the depth is derived from the engine's pipeline
size and the packed route replaces the header path, the knobs and the route switch going with
that landing. The header path's probe on node-b had logged "this stage names the packed layout"
on its first steps before the route was turned on: the v2 rule held on the GPU first.

## The fault arms on packed depth 2 (the spare's launch environment carrying the same knobs)

| trial | resumed at | coverage | re-prefilled | resumed step | sessions identical | detection | kill→resume | gap (instrument off it) | first differing position vs the control |
|---|---|---|---|---|---|---|---|---|---|
| node_fault 1 | 1526 | 1.0 | 0 | decode on both ranks | 8/8 | 11.4 ms | 0.247 s | 0.612 s | rank 0 none; rank 1 from 1526–1541 (the resume on) |
| node_fault 2 | 1377 | 1.0 | 0 | decode on both ranks | 8/8, request 4 kept | 11.1 ms | 0.545 s | 0.668 s | rank 0 none; rank 1 from 1377–1404 |
| single_death 1 | 1680 | 1.0 | 0 | decode on both ranks | 4/8, diverged at generated 3, 11, 15, 220, before the kill | 6.7 ms | 0.545 s | 0.672 s | rank 1 from 0–69, nearly every prompt position; rank 0 from 515–732 |

The resumed step's seam is closed: `is_prefilling` false on both ranks on the first batch after
the resume, request 4 kept in both node-fault trials. The instrument (`rows_checked` 20 steps
after the first step back, 1.57 s) is off the gap: the client-visible gap reads 0.61–0.67 s.

**Control against control.** Three adapter controls (node_fault 1 and 2, single_death 1) and
pass 5's three controls match each other at every token; the two pass-6 node-fault controls
match at every row of every request on both ranks (`first_differing_positions`: comparable, none).
A launch of the serving engine is row-exact against another on the same cache.

**The spare's rows (node-fault trials).** Rank 1's rows differ from the control's from the
resumed position on, with identical inputs (rank 0's rows match everywhere): the spare on
node-c had compiled the backbone on its own node. The same AOT artifact key on both nodes
(`torch_compile_cache/torch_aot_compile/679d0922…`, key `760834052d`), but Inductor's autotune
results inside it differ: `inductor_cache/j6/bb85e0…best_config` md5 3ae869f0 on node-b against
9cd6405b on node-c, the aotautograd entries with them. The spare ran the same fused kernels
with other tile configurations and rounded differently from the rank it replaced. At 22:14:52Z
node-b's cache was copied onto node-c (node-c's own kept aside; the best_config set then hashing
identically on both).

**The single-death run.** Its rank 1 rounded the prefill differently from the controls, on the
same node and artifact, before any kill. The cache says why: exactly one file under the
adapter's artifact was written in that launch's window, `inductor_cache/jx/b4bbf2e7…best_config`
at 22:22:20.735 (the launch's log line at 22:22:20): that launch autotuned one kernel fresh, by
timing, and landed in another rounding family; the stock run, with its own key (`49d4cf3c60`,
artifact `99bad6…` saved at 21:20:27) sits where its own autotune put it (its tokens diverge
from the adapter controls at the same sessions and positions as that run, 15/150/3/11). From
that write on, every launch loads the new best_config and lands in the new family.

One cause for both: a compile cache autotuned on a node by itself. Decision 0029 (the peer's):
the compile cache is part of a position, staged by the deploy to every node that may host one,
and every member reports the (key, digest) it loaded at registration, the coordinator holding
the engine's members to the first one's. Here: `everserve.compile_cache.digest` (core, with a
fixture), `nodes.sh cache_sync` (the serving node's cache onto the spare's node, both digests
recorded) called by the battery before every fault trial, `cache_digest.py` for the record.

## Rerun (22:30–22:46Z): two single-death trials, the spare on the staged cache

| trial | resumed at | coverage | sessions identical | first differing position vs the control | detection | kill→resume | gap |
|---|---|---|---|---|---|---|---|
| 1 | 1680 | 1.0 | 8/8 | none on both ranks | 8.3 ms | 0.545 s | 0.67 s |
| 2 | 1811 (1812 committed, capped) | 1.0 | 8/8 | none on both ranks | 8.3 ms | 0.274 s | 0.40 s |

With the serving node's cache staged onto node-c before each launch (`nodes.sh cache_sync`,
node-c's own copies kept aside), the spare computes row-exact against the control from the
resumed position to the end, on both trials. Both rerun controls match the pass-6 node-fault
controls at every row: the best_config written at 22:22 did not move the serving family.

What that file was: one Inductor reduction kernel (size hints x 65536, r0 128, the shape of
Qwen3's q/k norm; `inductor_cache/jx/…best_config`), rewritten by three launches in the window
(22:22:20 node-b, 22:40:10 node-c's spare, 22:40:36 node-b) with the same winner each time the
content could be read (XBLOCK 4, R0_BLOCK 128, 4 warps); node-c's own cold autotune of the same
kernel had chosen XBLOCK 64, R0_BLOCK 64, 16 warps, the spare family of the node-fault trials.
At the pin a kernel consults its persisted best_config only when the winner is among its
candidate configurations (`triton_heuristics.py` 292–312, `AutotuneCache.read_best`), and
benchmarks otherwise; the 22:22 launch's own content was overwritten before it could be read.
The setting that removes timing from reductions is Inductor's deterministic mode
(`torch._inductor.config.deterministic`: no tuning of RBLOCK or warps for reductions, one
configuration by rule), carried as `{"inductor_compile_config": {"deterministic": true}}` in
the engine's compilation config (decision 0029's amendment, the peer's `63b5d0b`), measured on
pass 7's throughput arms beside the staged cache.

The drill's `re_prefilled` read every request on trial 2 because it compared the first batch's
computed count (1811) against the committed count (1812) rather than the count resumed from
(`resume_point` caps at one below the token count by design); fixed to compare against
`resume_from`. No row or token effect.

## Cost

Three VMs from 19:53Z (pass 5) held through pass 6 and the rerun; released by hand at 22:48Z
(node-a and node-b deleted, node-c deleting): 2.9 h x 3 x $1.55/h ≈ $13.6 for passes 5 and 6
and the rerun together, on the user's word of 19:52Z; the day's dry runs ≈ $34 in all.
