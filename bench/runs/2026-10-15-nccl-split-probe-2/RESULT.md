# Result: a split reports success without its peer, so (C) is not built (NCCL probe 4)

**(C) is not built: #2 and #3 are falsified.** On 2026-10-15, on NCCL 2.29.7 in the engine's image
(`dev-51e2763`), on two L40S nodes, the probe ran beside the serving engine. Read against the
[pre-registration](PREREGISTRATION.md).

## The answers

| # | Read | Verdict |
|---|---|---|
| 1 | Both members' splits completed in 0.01 s (9 and 10 polls) with a child handle. Every child call returned 0 with no poll and the right sums, eager and graphed. | **holds** |
| 2 | Rank 1 formed the parent and left. Rank 0's split then read **success** in 0.58 s, with a child handle. Every call on that child returned 4 (`ncclInvalidArgument`). The aborts returned 0, the child's in 0.0 s and the parent's in 0.92 s. | **falsified**: the split completed without its peer |
| 3 | The child against `vllm` on the same pods, µs per all-reduce: eager 168.7 against 219.7 at 896 (−23 %) and 272.5 against 253.6 at 7,168 (+7.5 %); replay 156.4 against 156.7 at 896 (−0.2 %) and 240.9 against 227.1 at 7,168 (+6.1 %) | **falsified** at 7,168 |

a1 and a2 answered as in probes 2 and 3: still in progress at 10 s, aborts in 0.0 s and 0.65 s,
and the device ran a kernel after.

#2 decides it. A split whose member has gone reports a formed child that cannot serve. Under (C),
`form_planned` would report GroupFormed for a dead group. That is worse than a formation that
times out, because the cover would claim a set it does not hold.

## What the timings can and cannot say

The two blocking communicators on the same pods, `vllm` and the split's child, differ by 23 % per
eager call at 896. They ran minutes apart, beside a serving engine. So this method's eager noise is
above a 5 % bar, and #3's eager figures do not say whether the child runs like a fresh communicator.
The replays look steadier: 0.2 % apart at 896, 6 % at 7,168.

That bears on probe 3's figures too. Its 2.1× eager cost for (A) at batch 1 is within reach of
this noise. Its replay figure, 28 % slower at 896, is the one that stands well outside what the
replays here vary by. The decisions are not re-read: (A) was put out by its pre-registered rule. A
cost that is to decide anything again needs a method that can see 5 %: arms interleaved in rounds
within one process pair, the engine idle, thousands of calls.

## What comes next

With (A) and (C) out, the table's remaining way is to gate the blocking join on the control
plane's links. It shrinks the window in which a member's death hangs the others to the join itself,
but does not close it. The peer and I decide how that window is bounded, with this record.

The record is `run/`, where `nccl-probe/` holds each arm's answer and NCCL's log.
