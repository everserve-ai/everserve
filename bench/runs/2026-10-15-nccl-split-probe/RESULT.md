# Result: polling costs too much to serve through, and the split went unanswered (NCCL probe 3)

**(A) is out by its own rule. (C) is unanswered: the probe judged the split done before it was.**
On 2026-10-15, on NCCL 2.29.7 in the engine's image (`dev-51e2763`), on two L40S nodes, the probe
ran beside the serving engine. Read against the [pre-registration](PREREGISTRATION.md).

## The answers

**a1, a2, `vllm`, `blocking` and `nonblocking-polled`** answered as in probe 2:
- the aborts returned in 0.0 s and 0.64 s, and the device ran a kernel after;
- every call of the three joins ended in 0 with the right sums, eager and graphed;
- the polled arm needed a poll on every call.

**#3, cost, falsified.** Microseconds per all-reduce on rank 0 (rank 1 within 1 %):

| Decode size (bf16 elements) | Eager, blocking | Eager, polled | Replay, blocking | Replay, polled |
|---|---|---|---|---|
| 896 | 120.6 | 250.2 | 145.7 | 187.1 |
| 7,168 | 192.2 | 213.9 | 196.5 | 233.0 |

- **Eager:** the polled calls cost 2.1× at batch 1 and 1.11× at batch 8.
- **Replays:** these differ by 28 % and 19 %, though a replay issues nothing from the host. So the
  nonblocking communicator itself runs slower, not only its polls.
- **The decision:** the table's rule admitted (A) only if its eager cost at decode sizes was in the
  noise. It is not, so **(A) is out**. The replays add a warning against any nonblocking serving
  group.

**#1 and #2, the split: unanswered.**
- On the nonblocking parent, `ncclCommSplit` returned `ncclSuccess` at once, before the child
  existed. In `split-missing` it read rc 0, state 0, no child handle. NCCL completes the split
  asynchronously and fills the handle in later.
- The probe treated only a return of 7 as "in progress", so it took the 0 as done and issued the
  child's calls at once:
  - in `split-missing`, those calls went to a null child and returned 4;
  - in `split`, both members' last log line is the child's "Connected all rings", and both were
    ended by `timeout` at 180 s. Most likely the calls were issued on a child still being built.
- **One fact for (C) came out anyway:** with the split pending and its peer gone, aborting the
  parent returned 0 in 0.98 s.

**A later caveat on these timings.** Probe 4 ([RESULT](../2026-10-15-nccl-split-probe-2/RESULT.md))
timed two blocking communicators on the same pods 23 % apart per eager call at 896. So the eager
figures above are within reach of the method's own noise. The replay figure, 28 % at 896, stands
well outside what probe 4's replays varied by (0.2 %). (A) stays out by its pre-registered rule; a
cost meant to decide anything again needs a method that can see 5 %.

## What comes next

Probe 4 asks the split again with its completion read correctly: done only when the child has a
handle and the child's own state reads `ncclSuccess`, the parent's state read until then, all
against the 20 s deadline. The table and its decisions stand, with (A) now out.

The record is `run/`, where `nccl-probe/` holds each arm's answer and NCCL's log.
