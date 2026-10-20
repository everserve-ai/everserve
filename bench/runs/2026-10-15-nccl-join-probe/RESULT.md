# Result: a nonblocking join forms and is bounded, but serves only through calls that poll (NCCL probe 2)

**Answered: the table's last row.** On 2026-10-15, on NCCL 2.29.7 (`22907`) in the engine's image
(`dev-51e2763`), two processes on two L40S nodes, beside the serving engine:

| Arm | Formed | Collectives, eager and graphed |
|---|---|---|
| `vllm`, the control (vLLM's blocking `ncclCommInitRank`) | yes, 0.23 s after its peer joined | every call 0, sums right |
| `blocking` (`ncclCommInitRankConfig`, `blocking = 1`) | yes, 0.16 s | every call 0, sums right |
| `nonblocking` (`blocking = 0`) | yes, 0.16 s, state `ncclSuccess` | the first call 7 (`ncclInProgress`), then every call 4 (`ncclInvalidArgument`): the probe issued each next call without polling |
| `nonblocking-polled` (pre-registered in the addendum) | yes, 0.17 s | every call 0, sums right, **but every call answered 7 first and needed a poll**: 5 of 5 eager, 10 of 10 inside the graph's capture |
| a1: the server waits for a member that never comes | | in progress at 10 s; abort 0 in 0.001 s; the device ran a kernel after |
| a2: a member whose server exited | | in progress at 10 s; abort 0 in 0.65 s; the device ran a kernel after |

So a nonblocking join is bounded, by a deadline and an abort, and its communicator serves. It
serves only through calls that poll `ncclCommGetAsyncError` after every 7. vLLM's
`PyNcclCommunicator` raises on any code but 0, so a nonblocking cover group cannot be handed to
vLLM as it is. The design is the peer's and mine to choose (the addendum's two options).

The record is `run/`, where `nccl-probe-fixed/` holds the answers above with NCCL's log of each
process.

## How the run went

1. **The driver's three arms all failed the same way.** Every join was refused with "didn't
   receive a valid magic", even rank 0's own connection to its own root (`run/nccl-probe/`).
   Proof-23's b had the same cause.
2. **The cause was the probe's id.** The id's first eight bytes were pointer-like
   (`0x00007c6c371b7330`), where NCCL writes a random magic (`getRandomData`). The probe read the
   id through a temporary, `string_at(addressof(lib.ncclGetUniqueId()), 128)`. The struct was
   freed as soon as `addressof` returned, and the allocator's own pointer took the magic's place.
3. **The comparison on the head's pod, in one process:** a held struct gave a random magic and the
   temporary a pointer-like one. The shipped `mint_group_id()` holds its struct, which is why the
   engine's cover groups always formed. A test now holds it to that
   (`adapters/vllm/tests/test_group_id.py`).
4. **The table's first row:** the harness was wrong, so it was fixed and repeated. To spare a
   second rental, the driver was stopped before its take-down, and the arms were repeated on the
   same cluster by hand (`run/nccl-probe-fixed/run.log`). The logs were collected and the cluster
   taken down by hand.

A first attempt at the fix did not apply (an edit that failed silently). Its repeat ran the old
probe and failed as before, and was stopped. The staged copy is now checked before each run.

## What this does not show

- What the poll costs per call, against the blocking arm, at vLLM's sizes.
- A member lost after formation, mid-collective, on a nonblocking communicator.
- More than two members, and any other NCCL.
