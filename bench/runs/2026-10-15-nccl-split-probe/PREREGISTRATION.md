# Pre-registration: a blocking child split from a nonblocking parent, and what a poll costs (NCCL probe 3)

Written before the run. The run is `bench/harness/k8s/dynamo_proof.sh --losses 0 --nccl-probe`
on images `dev-51e2763`: 3 × L40S in eu-north1 or the capacity fallback, a 1.5 h watchdog, and a
cap of about $4. The probe ships to the pods at run time from `sg2-python`.

## Why

Probe 2 ([RESULT](../2026-10-15-nccl-join-probe/RESULT.md)) answered that a nonblocking join is
bounded, by a deadline and an abort, but its communicator serves only through calls that poll on
`ncclInProgress`, every call. vLLM's `PyNcclCommunicator` raises on that code. Two ways remain to
bound the join of a cover group, which becomes the serving group after a change:
- **(C):** form the parent nonblocking, then `ncclCommSplit` it into a child configured
  `blocking = 1`. The split is polled on the nonblocking parent against a deadline. If the child
  is blocking, its calls are plain and vLLM's path is unchanged.
- **(A):** keep the nonblocking communicator and give the cover group's calls our own entry that
  polls. That puts a poll on every host-issued call for the engine's life after the first change.

## The arms

Each runs on the same two pods (the head's and a node pod, on two nodes), under NCCL's log
(`INIT,BOOTSTRAP,NET`).

| Arm | What it does |
|---|---|
| `vllm` | the control, as probe 2 |
| `blocking` | as probe 2, plus `cost` |
| `nonblocking-polled` | as probe 2, plus `cost` with its polls |
| `split` | the parent nonblocking, polled to formed; `ncclCommSplit(color 0, key = rank)` with a child configured `blocking = 1`, polled to a 20 s deadline; then the child's eager and graphed all-reduces **with no polling** (a 7 is a failure), plus `cost`; then both aborted |
| `split-missing` | both form the parent. Rank 1 leaves without splitting. Rank 0's split is polled to its 20 s deadline, then the child (if it has a handle) and the parent are aborted, and the device is checked |

`cost` gives microseconds per all-reduce at decode's sizes (896 and 896 × 8, bf16) in two ways:
200 calls issued eagerly, and a graph of 50 calls replayed 20 times.

## Expected

| # | Expected | Falsified by |
|---|---|---|
| 1 | `split`: both members' splits end in `ncclSuccess` within the deadline. The child's every call returns 0, eager and graphed, with the right sums, without a poll. | any 7 or other code from the child, wrong sums, or a split not done in 20 s |
| 2 | `split-missing`: rank 0's split stays in progress to the deadline. Its aborts return 0 within 2 s, and its device runs a kernel after. | the split returning success or an error before the deadline (that is recorded, not a failure of (C)), an abort that hangs (timeout's exit 124), or a device that no longer runs |
| 3 | `cost`: each arm's replays cost the same within 10 %, since a replay issues nothing from the host. `nonblocking-polled`'s eager calls cost more than `blocking`'s. | replays differing by more than 10 % |

## What each outcome decides

| Outcome | Decision |
|---|---|
| #1 and #2 hold | **(C) is built.** `form_planned` joins nonblocking with a deadline and an abort, then splits a blocking child under the same deadline. The child is the cover group vLLM is handed. A member that never comes, or dies before or within the split, ends the formation at its deadline with GroupFailed, and the process serves on. |
| #1 holds, #2's abort hangs | (C) is not bounded inside the split. Recorded, and the peer and I decide between (A) and a bounded split. |
| #1 fails: the child answers 7, or it is not blocking | the child takes the parent's blocking mode. (C) is out. (A) only if its eager cost at decode sizes is in the noise against `blocking`, and that figure goes in the record. |

The decision is the peer's and mine, from this table, not after reading the results.
