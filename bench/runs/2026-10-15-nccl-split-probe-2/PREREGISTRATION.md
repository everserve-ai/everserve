# Pre-registration: a blocking child split from a nonblocking parent, its completion read right (NCCL probe 4)

Written before the run. The run is `bench/harness/k8s/dynamo_proof.sh --losses 0 --nccl-probe`
on images `dev-51e2763`, with three arms only: `vllm`, `split` and `split-missing`. That is
3 × L40S in eu-north1 or the capacity fallback, a 1.5 h watchdog, and a cap of about $4.

## Why

Probe 3 ([RESULT](../2026-10-15-nccl-split-probe/RESULT.md)) put (A) out. A nonblocking
communicator costs 2.1× per eager call at decode's batch 1, and 19–28 % per graph replay, where the
host issues nothing. So **the serving communicator must be blocking by construction**, whichever
join is chosen.

(C) is the remaining way to bound the join: form the parent nonblocking (bounded and abortable),
then split a blocking child for serving. Probe 3 left it unanswered. On the nonblocking parent,
`ncclCommSplit` returned `ncclSuccess` at once and filled in the child's handle later, and the
probe took the return as the split's end.

## What changed in the probe

- **The split's completion.** The split is done only when the child has a handle and the child's
  own state reads `ncclSuccess`. Until the handle comes, the parent's state is read, and an error
  there ends the wait. All of it runs against a 20 s deadline.
- **A test without a GPU** (`bench/harness/tests/test_nccl_probe.py`). NCCL is stubbed as probe 3
  found it: the split returns 0 with no child, and the handle comes later or never. The three
  tests pass on this probe and fail on probe 3's.
- **`vllm` is timed** (`cost`), as the child's baseline on the same pods.

## Expected

| # | Expected | Falsified by |
|---|---|---|
| 1 | `split`: both members' splits complete within 20 s. The child's every call returns 0, eager and graphed, with the right sums and no poll. | a split not done in 20 s, any code but 0 from the child, wrong sums |
| 2 | `split-missing`: rank 0's split is still in progress at the deadline. Its aborts (the child's, if it has a handle, and the parent's) return 0 within 2 s, and the device runs a kernel after. | the split completing without its peer, an abort that hangs (exit 124), the device not running |
| 3 | **(C)'s bar:** the child's cost is within 5 % of `vllm`'s on the same pods, eager and replayed, at 896 and 7,168. | more than 5 % on any of the four |

## What each outcome decides

| Outcome | Decision |
|---|---|
| #1, #2 and #3 hold | **(C) is built.** `form_planned` joins nonblocking under a deadline, splits a blocking child under the same deadline, and hands vLLM the child. A member that never comes, or dies before the split completes, ends the formation at its deadline: both communicators are aborted, GroupFailed is reported, and the process serves on. The parent is kept for the group's life, or destroyed once the child is formed, as the code's review decides. |
| #1 and #2 hold, #3 fails | The child inherits the parent's slowness, so (C) is not built. The record goes to the peer, and the way left to bound the join is gating it on the control plane's links. |
| #1 fails | (C) does not give a usable blocking child on this image. The same as the row above. |
| #2's abort hangs | (C) is not bounded inside the split. Not built. |

The decision is the peer's and mine, from this table, not after reading the results.
