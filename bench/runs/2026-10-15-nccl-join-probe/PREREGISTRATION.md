# Pre-registration: how a cover group's join is bounded, three arms (NCCL probe 2)

Written before the run. The run is `bench/harness/k8s/dynamo_proof.sh --losses 0 --nccl-probe`
on images `dev-51e2763`, proof-23's. That is 3 × L40S in eu-north1 or the capacity fallback, a
1.5 h watchdog and a cap of about $4. No loss is taken: the instance serves, and the probe runs
beside it.

## Why

The shared-groups pair joins a cover group by the blocking `ncclCommInitRank`, from an id the
group's server made and the links carried. Proof-23 showed that join works. If a member dies
after the id goes out, the other members' formation threads hang for good. The pair does not
land until the join is bounded.

Proof-23's probe answered a1 and a2: a nonblocking join is aborted at a deadline in under 0.7 s,
and its device is left usable. Its question b got no answer. Neither member of a nonblocking
two-member group formed in 120 s, while the engine's blocking groups formed between the same
pods in about 0.2 s. That probe had no control arm and NCCL's logging was off
([RESULT](../2026-10-15-dynamo-proof-shared-groups/RESULT.md)).

## What changed in the probe

- **`join --how`** selects one of three arms, each run on the same two pods (the head's and a
  node pod, on two nodes), one after another:
  - `vllm`: vLLM's `PyNcclCommunicator.from_unique_id_bytes`, the blocking `ncclCommInitRank`
    the pair uses. This is the control.
  - `blocking`: `ncclCommInitRankConfig` with the same configuration prefix and `blocking = 1`.
    It isolates the configuration's path.
  - `nonblocking`: the same with `blocking = 0`. This is the question.
- **The configuration is kept alive** for the probe object's life. In proof-23 it was a local
  freed when the call returned, while a nonblocking initialisation goes on reading on NCCL's own
  thread. That could be the fault on its own.
- **NCCL logs** at `NCCL_DEBUG=INFO`, subsystems `INIT,BOOTSTRAP,NET`, to each process's stderr
  (`nccl-probe/*.err`).
- **`--losses 0` takes no loss.** The loop no longer relies on `seq 1 0`, which counts down on
  BSD.

Every probe process is bounded by `timeout 180`, so a blocking arm that never forms ends with
exit 124.

## Expected

| Arm | Expected | Read from |
|---|---|---|
| a1, a2 | as proof-23: in progress at 10 s; the abort returns 0 within 2 s; the device runs a kernel after | `missing-*.json` |
| `vllm` | both members form within 30 s; every all-reduce returns 0 with the right sums, eager and graphed | `join-vllm-{0,1}.json` |
| `blocking` | as `vllm` | `join-blocking-{0,1}.json` |
| `nonblocking` | both members reach `ncclSuccess` within 30 s; every all-reduce returns 0 (never `ncclInProgress`) with the right sums, eager and graphed | `join-nonblocking-{0,1}.json` |

## What each outcome decides

| `vllm` | `blocking` | `nonblocking` | Reading | Decision |
|---|---|---|---|---|
| does not form | any | any | the probe's harness is wrong (the id's passing, the pods, the exec) | nothing else is read; the harness is fixed and the run repeated |
| forms | does not form | any | the configuration's path is wrong | the configuration is built from the image's own `nccl.h` layout, and the run repeated |
| forms | forms | does not form | nonblocking initialisation does not complete on this image; NCCL's log says where it waits | nonblocking is not the bound. The join is gated on the links instead: every member confirms it is alive before any joins, and a member lost inside that window ends its process, a bounded exit its node restarts |
| forms | forms | forms, every collective 0 | proof-23's b was the probe's fault | `form_planned` joins nonblocking with a deadline and aborts at it; a member whose peer never comes reports GroupFailed and serves on |
| forms | forms | forms, a collective returns 7 | a nonblocking communicator answers `ncclInProgress` to vLLM's calls | the options go to the peer with the record: forcing connections at formation, or the links-gated blocking join |

The decision is the peer's and mine together. It is made from this table, not after reading the
results.

## Addendum, written after the three arms and before the fourth (02:35Z)

The repeat with the probe's id fixed answered the table's last row. All three arms formed. The
`vllm` and `blocking` arms' collectives all returned 0 with the right sums, eager and graphed. The
`nonblocking` arm's first all-reduce returned 7 (`ncclInProgress`), and every later call returned 4
(`ncclInvalidArgument`): the probe issued the next call without polling, against NCCL's contract
for a nonblocking communicator.

A fourth arm, `nonblocking-polled`, asks the contract's own way, on the same cluster before its
take-down. After a call answered 7, poll `ncclCommGetAsyncError` until the state leaves
`ncclInProgress`, then issue the next call.

**Expected:** every call ends in 0, eager and graphed, with the right sums. Polling within a CUDA
graph's capture is a host call and is allowed.

**What it decides** (with the peer, from this, not after reading):
- **All 0 with the right sums, eager and graphed:** a cover group may be nonblocking, with the
  join bounded by a deadline and an abort. Its calls must poll on 7, so the group's communicator
  cannot be handed to vLLM's `PyNcclCommunicator` as is (its `NCCL_CHECK` raises on 7). The
  options to weigh: our own all-reduce entry that polls, against forming nonblocking and then
  splitting a blocking child.
- **Any call never leaves 7, a 4 after polling, or a capture error:** a nonblocking communicator is
  not usable for serving on this image. The join is bounded another way, its options to the peer.
