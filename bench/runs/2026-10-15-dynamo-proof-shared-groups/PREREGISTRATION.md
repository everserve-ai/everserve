# Pre-registration: shared groups on GPUs, a deletion then a crash, and NCCL's nonblocking join (proof-23)

Written before the run. The run is `bench/harness/k8s/dynamo_proof.sh --losses 2
--loss-how delete,crash --nccl-probe`:
- images from `sg2-python`: main at `3de89a0` with the shared-groups pair (`8b3c0ad`, `ac27bd4`,
  `72afaf8`), which is not on main and lands only if this run passes;
- 3 × L40S in eu-north1, or the capacity fallback; a 1.5 h watchdog; Qwen2.5-0.5B at TP 1 × PP 2.

## What differs from proof-22

- **A group is (scope, spans), formed once for every set that needs it.** Its server makes the
  NCCL id and reports it on its link, and the coordinator carries it to the other members'
  daemons. No member opens a port for a cover group, and no `TCPStore` is made for one.
- **Loss 1 is a deletion (noticed), loss 2 a crash (SIGKILL, no notice).** Proof-18 timed the
  first and proof-22 the second, both on the old rendezvous.

On this shape (one spare, PP 2) each plan's one set needs one group, so no group is shared.
Sharing is exercised by the unit tests. This run shows the id carried over the links, end to
end, on GPUs.

## Predictions

| # | Prediction | Read from | Falsified by |
|---|---|---|---|
| 1 | Both losses commit a REPLACE of rank 1 through a covered set, at epochs 1 and 2, and all 16 streams finish with every token. | the verdict, `load-N.json` | any stream error, or a change refused or not at the loss's epoch |
| 2 | Loss 1 (deletion) reaches the coordinator as a notice, and its worst gap is at most 1.0 s (proof-18: 0.64 s). | the coordinator's log, `load-1.json` | no notice, or over 1.0 s |
| 3 | Loss 2 (crash): detection, kill to the departure line, is at most 0.5 s (proof-22: about 2 ms), and the worst gap is at most 2.0 s (proof-22: 1.37 s). | `kill-2.txt`, the coordinator's log, `load-2.json` | over either bound |
| 4 | Before each loss, the engine reads Covered True, reason EveryLossCovered. | the driver's `the engine:` line | Covered False once rank 1's set formed |
| 5 | Every plan's groups form over the links: each covering plan reads "1 of 1 groups formed", and the event record holds no GroupFailed. | the driver's `cover:` lines; `engines-proof-events*.json` | any GroupFailed, or a set never formed within the driver's 600 s wait |
| 6 | The pod Grove makes in place of loss 1's stands as rank 1's spare and covers loss 2. The node crashed in loss 2 starts again in place and stands as a spare. | `serving-pods.txt`, its log, the cover line of loss 2 | either pod serving a rank, or never standing |

Mapping to proof-22: #1, #4 and #6 as there. #2 and #3 are proof-18's and proof-22's bounds for
their kinds of loss. #5 is the pair's own.

If #5 fails, the pair does not land. The record then says where the id stopped: the server's
report, the coordinator's relay, or the member's wait.

## The NCCL probe: three questions, asked after the losses on the same cluster

The pair's join is blocking. If a member dies between the id's relay and its join, the other
members' joining thread hangs, and a blocking `ncclCommInitRank` gives no handle to abort. The
probe (`bench/harness/k8s/nccl_probe.py`) asks whether NCCL's nonblocking initialisation bounds
that, on the engine's image, beside the serving engine.

The configuration is passed as the prefix NCCL has kept since 2.14 (size, magic, version 2.14.0,
`blocking = 0`). Every newer field takes NCCL's default.

| Q | Question | Expected | Read from |
|---|---|---|---|
| a1 | A group of two whose second member never comes (the server waits): is the state still `ncclInProgress` at 10 s, does `ncclCommAbort` return 0 within 2 s, and does the device still run a kernel? | yes, yes, yes | `nccl-probe/missing-server.json` |
| a2 | A member whose server made the id and exited: does the state end in an error or stay in progress, does the abort return 0 within 2 s, and does the device still run a kernel? | either state, then yes, yes | `nccl-probe/missing-member.json` |
| b | A group of two formed nonblocking across two nodes: does it reach `ncclSuccess` within 30 s, and does every all-reduce at vLLM's sizes, eager and then captured in a CUDA graph and replayed three times, return `ncclSuccess` (0, never `ncclInProgress`) with the right sums? | yes, yes | `nccl-probe/join-0.json`, `join-1.json` |

Q b's second half is the one the peer's two questions did not name. vLLM's `NCCL_CHECK` raises on
any code but 0. A nonblocking communicator may return `ncclInProgress` from a collective (NCCL
connects lazily at the first use of a path), so a group that forms but answers 7 under vLLM's
calls is not usable as vLLM's communicator.

What each answer decides:
- **a1, a2 and b all as expected:** `form_planned` joins nonblocking with a deadline and aborts at
  it. A member whose peer never comes reports GroupFailed and serves on. The collectives stay
  vLLM's.
- **An abort that does not return** (the probe's process ends at `timeout`'s 180 s, exit 124):
  nonblocking init does not bound the join on this image. The join is then gated on the links
  instead: every member confirms it is alive before any joins. A member lost inside that window
  ends its process, a bounded exit, and its node stands it again.
- **b returns `ncclInProgress` from any collective:** a nonblocking communicator cannot be vLLM's
  as it is. The options go to the peer with the record. One is forcing the connections at
  formation (NCCL's runtime connect off for cover groups). Another is the links-gated blocking
  join above.
