# Result: rank 1's node crashed twice, after the node fix (proof-21, attempts a–c)

**Verdict: fail, on loss 2, with a control-plane defect found.** On 2026-10-14,
`bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-how crash` ran against proof-20's
[pre-registration](../2026-10-14-dynamo-proof-crash/PREREGISTRATION.md). It used images built
from main at `d81cd20`, which carries 6220b01: a node whose worker died learns from the
placement that its rank left.

That fix holds on GPUs. The pod crashed in loss 1 ended, started again, stood as rank 1's spare,
and was covered at epoch 1 (proof-20's prediction 6).

Loss 2 crashed the spare that had taken rank 1. Its change was refused at delivery: the daemon
on the node crashed in loss 1 was still at epoch 0. The change failed and the engine went down.

The run's output, from attempt c, is `run/`. Each loss's kill, with its moment on the pod's clock
and the processes it killed, is `run/kill-N.txt`.

## The attempts

| Attempt | What happened | Whose |
|---|---|---|
| 21a (21:33Z) | the 3 × L40S node group in eu-north1 never came up: the CLI's create waited 30 minutes with no reason, the L40S quota free | the cloud's capacity; the harness waited blind. Fixed by c28dca5: the node group is asked for without waiting and the run's own bound (15 min) names "no capacity for N × <shape> in <region>". |
| 21b (22:52Z) | stopped after 5 minutes: `node-group create --async` answers with the operation's id as text, and the harness parsed it as JSON | harness, c28dca5's. Fixed: the answer is read as text and kept in the run's log, so a test can fake it from a recording. |
| 21c (23:06Z) | the nodes came up; loss 1 passed; loss 2 failed on the stale epoch below | control plane |

Estimated spend for proof-21, at list prices (the console has the billed figures):
- the dev-d81cd20 build, about $0.2;
- 21a and 21b, control planes without GPU nodes, about $0.15;
- 21c, 3 L40S for about 40 minutes, about $3.1.

About $3.5 in all, within the $8 booked.

## The predictions (attempt c)

| # | Prediction | Read | Verdict |
|---|---|---|---|
| 1 | both losses REPLACE rank 1 through a covered set at epochs 1 and 2, every stream finishing | loss 1: epoch 1 REPLACE, 8/8 streams. Loss 2: the change refused, the engine down, 8 streams 503 | **fail** (loss 2) |
| 2 | detection at most 0.5 s | loss 1: kill 23:24:22.929, the coordinator's "departed outside any change" 23:24:22.933, about 4 ms. Loss 2: 23:25:39.316 → 23:25:39.319, about 3 ms | **pass** |
| 3 | loss 1's worst gap at most 1.2 s | 0.578 s; committed 276 ms after the departure | **pass** |
| 4 | loss 2's worst gap at most 2.0 s | not read: the change was refused | **not read** |
| 5 | Covered True before each loss | before loss 1: Covered True (EveryLossCovered); before loss 2: Covered True, the re-cover formed at 23:25:20 | **pass** |
| 6 | the crashed pod's node ends, its pod starts again, asks and stands as a spare | `s9wnw` restarted, registered as rank 1's spare at 23:25:20 at epoch 1, covered under plan 4 | **pass** |

## The finding: a daemon whose member was lost without notice stays at the old epoch

- **What the record shows.** At 23:25:39.331 the coordinator started epoch 2's REPLACE of rank 1.
  At .333 it recorded `NoDelivery`: "node computeinstance-e00wv9c04r8gtn44b0 did not deliver the
  change: stale epoch: change is 2, this node is at 0". The change failed, and the engine read
  `Down` (ChangeFailed). e00wv9 is the node crashed in loss 1, where `s9wnw` stood as the spare.
- **Cause, as read from the logs.** A change goes to the lost rank's survivors and spares.
  - e00wv9's daemon was neither in loss 1: its member was the one lost. It never logged
    "membership committed at epoch 1", while the head's node's daemon did at 23:24:23.210.
  - The spare that registered there later heard the engine at epoch 1 on its own link. The
    daemon's own committed epoch stayed 0, and the next change through that node was refused.
  - A deletion (proof-18, proof-19) never showed this, because the leaving node takes part in
    its change.
- **Owner.** The control plane: the node daemon's committed epoch per engine, or the
  coordinator's delivery set. It is the same pattern 6220b01 fixed in the adapter, one layer
  down: a node must learn what happened to its rank whether or not it was in the change.

## What this does not show

- Loss 2's gap without notice.
- The head's own loss.
- A production-size model, more than one node per stage, and CoreWeave.
