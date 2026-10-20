# Result: rank 1's node crashed twice under load, each loss taken by a spare (proof-22)

**Verdict: pass.** On 2026-10-15, `bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-how crash`
ran on images built from main at `e56d9bd`. Those images carry:
- 6220b01: a node whose worker died learns from the placement that its rank left;
- 96814e7: a commit goes to the engine's audience, so a node that lost every member hears it.

Rank 1's serving process was killed with SIGKILL twice, with no notice. The second time it was
the spare that had taken rank 1 at the first. Each time the coordinator saw the departure within
about 2 ms and committed a REPLACE through a covered set, and all 16 streams finished with every
token. Every prediction of proof-20's [pre-registration](../2026-10-14-dynamo-proof-crash/PREREGISTRATION.md)
held.

The run's output is `run/`. Each kill, with its moment on the pod's clock and the process it
killed, is `run/kill-N.txt`.

## The predictions

| # | Prediction | Read | Verdict |
|---|---|---|---|
| 1 | Both losses REPLACE rank 1 through a covered set, at epochs 1 and 2, every stream finishing | epoch 1 through `0-R1@…ec8qw…:0`, epoch 2 through `0-R1@…wrdz5…:0`; 8 of 8 streams each, 6000 tokens, no error | **pass** |
| 2 | Detection, kill to the coordinator's departure line, at most 0.5 s | loss 1: 00:11:42.591 → 00:11:42.593; loss 2: 00:12:59.035 → 00:12:59.037; about 2 ms each | **pass** |
| 3 | Loss 1's worst gap at most 1.2 s | 0.597–0.602 s, at the change; committed 276 ms after the report | **pass** |
| 4 | Loss 2's worst gap at most 2.0 s | 1.366–1.376 s, at the change; committed 1,065 ms after the report | **pass** |
| 5 | Covered True before each loss | "Covered True (EveryLossCovered): … the head's node holds [0], whose loss restarts the engine", before both | **pass** |
| 6 | The crashed pod's node ends, its pod starts again in place, asks and stands as a spare | the pod crashed in loss 1 stood as rank 1's spare again, covered under plan 4 at epoch 1, 33 s after loss 1's commit; it took rank 1 at loss 2 | **pass** |

## What the changes cost

| | Loss 1 | Loss 2 |
|---|---|---|
| process killed | `VLLM::Worker_PP1` (pid 176), the original serving node's worker | `VLLM::Worker_PP1` (pid 198), the spare that took rank 1 |
| reported, from the departure | 00:11:42.593 | 00:12:59.037 |
| rank 0 acknowledged | +15.8 ms | +12.1 ms |
| rank 1 (the spare) acknowledged and committed | +276 ms | +1,065 ms |
| restore onto the spare's node | 300,984 rows | 1,443,000 rows |

As in proof-18 and proof-19, the change's time is the restore of rank 1's rows onto the spare's
node. The restore sends every row the rank has written, so it grows with the run toward the
cache's capacity (proof-18's record, "Open trades for Cort").

All three node daemons logged "membership committed at epoch 1" and "at epoch 2". That includes
the daemon on the node crashed in loss 1, whose daemon proof-21 left at epoch 0.

## The way here

| Run | Commit | What it found |
|---|---|---|
| proof-20 | 5776450 | loss 1 passed. The crashed node never learned its rank had left, so its pod never stood as a spare again. Fixed by 6220b01. |
| proof-21a–c | d81cd20 | a: no L40S capacity for 30 minutes. b: the harness read `--async`'s answer as JSON (c28dca5's slip). c: the node fix held; loss 2's change was refused by the daemon of the node crashed in loss 1, still at epoch 0. Fixed by 96814e7. |
| proof-22 | e56d9bd | pass |

## A recording for the harness

`node-group create --async` answered `opmk8snodegroup-e00tp71bg0b7sbja7c` (`run/run.log`). That is
the recording the harness's test fakes the CLI from:
`bench/harness/tests/fixtures/nebius-node-group-create-async.txt`.

## What this does not show

- The head's own loss under load.
- A production-size model, and more than one node per stage.
- CoreWeave.
