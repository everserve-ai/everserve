# Pre-registration: rank 1's node crashed twice under load (proof-20)

Written before the run. The run is `bench/harness/k8s/dynamo_proof.sh --losses 2
--loss-how crash`:
- images from main after `eb9a2c0`, the first proof with `22b1b2b` (Covered leaves out the head's
  node);
- 3 × L40S in eu-north1, a 1.5 h watchdog, Qwen2.5-0.5B at TP 1 × PP 2.

## What differs from proof-18 and proof-19

Each loss is a crash, not a deletion:
- Every process of the pod that holds rank 1 and serves a rank is killed with SIGKILL from inside
  the pod: vLLM's workers, and for loss 2 the spare that took rank 1, which serves in its own
  process.
- Nothing is noticed. The coordinator learns of the loss as the processes' departure ("departed
  outside any change: the rank is lost", 402503a), and starts the change then, not at a notice's
  deadline.

The killer prints its moment on the pod's clock (`run/kill-N.txt`). The coordinator's log stamps
the departure on the cluster's clock.

## Predictions

| # | Prediction | Read from | Falsified by |
|---|---|---|---|
| 1 | Both losses commit a REPLACE of rank 1 through a covered set, at epochs 1 and 2, and every stream finishes with all its tokens. | the verdict, `load-N.json` | any stream error, or a change refused or not at the loss's epoch |
| 2 | Detection, from the kill to the coordinator's departure line, is at most 0.5 s. A SIGKILLed process's daemon link reads EOF at once, and its daemon reports the departure up. | `kill-N.txt`; the coordinator's log | over 0.5 s on either loss |
| 3 | Loss 1's worst gap is at most 1.2 s: detection (≤ 0.5 s) + the change (proof-18/19: 0.36–0.37 s with a 420–447k-row restore) + the resume (about 0.3 s, the gap above the change in proof-18/19). | `load-1.json` | over 1.2 s |
| 4 | Loss 2's worst gap is at most 2.0 s: as #3 with proof-18/19's second change (1.13–1.14 s, the restore of rank 1's whole KV capacity, 1.57M rows). | `load-2.json` | over 2.0 s |
| 5 | Before each loss, the engine reads Covered True, reason EveryLossCovered, with the head's node named in the message. 22b1b2b makes that possible on this shape. | the driver's `the engine:` line; `engines-proof-loss-N.json` | Covered False once rank 1's set formed |
| 6 | The crashed pod's node process ends, and its pod is started again in place (a container restart, not a new pod). It asks again and stands as a spare. | `serving-pods.txt` restart counts; its log | the pod replaced by a new one, or the restarted container serving a rank |

What each would mean:
- #2 or #3 failing: detection by departure is slower than the notice path's change start. The
  record then says where the time went: the link's EOF, the daemon's report, or the
  coordinator's decision.
- #6: what the node does after its workers die is as built (`node_main`: an engine-down exit is
  3; a node whose workers all departed holds nothing). The record reads it rather than assumes it.
