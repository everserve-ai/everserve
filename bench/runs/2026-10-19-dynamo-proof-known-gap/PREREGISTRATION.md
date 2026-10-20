# Pre-registration: the worker-loss proof on the tree without the head's failover (proof-26)

Written before the run. Cort's word for it, 2026-10-19: "ok you have the go ahead for the proof
as soon as its ready".

The run is `bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-how delete,crash --platform
gpu-h100-sxm --preset 1gpu-16vcpu-200gb`:
- images built from the commit that adds this file, on main at `40f07499`: the head's failover
  removed (decision 0045), the spare's start-up replay cut to its core (decision 0044), the fix
  for a node new to the engine (`a791ffb3`), and the worker-loss end to end in CI (12 of 12 jobs
  green on PR #14); the record names the commit;
- 3 × H100 in eu-north1, a 1.5 h watchdog, Qwen2.5-0.5B at TP 1 × PP 2: the head (Dynamo's vLLM
  worker, node 0, holding rank 0), one node pod holding rank 1, one node pod standing as its spare.
  The shape of proofs 15 to 25.

**Cap.** About $4.45 per GPU-hour, three nodes, about 40 minutes: about $9. The 1.5 h watchdog
makes the worst case about $20. Building the images costs under $1.

## Why this run

The last worker-loss proof (proof-25, 2026-10-15) ran images `dev-9a40ce9`. Since then the tree
gained the spare's start-up replay and lost the head's failover. Two things on the worker-loss
path have never run on GPUs in this shape:
- **The start-up replay.** A spare no longer derives its KV cache configuration from its own
  worker. It asks the engine's core, through its daemon, for the calls the core sent its rank at
  its start, and runs them before it warms (`everserve_vllm.replay`). It ran on GPUs only under
  the removed head handoff.
- **The join's digest.** Every change's join offer names the digest of its core's calls, and a
  spare refuses an offer naming other calls or none (join version 8).

The tree also carries the fix for a node new to the engine (`a791ffb3`): a node whose daemon
first hears the engine after a commit is brought to its epoch, where it refused the next change
as stale and the engine went down. The CPU end to end (`test_worker_loss_e2e.py`) found it and
covers it. This run's shape does not exercise it, since Grove recreates a deleted or crashed pod
on the node it ran on, which has heard every commit; no prediction here reads it.

And the harness changed: the head node group, the head losses and the handoff reads are gone, and
the driver now waits for the coordinator's record to say the engine serves (`proof_verdict.py
serving`) after the head pod is ready, before any loss.

## Predictions

| # | Prediction | Read from | Falsified by |
|---|---|---|---|
| 1 | Both losses commit a REPLACE of rank 1 through a covered set, at epochs 1 and 2, and all 16 streams finish with every token. | the verdict, `load-N.json` | any stream error, or a change refused or not at the loss's epoch |
| 2 | Loss 1 (deletion) reaches the coordinator as a notice, and its worst gap is at most 1.0 s (proof-25: 0.67 s). | the coordinator's log, `load-1.json` | no notice, or over 1.0 s |
| 3 | Loss 2 (crash): detection, kill to the departure line, is at most 0.5 s (proof-25: about 1 ms), and the worst gap is at most 2.0 s (proof-25: 1.328 s). | `kill-2.txt`, the coordinator's log, `load-2.json` | over either bound |
| 4 | Before each loss, the engine reads Covered True, reason EveryLossCovered. | the driver's `the engine:` line | Covered False once rank 1's set formed |
| 5 | Every spare that stands is built from the replay: its log reads "rank 1's start-up calls read from the core" before its load, and the head's log reads "rank 1's replay: N calls written" for each, N being 3 or 4 (the layout, the memory profiling, the KV configuration, and the model length only if auto-fit changed it). No spare refuses a configuration that does not fit its GPU. | the spares' and the head's logs | a spare standing without the line, a replay refused, or a spare never standing |
| 6 | Every join is taken on its digest: each spare that takes rank 1 answers READY, and no log reads "built from other start-up calls". | the spares' logs; the verdict | any FAILED answer naming the start-up calls |
| 7 | The pod Grove makes in place of loss 1's stands as rank 1's spare and covers loss 2. The node crashed in loss 2 starts again in place and stands as a spare. | `serving-pods.txt`, its log, the cover line of loss 2 | either pod serving a rank, or never standing |
| 8 | The driver reads the engine as serving by its record before the first loss: its `serves:` line names every position held and a head registered. | `run.log` | the driver timing out on the record, or a loss taken before the line |
| 9 | Nothing is left behind: no ReleaseLeaked event and no stale exit (4) in the run. | `engines-proof-events*.json`; `serving-pods.txt` | any ReleaseLeaked, or a worker exiting 4 |

#1 to #4, #7 and #9 are proof-25's, with its bounds. #5 and #6 are the replay's first reads in
this shape. #8 is the harness's.

If #1, #5 or #6 fails, the record says where it stopped, the defect is fixed on CPU first, and
another run needs Cort's word. On green, these figures are the tree's current worker-loss numbers.
