# Pre-registration: the worker-loss proof on the open-source tree (proof-27)

Written before the run. Cort's word for it, 2026-10-20: "lets rip through first two, then we
will do gpu test and call it".

The run is `bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-how delete,crash --platform
gpu-h100-sxm --preset 1gpu-16vcpu-200gb`, proof-26's shape:
- images built from the commit that adds this file, on branch `oss-polish` at `8efdb855` plus
  this record; the record names the image tag;
- 3 × H100 in eu-north1, a 1.5 h watchdog, Qwen2.5-0.5B at TP 1 × PP 2: the head (Dynamo's vLLM
  worker, node 0, holding rank 0), one node pod holding rank 1, one node pod standing as its spare.

**Cap.** About $4.45 per GPU-hour, three nodes, about 40 minutes: about $9. The 1.5 h watchdog
makes the worst case about $20. Building the images costs under $1.

## Why this run

Proof-26 (2026-10-19) ran images `dev-80aa21e`, the last commit before the open-source audit.
Since then the tree changed on the GPU path in ways CI exercises on CPU only:

- **The audit's merge (PR #15, `11a3595f`).** Every permanent patch the adapter puts on vLLM or
  torch is one entry of `everserve_vllm/patches.py`, installed from the registry: the four
  pipeline methods are wrapped once with the route and the failure report together, and the
  warm-up wraps compose by a marker. The SHRINK reference tier left the shipped `everserve`
  package for `everserve.reference`. A missing or stale kernel build is refused rather than
  degraded to. Nothing at plugin load changes a vLLM process without `EVERSERVE_ENGINE`.
- **The signal codec (`efe401bb`).** Every signal between a process and its daemon converts
  through one registry of kinds rather than two chains of type checks. The coverage test fills
  every field of every kind; this run reads the codec on the wire, both ways, under a change.
- **The executor's change (`8efdb855`).** The carrier join and the local join share every step
  of a change but the join itself. This run's shape is the carrier's: the spares are the node
  pods', owned by their daemons, joined through the coordinator. The local join (an executor's
  own spare, the single-node shape) is not exercised here.

None of this changes what the proof measures. The predictions are proof-26's, with its figures
as the bounds, plus two that read the changed code directly.

## Predictions

| # | Prediction | Read from | Falsified by |
|---|---|---|---|
| 1 | Both losses commit a REPLACE of rank 1 through a covered set, at epochs 1 and 2, and all 16 streams finish with every token. | the verdict, `load-N.json` | any stream error, or a change refused or not at the loss's epoch |
| 2 | Loss 1 (deletion) reaches the coordinator as a notice, and its worst gap is at most 1.0 s (proof-26: 0.642 s). | the coordinator's log, `load-1.json` | no notice, or over 1.0 s |
| 3 | Loss 2 (crash): detection, kill to the departure line, is at most 0.5 s (proof-26: about 2 ms), and the worst gap is at most 2.0 s (proof-26: 1.135 s). | `kill-2.txt`, the coordinator's log, `load-2.json` | over either bound |
| 4 | Before each loss, the engine reads Covered True, reason EveryLossCovered. | the driver's `the engine:` line | Covered False once rank 1's set formed |
| 5 | Every spare that stands is built from the replay: its log reads "rank 1's start-up calls read from the core" before its load, and the head's log reads "rank 1's replay: N calls written" for each. No spare refuses a configuration. | the spares' and the head's logs | a spare standing without the line, a replay refused, or a spare never standing |
| 6 | Every join is taken on its digest: each spare that takes rank 1 answers READY, and no log reads "built from other start-up calls". | the spares' logs; the verdict | any FAILED answer naming the start-up calls |
| 7 | The pod Grove makes in place of loss 1's stands as rank 1's spare and covers loss 2. The node crashed in loss 2 starts again in place and stands as a spare. | `serving-pods.txt`, its log, the cover line of loss 2 | either pod serving a rank, or never standing |
| 8 | The driver reads the engine as serving by its record before the first loss: its `serves:` line names every position held and a head registered. | `run.log` | the driver timing out on the record, or a loss taken before the line |
| 9 | Nothing is left behind: no ReleaseLeaked event and no stale exit (4) in the run. | `engines-proof-events*.json`; `serving-pods.txt` | any ReleaseLeaked, or a worker exiting 4 |
| 10 | The codec decodes every signal the run exchanges: no log of any pod reads "which this build does not decode", "unknown to this build" or a `ValueError` from `everserve.contract.signals`. | every pod's log | any such line |
| 11 | No process of the engine ends on an Everserve error: no pod's log holds a Python traceback whose frames name `everserve_vllm` or `everserve`, and the only container restart is loss 2's crashed pod. | every pod's log; `serving-pods.txt` | a traceback naming the adapter or the core, or a second restart |

If #1, #10 or #11 fails, the record says where it stopped, the defect is fixed on CPU first,
and another run needs Cort's word. On green, the open-source tree's worker-loss path is proven
in this shape, and these figures are its current numbers.
