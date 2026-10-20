# Result: the worker-loss proof on the open-source tree, ten of eleven predictions held (proof-27)

**Verdict: pass on the record; prediction 11 falsified as worded.** On 2026-10-20,
`bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-how delete,crash --platform gpu-h100-sxm
--preset 1gpu-16vcpu-200gb` ran on 3 × H100 in eu-north1, on images `dev-645c75b`, built from
the commit that added the [pre-registration](PREREGISTRATION.md): branch `oss-polish` at
`645c75bd`, which is main after the open-source audit (PR #15) plus the signal codec's registry
(`efe401bb`) and the executor's shared change (`8efdb855`).

Rank 1's node pod was deleted, and then the spare that had taken it was killed with SIGKILL. Each
time a REPLACE committed through a covered set, and all 16 streams finished with every token.
The run's output is `run/`, curated by the rule in `bench/runs/README.md` (the pod logs the
reads below quote are not kept; the driver's log and the coordinator's records are).

## The predictions

Read against the [pre-registration](PREREGISTRATION.md).

| # | Prediction | Read | Verdict |
|---|---|---|---|
| 1 | Both losses REPLACE rank 1 through a covered set, at epochs 1 and 2, every stream finishing | epoch 1 "through covered set 0-R1@…e00vzky7…:0", epoch 2 "through covered set 0-R1@…e00zx77p…:0" (`engines-proof-loss-N.json`, `lastChange.cover`); 8 of 8 streams each, no error, none short (`load-N.json`) | **pass** |
| 2 | Loss 1 reaches the coordinator as a notice; worst gap at most 1.0 s | "notice for rank 1" at 12:18:27.758, committed at 12:18:33.157; worst gap 0.634 s | **pass** |
| 3 | Loss 2: detection at most 0.5 s; worst gap at most 2.0 s | kill at 12:19:49.129 (pod clock, `kill-2.txt`), the coordinator's departure line at 12:19:49.131, about 2 ms; committed at 12:19:50.017; worst gap 1.176 s | **pass** |
| 4 | Covered True before each loss | "Covered True (EveryLossCovered)" at 12:18:17 and 12:19:38 (`run.log`) | **pass** |
| 5 | Every spare that stands is built from the replay, N calls written for each, no replay refused | the head wrote "rank 1's replay: 3 calls written" at 12:17:37, 12:19:03 and 12:20:17, one for each spare that stood; each spare logged "rank 1's start-up calls read from the core"; no refusal in any log | **pass** |
| 6 | Every join is taken on its digest, none naming other start-up calls | "spare took rank 1's position at epoch 1 in 0.003 s" and "at epoch 2 in 0.002 s"; no "built from other start-up calls" in any log | **pass** |
| 7 | Grove's pod in place of loss 1's covers loss 2; the pod crashed in loss 2 starts again in place and stands as a spare | loss 2's set is `proof-0-nodes-w2jpf`, on loss 1's node; `proof-0-nodes-9pnpm` restarted in place (1 restart) and read its calls again at 12:20:17; the logs were collected before it registered | **pass** |
| 8 | The driver reads the engine as serving by its record before the first loss | "proof serves: Serving True (Members); 2 of 2 positions held; head on computeinstance-e00kd9r0…" at 12:17:59, before loss 1 at 12:18:18 | **pass** |
| 9 | Nothing left behind: no ReleaseLeaked, no stale exit (4) | no ReleaseLeaked in the events or any log; no stale exit in any log; the killed spare ended -9, as the kill made it | **pass** |
| 10 | The codec decodes every signal the run exchanges | no pod's log reads "which this build does not decode", "unknown to this build" or a `ValueError` from `everserve.contract.signals` | **pass** |
| 11 | No pod's log holds a Python traceback whose frames name `everserve_vllm` or `everserve`; the only restart is loss 2's crashed pod | the only restart is `proof-0-nodes-9pnpm`'s. But the head's log holds four tracebacks at 12:19:49, the crash's step, each ending `everserve_vllm.pipeline.MemberSetChanged: step aborted: the member set changed: the pipeline refused its group until the boundary`: the designed abort of the step a loss cut (ledger row A8a), raised in the worker and logged with its traceback by vLLM's worker loop (`multiproc_executor.py:1058`) before it crosses to the executor as text. No process ended on it; the change committed 0.9 s later. The crashed pod's previous log holds three torch tracebacks (`triton_bundler.py`: a cubin missing from the compile cache, torch's own warning), none naming Everserve's code | **fail as worded** |

## What it means

- **The open-source tree's worker-loss path is proven in this shape.** The patch registry, the
  codec registry and the executor's shared change ran under two losses with every stream whole.
  The figures are 0.634 s worst gap with notice and 1.176 s on a crash, with about 2 ms detection;
  proof-26 read 0.642 s, 1.135 s and about 2 ms on the tree before. One run each, 8 streams: not a
  measurement of the difference.
- **Prediction 11 was wrong as written, not the tree.** A step a loss cuts is aborted by raising
  in the worker (`MemberSetChanged`), and vLLM's worker loop logs every exception an RPC raises
  with its traceback, so the designed abort always leaves an ERROR traceback in the head's log at
  a change. The prediction should have excluded it. Whether the abort should cross vLLM's loop
  without that log line is an open item for the owner: it is vLLM's `logger.exception` in
  `WorkerProc.worker_busy_loop`, not ours, and quieting it means one more private line.
- **What this run does not read.** The local join (an executor's own spare, the single-node
  shape) is not exercised: this shape's spares are the node pods', joined over the carrier. Grove
  recreated each pod on the node it ran on, so the fix for a node new to the engine is not
  exercised either; the CPU end to end covers it.

**Cost.** The cluster ran from 12:01:43Z to its take-down at 12:32:44Z, 31 minutes of 3 × H100
at about $4.45 per GPU-hour: about $7. The image build's VM ran about 11 minutes, under $1.
