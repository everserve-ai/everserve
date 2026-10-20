# Result: the worker-loss proof on the tree without the head's failover, all nine predictions held (proof-26)

**Verdict: pass.** On 2026-10-19/20, `bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-how
delete,crash --platform gpu-h100-sxm --preset 1gpu-16vcpu-200gb` ran on 3 × H100 in eu-north1, on
images `dev-80aa21e`. That is main with:
- the head's failover removed (decision 0045);
- the spare's start-up replay cut to its core (decision 0044);
- the fix for a node new to the engine (`a791ffb3`);
- the worker-loss end to end in CI.

Rank 1's node pod was deleted, and then the spare that had taken it was killed with SIGKILL. Each
time a REPLACE committed through a covered set, and all 16 streams finished with every token. Every
spare was built from the core's replayed start-up calls, the first run of the replay in this shape.

The run's output is `run/` (the Dynamo platform's chart left out). Cort's word for it: "ok you have
the go ahead for the proof as soon as its ready".

## The predictions

Read against the [pre-registration](PREREGISTRATION.md).

| # | Prediction | Read | Verdict |
|---|---|---|---|
| 1 | Both losses REPLACE rank 1 through a covered set, at epochs 1 and 2, every stream finishing | epoch 1 through `0-R1@…e00cc245…:0`, epoch 2 through `0-R1@…e00eq32c…:0`; 8 of 8 streams each, no error | **pass** |
| 2 | Loss 1 reaches the coordinator as a notice; worst gap at most 1.0 s | "notice for rank 1" at 00:09:06.392; worst gap 0.642 s | **pass** |
| 3 | Loss 2: detection at most 0.5 s; worst gap at most 2.0 s | kill at 00:10:27.295 (pod clock), departure line at 00:10:27.297, about 2 ms; worst gap 1.135 s | **pass** |
| 4 | Covered True before each loss | "Covered True (EveryLossCovered)" at 00:08:56 and 00:10:16 | **pass** |
| 5 | Every spare that stands is built from the replay, N calls written for each, no replay refused | the head wrote "rank 1's replay: 3 calls written" at 00:08:24, 00:09:40 and 00:10:54, one for each spare that stood (the layout, the memory profiling, the KV configuration; no model-length change); each spare logged "rank 1's start-up calls read from the core"; no refusal in any log | **pass** |
| 6 | Every join is taken on its digest, none naming other start-up calls | "spare took rank 1's position at epoch 1 in 0.003 s" and "at epoch 2 in 0.002 s"; no "built from other start-up calls" in any log | **pass** |
| 7 | Grove's pod in place of loss 1's covers loss 2; the pod crashed in loss 2 starts again in place and stands as a spare | loss 2's set is `proof-0-nodes-mpw97`, on loss 1's node; `proof-0-nodes-555vh` restarted in place (1 restart), was told spare, and read its calls again at 00:10:54 | **pass** |
| 8 | The driver reads the engine as serving by its record before the first loss | "proof serves: Serving True (Members); 2 of 2 positions held; head on computeinstance-e00qppb6…" at 00:08:38, before loss 1 at 00:08:56 | **pass** |
| 9 | Nothing left behind: no ReleaseLeaked, no stale exit (4) | no ReleaseLeaked in the events or any log; no stale exit in any log; the killed spare ended -9, as the kill made it | **pass** |

## What it means

- **The worker-loss numbers on this tree** are 0.642 s worst gap with notice and 1.135 s on a
  crash, with about 2 ms detection. Proof-25 read 0.67 s, 1.328 s and about 1 ms on the tree before
  the replay and the removal. None of these is a measurement of the difference: one run each, 8
  streams.
- **The replay costs nothing in a take.** Each spare built from the replayed calls in 4.7 to 4.9 s
  (`kv_config_s`, vLLM's own memory profiling among them), at its build, while it stood. The takes
  themselves were 2 to 3 ms.
- **What this run does not read.** Grove recreated each pod on the node it ran on, so the fix for a
  node new to the engine is not exercised here. The CPU end to end covers it
  (`adapters/vllm/tests/test_worker_loss_e2e.py`).

**Cost.** The cluster ran from about 23:52Z to its take-down at 00:23:50Z, about 32 minutes of
3 × H100 at about $4.45 per GPU-hour: about $7. The image build's VM ran about 11 minutes, under $1.
