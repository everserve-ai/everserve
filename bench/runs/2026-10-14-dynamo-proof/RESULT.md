# Result: a node of a Dynamo instance taken away under load, its rank taken by a spare

**Verdict: pass.** On 2026-10-14, `bench/harness/k8s/dynamo_proof.sh` ran an NVIDIA Dynamo
instance on GPUs with Everserve installed as an operator installs it. It took the engine's
serving node away while eight long streams were running through Dynamo's frontend. The
coordinator handed that node's rank to the standing spare in 412 ms, and every stream finished
with all its tokens and no error. The worst gap between two tokens of any stream was 0.683 s,
at the change. The run's output is `run/`; the driver's own log is `run/run.log`.

The verdict is the coordinator's record, not the streams alone. The driver passes only when:
- before the loss, a formed set covers the loss of rank 1;
- after it, the engine's last committed change is a REPLACE of rank 1;
- every stream finishes without an error.

The checks are `bench/harness/k8s/proof_verdict.py`. Proof-13 is why: its streams finished
while the coordinator refused the change (see below).

## What ran

- **Cluster:** Nebius managed Kubernetes in eu-north1. Three `gpu-l40s` nodes, one L40S each.
  - Asked for at 12:31:46Z; GPUs advertised at 12:38:41Z.
  - Take-down asked at 12:50:14Z, under a detached watchdog set at 1.5 h.
- **Platform:** NVIDIA's GPU operator (device plugin only; the drivers are the node image's), then
  Dynamo's platform 1.5.1 with Grove.
- **Everserve:** its chart, as `deploy/kubernetes/everserve`: one coordinator and the node
  DaemonSet. The certificates were made before renting (`controller certs`).
- **Images**, built from main at `92d6204` (`bench/harness/build_images.sh`, from `git archive`):
  - control plane: `everserve:dev-92d6204`;
  - engine: `everserve-engine:dev-92d6204`, which is vLLM 0.31.0 with Dynamo's runtime 1.6.0
    and the adapter.
- **Instance:** `run/instance.json`, rendered by `everserve_dynamo.render` into a
  DynamoGraphDeployment and its policy (`run/proof.k8s.json`, `run/proof.engine.json`):
  - Qwen/Qwen2.5-0.5B-Instruct at TP 1 × PP 2: the head holds rank 0 and one node pod holds
    rank 1;
  - one spare node pod;
  - guarantee `continuous`;
  - 4096 KV blocks of 16 tokens.
- **Load:** after one warm completion, `bench/harness/k8s/stream_load.py` ran eight streams of
  6000 tokens each (about 40 s) through Dynamo's frontend. The serving node pod was deleted
  11 s into them (`kubectl delete pod`, as a node drain or an eviction does).

## What happened, from the record

| When (UTC) | What | Where it is read |
|---|---|---|
| 12:48:23 | the head's engine core serves membership changes (92d6204's hook) | head log |
| 12:49:03 | the spare node attached for rank 1 in 40.1 s; its compiled set was staged from the serving node (torch.compile 0.98 s, against 11.5 s on the serving node) | spare log |
| 12:49:04 | `SetCovered`: set `0-R1@…e00scg43…:0` (REPLACE of ranks [1]), 1 of 1 groups formed | events, run.log `cover:` |
| 12:49:16 | the serving node pod `proof-0-nodes-cp9gx` deleted | run.log |
| 12:49:17.860 | the coordinator heard the node's notice for rank 1, sent on SIGTERM (248e610), with its deadline 5 s on at 12:49:22.860; the node served on until then | coordinator log |
| 12:49:22.870 | `ChangeStarted`: epoch 1, REPLACE for ranks [1]; the head's step ended on the notice and completed the change in its own loop | events, head log |
| +0.3 ms | delivered to the survivors and the spare | `lastChange` |
| +9.8 ms | rank 0 acknowledged | `lastChange.acks` |
| +401 ms | rank 1, the spare, acknowledged: "took rank 1's position at epoch 1 in 0.003 s" | `lastChange.acks`, spare log |
| 12:49:23.274 | `ChangeCommitted`: epoch 1, REPLACE for ranks [1], 412 ms, through covered set `0-R1@…` | events, `lastChange` |
| 12:49:46 | 8 of 8 streams completed, 6000 tokens each by the server's usage count, no error | `run/load.json` |

Every stream's largest gap between tokens was between 0.670 s and 0.683 s, and in every stream it
fell at 12:49:22.93: the change. Grove made a new node pod in place of the deleted one
(`proof-0-nodes-wjktx`), and it was told spare.

## What the earlier runs found

Each finding was fixed before this run, and the fixes are on main.

| Run | Found | Fix |
|---|---|---|
| proof-7, 8 | a spare could not reproduce the serving rank's compiled set; the head's adapter logs were invisible | the compile-cache staging (decision 0043); `log_through_vllm` |
| proof-9, 10 | the step guard refused the spare on Dynamo's own `additional_config`; the driver's policy was not the instance's | the renderer's engine args for every process; the policy from the renderer's engine entry; the guarantee guard |
| proof-11 | the spare could not read vLLM's 0600 compiled files | the publisher shares them (g+r) |
| proof-12 | the loss of a deleted node pod was never reported | a departed serving worker is its rank's loss (402503a); a node pod leaves on SIGTERM with notices (248e610) |
| proof-13 | printed "passed" while the coordinator refused the change: the spare's set never formed, because the group met at 127.0.0.1 (the planner fell back to loopback with no daemon host); the taken pod served through its grace period | a group meets at its server process's own address (a4fdc50); the driver's verdict from the record (e31972b) |
| proof-14 | the set formed and the change started, but the head's engine core died at the first notice: the adapter's step hook wrapped `EngineCore.__init__`, and vLLM loads plugins inside that constructor, so a freshly started engine core (Dynamo's head) kept the stock step | the hook is a data descriptor on `EngineCore.step_fn` (92d6204) |

## Seen in the record, not yet fixed

- One millisecond after the commit, `GroupFailed`: rank 0 could not form `0-R1@…` again
  ("Address already in use"). The commit had just made that set the serving group. The next plan
  superseded it, so it cost nothing here, but it is a failure event after a clean REPLACE.
- One second after the commit, the engine read `Covered False`, with no set yet planned for the
  new spare. A run that takes the node a second time, after the engine reads covered again, is
  the next arm.
- The driver's verdict line prints "through through covered set" (`run/run.log`). The check is
  right; the wording is fixed in the commit that adds this record.

## What this does not show

- A loss without notice: a node that dies rather than being deleted. The departed-worker report
  (402503a) is that path, and it was not exercised here.
- The head's own loss: an engine restart under decision 0038's placement.
- A model of production size, or more than one node per stage.
- CoreWeave, which is the MVP's spot target.
