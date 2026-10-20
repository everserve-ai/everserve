# Pre-registration: the head's node lost under load, its standby taking the head (proof-24)

Written before the run, and read by both sessions (370f6ee). No bar here moves after this
commit. The run is `bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-target head
--loss-how crash --separate-head --second-engine --platform gpu-h100-sxm
--preset 1gpu-16vcpu-200gb --hours 1.5`:
- images built from main at `168cb40` or later with no engine or controller change between (the
  record names the image's commit): E1, E2/E4's core half, the separateHead render and the
  harness (`168cb40`), on the head failover's control plane (H1–H5, H4b's copy) and the
  coordinator's bring-up and takeover joins (`b9f85f8`, `e2524d0`);
- 6 × H100 in eu-north1, a 1.5 h watchdog, at most about $27 (Cort approved the spend
  2026-10-15, relayed by the rig session):
  - engine A, the one under test: `separateHead`, its head and standby on two nodes of the
    `<name>-heads` group (pod anti-affinity on hostname), and its 2 ranks and 1 spare on the GPU
    group, Qwen2.5-0.5B at TP 1 × PP 2. Each pod selects its group by `nebius.com/node-group-id`,
    checked on the live nodes before the install.
  - engine B, `proof-b`: the same model at PP 1 on one GPU node, no spare, its head beside its
    GPU. It is there so a stream broken by A's loss has an instance to migrate to (Dynamo 1.5.1
    fails a migration at once when no instance is registered, b23ce1d). Nothing about B is
    predicted beyond serving what migrates to it.
  - The head nodes are 1-GPU nodes because eu-north1 allows no CPU nodes. The core sees one GPU,
    since vLLM resolves its configuration from the device it sees; the GPU computes nothing. The
    production shape puts the head on CPU nodes; the mechanism under test is the same.
- 16 streams through Dynamo's front end, both engines in one global Dynamo namespace, with
  request migration on (`migration_limit` 3, the render's default). Under strategy C (0038,
  amended 2026-10-15) the client's stream is Dynamo's: a stream broken by the head's loss is
  re-sent, with every token already sent, to B or to A's new core. This build's new core does not
  adopt the mirrored block tables (KV adoption is a later landing): its KV cache manager starts
  empty, the survivors drop the lost core's requests when they move rings, and a stream re-sent
  to A is recomputed in full. The standby's copy of the head's table runs, and is predicted
  (#1, #8), but the new core does not read it in this build. The mirror's hold of outputs is off.
- `--losses 2`: the head is lost, its standby takes it, the lost pod restarts and stands as the
  new standby (#9), and then the new head is lost the same way. Every prediction is read for each
  loss n, from that loss's files (`load-<n>.json`, `served-<n>.json`, `after-<n>.json`,
  `served-after-<n>.json`); loss 2 runs only once #9 held for loss 1.

Before the rental, the vLLM-gated adapter tests run green in the engine image (build_images).
Built for it: `--loss-target head` (killer.py kills the engine core with SIGKILL, PID 1 out of reach,
tested on proof-25's process tables); the head node group and its id (mk8s, e2524d0); the pod
diff (snapshot.py); Dynamo 1.5.1's migration read at the pin (b23ce1d); the coordinator's side
of bring-up and takeover joins (b9f85f8, 4c87659) and `separateHead` (a3215cb).

## Predictions

| # | Prediction | Read from | Falsified by |
|---|---|---|---|
| 1 | Before the loss the head is covered: a standby stands on the second head node and its copy of the head's requests is caught up. Covered's message names no head reason, and `everserve_store_requests_copy_behind{engine}` reads 0 on the head's node. | `engines-proof-before.json`; the store's metrics | HeadUncovered, or the copy behind, at the loss |
| 2 | Detection, from the kill to the coordinator's "its head on <node> lost" line, is at most 0.5 s: the core's daemon link reads EOF at once, as a rank's does (proof-22: about 2 ms). | `kill-1.txt`; the coordinator's log | over 0.5 s |
| 3 | The engine is never down. A change of `OUTCOME_HEAD` commits at the next epoch, and the standby's node holds the head after it. | the coordinator's log; `engines-proof-after.json` | Down at any moment, or no head change committed |
| 4 | No worker restarts and no weight is loaded again: no pod of A's GPU group is gone, new or restarted, and no worker logs a model load after the kill. Only the lost head's pod restarts. | `pods-before.json` and `pods-after.json` diffed by snapshot.py; the workers' logs | a GPU-group pod gone, new or restarted, or a model load after the kill |
| 5 | Every stream finishes with every token it asked for, and each migrated stream resumed at the token index its client had reached: nothing repeated, nothing dropped. Exact equality with a separate run is not asked: other batching can flip an argmax with no defect. | `load-1.json`; the front end's migration lines, with each request's resume index | a stream error, a count short or long, or a resume index other than the tokens sent |
| 6 | The worst inter-token gap at the change is at most 5 s: detection (≤ 0.5 s), the epoch's barrier and the survivors' joins to the new core (the takers' joins: 0.3–0.6 s in proof-18 to 22), the core's attach (no profiling: the workers' KV configuration is reused), its registration with Dynamo's discovery, and the front end's migration, with the re-sent prefix's prefill on whichever engine took it. It is the first measure of this path; the bound is wide on purpose, and the record splits the gap by these moments. | `load-<n>.json`; the coordinator's change line; the front end's migration lines | over 5 s |
| 7 | Dynamo's front end migrates every broken stream within its retries, to B or to A's new core: each migration line names the instance it went to, within `migration_limit` attempts, and no client sees an error. | the front end's migration lines and attempt counts; `served-<n>.json`; `load-<n>.json`'s errors | a client error, or a migration that ran out of attempts |
| 7b | Recorded, not predicted: a stream re-sent to A's new core is recomputed in full (no adoption in this build), and the record gives each such stream's prefill time as the cost KV adoption would remove. If none went to A, it is recorded as none. | `served-<n>.json`; the new core's log for those requests | (a record, not a bar) |
| 8 | The standby's copy keeps up with the head's table through the run: `everserve_store_requests_copy_behind` reads 0 at the loss, and no answer carried `behind`. It is the control plane's half of adoption; the new core does not read the copy in this build. | the head's store metrics; the head's log (`behind` at warning) | the copy behind at the loss |
| 9 | After the change the head is uncovered until a standby stands again: Covered reads HeadUncovered ("no standby head stands"). The lost head's pod restarts within 120 s of the kill, whether Dynamo's worker (PID 1) exits on its core's death or Kubernetes' liveness probe ends it; the record says which. Its core then asks, stands as the standby, and the head is covered again. | `engines-proof-after.json`; the conditions' log | covered with no standby, the pod not restarted within 120 s, or the restarted core given the head |
| 10 | A's new core serves after the change: a ready head pod of A off the lost node registers its instance with Dynamo, and a batch of 8 streams sent to that instance by id once the change has committed finishes whole. This proves the new core registered and serves; whether Dynamo's router picks it unasked is #7's `served-<n>.json`. | `after-<n>.json`; `served-after-<n>.json` | no ready head pod within 600 s, or a stream of the batch short or in error |

What each would mean:
- #5 failing is the change's worst defect: a client told something twice or not at all. Under
  strategy C that is Dynamo's migration resuming at the wrong index, or our new core taking the
  re-sent prefix wrongly; the record names which.
- #7 failing reopens H2 (0038, amended 2026-10-15): Dynamo's front end does not reach the new core
  within its retries, and a Dynamo setting (a longer wait, more attempts) or the head record
  route is needed.
- #10 failing with #3 passing: the control plane changed the head, but the new core never served
  through Dynamo, so the takeover is unproven end to end.
- #9 failing on the restart (Dynamo's worker outlives its core and nothing ends it) is a defect of
  the integration: a dead head would hold its pod and node. It also stops loss 2.
- #6 over its bound with #2–#5 passing is a cost to find, not a correctness failure. The record
  splits the gap by the change's moments.
