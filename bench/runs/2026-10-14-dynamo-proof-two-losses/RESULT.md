# Result: rank 1's node taken away twice under load, each loss taken by a spare

**Verdict: pass, with one finding.** On 2026-10-14, `bench/harness/k8s/dynamo_proof.sh --losses 2`
ran an NVIDIA Dynamo instance on GPUs with Everserve installed. It took away the node pod
holding rank 1 twice, each time while eight long streams ran through Dynamo's frontend. Each
loss was a REPLACE committed through a covered set, and all 16 streams finished with every token
and no error.

The second loss is the one this run adds. Its spare was Grove's replacement for the pod taken in
the first loss, covered for rank 1 by a plan made at epoch 1.

The finding: the second loss's worst gap was 3.7 s, against 0.7 s for the first. The pod taken
the second time held rank 1 through a spare process, and it gave no notice. Its loss went
through the no-notice path instead. The fix is 42fe6bc and 2c0ede2 (below); a proof on their
images is the next run.

The run's output is `run/`. The driver's own log is `run/run.log`. Each loss's record is
`run/engines-proof-loss-N.json` and `run/engines-proof-events-loss-N.json`, and its streams are
`run/load-N.json`.

## What ran

As [proof-15](../2026-10-14-dynamo-proof/RESULT.md), with these differences:
- images `everserve:dev-993d266` and `everserve-engine:dev-993d266`, built from main at
  `993d266`;
- the driver's `--losses 2`.

Each loss runs these steps:
1. Wait for a formed set to cover rank 1.
2. Find the pod on the node the coordinator's record places rank 1 on.
3. Start 8 streams of 6000 tokens, and delete that pod 8 s in.
4. Pass only on a REPLACE of rank 1 at the epoch that loss commits.

The cluster:
- asked for at 13:43:23Z;
- instance applied at 13:51:04Z, serving at 14:00:34Z;
- taken down at 14:14:55Z, under a 1.5 h watchdog.

## What happened, from the record

| | Loss 1 | Loss 2 |
|---|---|---|
| covering set, at the take | `0-R1@…e6ah…:0` (the standing spare), **plan 2 at epoch 0** | `0-R1@…xdjeb…:0` (Grove's new pod on the freed node), **plan 4 at epoch 1** |
| pod taken (holder of rank 1, from the record) | `proof-0-nodes-z9z8c`, the original serving node | `proof-0-nodes-59v44`, the spare that took rank 1 in loss 1 |
| how the loss reached the coordinator | the node's notice (SIGTERM, 5 s lead) | **no notice:** "rank 1's worker … departed outside any change: the rank is lost" |
| change | epoch 1 REPLACE [1] through the covered set, committed in 357 ms | epoch 2 REPLACE [1] through the covered set, committed in 1,048 ms |
| acknowledgements | rank 0 +9.3 ms, rank 1 (the spare) +356.8 ms | rank 0 +10.8 ms, rank 1 (the spare) +1,048.3 ms |
| streams | 8 of 8, 6000 tokens each, no error | 8 of 8, 6000 tokens each, no error |
| worst gap between tokens | 0.657–0.668 s, at the change | **3.619–3.738 s**, before the change (see below) |

Between the losses, the coordinator covered rank 1 again on its own. Grove made a pod in place
of `z9z8c`, it stood as rank 1's spare (attached in 38.7 s, its compiled set staged), and plan 4
at epoch 1 covered rank 1 with it at 14:01:54, 58 s after the first change.

Two of the peer's fixes landed in `b0ecf3a`, and both are borne out:
- No `GroupFailed` after either commit (proof-15 had one). The new spare logged "cover plan 4
  at epoch 1 waits: epoch 2 committed" after its commit.
- The re-cover above.

## The finding: a spare that took its rank left without notice

- **What the record shows.** Every stream's worst gap in loss 2 fell at 14:02:04.8–08.5. That is
  before the change was reported, at 14:02:10.754, and the report was the departed-worker path
  (402503a), not a notice.
- **Cause.** The pod taken the second time was a standing node: its process ran one spare
  process per rank, and the spare that took rank 1 served it there, outside any node executor.
  On the pod's deletion, `node_main`'s leave saw no serving executor and stopped at once, with
  its spares.
  - The serving spare's steps stopped with it, and the streams waited until its process ended
    and the coordinator read the loss.
  - Proof-15's single loss never took such a pod.
- **Fix, 42fe6bc.**
  - A standing node passes the leave on to its spares (SIGUSR1; SIGTERM stays the stop at once).
  - A spare that took its rank gives it notice on its own link, as a serving node does.
  - It stops once a later placement puts the rank elsewhere, or at its bound.
- **Fix, 2c0ede2.** Found in that fix's review: every leaving process's notice is never under
  the policy's `noticeMinMs`. The coordinator reads a shorter one as no notice.

## What this does not show

- Loss 2 with a notice-sized gap: that is the next run, on the fix's images.
- A loss without notice from a node that dies rather than being deleted, the head's own loss, a
  production-size model, more than one node per stage, and CoreWeave, as in proof-15.
