# Pre-registration: the gated blocking join on GPUs, a deletion then a crash (proof-25)

Written before the run. The run is `bench/harness/k8s/dynamo_proof.sh --losses 2
--loss-how delete,crash --platform gpu-h100-sxm --preset 1gpu-16vcpu-200gb`:
- images built from `sg2-engine` at the commit that adds the guard's log line, right after this
  file (the record names its hash). That is main at `af0a613`
  with the shared-groups pair, and the bounded join Cort chose on 2026-10-15:
  - a member reports `GroupReady`, and the id is relayed only once every member is ready
    (`6bdfea3`, `91d99c7`);
  - the formation deadline, `cover.formTimeoutMs` (`6054ff1`, `d883af8`);
  - a release or a formation left behind is reported and retired (`87d01dd`, `2e524c0`,
    `a3ec227`);
  - the capture guard on vLLM's handles (`b3b6f4f`);
  - the stale-worker exit (`c6c52ae`) and the unknown-outcome decode (`9142082`);
- 3 × H100 in eu-north1 (Cort's word for today's proofs), a 1.5 h watchdog, Qwen2.5-0.5B at TP 1 ×
  PP 2.

**Cap.** About $4.45 per GPU-hour (our one bill), so three nodes come to about $13.35 an hour.
The run is about 40 minutes, so about $9. The 1.5 h watchdog makes the worst case about $20.
Building the images costs under $1.

## What differs from proof-23

Proof-23 joined a cover group by the blocking `ncclCommInitRank`, from an id its server made and
relayed. Its members joined as the id reached them: the server at once, the others on the relay.
Here, before any member joins:
- every member, the server included, reports `GroupReady` at the plan's generation;
- the coordinator relays the server's id only once every member is ready;
- the id's arrival is each member's go, the server's too.

So no member enters NCCL's blocking initialisation before every other has shown it is alive. The
window left, a member lost after the go and before its join, is bounded by the formation's
deadline. It is not injected here: the harness cannot time a window of milliseconds inside a
process from outside, and a fault hook in the product to do it is barred. Its path is covered by
the unit and coordinator tests named above.

The capture guard is on every NCCL handle vLLM holds, before vLLM captures its graphs. If any
capture in vLLM's warm-up reached NCCL, the engine would not start (#7).

## Predictions

| # | Prediction | Read from | Falsified by |
|---|---|---|---|
| 1 | Both losses commit a REPLACE of rank 1 through a covered set, at epochs 1 and 2, and all 16 streams finish with every token. | the verdict, `load-N.json` | any stream error, or a change refused or not at the loss's epoch |
| 2 | Loss 1 (deletion) reaches the coordinator as a notice, and its worst gap is at most 1.0 s (proof-18: 0.64 s; proof-23: 0.642 s, on L40S). | the coordinator's log, `load-1.json` | no notice, or over 1.0 s |
| 3 | Loss 2 (crash): detection, kill to the departure line, is at most 0.5 s (proofs 22 and 23: about 2 ms), and the worst gap is at most 2.0 s (proof-23: 1.387 s, on L40S). | `kill-2.txt`, the coordinator's log, `load-2.json` | over either bound |
| 4 | Before each loss, the engine reads Covered True, reason EveryLossCovered. | the driver's `the engine:` line | Covered False once rank 1's set formed |
| 5 | Every plan's groups form through the gate: each covering plan reads "1 of 1 groups formed"; the coordinator relays each group's id only after every member's GroupReady at that generation; the event record holds no GroupFailed. | the driver's `cover:` lines; the coordinator's log; `engines-proof-events*.json` | any GroupFailed, an id relayed before a member's GroupReady, or a set never formed within the driver's 600 s wait |
| 6 | The pod Grove makes in place of loss 1's stands as rank 1's spare and covers loss 2. The node crashed in loss 2 starts again in place and stands as a spare. | `serving-pods.txt`, its log, the cover line of loss 2 | either pod serving a rank, or never standing |
| 7 | The capture guard is on and costs nothing that shows: every engine pod logs "NCCL handles guarded against a graph's capture" before its warm-up, the engine starts (no capture reached NCCL), and #2's and #3's bounds hold. | the workers' logs; the instance serving | a pod not guarded, or the engine not starting with a capture refused |
| 8 | Nothing is left behind: no ReleaseLeaked event and no stale exit (4) in the run. | `engines-proof-events*.json`; `serving-pods.txt` | any ReleaseLeaked, or a worker exiting 4 |

Mapping to proof-23: #1, #4 and #6 are as there. #2 and #3 keep proof-23's bounds; on H100 the
figures are not comparable with L40S's, but the bounds hold either way. #5 adds the gate. #7 and
#8 are this landing's own.

If #1, #4, #5 or #7 fails, the pair does not land, and the record says where it stopped. On
green, the pair lands on main and the peer reports its CI.
