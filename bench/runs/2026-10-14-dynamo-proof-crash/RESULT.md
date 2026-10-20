# Result: rank 1's node crashed under load (proof-20)

**Verdict: fail, on loss 2, with one product defect found and fixed.** On 2026-10-14,
`bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-how crash` ran on images built from main at
`5776450`, against the [pre-registration](PREREGISTRATION.md) beside this record.

Loss 1 is the first loss without notice on GPUs:
- The serving node's workers were killed with SIGKILL under load.
- The coordinator saw the departure about 2 ms later.
- It committed a REPLACE of rank 1 through the covered set in 266 ms.
- All eight streams finished, the worst gap 0.562 s.

Loss 2 never came. The crashed pod's node process went on holding nothing and never ended, so
its pod never started again to stand as a spare, and no set ever covered rank 1 again.

The run's output is `run/`. The kill's own output, with its moment on the pod's clock, is
`run/kill-1.txt`.

## The predictions

| # | Prediction | Read | Verdict |
|---|---|---|---|
| 1 | Both losses REPLACE rank 1 through a covered set, at epochs 1 and 2, every stream finishing | loss 1: epoch 1 REPLACE through `0-R1@…c8s70…:0`, 8/8 streams. Loss 2 never taken. | **fail** (loss 2) |
| 2 | Detection, kill to the coordinator's departure line, at most 0.5 s | kill 20:53:55.748 (pod clock); "rank 1's worker … departed outside any change" 20:53:55.750 (coordinator): about 2 ms | **pass** |
| 3 | Loss 1's worst gap at most 1.2 s | 0.562 s; change committed 266 ms after the report | **pass** |
| 4 | Loss 2's worst gap at most 2.0 s | not taken | **not read** |
| 5 | Before each loss, Covered True, EveryLossCovered, the head's node named | before loss 1: "Covered True (EveryLossCovered): every node's and position's loss has a formed set; the head's node holds [0], whose loss restarts the engine". Before loss 2: never | **pass** for loss 1 |
| 6 | The crashed pod's node process ends, its pod starts again in place, asks and stands as a spare | the pod stood Running, 0 restarts, 20 minutes; its node process logged nothing after "engine proof is serving (2 of 2 positions held, epoch 1)" | **fail** |

## Loss 1, from the record

| UTC | What | Where |
|---|---|---|
| 20:53:45 | rank 1's set formed (plan 2 at epoch 0); the engine Covered True | run.log |
| 20:53:55.748 | the kill, on the pod's clock | `kill-1.txt` |
| 20:53:55.750 | "rank 1's worker on …c20vw… departed outside any change: the rank is lost" | coordinator log |
| 20:53:55.758 | the node's own report: "worker 1 is exiting; reported to the daemon" | the crashed pod's log, coordinator log |
| 20:53:55.762 | rank 1's rows staged from the head's node's store onto the spare's | coordinator log |
| 20:53:56.025 | "membership committed at epoch 1 (Replace, lost [1])": 266 ms after the report | coordinator log, `lastChange` |
| 20:54:24 | 8 of 8 streams finished, worst gap 0.562 s | `load-1.json` |

## The finding: a node whose worker died never learned its rank had left

- **What happened.** A change that replaces a rank goes to the rank's survivors and spares. The
  node whose worker died is neither, and it never hears the commit. `NodeExecutor` recorded a
  rank as left only from a commit it heard (`node_change`). So the crashed node never knew it held
  nothing: `serve()` never returned, the process never ended, and its pod never asked again.
- **Fix, 6220b01.** `NodeExecutor._left_by_placement` reads the placement every connection hears
  at every change. A dead worker's rank placed on another node at a later epoch has left, so the
  node then holds nothing, ends, and its pod asks again. Tested against the pinned vLLM.
- **Coordinator side.** The coordinator knew all along that the old pod held nothing. A condition
  saying so would show an operator such a pod. That is the peer's, after this fix.

## A harness fault in the same run

The crash's killer killed itself (`command terminated with exit code 137`, `run/kill-1.txt`). Its
own command line held the name it matched, `VLLM::Worker`. It killed rank 1's worker before
itself, as the record shows, but its account of what it killed was lost.

Fixed in 6220b01: the killer skips its own process and holds the names in pieces. It was checked
by extracting it as bash hands it over and parsing it.

## Next

proof-21: the same pre-registration on images with 6220b01.
