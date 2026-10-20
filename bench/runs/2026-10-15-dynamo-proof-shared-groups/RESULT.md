# Result: shared groups on GPUs, a deletion then a crash, and NCCL's nonblocking join (proof-23)

**Verdict: pass, all six predictions held.** On 2026-10-15, `bench/harness/k8s/dynamo_proof.sh
--losses 2 --loss-how delete,crash --nccl-probe` ran on images built from `sg2-python` at
`51e2763`. That is main at `3de89a0` with the shared-groups pair (`8b3c0ad`, `ac27bd4`,
`72afaf8`) and the probe.

Each cover group met over the control plane's links: its server made the NCCL id, and the
coordinator carried it to the other member. No port was opened for a cover group. Rank 1's node
pod was deleted (loss 1) and then the spare that had taken it was killed with SIGKILL (loss 2).
Each time a REPLACE committed through a covered set, and all 16 streams finished with every token.

**The NCCL probe answered a1 and a2. It did not answer b**: the probe's two-member group never
formed, so its collectives were never run. The section below says why that is not yet read as
NCCL's answer.

The run's output is `run/`. The probe's answers are in `run/nccl-probe/`.

## The predictions

Read against the [pre-registration](PREREGISTRATION.md).

| # | Prediction | Read | Verdict |
|---|---|---|---|
| 1 | Both losses REPLACE rank 1 through a covered set, at epochs 1 and 2, every stream finishing | epoch 1 through `0-R1@…tb1k24…:0`, epoch 2 through `0-R1@…hhvk74…:0`; 8 of 8 streams each, no error | **pass** |
| 2 | Loss 1 reaches the coordinator as a notice; worst gap at most 1.0 s | "notice for rank 1" at 01:47:40.072; worst gap 0.642 s | **pass** |
| 3 | Loss 2: detection at most 0.5 s; worst gap at most 2.0 s | kill at 01:48:56.528 (pod clock), departure line at 01:48:56.530: 2 ms; worst gap 1.387 s | **pass** |
| 4 | Covered True before each loss | "Covered True (EveryLossCovered)", before both | **pass** |
| 5 | Every plan's groups form over the links, no GroupFailed | plan 2 and plan 4: "1 of 1 groups formed"; each REPLACE set covered 226 ms and 204 ms after its plan was delivered; no group failure in the events or any log | **pass** |
| 6 | Grove's pod in place of loss 1's covers loss 2; the pod crashed in loss 2 starts again in place and stands as a spare | loss 2's set is the pod on loss 1's node (`hhvk74`); `w9ktp` restarted in place (1 restart) and its last answer is "told spare" | **pass** |

The two gaps match the runs they were bounded from: proof-18's deletion (0.64 s) and proof-22's
crash (1.37 s). Carrying the id over the links did not change the change's cost.

## The NCCL probe

NCCL 2.29.7 (`22907`), on the engine's image, beside the serving engine. One process ran in the
head's pod and one in a node pod, on two nodes.

| Q | Read | Answer |
|---|---|---|
| a1: the server waits for a member that never comes | state `ncclInProgress` at 10.0 s; `ncclCommAbort` returned 0 in 0.0 s; the device ran a kernel after | **as expected** |
| a2: a member whose server made the id and exited | state `ncclInProgress` at 10.0 s (no error); abort returned 0 in 0.66 s; the device ran a kernel after | **as expected** |
| b: a group of two formed nonblocking, then eager and graphed all-reduces | both members still `ncclInProgress` at the 120 s deadline; both aborts returned 0 at once; no collective ran | **not answered** |

a1 and a2 show that a nonblocking join can be bounded: an abort at the deadline returns, and the
process's device is left usable.

b is not read as "NCCL's nonblocking init does not form across nodes", for three reasons:
- The engine's own cover groups formed between the same two pods in the same run. Those joins are
  blocking (`ncclCommInitRank` through vLLM) and took about 0.2 s. So the network, the pods and
  the NCCL library are not at fault.
- The probe's run had no control arm. The same two processes were never shown forming a group by
  the blocking call, or by `ncclCommInitRankConfig` with `blocking = 1`. Without that, the probe's
  own path, its configuration prefix included, cannot be told apart from NCCL's nonblocking
  behaviour.
- NCCL's own logging was off (`*.err` are empty), so the record does not say where the
  initialisation waited.

## What comes next

The join stays as built: blocking, from the id carried over the links, which this run proves.
The question b leaves open is how to bound it. It is answered by a probe-only run, pre-registered
before it runs, with:
- three arms on the same two pods: vLLM's blocking `ncclCommInitRank` (the baseline this run
  already shows works), `ncclCommInitRankConfig` with `blocking = 1`, and with `blocking = 0`;
- `NCCL_DEBUG=INFO` for the init, bootstrap and network subsystems.

## What this does not show

- A group shared by two sets. On this shape each plan's one set needs one group. Sharing is
  exercised by the unit tests.
- The head's own loss under load.
- A production-size model, more than one node per stage, and CoreWeave.
