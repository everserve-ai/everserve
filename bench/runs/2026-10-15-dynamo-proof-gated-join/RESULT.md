# Result: the gated blocking join on GPUs, a deletion then a crash, all eight predictions held (proof-25)

**Verdict: pass.** On 2026-10-15, `bench/harness/k8s/dynamo_proof.sh --losses 2 --loss-how
delete,crash` ran on 3 × H100 in eu-north1, on images `dev-9a40ce9`. That is `sg2-engine`: main
with the shared-groups pair and the bounded join Cort chose:
- a member reports itself ready, and the group's id is relayed only once every member is;
- the formation deadline;
- leaks reported and retired;
- the capture guard.

Rank 1's node pod was deleted, and then the spare that had taken it was killed with SIGKILL. Each
time a REPLACE committed through a covered set, and all 16 streams finished with every token.

The run's output is `run/`. The read-only captures the peer asked for, for proof-24's harness
fixtures, are in `run/captures/`: the head's and a node pod's process tables, and the pods before
and after the losses.

The frontend ran with `DYN_MIGRATION_LIMIT=3`, since the renderer at `b508a85` sets it. No head
was lost here, so no stream migrated.

## The predictions

Read against the [pre-registration](PREREGISTRATION.md).

| # | Prediction | Read | Verdict |
|---|---|---|---|
| 1 | Both losses REPLACE rank 1 through a covered set, at epochs 1 and 2, every stream finishing | epoch 1 through `0-R1@…jh0j1…:0`, epoch 2 through `0-R1@…gdq11…:0`; 8 of 8 streams each, no error | **pass** |
| 2 | Loss 1 reaches the coordinator as a notice; worst gap at most 1.0 s | "notice for rank 1" at 14:08:19; worst gap 0.67 s | **pass** |
| 3 | Loss 2: detection at most 0.5 s; worst gap at most 2.0 s | kill at 14:09:41.305 (pod clock), departure line at 14:09:41.306, about 1 ms; worst gap 1.328 s | **pass** |
| 4 | Covered True before each loss | "Covered True (EveryLossCovered)", before both | **pass** |
| 5 | Every plan's groups form through the gate, with no GroupFailed | plans 2 and 4: "1 of 1 groups formed"; no GroupFailed in the events or any log. The ordering is shown by the formations completing (below). | **pass** |
| 6 | Grove's pod in place of loss 1's covers loss 2; the pod crashed in loss 2 starts again in place and stands as a spare | loss 2's set is the pod on loss 1's node (`gdq11`); `k4djc` restarted in place (1 restart) | **pass** |
| 7 | The capture guard is on, the engine starts, and #2's and #3's bounds hold | every serving worker logged "NCCL handles guarded against a graph's capture: _PP" at 14:07:13, before its warm-up; the engine started and served | **pass** |
| 8 | Nothing left behind: no ReleaseLeaked, no stale exit (4) | no ReleaseLeaked in the events or any log; the restarted container's last exit was 247 (its node process ending after its worker's SIGKILL, as in proofs 22 and 23), not 4 | **pass** |

**#5's ordering.** The coordinator does not log the gate's relay at info level, so "the id is
relayed only after every member's GroupReady" is shown another way. Under the gate a group's
server joins only on its own id coming back. The coordinator before the gate never sent a server
its own id, so every formation would have timed out there. Both plans' groups formed, so the
relay ran as gated.

**#7's spares.** The spares also logged their guard before their warm-up, as "none": a spare holds
no vLLM NCCL handle until it takes a position, when `point_pp_group_at` puts the guard on the
handle it is given. Its warm-up's captures could reach no NCCL communicator.

## What this does not show

- The window the gate leaves, a member lost after the go and before its join. Its bound (the
  formation deadline, the leak report, the retirement) is covered by unit and coordinator tests.
  Proof-25b is to show it with a frozen member.
- A group shared by two sets: on this shape each plan's one set needs one group.
- The head's own loss (proof-24), a production-size model, and CoreWeave.
