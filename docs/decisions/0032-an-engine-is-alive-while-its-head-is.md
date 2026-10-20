# 0032: an engine is alive while its head is

2026-10-10. Status: current, as amended below (2026-10-14); built.

**Context.** The coordinator knows an engine through the processes that register with their
node's daemon: its workers and its spares. The engine's head (the process that owns the
scheduler and the executor; vLLM's engine core) does not register. On the p3-resident rental's
node fault trial 2 (`bench/runs/PLANNED/p3-resident/run1`), node-b was lost; rank 0's worker
acknowledged the change; the head then shut the engine down on a pipeline refusal it read as an
ordinary failure (fixed in the adapter, `6db3808`); and 11 s later, when the spare's take
finished, the coordinator committed the change over an engine with no process left, and reported
it committed. Two gaps, one of them the coordinator's alone:

1. A commit trusted the acknowledgements and never asked whether the survivors that gave them
   were still there.
2. Nothing tells the control plane the head is gone, so an engine whose head exits between changes
   reads as serving until its workers' departures thin the view, and an operator's stacks and stop
   cannot reach it (`docs/PRODUCT-ARCHITECTURE.md`, the operator reach row).

**Decision.**

- *A change commits only over survivors still present.* Once the last acknowledgement is in, the
  coordinator checks that every survivor (an active rank the change does not replace) still has a
  registered process; if one left, the change fails (`ChangeError::Departed`, counted as
  `survivor_departed`) and is undone as a failed delivery is: the staging released and the spare's
  assignment dropped. Built.
- *The head registers.* The executor's daemon connection, which the head already opens before its
  workers start, registers as the head rather than as a client; drivers and tools stay clients,
  and their loss means nothing. One connection, so the head's loss and its link's loss are one
  event. It carries two pids: the engine core's, for stacks, and the launcher's (the process the
  operator started, the frontend's when the core runs in its own process, the same pid when it
  does not), which stacks reach too and an operator's stop of the engine is to be sent to (the
  operator API has no engine stop yet; designed, not built). The coordinator holds the head beside
  the engine's ranks, readiness reports it (`head`), and a stacks request names it
  (`{"head": true}`). Protocol 4: `ROLE_HEAD`, `Register.launcher_pid`, `Goodbye`.
- *The head's loss is the engine's.* The engine is down when the head's connection ends. A goodbye
  carries a cause, stopped or failed, with its detail, because the engine core shuts down in order
  on its error path too (trial 2 did): the executor says failed when it raised a failure that was
  not a loss, and stopped otherwise. Both mean down; only readiness's cause differs, and an error
  the scheduler raises outside the executor reads stopped, a residue named rather than closed by
  pinning the engine core's exit. On the engine down, a change in flight fails as above
  (`ChangeError::Down`), no change starts for it, readiness says down with the time and the cause
  (`down`), and `everserve_engine_down` reads 1. The daemon client has no reconnect, so a daemon
  restart under a live head is the engine down too; the engine is back when a relaunched head
  registers, at the coordinator's next epoch.
- *A change that fails after its delivery puts the engine down.* Once a change is delivered,
  processes have applied it: survivors refused their old groups, a spare took a position. If it
  then fails (a survivor's error, a survivor departed, a node's delivery failing midway, the head
  gone), that membership never commits, and before this decision the coordinator's epoch also
  stayed ahead of every daemon's, so every later change was refused and the engine was stuck
  without saying so. Now the engine is down (`change_failed`, with the error, unless the head's
  departure already said why), and the coordinator sends `Abandon` to every node with a process
  of the engine. The daemon takes the epoch as spent, forgets which spare was taking which rank,
  and forwards `Abandoned` to every process of the engine on its node: the head fails the engine,
  a spare that took a position for that epoch exits so its supervisor launches it standing again,
  and a worker only logs it, since the head's shutdown ends it.

**Built.** The coordinator and daemon halves and protocol 4; the adapter's half (the executor
registering as the head, its goodbye from `shutdown()`, and the head's and a taking spare's
reactions to `Abandoned`) lands beside them.

**Consequences.** A committed change means its survivors were present at the commit, not only
that they answered. A failed change is a down engine with its cause, never a stuck one. A spare's
take that outlives its engine is wasted work, not a false success. An adapter that does not
register a head gets the survivors check and the abandon, and its engine is never marked down by
a head's loss.

## Amendment (2026-10-14): a downed engine starts anew

Found on the first GPU proof under Dynamo: the head failed, the engine was down, Grove restarted
the head's pod, and every restart was told to stand as a spare for ranks that had had a holder,
so the engine never came back. An engine that is down and holds no serving process of its old
incarnation starts anew when a process of it asks what it is: its framework restarted it. The
coordinator records `EngineRestarted`, forgets the old incarnation's departed and lost ranks, its
change and its assignments, and decides the asker as at bring-up. While a process of the old
incarnation still serves, askers stand and nothing restarts, so two incarnations never mix; a
process of a downed engine ends on hearing it (`InstanceState` failed, decision 0041), so the
restart follows the last one's departure. Epochs keep rising across the restart, so no daemon
sees one go back. Built (2026-10-14), tested by a downed engine whose last worker leaves and
whose next asker is told to serve, and one that is told to stand while its old worker remains.

