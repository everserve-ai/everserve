# 0021: a lost rank is detected at the collective, not at its process's exit

2026-10-06. Status: current, as amended below (2026-10-14).

**Context.** The commit-path battery on the 72B shape (`bench/runs/2026-10-06-commit-path-72b/`)
held eight of nine predictions and failed the REPLACE pause: 6.8 s against a 4 s kill line, of
which 6.0 s was detection. The executor learns of a death by waiting on the worker process's
sentinel, which fires only when the process has exited, and a killed worker exits only after the
driver has torn down its CUDA context: a 72B shard, its KV and its pinned staging. On that host
(driver 570) the teardown took 6 s; on earlier hosts (driver 580) 0.55–0.80 s. Whether this host
is slower is not the point:

- detection by process exit is bounded by driver teardown, which we do not control and which
  grows with what the dead process held, and the frontier shapes hold more;
- detection by process exit never fires for a hung worker, and a hang is the failure class the
  runtime has been built to survive.

Everything after detection took about 0.8 s, half the frontier run's best.

**Decision.** The survivors detect a lost rank at the collective. The slot kernel already knows
which member's push has not arrived; after a bound it writes that member's row to a host-visible
word and keeps waiting, a watchdog thread on each worker reads the word and reports the rank lost
to the daemon through the same report the executor sends today, and the coordinator's change
proceeds as it does now, the executor's abort releasing the kernel at the boundary. The bound is a
setting in step periods, measured at startup, with a floor of tens of milliseconds, so a slow
step is never mistaken for a loss: the collective is only ever waiting on a member, and a member
silent for many step periods is gone or hung either way. The executor's sentinel stays as the
second signal, and a duplicate report of the same rank is one change (the coordinator already
folds them). The dead process's exit time is recorded separately in every fault run from now on.

**Consequences.** Detection takes tens of milliseconds on any driver, and a hung worker is
detected at all. The REPLACE pause on the 72B shape should fall to about 1 s, the post-detection
part the battery measured; the two-node pre-registration's fault bands assume this. A worker
reporting a peer lost is new authority for a worker, bounded by the daemon and the coordinator
exactly as the executor's report is. The NCCL path, used where the slot collective is not, keeps
its own abort bound and detects no faster; the slot collective is the product path. Needs a GPU
session to build and a REPLACE run to verify, scheduled before the probe-speed work.

**As built.** Built as decided, with two changes found while building.

- **Lateness is not loss: the late member's process decides.**
  - **Why.** A member can be late by far more than a few step periods and still be alive. In the
    commit-path battery, the first commit after a prefill held each rank's step thread for
    350–760 ms, by different amounts per rank, while the others waited in the kernel. The check on
    3× A6000 (`bench/runs/2026-10-07-detection-0021-72b/`) saw live ranks late by 149 and 200 ms against
    a 30 ms bound.
  - **How it works.** The watchdog checks the late member's process in `/proc` (each row records
    its process id in the region).
    - Exiting, killed, a zombie or gone: the member is dead, and is reported at once.
      `PF_EXITING` is set when the process starts to exit, before the driver's teardown, and
      `SIGKILL` is pending before that.
    - Alive: the lateness is recorded (`slot_watch.slow`). It is reported only past a hang bound
      (`EVERSERVE_HANG_S`, default 10 s), and the executor then kills the hung process so it
      cannot run beside its replacement.
  - **What it trades.** The bound can now be short (3 step periods, floor 30 ms, measured over
    steps 8–40) without a stall ever replacing a healthy rank. A hang is detected after the hang
    bound rather than after the step bound.
- **The coordinator did not fold duplicates; it does now.**
  - **What was wrong.** A second report of a rank in flight failed as an unknown rank. A report
    arriving after the commit would have started a second change, against the spare now standing
    in that position.
  - **The fix.** `RankLost` carries the epoch the reporter saw, and the coordinator keeps the
    epoch at which each rank last changed. Under the lock that claims epochs, it drops a rank
    changed since the report's epoch (`ChangeError::Stale`, counted as `folded`).
  - **The executor's side.** It reads the watchdogs' lost marks from the region while it waits on
    a step. Each mark carries its epoch, so a mark from before a change is never acted on after it.

- **Every frame from a worker says what it is.** The daemon used to read the frame after a
  `Change` as the worker's `Ack`. A survivor's report can arrive in that place: on 5× H100 and
  again on 5× A6000, a third survivor's report reached the daemon after it had sent the `Change`,
  was decoded as the `Ack`, broke the link, and failed the change. The `Ack` now travels as a
  `Signal` (`kind` `ack`). The daemon reads a `Signal` every time, gives an `Ack` to the change in
  flight, and forwards a report as a report. A report's epoch is the one its lateness began in,
  so a report sent after the worker applied the change still folds.

- **From the detection review.**
  - **The late word is a mask of every missing member.** A node's loss is several ranks at once,
    and must be one change for the set.
    - Every block's first thread times its own wait. Past the bound, it names the members whose
      words its block still lacks: exactly, for the batch it waits on, and by its first word for
      batches it has not reached.
    - Every block names for itself because a killed member's push can stop part way. A block
      that has its words finishes; one timing thread in block 0 would then name nobody, and the
      death would wait for the sentinel.
    - A device counter holds the blocks that have named. The last one to finish clears the word.
    - The watchdog reports every dead member of an episode in one report, and the lost mark
      carries the set.
  - **The hang clock does not run across a change.** From the change until this worker has ended
    two steps at the new epoch, a live late member is not reported hung. The replacement's first
    step waits on its restore, which took 3.1 s at 16K and grows with context. A dead member is
    still reported at once.
  - **Liveness from `/proc` assumes every member of a slot region is on this host.** That holds
    by construction, since the slot collective is scoped to one node's domain; `slotwatch.py`
    says so.

**On 3× A6000** (TP2, host-memory variant, driver 580): detected by the slot collective in 44 ms,
REPLACE pause 0.853 s, 8/8 sessions identical, no report in the no-fault run. The dead process
had finished exiting 2 ms after the kill on that host; its sentinel fired at 0.44 s.

**On 5× H100, 72B** (`bench/runs/2026-10-07-detection-0021-72b/`). The REPLACE kept every
session exact with a 1.250 s pause, but the death was detected by the sentinel at 0.364 s: that
host tore the process down in 0.365 s, and no survivor's slot kernel was ever late after the kill.
The survivors were waiting somewhere other than a slot all-reduce. Detection at the collective
covers only deaths that land in one.

**Where the survivors were** (`bench/runs/PLANNED/stall-stacks/`, 5× A6000, the survivors' Python
stacks 0.15 s after the kill).
- **The slot's coverage depends on the kill's point in the step.** Kills at 0.1, 0.3, 0.5 and 0.7
  of a step were detected by the slot in 43–46 ms. At 0.9 the sentinel found it, at 0.213 s.
- **At 0.9 the survivors had finished the step.** Every survivor's step thread was idle in vLLM's
  worker loop, waiting for the executor's next RPC. The executor was waiting for the dead rank's
  answer, which never came.
- **So a kill late in a step leaves no survivor in any collective.** The dead rank's queued GPU
  work completes the step's collectives after its process is killed. The next step is never sent,
  so no slot kernel waits on anyone.
- **It was not the logits all-gather,** which the 72B result had named as the likeliest place.

**The step-stall rule** (from the review of the H100 result). The slot's naming covers only a
death that lands in a slot all-reduce. So the watchdog also watches the engine's steps.
- **The rule.** While the executor has a step in flight and this worker has not ended a step
  within the slot's bound, the watchdog reads every member's process, at most every 10 ms. Any
  member that is exiting or killed is reported at once, the set together, through the same path.
- **What it covers.** A death wherever the survivors wait: in a collective the slot does not own,
  or, as the stacks showed, idle for a next step that the executor cannot send. No kernel change.
  A member alive but stuck in NCCL still falls to NCCL's own bound.
- **"In flight" is the executor's to say.** It is a count in the region of step RPCs
  (`execute_model`, `sample_tokens`) sent and not yet answered.
  - The worker's own begin and end of a step do not work as the signal. Its end comes when the
    forward is queued, before the logits are gathered and sampled. And in the case the stacks
    showed, every survivor has ended its step and waits for the next one: only the executor
    knows the step is unfinished.
  - At a change the count is reset to zero: the step a death cut short is never answered, and
    with two step RPCs out under async scheduling, one of them is never collected.
  - With nothing in flight nothing is scanned: an idle engine, or one shutting down while its
    workers exit. Without that gate, a worker exiting at shutdown would read as dead to every
    other worker's scan, and the coordinator would start a change in the middle of a shutdown.
- **Shutdown.** The executor also stops every worker's watch by RPC before it shuts the workers
  down.
- Both rules share the record of what has been reported at each epoch, so one death is one
  report from each worker.

**Amended** (2026-10-14). A serving worker that leaves its node's daemon outside any change, with
a goodbye or none, has lost its rank: the coordinator reports the loss as a worker's report would
be, folded with any other report of it. The twelfth GPU proof found the gap: under Dynamo the
stages meet over pipeline sends, not the slot collective, so no survivor's watchdog sees a stage
go, and a node pod deleted gracefully (every eviction, drain and spot reclaim under Kubernetes)
stopped its worker without a word, the engine stalled, and nothing replaced the rank. The
connection to the node's daemon is the coarser signal and the one every process has; the
collective stays the fast one where it runs. A rank a change already takes has left the members
when its worker departs, and a down engine takes no change, so neither is reported.
