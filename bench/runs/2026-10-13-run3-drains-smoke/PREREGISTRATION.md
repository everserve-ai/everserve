# Pre-registration: run 3, a smoke of what landed untested (2026-10-12)

Status: PLANNED 2026-10-12, written before any node is rented and before the harness additions it
needs (below) are built. Nothing below is changed after the first step runs; an amendment before
it is dated and says why. The rental is booked only on Cort's word, typed in the agent's session.

## Why this run exists

Three changes are on main, or held for this run, that no test has run where they matter. Each has
unit tests, but the part that only a real engine on real nodes can show has never run.

1. **The head fix** (branch `head-role-held`, held until this run; its hash at deploy is
   recorded). In run 2 (2026-10-10) node-b's node role registered as a second head of the engine
   (node-b's daemon log: `head of rig registered` at 18:24:17, 18:28:35 and 18:32:43, as its node
   role started). Since then the coordinator refuses a second head and counts it
   (`everserve_head_conflicts_total`). The fix makes a non-head node register as a client. Its
   test needs vLLM, so it has only ever skipped.
2. **Released workers** (`c5f6c5b`, decision 0033 step 3, item a). A worker moved off its node by
   a notice drains, closes its daemon link, marks its row released, and is ended by its node. A
   node whose positions all moved elsewhere ends. Its two `NodeExecutor` tests need vLLM and have
   only ever skipped.
3. **The Node resource** (`f348bb9`, decision 0036 step 2). Cordon and uncordon, drains written
   to the spec and sequenced under the policy's concurrent losses, a second drain of a scheduled
   node refused, and a stored drain resumed after a coordinator restart.

This is a smoke, not a battery: it reads each mechanism once, in order, on the cheapest shape that
holds all three.

## Shape

Nebius eu-north1, `gpu-l40s-a 1gpu-8vcpu-32gb`. Five VMs of one GPU each, node-a to node-e.
Qwen/Qwen3-1.7B, one engine `rig`, TP1 x PP3: stage 0 on node-a (the head and the coordinator),
stage 1 on node-b, stage 2 on node-c. Node-d holds a spare preloaded for rank 1 and node-e one
for rank 2 (`policy.py --members a:1 b:1 c:1 --spares d:1 e:1`). (Amended 2026-10-12, before
any step: the plan had node-d on a 2-GPU VM, but the region's only 2-GPU L40S preset,
`gpu-l40s-d 2gpu-64vcpu-384gb`, failed to start four times on 2026-10-08 for capacity, while
the 1-GPU preset has always come up. Same GPUs, same cost.) The policy's
concurrent losses is 1, one node's worth (`maxConcurrentLosses: 1`). The coordinator runs with a
file spec store (`--spec-store`). Four sessions of 512-token prompts generating 16,000 tokens,
enough decode steps to keep the engine under load through both drains and the restart (about
two minutes from the first drain to the second commit; see attempt 2 below). Every node runs one commit, the deploy's, recorded
with its hash: the head fix rebased onto main, with the harness additions below. Main must
carry the controller owner's landing after `f348bb9` first: it starts waiting drains in deadline
order after a coordinator restart (on `f348bb9` they start in the order the daemons say hello
again, which can flip node-b's turn to node-c's), it adds the blocked-drain state, and it adds
conditions and an event log this run reads (amended 2026-10-12, before any step, from the
controller owner's review).

## The harness (built before the deploy)

Amended 2026-10-12, before any step, to name what was built (on the harness branch, gated and
its commit recorded before the deploy):

- `nodes.sh`: `SERVING` names the nodes the engine spans, each started in its node role at its
  rank; `drain NODE LEAD_MS` for any node; `cordon`, `uncordon` and `specs` (GET `/v1/nodes`);
  `coord_restart` (the coordinator killed with SIGKILL, as a crash leaves it, then started again
  on the same lease and spec store); `SPEC_STORE`; the node role's exit status written to its log
  (`==== node_main exited <status>`); the node-a guards naming every test they ran (`-rA`).
- `policy.py --max-concurrent-losses`.
- The drill's `follow` arm: it completes every change the coordinator commits and sends or kills
  nothing.
- `bench/harness/twonode/drains.sh`, the driver of the steps below, keeping every API answer in
  `reads/` and every action in `actions.jsonl`. `drains_verdict.py` scores each prediction below
  from that record into `verdict.json`; its tests show a record of a run that went as predicted
  passing every line, and a record breaking each prediction failing that line and no other.
- Setup runs `test_node.py` on node-a with the pin guards, and fails on a failure or on any skip.
- The engine's and the nodes' conditions and event logs from the controller owner's landing are
  recorded at every read (readiness `conditions`, GET `/v1/engines/rig/events`, `/v1/nodes`
  `conditions`, GET `/v1/nodes/{node}/events`). They are reported and not scored, except the
  engine's `Covered` condition in N1.

## Attempts so far, and the amendment before attempt 3

Amended 2026-10-12 after attempt 2 and before attempt 3. No prediction below was read in either
attempt, and none is changed except where this section says.

- **Attempt 1** (17:12 to 17:16 UTC) never reached setup: the tenant's public IPv4 quota was 3,
  so node-c's and node-d's creates failed (ResourceExhausted, not retryable). Cort raised it to 10.
- **Attempt 2** (17:26 to 17:52 UTC) set up all five nodes on `3767782`; the node-a guards passed,
  17 of 17 and none skipped, both head tests among them, so the head fix went to main. The engine
  served, but the coordinator refused both spares for their compile caches (decision 0029): neither
  spare node had the serving nodes' caches staged, so each spare compiled its own. Eight sessions
  of 7,000 tokens ended in 80 s at about 1,250 tokens/s, before any drain could be asked. The VMs
  were released at 17:52 without the nodes' logs collected; the deploy, driver, drill and guard
  records are in `run/`.
- **For attempt 3:** the engine starts once without spares, so node-b and node-c compile under the
  run's configuration; node-b's cache is staged onto node-d and node-c's onto node-e; then the
  control plane starts again with the spares. A refused spare stops the run at once, with the
  refusal logged. Four sessions of 16,000 tokens; node-b's lead 20 s and node-c's 60 s. Every node's
  logs are collected on every exit. R1, R3 and R4 read the node-role log after its last launch,
  the run's own start, so the warm-up's stop is not read as the run's; N4 reads the restart as the
  coordinator's last launch, after the warm-up's and the run's.
- **Attempt 3** (18:10 to 18:56 UTC, on `60a1ee2`). Both spares were admitted on the staged caches.
  Node-b's drain moved rank 1 to node-d: notice 18:27:17.9, commit at epoch 1 at 18:27:38.3, on its
  deadline, the engine serving. Node-c's drain waited its turn and was released at that commit.
  The repeated drain of node-b was refused (409). Heads were only on node-a (H1 to H3 read green).
  Node-c's change then failed on every member, `KeyError: (0, 'R1', 'R2')`: an engine's sets are
  formed only at its start, and the spare that took rank 1 keeps its id `R1`, so an engine covers
  one transition. That is decision 0035's limit, whose process half is not built: no two changes
  on one engine can pass before it. The engine went down at 18:28:19. Also read: the released
  worker on node-b ignored its death pipe and then SIGTERM and was killed at 9.2 s; node-b's
  daemon never received the commit (it held no rank after it), so its node role never recorded
  rank 1 as left nor ended; the spares gauge stayed `warm` 2 while both spare nodes were cordoned.
  Logs collected in `run/collect/`.
- **For attempt 4** (amended 2026-10-12 after attempt 3, before attempt 4; no prediction read
  beyond what attempt 3 records): it runs only once every fix below is on main and a full pass is
  expected. The fixes: decision 0035's process half (online sets, ids renamed at commit); the
  commit delivered to every daemon with a process of the engine; the spares gauge following a
  cordon; a released worker killed at once (`b1edd9d`, since vLLM's graceful end destroys
  communicators whose peers moved on and, after the death pipe, ignores SIGTERM). R2 now predicts
  `killed`; N2 reads node-c's notice from the coordinator, the one record that always has it; N5
  also reads the coordinator's log once the run serves, since a restarted coordinator does not see
  an engine whose head is gone.
- **Cover planned, for attempt 4** (amended 2026-10-12, before attempt 4, from 0035's built halves):
  the policy has the coordinator plan every pre-formed group (`cover.enabled`, `policy.py
  --cover`). With TP1 each set is one fabric group over the three stages, served by rank 0; rank
  0 has no spare, so its loss is never covered and the Covered condition stays `False`
  (`LossUncovered`) throughout, its message listing the losses without a formed set by ranks. A
  cordon takes the cordoned spares' sets out of the plan, so their groups are released, and the
  uncordon's plan forms them again: the sessions start only once both losses are covered again
  (the drill's `--start-after`), so every change runs under load. After node-b's commit, node-c's
  change can go only through the epoch-1 plan's set for rank 2, formed with rank 1's new holder
  (0035's REPLACE gate), so node-c's lead is 120 s again and the record times that set's cover
  against node-c's deadline. If it is not covered by then, node-c's drain is blocked, which C2
  fails and the record keeps.
- **Attempt 4** (19:50 to 20:12 UTC, on `a238104`, about $3.70). The planned cover formed on
  the GPUs beside serving: both losses covered at 20:06:32, released by the cordon and covered again
  at 20:07:34. The sessions started and the drains were asked; node-b's change then failed on rank
  0, `KeyError: (0, 'R1', 2)`: the change path composed the replacement set with the plain spare id
  while the plan had formed `(0, R1@node-d:0, 2)`, and on a stage of one there was no stage entry
  for the set at all. Fixed before attempt 5 (the held set found by its spare, the stage's
  stand-in held with a plan's pipeline group); no prediction is changed. Attempt 5's cap is $9.50,
  under the balance left.
- **Attempt 5** (20:26 to 21:09 UTC, on `04208a1` then `4c361fb` on the same VMs, about $7.20;
  record in `run/attempt5/` and `run/attempt5b/`). On `04208a1` rank 0 found and staged the
  spare's set, then a plan delivered between the change's acknowledgements and the boundary
  dropped it, and the boundary's swap failed. On `4c361fb` (a plan during a change applied at its
  commit, the staged set served on), rerun as 5b on the held VMs: node-b's drain committed at
  epoch 1 at 21:03:15 on its deadline, and the verdict read H1 to H3, N1 to N3, N5 and C1 green.
  Node-c's change was then refused by the cover gate (C2 failed): the epoch-1 plan gave rank 2's
  set the old group's port, still bound by rank 0's rendezvous store (`EADDRINUSE`), so it never
  formed. Node-b's node role heard the commit but not the claimed change, so it never recorded rank
  1 as left (R4). Fixed after it: a released group's store closed before its release is reported;
  a node takes a commit's lost ranks from the commit; the planner keeps a port only for a group of
  unchanged holders, with a 60 s quarantine on a freed port; N5 no longer counts a head that
  stopped in order at the run's end. Run 3's balance after attempt 5: about $3.50.
- **Attempt 6** (22:31 to 22:54 UTC, on `df65e91`, about $3.90; record in `run/attempt6/`).
  Node-b's change committed at epoch 1, the epoch-1 plan formed rank 2's set again on a fresh port,
  and the cover gate let node-c's change through. It failed on node-e's spare, which was outside
  the epoch-1 membership: its runtime counted its own epoch and staged epoch 1 for the change at
  epoch 2, and the coordinator refused the acknowledgement. Fixed before attempt 7: a change moves
  every process to the change's own epoch. Attempt 7's cap is $7, under the balance left.
- **Attempt 7** (23:07 to 23:47 UTC, on `5f973a4`, about $6.60; what survived in `run/attempt7/`,
  read in `OBSERVED.md`). Both changes committed, node-b's at epoch 1 and node-c's at epoch 2,
  and the four sessions ran 16,000 of 16,000 tokens through them, the longest gap 0.58 s. Not
  scored: the budget's cap released the VMs before the driver collected, so the nodes' logs and
  the drill's result were lost. Three faults, fixed before attempt 8: rank 1's spare reported its
  live pipeline peers lost when a change refused its send (MemberSetChanged), and the coordinator
  refused that report's change; the driver waited for node-b's notice in node-b's daemon log,
  which does not log it, so the restart came after both commits (N4 not exercised, C2 read after
  node-c's commit); and the driver's waits counted polls, not seconds. Step 4 is amended: the
  restart is timed from the coordinator's log line for node-b's first notice, and runs beside
  the other drains' calls.
- **Attempt 8** (00:25 to 00:50 UTC 2026-10-13, on `97a8240`, about $4.20; record in
  `run/attempt8/`, verdict `fail ['R1', 'R2', 'R4']`). H1 to H3, R3, R5, N1 to N5 and C1, C2
  green: the restart came inside node-b's lead, the restarted coordinator read both drains from
  its spec store, re-sent node-b's notice with its deadline and committed 0.2 s after it, then
  node-c's at epoch 2; the four sessions ran 16,000 of 16,000 tokens. Red: node-b's worker was
  released at its commit but node-c's never was (R1, R2), and neither drained node's process
  exited (R4). Read from the record: a drained node hears only the commit for its own rank's
  change, whose `lost` names the ranks shrunk away and none replaced, so it took nothing and
  never held nothing; the head marked each departing rank dead on the claimed change, so the
  departures ran as faults, their drains not awaited; and node-c's worker never reached the
  boundary that releases it. No stack of it was taken. Fixed before attempt 9.
- **Attempt 9** (01:36 to 02:38 UTC 2026-10-13, about $10.20; the driver's reads in
  `run/attempt9/`, not scored). On `c495a5e` node-a's setup guard failed on a test fixture
  (f56bd9f); rerun on the same VMs, node-b's change failed for want of rank 1's acknowledgement:
  node-d's spare, preloaded for rank 1 and so carrying rank 1 before its take, read the
  departing change as its own and neither applied nor acknowledged it. Fixed (`453bbee`, a
  standing spare is never the departing process) and rerun on the same VMs: the restart inside
  node-b's lead, node-b committed at epoch 1 in 1.47 s, rank 2's set covered at epoch 1, node-c
  committed at epoch 2 in 0.54 s, the engine never down, both drained nodes `DrainDone`, the
  membership 0 on node-a, 1 on node-d, 2 on node-e. Then each VM's 60-minute watchdog, counted
  from its creation and run down by the two reruns, deleted the VMs at 02:37, a minute before
  the sessions' end and before the collect: R1, R2, R4 and R5 have no record. The next attempt
  runs once, from a fresh rental.
- **Attempt 10** (03:06 to 03:48 UTC 2026-10-13, on `d10a00e`, about $7; record in
  `run/attempt10/`, not scored: the driver was stopped once both changes were refused). Setup,
  warm-up, cover and the cordon round as predicted; the sessions ran to their end. Neither drain
  committed. The coordinator, restarted at 03:24:43 inside node-b's lead, planned the cover again
  as each daemon relinked (plans 6, 7 and 8 at epoch 0, each with new ports), and node-b's
  deadline came 0.52 s later: the REPLACE gate refused its change, `set 0-R1@node-d:0-2 is planned
  and not yet formed`, and node-c's at 03:26:28 likewise. The sets never formed again: a process's
  cover took plan 8's entry for a set it was already forming at plan 7's port, whose peers had
  moved on, and waited the store's 300 s. Fixed before attempt 11: the coordinator adopts the
  plan the daemons report after a restart and plans nothing until they have relinked, a drain the
  gate refused is retried once its set is covered, and a process forms each planned entry,
  rendezvous included, on a thread of its own.
- **Attempt 11** (04:17 to 04:41 UTC 2026-10-13, on `4df4aac`, about $4; record in
  `run/attempt11/`, verdict **`pass []`**: every prediction, H1 to H3, R1 to R5, N1 to N5, C1 and
  C2, green). The coordinator, restarted inside node-b's lead, adopted cover plan 6 from node-a's
  daemon, all five daemons relinked within 5 ms, and no plan was issued until the commit;
  node-b's change committed at epoch 1 1.7 s past its deadline, rank 2's set was covered again at
  epoch 1, and node-c's committed at epoch 2. Each drained worker was released at its change and
  ended by its node in 0.16 s, both drained nodes held nothing and exited 0, and the four sessions
  ran 16,000 of 16,000 tokens through both changes.

## Steps, in this order

Amended 2026-10-12, before any step: the engine is the drill, so the sessions start with it, and
steps 2 to 4 run while they do.

1. **Setup and guards:** setup on every node with the node-a guards. Stop here if a guard fails or
   skips. Then the warm-up start without spares, the compile caches staged b to d and c to e, the
   control plane again with the spares, the spec store empty, and the engine with its sessions.
2. **Registrations:** once both spares stand and the sets for ranks 1's and 2's losses are
   covered, read readiness, `/v1/nodes`, the event logs and the coordinator's metrics.
3. **Cordon:** cordon node-d and node-e and read them again; uncordon both and read them again.
   No drain runs while they are cordoned: with REPLACE out of reach, this policy's next outcome is REBUILD,
   which this smoke does not exercise. Wait for both losses covered again; read. Then the
   sessions start, and the drains wait for the drill's step 50.
4. **Two drains, sequenced, with a coordinator restart:** drain node-b (`lead_ms` 20,000), then
   node-c at once (`lead_ms` 120,000: its notices go out only when node-b's change ends, and its
   change needs the epoch-1 plan's set for rank 2 covered first), then node-b again with a
   lead 60 s longer; read. 10 s after the coordinator logs node-b's first notice (amended after attempt 7), kill the
   coordinator (SIGKILL) and start it again; read. Once node-b's change commits, wait for rank
   2's set covered at epoch 1, its time recorded; read. Wait for both changes; read. Then the sessions'
   end; read, and list node-b's and node-c's engine processes.
5. **Collect and release:** every node's logs and the drill's result, then the VMs released, then
   the verdict.

## Predictions: each a pass/fail line, with the record that decides it

**Head fix**

H1. **The guards ran:** `pin_guards_a.txt` shows `test_node.py`'s tests passed, among them
`test_only_the_head_registers_as_the_engine_s_head_and_says_goodbye` for both executors,
`test_a_change_moves_the_spare_s_row_in_and_clears_the_rows_that_left` and
`test_a_position_moved_to_another_node_has_left_not_been_taken_here`, with 0 skipped. **Fail:**
any failure or skip (setup stops).

H2. **Heads only on node-a:** node-a's daemon log has a `head of rig registered` for each start
(the warm-up's and the run's). Node-b's and node-c's have `client of rig registered` and no
`head of rig` line, for the whole run. **Fail:** any head
line on node-b or node-c.

H3. **No conflict:** readiness `head_conflict` is null at step 2 and at step 5, and
`everserve_head_conflicts_total` is 0 in the coordinator's metrics at step 5. **Fail:** non-null,
or above 0.

**Released workers (item a)**, read twice: node-b's worker at its drain and node-c's at its

R1. **Released:** each drained node's node-role log has `released worker <rank> ended: {...}`
for its rank (amended 2026-10-12, before any step: the head does not read a remote leaving rank's
boundary answer, so the answer's `released: true` reaches no record; the node's line is the
record, and the release can only follow the drain and the commit, which the unit tests order).
**Fail:** the line absent on either node, or its record `overdue` (amended after attempt 8:
the node ends a worker whose position left `hang_s` before unreleased, which skips its drain).

R2. **Killed at once:** that line's record has `how: killed`, `end_s` under 2. Its process is
absent from the node's process list at step 5. **Fail:** another `how`, a longer end, or the
process alive at step 5.

R3. **Its exit is not a loss:** no `worker <rank> is exiting` line in its node's node-role log, and readiness shows rank 1's member on
node-d and rank 2's on node-e at step 5. **Fail:** any such report, or the member missing or
on the drained node.

R4. **The drained node ends:** node-b's and node-c's node-role log has `this node holds
nothing of rig` and `==== node_main exited 0`. **Fail:** the line absent, a nonzero exit, or the process
alive at step 5.

R5. **Sessions continue:** all four sessions reach 16,000 tokens across both changes, the drill
records both changes, and it exits 0. **Fail:** a short session, a change missing, or another
exit. Token-for-token agreement with a no-fault
control is reported, not scored: this smoke runs no control.

**The Node resource**

N1. **Cordon shows, and is undone:** each cordon POST's 200 body (node-d's and node-e's) has
`unschedulable: true`, `written_by: "operator api"` and the reason given. Then:
- `/v1/nodes` shows node-d and node-e `cordoned: true`;
- readiness's `spares[]` entries on both have `state: "cordoned"`;
- readiness's `nodes[]` entries for node-b and node-c show `outcome: "rebuild"` and
  `reclaimable: false` (REPLACE out of reach);
- the coordinator's `everserve_spares{engine="rig",state="warm"}` reads 0, from 2 before;
- the engine's `Covered` condition is `False` with reason `LossUncovered`, its message listing
  the losses of ranks 1 and 2 with rank 0's (amended before attempt 4, for the planned cover).

After the uncordons, each POST body shows `unschedulable: false` at generation 2, and every read is back
as before (`warm` 2), and once the uncordon's plan has formed, `Covered`'s message lists only
rank 0's loss, `[[0]]`. **Fail:** any read not shown while cordoned, or not restored after.

N2. **The second drain waits its turn:** node-c's drain response has `waiting: true`, and the
coordinator logs `node-c: its drain waits: 2 positions would move at once, the policy allows 1`.
While node-b's change is in flight, `/v1/nodes` shows node-b `drain: "scheduled"` and node-c
`drain: "waiting"`, both `cordoned: true` (a drain cordons its node). Node-c's notices go out only
after node-b's change commits. The coordinator's `rig: drain of node-c releases ranks [2] at <ms>
ms` and its `rig: notice for rank 2` are both no earlier than its
`rig: membership committed at epoch` line whose `lost` is `[1]` (clocks against node-a's, as
recorded). **Fail:** `waiting` false, node-c's state not `waiting`, or node-c's notice before
node-b's commit.

N3. **A drain under way cannot be moved:** the repeated drain of node-b with another deadline
returns HTTP 409 with `its drain is underway (the notices went out); it cannot be withdrawn or
moved`, and `/v1/nodes` still shows node-b's first deadline. **Fail:** any other status, or the
deadline changed.

N4. **Stored drains survive a restart, in order:** the restarted coordinator logs `spec store:
file <path>, N resources` and reads both drains as waiting. Then:
- node-b's drain resumes first when its daemon says hello again: its notices are re-sent with the
  same deadline, and its change commits at that deadline, not later;
- node-c's drain logs `node-c: its drain waits` again, then runs after node-b's commit and
  commits;
- no drain call is repeated.

**Fail:** either drain lost, node-c scheduled before node-b, node-b's change committing after its
deadline plus the change's own time, or a change that never commits.

N5. **The engine stays up:** no change fails and the engine is never reported down
(`everserve_engine_down` 0; readiness without `down`; no `rig: down:` line in the coordinator's
log once the run serves), restart included. **Fail:** any.

**Cover** (amended 2026-10-12, before attempt 4)

C1. **Both losses covered before the drains:** readiness's `cover.sets` has the sets for `REPLACE
of ranks [1]` and `[2]` covered at the registrations' read and again after the uncordon.
**Fail:** either not covered at either read.

C2. **Rank 2's set covered again before node-c's deadline:** after node-b's commit, the
`covered_after_b` read has `cover.epoch` 1 or later with rank 2's set covered, and its time is
before node-c's deadline (its drain response's `deadline_ms`). **Fail:** not covered, or covered
after the deadline.

**Reported, not scored:** each change's pause and decision latency; the restart's time to the
coordinator's serving again; the release's `end_s`; the conditions and event logs at every read; whether the sessions were
still decoding at each commit; the spend at each
step.

## Cost

Five L40S GPUs (five 1-GPU VMs). Amended 2026-10-12, before booking: Cort read the L40S rate as
$2.00 per GPU-hour or less, and his balance is $17. He set the run at most an hour. So:

- **Time cap:** a watchdog releases every VM 60 minutes after the first create, whatever is
  running.
- **Money cap:** `nebius.py budget` reckons the spend at $2.00 per GPU-hour ($10 an hour for five
  GPUs) and releases every VM at $11, past the hour's $10 only if the time cap failed.
- **The estimate:** setup 15 min, guards and control plane 5, the engine to step 50 and the reads
  5, the drains under load with the restart about 5, the sessions' end, collect and release 10:
  about 40 to 45 min, about **$7.50** at $2.00. The bill is read afterwards and written into the
  record.

## What this run does not claim

- Exactness of the sessions through a planned move (no control runs).
- The head's node moving (decision 0033 step 4), online set formation, a returned node taking
  positions back, a software change: none is built.
- Any figure at frontier size, or a fault (kill) path: run 2 measured those, and nothing here
  changes them.
- Any figure leaves the repository only after its record is committed.
