# 0035: cover plans

2026-10-12. Status: current, as amended below (2026-10-13, 2026-10-15); built, the shared-groups
amendment included (protocol 12, proved in `bench/runs/2026-10-15-dynamo-proof-shared-groups`).

**Context.** A REPLACE is a pointer swap to a group pre-formed for the new member set (0005). Every
such set is planned once, when the engine starts:
- the static policy names them (`policy.sets_to_preform`);
- each process computes the same list (`worker.bootstrap_from_policy`);
- a set's rendezvous port is its index in that list (`shadow.rendezvous_plan`).
Nothing plans, forms or releases a set after that. Three consequences:

1. An engine covers exactly one transition. After a take there is no set for a second loss. The
   spare also keeps its replacement id (`R<r>`, `Collective.take`) for life, so the live member
   set's key depends on the engine's history. The batteries hid this, since they restart the
   engine between trials.
2. A spare that arrives later can't be covered. That covers a recovered node, a node returned
   from an upgrade (0033 step 3) and a spare lent from the pool (0031). The positions it could
   take have no set naming it.
3. Formation is built for a process that isn't serving yet. `form_shadow_groups` synchronizes
   the device after each group. vLLM's NCCL wrapper runs its warm-up all-reduce on the current
   stream during init (pin `6e517b15`, `pynccl.py:184-191`).

vLLM's elastic expert parallelism solves a neighbour of this, read at the pin. It forms standby
groups over a fresh store on a background thread while serving, and swaps at its commit
(`standby_state.create_standby_groups`, `elastic_execute.prepare_reconfiguration`). It does this
once per resize, for data- and expert-parallel groups only, with no fence and no abort. A death
during its prepare shuts the deployment down (0033).

**Decision.** The sets an engine holds are a cover plan. The coordinator computes it, every
process forms its part in the background, and acknowledgements make coverage a fact the planner
reads.

- *One plan, one owner.*
  - `CoverPlan{engine, generation, epoch, placement, sets}`:
    - `placement` is each rank's holding process at the committed `epoch`;
    - each set names its full member set, its scope (domain or fabric), the members its group
      spans, and the host and port it rendezvouses on.
  - The coordinator computes it. The planner is layer 6 (0034), and coverage gates layer 5's
    REPLACE. It holds the placement, the spares and their eligibility (0030), and the ports. The
    policy is its input, and `policy.sets_to_preform` moves into it.
  - The plan is delivered over the daemon link to every process of the engine, workers and
    spares alike. A process that registers while a plan stands receives the current plan with
    its registration's answer, not only at the next generation.
  - Bootstrap is generation 0 of the same plan, so there is one path.
  - `generation` is separate from `epoch`. A spare's arrival or departure changes the plan with
    no membership change, and a commit changes the epoch and then the plan.
  - The plan survives the coordinator. A restarted coordinator, or a standby taking over (0014),
    first hears each daemon's hello. The hello carries, for every process on the node, the sets
    it holds formed with their generations, and the highest generation the daemon saw. The new
    coordinator numbers its plans above that highest generation. It treats the reported sets as
    the engine's cover until its first plan supersedes them. A failover never resets the
    generation to 0, which would make every process abort its formations and form again.
    Amended 2026-10-13, after run 3's attempt 10 (`bench/runs/2026-10-13-run3-drains-smoke`): the hello also carries the last plan the daemon
    forwarded, ports included, and the new coordinator adopts the newest one any daemon reports,
    with its sets. It plans only once the daemons its policy and that plan name have relinked, or
    ten seconds after it started, and its first plan then keeps every group whose processes are
    unchanged, ports and all, so it delivers nothing. Planned from the first daemons to relink,
    each relink delivered another plan with other ports, the processes' cover was undone, and a
    drain due 0.5 s later was refused as uncovered.
- *A cover budget.* Cover costs world × spares. A set exists per pair of a coverable loss and a
  spare that can take it, and every member holds each set's NCCL state: about 66 MiB per group per
  GPU in harness week five (`groups.py`; not yet read on H100). At 16 positions and two
  any-rank spares that is 32 sets, about 2 GB per GPU for one scope. A set covered in both
  scopes holds two groups, a domain one and a fabric one, so a set's cost is the pair's.
  - The policy gives a cover budget, in sets or bytes per GPU, counted over both scopes.
  - The planner covers within it in order: whole-node losses first, then single positions, then
    a second spare for a loss already covered.
  - Readiness reports each engine's covered, missing and over-budget losses.
  - The budget's default comes from the per-set bytes read on the next rental, not from this
    estimate.
- *Member ids are normalized at every commit.* At the commit the process that took rank `r` is
  rank `r`. `R<r>` exists only inside replacement sets, for a spare that has not taken. The live
  member set is `0..world-1` after every change, so a plan computed from placement and every
  process's own key agree. The relabel is local (`Collective.rename_at_commit`): groups are keyed
  by member set, and their NCCL ranks are positions already.
- *Formation runs beside serving, fenced by generation.*
  - Each process forms the newest plan's sets that contain it, on a formation thread. The NCCL
    init's warm-up and the group's warm-up run on a stream of their own, with no device
    synchronize.
  - It releases every set the plan no longer names, beside the change path (0025).
  - A newer generation aborts a formation in progress, and so does a claimed change (0023) whose
    lost ranks are members of the forming set.
  - The change path never waits for formation.
- *Acknowledgements.* Each process reports `SetFormed{generation, set, scope, seconds}`,
  `SetFailed{generation, set, scope, error}` and `SetReleased{generation, set, scope}`.
  - A set is covered when every member has reported it formed, in both scopes.
  - A port is allocated per rendezvous host, not per engine, since engines share hosts. It
    returns to the host's pool only when every member has reported the set released. A port
    still held by a release is never handed out again.
- *Coverage gates REPLACE.*
  - The planner chooses REPLACE onto a spare only through a covered set, after the capability
    check (0034): the spare's software identity and compile pair against the position's.
  - A spare is ready to take a position when it has registered warm, with its identity, and the
    set naming it for that position is covered. That is the returned spare's ready signal; there
    is no separate one.
  - A loss with no covered set is refused at once, and the refusal names the missing set.
  - A policy flag may allow forming the set on the critical path instead, as a slower REPLACE
    that the change record reports as formed in the change. A change never waits silently for
    cover.
- *Protocol 8* (6 and 7 went to other decisions first). `CoverPlan`, `SetFormed`, `SetFailed` and
  `SetReleased`, defined in `signals.proto` and carried on both links; `coord.proto` adds the
  daemon's forwarded `CoverReport`, and the hello's `plan_generation`, `held` groups and
  `advertise_host`. The fields land first; the number moves to 8 with the processes' half, in one
  push, since a version mismatch is refused (0027).
- *Not closed here.* The plan carries each set's rendezvous host and port. The rendezvous store
  NCCL forms over is unauthenticated, as it was before this decision. Its authentication stays
  an open item, and nothing here should be read as closing it.

**The return, as the plan sees it.**
1. A node comes back. Its daemon says hello, and a spare is launched on it on demand (0031). The
   spare loads, warms and registers with its identity.
2. Plan `g+1` adds the sets naming it. Once they are covered, the positions the shared spare's
   node took are drained onto it: a REPLACE through covered sets.
3. Plan `g+2` restores cover for the shared spare, now free.

Every transition after the first gets its cover back the same way, so consequence 1 above closes
with step 3.

**Built in order.**
1. The coordinator's split along 0034's layers lands first, so the planner goes into the
   placement and transition modules.
2. Then, in parallel:
   - **Coordinator:** protocol 6, the planner (placement, spares, budget, ports per host) and
     coverage in the view and readiness.
   - **Adapter and exchange:** `rename_at_commit`, the stream-isolated formation with its
     generation fence and abort, the releases and the three reports.
3. Bootstrap reads generation 0 from the plan once both halves are in.

Agreed with the adapter's half (2026-10-12): a group's `spans` is its communicator's rank order and
its `server` runs the rendezvous store; a daemon releases every group a process held when its
connection ends, reporting each for that member; a plan always names the current membership's
sets, and a process refuses to release a group it serves on whatever a plan says; no process
forms between a change and its commit, and a formed report counts only at a generation planned
for the committed epoch's placement; generations only move forward in each process; a failed
group is retried by the controller's backoff, never by the process. Built (2026-10-12): the fields;
the planner (`policy::cover`: the current set always, then each node's loss, each position's, and a
second spare, within a budget of groups per process, ports per host kept across plans and never
reused while held); and the daemon's half (the newest plan kept and forwarded to every worker and
spare, and to each that registers while it stands; the reports forwarded; held groups in the
hello and released with their process). Next: the coordinator's plan state, delivery behind the
policy's flag, coverage in readiness and conditions, and the REPLACE gate.

The coordinator's plan state is built (2026-10-12), behind the policy's `cover` (off by default):
the plan derived again after every change to an engine's view and delivered as a new generation
when its groups change, above every generation a hello reported; reports counted only at the
delivered generation, so coverage is rebuilt from current reports; a group the new plan drops
keeps its port until every member that held it reports it released, or its node is gone; the
hello's held groups stand as cover until the plan's own reports replace them; the hello carries
every standing spare's registration, from which a restarted coordinator derives the spares'
admission and the cover. Readiness shows the plan and its coverage, and the Covered condition
follows it. The REPLACE gate is built (2026-10-12): with cover planned, a change goes only through
a covered set answering its loss, taking exactly the spares it names, or is refused at once naming
what is missing, unless the policy forms on the critical path; the record says which. A failed
group is retried by delivering the plan again after a backoff (one second, doubling to a minute).
A plan's placement names each rank's holder by node and device, so a
process tells a rank's holder changed between two plans, a take on the same node included; the
hello carries every serving registration, from which a restarted coordinator rebuilds each
position's device and identity. The coordinator's half is complete; the processes' half turns it
on with the bump to 8.

The process half is built (2026-10-12), and with it protocol 8. Each process holds its part of
the plan (`everserve.comm.cover.Cover`): plans at or below the generation applied are ignored; a
plan at a newer epoch first lets go of every held group naming as a rank a position whose holder
changed (which is how a spare standing for any position, which runs no commit, follows); groups the
plan no longer names are released beside serving, except the one the process serves on, which is
refused with the reason; kept groups are reported formed again at each generation in no time;
missing ones are formed on the cover's own thread in plan order, never between a change and its
commit, a failure reported once and never retried by the process. At a commit the process sets
aside every held set naming a lost position as a rank and renames each taker to its rank
(`Collective.rename_at_commit`). A spare's id names its process (`R1@node-d:0`), matching the
plan's `SpareRef`. The vLLM adapter forms a group on the cover's own CUDA stream, vLLM's
initialisation warm-up and its own on it, and synchronises only that stream
(`groups.form_planned`). Not yet measured: formation's effect on decode ITL p99, and a formation
blocked on a dead peer, which still runs to its store's timeout on the cover's thread.

**Measured before it is claimed.**
- Decode ITL p99 while a plan forms, against none forming. If formation shows in it, the plan
  forms one set at a time, between steps.
- The bytes per group per GPU on H100, read for the domain and the fabric scope apart, for the
  budget's default.
- Abort mid-init needs an NCCL init that can be abandoned. vLLM's wrapper at the pin binds only
  `ncclCommInitRank` (blocking) and `ncclCommAbort`. The preferred path is our own binding of
  `ncclCommInitRankConfig` (nonblocking) and `ncclCommGetAsyncError` to the same library, a
  `PRIVATE-INTERNALS.md` row. It is decided after the jitter reading. Without it, a formation
  blocked on a dead peer runs to its store timeout on its own thread. That is harmless to
  serving, but it delays the next plan.

*Amended 2026-10-15: a group's formation is bounded, and a failed group is formed afresh.* A
planned group's members have the policy's `cover.formTimeoutMs` (30 s unless given) to form it,
and the coordinator the same and a transit margin: a group some member has not reported formed
by then is failed, as if a member had said so. Until a failed group's retry is due, plans leave
it out, with every set that needs it, so each member lets go of what it holds of it; the retry
plan names it again, and every member forms it anew. A member whose formation reported success
without its peers (NCCL's split does, NCCL probe 4) would otherwise hold a dead communicator,
and a retry that named the same group kept it. The groups the engine serves on are not the
plan's to form, and are never failed or left out. A deadline counts nothing once a later plan
or a change has come: during a change members only record plans, and the plan after its commit
has a deadline of its own.

*Amended 2026-10-15: a group's join is gated on its members.* A group is joined with the
engine's own blocking communicator from its server's id, the same NCCL communicator a
nonblocking one would have to beat (NCCL probe 5: a split's child 8–12 % slower per call on
H100). Every member, the server included, reports that it holds the plan and waits for the
group (`GroupReady`); the coordinator relays the server's id to the members only once all of
them did at the plan's generation, and a member's receipt of the id is its go. A daemon that
links is given only ids every member is ready for. A member that dies after the go and before
it reaches the group's root hangs the others' blocking join, which NCCL bounds nowhere
(bootstrap.cc at v2.29.7: accept and recv without a deadline). Decided: the window is accepted, with its
consequence bounded: the formation deadline above fails the group and retries it fresh, and the
process whose formation thread is stuck reports it as a release left behind (0025), so a spare
takes its rank by notice with no gap.

**Consequences.**
- An engine's cover is a reported state with a budget, not a startup side effect. A second
  failure, a returned node and a lent spare are covered by the same mechanism as the first.
- Workers and spares stop reading the policy to plan sets. The execution layer forms what it is
  told and reports what it formed (0034's dependency rule).
- Cover at frontier width is a policy choice with a stated cost, and readiness shows what is
  uncovered rather than leaving it to a timeout.

## Amendment (2026-10-13): a group is shared by every set that needs it

Decided 2026-10-13: built after the first GPU proof on Kubernetes and before any engine past two
nodes, because the proof's two-node engine covers its losses under the budget and a wider one does
not (the figures below). Built, protocol 12, and proved on GPUs in
`bench/runs/2026-10-15-dynamo-proof-shared-groups` (proof-23).

**Context.** A set re-formed every group it names, so a plan held one group per set and scope
for every member of the set, and grew with the square of the engine's world. Measured with the
planner (tp8, one spare node of eight, REPLACE; `policy::cover`'s test
`a_plans_groups_grow_with_the_square_of_the_world_until_groups_are_shared`): with no budget, a
16-GPU engine's plan has 350 groups, a 32-GPU engine's 828 and a 64-GPU engine's 2,192, about
730 KB encoded, sent to every daemon and every process of the engine. Under a budget of 8
groups a process, a 64-GPU engine plans 4 sets and leaves 69 of its single losses uncovered, since every set costs every process its two groups, one
per scope. Yet a rank's loss changes one stage's domain group and one column's fabric group; the
other groups a set names are the current ones, which the processes already hold as their live
communicators.

**Decision.**
- A group's identity is its scope and its spans, the members in communicator rank order, not
  its set. Two sets that need a group of one scope over the same spans need one communicator, and
  it is formed once.
- A plan carries each group once (`CoverGroup` loses its `set`), and its sets apart: `CoverSet`
  holds the whole member set, the plan's groups it needs by index, both scopes, and its purpose
  for people. A process forms every group whose spans contain it, once however many sets name it,
  keeps it while any set of the newest plan names it, and releases it when none does. A group
  carried from one plan to the next keeps its port and is matched by its scope and spans, so it
  is not formed again.
- A set swap points the collective, scope by scope, at the group of the set whose spans contain
  the process. A scope whose group does not change is not swapped: the process keeps the object
  it has. The current set's groups are the live communicators, so the current set costs nothing,
  and every set names them for the stages and columns its loss leaves alone.
- Reports name a group by its scope and spans at a generation (`GroupFormed`, `GroupFailed`,
  `GroupReleased`); a set is covered when every group it names is formed by all its spans.
- The budget counts distinct groups a process holds. A rank's loss costs new groups only to the
  processes in its stage and its column, and to the spare standing in for it, so coverage scales
  with the engine and the plan with its sets, not with sets times world.
- Protocol 12: the coordinator and the processes change together.

**Split.** The planner, the wire and the coordinator's coverage are the control plane's; the
process side (held groups keyed by scope and spans and counted by the sets that name them, sets
as lists of groups, the swap's lookup, the renaming of spans at a commit) is the engine's.
