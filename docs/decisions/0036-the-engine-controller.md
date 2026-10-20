# 0036: the engine controller: desired state, observed state, reconciliation

2026-10-12. Status: current, as amended below (2026-10-13, 2026-10-15); steps 1 to 4 built, the
Kubernetes spec store included (each step's status below); of step 5, 0035's cover plans are built
and 0033's kinds beyond REPLACE are not.

**Context.** Layer 5 (0034) handles commands. A daemon reports a loss and the coordinator runs one
REPLACE; an operator calls drain and it runs one drain. A command that fails is abandoned, the
engine is marked down (0032), and nothing tries again unless someone notices and asks again.
That was enough while the only transition was a fault's replace. It is not enough for what 0033
adds. A resize is a goal ("eight wide by three o'clock") that may take several transitions, any
of which a fault can interrupt; an upgrade walks node by node; a returned node rejoins as cover
(0035). Written as commands, each of those needs its own handling of every interruption, and an
operator has no single place to say what an engine should be or to read why it is not.

Production orchestration settled this long ago. A Kubernetes controller, a Ray autoscaler and a
cloud's managed instance group all hold the desired state of a resource apart from its observed
state, and run a loop that takes the next safe step to close the difference, again after any
failure, until they agree or a stated condition says why they cannot. 0035's cover plan already
works this way for communicator sets. This decision makes it the shape of layer 5.

**Decision.** Every resource Everserve manages has a *spec*, what it should be, written only by
operators, schedulers and autoscalers through the API, and a *status*, what it is, written only
by the control plane from what the nodes report. A controller per resource kind reconciles the
one toward the other. The transition machine (0033) is unchanged: it executes one transition; the
controller decides which transition comes next.

## Resources

| Resource | Spec (desired, written through the API) | Status (observed, written by the control plane) |
|---|---|---|
| **Engine** | the model and the software identity it must run (0034); the parallel layout and target widths; the guarantee class (0030); the KV protection mode; the cover it must hold (which losses, the budget, 0035); the pool it draws spares from; a paused flag | the epoch; the members and the process at each position; each position's identity; the cover held and missing; the change in flight; the last change's record; conditions |
| **Node** | schedulable or cordoned; drained by a deadline; the software identity it should run | the daemon's link; the devices; the processes on it; the identity they run; conditions |
| **Pool** (0031) | the engine shape it serves; what may join it; how many spares it wants; how it holds them | the spares standing, their identities and readiness; the deficit |

Every spec carries a `generation`, raised by every write. Every status carries the
`observedGeneration` the controller last acted on, so a reader knows whether the status reflects
the spec it sees. A spec is never written by the control plane, and a status never by an operator.

*Conditions* are how status says what it means. Each is a type with true, false or unknown, a
reason in one word and a message in a sentence, and when it last changed. The Engine has at
least: **Serving**; **Converged** (status equals spec); **Progressing** (a transition is in
flight toward the spec); **Covered** (the cover the spec requires is held); **Degraded** (serving,
short of the spec, with the reason); **Down** (0032, with the cause). A scheduler or a console
reads conditions, never infers state from fields.

## The controller

- **The node reacts; the controller chooses.** Detecting a loss, fencing the dead process and
  refusing its groups belong to the node and the executor, and never wait for a reconcile: the
  executor marks the rank lost and its step raises, every group with a dead peer is refused at
  once (0025), a hung rank is killed by the slot path, a failed pipeline operation reports its
  peers through the worker's link (0023). What replaces the lost position is the controller's
  choice. Nothing on the node's reaction path is ever routed through the work queue.
- **Level-triggered, with event order resolved at the fold.** A reconcile reads the current spec
  and the current status, never the event that woke it, so a lost event loses nothing: the next
  reconcile sees the same difference. Events only update status and enqueue the resource. Their
  order is resolved where they are folded into status: a departure older than its rank's last
  change is dropped there (`WorkerEvent.epoch`, `f33a779`), so no reconcile ever sees it.
- **One resource at a time, many at once.** A work queue keyed by resource: one reconcile of a
  given engine at a time, different engines in parallel. A key enqueued while it reconciles runs
  again after.
- **A reconcile never waits.** It reads the cached spec (kept current by the store's watch) and
  the status in memory, issues requests whose outcomes return as events, and returns. Writes of
  status, the event log and the audit trail go to the store off the reconcile's path,
  asynchronously. Its wall time is held under a millisecond, measured, so a loss for an engine
  never waits behind a slow reconcile of the same engine, and the REPLACE path writes nothing to
  the store before its change goes out, so an unreachable store never delays a fault.
- **One step is one transition.** A reconcile folds every position without a holder in the
  status it read into one transition: a whole node's loss is one REPLACE with several takes at one
  boundary, never one per rank. A loss that arrives while that REPLACE is before its fence joins
  it (the machine amends the change); past the fence it becomes the next change. Whole-node
  losses first (0035).
- **Faults keep their speed.** A loss enqueues the engine at the head of the queue, and the
  reconcile's answer is a REPLACE through covered sets in the same pass. The decision costs a
  lookup inside the reconcile's bound. The fast path is the reconcile, not a bypass of it.
- **Two kinds of work in flight.** A *transition* is exclusive per engine: at most one at a time.
  A *preparation* (forming cover, 0035; launching a spare) runs in the background, fenced by its
  own generation, and is never "the change in flight". A loss never waits on a preparation: its
  REPLACE uses sets already covered, and aborts any formation that conflicts with it.
- **Priority and preemption.** Unplanned transitions (a loss) preempt planned ones (a resize, a
  relocate, an upgrade step). A planned transition not yet past its fence is abandoned at once,
  its prepared state released in the background and finished at a boundary (0025), and the
  REPLACE goes out in the same pass, never after the release. A transition past its fence
  finishes or abandons under the machine's own rules; the controller never interrupts a swap.
- **Planned work never spends required cover.** A relocate, a resize or an upgrade step may use
  only spares and sets beyond what the spec's cover requires (0035's budget), so no loss during it
  finds its cover taken, and no fault ever depends on an abandon succeeding.
- **Backoff and give-up.** A step that fails is retried with exponential backoff per resource,
  with jitter. After a bounded number of attempts, or when the policy has no feasible step (no
  covered set, no admissible spare, a refused identity), the controller stops trying that step,
  sets Degraded or Down with the reason, and waits for a spec change, a new resource, or the
  backoff's ceiling. It never retries a hopeless step in a tight loop.
- **No flapping.** The controller enforces a minimum dwell between opposite width changes of one
  engine, from the policy. Hysteresis (when to scale) belongs to the autoscaler that writes the
  spec, outside; the dwell is the controller's guarantee that a noisy writer cannot oscillate an
  engine faster than its transitions are safe.
- **The policy decides kinds, not the controller.** The controller computes the difference
  (positions without a holder, widths away from target, a node to drain, cover missing, an
  identity to change) and asks layer 6 which transition kind closes it and from which pool. The
  controller holds the loop; the policy holds the rules (0034).

## Cover, as the controller sees it

The cover an engine must hold (which losses, within which budget) is spec. 0035's cover plan is
layer 6's answer to the difference between that and what is held: derived, never stored, so it
lives in neither the spec store nor status. The sets each process holds are status, rebuilt from
the hellos (0035's reports of the sets held and the highest plan generation seen); covered and
missing feed the Covered condition. Forming and releasing sets are the controller's preparations;
`SetFormed`, `SetFailed` and `SetReleased` are its events. The plan's own generation stays
separate from the spec's: it changes with membership as well, with no spec write, and its fence
is what makes a stale formation abortable. Status records the plan generation each process holds.

## Where desired state lives

Observed state stays where it is: the nodes are its source, and a coordinator rebuilds it from the
daemons' hellos on start or failover (0014). It is never written to a store, because a stored
copy of what the nodes hold is a second truth to disagree with them.

Desired state must survive a coordinator's restart and its standby's takeover, so it lives in a
store behind one trait, `SpecStore`: read, write with optimistic concurrency (a write names the
version it read and fails if another landed since), and watch. Everserve does not implement
consensus. Three implementations, one contract:

- **Kubernetes:** Engine, Node policy and Pool are custom resources; the API server is the store,
  its `resourceVersion` the concurrency token, its watch the feed, and the coordinator is their
  operator. Kubernetes' own RBAC and audit apply unchanged.
- **etcd** for bare metal and other orchestrators, the same semantics through etcd's revisions
  and watches, beside the lease (below).
- **A local file** for one coordinator host, written whole and atomically. It has the limit of
  the file lease beside it: a standby must share the file system with both. A standby on another
  host needs etcd, which replaces the file lease and the file store together (step 3).

Writes go through the operator API (layer 7), which validates them (feasible widths for the
model, a known pool, an identity that names real fields) before they reach the store, and
records who wrote what and when. The imperative calls become spec writes: a drain sets the
node's drained-by deadline; a cordon sets it unschedulable; a resize sets the engine's target
widths. The calls stay as conveniences over the same writes.

## One writer, fenced

Exactly one coordinator acts at a time: it holds a lease in the same store (a Kubernetes Lease,
an etcd lease), renews it, and stops issuing anything the moment it cannot renew. The lease's
term is a fencing token: the coordinator sends its term on every message to a daemon, and a
daemon refuses any message from a term older than the newest it has seen. The daemon is where the
term is checked; workers are fenced by the epoch already, and the engine contract carries no
term. A deposed coordinator
that has not yet noticed its lease lapsed can therefore not move anything. Epochs stay monotonic
across a takeover because the new coordinator resumes above the highest epoch the daemons hold
(0014). The standby reads the store and the daemons' hellos; it acts only once it holds the
lease. Designed in 0014; this decision makes the term a fence and a requirement before any
partner runs the control plane.

Built (2026-10-12): the file lease's term, raised at every acquisition and kept in its file; the
term on every coordinator message; the daemon's refusal of an older term, a fenced delivery
answered at once; the daemon's newest term in its hello, and a coordinator that hears a newer one
claiming no change.

Built (2026-10-12): the etcd lease, for a coordinator with a standby on another host
(`--lease etcd://host:port[,host:port...]/key`, `--lease-watch` the same on the daemons). A
contender grants itself an etcd lease and puts the key only if it does not exist; the term is
that write's revision, the key's create revision. Every write of the record is guarded by "the
key's create revision is still the term".

The backend owns the hold, as etcd's `concurrency.Session` and client-go's `LeaderElector` do:
taking the lease returns a `Hold` whose own task renews it every quarter of the TTL
(`--lease-ttl-secs`, default 10), a keep-alive and then a check that the key is still its own,
since a key deleted under a live etcd lease is free for a contender. The hold ends, cancelling its
`ended` token, three quarters of the TTL after it sent its last answered renewal, by its own
monotonic clock and whether or not a renewal is still in flight; etcd expires the key no earlier
than a full TTL after it received that renewal. A request's timeout bounds only how long a call
may take. The coordinator acts under the hold's token itself, with nothing relaying it: once it is
cancelled the coordinator sends no daemon anything, claims no change, and exits 75 (`EX_TEMPFAIL`)
so its supervisor restarts it as a contender. Contenders retry every one to two seconds, drawn
afresh. The tests start a real etcd and fail without one; a partition is a relay that goes silent,
and the hold must end exactly at its deadline, before the contender is let in. Each mechanism has a
test that fails when it is removed.

Built (2026-10-13): the Kubernetes lease (`--lease kubernetes://namespace/name`, the daemons'
`--lease-watch` the same, through the shared API access `--kubernetes-endpoint`,
`--kubernetes-token-file`, `--kubernetes-ca-file`). A `coordination.k8s.io/v1` `Lease`, reached
with the in-tree client the node's notice source uses rather than a second Kubernetes client. The
API server expires nothing: as client-go does, a contender takes a held lease only once its
version has stood unchanged for its whole duration, by the contender's own monotonic clock from
when it first saw that version, and the holder's hold ends at three quarters of the duration after
its last answered write (the renewal the etcd backend shares, `lease::renewal`). The term is the
`everserve.ai/term` annotation, raised in the compare-and-swap write that takes the lease. Every
write names the `resourceVersion` it read; a conflict is read again and written once more only
while the object still has this hold's holder, term and `uid`, so a hold never continues on a
recreated object. Release clears the holder and keeps the term. Tested against envtest's
`kube-apiserver` over a test etcd: one hold at a time at rising terms, a partition (a silent
relay) ending the hold at exactly three quarters of the duration before the contender is let in,
renewal past the first deadline, another writer's takeover ending the hold, another writer's label
survived, a recreated object refused, and a coordinator process exiting 75 when another writer
takes its object; removing the staleness wait, the term's rise, the `uid` check or the conflict's
second write each fails a test. RBAC must deny `delete` on the `Lease` to every role; the manifest
that grants the rest lands with the Kubernetes deployment (step 4).

Decided (2026-10-12): daemons watch the lease, read-only. The lease's record names its holder,
its term and the address daemons link to; a daemon raises its fence to the record's term as soon
as it reads it, so a deposed coordinator's messages are refused even while its link still stands,
and it links to the address the record names, so the lease is also how daemons find the active
coordinator. This is the shape Kafka's controller epoch and Kubernetes' leader leases take: agents
learn the leader from the store, not from the leader. A daemon's access is read-only and scoped to
the lease alone: one `Lease` object through the Kubernetes API, one key in etcd, or, for one host
and the tests, a file beside the lease file. Linking every daemon to every coordinator was the
alternative; it adds connections without settling who leads.

Checked against Kubernetes' client-go leader election, etcd's `concurrency` election, Kafka's
controller epoch and Kleppmann's fencing tokens (2026-10-12), each backend must meet these:

- The term only rises and is never reused, and is raised in the same compare-and-swap write that
  takes the lease. For etcd it is the leader key's create revision; for Kubernetes an annotation on
  the `Lease`, since `leaseTransitions` does not rise when the same holder takes it again and starts
  over when the object is recreated; deleting the `Lease` is denied to every role, and a change of
  its `uid` is a failure.
- A lease outlasts the holder's renewal deadline plus its retry period, with a margin, checked at
  start. The holder stops acting by its own monotonic clock, counted from when it sent its last
  successful renewal, before any contender may take the lease as expired. Built for etcd, where
  every period is a fixed fraction of one TTL, so no setting can break the order.
- No one compares a timestamp from another host with its own clock: a contender times a lease by
  its own monotonic clock from when it last saw the record change. `renewed_at_ms` is for people.
- Contenders retry with jitter. Built.
- A holder that shuts down in order finishes its work, publishes that no one holds the lease, keeps
  the term, and lets it go; daemons then hold their reports. Built for the file and etcd backends.
- A daemon never lowers its fence; a record it cannot read leaves its link and fence as they are.
  A daemon that follows the lease links only after its first read, so a restart cannot accept a
  deposed coordinator's message first. Built.
- The coordinator's other writes carry the term as well: a spec store write is refused below the
  term its store holds, as Kafka fences a controller's metadata writes. Built (2026-10-13): the
  snapshot keeps the newest term that wrote to it and every write names its coordinator's term;
  a coordinator that knows its hold ended writes nothing (the operator API answers 503, write
  through the holder); the file store takes an exclusive lock on `<file>.lock` for each read,
  check and rewrite, so the checks hold across coordinators on a file system they share. The
  etcd and Kubernetes spec stores take the same contract when they are built.

## Failure modes

| Failure | What happens |
|---|---|
| The coordinator dies mid-transition | Daemons keep serving. The standby takes the lease and rebuilds status from the hellos. If any member reports the change's epoch, the change committed past its fence and status is rebuilt as committed, with the spares that took as members; if none does, it is abandoned (0032), the daemons take its epoch as spent, and the next reconcile plans again. |
| The spec store is unreachable | The coordinator keeps the last spec it read and keeps reconciling toward it, so faults are still handled. Spec writes are refused at the API with the reason. A condition says the spec may be stale. If the store holds the lease and the lease cannot be renewed, the coordinator stops acting, as above. |
| A partition between the coordinator and some nodes | Those nodes' engines keep serving; changes needing them wait; daemons that miss heartbeats are treated as lost by the rules of 0023. The fencing term keeps a coordinator on the far side, if any, from acting. |
| Two writers disagree on the spec | Optimistic concurrency: the second write fails with the version it lost to; the writer reads and decides. The controller never merges specs. |
| A spec the cluster cannot satisfy | No step is taken that cannot finish; Degraded names what is missing (capacity, cover, an admissible identity), and the engine serves as it is. |
| A transition fails partway | The machine abandons it (0032); the controller backs off and plans again from the new status, or gives up with the reason. |

## Observability

Per resource: the conditions, the last change's record (0032, `bb13db6`), and an event log of
what the controller decided and why, bounded and readable through the API. Metrics: reconcile
duration and outcome, work queue depth and age, retries and give-ups by reason, time spent not
converged per engine, spec writes by writer. A spec write's author and time are kept for audit.
Built (2026-10-12): `ControlMetrics` carries each of them; time not converged is the share of a
range during which `everserve_engine_condition` holds a status short of the spec, not a counter
kept across restarts.

## Testing

*Intended, not built: no simulation, fake clock or generated sequences exist in `controller/tests`.
What exists is `controller/tests/cell.rs` (synthetic cells: one coordinator, fake daemons over the
real link, changes driven through it at cell size) and `controller/tests/takeover.rs`, beside the
integration tests named at the end of this section.*

The controller is tested without nodes, by a deterministic simulation: a fake clock, a fake
machine whose transitions succeed, fail or hang on a schedule, and generated sequences of spec
writes and faults. Properties checked on every sequence: it converges when the spec is
satisfiable; it never takes a step after its give-up until something changes; it never runs two
transitions of one engine at once; it never oscillates faster than the dwell; a loss during a
planned transition is handled within the fault path's bound; a REPLACE's decision latency stays
under its bound while a planned transition and a formation are both in flight; a whole node's
loss is one REPLACE; a planned transition never holds required cover. The integration tests that exist
(real daemons, fake workers) run unchanged against the controller, and the hardware batteries
stay the measurement.

## Layers

The controller is layer 5's top half; the machine its bottom half; the membership view between
them. The spec store is control-plane infrastructure behind its trait, used by layers 5 to 7
and visible to no other. Layer 6 is consulted for kinds, never runs the loop. Layer 7 writes specs
and reads status and conditions, and nothing else. 0034 is amended accordingly.

## Order

1. The controller loop with REPLACE as its only kind: reports of a loss recorded against their
   engine, a work queue keyed by engine, a reconcile that folds every pending report into one
   transition and starts it as its own task, one transition per engine at a time, and the
   transition's outcome answering every report it folded. Every existing test passes through the
   loop unchanged. The change record's report time is the first folded report's, and
   `everserve_change_decision_seconds` measures report to delivery; the dry battery's gap on its
   own shape is read on the next rental against the last, measured, not assumed. Built
   (2026-10-12). Two pieces of this step follow it rather than lead: a loss arriving while a
   change is before its fence joins that change (the machine's amendment) instead of becoming the
   next one; and the spec and status types with the spec store, which land with their first
   reader (the conditions, then a desired width), so that no type exists before something uses
   it.
2. Conditions, the event log and the metrics; the Node and Pool resources (0031). The Node
   resource is built (2026-10-12): `spec.proto`'s `NodeSpec` (cordoned, drained by a deadline,
   the reason) in a `ResourceSpec` with its generation and its last writer; `SpecStore` with an
   in-memory and a file implementation, writes checked against the generation they read; a node
   controller with its own work queue keyed by node, woken by each transition's outcome, which
   starts a drain when the policy's concurrent losses allow it and leaves it waiting otherwise; a
   drain whose notices went out cannot be withdrawn or moved; a cordoned node's spares take no
   position, in the claim and in readiness; a stored drain resumes when a restarted
   coordinator hears its node. `POST /v1/drain` and a node notice write the spec; cordon and
   uncordon are routes. The node's software identity in its spec and the store's watch follow
   with their first reader. Conditions and the event log are built (2026-10-12): the Engine's
   Serving, Progressing, Covered, Degraded and Down, derived from observed status in one place
   after every change to an engine's view (Converged waits for the Engine's spec); the Node's
   Draining and Degraded; a bounded log of each resource's events, readable through the API. A
   drain whose change is refused is blocked, its Degraded condition says why, and its spec may
   be withdrawn or given another deadline; drains start in deadline order, so a restart resumes
   them in the order they were due whichever node says hello first. Both the order and the
   concurrent losses are per engine (amended 2026-10-14, the fleet-scale review): a drain waits
   only on drains of nodes holding one of its engines (by their positions, or the policy's
   placement while those are unknown), and counts only its engines' moving positions, so one
   engine's maintenance never holds another's. The Pool resource, and the
   reconcile and queue metrics, are next.
3. The lease's term as a fence on the daemon link, with the etcd implementation and the standby.
   Built (2026-10-12), but for the spec store's fence and its etcd store.
4. The Kubernetes spec store, as amended below.
5. Then 0035's cover plans and 0033's kinds (relocate, upgrade, resize) as the controller's next
   steps, each a new answer from the policy and a new transition for the machine.

*Amended 2026-10-15: what a coordinator knows when it starts or takes the lease.* It knows an
engine's ranks only from the daemons' hellos, which carry each node's workers, spares, head and
standby head. Each engine's membership, every rank's holder, is kept in its status once its
incarnation first holds every rank and whenever a commit moves one, and cleared when it goes
down. A coordinator claims no change of an engine with a kept membership until it knows every
rank of it: a change claimed before the hellos came was delivered to part of the engine, or (a
head lost just after a takeover) to none of it, and committed without the ranks that never heard
it. A rank is never lost for being unheard alone, so a coordinator's restart is no change of
serving ranks. It is lost by the node-loss rule, applied to a node the coordinator has never
heard: a daemon that can reach it relinks within its longest reconnect backoff, so one not heard
within that and the node-loss allowance (about 8 s at the defaults) after the first daemon's
hello cannot reach it. The clock starts at the first hello, not at the listening, since that
shows the daemons' whole path carries (a Service's endpoints, DNS, TLS). An engine with no kept
membership never held every rank: its ranks still registering are not lost, and its changes are
claimed on what is known, as before. A coordinator that no daemon reaches within a minute, with
engines that served, says it is unreachable and changes nothing. How long the daemons took to
relink after a start is measured, against the allowance. A daemon whose hello is from before
the engine's committed epoch missed a commit: its ranks are members only where the committed
membership holds them on its node (it acknowledged the change and missed only the commit's
message), any other is not, whoever holds it now, and the node is sent the membership as it
stands. The coordinator's view decides it, not the commit's record, which is written a moment
after the view moves. The daemon, on any commit, stops counting
a worker the membership does not place on its node and sends it the membership: a process left
out by a later epoch has left, and asks anew (a takeover mid bring-up committed an epoch on the
ranks it knew, and a node relinking after it had its ranks taken as members of an epoch they
never heard).

**Consequences.** Every transition kind after REPLACE is a new way to close a difference, not a
new command path. An operator states what an engine should be in one place and reads in one
place why it is not. The control plane can lose its coordinator, its store or an event without
losing the engine's intent. And the shape is the one an operator already runs elsewhere, which is
what a reviewer from a platform team will look for first.

## Amendment (2026-10-13): the Kubernetes spec store

Decided 2026-10-13. The Kubernetes API has no transaction across objects, and a lease is
not a fence: client-go's leader election says so in its own source, and CloudNativePG documents
that its lease cannot prevent split brain. A store whose every write is refused once another
holder took it needs each object it writes to carry the fence. Patroni's Kubernetes store is the
shape: the leader's state and its fence on one object, every write a replace guarded by the
`resourceVersion` the writer last read or wrote, and a conflict answered by stepping down, never
by reading again and retrying. One object for the whole store does not fit: an engine's status
held its cover plan, which grows with the square of its world (3.9 MB for 128 GPUs with no group
budget, measured with the planner), and etcd holds an object of at most 1.5 MiB.

- **Shards.** One root object holds the term and the specs (nodes, pools: cordons, drains). One
  object per engine holds its status: the change in flight, the assignments and the highest cover
  plan generation. Nothing the coordinator writes spans two objects: a change is one engine's,
  and a node's drain is its spec on the root, then a separately logged change in each engine it
  touches. The objects are an `EverserveState` custom resource, each a `SpecSnapshot` encoded.
- **The cover plan is not stored.** Every daemon is sent the whole plan and reports it in its
  hello; a successor adopts the newest it hears, and keeps the ports of groups processes hold
  from their reports. Only the plan's generation is stored, before the plan is delivered, so a
  successor numbers its first plan above any that a daemon lost since may have heard (0038,
  amended).
- **Taking the store.** A coordinator that starts or takes the lease takes the store before it
  acts (`SpecStore::take`): for each object, it reads it, refuses if a newer term is on it, and
  replaces it with its own term, guarded by the version it read; what it read is what it acts
  on. A write of the deposed holder's that landed before is in that read; one after is refused.
  A conflict during the take means the deposed holder wrote between the read and the replace:
  the take reads again. The coordinator acts under the term it took the store under, and no
  other.
- **Writing.** Every later write replaces one object, guarded by the version this coordinator
  last read or wrote, never one read again just before. A conflict means another writer changed
  the object since this coordinator took it, which only a newer holder does: the coordinator
  stops acting and exits 75, as when its lease ends or a write outruns its deadline. One write
  is in flight per object: an engine's writes already go one at a time under its status lock,
  and the root's under a lock of its own.
- **A new engine's object** is created with the term, refusing if it exists: an object this
  coordinator did not take was made after it took the store, by a holder deposed since, and the
  coordinator stops acting rather than write over what it has not read.
- **End to end.** The daemons already refuse a message under a term below the newest they have
  seen, from the lease's record or from a message (decision 0036, step 3): a deposed holder's
  command in flight is refused once a daemon heard the new term, and before that the holder
  logged it first, so the successor's take found it.

Built (2026-10-13): the take (`SpecStore::take`, `Coordinator::with_store` takes the store under
the term it then acts under, `ActError::TakenUnder` otherwise), a write the store fences ending
the coordinator's hold, and the stored plan replaced by its generation (`EngineStatus.plan_generation`),
for the file and memory stores. Built (2026-10-13), the Kubernetes store
(`--spec-store kubernetes://namespace/name`, `spec::kubernetes::KubernetesStore`, the chart's
`crds/everservestates.yaml` and `coordinator.specStore`): a new engine's object registered on the
root before it is made, and a registered object a taker finds missing made by the taker, so a
deposed holder makes nothing a taker could miss (refining the create-if-absent of the plan above,
which let a deposed holder's create make the holder's own conflict). Tested against a real API
server: a newer term takes every object and reads what was written before it; the deposed
holder's spec, status and new engine are refused, and the refusal ends its hold; a registered
object not yet made is made by the taker; a 128-GPU engine's status fits its object; the
coordinator's role makes and replaces the objects and deletes none.
