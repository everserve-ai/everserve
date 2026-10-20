# 0038: a GPU node is disposable, and the engine's head fails over

2026-10-13. Status: reversed by [0045](0045-the-heads-loss-is-a-known-gap.md) (2026-10-19) for the
head's failover; kept as the record of the design. The change log stands and is built: the
coordinator logs a change before its delivery and a successor finishes it ("The change in flight
survives the coordinator" below). The head's failover is a known gap, its code removed; the
sections on it below are its design as it stood, kept as the record of what was tried.

**Context.** Run 3 (`bench/runs/2026-10-13-run3-drains-smoke`) proved a rank's node can leave, planned or not, and the engine keeps serving
(0033, 0035). The engine's head, its scheduler and the client-facing server, cannot: it runs
beside rank 0, and losing its node takes the engine down (0033 step 4, not built). The product's
premise is that GPU nodes come and go, so a design that makes one GPU node special leaves room
for exactly the failure the product exists to absorb. Head failover is in the MVP's scope.

## The rule

*Reversed by 0045: this section is the design as it stood, not the system.*

> A GPU node holds only its own GPUs' work, and copies of state that exist elsewhere. The only
> copy of anything, and anything that decides for more than one node, lives on the CPU tier.

The CPU tier is a few small on-demand nodes, each role on it held by one process with a standby.
Placement is checked against the rule rather than argued case by case: the review checklist asks
of every landing whether it puts the only copy of something, or a decision for more than one
node, on a GPU node.

## Where things live

| | holds | why there |
|---|---|---|
| **GPU node** (disposable) | the GPU workers and the spares | its GPUs' work |
| | the node daemon | a local agent; holds copies (its fence, the plan it forwarded) and transient state (reports held while unlinked); decides nothing for another node |
| | store replicas of KV rows | copies, by the protection mode |
| | the notice sources | a node's own notices are read on the node |
| **CPU tier** (stable, with standbys) | the coordinator | decides for every node; the lease picks one |
| | the lease store (etcd, or the Kubernetes API) | the lease's only copy |
| | the spec store | desired state, and the controller state a restart needs: each engine's change in flight, its assignments and its highest cover plan generation |
| | the engine core and a warm standby core | decides for every rank of its engine |
| | the request-state mirror | the only copy of what in-flight requests are and how far they got |
| | the API front ends, more than one | hold client connections; stateless |
| | the operator API, metrics, alerting | the operators' view and controls |

The CPU tier's minimum is two small on-demand nodes, the holder and the standby of each role
spread across them, and three etcd members where etcd keeps the lease.

Two things broke the rule before this decision and are corrected by it. A restarted coordinator
took an engine's cover plan only from the daemons' hellos (0035). And a change in flight
(claimed, delivered and acknowledged, not yet committed) existed only in the coordinator's
memory. A coordinator lost there left the daemons at epoch N and the processes acknowledged at
N+1; its successor knew nothing of the change, and a daemon takes any change at its epoch plus
one, so the successor could claim another N+1 with other content: two changes under one epoch.

## The change in flight survives the coordinator

*Current: this section stands and is built; 0045 keeps it.*

The coordinator keeps a log of each change, as a two-phase commit's coordinator does, in the
engine's status (0036) in the spec store, fenced by its term:

- *The claim is logged before the first delivery*: the change as delivered (its epoch, lost
  ranks, outcome, takes and departing ranks), what each node is sent, and the membership it
  leads to. *The commit is logged before the first `Committed` goes out*, as sent. A write that
  fails or is fenced stops the change before anything leaves: the coordinator is deposed (0036),
  or the change fails before its delivery and nothing was applied.
- *A restart resumes the logged change; it never claims beside it.* The successor builds the
  engine's view from the hellos, applies the logged claim to it, and continues from the step the
  log reached. A logged commit is sent again to every daemon whose hello is below its epoch; a
  daemon ignores a `Committed` at or below the epoch it holds. A logged claim is delivered
  again, the same change to the same receivers, byte for byte, and committed when every
  acknowledgement is in. Every receiver takes a delivery it already acknowledged as
  acknowledged: a survivor finds the lost ranks already in its fault, and a spare that already
  took the position neither stages nor restores again.
- *Only the logged change is sent again.* When a receiver the log names is gone (a taking
  spare, or a survivor missing from the hellos), sending the change again is not possible, and
  a change placed afresh under the same epoch would be a different change. The change fails
  after its delivery (0032): the engine is down, and every daemon takes the epoch as spent.
- The engine's status also keeps the highest cover plan generation numbered, written before
  the plan is delivered, so a restart numbers its first plan above any a lost daemon heard. The
  plan itself is the daemons': every daemon is sent the whole plan and reports it in its hello,
  and a restart adopts the newest it hears (amended 2026-10-13, 0036's Kubernetes store: a plan
  grows with the square of the engine's world, 3.9 MB at 128 GPUs unbounded, and nothing in it
  but its number is lost when the coordinator is).
- Each change pays two fenced store writes. They are measured on rented GPUs and published with the
  change's budget.

## What the store guarantees, by protection mode

- *Continuous*: every committed step replicates to `replication_factor` nodes (policy, default
  one), placed in failure domains apart from the source's and from each other's: capacity class
  (spot, on-demand), zone, rack, as the planner already keeps groups inside NVLink domains. A
  reclaim takes many nodes of one type and zone at once, so a replica beside its source protects
  little. The rows survive the loss of up to `replication_factor` nodes that share no failure
  domain with each other.
- *On notice*: every announced reclaim whose notice covers the staging; readiness states the
  window each engine needs.
- *None*, and a loss without notice under *on notice*: the lost ranks' sessions re-prefill.

## Head failover: the control plane

*Reversed by 0045: this section is the design as it stood, not the system.*

- *The head is a role the membership names beside the ranks* (the standby's role, its
  registration, readiness and the head's cover built 2026-10-15, H1a; who is told which, H1b, the
  same day: an engine core asks undecided with `head` set, and the first on one of the policy's
  `headNodes` is told HEAD, its assignment kept in the store under the fence, the next on another
  head node HEAD_SPARE; the change, H5, built the same day on the control plane's side, the
  engine's attach, E4, not yet), never a rank: the core process
  (role HEAD) on a CPU-tier node, reached by the engine's alias (below). A warm standby core (role HEAD_SPARE) stands on another
  CPU-tier node, in another failure domain, placed by the planner from the policy's CPU-tier
  nodes. A single-node engine may run its core beside its GPU, a placement the policy names;
  its head is then covered like a rank.

  *Amended 2026-10-15, built: a separate head on nodes the platform chose.* Under a framework
  that schedules the head's pods (Dynamo on Kubernetes), the head's nodes are known only once
  scheduled, so the policy cannot name them. `separateHead` declares the shape instead: the
  core runs apart from its GPUs, and its loss is a loss to cover, as with `headNodes`, which,
  when named, still restricts where. With `separateHead` and no `headNodes`, a core on any node
  may be told HEAD or HEAD_SPARE, and the coordinator keeps the head and the ranks apart in both
  directions. A core is refused on a node that holds a rank of its engine, or a process told
  to serve one. A rank's process is refused on the node of the head, of the standby, or of a
  core told either and still loading. Without either field a core is refused: the head runs
  beside its GPUs.

  *Amended 2026-10-15, built: the ranks set the engine's step config, never its head.* The step
  guard holds every process to the engine's step config, the one its first process reported.
  A separate head builds its configuration from a declared hardware profile, not from a device
  it sees. If the head came first, a profile that did not reproduce the ranks' configuration
  would become the engine's, and every rank would be refused as the one that differs. So only a
  rank's process sets it, by its ask or its registration. A core asking for a separate head
  before any rank has set it is not answered yet; its daemon asks again, and the core is
  answered once a rank has. Its step config is then compared with the ranks', and a mismatch is
  refused at the head, naming the fields. A coordinator that starts or takes the lease takes the
  step config from the serving ranks its daemons' hellos report. Before this it took it from
  whichever process asked first after the takeover.

  *Amended 2026-10-15, built: shape A, the head beside rank 0, all spot.* The rule: no head that
  pretends to a GPU it lacks, and no head or standby node that is not preemptible, because a head
  configured from a declared profile rather than a device it sees cannot be held to its ranks'
  configuration (the amendment above), and an on-demand node per engine is a cost the product does
  not require.
  With `headOnRankNodes` the core runs as a process on one of the engine's own GPU nodes, and a
  standby core on another. The head moves only on a
  notice. A notice for the head's node (a cloud's, or the operator's `POST /v1/drain`, which is
  how tests send a synthetic one) moves the node's ranks to their spares `headHandoffMs` (30 s
  unless given) before the deadline, not at it. Once that change commits, the head is sent a
  retire notice (`ForwardNotice.to_head`, reason RETIRED) on its own link, and it stops taking
  requests and leaves. Its leaving is taken by the standby as a lost head's is. A notice that
  leaves less than `headHandoffMs` moves the ranks at the deadline as before and records
  HeadHandoffSkipped: the head stays, and its node's loss restarts the engine. A head node lost
  with no notice restarts the engine (0032), and Dynamo moves its streams to the instance's other
  replicas. The rules keeping a separate head and the ranks on different nodes apply to
  `separateHead` and `headNodes` only.

  The control plane starts the core: there is no head pod. At bring-up, when no head, head
  assignment or start stands, rank 0's node gets the engine a core, as vLLM's and Dynamo's
  leader is rank 0's: a `StartHead` goes to the node process that asked for rank 0's stage, on its
  asking connection, once it is told to serve, and the process starts the core in its own
  container, on the node's real GPU. Under this shape the head's node's ranks count in the Covered
  condition like any node's: a notice moves them before the head, so they need a formed set, and
  an engine whose head's node has no spare is not covered. (Found by the handoff measurement,
  `bench/runs/2026-10-15-dynamo-handoff-measure`, attempt 2: the core had started on rank 1's node, which the cell's one spare
  could not cover, while Covered read true.) A head node's drain is done once its ranks and its
  head have both left (DrainDone, as for any drain), and the operator returns the node to
  service by clearing its drain and cordon. The head's change wakes the node's drain as a rank
  change does; before this, a node whose last position was the head stayed draining, and every
  write that cleared it was refused. A head change ends at once on either side's failure. A survivor that cannot
  acknowledge (it died taking the new core's join), or a standby that could not take, ends the
  change, which is abandoned as any failed change is. Before this, the change waited out the
  other side's two-minute timeout (the handoff measurement, attempt 5). Every change is also delivered
  to the head's node, whatever that node still holds, since a head hears a change through its
  node's delivery. A head beside ranks that all depart, or apart from its GPUs, holds no member;
  before this its node got no delivery, the core never heard the claim, and it stopped at a
  commit it could not apply (the handoff measurement, attempts 5 and 6).

  *Amended 2026-10-16, built (the control plane's half): a spare's start-up replay.* A spare
  builds as its rank's worker was built: it receives what the engine's core sent that rank at its
  start, rather than deriving any of it (the rule the construction audit set: what a spare derives
  for itself can differ from what the core sent, as its derived KV configuration did). It asks on a
  connection of its own whose first frame is a `ReplayAsk`; that connection registers nothing.
  Its daemon names it (node, daemon instance, connection serial) and passes the ask up. The
  coordinator hands it to the engine's head, and the head answers with a `ReplayOffer`: a queue it
  made for that spare alone, opaque to the control plane. The offer is taken from the head's node
  only, and delivered on the asking connection alone. The control plane keeps nothing per ask.
  With no head standing, or the asker gone, an ask or offer is dropped, and the spare's bounded
  wait asks again. The engine's side (the core recording and serving its start-up calls, the
  spare replaying them) is the engine's half, built in the amendment below. The placement every process of the engine is handed names the
  head process (`EnginePlacement.head`: its node, and its connection to that node's daemon, which
  no other process has). A spare standing sees a new head there (a bring-up, a take, a restart),
  asks again, and leaves by name if the new core sent its rank something else: a spare can stand
  across an engine's restart, since a downed engine restarts once its members are gone, and must
  not take a rank with a former core's start-up. On a notice for the
  head's node with no standby standing, a successor is started the same way on the node holding
  the lowest other rank, never the doomed node, a draining node, or a node already running a
  core (the head, a standby, or a core told it stands and still starting). One start is
  outstanding per engine. A core that does not ask for the head within `startHeadMs` (5 minutes
  unless given) is given up on (HeadNotStarted), and the next node is asked. The head is retired
  only once a standby stands, by half `headHandoffMs` before the deadline, and only after its
  node's ranks have moved; if no standby stands in time, or the ranks' change is refused, it
  stays, and HeadHandoffSkipped names why. Under this shape the head is covered while another of the engine's
  nodes can start a successor. A head lost with no notice settles Down, with the reason named,
  and is restarted. Measured: nothing yet. The cold start of a core on a node, and so which
  clouds' notice windows a handoff fits, is the GPU rental's to measure. The engine's side (the
  started core, its retire on SIGTERM) is the engine's half, built beside it.

  *Amended 2026-10-16, built (the engine's half): a spare's start-up replay.* The calls are
  what vLLM's engine core sends its workers before it warms them: the KV cache layout it
  resolved, the memory profiling, the model length auto-fit settled on, and every rank's KV cache
  configuration. The executor records them as it broadcasts them; a standby taking the head
  records the two it sends nowhere, since its workers hold their caches. The head serves each ask
  on a vLLM `MessageQueue` with one remote reader, made for that spare, once its record is
  complete. The spare checks the rank's configuration fits its GPU (refused by name otherwise)
  and runs the calls on its worker by name, in the core's order, before it warms. Before this, a
  spare derived each from its own worker: that matched the engine only where every pipeline
  stage holds one kind of KV cache, and it missed the layout the core sets after building, so
  the standby taking the head refused the spare (the handoff measurement, attempt 6). A
  spare standing for any position reads every eligible rank's calls while it stands. A standing
  spare asks again once at its start and at each new head process its placement names, and
  leaves, naming why, if the calls differ from the ones it was built from: the engine restarted
  with another configuration while it stood. Every change's join offer also names the digest of
  its core's calls, and a spare built from others refuses it. The spare's worker runs vLLM's own
  memory profiling, as the engine's workers did, with no size given it to skip it: 4.49 s on H100
  (`bench/runs/2026-10-15-dynamo-handoff-measure/RESULT.md`, attempt 7; the battery's 2.9 s
  estimate was low), at its start for a spare built as its rank, in the take for one standing for any position.
- *Head cover.* The Covered condition and readiness include the head: covered when a standby
  core stands on another node and the request-state mirror is caught up on that node.
- *Detection.* A core's exit is reported by its node's daemon, as a worker's is; its node's loss
  by the coordinator's heartbeats (0023); a notice on its node, or a drain, starts a planned head
  move, the standby pre-adopting during the lead.
- *The change.* Losing the head is a change like any other: one epoch, one barrier, alone or with
  ranks lost in the same event. The surviving workers fence at the epoch, so nothing from the old
  core is acted on after it (the epoch is the one fencing token per engine). The standby core
  takes the head; the coordinator routes the new core's join offers to every surviving rank and
  every taker at the change's epoch, and their answers back, as it routes a taker's today; the
  commit names the head's placement.
- *The change, as built (H5, agreed 2026-10-15 with the engine side's E4; the control plane's
  side built the same day: a head departure with a standby standing claims the change, a
  standby on other software or one that answers with an error leaves the engine down with
  `HeadNotTaken` or the change's failure, and the commit keeps the new head's assignment).* A head lost while a
  standby stands on another of the policy's head nodes is a change of `OUTCOME_HEAD` at the next
  epoch, through the same two-phase machine as a rank's, the epoch its fence; with no standby, or
  with a rank lost in the same event, the engine goes down as before (joining the two is a later
  landing). The standby is refused the head when its software identity differs from the head's
  (0034). Every surviving rank's daemon is delivered the `Change`, and the standby's daemon a
  `TakeHead` carrying the placement and the request mirror's (epoch, step). The new core offers
  each survivor a join (its ring's handle, which carries its address), routed as a taker's is; each
  survivor answers as a spare does, with a new response queue's handle, its address and its last
  executed step, since a ring fixes its readers when made and every survivor's queue had the dead
  core as its reader. A survivor acknowledges the epoch once it has answered; the new core
  acknowledges it once it has subscribed every answer; the commit waits for both, so it means both
  directions are attached. It moves the head's assignment to the standby, which is then the head,
  and Covered reads HeadUncovered until another standby stands. Two invariants the order rests on:
  a survivor leaves the old ring when its daemon delivers the `Change`, since nothing on a dead
  ring can tell it, finishing a step it is in first; and a token reaches the front end only once
  the mirror holds it, so every step past the mirror's point was never seen by a client and the
  new core runs it again from the mirror, the last stage's RNG reset from the mirror's state.
- *The head record.* The coordinator publishes each engine's head (its holder, epoch and
  placement), as it publishes the lease's record (0036): `GET /v1/engines/{engine}/head`, and
  with `?watch` an answer at each change, authorized as `get engines/head` (0039), so a viewer's
  credential is enough. Front ends follow it; they run anywhere on the CPU tier and need no daemon.
  *Amended 2026-10-15, H2: no head route is built.* A framework's front end finds the new core
  through the framework's own discovery, as it finds any new worker: Dynamo's standby core
  registers its endpoints only once it is the head, since it waits inside its engine's
  construction for the change, and Dynamo's front end has no hook for an address given from
  outside. The engine's status names the head's and its standby's address (`head.host`), read
  and watched through `/v1/engines` with a viewer's credential, for operators. Two things the
  Dynamo proof (E5) must show rather than assume, and which reopen this if they fail: the lost
  core's instance leaves the front end's router promptly (its discovery registration is held by
  a lease until that expires), and a stream on the lost core is resumed on the new one without
  re-emitting what its client was sent.
- *One mechanism for bring-up and failover.* A core and its ranks meet by the carrier join the
  engine side already uses for a change's takers (0024): the core starts no worker of its own,
  each rank's process starts on its GPU node (launched by its daemon, as spares are), and each
  joins the core and answers with its endpoints over the control plane. An engine's first start
  is that join at epoch 0, for every rank; a head change is the same join, at the change's epoch,
  of every surviving rank to the standby core. The coordinator routes the joins in both cases, as
  it routes a taker's today.

  *Amended 2026-10-15, built: the bring-up's route.* At epoch 0 no rank's worker has
  registered, and none may yet: a worker's registration makes its rank a member, and an engine
  whose every position is held is reported serving (0041) before any weights load. The holder of
  a rank's join at bring-up is the process told to serve it (0041's assignment), from the answer
  until its worker registers. The coordinator routes a core's offer for the rank to that
  process's node, ahead of any spare standing for it, and an offer held because it came before
  the answer goes out with the answer. The daemon hands it to the process on its asking
  connection, which the process keeps open while it loads, and sends up the process's answer on
  that connection for the engine's clients. A pipeline stage's process asks for its stage and
  answers for each of its ranks. Nothing on the wire is new. The engine side (E1) is not yet
  built.
- *The request-state mirror* is a table of the store, replicated by policy from the core's node
  to the standby's node and fenced by epoch, like a rank's rows: the standby adopts from its own
  node's store.
- *Releasing a node* (built 2026-10-15). `GET /v1/nodes/{node}/release` answers whether a node
  may be given back:
  it holds no position (no rank, no head, no spare taking one), nothing is staged onto or from
  it, and no change in flight touches it; with `?wait_ms=N` it answers when that becomes true or
  the time runs out, with why not. Maintenance and reclaim tooling asks it, rather than reading
  a node's conditions and waiting for `DrainDone`. Authorized as `get nodes/release` (0039).

## Head failover: the engine

*Reversed by 0045: this section is the design as it stood, not the system.*

- *The head is a role, reached by an alias.* Clients, front ends and operators name an engine and
  its head, never a node or a rank: the alias resolves to the core holding the head position at
  the current epoch, as a Service name resolves to the pod behind it. No GPU position is special.
  Position 0 is a rank like any other; the roles a position carries inside the engine (vLLM's
  output rank, a stage's driver rank) belong to the position and move with it to whoever takes it,
  as rank 2's did, the output rank, to node-e's spare in run 3.
- *A core starts no worker.* On a multi-node engine the executor runs in carrier mode from the
  start: it binds its broadcast ring with no local reader, offers a join to every rank, and installs
  each rank's response queue from the rank's answer over the control plane (0024), never through a
  local worker. Stock vLLM learns remote response queues through its local worker 0
  (`MultiprocExecutor._init_executor`), so this mode is the adapter's, with no vLLM edit. A
  single-node engine that co-locates its core by policy may keep local workers.
- *The endpoints the change carries are the join's.* The control plane routes the join payload
  unread, as it does a taker's today: the core's offer (its ring's handle, the boundary's tag, the
  epoch) to each rank, and each rank's answer (its response queue's handle) back. The payload is
  the engine's declared format (`everserve_vllm.join`), not a field of the change.
- *The standby core is warm.* It has loaded the engine's configuration and tokenizer, built its
  scheduler with the engine's KV configuration (from the ranks' registrations), bound its ring,
  and its node's store holds the request-state mirror, which the store replicates there. At a head change it offers a join
  to every surviving rank and to every taker. A surviving worker, idle since its core died,
  subscribes to the new ring and answers at the change's boundary, as a survivor answers a taker's
  boundary today. Workers keep their weights, compiled graphs and KV cache. vLLM's engine core is
  built over an executor class that is ours, and sets itself up only through executor operations
  (`get_kv_cache_specs`, `determine_available_memory`, `initialize_from_config`,
  `compile_or_warm_up_model`, `v1/engine/core.py`); in attach mode these answer from the running
  workers, with nothing reallocated, reloaded or recompiled.
- *Resume from committed state.* One contract operation, in `everserve.contract`, which every
  engine adapter implements: stop at a boundary, apply the committed change, resume each in-flight
  request from its committed state. The control plane knows changes and nothing of steps. How a
  request resumes is the engine's: the vLLM adapter re-admits each request from the mirror and
  restores its committed KV rows from the node stores into new blocks on every rank, a host-to-
  device copy of the live KV (seconds, not milliseconds). Adopting the requests' block tables in
  place, over blocks the surviving workers still hold, is an optimisation, vLLM's alone. The
  change's record names which resume ran, and a request the store cannot restore is re-prefilled
  and counted.
- *What the mirror holds.* Engine-neutral, per in-flight request, written at the commit boundary
  that fences each step's KV rows, so the mirror never runs ahead of the rows: the request's id,
  its prompt's token ids, its sampling parameters, its committed output token ids, its sampler's
  RNG state, and the store keys of its committed rows. Nothing of vLLM's scheduler (block
  tables, prefix-cache maps, preemption state) is in it; a block table rides beside the request
  only as the optimisation's hint, ignored by an engine that has none.
  *Amended 2026-10-15, the wire agreed for building (H4, E3); the store's table, its records and
  its reservation built the same day (H4a), its replication to the standby's node not yet.* The table is a log of records per
  request, applied in order and folded by the store; the head's process sends them, one sender
  per engine, through the mirror client it already holds, at the boundary that fences the step's
  rows. `admit` carries the prompt's token ids and the parameters once. `grow` carries a commit's
  new output tokens and the KV blocks it added. `reset` marks a preemption by recompute: blocks
  and cached outputs cleared, prompt, parameters and emitted outputs kept. `finished` drops the
  request. The blocks are the engine's ids, one list per KV cache group, unread by the store. They
  are what keeps a head change from recomputing every running request: the KV stays on the
  surviving workers, and what dies with the head is the map from request to blocks. The sampler's
  RNG state is carried only where the head's node also holds the last pipeline stage, since
  sampling runs there. A standby reads the folded state from its own node's store. The table is
  bounded by the engine's running requests times its longest context, four bytes a token, and a
  store sizes its refusal from that, so a refusal is a sizing error caught at prepare, never a
  runtime surprise.
  *Amended 2026-10-15, the standby's copy agreed for building (H4b, E3) and built the same day, the stores' side and then the coordinator's and the daemons'; the engine's side (E3) not yet.*
  The head's node's store copies the table to the standby's node's store synchronously, so a
  head change never resumes from a mirror behind what a client was sent. The coordinator names
  the standby's store to the head's store, as it names a rank's replicas. Every put goes, in
  order, over one link per engine, and the store answers the put only once the standby's store
  has applied it. The answer names the engine, epoch and step it covers, and answers are
  cumulative: an answer for a step covers every step before it. The head's process holds a step's
  outputs, its aborts and its finishes among them, until the answer for that step, so a client
  never sees a token the standby could not resume from. On every connection the head's store
  sends the whole folded table before the next put, as a carrier sends a rank as held. A standby
  that does not answer within the policy's `mirrorAckMaxMs` is behind, and a head change refuses
  while it is: resuming from a mirror behind the clients would repeat or drop their tokens. So
  the store answers from its own table only once the coordinator has kept the standby as behind
  in the engine's status, as a Kafka leader writes its shrunk in-sync set before it commits
  without a follower: the store tells its daemon, the daemon the coordinator, and the store
  answers the held puts when the coordinator's keeping comes back. A head's node that dies in
  between leaves puts unanswered, never a standby wrongly trusted. A coordinator out of reach
  holds the puts; that costs availability, not correctness, and shows in the answer's latency.
  The engine's Covered condition is HeadUncovered while the standby is behind; clearing it once
  the standby has caught up needs no such order, since a late clear only refuses a change longer.
  The store measures the answer's latency and the time spent behind. The sampler's RNG state is
  carried only for seeded requests.
  Only a head whose software says it writes the table (`mirrors_requests`) is copied, and only
  its standby needs the copy caught up to cover it; a head that does not changes without one and
  resumes nothing from a mirror. The copy's state is kept in the engine's status, so a successor
  never counts a standby that was behind; the head's daemon reports it again whenever it is told
  the copy, newly or by a successor. The standby's daemon has its store expect the copy from
  nobody before it tells its core to take the head.
  *Amended 2026-10-15, found building H4b.* A coordinator that starts or takes the lease learns
  the head and its standby from the hellos, as it learns the ranks', and claims no change until it
  knows every rank's holder (decision 0036, amended the same day): a head lost just after a
  takeover was changed with deliveries to nobody, which committed and left every rank on the
  lost core.
- *Front ends.* More than one replica, stateless, behind a load balancer, anywhere on the CPU
  tier: they need no node daemon. A front end learns the engine's head from its framework's
  discovery (the head record above, as amended). Every request has a
  client-visible id, and a resubmission with the same id resumes it rather than starting another.
  A stream is resumable: a client that loses its front end reconnects to any front end with the
  request's id and the last offset it received, and the front end serves from the offset. At a
  head change a front end resubmits each of its streams' requests by id to the new core, which
  resumes them from the mirror, and the clients see a pause, not an error. vLLM's own front end
  treats a dead core as final (`core_client.py`: `engine_dead` is set and never cleared), so the
  front end is the adapter's: it builds a new engine client against the new core rather than
  reviving the old one.
  *Amended 2026-10-15, the rule: the client's stream is its router's; the engine is ours.*
  A head change keeps the engine serving, its workers' KV in place, and makes a resumed request
  cheap; the client's stream belongs to the layer that holds the client's connection. Where the
  framework's router migrates a broken stream to another instance, as Dynamo's front end re-sends
  a request with every token it already sent, the stream's continuity is the router's: the new
  core takes the migrated request as a new one whose prefix is the prompt and the tokens sent,
  and adopts the mirrored block tables into its prefix cache, so the workers' KV serves it with
  nothing recomputed. The request mirror is written all the same, for that adoption; holding a
  step's outputs until the standby's store has it would only add latency under such a router, so
  it is an integration setting, off where the router migrates. Where the framework has no
  migration (vLLM's own front end, which treats a dead core as final), the mirror's hold and a
  resuming front end are the way a stream survives, built when such an integration is. Dynamo's
  migration is read at the pinned version before the Dynamo integration relies on it.
- *Exactness, stated by class.* A resumed request continues token for token only with
  batch-invariant kernels, the mirrored RNG state and the engine's deterministic compile (0029);
  otherwise it is a continuation from its committed tokens, correct but not identical. The engine's
  policy names its class, and the record says which held.
- *A change's budget.* The client-visible stall of a head change is detection, the epoch's
  barrier, the joins, and the resume; each is measured on rented GPUs and published beside the
  rank-change figures. The CPU-tier core's per-step cost (its scheduler output always crossing the
  network) is measured against a co-located core on the same engine.
- *Additive on the wire.* The join payload follows 0040's discipline: a build reads the versions
  from its oldest to its current, a field absent means what the older build did, and an attach
  offer (a survivor joining a new core, with nothing to restore) is a new kind an older build
  refuses by name rather than misreads.

**Consequences.**

*Reversed by 0045: this section is the design as it stood, not the system.*

- Head loss stops taking an engine down; a multi-node engine's core sits on the CPU tier by
  default, and a measured hop decides any single-node exception.
- A neocloud runs a small CPU tier beside its GPUs; the docs state its size.
- The coordinator gains durable controller state in the spec store: each engine's change in
  flight, the coordinator's log of a two-phase commit, its assignments, and its highest cover
  plan generation.
