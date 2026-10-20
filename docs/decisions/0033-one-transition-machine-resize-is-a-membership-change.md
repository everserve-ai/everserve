# 0033: one transition machine; resizing an engine is a membership change

2026-10-11. Status: design, partly built: the machine's REPLACE path, the software identity (step 2)
and the engine controller (0036) are built; the target layout on the change, the planner, RELOCATE as
an upgrade's evacuation, SHRINK, GROW, UPGRADE and lending are not. Step 4 (moving the head) went to
0038 and is reversed by 0045.

**Context.** Every measurement so far has been a fault: a node or a GPU lost, a spare takes its
position, the sessions continue. Two H100 runs of the p3-resident shape show that move exact on a
70B engine. But the cost a GPU operator carries every day is not faults. It is capacity sized
for peak and idle the rest of the time, and capacity that arrives minutes late when load rises.
A frontier MoE served wide is one engine of 16 to 72 GPUs that takes about 23 minutes to start
cold, so scaling it by whole replicas is too coarse and too slow. Nebius paid for this pain on
2026-10-01 (Inferize: snapshot restore of a whole single-node replica). The engines are
beginning to answer it from inside: vLLM's elastic expert parallelism resizes an engine's
expert-parallel width at tensor parallelism 1, on Ray; Dynamo has a proposal that scales
"membership, not pod count", with its autoscaler gated off. Nobody has publicly resized a
multi-node 300B+ engine under load (read in an earlier private harness, not in this repository; the figures there are the vendors'
own).

Decision 0030 already named the transitions (REPLACE, RELOCATE, SHRINK, GROW) and the five
primitives that make a placement mutable. This decision says they are one machine, what a resize
moves that a replace does not, and which of them is the product.

**Decision.** There is one transition machine. Every change to an engine's placement, whatever
asked for it, is the same object, goes through the same phases, uses the same control plane and
the same data plane, and ends committed or abandoned. What differs between kinds is the plan:
which positions change, what state moves, and where it comes from.

*The transition.* A transition takes an engine from one layout to another at the next epoch. A
layout is the engine's parallel widths (tensor, pipeline, data, expert), its expert map (which
experts each expert-parallel rank holds) and its member set (which device holds each position).
Today a change carries only lost ranks and the spares that take them; it carries the target
layout from now on, and a replace is the case where the target layout equals the current one
with positions remapped.

*The kinds.*

| Kind | Asked for by | Layout | Planned |
|---|---|---|---|
| REPLACE | a detected loss (fault) | unchanged; the lost positions move to spares | no |
| RELOCATE | a notice (a spot reclaim's warning), a drain, a cordon | unchanged; the named positions move | yes |
| GROW | an operator or an autoscaler | data or expert width up | yes |
| SHRINK | an operator or an autoscaler; a loss with no spare | data or expert width down | yes, or no for a loss |
| UPGRADE | an operator, node by node or engine by engine | unchanged; the software identity changes | yes |

A scheduler or an autoscaler decides when and to what size; Everserve decides how, and refuses,
with the reason, a transition it cannot make. Scheduling stays outside (0030).

*The phases, the same for every kind.*

1. **Plan.** From the current layout, the target and the topology: which positions change hands,
   which weights and which KV rows move from where to where, which groups and graphs the target
   needs, the pause bound and the bytes to move. Infeasible plans are refused before anything
   moves.
2. **Prepare, off the critical path.** Claim the capacity from the pool (0031), make the target's
   weights resident where they will run, form and warm the target's groups (0024, 0025), prepare
   the target's graphs, stage the KV rows the new owners will need (0030's staged replication, the
   streamed snapshot of `a830e98`). For a fault there is no time: the prepared state is whatever
   standing spares and continuous replication already hold.
3. **Fence** at a step boundary (immediately for a fault): the step in flight finishes or is
   aborted to the boundary (0021, 0025).
4. **Swap** onto the prepared groups and layout: a pointer swap, never a formation (0005).
5. **Restore** the KV each new owner needs, from the staged rows.
6. **Resume** every session from its committed position.
7. **Commit** the epoch, or **abandon** it with the engine down and its cause (0032).
8. **Release** the capacity the target no longer uses: back to its pool, or lent (below).

*What moves, by kind.* This is where the kinds differ, and the differences are real:

| | REPLACE | RELOCATE | GROW | SHRINK |
|---|---|---|---|---|
| Weights | the lost position's shard, host memory to the spare's GPU | the moving position's shard, the same way | the new positions' shards; under expert parallelism the expert map changes, so survivors gain or give up experts too | survivors take the leaving positions' experts; under data parallelism nothing |
| KV | the lost rank's rows, from its backup replica | the moving rank's rows, from its live holder, a final delta at the fence | none required: new positions start empty and take new requests; moving some sessions to them is an optional rebalance | every session the leaving positions hold moves to a survivor, each session to one; from the live holder when planned, from the backup after a loss |
| Groups | the target set exists already (shadow groups) | the same | new world size: groups for each prepared target width | the same |
| Graphs | the spare's own, same shapes | the same | shapes change on every rank (experts per rank, batch per rank): prepared per target width | the same |
| Scheduler | unchanged | unchanged | gains data-parallel ranks and routes to them | stops routing to the leaving ranks, then loses them |
| State comes from | a backup | the live holder | resident host copies | the live holder, or a backup after a loss |

Two distinctions carry the most weight. A replace keeps the layout, so only the moving position's
state moves; a resize changes the layout, so state moves between survivors as well, and every
rank's groups and graphs change. And a fault is the only unplanned kind: it alone restores from a
backup; every other kind copies from a live holder, ahead of the fence, which is why planned
transitions can be faster than a fault.

*Upgrades.* An upgrade is a transition whose destination differs from its source in software
identity (0034), not a separate mechanism. Its semantics follow from what changes: a host or
driver upgrade is RELOCATE off the node, the node upgraded and returned as capacity; an
engine-version upgrade moves positions only where the destination can read the state's format,
checked before anything moves, and is refused otherwise; a model-weights rollout moves no state,
since an old model's KV is not the new model's, and its sessions finish on the old weights or
recompute their context under the new ones. An upgrade still restarts something; what the
machine removes is the restart's dependence on the requests in flight.

*Capability before movement.* Every plan checks, before anything moves, that each destination can
take what it is given: its software identity against the state's format, its compile key against
the position's (0029), its device class against the guarantee class (0030). Hardware membership
and software version are separate identities (0034), so no plan assumes a replacement runs the
software its source ran.

*The widths that change.* Data and expert widths only. Tensor and pipeline widths stay fixed for
an engine's life: changing them re-shards every tensor or repartitions every layer, which no
transition that keeps sessions can do in seconds. A position under tensor parallelism is a group
of GPUs that moves whole (0022). So a dense model resizes by data width, and a mixture-of-experts
model by its data-parallel attention and expert-parallel width together, the shape wide expert
parallelism already serves in.

*Guarantees.* REPLACE and RELOCATE keep the exact class (0030): same layout, same kernels, same
compile. GROW and SHRINK change how each step's batch is split across ranks and which rank
computes which expert, so outputs stay identical only with batch-invariant kernels and a combine
order fixed by expert rather than by rank. Without both, a resize runs under the continuous
class: every session continues from its committed position, and numerical difference is stated,
not promised away. Whether a given engine keeps exactness across a resize is measured, never
assumed.

*Signals.* Kinds are asked for by signals, which the policy maps to a kind: a loss (a daemon's
report, a lost link), a notice with a deadline, an operator's drain or cordon or resize, an
autoscaler's resize, and a lender's recall of GPUs it lent. The control plane's API gains one
call, a resize of an engine to a data or expert width by a deadline with a reason, which returns
the change; drain and cordon stay as they are. Readiness reports, per engine, the widths it has
prepared, the cost of reaching each (pause bound, bytes to move) and the change in flight
(`a82f743`).

*Lending.* Capacity a shrink releases returns to its pool, or is lent to a borrower (a batch
tier, an RL rollout fleet) with a recall: the borrower is told, finishes or checkpoints within a
bound, and the GPUs come back for a grow. Designed, not built.

*The engine's own resize.* One resize path, Everserve's, with one set of semantics for every
shape. The engine's orchestration of a resize is never handed a transition unless it meets this
decision's contract (no error window, sessions kept across a shrink, a fault path, any rank
removable, a new rank's identity checked); vLLM's does not, read at the pin and at the latest
release. Its elastic expert parallelism is a planned resize behind one call nothing in vLLM
makes, on the Ray data-parallel backend, with no pipeline parallelism and no group across nodes;
it removes only the highest ranks, has no fault path, errors every request while it commits,
aborts the removed ranks' requests rather than moving them, and checks no new rank's version or
weights. It is read again at every pin move.

Engine components are reused only where verified to fit inside this path, and only on the side
of the seam they belong to (0034). vLLM's expert load balancer, read at v0.31.0 (`db9527a`,
2026-10-12; byte-identical at the pin for every file that decides it):
- *Its transfer is not reused.* It moves experts only between live GPU ranks, over collectives
  that include every rank, so an expert whose only copy is on a dead rank cannot be reached, and
  one with no sender is skipped silently, leaving a row's weights stale under a map that names
  the new expert. A change in the number of ranks runs only through the elastic orchestration's
  state. A failed move leaves weights and maps disagreeing, and a failed background move wedges
  its peers.
- *Its placement algorithm is not called from vLLM.* Which expert goes on which rank is
  engine-neutral arithmetic, so it lives in Everserve's planner (layers 5 and 6), Everserve's own
  or a vendored copy under its licence, the same for every engine; the planner never imports an
  engine.
- *Its router tables are the vLLM adapter's swap point.* Installing a new expert map is
  engine-specific by nature: the adapter writes vLLM's map tensors in place at a fenced step,
  after the weights land, all layers in one commit, the old maps kept until then so an abandon is
  clean, and captured graphs keep working because they read those tensors. This sits behind one
  engine-contract operation, apply this expert map at the fence, which another engine's adapter
  implements its own way.

Everserve builds the expert transfer, with sources that include host memory (the resident copies)
and the store and no collective that spans a dead rank, the abandon and commit across layers, and
the change in rank count.

**What is built, and what this adds.** Built: the change's claim, delivery, acknowledgement,
commit and abandon; epochs and fences; the store as the KV's address space with staged and
streamed replication; shadow groups; resident weights; any-position spares; the head's
registration; readiness with the change in flight and the last change's record; guarantee classes
and the compile hold. Built only for REPLACE: the plan (every change keeps its layout). Designed,
not built: the target layout on the change; the planner; expert moves between ranks; KV moved to
several receivers, each session to one; graphs and groups prepared per target width; the
scheduler's ranks added and removed; the resize call; lending.

**Order** (amended 2026-10-12). Each step is tested without a GPU first where the machinery
allows, then on a node. The code's layers come before any new kind (0034).

1. The coordinator split along 0034's layers, behaviour-preserving (control plane).
2. The software identity on registration and in the view (built, `2fcad06`); then the engine
   controller (0036): every kind is asked for by a difference between an engine's spec and its
   status, not by a command, and the target layout on the change, with REPLACE and RELOCATE
   expressed through both, so the existing kinds run on the general path before a new kind does
   (control plane, with the adapter's registration).
3. An evacuation that is a real upgrade, scoped first to one node that does not hold the engine's
   head: under load, the node drained, its vacated workers released cleanly, its software
   restarted at the same identity, the node returned as a spare and taking its positions back,
   every session continuing; a true version change follows once the compatibility window exists
   (0034, step 5). What it needs beyond the relocate path that exists: groups formed online for a
   node that returns, where today every replacement set is formed once at the engine's start
   (the critical piece, designed before it is built, read against vLLM's background group
   formation); vacated workers released and their node's processes ending when they hold
   nothing; a ready-to-take signal from a returned spare; a move and return call to a chosen
   node; cordon, uncordon and pools in the operator API (0031); spares launched on demand, not
   only when a daemon connects; a drain held to the policy's concurrent losses; several nodes in
   sequence with a readiness check between them (both).
4. Moving the engine's head: the head holds request state and does not move with a drain today,
   so upgrading the head's node takes the engine down. Its own design and its own measurement
   (both).
5. The adapter's reading of the engine's own resize at the pin, from code (done 2026-10-12, in
   "The engine's own resize" above); a measurement of its cost and its error window on a node
   comes with step 6 (adapter).
6. SHRINK and GROW by data width on a dense model: positions added and removed, the scheduler's
   ranks with them, sessions moved off leaving positions (both).
7. Expert width: the expert map in the layout, experts moved between ranks from resident copies,
   groups and graphs per prepared width (both).
8. A measured resize run, pre-registered: a mid-size mixture-of-experts model across two nodes
   under steady load, shrunk from 16 GPUs to 8 and grown back, three times. Kill conditions are
   set in its pre-registration; the starting point is any token gap over 5 s, any output that
   differs from a static run at the same width under the class the engine claims, or more than
   30 s of reduced throughput per resize.
9. Cross-version engine handoff: the state format descriptor, the capability check, a declared
   format for the join's payload (a pickle today, which holds only between identical software),
   and a translation where formats differ, tested as its own capability (both).
10. Lending, after a resize is measured.

**Consequences.** The product is the transition machine, and resize is the transition the market
pays for; fault and spot recovery are its unplanned cases, and upgrades its most frequent planned
one. What makes the position last is not the machine alone but what it carries that no single
engine does to itself: state that outlives its engine, transitions across nodes, versions and
clusters, a fleet neutral to engine and hardware, and guarantees someone else stands behind
(`docs/PRODUCT-ARCHITECTURE.md`, why it lasts). The engine-side hooks go upstream on purpose, so
an engine's native support ships what this system drives. This amends 0004, which ranked REPLACE
as the product and SHRINK as a tier: REPLACE stays the first transition and the exact one, and
the order of work follows this decision. The fault path does not fork: the existing kinds move
onto the general plan first, so every later kind inherits what two rentals proved. The largest
risk is that the engines' own resize, with Dynamo's control loop, ships across nodes before this
does; the work order puts the reading of what the engine already does second, so Everserve builds
only what the engine does not.
