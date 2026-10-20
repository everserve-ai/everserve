# 0030: a persistent logical engine over a mutable placement

2026-10-09. Status: current, as amended below (2026-10-13); the degrade-then-restore path is designed,
not built.

**Context.** Pass 11 of the two-node dry run (`bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass11`)
closed the loop on REPLACE: a node dies, a spare
takes its positions, every session continues from its committed position with every K/V row equal
to the control's, 0.3 s after the kill, under a check that holds every process to one compile
(0029). The shape it ran on, one warm spare per position on a spare node holding every rank's KV
mirror, proves the mechanism and is a shape nobody runs: a second engine's worth of GPUs per
replica, and a mirror that cannot hold a frontier replica's KV, let alone a fleet's. The
long-term vision this decision adopts (2026-10-09): physical placement and parallelism are mutable
properties of a running engine, its logical identity and active request state preserved; REPLACE
is the first transition of several (RELOCATE, SHRINK and GROW, heterogeneous placement, repeated
transitions); every accepted transition leaves consistent membership and recoverable request
state; guarantees are explicit per transition; scheduling policy stays outside.

**Decision.** The engine is a logical object, its name, membership epoch, request state and
committed KV, over a changing, topology-constrained set of physical resources. Five primitives make
the placement mutable, and every landing from here is placed against them:

1. *Weights resident in host memory* on any node that may host one of the engine's positions
   (0028, generalised beyond the spare). Taking a shard is a copy from host memory, two to four
   seconds for a 44 GB share. A resident spare is the first use; RELOCATE is the same copy under a
   planned change; GROW is the same copy onto a new node.
2. *The store is the KV's placement-independent address space.* Rows are addressed by engine,
   rank, table and slot, never by device. A replica's rows replicate into its own sibling nodes'
   stores, in host memory the replica's nodes already have, over the replica's own fabric; a
   position restores from any store that holds its rows, local or remote, over the same wire.
   A spare whose own node holds a replica of the rank restores from it; the rows are staged from
   another replica only onto a spare's node that holds none (amended 2026-10-13, after run 3's
   attempt 11, `bench/runs/2026-10-13-run3-drains-smoke`, staged them over a replica the spare already held).
   This reverses 0016's "spare nodes only" for the shared-pool shape: a shared spare holds no KV
   until it joins. A change of KV ownership (SHRINK, GROW) is a retargeting and a restore of the
   rows the new layout owns, gathered from the tables that hold them.
3. *Pre-formed, warmed groups* (0024, 0025, 0028's attachment rule): a change is a pointer swap
   onto groups that exist and have run, never a formation on the critical path.
4. *The control plane as it stands*, epochs, topology, readiness and policy, with one addition:
   a transition is an object with feasibility, prepare, execute and observe, generalising the
   drain and the fault path that exist. Scheduling stays outside; the readiness API is what a
   scheduler reads.
5. *A shared spare pool*: a spare stands for any position of any engine it is warm for (0028's
   slots without a rank), warmed and registered per attachment, held to its position's compile
   (0029).

Guarantees are per transition class, stated with every published figure:

- **Exact**: the device class and the parallel layout are unchanged (REPLACE, RELOCATE). Every
  session continues from its committed position; tokens and K/V rows equal an uninterrupted
  engine's, under 0029's compile check and deterministic kernels. Measured by `first_differing_
  position` and `sessions_kept_identical`.
- **Continuous**: the layout or the class changes (SHRINK, GROW, heterogeneous placement). Every
  session continues from its committed position with recoverable request state; numerical
  behaviour is stated, not promised identical, since a reduction's order changes with the split.
  SHRINK stays gated by `placementIndependentModel` until a model is ported to arithmetic that
  does not depend on its split, and no served model is.

The class is the engine's, in its policy entry (`guarantee`, exact unless it says continuous), and
the mechanisms follow from it rather than from knobs of their own: deterministic kernels and the
compile-cache hold (0029) serve the exact class, and an engine under the continuous class chooses
its kernels by timing and has a compile difference reported, never refused. Both classes report a
pause bound and a capacity change. A transition the policy or the topology
cannot satisfy is refused, with the reason, before anything moves.

**KV protection modes.** The store and the staged replication give an engine three ways to hold
its KV against a loss, chosen per engine in the policy, each with its guarantee stated beside
every published figure:

- *Continuous*: every committed step replicates to the engine's replica nodes as it happens.
  Covers faults and notices alike; the exact class on a REPLACE. Costs host memory equal to one
  rank's share per replica and the replication bandwidth, nothing in device memory.
- *On notice*: no steady mirror. When a notice arrives (a spot reclaim's warning, a planned
  drain), the coordinator stages the leaving ranks' rows onto their spares' nodes, the same
  staged replication a REPLACE uses, started at the notice instead of the loss; the carrier sends
  each rank as held and then the live steps until the deadline, and the planned change swaps onto
  spares that are caught up. Covers every announced reclaim (AWS gives two minutes, GCP and Azure
  thirty seconds; a frontier node's share of a few hundred gigabytes crosses the fabric in under
  twenty) at no steady cost. A loss without a notice brings the engine back in seconds on the
  resident weights, and the lost ranks' sessions are re-prefilled from their prompts: the
  continuous class is not promised in this mode, and the record says so.
- *None*: resident spares only; every loss re-prefills.

Two rules hold in every mode. A rank restores all or nothing: a spare whose store is not caught up
for the rank at the deadline restores nothing for it and its sessions re-prefill, since a partial
restore mixes two states and is worse than none. And readiness states the notice window each
engine needs, from its ranks' bytes and the fabric rate the deploy measured, so a scheduler knows
whether a cloud's notice is long enough for this engine before it bids there. The on-notice mode
is built after the shared-pool primitives, from landing 2's staging with the notice as its
trigger, and the record gains a notice arm that measures the burst against the window.

**A spare for any position, held and retired.** A spare standing for any position forms and warms
one replacement set per position it is eligible for before it registers, and registers with no
compile pair, since it has compiled nothing. A change names it in `Change.take` by node and device
with the position's compile pair as the serving rank last reported it; the spare attaches, compares
its own pair, and refuses the change naming both when they differ, so a spare running other
kernels never takes a rank (0029 moved to the join). At the take its other identities' sets are
refused and released through 0025's releases, never left allocated. A survivor that holds its half
of a set naming the spare under another identity retires it by the same rule: the take names the
spare's node and device, the policy places each replacement identity on a slot's node and device,
so every survivor computes alone which of its sets name a spare that is now serving elsewhere and
refuses them.

**When no spare is ready: degrade, then restore (designed, not built).** A loss with no spare
standing today refuses the change and falls back to a rebuild, which drops every session on the
replica. The intended path keeps the replica serving at reduced capacity instead. The survivors take
the lost rank's share of the weights from host memory and give up KV space to fit it; for a
frontier model the weights leave that room (a 671B model in FP8 across 16 GPUs holds about 42 GB
of weights per GPU, about 45 GB across 15, so the KV shrinks by about a tenth). Sessions that no
longer fit are parked, not dropped: their rows stay in the store at their committed positions, and
they resume exactly when capacity returns. When a spare arrives, freshly provisioned or freed
elsewhere in the fleet, the replica grows back and the parked sessions resume. Pipeline parallelism
is the tractable case and can stay exact, since each layer is computed as before on another GPU;
tensor parallelism must keep the attention heads evenly divided and changes results at rounding
level, the continuous class. The limit is the weights: a loss that leaves them no room still
pauses the replica until a spare arrives, with every session's rows kept. Moving layers between
pipeline stages inside a running vLLM is the largest single piece of this, and it is built after
the shared-spare path is proven on hardware.

**Row checksums.** TLS covers the wire and ECC the memory; a software defect writing wrong rows
is what a checksum is for. The store takes a CRC32C of every row when it writes it, at the commit
from the staging file and at a replicated apply, and keeps it beside the row; replication carries
each row's recorded checksum, so a row damaged in the source's memory after its write is caught at
the replica rather than re-blessed there, and a replica refuses any step with a row that does not
match, or with no checksums, writing nothing of it (protocol 2). A refused step leaves the replica
uncovered, and the source sends the rank as held before its next step, so a replica never holds a
row its source did not write. The rows diagnostic reports each row's checksum. Not covered yet,
and stated with every published figure until it is: the worker's path from the device to the
staging file (the checksum is taken after it) and the restore's copy from the table to the device
(the client does not verify it); both are the adapter's, and the second costs a pass over every
restored byte on the critical path, so it waits for a device-side check. The hash runs at memory
speed beside the copy it follows.

**The next loop and the rental.** The next loop builds primitives 1, 2 and 5 for the shape a
customer runs: weights resident on the spare node, replication targeted at the replica's siblings,
the spare restoring from a remote store, a spare for any position. Each lands through the gate with
its test and is proven on one-GPU pods under $20 before any node is rented, as this week's loop
was. Then one rental on the deployment shape: two serving nodes and one shared spare node with
nothing preloaded but the weights in host memory, whose headline is the replica's recovery in
seconds against a stock restart in minutes, sessions exact. The two-node battery as pre-registered
(`bench/runs/PLANNED/p2-two-node`) stays the dry-run shape and is superseded as the rental; a new
pre-registration is written before booking, and no figure leaves the repository before its record.

**Consequences.**

- Decisions 0016 (replication to spare nodes) and MULTINODE's M3 and M5 are reversed for the
  shared-pool shape by this decision; their text stays as the record of the mechanism's proof.
- Repeated transitions need the pool to refill: a spare that took a position leaves the pool and
  the daemon's supervisor launches its replacement; a recovered node rejoins as a spare node.
- Coordinator failover (M1's lease) is on this path, since a control plane that drives many
  transitions over an engine's lifetime cannot be a single process.
- Layout change and graph recapture under SHRINK and GROW, and expert parallelism across nodes,
  are the unproven parts of the vision; they are measured when they are built, not projected.
- The public claim until then is the exact class: replacement with a spare, byte-identical, in
  seconds, on spot-priced nodes.
