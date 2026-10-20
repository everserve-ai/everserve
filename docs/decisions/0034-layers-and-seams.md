# 0034: layers and seams

2026-10-12. Status: current, as amended below (2026-10-16); steps 1 to 3 and 5 built, step 4 (the
state format descriptor) not.

**Context.** 0033 makes every change to an engine's placement one transition machine, and the
product's lasting value sits in what no single engine does to itself: state that outlives an
engine process, transitions across nodes, versions and fleets, and a control plane neutral to the
engine and the hardware (`docs/PRODUCT-ARCHITECTURE.md`, why it lasts). For that to grow as
extensions of one system rather than as features bolted onto one module, the system needs layers
with one job each and contracts between them that an engine, a scheduler or a later product can
be built against. The code is partly there. The store is a clean layer; the wire is one; the
engine adapter is separate from the engine-agnostic runtime it uses. The coordinator is not: one
module of about 2,600 lines holds the daemon links, the membership view, the transition machine,
fleet policy (spare pools and admission, compile holds, replication targets, spare launches) and
operator views (readiness, stacks, drain).

**Decision.** Seven layers, dependencies pointing down only, and every crossing a typed contract
(a protobuf message or a trait), never a shared structure.

| Layer | Owns | Does not know | Code today |
|---|---|---|---|
| 1. Wire | Framing, transports, identity (mutual TLS), protocol versions | What a message means | `wire`, `proto/` |
| 2. State | The KV as an address space: rows by engine, rank, table and slot; durability (replication, checksums, fences, snapshots); a state format descriptor | Engines, policy, why a fence moved | `store`, `everserve.state` |
| 3. Execution | An engine's processes: how a position is fenced, swapped, restored and resumed in that engine; the engine-agnostic runtime it builds on | Other engines, the fleet | `adapters/vllm` over `everserve.comm` |
| 4. Node agent | One node's processes: registration, supervision, signals, local discovery, notice sources | Policy, other nodes | `controller/src/node`, `notice`, `discovery` |
| 5. Transitions | The engine controller, which reconciles each resource's observed status toward its desired spec (0036); the membership view (members, epochs, fences, layouts); and the transition machine (plan, prepare, fence, swap, restore, resume, commit, abandon, release) | The rules for which kind closes a difference (layer 6) | inside `controller/src/coord` |
| 6. Fleet and policy | Pools, placement and topology fit, guarantee classes, which kind satisfies a signal, lending | Engine internals | inside `controller/src/coord`, `policy`, `topology` |
| 7. Surfaces | The operator API (readiness, resize, drain, cordon, stacks, changes), the CLI, metrics, the console, the fleet layer across clusters | Anything but requests and views | `readiness`, `http`, `cli`, `metrics` |

*The seams.* Five contracts carry everything across the layers, each versioned:

1. **The engine contract** (layer 3 to 4 and 5): a process registers its role, its position and its
   software identity; it receives a change and acknowledges it, takes part in a join, hears an
   abandon, and says goodbye with a cause (signals.proto). An engine is supported when its adapter
   implements this contract; nothing above layer 3 names an engine.
2. **The state contract** (layer 3 and 5 to 2): open, prepare, commit, restore, replicate, expect,
   with a state format descriptor on every rank's tables (mirror.proto).
3. **The node contract** (layer 4 to 5): the daemon's link to the coordinator (coord.proto).
4. **The transition contract** (layer 6 to 5): a request for a transition (a kind, a target
   layout, a deadline, a cause) and its outcome. The fleet asks; the machine plans, executes or
   refuses with the reason.
5. **The scheduler contract** (layer 7, outward): readiness, the change in flight and the last
   change's record, resize, drain and cordon, the pools. Everything a scheduler, an autoscaler or
   a console needs, and nothing it can use to reach below layer 5.

*Identities, kept apart.* Five things that are easy to conflate stay separate types, because each
changes under a different transition:

- the **engine**: its name, a logical object that outlives everything below it (0030);
- the **position**: a place in the engine's layout (a rank under data, tensor, pipeline and expert
  widths);
- the **member**: the device or group of devices holding a position now;
- the **process**: the pid and role registered on a node, the engine's head included (0032);
- the **software identity**: the engine and adapter versions, the model and its weights' digest,
  the compile key (0029), and the KV format.

A replace changes the member; a resize changes the layout and so the positions; an upgrade
changes the software identity while the engine and its sessions stay. No code assumes two of
these change together.

*Capability before movement.* No transition moves anything before the planner has checked that
every destination can take what it is given: the destination's software identity against the
state's format descriptor, its compile key against the position's (0029), its device class
against the guarantee class (0030). A destination that cannot is refused with the reason, and
the source keeps serving. This is the check that makes an upgrade, a heterogeneous placement or a
cross-engine move a later transition kind rather than a rewrite.

*Upgrades, by what they change.* An upgrade still restarts something. What the system adds is
that the restart no longer depends on the requests in flight. Three kinds, with different
semantics:

| Upgrade | What changes | What the system does |
|---|---|---|
| Host OS or GPU driver | the node | cordon, RELOCATE every position off it, upgrade, uncordon, and the node returns as capacity |
| Engine or adapter version | the software identity; possibly the KV format | prepare destinations at the new version, check the state's format against them, move positions; refused where the format differs and no translation exists |
| Model weights | the model, so the KV | not a state move: an old model's KV is not the new model's. Sessions finish on the old weights while new ones start on the new, or their context is recomputed under the new weights; the policy says which |
| The control plane itself | the coordinator, a daemon | coordinator failover (a lease and a standby) and a protocol compatibility window, so mixed versions talk during the roll (`PRODUCT-ARCHITECTURE.md`, enterprise gaps) |

The engine's head holds request state, so upgrading it is an engine-version upgrade of the head's
process, and needs the head's state to move or be rebuilt; it is never assumed away.

**What changes in the code, in order, each behaviour-preserving with the tests unchanged.**

1. The coordinator splits along layers 4 to 7, inside the controller crate: the daemon links; the
   membership view; the transition machine; state placement (staging and replication targets);
   the fleet (pools, spare admission and launches, compile holds, outcome choice); and the views
   the surfaces read. Each becomes a module whose dependencies point down.
2. Registration carries the software identity, and the view keeps it per process and per
   position (layer 5's identity types).

   *Amended 2026-10-16, built: the members are held to one another.* A spare is held to its
   position's identity before it is admitted. The members are compared with one another on the
   same fields as they register (engine, its commit, the adapter, NCCL, the checkpoint's digest,
   the model's dtype, the guarantee, the layout, the KV's dtype and block size, and the step
   config; the CUDA version is not, since it is compared only for an exact spare). Two that differ
   are named in the Degraded condition (MembersDiffer), with the field and both values. The
   handoff measurement's ranks had registered two checkpoint digests for the same weights, read
   by two methods from two Hugging Face cache layouts, and served together. A spare was refused
   for one of them before anything said so.
3. The engine controller (0036, step 1): spec and status for the Engine, the spec store behind
   its trait, and the reconcile loop with REPLACE as its first kind, the existing calls rewritten
   as spec writes and events, behaviour-preserving. The change carries the target layout, and the
   existing kinds run through it (0033).
4. The store's tables carry a state format descriptor, and the planner checks capability before
   any movement.
5. The protocol compatibility window: a build accepts its predecessor's version, with every
   field added since read as absent by contract, replacing strict equality (0027) for the roll.

Crate boundaries follow modules only where a second consumer appears (a fleet layer across
clusters, a second engine's adapter); a module boundary with typed contracts is the seam until
then.

**Consequences.** Every product the lasting position names becomes a new transition kind, a new
adapter or a new surface, never a change through the layers: an upgrade is a transition whose
destination has a different software identity; a second engine is a second adapter of the same
contract; a fleet console is a surface over the scheduler contract. The coordinator split comes
before any new transition kind lands. The adapter's own side of the engine contract (registration
of the software identity, a format descriptor for its KV) is the adapter's to build against these
types.
