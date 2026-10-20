# The single runtime

Status: ADOPTED 2026-10-04 (decisions 0004–0007). The design of the one runtime inside a worker:
policy, not code paths, decides what happens on a membership change. The decisions this document
put as open (**D1** to **D5**, section 7) were ruled the same day: D1 by 0004, D2 by 0005, D3 by
0006, D5 by 0007.

What has moved on since, each in its decision:

- A spare holds the whole model in host memory and loads the shard of the position it takes,
  standing for any position rather than one preloaded rank (section 4): 0028, 0030.
- The member sets an engine pre-forms are planned by the coordinator as cover plans and formed
  while the engine serves, not fixed at start-up (section 3): 0035.
- A spare is built from the start-up calls the engine's core sent its workers, replayed to it
  before it warms: 0044.
- The engine's head, the scheduler's process on rank 0's node, does not fail over; its loss
  restarts the engine and is a known gap: 0045.

## 0. The constraint: no structural refactor between here and a neocloud deployment

Everything below is designed for the deployed state: a 16-GPU engine over two or more nodes on a
neocloud's preemptible fleet, a spare pool, a controller that runs unattended, and an operator who
never reads our code. Phase 1 builds a subset of that structure, never a stand-in for it. The
rules this imposes:

1. **The controller is the Rust daemon from the first commit.** There is no in-process Python
   controller that gets replaced. The phase-1 drill driver is a client of the daemon's protocol.
2. **One wire protocol, one schema.** Signals between controller, workers and the mirror are
   defined once (protobuf, `proto/`) and generated for Rust and Python. The Python
   ``multiprocessing.connection`` channels in `core` are the test transport and keep the
   same message set; the production transport is Unix domain sockets on a node and mTLS TCP across
   nodes, both behind the same client and listener types.
3. **State is a network service from day one.** `CommittedKV` is implemented by the `store/`
   process (Rust) with a placement rule: a rank's mirror lives on the node that will restore it,
   so a restore is a local copy. Single-node is the case where every node is the same node. The
   Python store in `core` stays as the exact, synchronous test transport.
4. **Kernels are production kernels.** The slot collective is written in `core/csrc` as a CUDA
   kernel with the host-memory and the peer-to-peer variants selected by topology. No Triton
   prototype lands in the product tree.
5. **Multi-node is the default shape.** Member sets, shadow groups, the spare's rank id, the
   mirror placement and the notice path are all specified over (node, device) pairs. A
   single-node engine is a degenerate configuration, not a separate path.
6. **Policy is data.** Which outcomes a deployment may use, how many spares it holds, what a
   notice deadline means, are a configuration document the controller reads, not constants in
   code. The same binary serves every customer.
7. **The adapter boundary is the upstreamable contract.** Every private engine line is listed and
   the adapter exposes only `docs/CONTRACT.md`. When vLLM or SGLang carry the hooks, the adapter
   shrinks and nothing above it moves.

## 1. What the runtime collapsed

It collapsed five mechanisms of an earlier harness, each its own module and code path chosen by
flags, into one runtime. The mechanisms were: *strict* (survivors shrink the group and absorb the
lost rank's units, a step guard re-runs the step, KV restored from the step-committed store; needs
a ported model), *relaxed* (kill the workers, rebuild the executor at a smaller TP, replay every
running request; stock model code, the slowest pause and a re-prefill), *warm standby* (a second
idle executor for the post-fault shape, swapped in; a standby GPU set held idle), *spare
replacement* (a pre-loaded process on a spare GPU joins as the dead rank at constant world size),
and *shadow + mirror* (pre-formed survivor and replacement communicators, an indirected handle so
captured graphs survive, an asynchronous device-to-host KV mirror; a fault is a pointer swap and a
KV copy, with no re-prefill, on stock model code). Two facts fell out: the bottom rung does not
need our model code, and what the ported cores add is shrinking without a spare and bit-identical
output; and every path is the same four moves in a different order, detect, stop the collectives
with the dead rank, restore state onto the new member set, resume, differing only in the member
set, where state comes from, and whether the model's shapes allow the new set at all. So the
runtime is one state machine over member sets with three *outcomes*, and the earlier mechanisms
are points in the outcome × model × state-source space. None of the harness's modules survives in
this repository; the runtime below is what replaced them.

## 2. The three outcomes

On a rank leaving (fault or notice), the controller chooses one:

| outcome | new member set | model code | sessions | when |
|---|---|---|---|---|
| **REPLACE** | survivors + a spare taking the dead rank's id, constant world size | stock or ours | kept in place; KV from the mirror, tail recomputed | a spare is ready |
| **SHRINK** | survivors only | ours (placement-independent) | kept in place; KV from the committed store | no spare, model is ported |
| **REBUILD** | survivors only, new executor at a smaller parallel size | stock | kept by the scheduler, KV recomputed from tokens | no spare, model not ported |

REPLACE is the default and the v1 product. SHRINK is the strict tier. REBUILD is the floor every
deployment gets for free. A deployment's *policy* ranks the outcomes it can use; the runtime
executes whichever is chosen with the same machinery.

## 3. The runtime inside a worker

Four layers, each a module in `core/`:

```
model forward  ──collectives──▶  Collective (indirected)  ──▶  live group | shadow group
      │                                                             ▲
      │ per-step KV writes                                          │ swap (epoch)
      ▼                                                             │
CommittedKV (mirror | store)                      Membership runtime (step guard)
      ▲                                                             ▲
      │ restore                                                     │ signals
      └──────────────────── adapter ◀──── controller ───────────────┘
```

**Collective.** One object the model (or the engine's all-reduce, via the graph-safe custom op)
calls for every collective. It owns the live group, the pre-formed shadow groups for every member
set the policy allows and the launched spares can play (`comm.shadow.member_sets`,
`policy.sets_to_preform`: a replacement set per spare, since a spare is built as one rank; since
decision 0035 the coordinator plans these sets as cover plans and the groups form while the engine
serves), and the handle captured graphs read at call time (`comm.handle.GroupHandle`). Two
implementations of the actual collective:

- **wrapped NCCL** for prefill-size tensors: resolved by name at call time, never captured;
- **the slot collective** for decode-size tensors inside FULL CUDA graphs: a kernel that reads a
  device-visible membership slot (mask, abort word, sequence counters) on every replay, so a
  captured graph follows a membership change without being recaptured. Both variants, peer-to-peer
  and host memory, are one kernel in `core/csrc/slot.cu`, selected by topology (decision 0006).

A membership change is: set the abort word, swap the handle, reset the slot's counters, bump the
epoch. No rendezvous, no NCCL init, no graph capture on the critical path.

**CommittedKV.** `save(step)`, `committed(request) -> tokens`, `restore(target)`. One interface,
implemented by the `store/` service: each step's new KV rows leave the GPU by one kernel into
host-mapped memory on the worker's node (`core/csrc/gather.cu`) and are replicated to the stores
the policy names, the engine's own member nodes by default (decision 0030), so a spare restores
from a store that holds the rows, local or over the wire. Rows carry the step's membership epoch,
and the service reports the last committed token per request so the scheduler recomputes only the
tail. The per-step synchronous put in `core.state` is the same interface over loopback, kept for
exactness tests and for SHRINK in CPU tests. Replication between stores runs over the wire crate's
transport, a Unix socket on one node and TCP under mutual TLS across nodes
(`store/src/replication.rs`); RDMA is not built.

**Membership runtime.** `comm.runtime.Runtime`: epoch, fault flag, recover, commit, agree, re-run,
quarantine, and two more inputs: a *notice* (planned departure at a deadline: stage KV, then swap
at the deadline with no fault ever observed by the engine) and a *join* (a spare taking a rank id
at constant world size). Both are membership changes through the same epoch machinery.

**Adapter.** Everything that touches the engine: the executor subclass (worker death is an event,
RPC without the dead, ring sweep), the connector or runner hook that feeds CommittedKV, the custom
all-reduce op or the model registration, the replacement process, the scheduler replay for
REBUILD and for the tail after a restore, and the capacity signal. Listed line by line in
`adapters/vllm/PRIVATE-INTERNALS.md`.

## 4. The spare

A spare is a GPU with a *replacement process* on it: the engine's worker, built outside the
engine's world from the start-up calls the engine's core sent its workers (decision 0044), its KV
allocated, its graphs captured against a local stand-in group, its shadow communicators formed,
idle. As first designed it held the shard of the one rank it stood in for; since decisions 0028
and 0030 it holds the whole model in host memory and loads the shard of the position it takes,
so one spare stands for any position. On REPLACE it swaps its handle to the replacement set,
copies the dead rank's rows into its KV, attaches to the executor's ring and starts answering as
that rank. Measured in this repository: a REPLACE under load, a node of a pipeline-parallel
engine taken away, committed in 412 ms with every stream finishing
(`bench/runs/2026-10-14-dynamo-proof`).

A spare costs one GPU per position it may take at once. Without a spare the policy falls to
SHRINK or REBUILD.

## 5. Notice-driven change

A preemption notice arrives with a deadline. The runtime stages the doomed rank's KV to the
replacement in rounds (`placement.staging.blocks_to_stage`), then at the deadline performs the
same swap as a fault, except the departing rank is still alive and finishes its step. The engine
never observes a fault; the pause is the swap alone. This is the path a spot fleet lives on.

## 6. Per-architecture cost

With REPLACE on stock model code, a new architecture costs nothing. The graph-safe custom op
replaces the engine's tensor-parallel all-reduce for every model at once.

SHRINK needs the model's shapes to be placement-independent: fixed KV head slots, fixed expert
chunks, slot-per-unit reductions. That is a port of the model file (`arith.cores` is the pattern).
Each port is weeks and must track the engine's model file. SHRINK is therefore the strict tier,
bought per architecture by customers who need shrink-without-spare or bit-identical output. No
served model is ported (decision 0030).

## 7. Decisions

Closed by the no-refactor constraint (section 0): the controller is the Rust daemon from the start;
the mirror is a network service with node placement from the start; the protocol is one schema;
kernels land in `csrc`.

Put as open on 2026-10-04, and ruled the same day:

- **D1. v1 outcome.** REPLACE on stock model code is the product; SHRINK is a tier added per
  architecture; REBUILD is the free floor. The alternative, SHRINK first, puts the model-port cost
  on the critical path to a first customer. *Ruled as recommended: decision 0004.*
- **D2. Shrink primitive.** `ncclCommShrink` versus a membership mask on a fixed-size communicator
  (Mooncake's approach inside SGLang). The slot collective already masks; the wrapped NCCL path for
  prefill can do either. *Ruled: pre-formed shadow groups, since NCCL's own collectives over a
  communicator with a dead member hang or abort: decision 0005.*
- **D3. Slot-collective topology variants.** Host-memory (proven on hosts with peer-to-peer
  disabled) and peer-to-peer device memory (NVLink hosts). Both in `csrc`, selected by topology;
  the question was whether the P2P variant is phase 1 or phase 2. *Ruled: peer-to-peer is the
  production kernel, host memory the fallback, both built: decision 0006.*
- **D5. Spare granularity.** One pre-loaded spare per engine versus one spare per N engines
  loading on notice (cheaper, slower). *Ruled: a controller-managed pool, per-engine pre-loaded
  spares first: decision 0007; then a spare resident in host memory standing for any position:
  0028, 0030.*
