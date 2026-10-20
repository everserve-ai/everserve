# 0022: a replacement stays in its captured group's domain, or the whole group moves

2026-10-07. Status: current.

**Context.** The two-node pre-registration's arm 6 replaced one rank of node-b with a spare on
node-c, which `docs/MULTINODE.md` section 4 cannot support: the tensor-parallel group's
all-reduce is captured in CUDA graphs on the slot kernel, whose peer-to-peer variant needs the
domain's links and whose host-memory variant needs the node's shared memory. A group split across
nodes would need the fabric slot kernel (decision 0015, phase 2) or a recapture. The review of
item 3 (2026-10-07) found the contradiction. The two-node smoke found the same day that stock
vLLM's headless node shuts every local worker down when one dies, which would have forced the
node-level outcome by accident rather than choice.

**Decision.** The policy's engine entry declares whether the engine's captured collectives can
span the fabric (`fabric_collectives`), false for every engine until it has a fabric path. While
false, a replacement must sit in the domain of its tensor-parallel group's survivors; when the
spare that would take a lost rank is outside that domain, the loss becomes the loss of the whole
group: its alive ranks join the lost set and are handed over the way a notice hands a rank over,
and the group's spares take every position in one change. Every rank of a moved group needs a
spare, and the group's spares must share one domain, or the change is refused with the reason.
The tensor-parallel group of a rank is the block of `tensor_parallel` consecutive ranks holding
it, which is how vLLM lays ranks out. Expert and pipeline groups are not constrained: their
collectives are fabric collectives outside the graphs.

The adapter's executor runs on every node, in a node role on the headless ones, so a death on
any node is an event there as on the head, and the whole-node outcome is this rule's choice, not
the engine's default.

**Consequences.** A single GPU death on a serving node costs that node's whole handover, which
is the product case the design opens with: nodes are the unit a neocloud reclaims. The
two-node pre-registration loses its cross-node single-rank arm and keeps the node arms. The
coordinator reads domains from the topology landing 4 built, falling back to the node name. When
an engine gains a fabric path, its entry flips the flag and single-rank replacement across nodes
returns with nothing else changed.

**As built (2026-10-07).** The node role is `everserve_vllm.node.NodeExecutor`, run by `python -m
everserve_vllm.node_main` in place of `vllm serve --headless` (whose executor is stock and not
selectable). It creates the node's slot region, named for the engine so the daemon's spares map
it (decision 0024), reports a worker's exit to the node's daemon, reads the region for the ranks
a worker's slot watch found lost and fences a hung one, and keeps the region's order and words
across a change; it sends no step and joins no spare. Decision 0021's stall gate on a node
without the engine is the members' own step words in the region (set when a step RPC is received,
cleared when answered, so a member that dies inside a step leaves the step in flight until the
change clears its word), the same rule with no new wire. Each member writes its own row's order
into the region at the boundary, from the change its group applies, so no node's executor writes
an order at a change and nothing is waited for; the executor writes the initial order when it
creates the region. The region's marks name positions in the stage; reports name engine ranks.
