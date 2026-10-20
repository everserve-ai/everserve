# 0015: fabric collectives run outside the graphs first; the fabric slot kernel ships in phase 2

2026-10-06. Status: current; the fabric slot kernel it schedules for phase 2 is not built. A ruling on
`docs/MULTINODE.md` M2.

**Decision.** Every collective is classified by scope. Inside a domain the slot collective applies
as it does today, captured in CUDA graphs. Across the fabric, the first multi-node engines run
their collectives (pipeline send and receive, expert all-to-all) outside the captured graphs
through a handle that reads the member set at call time, at the cost of a launch per fabric
collective per step. The two-node run measures that cost and reports it.

**In scope, before a design partner.** The fabric variant of the slot collective: the same push
protocol as the peer-to-peer kernel with the push over the fabric by GPU-initiated RDMA, captured
like the other variants, the third implementation of the same `SlotCollective` contract. It is a
phase 2 landing, not phase 3, and it ships before the first design partner runs a multi-node
engine. The two-node run does not wait for it.

**As built, the pipeline over the family (2026-10-07).** The three-node smoke on Nebius found
that at the vLLM pin the pipeline's traffic never went through the device communicator the
family swapped: the intermediate tensors travel by `isend_tensor_dict` / `irecv_tensor_dict`,
metadata over the PP group's CPU group and tensors over its torch device group, and the sampled
tokens the last stage broadcasts each step go over a sibling torch group `PPHandler` makes at
init. All three are torch process groups formed once over the world: a dead rank cannot leave
them, and a spare outside the world has none. So the exchange is now ours: a named-tensor
protocol in core (`everserve.comm.exchange`: the metadata as one device tensor ahead of the
tensors, over any communicator with send and receive, the shape every pipeline-parallel engine
needs), and in the adapter (`everserve_vllm.pipeline`) vLLM's four tensor-dict methods on PP
groups and the sampled-token broadcast routed over the family's current group from bootstrap
on, survivors and spares alike, so a change swaps the group and nothing else; vLLM's own path
before bootstrap and on the stock engine. The spare's stub PP group carries its stage. Ledger
rows A33–A35.
