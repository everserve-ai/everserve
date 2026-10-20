# Roadmap

What is shown is in `README.md` and indexed in `bench/runs/README.md`; what is decided is in
`docs/decisions/`. This is what comes next, in order, and why. Dates are not promised.

## Next

1. **The head's loss.** The one position without a spare (decision 0045). Closing it takes two
   capabilities upstream: vLLM letting an engine core attach to running workers through a
   supported executor API, and Dynamo keeping an instance's identity when its worker process
   changes, so its router waits for the new core rather than migrating away. Both are proposals
   to make, with the adapter's start-up replay (decision 0044) as the piece that already works.
2. **The target cloud.** The proofs ran on Nebius managed Kubernetes. The product's target is a
   neocloud's spot tier with its own preemption signal (decision 0041 names CoreWeave's labels);
   a proof there, with the cloud's notice driving the drain, is the next record.
3. **Hostile failures.** A hung rank and a slow rank are detected at the collective today
   (decision 0021) and treated as lost; a flapping link and a reclaim during a change each need a
   pre-registered drill with a pass condition under `bench/runs/`.
4. **Resizing.** GROW, the fourth outcome: a shrunken engine adds ranks when capacity returns.
   Decision 0033 designs it as a membership change through the one transition machine; it is
   not built.
5. **Degradation under a capacity crunch.** Spares first, then SHRINK with admission closed,
   then a drain at step boundaries, stated in the policy and enforced by the coordinator.
   Designed in decision 0033, not built.
6. **The adapter contract upstream.** `docs/CONTRACT.md` names what an engine exposes for
   Everserve to keep it serving through a change; every private vLLM line the adapter reaches
   is listed with the upstream hook that would replace it (`adapters/vllm/PRIVATE-INTERNALS.md`).
   Proposing those hooks shrinks the adapter and moves nothing above it.

## Not planned here

A second engine adapter (SGLang, TensorRT-LLM): the contract is written to admit one, and none
is in this repository.

## Neighbours

`docs/REVIEW-industry-shapes.md` records how Dynamo, vLLM and NIXL do what Everserve does and
what was adopted or deliberately done otherwise. The competitive read as of 2026-10-06, kept
for the record: no shipped system replaced a dead tensor-parallel or attention rank at constant
world size with the lost rank's sessions and KV kept; vLLM's elastic expert parallelism and
fault-tolerance work, TensorRT-LLM's wide-EP fault tolerance and Dynamo's request migration and
shadow-engine failover each cover a part of it, and the published research that shows
tensor-parallel shrink with KV kept (FailSafe, arXiv 2511.14116; AnchorTP, arXiv 2511.11617)
has not been released into an engine. Any of these can change in a release; re-check before
making a claim against them.
