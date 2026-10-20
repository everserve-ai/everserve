# Pre-registration: the frontier-size run (skeleton, after the two-node battery)

Written 2026-10-08 as a skeleton. The predictions are filled in only once the two-node battery
(`../p2-two-node/PREREGISTRATION.md`) is green, because each of them is that battery's figure
scaled by what this shape changes, and nothing here is a number until then. Nothing is rented for
this run until the two-node results are committed.

## Why this run exists

The two-node battery measures every cross-node piece on a 72B dense model. The figure a buyer
will ask for is a frontier-size model: a 600B-class mixture of experts, long contexts, expert
parallelism across the fabric. The same two-by-eight rental holds one: a 671B-parameter model at
FP8 (DeepSeek-V3 or R1) fits on sixteen H100s with room for its KV, and the adapter's
expert-parallel path is in the single-node battery (`fault-ep`). This run moves nothing new; it
measures the mechanism's cost where the cost is largest.

## What runs

A 671B FP8 mixture-of-experts model, the layout the engine's own recipe gives it on two 8-GPU
nodes (expert parallelism across the sixteen GPUs, pipeline across the nodes, the exact layout
written in the run log before the first measured arm), sixteen sessions, the determinism
settings, five trials per measured arm as in the two-node battery. The arms are the two-node
battery's 1 to 8, with two more:

9. **Long context at 131,072 tokens**, one session, REPLACE mid-step: the restore of one
   position's share of a context eight times the two-node battery's, for the curve that decides
   whether a buyer at that context needs the fabric's speed or a bigger spare pool.
10. **Expert loss**: one GPU holding experts killed mid-step, the spare taking its experts'
    position across the fabric.

**Recorded:** as the two-node battery, plus the KV bytes per token the model's attention keeps
(the compressed KV of this model family is small per token; the restore's size is the figure, not
the token count) and the expert-parallel collective launches per step.

## Predictions

To be written from the two-node results, each as that battery's figure scaled by:

1. **Adapter overhead** against stock: the two-node figure plus the expert-parallel collectives'
   share, which the single-node `fault-ep` battery put under one point.
2. **Node fault, single death, expert loss pauses:** the two-node median plus the restore's growth,
   which is linear in one position's rows; the detection, claim, commit, join, boundary and resume
   spans do not depend on the model.
3. **Long context:** restore seconds at the fabric's measured bandwidth against the position's
   rows, within the factor the two-node arm 8 established.
4. **Sessions continue** identical to a control: 16/16. This does not scale; it holds or it does
   not.
5. **Replication keeps up at steady state** and the prefill burst's lag, with the model's KV bytes
   per token in place of the dense model's.

## Platform and cost

The two-node battery's rental, extended: sixteen H100s on two nodes at about $56/h, about three
hours more including the model's load from storage, which at this size is the largest single
cost and is recorded. Cap to be set with the two-node results.

## What this run does not claim

More than two nodes, mixed hardware, the fabric slot kernel, Kubernetes installation.
