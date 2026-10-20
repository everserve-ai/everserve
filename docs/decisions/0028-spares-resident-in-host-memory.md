# 0028: a spare that holds every shard in host memory and loads the one it takes

2026-10-08. Status: design; superseded by 0030 (2026-10-09), which carries the resident spare as its
primitives 1 and 5 and built them (the p3-resident runs, `bench/runs/PLANNED/p3-resident`).

**Context.** A spare stands in for one position and holds that position's share of the weights,
preloaded, so a change is a pointer swap and a restore. Shares differ by position, under tensor
parallelism and more so under expert parallelism, so a preloaded spare can take one position and
no other; the policy says which (`spares[].rank`). For an engine with a spare node paired to each
serving node that is the right shape and the one the two-node battery measures. For an engine of
sixteen nodes it is not: covering any single loss with preloaded spares takes as many spare GPUs
as the engine has, a second engine, the cost the design exists to avoid.

**Decision.** A second spare loading mode in the policy's spare pool: resident. A spare node
keeps the whole model's weights in pinned host memory, which a GPU node of the class that serves
a frontier model has (a 671B model at FP8 is about 700 GB), and runs one warm spare process per
GPU that holds no shard. When a change gives a spare a position, the join names it, as today, and
the spare copies that position's shard from host memory to its GPU over PCIe before it restores
the position's KV as now. The pause gains one term, the shard copy, two to four seconds for a
44 GB share, and one spare node covers any single loss anywhere in the engine; two cover any two.
Preloaded stays the mode for engines whose spare nodes are paired to their serving nodes, and the
policy chooses per pool.

**Consequences.**

- Every published pause names its spare mode and the spare cost beside it: preloaded, fast, a
  paired spare node; resident, the shard copy more, one spare node for an engine of any size.
- The shard copy is measured on its own in the two-node battery (arm 9) before the mode exists,
  so the large-engine figure is a measured sum, and end to end in the frontier battery once the
  mode is built.
- Replication is unchanged: a position's rows go to the node whose spare can take it, which in
  this mode is every spare node, so the replica count per rank is the spare node count. The
  policy's spare slots name a node and a device and, in this mode, no rank.
- The spare process's bootstrap, which today loads one share, loads none and maps the host copy;
  the join's handler copies the share. That is the adapter's landing, after the two-node battery.
- Warmth is a property of an attachment, never of the spare process. A spare registers for an
  engine only once it is warm for that engine's position: its pre-formed groups formed and warmed
  (one collective of every kind the engine issues on them), the engine's runtime kernels compiled
  (vLLM's own kernel warm-up, with the pipeline exchange stubbed on a stage past the first), both
  before its `Register` for that engine goes out. The preloaded spare of the two-node battery
  attaches once, at start, so it warms once and then registers. A resident spare attaches when it
  takes a share, warms for that engine then, and registers for each engine it stands ready for
  after warming for it; nothing in the warm-up reads a process-wide "warmed" flag, and a spare
  allotment shared across engines is this same rule applied per engine. The record proves it as it
  proves the rest: the first step after a resume reads at the steady step's size (pass 9 of the two-node
  dry run, `bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass9`, 2026-10-09, read 0.63 s on a cold spare and 0.06 s on a warm one).
  *Amended 2026-10-09 (pass 13, `bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass13`):* warmth has two parts, and a spare standing for any position has
  only the first at standing. Its communicators are formed and warmed for every position it is
  eligible for before it registers. Its kernels are compiled by the engine's own warm-up, which
  needs a worker built for one stage, so a rankless spare builds its worker and warms its kernels at
  the take, for the stage it takes; the record names that term. Kernel warmth at standing for a
  spare that is not preloaded would need a worker built per stage ahead of the take; that is the
  pre-attachment trade (a slot naming the positions it pre-attaches to), designed as an opt-in
  setting and not built. Until it is, the take's kernel warm-up is part of every rankless spare's
  recovery time.
