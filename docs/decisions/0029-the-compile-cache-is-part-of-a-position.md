# 0029: the compile cache is part of a position

2026-10-08. Status: current; 0030 moved the check to the join, and 0043 fetches the cache to a spare's
node under Kubernetes.

**Context.** Pass 6 of the two-node battery's dry run
(`bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass6`) read every fault trial exact at the token
level and then read the K/V rows themselves, per position, against the control's. Two findings.
The spare on node-c matched the control below the resume and differed at rounding level from the
resumed position on, with identical inputs and weights: vLLM compiles the backbone and both nodes
loaded the same compiled artifact, but inside it the Inductor autotune results differed, the tile
configuration for each fused kernel chosen by timing on each machine. Then one launch on node-b,
nine launches after the cache was first filled, autotuned one more kernel fresh (one `best_config`
written in that launch's window) and landed in another rounding family for stage 1's prefill,
diverging from the control before any kill; every launch after it loads that file and lands with
it. Two launches of the same engine on the same cache match at every row, so a row-exact spare is
possible, and what decides it is which compiled kernels a process runs with, not the exchange,
the restore or the resume.

**Decision.** An engine's compile cache is part of each of its positions, as its weights are. It
is produced once, by a warm-up of the engine, and staged to every node that may host one of the
engine's positions by the deploy that stages the weights, every rank's subtree included, so that
every launch of the engine, serving rank or spare, is a cache hit and no node autotunes by itself.
Each process reports what it actually runs with: after its warm-up, when it registers, a `Register`
carries `CompileCache { key, digest }`, vLLM's `torch_compile_cache` key and a digest over every
file under that key and the artifact it references (`everserve.compile_cache.digest`, one
implementation, which the deploy's record uses too). The coordinator holds every process of an
engine to one key, the artifact its first registered process reported, and holds each spare to the
pair (key and digest) of the serving rank it stands in for: on a pipeline engine each stage
compiles its own graph into the artifact, so a rank's digest is its own and the digest a process
reports covers its rank's subtree and the kernels its graph names (pass 7 read two serving ranks
with one key and two digests). A spare with another pair is not admitted to the pool, or leaves it
when its rank registers with another, both pairs in the refusal; a serving rank with another key is
reported, per rank; and the engine's readiness says `compile_cache_consistent: false` unless every
serving rank reported the key and every spare in the pool reported its rank's pair, which a
scheduler, and the battery before every trial, refuses to run on.

**Consequences.**

- A launch that autotunes anything fresh has a digest other than the engine's and is caught at
  its registration, since the adapter registers after `compile_or_warm_up_model`. The staged
  cache is what makes that launch not happen; the check is what makes it visible when it does.
- The digest covers the whole key directory, so a node staged with one rank's subtree and not
  another's differs from a fully staged node and is caught the same way.
- The coordinator cannot close a serving rank today; a serving rank with another cache is reported,
  not refused. Closing it at the coordinator's word is a mechanism of its own and a follow-up,
  with coordinator failover (M1).
- Two engines with different compilation configurations have different keys; stock vLLM has its
  own key and its own autotune, so a row-level comparison against stock is not defined, and the
  two-node pre-registration scores rows against the engine's own first member.
- Every run record carries the key and digest of every launch beside its figures.
- The Triton cache is staged too, for speed, and it is not in the digest. vLLM's own Triton
  kernels on the serving path (the v2 runner's input batch, block table, buffer and sampler
  kernels, the attention, rotary, layernorm and activation layers) choose no configuration by
  timing at the pinned vLLM: no active `@triton.autotune` (the one in `prefix_prefill.py` is
  commented out), and attention runs FlashAttention 2, a CUDA kernel. Their configurations follow
  from source and shapes, so a cold cache costs compile time only (pass 14 of the two-node dry run,
  `bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass14`, 2026-10-09: 15.9 s on a fresh spare node, about 2 s on a staged one). A model that reaches an
  autotuned Triton kernel, at this pin or a later one, would need the staged Triton cache in the
  digest. A check at every pin move that fails on an active `@triton.autotune` on the serving
  path is designed for that reason and not built; until it is, a pin move re-reads the serving
  path for it by hand.
- Timing never chooses a kernel's numerics on a serving node. A launch consults a persisted
  autotune result only among its own candidates and benchmarks otherwise, and pass 6 found the
  q/k-norm reduction's configuration re-decided by timing in three launches with its result on
  disk and the artifact loaded. The engine therefore runs with Inductor's deterministic mode
  (`inductor_compile_config: {"deterministic": true}` in vLLM's compilation config, torch 2.13's
  `torch._inductor.config.deterministic`), under which a reduction's block and warp configuration
  is chosen by rule and never benchmarked; pointwise kernels still are, and their configuration
  does not shift numerics. The staged cache then carries rule-chosen configurations only, and the
  digest guards what a process loaded. The setting is the engine's, in its own launch, never read
  from the environment; its cost is measured on the battery's throughput arms beside the staged
  cache.
