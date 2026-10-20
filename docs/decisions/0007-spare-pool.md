# 0007: the spare is a controller-managed pool resource; per-engine pre-loaded spares ship first

2026-10-04. Status: amended by 0030 and 0031 (2026-10-09); current as amended: the pool holds
spares for any position and belongs to the fleet.

**Context.** Only a spare that already holds the shard, the KV allocation, the captured graphs and
the shadow communicators recovers a no-notice fault in 2.6 s. A spare shared across engines must
load the right shard on notice, which for a frontier rank is tens of gigabytes, and is useless
without a notice.

**Decision.** Spares are a pool owned by the controller, each with a state of "loaded for (engine,
rank)". The per-engine pre-loaded configuration is the v1 pilot's promise and ships first. A
shared-spare configuration is a scheduling policy over the same pool, added later with no change
to the runtime.

**Consequences.** The policy document expresses spare count and loading rule; the replacement
process is the same in both configurations.
