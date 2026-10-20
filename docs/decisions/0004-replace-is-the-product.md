# 0004: REPLACE on stock model code is the product; SHRINK is a tier; REBUILD is the floor

2026-10-04. Status: amended by 0033 (2026-10-11); current as amended: REPLACE stays the first
transition and the exact one, and resize is the product's planned transition.

**Context.** The 2.56 s bottom rung of the recovery ladder was measured on stock vLLM model code
with a pre-loaded spare (week five of an earlier private harness, not in this repository). The ported model cores are what make
shrinking without a spare possible and the output bit-identical; they are not what makes recovery
fast. A port is weeks per architecture and must track the engine's model file.

**Decision.** REPLACE (a spare takes the dead rank's id at constant world size, sessions and KV
kept) is the v1 product and works for every model the engine supports. SHRINK (survivors absorb
the lost rank's units) is a tier bought per architecture by customers who need shrink-without-spare
or bit-identical output. REBUILD (rebuild the executor at a smaller parallel size, replay requests)
is the floor every deployment gets. All three run on the same membership machinery; the policy
document ranks them.

**Consequences.** No model port is on the path to a first customer. `arith.cores` stays as the
SHRINK pattern and the reference for kernel checks.
