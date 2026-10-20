# 0005: pre-formed shadow groups are the primary shrink mechanism

2026-10-04. Status: current; the sets an engine pre-forms are planned by 0035's cover plans.

**Context.** A membership mask on a fixed-size NCCL communicator is not viable for stock
collectives: NCCL's own operations over a communicator with a dead member hang or abort. Masking
works only inside a collective we own, which is why Mooncake wrote its own backend for SGLang.

**Decision.** Every member set the policy allows is pre-formed as its own communicator while the
engine is healthy; a membership change is a pointer swap. `ncclCommShrink` with abort is the
fallback for a set that was not pre-formed, which policy keeps rare. The membership mask lives only
in our slot collective kernel.

**Consequences.** `Collective` owns live and shadow groups and the swap; the policy document lists
the member sets to pre-form; the 8-GPU run measures the fallback's cost so the policy can bound it.
