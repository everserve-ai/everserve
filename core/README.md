# everserve (core)

The engine-independent runtime. It knows tensors, process groups, KV slots and expert chunks.
It never imports an engine; the dependency arrow is adapter -> core.

| module | owns |
|---|---|
| `everserve.comm` | communicator lifecycle: the live group, pre-formed shadow groups for survivor and replacement sets, the cover plan's groups formed while serving, the indirected handle captured graphs call through, the slot collective's driver, abort and swap, the membership runtime: the change applied at a step boundary (the product path, `Runtime.at_boundary`) and the step guard the reference tier runs under (`Runtime.run_step`) |
| `everserve.state` | the step-commit protocol (`CommittedKV`), the mirror client over the store's protocol, and the paged cache layouts a head slot's rows are read and written through |
| `everserve.contract` | the engine contract's process side (decision 0034): `signals` (the event model adapters raise and the controller consumes), `daemon` (a process's connection to its node's daemon), `head` (the engine's head on the control plane, decision 0032), `compile_cache` (the (key, digest) pair a member reports, decision 0029) |
| `everserve.kernels` | builds and loads the CUDA extension from `csrc/`, or says it is unusable here so callers take their torch paths |
| `everserve.policy` | the policy document on the runtime's side: which outcomes it may use, the choice for one event |
| `everserve.outcome` | REPLACE, SHRINK, REBUILD (`docs/RUNTIME.md` section 2) |
| `everserve.discovery` | reading the daemons' records (the file backend) |
| `everserve.podcontract` | what an engine's pod on Kubernetes mounts and sets to reach its node's daemon |
| `everserve.wire` | the frame protocol: a 4-byte big-endian length, then one protobuf message, as `wire/src/frame.rs` |
| `everserve.proto` | the schema generated from `proto/` (`scripts/gen-proto.sh`, committed) |
| `everserve.reference` | the reference tier (decision 0004), on no product path and imported by no adapter: the executor-to-worker fault channel, rejoin at constant world size, the Python KV store, the loopback KV copy and the `CommittedKV` over it, the placement plan (fixed KV slots, fixed-shape expert chunks, who computes what after a loss), staging on notice, recapture after graphs were lost, and `reference.arith`: placement-independent reductions and the reference cores (one producer per element, bit-identical output across membership changes), the pattern a SHRINK port of a model follows. What the CPU tests run SHRINK against; shipped so the tests and a reader of the mechanism have it |
| `csrc/` | the C++/CUDA torch extension `everserve_kernels`: `gather.cu`, the mirror's one-launch step gather (decisions 0010, 0011); `slot.cu`, the slot collective in both variants, peer-to-peer and host memory (decision 0006); `kernels.cpp`, the bindings. Its README says how it is built |

Nothing in the step path executes Python beyond a call into the extension.

What this package has to reproduce is measured under `bench/runs`: a rank's node taken away under
load and its position taken by a spare with every stream finishing, the REPLACE committed in
412 ms (`bench/runs/2026-10-14-dynamo-proof`); a loss by notice with a worst gap between tokens
of 0.642 s and a crash with 1.135 s, the spare built from the core's replayed start-up calls
(`bench/runs/2026-10-19-dynamo-proof-known-gap`); and, on one node, every session token-identical
across a REPLACE and a notice (`bench/runs/2026-10-06-exact-replay-4gpu`).

The CPU tests in `tests/` run four gloo processes as tensor-parallel ranks and kill one. They are
the reference tier's spec (decision 0004): a rank lost between steps or mid-step must leave every
token and logit unchanged. They run the SHRINK outcome on `reference.arith.cores`, which no served
model is ported to; the product runs REPLACE, whose GPU evidence is the runs above.
