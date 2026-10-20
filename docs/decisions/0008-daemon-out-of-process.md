# 0008: the controller runs out of process; in-process bindings only for kernels

2026-10-04. Status: current.

**Context.** Dynamo exposes its Rust runtime to Python in-process through PyO3 (`lib/bindings/python`)
and runs engine workers as separate processes that talk to it over TCP. Our daemon's job is to act
when a worker process dies, so it cannot live inside that process.

**Decision.** The node daemon is its own process; workers link to it over a Unix socket with the
protobuf frame protocol. The only in-process Rust or C++ in a worker is the slot-collective kernel
extension, where the call has to be on the GPU stream.

**Consequences.** A worker's death is observed, not suffered, by the controller. The link protocol is
the one place the two languages meet, and it is the schema in `proto/`.
