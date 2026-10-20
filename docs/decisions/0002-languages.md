# 0002: Python where the engines force it, native everywhere else

2026-10-03. Status: current.

**Context.** Engines are Python over CUDA; workers call us from inside a forward. Anything per
step or per byte on the CPU is where Python costs.

**Decision.** `core` API and adapters in Python. `core/csrc` is a C++/CUDA torch extension for
every per-call or per-byte path. `controller` and `store` in Rust: long-running, unattended,
concurrent, at a customer. Phase 1 has no controller beyond a drill script, so there is no
rewrite cost.

**Consequences.** A wheel per torch and CUDA version, as every engine pays. Nothing in the step
path executes Python beyond a call into the extension.
