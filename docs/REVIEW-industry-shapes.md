# Review: how the production systems next to us are built, and what we take from them

*A review note from 2026-10-04: a snapshot, kept for the record and not maintained. The current design is docs/ARCHITECTURE.md and docs/decisions/.*

2026-10-04. Read against the sources, not from memory: `ai-dynamo/dynamo` (runtime crate
`lib/runtime`, Python bindings `lib/bindings/python`, workspace `Cargo.toml`, `rust-toolchain.toml`,
`worker.rs`, `runtime.rs`, `config.rs`, `error.rs`), `ai-dynamo/nixl` (README), and vLLM at the adapter's
pin (`kv_transfer/kv_connector/v1/base.py`, `plugins/__init__.py`,
`v1/executor/abstract.py`). Each row says what they do, what we do, and whether we change.

## Repository and toolchain

| they | we | ruling |
|---|---|---|
| Dynamo: Rust workspace, `edition = "2024"`, `rust-toolchain.toml` pinning one channel (`1.96.1`), workspace dependencies, exact-pinned tokio (`=1.52.4`) | Rust workspace, edition 2024, `rust-toolchain.toml` pinned (`1.97.1`), lockfile committed | **same**; the pin was added the same day after CI and the local gate disagreed on a lint |
| SPDX header on every source file (Dynamo and vLLM both) | none | **adopt**: SPDX on every file, checked by the gate. `Apache-2.0` on every file (decision 0046) |
| Dynamo: `lib/` crates by concern (`runtime`, `llm`, `kv-router`, `kvbm-*`, `memory`, `tokens`), Python bindings as one crate | `core/`, `adapters/`, `controller/`, `store/`, `bench/` | **same shape**; split crates further only when a second consumer exists |
| Dynamo workspace lints: not pedantic by default | `clippy::pedantic` warn, `-D warnings` in CI | **keep ours**; stricter is cheaper now than later |

## Process model and language boundary

| they | we | ruling |
|---|---|---|
| Dynamo exposes the Rust runtime to Python **in-process** via PyO3/maturin (`lib/bindings/python`, crate `dynamo-py3`, `cdylib`); engine workers run as separate processes (`python -m dynamo.vllm`) that talk to the Rust frontend over TCP | Rust daemon is its own process; the Python worker links to it over a Unix socket | **keep ours, deliberately**. The daemon's job is to outlive the worker that dies; an in-process binding would die with it. Record this as decision 0008. PyO3 is the right tool for the slot-kernel bindings, which are in-process by nature |
| Dynamo `Worker::execute`: one tokio runtime per process (`OnceCell`), SIGINT/SIGTERM trap, graceful shutdown with a timeout from `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT`, exit code on overrun | `tokio::runtime::Runtime::new()` in the CLI, `serve` loops forever, tests `abort()` the task | **adopt**: a `CancellationToken` through `Daemon::serve`, signal handling in the CLI, a graceful period, tests cancel instead of abort |
| Dynamo config: `figment` merging defaults, files and `DYN_*` environment; `environment_names` module lists every variable | hand-rolled `--policy`/`--socket` parsing | **adopt** the shape with less machinery: `clap` with `env = "EVERSERVE_*"` on every flag, one module that names the variables |
| Dynamo `error.rs`: one semantic error contract for the request path (class + stable reason catalog, diagnostics never reach clients) | per-module `thiserror` enums | **keep ours for now**; we have no client-facing request path. Revisit when the coordinator exposes an API |
| Dynamo transports: TCP (+ `rustls`), etcd and NATS optional, ZMQ; discovery: Kubernetes native by default, file-based for dev | Unix socket on a node, mTLS TCP across nodes planned; discovery: file backend landed (`controller/src/discovery`), Kubernetes `EndpointSlices` in phase 2 | **adopted** their discovery split; nothing mandatory outside Kubernetes |
| Dynamo observability: `tracing` + OpenTelemetry, Prometheus registry, `tokio-console` feature, NVTX feature | `tracing` only | **adopt in phase 2**: Prometheus metrics module, OTLP export behind a feature |

## Engine interfaces (vLLM)

| vLLM | we | ruling |
|---|---|---|
| Plugin groups with explicit process scope: `vllm.general_plugins` loads in **all** processes (API, engine core, workers); platform, IO-processor, stat-logger and endpoint groups are scoped; `VLLM_PLUGINS` allowlists; one load per process | we register through `vllm.general_plugins` | **same**; write the process-scope facts into the adapter README so nobody registers a worker-only side effect in a group that runs in the API server |
| `distributed_executor_backend` accepts a qualified class name and checks `issubclass(..., Executor)` | our executor subclass is selected that way | **same** |
| `KVConnectorBase_V1`: scheduler side (`get_num_new_matched_tokens`, `update_state_after_alloc`, `request_finished`, `take_events`), worker side (`start_load_kv`, `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`, `get_finished`), plus `handle_preemptions()` for evicted blocks | `CommittedKV.save`/`commit` mirror `save_kv_layer`/`wait_for_save` | **adopt**: add a preemption hook to `CommittedKV` (`evicted(request_id, blocks)`) so the committed-token count drops when the scheduler evicts; put the mapping into `docs/CONTRACT.md` |
| Connector docstring states side-effect freedom for `get_num_new_matched_tokens` and asynchronous saves | our commit is synchronous inside the step guard by design | **keep**; the asynchronous mirror path reports its completion fence, which is the same contract |

## NIXL

| they | we | ruling |
|---|---|---|
| Point-to-point transfer library with backend plugins (UCX, GDS, object store), Python API over a C++ core, Linux only, wheels bundle backends per CUDA version | the mirror service moves committed rows between nodes | **candidate** at the time; the store's replication was then built over its own TCP transport under mutual TLS (`store/src/replication.rs`), and NIXL is not used |

## Practices we were already doing that they also do

Pinned toolchain in the repo; generated schema code checked for freshness; integration tests over real sockets with fake peers; a library crate with a thin binary; typed errors with `# Errors` documentation; tests that spawn real processes for the distributed paths.

## Actions from this review

1. SPDX headers everywhere, gate-checked (this landing).
2. `CancellationToken` and signal handling in the daemon; tests cancel (this landing).
3. `clap` CLI with environment-backed flags and a named-variables module (this landing).
4. Decision 0008: out-of-process daemon, in-process bindings only for kernels.
5. `CommittedKV.evicted` and the connector mapping in `docs/CONTRACT.md` (done).
7. File discovery backend with the daemon publishing its record (done); Kubernetes backend in phase 2.
6. Prometheus in the shared `telemetry/` crate with `/metrics` and `/healthz` on the daemon and the store (done); OTLP behind a feature later; NIXL evaluated on the two-node run.
