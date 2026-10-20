# Architecture

What runs, where, and how the parts speak to each other. The choices behind it are the
decision records in `docs/decisions/` (its README indexes them with their status). This
document describes the tree as it is; a change to a boundary updates it in the same commit.

## What Everserve does

A multi-GPU inference engine's rank set is fixed at start-up: losing one GPU loses the engine
and every session on it. Everserve makes the membership a runtime variable. A rank's node can be
lost or drained, a pre-loaded spare takes the rank, and the sessions continue from their
committed state (decisions 0004, 0013, 0030). The engine's head, the process that holds the
scheduler, is the one position without a spare: its loss restarts the engine, and the router
above moves its streams (decision 0045).

## Packages

Each top-level package has its own manifest, tests, README and CHANGELOG, and imports another
package only through its published API, so any one can be split out with `git subtree split`
and a CI change (decision 0001). The dependency arrow is adapter to core; `core` imports no
engine (decision 0002).

| package | language | what it is |
|---|---|---|
| `core/` | Python, with a C++/CUDA extension in `csrc/` | the engine-independent runtime: the collective a model's traffic goes through (`everserve.comm`), the slot collective and its region, cover plans, the exchange between pipeline stages, the step guard; committed KV through the node's store (`everserve.state`); the contract a process speaks to its node's daemon (`everserve.contract`); the kernels' loader |
| `adapters/vllm/` | Python | the vLLM plugin: the executor that treats a worker's death as an event, the KV connector, the spare and node processes, the start-up replay, the joins; every private vLLM line it reaches is listed in `PRIVATE-INTERNALS.md` for the commit in `PIN` |
| `integrations/dynamo/` | Python | the NVIDIA Dynamo integration: renders an instance's graph and reports its state in Dynamo's terms (decision 0041) |
| `proto/` | protobuf | the one wire schema every process speaks: signals, the coordinator protocol, the mirror service, the policy document, the spec store, discovery; Rust and Python are generated from it |
| `wire/` | Rust | the generated schema types, the frame protocol, the Unix and mutual-TLS transports |
| `telemetry/` | Rust | `/metrics` and `/healthz` for every Rust process |
| `controller/` | Rust | the control plane: the node daemon and the coordinator, the notice sources, the lease, the spec store, node certificates, the operator API, the CLI |
| `store/` | Rust | the KV mirror process on every node: committed rows in shared memory, replication to other nodes' stores, the compile cache |
| `deploy/` | Helm, systemd, Grafana, Docker | the chart, the bare-metal units, the dashboard and alerting rules, the control plane and engine images |
| `bench/` | Python, bash | the harnesses and every run's record, each with its pre-registration |

## Processes at runtime

| process | package | one per | role |
|---|---|---|---|
| engine core (vLLM's `EngineCore`, the API server above it) | the engine, with the adapter loaded as a plugin | engine | unchanged vLLM: the scheduler and the request queue; Everserve's executor beneath it keeps it alive through a change and completes the change in its own step loop |
| worker | the engine, with the adapter | GPU | the model's forward; every collective goes through `core`'s collective, so the group beneath it can be swapped at a step boundary |
| spare | `adapters/vllm` (`spare_main`) | spare GPU, or spare node | a worker built ahead of a loss, from the start-up calls the engine's core sent its rank (decision 0044), standing until a change gives it a position |
| node process | `adapters/vllm` (`node_main`) | engine node | asks the coordinator what its node's GPUs are for, starts the workers or the spare, and leaves on notice |
| node daemon | `controller` | node | the coordinator's agent: holds the links to every process on the node, reports what they say, delivers changes and commits, is issued the node's certificate, reads the node's own preemption notice |
| store | `store` | node | holds each rank's committed KV rows off the GPU, replicates them to the nodes that may restore them, serves a restore to a spare (decisions 0009, 0011, 0016 as amended by 0030) |
| coordinator | `controller` | cluster, under a lease | the one writer of every engine's epoch: decides each change, logs it, drives it through the daemons, keeps the spec store, serves the operator API (decisions 0014, 0036) |

On Kubernetes the coordinator is a Deployment of two under a `coordination.k8s.io` Lease, the
daemon and the store a DaemonSet, the engine's processes the pods Dynamo schedules
(`deploy/kubernetes/README.md`, `integrations/dynamo/README.md`). On bare metal the same
binaries run as systemd units (`deploy/baremetal/README.md`).

## Boundaries

- **core to engine: none.** `core` knows tensors, process groups, slots and rows.
- **adapter to core.** The adapter wires the engine's model, executor and connector to `core`'s
  API, and owns every private-engine line, each listed by file and function for the pin.
- **process to daemon: the contract.** `proto/everserve/proto/signals.proto`, spoken over a
  Unix socket on the node (`core/everserve/contract`): a process registers with its identity
  (decision 0034), hears changes, commits, notices, cover plans and the engine's placement, and
  acknowledges a change once it has moved. Every first message carries the protocol version,
  and another protocol is refused by name (decisions 0027, 0040).
- **daemon and coordinator: `coord.proto`** over mutual TLS across nodes, a Unix socket on one;
  a peer is who its certificate says (decision 0026). The daemon's hello carries everything a
  coordinator that starts or takes the lease needs to rebuild its view (decision 0036).
- **worker and store: `mirror.proto`** on a Unix socket for control; the rows cross in shared
  memory and never the socket (decision 0011). Stores replicate to each other over mutual TLS.
- **operator and coordinator: the operator API**, REST over TLS, every caller authenticated and
  authorized by the cluster or by certificate (decisions 0039, 0042); `docs/api/openapi.json`
  is generated from the routes and tested against them.

## A change, end to end

1. **Detection.** A lost rank is seen at the collective (decision 0021): the survivors' slot
   kernel names the row it waits on past a bound, the executor reports it, and the engine core
   stays up. A node is lost when its daemon's link ends without a goodbye or misses its
   heartbeats (decision 0023). A notice (a cloud's preemption signal, an operator's drain)
   drives the same change before the loss.
2. **The claim.** The coordinator chooses the outcome by policy (REPLACE with a spare; SHRINK
   is the reference tier, decision 0004), logs the change in the spec store, and delivers it to
   every daemon of the engine, which hands it to each process (decision 0038's change log,
   which stands).
3. **The move.** At a step boundary each survivor swaps to a group pre-formed under a cover plan
   (decision 0035); the spare restores the lost rank's rows from its node's store and joins
   through the engine's own queues (decision 0024). Each process acknowledges once moved.
4. **The commit.** With every acknowledgement in, the coordinator logs and sends the commit;
   the engine core completes the change in its own step loop and resumes every request from
   its committed position, one step at a time, in the same shape (decision 0013).

The numbers: a REPLACE under load in 412 ms with every stream continuing
(`bench/runs/2026-10-14-dynamo-proof`), 0.64 s with notice and 1.14 s on a crash on the current
tree (`bench/runs/2026-10-19-dynamo-proof-known-gap`). `bench/runs/README.md` indexes every
proof.

## Build and gates

- `scripts/preflight.sh`: format, lint, the schema's freshness, the engine-free tests, judged
  by exit code. It builds the store and the controller first, since the Python tests drive the
  real binaries.
- `scripts/prepush.sh`: preflight with the slow tier, and the doc build with warnings denied.
- CI (`.github/workflows/ci.yml`): every crate's and package's tests, the chart against a real
  API server, the alerting rules under Prometheus's own tool, the published API document
  against the routes, the dependencies' advisories and licences, the image's SBOM, and the
  worker-loss end to end on the real control plane.
- A release (`.github/workflows/release.yml`): a tag publishes the image and the chart at one
  version, each signed by its digest.

GPU runs are measurements, not tests: each is pre-registered under `bench/runs/` and its
record committed before any number is written in prose.
