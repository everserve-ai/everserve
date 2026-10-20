# Everserve

The runtime that keeps a multi-GPU inference engine serving while the GPUs under it change.
A rank can leave, a spare can join, and sessions do not notice.

Today an engine's rank set is fixed at startup, so a 16-GPU model lives or dies with every one
of its nodes. Everserve makes membership a runtime variable. The consequence: frontier models on
preemptible capacity, at the spot price, without a replica of headroom per replica.

## Layout

| path | what | language |
|---|---|---|
| `core/` | engine-independent runtime: communicators, placement, arithmetic, committed state, signals | Python API over a C++/CUDA extension |
| `adapters/vllm/` | the vLLM plugin; every private-engine line listed in `PRIVATE-INTERNALS.md` | Python |
| `integrations/dynamo/` | the Dynamo integration: renders an instance's graph, reports its state in Dynamo's terms (decision 0041) | Python |
| `controller/` | node daemon and coordinator: notices, spare pool, rejoin, the spec store, node certificates, the operator API, metrics, CLI | Rust |
| `store/` | the KV mirror process | Rust |
| `wire/` | the one wire schema (`proto/`), the frame protocol, the Unix and mutual-TLS transports | Rust |
| `telemetry/` | `/metrics` and `/healthz` for every Rust process | Rust |
| `proto/` | the schema every process speaks, Rust and Python generated from it | protobuf |
| `deploy/` | the Kubernetes chart, bare-metal units, the Grafana dashboard and alerting rules | |
| `bench/` | drills and measurements; every result with its pre-registration | scripts |
| `docs/` | architecture, the recovery ladder, the adapter contract, decisions | |

Each package is splittable on its own: own manifest, tests, README and CHANGELOG, and no
imports across package boundaries except through the other package's published API. The
dependency arrow is adapter -> core. `core` never imports an engine.

## Status

What is shown, and where; nothing here is claimed past its record. `bench/runs/README.md`
indexes every proof.

- **Under NVIDIA Dynamo on Kubernetes, with GPUs:** a node of a serving instance taken away under
  load, its rank taken by a spare in 412 ms, every stream continuing
  (`bench/runs/2026-10-14-dynamo-proof/`, pass); two losses, a deletion then a crash, each taken
  by a spare built from the engine core's own start-up calls, 0.64 s with notice and 1.14 s on a
  crash, on the current tree (`bench/runs/2026-10-19-dynamo-proof-known-gap/`, pre-registered,
  three H100, pass).
- **On GPUs, under vLLM, across nodes:** ranks drained off a node onto preloaded spares while the
  engine serves, a stored drain resumed after a coordinator restart, a second head refused
  (`bench/runs/2026-10-13-run3-drains-smoke/`, pre-registered, five L40S VMs, pass).
- **On Kubernetes, without GPUs:** the chart and the node DaemonSet installed on a managed
  cluster, each node issued its certificate by the coordinator, one process per pipeline stage
  told to serve and the rest to stand, the placement reaching every process, a coordinator's
  restart keeping its decisions through the Kubernetes spec store
  (`bench/runs/2026-10-13-k8s-e2e/`, three CPU nodes on managed Kubernetes 1.33, pass).
- **In CI:** every crate's and package's tests, the chart and its roles against a real API
  server, the alerting rules loaded and tested by Prometheus's own tool, the published API
  document against the routes that serve it, the dependencies' advisories and licences
  (`deny.toml`, pip-audit), the image's SBOM.
- **Released:** a tag publishes the image and the chart, each signed by its digest with the
  release workflow's identity (`.github/workflows/release.yml`); run as a dry run, not yet on a
  tag.
- **The known gap:** the engine's head, the process holding the scheduler, has no spare. Its
  node's loss restarts the engine, and the router above (Dynamo's frontend) moves its streams to
  another instance; with one instance, those streams fail (decision 0045, which also names what
  would close it). An operator who wants the head's node off spot sets `headOnDemand` and pays
  one on-demand node per engine.

Operating it: `docs/RUNBOOK.md` (reaching the API, moving work off a node, every alert),
`docs/api/openapi.json` (every route and object, decision 0042), `deploy/kubernetes/` and
`deploy/baremetal/` (installing).

## History

The repository was published on 2026-10-20 as one commit, the tree of a private repository
developed from 2026-10-01 (decision 0046). That repository stays private as the archive of the
development history. The decision records (`docs/decisions/`) and the run records
(`bench/runs/`) cite commits of it by hash: those hashes are evidence in the archive, not links
into this history. What landed in each package is in its `CHANGELOG.md`.

## Working here

- Inner loop: `scripts/preflight.sh`. Judged by exit code. `CONTRIBUTING.md` lists what it
  needs on `PATH`.
- Before a push: `scripts/prepush.sh`. Push only on its green.
- GPU runs: `bench/`, pre-registered, results committed before any number is written in prose.
- Nothing under `vllm/` is ever edited. The adapter lists every private line it depends on.

See `docs/ARCHITECTURE.md` (what runs where), `docs/decisions/` (why, indexed with each
record's status), `docs/GLOSSARY.md` (the words), `docs/RUNBOOK.md` (operating it) and
`docs/ROADMAP.md` (what is next). Apache-2.0 (`LICENSE`); security reports go through
`SECURITY.md`.
