# Everserve

Everserve keeps a multi-GPU vLLM engine serving when one of its nodes goes away.

A large model is usually served by one engine spread over several GPUs, often over several
machines. vLLM fixes that set of GPUs when the engine starts. If any one of them is lost, to a
spot reclaim, a hardware fault or a drain, the whole engine dies and every request in flight
dies with it. Everserve adds a control plane and a vLLM plugin that let a lost GPU's place be
taken by a spare while the engine keeps running, and the requests that were in flight continue
from where they were.

It is built for NVIDIA Dynamo on Kubernetes, and is Apache-2.0.

## How it works

1. **Spares stand ready.** Beside each engine, extra GPU nodes run spare worker processes.
   A spare loads the model and is built from the exact start-up calls the engine's core sent to
   the rank it stands for, so it is a drop-in copy of that worker, warm and waiting.
2. **Each node keeps a copy of its KV cache off the GPU.** As the engine runs, every worker
   writes its committed KV blocks to a small store process on its node, in shared memory. The
   stores replicate those rows to the nodes whose spares might need them.
3. **A loss is detected at the collective.** The workers' all-reduce goes through Everserve's
   own collective, which notices a dead member within a few milliseconds instead of hanging.
   A planned loss, such as a spot reclaim with notice or a drain, is reported ahead of time.
4. **The coordinator commits a change.** One coordinator process decides the new membership,
   logs it, and delivers it to every worker through a daemon on each node.
5. **The spare takes the rank at a step boundary.** The surviving workers swap to a new
   process group, the spare restores the lost rank's KV rows from the stores, and the engine's
   scheduler resumes the running requests from their committed tokens. Clients see a pause of
   under a second, not an error.

One position has no spare: the engine's head, the process that holds the scheduler. If its node
is lost the engine restarts, and the router above it (Dynamo's frontend) moves the streams to
another engine. The roadmap says what closing that gap needs from vLLM and Dynamo.

## What it has been shown to do

Every number below comes from a run whose predictions were written down before it started and
whose record is in this repository (`bench/runs/README.md` indexes them all).

| What | Result | Record |
|---|---|---|
| A node pod deleted under 8 concurrent streams, under NVIDIA Dynamo on Kubernetes | the rank taken by a spare in 412 ms, every stream completed | `bench/runs/2026-10-14-dynamo-proof/` |
| The same, twice: a deletion with notice, then a SIGKILL of the spare that took over, 3 × H100 | worst pauses 0.63 s and 1.18 s, about 2 ms to detect the crash, all 16 streams completed | `bench/runs/2026-10-20-dynamo-proof-oss-polish/` |
| Ranks drained off a node onto spares while serving, across five nodes, then a coordinator restart mid-drain | the drain resumed from its stored state; a second head refused | `bench/runs/2026-10-13-run3-drains-smoke/` |
| The control plane installed on managed Kubernetes without GPUs | certificates issued to every node, placement delivered to every process, the coordinator's decisions kept across its restart | `bench/runs/2026-10-13-k8s-e2e/` |

Each run used one engine of a small model (Qwen2.5-0.5B, tensor-parallel 1, pipeline-parallel
2). Larger models and wider layouts are supported by the design and not yet measured.

## What you need

- A Kubernetes cluster with GPU nodes (Kubernetes 1.30 or later), NVIDIA Dynamo's platform
  installed on it, and `helm` and `kubectl` on your machine.
- The control-plane image. A release publishes it to `ghcr.io/everserve-ai/everserve`, signed;
  the chart pulls it by default.
- The engine image, which carries Dynamo's vLLM runtime at vLLM 0.31.0, the Everserve plugin and
  the prebuilt kernels. It is built from `deploy/images/engine` on any Docker host and pushed to
  a registry your cluster can pull from. No engine image is published yet.
- Spare capacity: at least one more GPU node than the engine needs.

Everserve runs with vLLM 0.31.0 only. The plugin reaches into vLLM in a number of places that
are not public API, and every one of them is listed with its reason in
`adapters/vllm/PRIVATE-INTERNALS.md`. A nightly job runs the plugin's tests against vLLM's main
branch so that a break there is seen the day it happens.

## Try it

The whole path, from an empty cluster to a node loss under load, is scripted in
`bench/harness/k8s/dynamo_proof.sh`. It rents a cluster on Nebius, installs everything, runs the
losses and takes the cluster down; it is the most reliable guide to the steps. By hand:

1. **Build the engine image** and push it:
   `deploy/images/engine/build.sh --tag <registry>/everserve-engine:<tag> --push`
2. **Install the control plane** (the coordinator, and a daemon and a store on every node).
   `deploy/kubernetes/README.md` has the four steps: the TLS secret, the namespace, the policy
   and `helm install`.
3. **Describe your engine** in one JSON file: model, vLLM arguments, GPUs per node, how many
   spare nodes. `integrations/dynamo/README.md` shows the file.
4. **Render and apply it:**
   ```
   python -m everserve_dynamo instance.json --out deploy/
   kubectl apply -f deploy/<name>.k8s.json
   ```
   The renderer also writes the chart values for the node stores and the engine's entry for
   the control plane's policy.
5. **Watch it.** `everserve-controller` is the command-line client for the coordinator's API:
   which engines are up, which ranks are covered by a spare, every change and why. The runbook
   (`docs/RUNBOOK.md`) covers reaching the API, draining a node on purpose, and every alert the
   Grafana rules can raise.

Then delete one of the engine's node pods and keep sending requests.

## What it does not do yet

- **The head's node.** Losing it restarts the engine (decision 0045). Put the head on a node
  that is not preempted (`headOnDemand`), or run two engines behind Dynamo's frontend so its
  streams migrate.
- **Growing back.** A lost rank is replaced by a spare. When no spare is left the engine can
  shrink to a narrower layout for models that allow it, but it does not grow again when
  capacity returns.
- **Other engines.** Only vLLM, and only through this plugin. `docs/CONTRACT.md` describes what
  an engine has to expose for Everserve to keep it serving, as a proposal for upstream.
- **Other orchestrators.** Kubernetes with Dynamo, or bare metal with systemd units
  (`deploy/baremetal/`). Nothing else is integrated.

`docs/ROADMAP.md` lists what comes next.

## The repository

| Directory | What it holds |
|---|---|
| `controller/` | The control plane, in Rust: the coordinator, the node daemon, the operator API and CLI |
| `store/` | The per-node KV store, in Rust |
| `wire/`, `proto/`, `telemetry/` | The protobuf schema every process speaks, its transports, and metrics |
| `core/` | The engine-independent runtime, in Python with a CUDA extension: the collective, the KV commit path, the contract between a process and its daemon |
| `adapters/vllm/` | The vLLM plugin |
| `integrations/dynamo/` | The NVIDIA Dynamo integration: renders an engine's deployment and reports its state in Dynamo's terms |
| `deploy/` | The Helm chart, the images, bare-metal units, the Grafana dashboard and alerts |
| `bench/` | The test harnesses and the record of every GPU run |
| `docs/` | Architecture, the runbook, the glossary, the roadmap, and 46 dated decision records explaining why things are the way they are |

Start with `docs/ARCHITECTURE.md` for what runs where, and `docs/GLOSSARY.md` for the words
(rank, spare, head, notice, cover set) that the rest of the documentation uses without
explanation.

## Contributing

`CONTRIBUTING.md` has the setup and the rules. The short version: `scripts/preflight.sh` is the
inner loop and `scripts/prepush.sh` the gate before a push; nothing under vLLM's own source is
ever edited; every GPU number has a pre-registered record; commits are signed off
(`git commit -s`). Security reports go through `SECURITY.md`.

## History

This repository was published on 2026-10-20 as a single commit, taken from a private repository
where Everserve was developed from October 2026. That repository stays private. The decision
records and the run records cite commits of it by hash; treat those hashes as references into
the archive, not links into this history. What landed in each package is in its `CHANGELOG.md`.
