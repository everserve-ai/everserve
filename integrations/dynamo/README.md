# Everserve under NVIDIA Dynamo

Everything Dynamo-specific lives here (decision 0041). Dynamo decides *how many* instances there
are and where each request goes. Everserve keeps each instance serving while its nodes change:
it replaces a lost or drained rank in place, and Dynamo's router sees nothing.

## An instance

An instance is described in one JSON file:

```json
{
  "name": "llama",
  "namespace": "serving",
  "image": "<registry>/everserve-engine:<tag>",
  "dynamoRuntimeVersion": "1.6.0",
  "imagePullSecrets": ["<registry secret>"],
  "engineArgs": ["--model", "/models/llama", "--tensor-parallel-size", "8",
                 "--pipeline-parallel-size", "2", "--max-model-len", "8192",
                 "--num-gpu-blocks-override", "16384", "--block-size", "16"],
  "gpusPerNode": 8,
  "spareNodes": 1,
  "guarantee": "exact",
  "head": {"nodeSelector": {"<on-demand label>": "<value>"}},
  "nodes": {"nodeSelector": {"<spot label>": "<value>"}, "tolerations": []},
  "nodeResources": {"rdma/ib": "1"},
  "metricsNamespaces": ["monitoring"]
}
```

- `image` is Everserve's engine image, built from `deploy/images/engine`. It carries Dynamo's
  vLLM runtime and the adapter.
- `dynamoRuntimeVersion` is the Dynamo runtime that image carries (`deploy/images/engine/
  DYNAMO_PIN`). Dynamo's operator reads that version from an image's tag, and an Everserve
  image's tag is never one, so the renderer refuses an instance without it unless the image's
  tag is a version. It is `1.6.0` beside a platform at 1.5.1 because the pin is a commit of
  Dynamo's main after the 1.5.1 release, the first whose vLLM runtime is the adapter's vLLM
  0.31.0, and that commit's runtime reports itself as 1.6.0.
- `migrationLimit` (default 3) is how many times Dynamo's frontend re-sends a broken stream to
  another instance, with the tokens it has generated (`DYN_MIGRATION_LIMIT`; Dynamo 1.5.1 has it
  off, 0, by default). A lost head breaks every stream it serves, and only migration continues
  them:
  - **With two or more engines behind the frontend**, the broken streams continue on another one
    at once. That engine prefills each stream's whole context, and the client sees one stream.
  - **With a single engine**, nothing in Dynamo 1.5.1 lets a request wait for an instance.
    Until the engine restarts, migration finds no instance and spends its retries at once, so
    those streams fail. Run two engines where a client must never see a head's loss
    (`bench/runs/2026-10-15-dynamo-migration-read`).

  The head's loss is a known gap (decision 0045): Everserve replaces any other node's ranks with
  a spare, but not the head's, whose loss restarts the engine. Migration is what continues its
  streams.

  0 turns migration off.
- `globalDynamoNamespace` (default false) puts every component of the instance in Dynamo's
  global namespace rather than its graph's own. Instances so placed serve their model behind
  every one of their frontends, which is how a second engine takes the streams a lost head
  breaks. The renderer then admits the namespace's Dynamo frontends to the instance's head, by a
  NetworkPolicy of its own (`<name>-frontends`); the engine's nodes stay reachable only from the
  instance's own pods.
- `engineArgs` are vLLM's, the same for every process of the instance. They must state the KV
  cache: `--num-gpu-blocks-override` and `--block-size`. Each node's store keeps a row for every
  slot the engine can write, so the two must agree, and vLLM otherwise sizes the cache from the
  GPU's free memory only once it runs (see "Sizing the node store").
- `imagePullSecrets` are for a private registry. `metricsNamespaces` are the namespaces (a
  Prometheus's) allowed to reach the engine pods' metrics and health ports.

Render it with:

```
python -m everserve_dynamo instance.json --out deploy/
```

This writes three files:

- `deploy/llama.k8s.json`, which you apply with `kubectl apply -f`. It contains:
  - a DynamoGraphDeployment (`nvidia.com/v1beta1`) with three components:
    - `Frontend`: Dynamo's frontend, unchanged.
    - `head`: Dynamo's vLLM worker as the engine's node 0, running Everserve's executor. It is the only pod that registers the instance with Dynamo, and `headOnDemand` places it on capacity that is not preempted (0041): its node's loss restarts the engine (0045).
    - `nodes`: the engine's other nodes plus its spare nodes.
  - a headless Service, `<name>-rendezvous`. The nodes reach the head's rendezvous through it before the head is ready. Dynamo's operator keeps a Service of its own named for each component (`<name>-<component>`), so ours never takes such a name.
  - a NetworkPolicy, `<name>-engine`. Only the instance's own pods reach the head and the engine's nodes, whose transports (vLLM's rendezvous store, its workers' output queues, NCCL's sockets) carry no authentication of their own. The namespaces in `metricsNamespaces` reach their metrics and health ports alone. The frontend is left open to its clients.
- `deploy/llama.values.json`: values for Everserve's chart (`helm install … -f deploy/llama.values.json`). They hold the node stores' `node.store.slots`, the engine's KV blocks times its block size.
- `deploy/llama.engine.json`: the engine's entry for Everserve's policy, which you add to the chart's `policy.engines`. It is shape only (0036): the widths and the head's placement, no members. The chart's policy must enable cover (`"cover": {"enabled": true}`), and both the coordinator and the processes refuse it otherwise: the instance's spares stand after bring-up, and only the sets cover plans as they stand can replace a rank.

Each pod of `nodes` registers with its node's daemon as undecided. The control plane tells it either to serve a rank or to stand as a spare. When Grove recreates a lost pod, the new pod becomes the instance's next spare, never a second holder of the rank.

The instance uses two worker components rather than Dynamo's `multinode`. With `multinode`, the leader would be a GPU worker launched with vLLM's own executor backend, and a spare node would join the instance's collectives.

Every GPU pod meets the pod contract in `everserve.podcontract`:
- the node daemon's socket directory, the node's shared-memory directory and its compiled-kernel directory (`/var/lib/everserve/vllm-cache`, vLLM's `VLLM_CACHE_ROOT` in the pod, decision 0043), each mounted from the host at its own path;
- the node's name and the pod's address, taken from the downward API.

## Sizing the node store

Each GPU node's store keeps the KV rows of the ranks it holds, so a rank lost on one node can be
restored on another. Two chart values size it, and the store refuses to start, or refuses the
engine when it registers, when either is wrong:

- `node.store.slots`: the engine's KV blocks times its block size. The renderer writes it to
  `<name>.values.json` from `engineArgs`.
- `node.store.budgetBytes`: the bytes of the store's tables, in the node's shared memory
  (`/dev/shm`). It must hold the tables of every rank the node's store keeps: its own ranks', and
  those of the ranks it replicates for (under decision 0030, the engine's other members'). The
  renderer cannot compute it, because it depends on the model's shape. For each rank held:

  ```
  layers on the rank × KV heads on the rank × slots × 2 (K and V) × head_dim × the KV dtype's bytes
  ```

  For example, Qwen2.5-0.5B (24 layers, 2 KV heads, head_dim 64, bf16) at pipeline width 2 has 12
  layers per rank. At 65,536 slots, one rank is 12 × 2 × 65,536 × 2 × 64 × 2 B ≈ 403 MB. A node
  holding its own rank and one other's needs about 805 MB.

The chart derives the store container's memory limit from `budgetBytes`, and the node's
`/dev/shm` must have room for it.

## When a node goes, and when something fails

A node pod being deleted (a drain, an eviction, a spot reclaim) is told by SIGTERM. Then:

- **A serving node** gives its ranks notice, 5 s ahead and never less than the policy's
  `noticeMinMs`. It keeps serving until the coordinator has moved them to their spares, at most
  60 s, and stops within the pod's 90 s grace period.
- **A spare node whose spare took a rank** does the same for that rank. Its node process tells
  the spare over a pipe it owns.
- **A node that dies without notice** has its loss reported as its workers' departure, and the
  change follows.

What an operator sees, and what to do:

| Seen | Meaning | What to do |
|---|---|---|
| node pod exits **2** | A guard refused it, and that is permanent: its step configuration differs from its engine's (`StepRefused`), it was started for another guarantee class (`GuaranteeRefused`), or a spare's compiled set could not be staged. The engine's events name the reason. | Fix the instance (`engineArgs`, `guarantee`, the image) and render it again. Kubernetes restarts the pod as it would any exit, and it is refused again until the instance is fixed. |
| node pod exits **3** | Its engine is down (the head stopped or failed, or a change failed after it was delivered). The coordinator starts the engine anew once no process of the old one is left. | Nothing. The pods restart, ask again, and serve once the engine is up (`EngineRestarted`). Streams in flight when the engine went down fail. |
| a worker exits **4** | A later commit displaced it (`EXIT_STALE`): its rank serves elsewhere, for instance a node that came back after a takeover committed a change it never heard (`StaleRanks` in the engine's events). Where its node ends released processes, the worker is released instead and its node ends once it holds nothing. | Nothing. The pod restarts, asks again, and stands or serves at the current epoch. |
| coordinator exits **75** | Its hold of the lease ended (`EX_TEMPFAIL`): another replica leads. | Nothing. It restarts and stands by. |
| head will not start: "Everserve's step hook was not installed" | The `everserve` vLLM plugin did not load in the engine core's process, so the engine would serve without membership changes. | Check that the image has `everserve-vllm` installed, and that `VLLM_PLUGINS`, where it is set, names `everserve`. |
| `Degraded`, reason `NotServingChanges` | The registered head did not say it serves membership changes, so the coordinator changes nothing on this engine. | As the line above. |
| `Covered` False, `Forming` | Every loss without a formed set has a planned one still forming (a spare that stood up recently). The message lists the losses. | Wait. A spare's set forms within about a minute of the spare starting. |
| `Covered` False, `LossUncovered` | At least one loss has no planned set at all: no spare can take it. The message lists the losses. A loss of the ranks the head's node holds is never among them: it restarts the engine (`Down` below, decision 0045), and the message names those ranks ("the head's node holds […]"). | Add spare nodes. |
| `Down` True | The engine is down: its head stopped or failed (`HeadStopped`, `HeadFailed`), or a change failed after it was delivered (`ChangeFailed`). Losing the head's pod is this, not a replacement. | Nothing. The engine is started anew once its old processes are gone (`EngineRestarted`), and streams in flight when it went down fail. |
| event `RefusedUncovered` | A loss came before any set covering it had formed, so its rank could not be replaced. | As for `Forming` and `LossUncovered`. |

The engine's state is `GET /v1/engines/<name>` on the coordinator: its conditions, the last
change (`lastChange`) and its cover. Its events are `GET /v1/engines/<name>/events`, and
`everserve-controller get` reads both. Every event above is there with its reason and the
coordinator's words.

## Requirements

- Dynamo's operator, at a release serving `nvidia.com/v1beta1`.
- Everserve's control plane (`deploy/kubernetes`) and its node DaemonSet on every GPU node the instance may use.
- An image with Dynamo's vLLM runtime at vLLM 0.31.0 and `everserve-vllm` installed.
- A trusted network between an instance's nodes, as vLLM's own multi-node serving requires.
  The engine core and its workers exchange pickled Python objects over vLLM's queues with no
  authentication, and a spare reads the core's start-up calls the same way: what the core sent
  its workers when it configured their KV caches, replayed to the spare before it warms
  (decision 0044). Only the queue's address crosses the control plane, as declared JSON.

**A spare builds only while a head stands.** A spare is built from the calls the engine's core
sent its workers, so one starting while no core holds the head (the engine's bring-up, a
restart) waits for the next core before it can warm, asking again within its bound.
Spares already standing are unaffected. A spare whose configuration differs from the engine's is
refused before anything moves (`SpareRefused`), and every join names the calls of the core that
offers it, which the spare checks against its own as a last guard.

## Built and not

- **Built, and shown on GPUs** (`bench/harness/k8s/dynamo_proof.sh`, three one-GPU nodes, L40S
  or H100, a pipeline-parallel engine under Dynamo's platform 1.5.1):
  - The serving node taken away under load, and its rank taken by the spare in 412 ms. Every
    stream through Dynamo's frontend finished:
    [2026-10-14-dynamo-proof](../../bench/runs/2026-10-14-dynamo-proof/RESULT.md).
  - The same rank taken away twice, the second time from the spare that took it, both by notice:
    [2026-10-14-dynamo-proof-pipe-leave](../../bench/runs/2026-10-14-dynamo-proof-pipe-leave/RESULT.md) and
    [2026-10-14-dynamo-proof-s1](../../bench/runs/2026-10-14-dynamo-proof-s1/RESULT.md).
  - The renderer; the node role's undecided registration for its slot (`node_main --slot`), and
    the head's for node 0.
  - The engine's policy, placement and state on every process's link, so no pod mounts a policy
    file.
  - The head's executor given as `mp` and resolved to Everserve's in the engine's process (vLLM
    0.31 refuses a pipeline-parallel engine whose executor is named by its class's path).
  - A node that dies rather than being deleted, twice, without notice:
    [2026-10-15-dynamo-proof-crash-3](../../bench/runs/2026-10-15-dynamo-proof-crash-3/RESULT.md).
  - On the tree without the head's failover, a deletion then a crash, each spare built from the
    core's replayed start-up calls (decision 0044), the worst gap between tokens 0.642 s on the
    notice and 1.135 s on the crash:
    [2026-10-19-dynamo-proof-known-gap](../../bench/runs/2026-10-19-dynamo-proof-known-gap/RESULT.md).
- **Not shown yet:** a production-size model, and more than one node per pipeline stage.
- **A known gap:** the head's own loss restarts the engine (decision 0045).
- **Designed, not built:** reporting the instance's state to Dynamo (its health and metrics), and
  translating Dynamo's signals (0041's four obligations, beyond publication).

## Dynamo's schema

`schema/dynamographdeployment-v1beta1.json` is the `v1beta1` `openAPIV3Schema` from Dynamo's CRD, `deploy/operator/config/crd/bases/nvidia.com_dynamographdeployments.yaml` at `ai-dynamo/dynamo` commit `a6df4fbbf575a7ea46f50ebdf1c9bad042568eb6`. It is © NVIDIA Corporation and licensed Apache-2.0. The tests validate every rendered graph against it.
