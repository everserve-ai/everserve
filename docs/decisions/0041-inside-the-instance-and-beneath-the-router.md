# 0041: inside the instance, beneath the router

2026-10-13. Status: amended by [0045](0045-the-heads-loss-is-a-known-gap.md) (2026-10-19); current as
amended: the head does not fail over; its loss restarts the engine, and the router migrates its
streams. Where this record says otherwise, 0045 holds. Built, both halves: the control plane's
(undecided processes decided once, the policy, the placement and the instance's state pushed to
every process, the cluster's Nodes followed) and the Dynamo integration's (`integrations/dynamo`);
its GPU proofs are `bench/runs/2026-10-14-dynamo-proof` and the dated `dynamo-proof-*` directories
after it, through `bench/runs/2026-10-19-dynamo-proof-known-gap`.

**Context.** Serving frameworks orchestrate above engines: NVIDIA Dynamo, llm-d and KServe own
routing, request scheduling and instance counts across a fleet. Everserve keeps one instance
serving while its nodes change: a lost or drained GPU's rank is replaced in place, the head fails
over (0038), and the engine never restarts.

The frameworks treat an instance as the unit of failure. llm-d's own resilience proposal says
that today "the entire instance restarts with minutes of full downtime and all in-flight requests
dropped" [1]. A multi-node instance runs as a gang: a LeaderWorkerSet group, whose default restart
policy recreates every pod of the group when one is recreated [2], and which KServe sets in code
[3]; or a Grove replica under Dynamo, terminated whole once it stays below its floor past a delay
[4]. Dynamo's request migration, off by default, re-sends a broken stream's prompt and generated
tokens to another instance [5]. None of them replaces a rank of a running tensor- or
pipeline-parallel instance. The two are complementary only if each knows where the other stops,
and a customer who runs one of these frameworks must be able to add Everserve without changing
their router.

**Decision.**

## Dynamo first, and only Dynamo built

- The MVP integrates with Dynamo, and builds that integration only. llm-d and KServe are designed
  for, so the contract below must fit them, and nothing of theirs is built: no LeaderWorkerSet
  backend, no endpoint-picker readiness logic, no resilience-operator adapter.
- Standalone, Everserve under stock `vllm serve` with its plugin and no framework, stays possible
  as it is today, with nothing built for it. 0038's stateless front end leaves the MVP.
- The product's target is CoreWeave's Kubernetes service, on spot nodes [14]; the proofs ran on
  Nebius managed Kubernetes in eu-north1 (`bench/runs/2026-10-14-dynamo-proof/RESULT.md`).

## The layering rule

- The framework owns every decision between instances: how many there are, which one a request
  goes to, and moving a request from one to another.
- Everserve owns every decision within an instance: which GPU holds which rank, replacing a lost
  or drained rank, the head and its failover, cover (0035), spot or on-demand, spares and drains.
- An instance keeps its endpoint, its model and its API through every change Everserve makes
  inside it.
- Everserve never hides a change in capacity. An instance serving degraded or smaller (SHRINK, an
  uncovered loss) says so in its vLLM-compatible metrics and in its status, so the framework's
  router and planner see it.
- Everserve never scales instance counts and never routes between instances.

## Lifecycle beats placement

The two control planes share no verb. The framework owns an instance's existence: it creates and
deletes replicas. Everserve owns where an existing instance runs. In integrated mode Everserve's
API has no verb that creates or deletes a framework's instance, and the framework has none for
placement, so neither can undo the other.

Where the two meet in time, lifecycle wins. A framework that deletes an instance while Everserve
is changing it ends the change: it fails as a change does after its delivery (0032), the
instance's spares are released, and Everserve's API refuses any further action on an instance
being torn down.

## Two loops, joined only by information

- *How many instances* is the framework's lever. A customer's autoscaler talks to Dynamo, or to
  Kubernetes, as it would without Everserve, and each new instance comes up protected.
- *How an instance sits on hardware* is Everserve's: spot or on-demand, its spares, its drains,
  driven through Everserve's API and never through the framework.
- Neither loop calls the other. Everserve's metrics and status show a degraded or resized instance
  to the framework's planner; the framework's scaling creates pods Everserve then protects.

## Within-instance resize: designed for, not built

Shrinking or growing one instance (an engine on eight nodes, under-used, giving four back) is
placement within an instance, so it is Everserve's. Its entry point is the engine's size in its
spec, on the one API, set by an operator, a customer's autoscaler or, later, Everserve's own
policy; the framework sees the change through metrics and may add replicas. The first form is
data-parallel: one data-parallel group retires, and its in-flight requests' KV moves to the
survivors by the restore path spares use, with no weights moved. Resharding tensor or pipeline
parallelism over fewer nodes (weights and KV laid out again, kernels recompiled, a fit check) is
research and out of scope. Today Everserve replaces at constant size, and the runtime can shrink a
set; an instance resize does not exist. Outside the MVP.

## One interface

Operators, dashboards, gateways, autoscalers and frameworks see one model: engines, nodes, pools,
drains, the head, cover and release, each a resource with a spec (desired, written through the
API) and a status (observed, written by the control plane, with conditions; 0036). The same verbs
and the same authentication (0039) serve every caller:

- every resource is listed and read, and watched with one list+watch mechanism (a resource
  version on every object, as Kubernetes' watch), never by a special long-poll per question;
- a write names its resource in its path, so RBAC can scope a caller to the nodes or pools it may
  touch (`resourceNames`); a drain is a node's spec, not an action whose body names the node;
- what an action asks for is spec, and what came of it is status: a node's `cordoned` and
  `drain { deadline }`, and its conditions Cordoned, Draining, Blocked (with the reason) and
  Released.

On Kubernetes the native forms mirror this model field for field and are not a second design: a
custom resource per kind with its spec and status subresource, and a disruption budget that holds
an eviction until the node's Released condition.

## The framework-integration contract

Anything specific to a framework lives in its own folder behind one contract, as anything specific
to an engine does (0034): `adapters/<engine>/` implements the engine contract, and
`integrations/<framework>/` implements the framework-integration contract, defined in the core. The
MVP has `integrations/dynamo/` only. No framework's name appears outside its folder. Every
integration meets four obligations:

1. *Publish the instance the way the framework discovers it*, and keep it published through every
   change inside the instance. Dynamo: the worker registration under the leader runtime's lease
   [7]. llm-d (designed for): the pool's labels and readiness on the head's pod [6].
2. *Report the instance's state in the framework's terms*: serving, recovering, degraded (capacity
   down), failed; and starting, before every position is held, which a framework reads as not
   yet registered. Built, the control plane's half (2026-10-13): the coordinator derives it in
   one place from the engine's view, the first that holds of failed (down, 0032), recovering (a
   change in flight), degraded (ranks lost under SHRINK), serving and starting, and sends it
   (`InstanceState`) through each daemon to every process of the engine as it changes, before a
   change's first delivery; readiness reports it too.
3. *Translate the framework's disruption and scaling signals into calls on the one API*, never
   into a side channel.
4. *Own the framework's pod group through the launcher seam*. Dynamo: Grove. llm-d (designed for):
   LeaderWorkerSet.

The contract's control-plane half (state, signals, the API calls) and its engine-facing half
(publication, the worker's registration) are held by the control plane's and the engine's owners
respectively, and `integrations/dynamo/` is built by both, each piece by the owner of the half it
touches.

## Touch points with Dynamo

Each is a contract the integration meets with what Dynamo already reads, or a named upstream
change where it reads nothing yet.

- *The instance's registration survives a rank replacement.* Dynamo routes to the instance its
  leader runtime registered [7]. Replacing a non-head rank leaves the leader and its lease alone,
  so Dynamo's router sees nothing. The MVP's first proof: a non-head node lost or drained under
  Dynamo, sessions continuing, Dynamo's router unaware. No Dynamo change is needed.
- *Grove keeps the replica through a replacement.* Grove terminates a replica only once a clique
  has stayed below its `minAvailable` past the replica set's `terminationDelay`, four hours by
  default [4]; a replacement commits in seconds. Grove recreates the lost pod [4], and by then
  Everserve's spare holds the lost rank. The pod's template runs a worker for that rank, with the
  node count and the leader's address the operator injected, which would join the instance's
  collectives as a second holder of the rank. So the integration owns what a group pod's process
  becomes: it starts as Everserve's process, which asks the control plane whether its rank is
  held, and stands as a spare if it is, or serves as the rank's worker at bring-up if it is not.
  The recreated pod is then the instance's spare for its next loss.
- *Every process of the instance resolves one configuration.* Dynamo's vLLM worker adds to the
  configuration it starts the engine with (it sets `additional_config`'s
  `consolidator_endpoints`, None without its KV block manager), and vLLM hashes that into its
  configuration and compile cache keys; a node resolving the same engine arguments without it
  differed from its head there alone, and the step guard refused the head (the Dynamo proof,
  2026-10-14). The integration gives every process of the instance the same `--additional-config`
  (`everserve_dynamo.vllm.engine_args`), and the engine image's build refuses a Dynamo whose
  worker adds a key the renderer does not render. With Dynamo's KV block manager the endpoints
  are the head's own and no node could state them: that mode is not supported until the head
  hands its resolved configuration to its nodes.
- *Which process serves a rank is decided once.* A group pod's process registers before it loads
  anything, as undecided for its rank, and the registration's reply tells it to serve as the
  rank's worker or stand as a spare. Exactly one process is told to serve a rank at an epoch; every
  other is told to stand. The answer is written to the engine's status in the store, fenced by the
  term, before the reply goes out, so a coordinator's restart never gives it twice. A rank held,
  by a registered worker or by a logged answer, is answered "stand". An answer is released when
  its process departs before registering as the worker, or its node is lost; otherwise the rank's
  holder changes only through a committed change, never through a registration. Grove's upstream work on stable capacity
  slots and an authorized release of a named pod (GREP-0793 [8]), a runtime floor apart from the
  admission floor (#836 [9]), and terminal states and restart policies (#877 [10]) is Everserve's
  path to a drain of a named pod and to spares Grove places.
- *Recovery goes before Dynamo's migration.* While Everserve replaces a rank or fails the head
  over, the integration reports the instance recovering. Dynamo has no state that holds its
  migration off: migration fires on a connection failure or a migratable stream error [5], and a
  failed worker is set aside for a few seconds [7]. The upstream ask is a recovering state (or a
  per-request hold) that Dynamo's frontend reads, so that Everserve recovers first and Dynamo's
  migration (`--migration-limit` of one or more) is the fallback once Everserve declares the
  instance failed (an engine Down, 0032).
- *The instance keeps its identity across a head failover.* A new head registers under a new
  lease, which reads as one instance leaving and another joining: the router's cache-aware state
  for it is lost until republished, and streams on the old registration break and fall to
  migration. The upstream ask, beside the recovering state, is a way for a replacement leader to
  keep the instance's identity, or an alias the router resolves. Until Dynamo has it, head failover
  under Dynamo is degraded and correct: the workers and their KV are kept, and requests resume by
  id when Dynamo's migration lands them on the instance.
- *A resubmitted request resumes; it is never run twice.* A request resubmitted with an id the
  request-state mirror holds resumes from its committed tokens rather than starting again (0038),
  so Dynamo's migration onto the same instance lands on Everserve's resume and never duplicates it.
- *Drains and preemptions come to Everserve.* A node cordoned or tainted for disruption is drained
  by Everserve, which watches the Node and starts the drain itself; a disruption budget, counting
  whole groups rather than pods (a pod-counting budget under-protects a gang and can deadlock a
  drain [11]), holds an eviction until the node's Released condition. A provider's preemption is a
  notice to act on at its first sign: CoreWeave's spot pools cordon a node and label it
  `qos.coreweave.cloud/pending-preemption: true` (event `Preemption pending`) at once, drain it
  themselves two minutes later, sending each pod SIGTERM after its grace period, and remove it
  within seven minutes [14]. The lead Everserve can use is two minutes and the grace period, not
  seven, so the Node watch acts on that label, and a drain started then commits well inside it.

## Designed for, not built: llm-d and KServe

The contract must fit them, and these are where it would meet them; none is built.

- llm-d's endpoint picker watches pods by its pool's label selector and routes to a Ready pod's
  address, honouring custom readiness gates [6]: the head's pod would carry the pool's labels and
  a readiness gate Everserve sets false during a head failover and during a change that outlasts
  the engine policy's bound. Not for every replacement: a rank replacement pauses the instance for
  about a second (0.56 s and 1.67 s on run 3, `bench/runs/2026-10-13-run3-drains-smoke`) while the head keeps accepting, and taking the pod
  out of routing for it could outlast the pause.
- LeaderWorkerSet's `None` policy restarts only the failed pod [2]; an llm-d deployment can set it.
  KServe sets `RecreateGroupOnPodRestart` in code [3] and would need a field to change it.
- llm-d's Inference Resilience Operator, in early development [12], recovers an instance through a
  `RecoveryRequest` (RESET_DEVICE, REBOOT_NODE, REPLACE_NODE) carried out by an engine's
  `EngineAdapter` [1], and targets data- and expert-parallel engines at tensor parallelism one, as
  vLLM's elastic expert-parallel work does [13]. Everserve is complementary, replacing a tensor- or
  pipeline-parallel rank in place at full capacity; an Everserve `EngineAdapter` would keep two
  orchestrators from acting on one fault.

## Pod ownership

Inside the framework's group: the framework creates and owns the instance's pods, and Everserve
decides what each GPU's process does. Everserve creates no pod beside a framework's controller, so
two controllers never act on one group.

Open, decided with the Grove launcher backend: spares beyond the pods Grove recreates. A Grove
group, or a LeaderWorkerSet one, cannot take in a pod it did not create. Either an upstream hook
lets the group adopt a member or replace one with a named pod (GREP-0793's authorized release is
the start of it [8]), or spares stand outside the group and register through Everserve's control
plane, taking a rank without becoming a member of the framework's group.

## Build order

1. Dynamo's worker integration: Everserve's layer loading inside Dynamo's vLLM worker, its
   registration surviving a rank replacement. The first proof: a non-head node lost or drained
   under Dynamo on spot nodes (the product's target is CoreWeave; the proofs ran on Nebius), sessions
   continuing, Dynamo's router unaware.
2. Resume by request id.
3. The core on the CPU tier, and its standby (0038).

The upstream asks (a recovering state, and an instance identity across a head failover) are
pursued with Dynamo in parallel.

**Consequences.**

- `integrations/dynamo/` holds everything Dynamo-specific; the core gains the framework-integration
  contract.
- 0038's front end is outside the MVP; under Dynamo the frontend is Dynamo's, and Everserve's
  contract with it is the registration, the recovering state and resume by id.
- The operator API moves to the one model: resources with spec and status, a generic watch, and
  node-scoped writes. 0038's release endpoint and head record become a node's Released condition
  and an engine's head in its status, read and watched like any other. An instance being torn
  down by its framework refuses every further action.
- The Kubernetes work (the chart, 0037's node identity, the custom-resource spec store, the Node
  watch and the group disruption budget) is what the integration rests on.
- An engine's policy states its shape (its parallel layout and what it requires), never where it
  runs: under a framework, the scheduler places the pods and the processes that register place
  the engine, which is status, as 0036 has it. Node placement in the policy is an optional pin, for
  bare metal and a single node run by hand.
- The launcher seam gains a Grove backend, in which the framework's group owns the pods.
- Engine neutrality matters more: these frameworks run several engines, and the engine contract
  (0034) is what a second engine's adapter implements.

**Sources** (checked 2026-10-13)

1. llm-d, `proposals/inference-resilience-operator.md` (llm-d/llm-d, main).
2. LeaderWorkerSet, `api/leaderworkerset/v1/leaderworkerset_types.go`: `RestartPolicyType`
   (kubernetes-sigs/lws, main).
3. KServe, `pkg/controller/v1alpha2/llmisvc/workload_multi_node.go`: `RestartPolicy:
   lwsapi.RecreateGroupOnPodRestart` (kserve/kserve, main).
4. Grove, `operator/api/core/v1alpha1/podcliqueset.go` and `podclique.go`: `MinAvailable`
   (default 1) and `TerminationDelay` (default four hours) (ai-dynamo/grove, main).
5. Dynamo, request migration: off by default (`--migration-limit 0`); triggered by a connection
   failure at setup or a migratable stream error; a fresh stream carries the prompt and every
   generated token (docs.nvidia.com/dynamo, fault tolerance, v1.5.1).
6. llm-d-router, `docs/discovery.md`: pods by selector, aggregate Ready, readiness gates honoured.
7. Dynamo, discovery plane and distributed runtime: registration under the runtime's lease, and a
   failed instance inhibited for `DYN_RUNTIME_INHIBITED_DURATION_SECS` (v1.4.2).
8. ai-dynamo/grove#823, GREP-0793: capacity identity and authorized release (draft); from
   ai-dynamo/grove#793, closed as completed 2026-10-01.
9. ai-dynamo/grove#836, open.
10. ai-dynamo/grove#877, open.
11. pdb-operator (github.com/pdb-operator/pdb-operator): pod-counting budgets and gangs.
12. llm-d-incubation/llm-d-resiliency-operator, README: early development.
13. vLLM, elastic expert parallelism (vllm.ai blog, 2026-05-14).
14. CoreWeave, Spot Node Pools, the preemption sequence (docs.coreweave.com/platform/
    capacity-plans/spot-node-pools); released 2026-03-10; not for Superchip instances.
