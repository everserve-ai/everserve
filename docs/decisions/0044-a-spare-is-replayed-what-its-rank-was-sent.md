# 0044: a spare is replayed what its rank was sent

2026-10-16. Status: current; built, both halves. First run on GPUs in the handoff measurement
(`bench/runs/2026-10-15-dynamo-handoff-measure`, attempts 7 and 8, 2026-10-16), under a core on the
engine's own nodes; run in the worker-loss shape the Dynamo proofs ran on 2026-10-19
(`bench/runs/2026-10-19-dynamo-proof-known-gap`, proof-26, pass: every spare that stood was built
from the replay, each join taken on its digest). Recorded on its own 2026-10-19, when the head's failover it was built beside
was removed (decision 0045); before that it was two amendments to 0038. Cut to its core the same
day: what existed for a successor core (a standing spare re-asking at each new head, the core's
incarnation, calls read ahead of a take, a queue kept per asker) is gone.

**Context.** A spare is a worker built outside the engine's world, ahead of the loss it covers.
vLLM's engine core configures its workers' KV caches with broadcasts sent once, before it warms
them (`EngineCore._initialize_kv_caches` at the pinned vLLM, 0.31.0, `adapters/vllm/PIN`): the KV
cache layout it resolved, the memory profiling,
the model length auto-fit settled on (only when it changed one), and every rank's KV cache
configuration. A spare receives none of them. It used to derive each from its own worker, which
agrees with the core only where every pipeline stage holds one kind of KV cache, and it missed the
layout the core sets after building (`bench/runs/2026-10-15-dynamo-handoff-measure`, attempt 6,
2026-10-16). A spare
whose cache differs from the engine's workers' corrupts a rank it takes.

**Decision.** A spare builds as its rank's worker was built: it receives what the engine's core
sent that rank at its start, replayed, never derived (the rule the construction audit set: what a
spare derives for itself can differ from what the core sent, as its derived KV configuration did).

- *The core records* each start-up broadcast as it sends it (`everserve_vllm.replay.Recorder`),
  and the record closes when it warms its workers. The calls are then fixed for the core's life.
- *The spare asks* through its node's daemon, on a connection of its own whose first frame is a
  `ReplayAsk`; that connection registers nothing, since a spare registers only once warm. Its
  daemon names the asker (node, daemon instance, connection serial), the coordinator hands the
  ask to the engine's head, and the head answers with a `ReplayOffer`: a vLLM `MessageQueue` with
  one remote reader, made for that spare alone and closed when it has read. The control plane
  carries the queue's handle unread and keeps nothing per ask. With no head standing, or the
  asker gone, an ask or offer is dropped, and the spare's bounded wait asks again.
- *The spare checks and runs them.* It refuses by name a configuration that does not fit its GPU,
  and runs the calls on its worker by name, in the core's order, before it warms. Its worker runs
  vLLM's own memory profiling, as the engine's workers did: 4.49 s on H100
  (`bench/runs/2026-10-15-dynamo-handoff-measure/RESULT.md`, attempt 7). A spare built as
  its rank asks and runs them where it is built; a spare standing for any position asks at its
  take, for the rank it takes, and pays the run there.
- *A join is checked against them.* A spare whose step configuration differs from the engine's
  is refused by the coordinator before anything moves (decision 0034), so an engine restarted with
  another configuration while a spare stood never offers it a position. The digest is the last
  check: every join offer names the digest of its core's calls, and a spare refuses by name an
  offer naming other calls or none, failing that change rather than serve with a cache the
  workers were not built with.

**Consequences.**

- A spare's KV cache is the engine's by construction, for any mix of cache kinds across stages
  and any field the core sets, in vLLM's own objects.
- The calls cross the network pickled by vLLM's queue, as every vLLM broadcast is: the trust model
  is vLLM's multi-node one, a trusted network between an engine's nodes (the Dynamo integration's
  README states it). Only the queue's handle crosses the control plane, as declared JSON.
- A spare cannot build until the engine's core has warmed its workers; a spare started before
  then waits (up to 30 minutes, for a large model compiled cold).
- Measured on H100 (`bench/runs/2026-10-15-dynamo-handoff-measure/RESULT.md`, attempt 7): the ask
  answered in 4 ms with three calls. Run in the worker-loss shape the Dynamo proofs ran on
  2026-10-19 (`bench/runs/2026-10-19-dynamo-proof-known-gap`, proof-26, pass): the head wrote three
  calls for each spare that stood, and each join was taken on its digest.
