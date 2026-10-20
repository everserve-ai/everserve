# 0045: the head's loss is a known gap

2026-10-19. Status: current. Reverses the head's failover in
[0038](0038-a-gpu-node-is-disposable-and-the-head-fails-over.md); the code that built it is
removed.

**Context.** Everserve replaces a lost or leaving GPU node's ranks with a spare while the
engine serves, and its sessions continue (the Dynamo proofs, `bench/runs/README.md`). The
engine's core, vLLM's scheduler and executor and Dynamo's client-facing worker in one process,
is not a rank: it runs on rank 0's node, and that node's loss ends the engine. Decision 0038
set out to make it fail over, and from 2026-10-13 to 10-19 it was built in two shapes: a core on
a CPU head node with a warm standby, then a core on one of the engine's own GPU nodes, handed
to another of them on a notice for its node.

The handoff measurement ran the second shape eight times on H100
(`bench/runs/2026-10-15-dynamo-handoff-measure`). The eighth attempt was the first to commit a
head change: rank 0 moved in 0.44 s, the successor took the head, and the change committed
5.4 s after the move. Every attempt found a new defect, and the eighth found two in the design
itself: Dynamo's graceful shutdown killed the retiring core before its goodbye, so a planned
retire read as a loss, and the retiring core's streams found no worker to migrate to, since the
successor registered with Dynamo only after its take. An order that fixes both exists (the old
core parks between steps and acknowledges the head change; the successor says when Dynamo
routes to it; only then is the old core told it was replaced) and was not landed.

**Decision.** The head's loss is out of the product's scope for now: a known gap, stated
wherever the product is described, and the code that built its failover removed rather than
kept dormant. Three reasons.

1. *It is not an adapter.* vLLM's engine core is one process that owns the scheduler, its
   request queue, the KV block tables and the prefix cache, and vLLM has no supported way for a
   second core to adopt workers that are running. Doing it from outside took a replacement
   worker loop, a standby attaching to live workers' KV caches, a start-up path beside vLLM's,
   cross-node rings vLLM's executor never makes, and a handshake between the old core and the
   coordinator to stop at a step boundary: a second engine lifecycle beside vLLM's, every piece
   of it a private line in `PRIVATE-INTERNALS.md`. 0038 had planned it as adapter code over the
   executor's own operations; building it showed those operations do not reach the workers'
   running state.
2. *What was built kept no sessions.* The successor started with an empty scheduler; the
   router re-sent the streams and the successor prefilled them again. The part of the design
   that would have kept them, the request-state mirror the successor adopts block tables from,
   was built but never written by any core and never exercised. What the built handoff saved
   over a restart was the surviving nodes' weights staying loaded.
3. *Its worth is bounded, and known.* On spot, the head's node is as reclaimable as any other:
   of an engine's N nodes, one holds the head, so about one reclaim in N ends the engine and
   drops its streams rather than moving its ranks. With two nodes, every proof's shape, that is
   half. The mitigations below cost one on-demand node per engine, or a second instance; the
   failover would have cost the second engine lifecycle above, in every release, for the
   remaining fraction. The gap is stated as a fraction so an operator can price it; it is not
   claimed small.

**What happens on the head's loss.**
- *Under Dynamo*, the frontend re-sends each broken stream, with the tokens it already sent, to
  another instance of the model (`migrationLimit`, 3 in the rendered frontend;
  `globalDynamoNamespace` lets instances in separate graphs take each other's streams), which
  prefills it again and continues. The lost engine restarts. With one instance there is nothing
  to migrate to, and those streams fail (`integrations/dynamo/README.md`). Read at the pinned
  Dynamo in `bench/runs/2026-10-15-dynamo-migration-read`.
- *An operator who wants the core's node off spot* sets `headOnDemand` (decision 0041); the
  engine's `HeadOnSpot` condition says when it is on spot anyway. That trades the gap for one
  on-demand node per engine.
- *Standalone* (`vllm serve` and the plugin, no router), a lost core drops its streams and
  clients resubmit.

**What would close it.** Two capabilities upstream: vLLM letting an engine core attach to
running workers through a supported executor API, and Dynamo keeping an instance's identity
when its worker process changes, so its router waits for the new core rather than migrating
away. With both, a handoff is adapter code over supported interfaces, the spare's start-up
replay (decision 0044) is the piece already built for it, and this record is reversed in turn.

**Consequences.**

- The engine's core runs on rank 0's node, its loss restarts the engine, and the router migrates
  its streams. Any other node's loss is a replacement by a spare with the sessions kept.
- Removed: the core on a head node or on the engine's own nodes, the standby core and its take,
  the planned handoff and its retire, the nodes starting cores, the request-state mirror and its
  copy, the head's cover, and their tests and harness.
- Kept: the coordinator's change log and its resume after a restart (0038), the engine's head
  named in its placement, and the spare's start-up replay built for the handoff, which every
  spare uses (decision 0044).
- The protocol's fields and messages for the head's failover are reserved by number and name,
  never reused. The join format's offer kind, which only a take needed, is gone (join version 8):
  every offer names the start-up calls of the core that makes it.
- `README.md` states the gap beside the product's claim; `integrations/dynamo/README.md` states
  the single-instance case. The last tree with the removed code is `019c07fe`; 0038's text keeps
  its design, each reversed section marked.
