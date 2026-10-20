# The adapter contract

What an engine has to expose for Everserve to keep it serving through a membership change.
This is the interface the adapters implement today and the one we would propose upstream in
phase 3. Every item names the vLLM surface it maps onto (adopted) and the SGLang one. No SGLang
adapter exists; every SGLang column is open.

1. **Collectives through a swappable handle.** The model's tensor-parallel and expert traffic
   goes through a communicator the runtime owns (`Collective`), so the group can be aborted and
   replaced without the model knowing. vLLM: the graph-safe custom all-reduce op routed under
   `GroupCoordinator.all_reduce` for stock models, and the pipeline's sends over the fabric
   family (`everserve_vllm.collective_op`, `everserve_vllm.pipeline`; ledger A12, A33); no model
   class is ported yet (`models/__init__.py` registers nothing), so no model owns its
   communicator. SGLang: `GroupCoordinator` rebuild (open).
2. **Worker death is an event.** The executor reports a lost worker and keeps the engine core
   alive instead of shutting down. vLLM: `EverserveExecutor`, a subclass of the multiprocess
   executor (`everserve_vllm.executor`); the engine is given `--distributed-executor-backend mp`,
   which the plugin resolves to it in a process whose environment names an engine
   (`everserve_vllm.backend`, ledger A48), since vLLM 0.31 refuses a qualified class name for a
   pipeline-parallel engine. SGLang: watchdog policy hook (open).
3. **A per-step commit point with the slot ids written, and the request lifecycle.** So
   committed KV can be mirrored, a restore knows the committed length per request, and that
   length stays true through preemption. vLLM: the `KVConnectorBase_V1` lifecycle, mapped below.
   SGLang: HiCache storage backend plus decode offload outside PD mode (open).
4. **Fixed KV slot and expert chunk layout, set at launch.** So placement after a change is
   arithmetic, not a reshuffle. vLLM: the paged layout as vLLM allocates it, read by shape
   (`PagedKV.of`); a REPLACE restores a rank's rows to the same head slots. A model-owned layout
   waits on a ported model (SHRINK; none yet). SGLang: `--init-expert-location`, dispatcher
   registry (open).
5. **Capacity signal to the scheduler.** The scheduler learns the new capacity without being
   told why. vLLM: the unchanged scheduler sees the engine resume. SGLang: DP controller mask
   (open).
6. **Membership from outside.** The controller's signals (`proto/everserve/proto/signals.proto`)
   are the only way membership changes; the engine never decides on its own. vLLM: the
   `DaemonConnection` in the worker. SGLang: same link (open).

## Item 3 on vLLM: the connector lifecycle and `CommittedKV`

`everserve_vllm.connector.EverserveConnector` is the vLLM class; it delegates to
`everserve_vllm.lifecycle.Lifecycle`, which holds the `everserve.state.committed.CommittedKV`
transport the worker's bootstrap attaches.

| vLLM `KVConnectorBase_V1` | side | `Lifecycle` and `CommittedKV` | what it means here |
|---|---|---|---|
| `register_kv_caches(kv_caches)` | worker | the tensors kept on the connector; the worker's bootstrap registers them with `MirrorCommittedKV.register` (`core/everserve/state/mirror.py`), the production transport. `KVCopy.register` (`core/everserve/state/kvcopy.py`) is the loopback transport the tests use | learn the paged layout; the store opens one table per (layer, held head slot) now, so the first commit costs what every later one does |
| `build_connector_meta(scheduler_output)` | scheduler, per step | `Lifecycle.record` → `StepRecord` | the committed count per scheduled request (computed before the step plus scheduled in it) and the requests preempted, carried to the workers as the connector metadata |
| `request_finished(request, block_ids)` | scheduler, once | `finished(request_id)` | forget the request; we never hold blocks asynchronously, so return `(False, None)` |
| `get_num_new_matched_tokens(request, n)` | scheduler | none: `(0, False)` | no external hits. After a change the scheduler resumes each running request from its committed count directly (`resume.resume_running`, decision 0013), not through the connector: nothing is preempted, freed or looked up in the prefix cache |
| `handle_preemptions(metadata)` | worker, start of step | `evicted(request_id, 0)` per preempted request, then the runtime's attempt callbacks (`clear`) | a preempted request keeps no blocks, so nothing of it is committed; anything uncommitted is dropped before the step |
| `save_kv_layer(layer, kv, attn_metadata)` | worker, per layer | `save(layer, kv, slots)` | keep this step's rows for every unit held; `slot_mapping` gives the slots; nothing inside a graph capture |
| `start_load_kv(forward_context, attn_metadata=)` | worker, before the forward | none loaded; the step's slot ids kept | we load nothing per step. The attention metadata the model runner passes is kept so `wait_for_save` can save the layers FULL graphs hide from the per-layer hook (ledger A22) |
| `wait_for_save()` | worker, per step | the layers not yet saved, then through the runtime's commit callbacks `commit(epoch, step, committed_tokens)` and `fence(step - 2)` | start making this step's rows durable without waiting (device copy on a side stream into pinned memory, send from a thread, counts after their rows); then wait for the commit two steps back to be acknowledged. A step a change interrupted is not committed: its rows are cleared and the scheduler replays its tokens |
| `update_state_after_alloc`, `wait_for_layer_load`, `get_finished` | both | no-ops (`get_finished` the base's default) | we load nothing per step; a restore is a bulk copy at a membership change, not a step action |

Two facts the mapping depends on: the connector is named by its module path in
`--kv-transfer-config` (`engine_args.CONNECTOR`: `EverserveConnector` from
`everserve_vllm.connector`, `kv_both`), so vLLM builds it on both sides and the plugin registers
nothing for it; and the connector's worker-side methods run on the model runner's thread, so
`commit` is on the step path by construction.

### The one-step fence (decision 0009)

The first port committed synchronously on the step path: the rows left the device after the
forward and the agreement waited for the store's acknowledgement. On the four-GPU run that cost one
decode step per step (27.8 % overhead, `bench/runs/2026-10-05-adapter-port-4gpu/RESULT.md`).

The commit is now asynchronous with a fence one step behind:

- `save(layer, kv, slots)` issues the layer's rows towards the host as the forward runs: one
  gather for every held head slot on the compute stream, one copy on a side stream into a pooled
  staging file on shared memory, pinned for the GPU. Nothing on the step thread reads device
  memory (decision 0010).
- `commit(epoch, step, counts)` records one event behind the last copy and queues the step for a
  sender thread, which waits for the event and names the staged rows to the store in one
  descriptor of the staging file's layout, whatever the layer count (decision 0019). No row bytes
  cross the socket (decision 0011). The per-request counts travel with it, so a count is never
  ahead of its rows.
- `fence(step - 2)` waits until the store has acknowledged the commit two steps back (decision
  0019; one step back until the frontier run showed the tail of a commit exceeds a step's lead at
  frontier shape). It runs inside the step guard before the agreement collective, so a step the
  survivors accept has every rank's rows durable **through step N-2**. Two steps' rows are in
  flight.
- A restore is exact to the restored rank's own last acknowledged commit (the store returns that
  rank's counts with its tables), and the scheduler replays from those counts. The restoring rank
  maps the tables from shared memory and copies the written rows to its GPU. After a fault during
  step N the replay is at most three positions: N, N-1 and N-2 as their commits stand, each a
  one-token decode step (decision 0013). That is the whole cost of the fence, paid only on a fault.
- `evicted` and `finished` are ordered behind the commits already queued, so a lowered count can
  never be overtaken by rows the scheduler has since freed.
- At most `depth` steps (three by default) are in flight. A commit with the pipeline full waits for
  the oldest step first and counts the wait, so a store that falls behind shows up as a number in
  the run report, not as unbounded host memory.
- A send that fails surfaces on the step thread at the next `fence` or `commit` and poisons the
  transport, so the step is not agreed and the worker fails the way a synchronous commit failure
  failed. The transport does not retry: that is the daemon's decision, with the membership.

The invariant is unchanged in kind: survivors never accept a step whose state cannot be restored.
What changed is the step it is measured at.
