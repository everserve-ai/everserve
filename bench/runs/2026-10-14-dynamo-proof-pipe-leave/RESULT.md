# Result: two losses of rank 1 under load, both by notice, the second from a spare that took the rank

**Verdict: pass.** On 2026-10-14, `bench/harness/k8s/dynamo_proof.sh --losses 2` ran on images
built from `f22153e`. It took away the node pod holding rank 1 twice, each time while eight long
streams ran through Dynamo's frontend. Both losses reached the coordinator as notices, and each
was a REPLACE committed through a covered set. All 16 streams finished with every token and no
error.

The second loss is the one this run adds. The pod taken was the spare that took rank 1 in the
first loss:
- proof-16 took that pod without notice: a 3.7 s gap.
- proof-17 took it with a notice that never left the spare (below).
- Here it gave notice through its pipe, and the gap is 1.17 s.

The run's output is `run/`, with each loss's record in `run/engines-proof-loss-N.json`,
`run/engines-proof-events-loss-N.json` and `run/load-N.json`.

## What ran

As [proof-16](../2026-10-14-dynamo-proof-two-losses/RESULT.md), on images
`everserve:dev-f22153e` and `everserve-engine:dev-f22153e`. They carry:
- the spare's leave over a pipe its node hands it (f22153e);
- the notice lead held to the policy's minimum (2c0ede2);
- the coordinator's link outbox (1f19f2e).

The cluster:
- asked for at 15:26:38Z;
- serving at 15:45:01Z;
- taken down after the verdict, under a 1.5 h watchdog.

## What happened, from the record

| | Loss 1 | Loss 2 |
|---|---|---|
| covering set, at the take | `0-R1@…d3csp…:0`, plan 2 at epoch 0 | `0-R1@…hkc4y…:0` (Grove's new pod on the freed node), plan 4 at epoch 1 |
| pod taken | `ztbmw`, the original serving node | `8862r`, the spare that took rank 1 in loss 1 |
| notice heard by the coordinator | 15:45:28.569 | 15:46:44.337 |
| change reported (the notice's deadline) | 15:45:33.569 | 15:46:49.337 |
| committed | +363.1 ms | +1,130.7 ms |
| acknowledgements | rank 0 +13.7 ms, rank 1 (the spare) +362.9 ms | rank 0 +13.8 ms, rank 1 (the spare) +1,130.6 ms |
| streams | 8 of 8, 6000 tokens each, no error | 8 of 8, 6000 tokens each, no error |
| worst gap between tokens | 0.638–0.645 s, at the change | 1.158–1.171 s, at the change |

In both losses the spare's own take was 3–4 ms ("took rank 1's position … in 0.004 s",
"… in 0.003 s"). Between the notice and the acknowledgement, the time is the restore of rank
1's rows onto the spare's node, staged from the head node's store when the change starts.

## The restore's cost, and why

The two restores, from the stores' and spares' logs:

| | Loss 1 (empty target) | Loss 2 (target holding a stale copy) |
|---|---|---|
| staged (spare) | 15:45:33.583 | 15:46:49.350 |
| the restoring store | `restore since step 0: 447456 rows` (store nmh7v, 15:45:33.868) | `restore since step 0: 1572480 rows` (store cndv8, 15:46:50.327) |
| the target, as the sender read it | `holding nothing at epoch 0 step 0` | `holding rows at epoch 0 step 4682, this store at epoch 1 step 16528` |
| acknowledged (spare) | 15:45:33.951 (**0.37 s** after staging) | 15:46:50.468 (**1.12 s** after staging) |

The target's stale copy costs nothing. Loss 2 sent 3.5× the rows in 3.0× the time.

What sets the size is **confirmed in code**:
- `snapshot_run` in the store's `storage.rs` sends every slot whose `written_step` is not zero,
  and `restore` in `service.rs` sends every window written since the step asked.
- Nothing clears a slot's `written_step` when the engine frees its block.
- So a restore from step 0 grows with the store's steps (16528/4680 = 3.53; the rows'
  ratio is 3.51) toward the rank's whole KV capacity.
- Loss 2's 1,572,480 rows are within 0.03% of rank 1's 1,572,864: 4096 blocks × 16 slots ×
  12 layers × K and V.

Clearing a slot when its block is freed would be wrong while the prefix cache is on (it was:
the head logged a 33.9% hit rate). A freed block stays cached and valid until it is evicted, and
a later request takes it as a hit without rewriting it. On a loaded engine with prefix caching,
the cached rows fill toward capacity anyway.

## Open trades for Cort

Each shortens the restore that sets the change's gap. Each costs something the design chose not
to spend. The numbers are this run's: 0.37 s for 447k rows, 1.12 s for 1.57M rows.

- **(a) Continuous replication to the covering spare's node.**
  - Buys: the change restores only the steps since the last replicated one, close to nothing.
  - Costs: one more copy of each rank's rows in host memory for each covering spare, and the
    bandwidth to keep it current.
  - Decision 0030 chose not to (a shared spare holds no KV until it joins), reversing 0016.
- **(c) At the change, restore only the blocks running requests hold, and drop the free cached
  blocks from the head's prefix cache.**
  - Buys: a restore sized by live requests, not by the cache.
  - Costs: the prefix-cache hits on the dropped blocks right after a change.
  - Needs the adapter to reach into vLLM's block pool: `reset_prefix_cache` refuses while any
    block is in use.

## Earlier on the way here

- proof-17 (12933dc) failed its second loss, and the verdict caught it.
- The spare's leave was SIGUSR1, the signal the plugin registers `faulthandler` on for stack
  dumps. vLLM loads the plugin after the spare set its handler, so the leave dumped stacks.
- The spare gave no notice and served on through its pod's grace period, and no change came.
- Fixed by f22153e, which hands the leave over a pipe the node owns, not a signal.

## What this does not show

- A loss without notice from a node that dies rather than being deleted.
- The head's own loss.
- A production-size model, more than one node per stage, and CoreWeave.
- The coordinator's audience change (S1, 4ce8921), whose first GPU run is the next proof.
