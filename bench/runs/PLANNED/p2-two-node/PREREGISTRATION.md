# Pre-registration: the two-node run (roadmap step 5, decision 0018)

**Amended 2026-10-08, before any node is rented**, after the three-node smokes of 2026-10-07/08
(`bench/runs/PLANNED/two-node-0024-smoke/NOTE.md`): the measured arms are repeated five times and
reported as a median and a worst case; the long-context arm is scored, not post-hoc; the
predictions for the pause assume decision 0025 as amended the same day (the boundary reaps
releases instead of waiting for them), with the smoke's 1.08 s replacement as the figure the
change is measured against; every cross-node span carries the nodes' measured clock offset; the
claim-to-commit time is recorded as three spans. The original text is in the history of this
file; nothing below was changed after a measured arm.

Status: PLANNED 2026-10-06, written before item 5 of `docs/MULTINODE.md` section 11 is built,
predictions from the single-node figures (`bench/runs/2026-10-06-frontier-1node/` and the
commit-path battery when it has run). Three 4x H100 SXM pods in one region (secure cloud,
NVLink inside each, the provider's fabric between them), a 70B-class dense model at TP4 inside each
serving node and PP2 across the two, the third node the spare node. Cap $100. The run log names
every pod, its price and every arm's spend; the watchdog is the estimate plus fifteen minutes,
never the budget.

## What must be on main first

Named here so the run is not started against a tree that cannot run it:

0. **The node role** (section 2): the adapter's executor on the headless nodes with the per-node
   in-flight gate, so arm 6 is detected on node-b itself.
1. **Fabric collectives outside the graphs** (section 4, item 5): the engine's pipeline send and
   receive across nodes through a handle that reads the member set at call time and is never
   captured; the shadow sets for cross-node member sets pre-formed at startup. The adapter today
   handles the tensor-parallel all-reduce only.
2. **The executor joining several spare processes at one boundary** (item 2's remainder): a
   node's four positions taken by four spares in one change.
3. **The harness on three hosts:** the coordinator on the spare node over mTLS TCP (`--listen
   host:port`, `--tls-*`), one daemon and one store per node (`--coordinator`, `--mirror-replica`
   on the spare node's store, `--replica-listen` with TLS), `docs/examples/policy-two-node.json`,
   the engine's eight workers across two hosts, and the spare node's four spares. The notice is
   the operator's: `POST /v1/drain` on the coordinator.
4. The commit-path battery's verdicts, so the single-node bands below are current.

Everything else this run exercises is on main: coordinator over daemons with epoch fencing
(`0e1f3d2`, `9657af4`), changes as sets (`56c5d41`, `5ed4a19`), store replication (`d704d0f`),
topology and the spare pool (`6a3fca0`), the readiness API and the drain (`824a494`), notice
sources (`2550119`).

## Why this run exists

Phase 1 proved the mechanism inside one node: a rank replaced at stock speed with every session
kept. The product is the multi-node shape (section 0): an engine wider than a node, a node that
leaves, and a spare node that takes its positions. Nothing cross-node has been measured. This run
measures every cross-node piece once, on the smallest shape that has all of them.

## What runs

Qwen2.5 72B Instruct (or Llama 3.3 70B Instruct if a token is available; the model is written in
the run log before the first measured arm), bf16, TP4 x PP2, 8 sessions x 2,048-token prompts,
3,000 decode steps so a kill at step 100 lands while the engine is serving, the determinism
settings. In this order; the two arms whose pause goes on the site (5 and 6) run **five trials**
and the other measured arms (4 and 8) **three**, each trial a fresh kill on a fresh control, and
the figure reported is the median with the worst case beside it:

1. **stock x1**: vLLM alone at TP4 x PP2 across the two serving nodes, no spare node.
2. **adapter x2**: the engine under the adapter, the spare node's four spares warm, replication
   on.
3. **adapter, mirror off x1.**
4. **node drain x3**: `POST /v1/drain {"node": "node-b", "lead_ms": 30000}` after step 100. The
   spare node takes node-b's four positions in one change at the deadline; node-b keeps serving
   until then.
5. **node fault x5**: node-b's four worker processes and its daemon killed together during the
   step after step 100 (the host stays up; the loss is the engine's half). Detected by the
   coordinator as the daemon's link closes (decision 0023).
6. **single death on node-b x5**: one of node-b's worker processes killed mid-step. Its group
   cannot be split across the fabric, so the whole group moves to the other node's spares
   (decision 0022); detected on node-b by its own executor's watchdogs (the node role, section 2
   of `docs/MULTINODE.md`), the most common fault a neocloud hands over.
7. **notice reference x1**: no fault, the token count of arm 4, not scored.
8. **long context x3**: one session at 65,536 tokens of context, REPLACE mid-step on node-b, for
   the restore time and the pause of a long context on this shape; the single-node 16K restore
   was 3.1 s and linear in rows. Scored: prediction 12.
9. **shard load x5, reported, not scored**: on the spare node, the time to copy one position's
   weight shard (the 72B model's share of one rank at TP4 x PP2, about 18 GB in bf16) from pinned
   host memory into a warm process on one GPU. The one term a host-resident spare mode adds to the
   pause (decision 0028, design only): the site's large-engine pause is this run's pause plus this
   term, stated as such, until that mode is measured end to end in the frontier battery.

A cross-node single-rank REPLACE is not an arm: a tensor-parallel group cannot span the fabric
until the engine has a fabric path, and a single death on a serving node moves the whole group
(decision 0022). Arm 6 is that case, measured; arm 5 is the node.

**Recorded:** `results.json`; `/metrics` from the coordinator, every daemon and every store after
every arm (`everserve_store_replica_lag_steps`, `everserve_store_replicated_total`,
`everserve_store_replication_dropped_total`, `everserve_spares`, `everserve_change_seconds`);
`GET /v1/readiness` before and after every change; the timeline spans per rank, with the
coordinator's claim-to-commit as three spans (deliver, acknowledgements, commit) and the kill
stamped on the clock of the node where it lands; the clock offset between every pair of nodes,
measured at the start and the end of the run, beside every cross-node span; the fabric
collective launches per step; the shadow sets' device memory per rank; every rank's event log in
`flake/`; topology (`nvidia-smi topo -m` per node, the fabric's bandwidth as `nccl-tests`
all-reduce between the two serving nodes, once); versions; spend.

## Predictions (kill conditions in bold)

1. **Adapter overhead** against stock, wall-clock tok/s over steps 10–200, each adapter run: under
   3 %. **Kill: over 5 %.** The fabric collectives run outside the graphs, one launch per pipeline
   boundary per step; the single-node floor is within 1 % of stock.
2. **Adapter against mirror-off:** within 2 points. **Kill: more than 4 points apart.**
3. **Node drain pause**, the largest inter-token gap over the drain, both runs: under 1.0 s.
   **Kill: no swap, or over 3 s.** The single-node notice swap is 0.30 s with the incremental
   restore; four positions restore in parallel from the spare node's own replicas; the pipeline
   group's swap is a pointer swap.
4. **Node fault pause**, the largest inter-token gap, median of five: 0.5–1.5 s, of which
   detection under 0.1 s: the coordinator sees node-b's daemon link close when its processes are
   killed (decision 0023), and the rest is four restores in parallel from the spare node's own
   replicas, the join through the daemons, the boundary (a sync and a swap, no wait for a
   release, decision 0025 as amended) and the resume. The three-node smoke replaced one position
   in 1.08 s with the boundary waiting 0.5–0.8 s for a release it did not need. **Kill: median
   over 4 s, or detection over 0.5 s.** Reported beside it: the head executor's time from the
   kill to its step abort, which is the forwarded change's arrival.
4b. **Single-death pause** (arm 6), the largest inter-token gap, median of five: 0.5–1.5 s, of
   which detection under 0.05 s by node-b's node role reading `/proc` (the smoke: 5–50 ms).
   **Kill: median over 4 s, or detection over 1 s** (the two-second failed-send fallback is the
   kill: it means the node role did not act).
4c. **Worst case of five** for arms 4 and 6: within 2x the median. **Kill: over 3x**, which
   would mean a mechanism with a mode, not a distribution.
5. **Sessions continue** in arms 4, 5 and 6: 8/8 kept, tokens identical to the no-fault run's after
   the change, resumed from the committed count, no re-prefill. **Kill: any session lost,
   re-prefilled or mismatched.**
6. **Restored rows equal** the replaced ranks' committed rows, every table of every replaced
   position. **Kill: any mismatch.**
7. **Replication keeps up at steady state:** over the decode window, `everserve_store_replica_lag_steps`
   at most 2 on every rank, and `everserve_store_replication_dropped_total` zero throughout.
   **Kill: a lag over 8 at any decode step, or any drop.** The fence is three deep; a replica
   more than that behind means a restore would miss committed rows. The prefill burst is reported,
   not scored: about 0.65 GB per rank from four ranks, whose lag is set by the fabric's bandwidth,
   and on a fabric below the floor in **Platform** it is seconds.
8. **The readiness view is true:** before arm 4, `GET /v1/readiness` reports node-b reclaimable
   with outcome `replace`; after it, node-b holds no rank, node-c holds the four, and the engine's
   `expected_pause_ms` for `replace` equals the measured change. **Kill: the view disagrees with
   what happened.**
10. **One epoch across three daemons:** every change is delivered to every daemon holding a
    survivor or a spare and committed at one epoch; no daemon refuses a delivery. **Kill: a
    refused delivery or a split epoch.**
11. **Shadow sets:** formation under 60 s across the fabric, device memory per rank under 2 GB.
    **Kill: formation fails.**
12. **Long context** (arm 8), median of five: the restore of one position's rows for a 65,536-token
    session completes within 2x the time the restore's bytes take at the fabric's measured
    bandwidth, and the session continues identical. **Kill: over 4x, or a mismatch.** Reported
    beside it: restore seconds against rows, so the curve a longer context would follow is the
    figure, not one point.
13. **Claim to commit** across three daemons, median: under 0.15 s, as the first exact smoke arms
    (0.09–0.14 s); the last two measured 0.29–0.39 s unexplained. **Kill: over 0.5 s.** The three
    spans say where it goes.
14. **Restore coverage** is one: the rows the spare restores equal the rows the replaced position
    had committed, every table, every request, prompt and decode alike (`restore_coverage` in
    the record, rows restored over tables times the sum of committed counts). **Kill: under 1.0.**
    Dry run 1 restored the decode region only (0.62, 0.53), because the replica never received
    the prefill; the exactness it reported was the small model's indifference to random-token
    prompts, not the mechanism's, and this prediction is what makes prediction 5 mean what it
    says.
15. **No request is re-prefilled** after a change: every request resumes at its committed count, or one below it when the last sampled token died
    with the rank before it was delivered, which the record names per request (`resume_from`,
    `committed`, the cause), and the first batch after the resume computes one token per
    request, as a decode (`requests_re_prefilled` in the record). **Kill: any resume below
    that, or any first batch computing more than one token for a request.** A re-prefill would reproduce the
    prefill-against-decode numerics seam of runs 10 and 11 and read as a divergence that is not
    the restore's.

**Reported, not scored:** the fabric's measured all-reduce bandwidth; the fabric collective
launches per step; restore seconds per position against the single-node 0.36–0.41 s; the pipeline
bubble under the adapter against stock; the post-hoc long-context restore.

16. **Every process of a launch runs on the engine's compile cache** (decision 0029): the
    record carries each launch's `torch_compile_cache` key and digest per rank, every trial starts
    with the readiness view's `compile_cache_consistent` true, and no autotune file is written
    under the engine's key during a trial, with the engine launched in Inductor's deterministic
    mode (`inductor_compile_config: {"deterministic": true}`), under which no reduction kernel's
    configuration is chosen by timing. **Kill: any trial started inconsistent, or any autotune
    file written during one.** Pass 6 of the dry run (2026-10-08) found the spare's rows
    differing from the control's at rounding level from the resume on because node-c's Inductor
    autotune chose other kernel tiles than node-b's under the same compiled artifact, and one
    launch on node-b autotuning one kernel fresh and landing in another rounding family before
    any kill. Row-level predictions (5, 6 and `first_differing_position`) are scored against the
    engine's own first member, never against stock, whose key and autotune are its own.

## Platform

The two-node smoke of 2026-10-07 (`bench/runs/2026-10-07-two-node-smoke/`) measured ordinary
RunPod pods' private network at 82–91 Mbit/s, a shaped user-space tunnel, and no traffic at all
between two pods in one region. This run cannot run on ordinary pods. It needs nodes on a flat
fast network: RunPod's Instant Clusters, or a provider's cluster offering. The floor is a measured
inter-node bandwidth of 25 Gbit/s by `nccl-tests` or `iperf`, below which the run proves the
mechanism but not the design's target and prediction 7's prefill lag is not a product figure.

If the platform's unit is the 8-GPU node, the shape is two 8x H100 nodes, each serving four
ranks and holding the four spares for the other node's positions; a node loss moves the whole
engine onto the survivor, pipeline over NVLink. That exercises everything this run exists for.
The three-node shape of decision 0018 waits for a provider with 4-GPU nodes on a fabric.

## Cost

Twelve H100s at the rates paid so far, about $3.50 per GPU-hour, is about $42/h; sixteen on two
8-GPU nodes about $56/h. With five trials on the two site arms and three on the others the
battery is about three hours including setup and model load: about $140 to $180, **cap $190**,
raised from $120 by the amendment of 2026-10-08 for the repetition, to be confirmed by Cort
before booking.
On A100 80 GB nodes, on which the 72B model fits at TP4, about half.

## What this run does not claim

Eight-node engines, MoE expert parallelism across nodes, mixed hardware, the fabric slot kernel
(decision 0015, before a design partner), Kubernetes installation, contexts beyond arm 8, the
host-resident spare mode end to end (arm 9 measures its one extra term), and frontier-size models: those are the second battery on the same rental,
`bench/runs/PLANNED/p3-frontier-moe/PREREGISTRATION.md`, written once this one is green.
