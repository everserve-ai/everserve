# Pre-registration: the resident shared-spare run (decision 0030)

Status: PLANNED 2026-10-09, written before any node is rented and before the shape has run on
anything larger than a 1.7B model. It supersedes `bench/runs/PLANNED/p2-two-node` as the rental
(decision 0030): that battery's shape, a warm spare per position on a spare node holding every
rank's mirror, proved the mechanism and is not a shape a customer runs. Nothing below is changed
after a measured arm; an amendment before the first arm is dated and says why.

Three single-GPU H100 80 GB nodes in one region. A 70B-class dense instruct model in FP8 (the
exact checkpoint written in the run log before the first measured arm), TP1 x PP2 across node-a
and node-b, each node holding one stage. node-c is the shared spare node: the whole model resident
in pinned host memory, one spare process standing for any position with no shard loaded, its
communicators formed and warmed. Each serving node's KV replicates to its sibling (no
`replicaNodes`); node-c holds no rows until a change stages them. Every node runs one named commit,
recorded with its hash.

## Why this run exists

The dry runs of 2026-10-09 (passes 9 to 15, `bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass*`)
proved the resident shape on a 1.7B model: every fault trial row-exact, the lost rank's rows
staged from its sibling, the backup restored after the take, and a node-fault gap of 10.2 s
against a stock restart of 393 s. Two things those passes cannot say: what the terms become at a
size where weights and KV are tens of gigabytes, and what a stock restart costs at that size. This
run measures both on the cheapest hardware where they are real.

## What must be on main first

1. Everything the dry runs used, at one commit: resident weights, the rankless spare, sibling
   replication, staged restore, the guarantee class (exact), the compile-cache check (0029), row
   checksums (protocol 2), recorded KV sizing.
2. The pinned deploy: every node syncs and builds exactly the commit the driver names.
3. An FP8 dry pass on the cheap one-GPU shape before any H100 is booked: a small model's FP8
   checkpoint through the resident load, the take, Inductor and graph capture, every fault arm
   exact. FP8 has run in no dry run so far, and the resident loader refused every FP8 take until
   2026-10-09 (it required parameters the checkpoint does not carry, which vLLM itself creates).
4. The take's terms in the change record: the spare's join answer carries its attachment record
   (module construction, KV configuration, weights to the device, warm-up), so the summary scores
   them rather than a log.
5. A smoke of a few minutes on the three H100 nodes before the first measured arm: the model loads
   under the adapter at TP1 x PP2 in FP8 with deterministic kernels, one step serves, readiness
   reads consistent. If it fails, the run stops and the cost so far is recorded.

## What runs

Eight sessions of 2,048-token prompts generating 1,792 tokens, greedy, seeded, in every arm.

1. **stock throughput x1**: vLLM alone at TP1 x PP2, no spare node.
2. **adapter throughput x2**: the engine under the adapter, packed exchange at depth 2, mirror on.
3. **adapter, mirror off x1.**
4. **stock restart x3**: node-b's vLLM process killed mid-decode, the engine relaunched on the
   same nodes by the stock path; the time from the kill to the first token served again. Sessions
   are lost by construction, recorded as 0/8.
5. **node fault x5**: node-b's worker and daemon killed together mid-step; node-c's spare takes
   rank 1.
6. **single death x5**: node-b's worker killed mid-step, its daemon alive.
7. **long context x3**: one session at 32,768 tokens of context beside the eight, node fault
   mid-decode: the staged copy and the restore at a size the dry runs never reached. The engine's
   block count and every store's budget are sized for it (about 5 GB of one rank's KV at 32k in
   bf16), and the sibling stores hold each other's rows in host memory.

After every fault trial the battery reads both stores with the engine quiescent: backup coverage
for both ranks at the step both stand at.

## Predictions (kill conditions in bold)

1. **Sessions continue exactly**, arms 5 to 7: every session's tokens identical to the control's,
   every trial. **Kill: any mismatch.**
2. **Rows equal**: `first_differing_position` none on both ranks, every fault trial (comparable:
   no preemption, block tables prefixes). **Kill: any differing position.**
3. **Restore coverage** 1.0, every fault trial. **Kill: under 1.0.**
4. **Backup coverage after the take** 1.0 for both ranks, every fault trial. **Kill: under 1.0.**
5. **No request re-prefilled**; a resume one below the committed count only with its cause named.
   **Kill: any resume lower, or any first batch computing more than one token for a request.**
6. **The compile cache holds**: readiness reads consistent before every trial, the spare's pair
   equal to its position's at the join, no autotune file written during a trial. **Kill: any.**
7. **Row checksums**: `everserve_store_refusals_total` with reason `checksum` or `unchecked` is
   zero on every store at the end of every trial. **Kill: any.**
8. **Node fault gap**, kill to first token after the change, median of five: 10 to 25 s, the take
   dominated by the worker's build and warm-up. **Kill: median over 40 s.**
9. **Single death gap**, median of five: 8 to 20 s. **Kill: median over 40 s.**
10. **The take's terms**, each reported per trial: weights to the device 1.5 to 4 s for a
    35 GB stage share from pinned memory (**kill: over 10 s**); KV configuration under 2 s with
    profiling skipped (**kill: a profiling run logged**); module construction and warm-up
    reported, not scored, since they are what this run exists to measure at size.
11. **Long context**, median of three: the staged copy of the lost rank's rows finishes before
    the spare asks for them, and the restore to the device is under 2 s. **Kill: the spare's
    staged wait over 5 s.** The copy's bytes and seconds are reported.
12. **Stock restart**, median of three: 5 to 20 minutes. Reported as the comparison, not scored.
13. **Adapter throughput** against stock, steps 10 to 200, each adapter run: within 5%.
    **Kill: more than 10% under.** The mirror's cost is arm 2 against arm 3, reported.

**Reported, not scored:** the clock offsets between every pair of nodes; the fabric's measured
bandwidth; the shadow sets' formation time at standing; the spare's resident host memory.

## Platform

Three single-GPU H100 80 GB VMs, one region, on one provider; at least 200 GB of host memory on
node-c for the resident model and its pinned buffers (the pin itself, about 45 s at standing, is
off the change's path), and room on node-a and node-b for each other's replica in host memory.
Every node fetches the checkpoint at setup: about fifteen minutes. The run log names each VM, its price, its
driver, CUDA, torch and vLLM versions, and the commit every node built.

## Cost

Three H100 VMs at about $3 per GPU-hour is about $9 an hour. The battery is about four hours
including setup, model staging and the stock restarts: about $36. **Cap $60.** The watchdog is the
estimate plus fifteen minutes, never the cap. The rental is booked on Cort's word only.

## What this run does not claim

- Tensor parallelism inside a node, and therefore decision 0022's group widening and the slot
  kernel at size: every rank here is one GPU. That needs multi-GPU nodes and is the next run.
- Frontier size, mixture-of-experts, expert parallelism across nodes.
- A second loss: after the take the pool is empty; the backup coverage reading stands in for it.
- The on-notice KV mode, degrade-then-restore, coordinator failover: designed, not built.
- Any figure leaves the repository only after its record is committed.

## Amendment, 2026-10-09, before any node is rented

Why: pass 16 (`dryrun-1gpu-fp8-pass16b`, f1a59cc) ran FP8 through the resident shape and found a
cold-node warm-up term the staged caches do not cover, and a reading of the battery against this
text found settings that differ from it. Each item below replaces the text above where they
differ.

1. **The checkpoint.** `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` at revision
   `f50dbad2c84590ca17dc51e207c34321b65ff14b`: compressed-tensors, FP8 weights per channel with
   dynamic per-token activations, 80 layers, 8 KV heads; node-b's stage begins at layer 40. On
   Hopper it runs CUTLASS `scaled_mm`. The block-scaled variant is not used: on Hopper it routes
   to kernels that may compile at run time, a second unnamed warm-up term and a determinism risk.
2. **Prerequisite 3, extended.** Pass 16 proved vLLM's `fp8` method, which on Ada runs
   weight-only through Marlin. This checkpoint loads through compressed-tensors, a different path
   through the resident loader. A dry pass with `RedHatAI/Qwen3-1.7B-FP8-dynamic` (e776dcc), the
   same scheme, on the one-GPU shape, every fault arm exact, is prerequisite 3b.
3. **Prerequisite 6, new: the repeat arm.** The engine under the adapter serves the same sessions
   twice with no fault, and the tokens agree. It runs in the smoke on the H100 nodes, and the
   battery exits before any later arm unless it is green. Every fault trial is scored against an
   adapter control, so this proves the comparison holds on Hopper's FP8 kernels before a mismatch
   can count against the take.
4. **Prerequisite 5, extended.** The smoke runs one node fault, node-c's first take, with the
   files written on node-c read per start (`written_since.json`: directory, files, bytes, and
   the process's exit kind). The warm-up term the reads name, and the files, are written here,
   dated, before arm 1.
5. **Cold and warm takes.** Between trials every process stops (SIGINT, then SIGKILL after
   20 s, the exit kind recorded) and the controller relaunches them; node-c's daemon starts a
   fresh spare, and node-b's compile cache is staged onto node-c over its own. The page cache,
   every on-disk cache outside the staged compile cache, the built kernels and the environment
   persist. A take is **cold** when it is node-c's first in the battery (take index 1) and
   **warm** otherwise; the reads confirm or contradict the rule, and a contradiction is reported.
   Predictions 8 and 9 are scored over warm takes. Cold takes are reported with their reads.
6. **Arm 8, new, conditional: node fault, cold node x3.** Before each trial the battery removes
   from node-c the directories item 4's reads name outside the staged set, so each take is a
   first take. If staging covers the term once named, the cold gaps fall inside prediction 8,
   and this arm shows it. The removal set is written here, dated, before arm 1, and the record
   names it. It never includes the install's own artifacts (the environment, the checkpoint, the
   built kernel extension, the binaries): removing one measures a broken install, not a cold
   node. If item 4 names no file, the arm does not run and the term is reported as one staging
   cannot cover.
7. **Session settings, as the driver pins them.** 8 sessions, 2,048-token prompts, 1,792
   generated tokens; arm 7 at 32,768 tokens of context. Arm 7 as built serves its one long
   session **alone**, not beside the eight, which isolates the staged copy; the text above is
   amended to match.
8. **Store sizing.** A store allocates each table at the engine's full slot count, so its
   budget follows the block count, not the tokens served: per rank, 40 tables of 4,096-byte rows
   per slot, and each serving node's store holds its own rank and its sibling's. At 2,048 blocks
   (8 sessions of 240 blocks, with margin) that is 10.7 GB; arm 7's 2,176 blocks are 11.4 GB.
   Every store's budget is 16 GB; the battery refuses to start an arm when the store's directory
   has less free space than the budget. The
   "about 5 GB" above is one rank's rows at 32k only.
9. **The stock restart's conditions.** The record names the stock launch command and the
   checkpoint's page-cache residency on node-a and node-b at relaunch, as it stands: a warm page
   cache, the stock path's best case.

Cost: arms 8 and the repeat add about twenty minutes; the estimate is about $40, the cap stays
$60.

## Clarification of the amendment, 2026-10-10, before any node is rented

Why: passes 17 and 18 (`dryrun-1gpu-fp8-pass17`, `-pass18`) read the files a cold take writes.
Pass 17's first take wrote 33.7 MB of Python bytecode compiled at first import; with bytecode
compiled at install (364b35a), pass 18's first take wrote nothing beyond what an idle node writes,
and its take equalled the warm one (5.74 s against 5.75 s).

1. **The baseline.** Before its first arm the battery reads every node over 60 s with no start
   (`written_since_idle_baseline_<node>.json`). A take's files beyond the baseline are its running
   read less every root the idle read touched.
2. **"Names a file" (items 4 and 6) means bytes beyond the baseline above zero.** A zero-byte file,
   such as the lock pass 18's take left under the FlashInfer cache, is listed in the record and
   does not count.
3. **If the smoke's cold take writes zero bytes beyond the baseline,** arm 8 does not run, and
   every cold take is scored against prediction 8's kill (a gap over 40 s fails) instead of only
   reported. Each cold take's bytes beyond the baseline are a prediction of their own: zero.
   **Kill: any byte.**

## Amendment, 2026-10-10, before any node is rented: what a replica restore would discard

Why: the comparison a buyer asks for is not only a stock restart but a whole-replica restore
from a snapshot, which brings back a warmed engine and none of the sessions in it. This run
measures no such restore. It derives, from records it already takes, what one would have
discarded at each fault, so the comparison is stated rather than implied. Nothing here is scored.

Per fault trial, under "reported, derived" in the summary, never beside the scored gaps:

- `replica_lost_sessions`: the sessions not finished at the kill.
- `replica_lost_prompt_tokens`: the sum of their prompt lengths.
- `replica_lost_generated_tokens`: the sum of the tokens each had committed at the fault, from the
  change record (listed per session).
- `replica_redo_retry_s`: the prompts prefilled again and each session's tokens generated again,
  as prompt tokens over stock's prefill rate plus the longest session's generated count times
  stock's decode step.
- `replica_redo_continuation_s`: prompt and generated tokens prefilled again, nothing generated
  again, over stock's prefill rate.
- `replica_restore_gpus` (the engine's TP x PP) beside `take_gpus` (the positions the change
  took).

The rates come from stock throughput trial 1 on the same nodes: the prefill rate from the first
step's start to the last session's first token, and the decode step as the mean step over steps
10 to 200. An adapter run's first step is never used: under the fault reads it is inflated. Arm 7
has no stock run at its length; its redo uses the same rates scaled linearly, and the record says
so.

Both redo figures are **lower bounds where approximate**: linear prefill under-counts attention at
long prefixes, steps 10 to 200 under-count decode at later positions, and a batch of eight short
sessions under-counts one long one. The one cost that could err the other way, a first step's
one-time cost, is stable across six stock records but not separable. Neither figure includes the
replica's own restore, which this run does not measure. On this shape a replica restore needs two
GPUs against the take's one; the figure exists here so that the multi-GPU and frontier runs carry
it from their first record.
