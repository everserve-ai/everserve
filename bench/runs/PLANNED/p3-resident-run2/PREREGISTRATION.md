# Pre-registration: the resident shared-spare run, again (p3-resident run 2)

Status: PLANNED 2026-10-10, written before any node is rented. Nothing below is changed after a
measured arm; an amendment before the first arm is dated and says why.

## Why this run exists

Run 1 (`bench/runs/PLANNED/p3-resident/run1`, NOTE.md) killed its own exactness claim: node fault
trial 2 lost the engine, a pipeline refusal read as an ordinary failure. Its long-context arm never
ran: the store's staging files filled node-b's shared memory. Every other completed trial was
exact. The claim is made again only by a run pre-registered for it, which this is. It runs the two
arms run 1 left unproven in full, and enough of the rest to read the fixes in place.

What changed since run 1, each on main before the deploy:

- A step a membership change ended raises as the loss it is (`6db3808`, `fed63b4`), with a pin
  guard on node-a in setup (`df748a4`).
- A change commits only over survivors still present, and the engine's head registers; its loss,
  or a change that fails after its delivery, puts the engine down with its cause (decision 0032).
- The store removes a connection's staging files when it ends and sweeps them at start
  (`58f4877`, protocol 3; protocol 4 with 0032).
- The resident copy is held at its exact size (`d348ec3`): run 1's spare pinned 124.6 GB for a
  72.7 GB model.
- The record splits the stock restart, names every file a take writes, scores the compile cache
  and reports the parts of the KV configuration term; the harness stops at the spend cap.

## Shape

As run 1: three single-GPU H100 80 GB VMs in one region, RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
at `f50dbad2c84590ca17dc51e207c34321b65ff14b` (the resolved revision checked on every node),
TP1 x PP2 on node-a and node-b, node-c the shared resident spare, sibling replication, the exact
guarantee. Eight sessions of 2,048-token prompts generating 1,792 tokens; 2,048 blocks; every
store's budget 16 GB; arm 6 at 32,768 tokens of context, one session alone. Every node runs the
commit the driver names, recorded with its hash.

## What runs, in this order

The order puts the arms run 1 left unproven first, so a stop at the cap or the watchdog cuts the
least.

1. **Smoke:** the repeat (the same sessions twice, no fault, tokens agree), then one node fault,
   node-c's first take (cold). The battery continues only if the repeat agrees, the node fault is
   exact, its gap is at most 40 s and the take wrote zero bytes beyond the idle baseline.
2. **stock throughput x1:** vLLM alone, the same-day reference for the overhead.
3. **adapter throughput x1:** the engine under the adapter, mirror on.
4. **node fault x5:** node-b's worker and daemon killed together mid-step; node-c's spare takes
   rank 1.
5. **long context x3:** one session at 32,768 tokens of context, node fault mid-decode.
6. **single death x3:** node-b's worker killed mid-step, its daemon alive.

After every fault trial the battery reads both stores with the engine quiescent: backup coverage
for both ranks.

## Predictions (kill conditions in bold)

Takes are cold at node-c's first take of the run and warm otherwise; gaps are scored over warm
takes, and the cold take against prediction 7's kill.

1. **Sessions continue exactly,** arms 4 to 6 and the smoke: every session's tokens identical to
   the control's, every trial. **Kill: any mismatch, or any trial that ends without its record.**
2. **The engine stays up:** no change fails and no engine is reported down in any trial
   (`everserve_engine_down` 0, readiness without `down`). **Kill: any.**
3. **Rows equal:** no differing position on either rank, every fault trial. **Kill: any.**
4. **Restore coverage** 1.0 and **backup coverage after the take** 1.0 for both ranks, every
   fault trial. **Kill: under 1.0.**
5. **No request re-prefilled;** a resume one below the committed count only with its cause named.
   **Kill: any resume lower, or any first batch computing more than one token for a request.**
6. **The compile cache holds** (`compile_cache_consistent`: readiness consistent before the trial,
   the taking spare's pair equal to its position's, no autotune result written during a trial).
   **Kill: any false.** **Row checksums:** no store refusal for checksum or unchecked. **Kill:
   any.**
7. **Node fault gap,** kill to first token after the change, median of five warm takes: 10 to
   25 s. **Kill: median over 40 s; the cold take over 40 s.**
8. **Single death gap,** median of three: 8 to 20 s. **Kill: median over 40 s.**
9. **The cold take writes nothing:** zero bytes beyond the idle baseline, zero-byte files listed
   and not counted. **Kill: any byte.**
10. **The take's terms,** each per trial: weights to the device 0.5 to 1.5 s (run 1: 0.72 s on
    every take; **kill: over 10 s**); KV configuration 4 to 6 s, its parts reported (run 1: 4.0
    to 5.2 s; **kill: over 10 s, or a profiling run logged**); module construction and warm-up
    reported.
11. **Long context,** median of three: the staged copy of the lost rank's rows finishes before the
    spare asks for them (**kill: the spare's staged wait over 5 s**), and the restore to the
    device is under 2 s. The copy's bytes and seconds are reported.
12. **Shadow formation at standing** under 30 s now that the resident copy is held at its size
    (run 1: 64.7 to 146.7 s). **Kill: median over 60 s.**
13. **Adapter throughput** within 5% of stock, steps 10 to 200. **Kill: more than 10% under.**
14. **Every stop clean:** no process the battery stops is killed at its stop, stock's own
    processes excepted. **Kill: any of ours killed.**

**Reported, not scored:** the clock offsets; the fabric's bandwidth; each acknowledgement's time
at its daemon (which process a change waited on); the join route's stamps; the replica-restore
loss on every fault trial (derived, lower bounds); the spend at each arm.

## Cost

Three H100 VMs at the billed $4.45 per GPU-hour, $13.35 an hour. From run 1's minutes: setup 15,
smoke 27, stock 11, adapter 10.5, node fault 70, long context 48 (16 a trial, run 1 having no loss
trial to time), single death 45, release 5: 231.5 min, about **$51.50**. **Cap $60**, enforced by
the harness from the rate and the VMs' start times; the watchdog is the estimate plus fifteen
minutes. Booked on Cort's word, given 2026-10-10 in the agent's session.

## What this run does not claim

- Tensor parallelism inside a node, frontier size, mixture-of-experts, expert parallelism across
  nodes.
- A second loss: the pool does not refill (decision 0031, not built).
- The on-notice KV mode, degrade-then-restore, coordinator failover.
- Any figure leaves the repository only after its record is committed.
