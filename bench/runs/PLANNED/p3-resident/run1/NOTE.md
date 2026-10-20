# The p3-resident rental, run 1 (2026-10-10 10:14–15:08 UTC): a 70B FP8 engine across two H100 nodes, a shared resident spare

The run `PREREGISTRATION.md` describes, on Cort's word: three single-GPU H100 80 GB VMs
(Nebius eu-north1, `gpu-h100-sxm` `1gpu-16vcpu-200gb`), RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
at `f50dbad2c84590ca17dc51e207c34321b65ff14b`, TP1 x PP2 on node-a and node-b, node-c the shared
resident spare. Deployed at `20853ae`; the smoke ran again at `fb3a48f` after the fix below. Eight
sessions of 2,048-token prompts generating 1,792 tokens in every arm but arm 7.

**Spend: $65, over the $60 cap.** That is the billing page's figure, as Cort read it. The VMs ran
from 10:14Z to 15:05, 15:07 and 15:08Z, 14.6 GPU-hours. This note first estimated $43 at an
assumed $2.95 per GPU-hour, which the CLI cannot confirm. $65 over those hours is about $4.45 per
GPU-hour, everything billed included. The re-armed watchdogs (16:00Z) were set against the $43
estimate. The cap was never checked against the real rate, so nothing stopped the run at $60.

## Result

| arm | trials | verdict | headline |
|---|---|---|---|
| 1 stock throughput | 1 | measured | 271.15 tok/s, steps 10 to 200 |
| 2 adapter throughput | 2 | within 5% | 1.38% and 1.15% under stock |
| 3 adapter, mirror off | 1 | measured | 1.26% under stock; the mirror's cost is inside the noise |
| 4 stock restart | 3 | measured | first token 403.3, 402.6, 404.4 s after the kill, 0/8 sessions kept |
| 5 node fault | 5 | **red** | 4 trials exact (gap 14.39, 13.72, 17.23, 14.18 s); **trial 2 shut the engine down** |
| 6 single death | 5 | green | 5 trials exact, gap median 17.09 s (14.23 to 17.67) |
| 7 long context | 3 | **red** | **no trial ran**: node-b's store directory was full of staging files |

The smoke, before the battery: the repeat agreed 8/8 for all 1,792 tokens, and node-c's first take
(the cold one) kept 8/8 exact with a gap of 18.99 s and zero bytes written beyond the idle baseline.
So arm 8 did not run, and the cold take is scored against prediction 8's kill, which it passes.

## The predictions

1. **Sessions continue exactly: killed.** Node fault trial 2 lost the engine. Every trial that
   completed was exact: 8/8 sessions identical to the control on 4 node fault trials, 5 single
   death trials and the smoke's node fault. Arm 7 has no trial.
2. Rows equal: no differing position on any completed fault trial.
3. Restore coverage 1.0 on every completed fault trial.
4. Backup coverage after the take 1.0 for both ranks on every completed fault trial.
5. No request re-prefilled and no resume below the committed count.
6. The compile cache held: no inconsistent readiness read was logged, and the staged cache's
   digests on node-c equalled node-b's before every fault trial (`cache_digest_{b,c}.json`). No autotune file was written during a trial. The 55,832 bytes below are Triton's
   compiled driver helper, not an autotune file.
7. Store refusals for checksum or unchecked: zero on every store, every trial.
8. Node fault gap: the median of the four warm takes that completed is **14.28 s**, inside 10 to
   25 s, and the cold take's 18.99 s is under the kill. This is **not the median of five** the
   prediction names: trial 2 has no gap.
9. Single death gap: median of five **17.09 s**, inside 8 to 20 s.
10. The take's terms. Weights to the device were **0.72 s on every take, under the predicted 1.5 to
    4 s** (the kill is over 10 s). KV configuration was **4.0 to 5.2 s on every take but one
    (2.14 s, single death trial 1), missing "under 2 s"**. No profiling run was logged, so the
    kill is not met, but what the term spends its time on is not yet named. Modules took 2.3 to
    3.4 s and warm-up 3.8 to 4.8 s; both are reported, not scored.
11. Long context: **no reading.** See below.
12. Stock restart: median 403.3 s (6.7 minutes), inside 5 to 20 minutes. Split after the run
    (`summary_split.json` in each trial, by `stock_restart.py` as amended; medians):
    - kill to relaunch: 316.9 s. Of that, about 300 s is vLLM's worker RPC timeout on node-a:
      trial 2's drill log has `RPC call to sample_tokens timed out` at 12:13:29Z, against the kill
      at 12:08:28.9Z. The exit was not stamped in this run, so detection is read from the log, to
      the second.
    - relaunch to the relaunched engine's first step: 84.7 s, of which 25.4 s is the engine's own
      load (`engine_ready_s`). The other 59 s goes to launching both halves before vLLM's load
      timer starts, unnamed.
    - first step, the eight prompts prefilled again: 2.4 s.
13. Adapter throughput: 1.38% and 1.15% under stock, inside 5%.

**Reported, not scored:**
- Shadow formation at standing took 64.7 to 146.7 s on the battery's arms (10.9 and 11.4 s in the
  smoke). That is the spare's pin, which takes 53 to 178 s under the over-pinning named below.
- Clocks of node-b and node-c against node-a: offsets within 0.052 ms over the run's ten reads,
  each with an error under 0.15 ms.
- Derived replica-restore loss on every completed fault trial: 8 sessions lost, a redo of 9.9 to
  13.2 s by retry and 2.7 to 2.8 s by continuation (both lower bounds, the replica's own restore
  not included), against 2 GPUs for a replica restore and 1 for the take.

## Node fault trial 2: the engine shut down on a pipeline refusal

Node-b was lost at step 100. 0.4 s later, node-a's sampled-token broadcast refused its pipeline
group on its daemon link's word, before the coordinator's word reached the executor. vLLM's worker
RPC carries a failed call as the exception's text alone. The refusal crossed it as an ordinary
failure, and the engine core shut down (`node_fault_2.drill.log`, rc 1). The change itself
committed (epoch 1).

Fixed on main at `6db3808` and `fed63b4`, not in this run:
- Every error a step raises because the member set changed opens its message with one marker.
- The executor reads a failed step RPC that carries the marker as the loss, once its own record
  holds that loss, within `EVERSERVE_HANG_S`.
- `PRIVATE-INTERNALS.md` A8a names the transport the fix relies on.
- A guard that needs an installed vLLM fails if that transport changes. Setup runs it on node-a
  before any arm (`df748a4`).

The battery wrote no `summary.json` for the trial, so `arm.json` read green over four trials.
Battery now records a trial that ended without its record as a failed, red trial of its arm
(`6db3808`).

## Long context: node-b's store directory filled with staging files

Arm 7's control ran clean. The battery then refused the loss trial's start: node-b's
`/dev/shm/mirror` had 13.5 GB available, under the store's 16 GB budget. Trial 2's control was
refused the same way, and the arm stopped.

The directory, read on node-b before release (`battery/arms/long_context/node_b_shm_after.txt`),
held 80.6 GB of `stage-<pid>-{1,2,3}.rows` files:
- one set for every start of node-b's worker from 10:35Z on;
- 2.7 GB each, and 5.4 GB for the long-context start.

A worker's staging files went only at a clean close, and every killed worker left its set. Fixed
in the store on main at `58f4877`, not in this run:
- each connection's staging files are named by the store;
- they are removed when the connection ends, however it ends;
- leftover staging files are swept when the store starts;
- protocol 3.

The harness does not clean the directory: a customer's node would hit the same leak.

## Writes beyond the idle baseline

These were scored against the clarification's per-take reads.

- **Cold takes** (the smoke's, and arm 2's to 4's node-c starts at take index 1): zero bytes, one
  zero-byte file in `~/.humming/tmp`.
- **55,832 bytes in `~/.cache/vllm`, on node fault trials 1 and 3:** Triton's compiled driver
  helper, rebuilt because staging the compile cache replaces the directory. The install should
  build it, so staging never carries node-local artifacts.
- **122 to 128 MB on adapter trial 1, node fault trial 5 and single death trial 4:**
  `/var/cache/apt` and `/var/lib/ubuntu-advantage/apt-esm`. That is the OS's periodic apt refresh
  landing inside a take's window; the take wrote none of it. The baseline does not include the
  refresh because it runs on a daily timer. Earlier notes called this "61 MB in the adapter arm";
  that figure was the arm's mean.

The `predicted` and `kill` fields printed in each `summary.json` come from the harness's own table
of an earlier pre-registration (for example "0.5-1.5, median > 4" on the gaps). This note scores
against `PREREGISTRATION.md`. The summary should carry the pre-registration it is scored against.

## Interventions during the run, each in `deploy.log`

- The smoke's idle baseline read passed an empty `--start` before any start existed. The smoke was
  stopped before node-c's first take, fixed at `fb3a48f`, and run again.
- The kill-time rows file (`*.rows_rank1.pt`, 2.8 GB at 70B) was re-copied by every collect. From
  11:49Z each one was parked in `/workspace/cores/rows_at_kill/` on its node, with a link left at
  its path, so `kv_rows.py` still read it and collect carried only the link. The files left with
  the VMs, and the comparisons made against them are in each trial's record. Collect should skip
  them.
- `fincore` is in `util-linux-extra` on Ubuntu 24.04. It was installed on node-a and node-b at
  12:06Z, so stock restart trial 1's page-cache read is empty. Trials 2 and 3 found the
  checkpoint fully resident. Setup should install it.
- The watchdogs were re-armed twice, to 15:40Z and then to 16:00Z, as the projections moved. They
  did not fire; the driver released the VMs at 15:05 to 15:08Z.
- Stock throughput's stop killed node-b's stock headless process after 20 s (`stops_killed` 1).
  Every adapter arm's stop was clean.

## Found, for the next run

- The resident loader's over-pinning: torch's caching host allocator rounds each pinned buffer up
  to a power of two. The spare holds 124.6 GB for a 72.7 GB model (15 files of about 4.9 GB,
  each 8 GiB), and the pin takes 53 to 178 s. The fix is exact-size buffers registered with the
  driver.
- The KV configuration term (4 to 5 s against under 2 s) needs its parts named.
- The stock restart's 59 s between the relaunch and vLLM's load timer needs its parts named;
  `stock_restart.py` now records the split, with the dead engine's exit stamped.
- A follow-up run with `6db3808` and the store's staging fix, to read arm 5 over five trials and
  arm 7 at all. Only on Cort's word.
