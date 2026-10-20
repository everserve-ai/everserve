# The p3-resident run 2 rental (2026-10-10 17:52–21:01 UTC): run 1's fixes on the same shape, every fault trial exact

The run `../PREREGISTRATION.md` describes, on Cort's word (2026-10-10, "i just added $60 again ...
feel free to run the session"): three single-GPU H100 80 GB VMs on Nebius eu-north1
(`gpu-h100-sxm` `1gpu-16vcpu-200gb`); RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic at
`f50dbad2c84590ca17dc51e207c34321b65ff14b`; TP1 x PP2 on node-a and node-b; node-c the shared spare,
holding the whole model in host memory.

The driver named `a6d07c3`. Every measured arm ran at `c5bea92`: see the intervention below. The
setup guards on node-a ran 8 tests and skipped none, including the resident copy page-locked at its
own size (`setup/pin_guards_a.txt`).

**Spend:** about **$41.70**, under the $60 cap: three VMs from 17:52:15Z to 20:58, 20:59 and
21:01Z, 9.37 GPU-hours at the billed $4.45 per GPU-hour. `nebius.py budget` read $40.22 at 20:53Z.
The billing page is the record. Neither the budget cap nor the watchdog fired.

## Result

| arm | trials | verdict | headline |
|---|---|---|---|
| smoke: repeat | 1 | green | the two runs agree 8/8 for all 1,792 tokens |
| smoke: node fault (cold take) | 1 | green | 8/8 exact, gap 13.16 s, 0 bytes beyond the idle baseline |
| stock throughput | 1 | measured | 270.64 tok/s, steps 10 to 200 |
| adapter throughput | 1 | green | 1.25% under stock |
| node fault | 5 | green | **5/5 exact**, gap median **13.06 s** (10.03 to 14.58) |
| long context (32,768 tokens) | 3 | green | **3/3 exact**, gap median 27.93 s (22.01 to 30.06) |
| single death | 3 | green | 3/3 exact, gap median 12.48 s (9.83 to 12.50) |

No trial failed. No engine went down in any trial (`engine_down` false on every one). Every stop
of ours was clean.

**Run 1's failure recurred and was handled.** In node fault trial 4 the kill landed 68% into step
288. Rank 0's sampled-token step came back refused 5.7 ms after the coordinator's word
(`step_refused`: "step aborted: the member set changed: the pipeline refused its group until the
boundary"). The executor read the refusal as the loss. The spare took the position 14.4 s later,
and every session stayed exact. This is the race that shut the engine down on run 1's node fault
trial 2, fixed at `6db3808`. Long context trials 1 and 2 hit the same refusal and were handled the
same way.

## The predictions

1. **Sessions continue exactly: holds.** Every fault trial was exact (the smoke's, 5 node fault,
   3 long context, 3 single death), and no trial ended without its record.
2. **The engine stays up: holds.** `engine_down` read false on every trial: no change failed, and
   no readiness read before or during a trial said down.
3. **Rows equal: holds.** No differing position, and 0 restored rows mismatched, on every fault
   trial.
4. **Restore coverage and backup coverage after the take:** 1.0 on every fault trial.
5. **No request re-prefilled, no resume below the committed count:** 0 on every fault trial.
6. **The compile cache holds: killed** on check (b), on every take (the smoke's node fault, all
   five node faults, all three long contexts, all three single deaths).
   - **Why check (b) fails.** It compares node-b's and node-c's whole-node digests
     (`cache_digest.py`), which hash every file under the artifact. `cache_sync`'s in-place build
     of Triton's driver helper (`cuda_utils.cpython-312-x86_64-linux-gnu.so`, `a6d07c3`) adds a
     node-local file to node-c's staged cache after node-b's digest is taken.
   - **What held.** Readiness read consistent on every trial, no take was refused, and no autotune
     result was written. The comparison the coordinator enforces is the per-rank digest
     (`rank_digest`, `PRIVATE-INTERNALS.md` A40), which leaves the helper out.
   - **Re-derived after the run with that comparison** (`compile_cache_pairs_rederived.json`;
     this never replaces the scored line). In 7 takes, the spare's registered pair equals its
     position's. In the other 5 (node fault 2 and 4, long context 1 to 3), the harness's readiness
     read landed between the change's claim and its commit, when the view holds no rank 1. So
     those 5 are not read. The readiness view reports the claimed epoch as if committed; the fix is
     the peer's, to follow.
   - **Row checksums:** no store refusal on any trial.
7. **Node fault gap: holds.** Median of five warm takes 13.06 s, inside 10 to 25 s. The cold take,
   13.16 s, is under 40.
8. **Single death gap: holds.** Median of three 12.48 s, inside 8 to 20 s.
9. **The cold take writes nothing: holds.** 0 bytes beyond the idle baseline. Two zero-byte files
   are listed: `~/.cache/flashinfer` and `~/.humming/tmp`.
10. **The take's terms.**
    - Weights to the device: 0.72 s on every take, inside 0.5 to 1.5 s.
    - **KV configuration: 1.99 to 2.10 s on the 2k-context takes, 3.04 to 3.07 s at 32k. Outside
      the predicted 4 to 6 s, below it**, the kill not met (no profiling run logged). The parts
      did not arrive. The spare runs as `python -m everserve_vllm.spare_main`, so the parts were
      kept in `__main__`, and `attach()` read the imported module's empty copy. The drop from run
      1's 4.0 to 5.2 s is therefore unexplained here. Run 1's spare held 124.6 GB pinned on a
      196 GB node, and it may have squeezed this term; that is a guess, not a reading.
    - Module construction 1.9 to 2.2 s; warm-up 3.4 to 3.6 s (5.1 to 5.3 s at 32k).
11. **Long context: killed** on the staged wait. Median 6.32 s (6.91, 6.32, 4.58) against the kill
    at over 5 s. The restore to the device holds: median 1.12 s, under 2 s. The cause is in
    node-a's store (the peer's reading of the store logs): it builds the whole rank's snapshot
    under its lock, then clones it before sending, so about 5.4 GB is copied twice before the
    first byte leaves. The send then runs at under 1 GB/s against a measured 59 to 66 Gbit/s
    fabric. A streamed snapshot is the fix, being gated as this is written.
12. **Shadow formation at standing: holds.** 0.42 to 0.52 s against under 30 s (run 1: 64.7 to
    146.7 s), with the resident copy held at its own size (`d348ec3`).
13. **Adapter throughput: holds.** 1.25% under stock.
14. **Every stop of ours clean: holds.** `stops_killed` 0 on every trial. Stock's own: node-b's
    stock headless was killed at its stop (`stock_stops_killed` 1), as on run 1.

**Reported:**
- Clock offsets 0.06 to 0.11 ms between node-a and the others.
- The replica-restore loss, derived per fault trial in each summary.
- The spend at each trial line, in `battery/run.log`.
- Where each kill landed (`change.kill_in_step`): the node fault kills fell 11% to 68% into their
  step.
- `instrument_pause_s`, the drill's native-stack dump of the survivors three seconds after the
  kill, outside every gap: about 49 s on the 2k-context arms and 165 s at 32k.

## Where the gap goes

From node fault trial 5's change record:

| part | node fault trial 5 | long context trial 2 |
|---|---|---|
| kill to loss heard | 0.012 s | 0.012 s |
| take (weights, modules, KV configuration, warm-up) | 8.77 s | 13.38 s |
| staged rows' wait | 0.30 s | 6.32 s |
| restore to the device | 0.70 s | 1.11 s |
| join answers back (`join_legs_s.answers_back`) | 3.10 s | 9.03 s |
| commit to the first step's end | 0.15 s | 0.20 s |
| **gap** | **13.06 s** | **30.06 s** |

The join answers' leg grows with context (3.1 to 9.0 s), and its cause is not yet named.

## Intervention, in `deploy.log`

At 18:06Z the smoke's battery died before any measurement: `PY: unbound variable`.
`${PREDICTIONS:?... the run's pre-registration}` opened a single-quoted string at the apostrophe,
which closed 28 lines on, so every assignment between was skipped. `bash -n` passed, the quotes
balancing.

Fixed and gated at `c5bea92`: the message reworded, and a harness test that fails on an unclosed
quote in any parameter expansion. The smoke reran at 18:16Z within the hold, on the same VMs.

That commit also swept in this run's partial `deploy.log`, `setup/pin_guards_a.txt` and the
generated `bench/harness/policies/battery.json`, which stays uncommitted. This record's commit
restores that file.

## Found, for the next run

- **The OS's apt cache still refreshes inside takes:** 122 MB of `/var/cache/apt/pkgcache.bin`
  and `srcpkgcache.bin` in node fault trial 4 and single death trial 1, plus the ESM caches. Setup
  disables `apt-daily.timer` and `apt-daily-upgrade.timer`, so something else runs apt; the ESM
  refresh is a candidate. Not named without a node. Both takes were warm, so no prediction reads
  it.
- **Long context trial 1's take compiled three of vLLM's input kernels** (218,259 bytes across 24
  files in `~/.triton/cache`): `_gather_block_tables_kernel`, `_prepare_prefill_inputs_kernel`
  and `_prepare_pos_seq_lens_kernel`. These are specialisations node-b's staged cache did not
  hold. Trials 2 and 3 found them in node-c's own cache.
- **The KV configuration's parts:** kept in a module the spare imports, not `__main__`.
- **Check (b):** compare the pairs the coordinator enforces. Readiness reports the committed
  epoch, the change in flight apart (the peer's).
- **The join answers' leg:** 3 s at 2k context and 9 s at 32k, to be named.
