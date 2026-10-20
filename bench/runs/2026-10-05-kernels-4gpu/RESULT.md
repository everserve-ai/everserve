# Result: the kernels on the four-GPU rig, peer-to-peer on

**Pod and software.**
- Run 2026-10-05T17:47Z to 18:08Z on RunPod pod `3ocsxq7ebzoe4w`: 5× A100-SXM4-80GB, NVLink (NV12,
  every pair), secure cloud, US, $7.95/h. Driver 595.91.07, no `cuda-compat` (it breaks CUDA on
  this driver; see the harness fix below). Topology in `data/topology.txt`.
- vLLM at the pin (`6e517b15`), torch 2.13.0+cu130. **No file under `vllm/` was edited.**
- 5× A100 had no supply at first. A 5× H100 pod came up at $17.45/h, which covers about 34 minutes
  under the cap against about 50 needed; it was terminated after 22 s (about $0.03).

**Peer-to-peer and the slot variant.** No `NCCL_P2P_DISABLE`. The slot variant was **not logged**:
the adapter does not record what `choose_variant` returned, and the region file went with
`/dev/shm`. It was peer-to-peer by every indirect sign: NV12 between every pair, 21 kernel tests
passed on the pod (the skip count was not logged), and the adapter's 905 tok/s equals the build
session's peer-to-peer runs (904). This is a recording gap against the pre-registration.

**Data directory.** `/dev/shm` was 274 GB; the store ran with `--data-dir /dev/shm/everserve`.

**Tree.** `7d44e95` (preflight and CI green). The only change on the pod is the bench-only fix to
`adapter_port.sh` (keep `cuda-compat-13-0` only if CUDA works with it), committed with this result.
`PREREGISTRATION.md` is unchanged since `7d44e95`.

**Stock ran with vLLM's default custom all-reduce on** in all three runs (`custom_all_reduce: true`);
the fallback to off was not needed. The adapter arms ran with it off, as pre-registered.

**Records.** `results.json` (every figure per run, the metrics per arm, the store span per rank,
each abort, SHA-256 of `data/`), `run.log`, `data/` (not in git), `flake/` (every rank's event log
for the REPLACE and notice runs).

18 runs, all pre-registered, in order, on this pod. Nothing was repeated or added afterwards.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | adapter overhead < 3 % (> 5 %) | 7.81, 7.82, 7.87 % | **dead** |
| 2 | mirror-off overhead < 1 % (> 3 %) | 7.42, 7.28, 7.41 % | **dead** |
| 3 | REPLACE pause < 1.5 s every run (> 2.5 s), the abort named | 0.94–1.50 s; the largest 1.498 s | **alive** |
| 4 | notice: swap in all, < 0.3 s (no swap, > 1.0 s, a fault) | swap 3/3, 0 faults; 0.753, 0.212, 0.199 s | **weakened** |
| 5 | store span (`sender`) p50 < 1 ms every rank (> 3 ms) | 0.31–0.49 ms, every rank, every mirror-on run | **alive** |
| 6 | spare restore < 0.4 s (> 1.0 s) | 0.155–0.179 s | **alive** |
| 7 | sessions kept, 20 tokens equal, replay past the prompt (any lost, re-prefilled, mismatch) | 8/8 kept, replay from 2,149; **session 1 mismatched in 6/6 REPLACE runs** | **dead** |
| 8 | restored rows equal the dead rank's (any mismatch) | 48/48 tables, 9/9 runs | **alive** |
| 9 | commit within 100 ms of resume, epochs applied | 22.9–27.7 ms REPLACE, 52.4–60.6 ms notice; epoch 1 everywhere | **alive** |

## Overhead (all from this pod, this session)

| Arm | tok/s, steps 10–200 | Mean | Overhead against stock |
|---|---|---|---|
| stock, custom all-reduce on | 981.5, 983.2, 982.0 | 982.2 | — |
| adapter, mirror off | 909.4, 910.7, 909.4 | 909.8 | 7.37 % |
| adapter | 905.5, 905.5, 905.0 | 905.3 | 7.83 % |

**What killed predictions 1 and 2 is the baseline, not the adapter.** The adapter's throughput is
what the build session measured (904–907 tok/s). Stock with custom all-reduce off ran 837 tok/s
there; with it on it runs 982 here. With vLLM's custom all-reduce, stock runs about 8 % faster
end to end than the adapter on NVLink A100s. The slot collective beats pynccl (16–18 µs against 20.2 at
16,384 elements), but the custom all-reduce is the baseline the adapter has to match. The mirror
costs 0.5 % (909.8 against 905.3).

## The REPLACE pause, with each abort

Seconds. The aborts are each survivor's `ncclCommAbort` of the live TP handle and of the EP
group's pynccl handle, run together; the torch group's abort was 0.001–0.002 s in every run.

| Run | Pause | Detect | Change | Abort, live TP (ranks 0/1/3) | Abort, EP (ranks 0/1/3) | Restore | Resume | First token | Daemon change |
|---|---|---|---|---|---|---|---|---|---|
| between 1 | **1.498** | 0.356 | 0.279 | 0.391 / 0.391 / 0.392 | 0.430 / 0.429 / 0.429 | 0.156 | 0.026 | **0.682** | 0.434 |
| between 2 | 0.940 | 0.377 | 0.254 | 0.403 / 0.405 / 0.403 | 0.380 / 0.373 / 0.382 | 0.155 | 0.025 | 0.130 | 0.409 |
| between 3 | 1.131 | 0.338 | 0.442 | 0.403 / 0.404 / 0.398 | 0.380 / 0.380 / 0.617 | 0.179 | 0.027 | 0.145 | 0.620 |
| mid 1 | 1.096 | 0.411 | 0.368 | 0.092 / 0.520 / 0.126 | 0.366 / 0.365 / 0.365 | 0.157 | 0.028 | 0.129 | 0.524 |
| mid 2 | 1.125 | 0.413 | 0.393 | 0.531 / 0.535 / 0.540 | 0.501 / 0.547 / 0.541 | 0.159 | 0.025 | 0.132 | 0.551 |
| mid 3 | 1.064 | 0.382 | 0.368 | 0.045 / 0.061 / 0.524 | 0.503 / 0.053 / 0.053 | 0.160 | 0.023 | 0.127 | 0.527 |

- **The abort sets the change, as stated before the run.** The slowest abort on any survivor is
  0.39–0.62 s, and the change is 0.25–0.44 s after detection plus the restore.
- **The first REPLACE run passed by 2 ms, and not because of the abort.** Its detection, change and
  restore are ordinary; its first token after resuming took 0.68 s against 0.13 s in the other
  five. See the surprises.
- Mid-step, some survivors' aborts return in 0.05–0.13 s: a handle with no kernel in flight
  tears down fast.

## Driver pause against the daemon's change

| Kind | Run | Driver pause (largest gap) | Daemon `everserve_change_seconds` |
|---|---|---|---|
| REPLACE | between 1–3 | 1.498, 0.940, 1.131 | 0.434, 0.409, 0.620 |
| REPLACE | mid 1–3 | 1.096, 1.125, 1.064 | 0.524, 0.551, 0.527 |
| notice | 1–3 | 0.753, 0.212, 0.199 | 0.176, 0.165, 0.168 |

The daemon's change excludes detection and the engine's first step after it resumes.

## The store span (prediction 5)

`sender` (copies landed → the store's acknowledgement) per rank, p50: 0.31–0.49 ms in every
mirror-on run; p90 1.05–2.73 ms. Reported beside it, without a verdict: `fence_wait` p50
6.48–6.75 ms and `event_lag` p50 14.9–15.2 ms, as in the build session. The step thread's lead
over the GPU under async scheduling is unchanged.

## Prediction 7: what failed

- **Session 1's tokens after the fault differ from the no-fault run in all six REPLACE runs**, both
  kill timings. It diverges at token 114 every time; the kill came after token 101 (between) or
  102 (mid). The other seven sessions match. No session was lost or re-prefilled; every replay
  started at token 2,149, past the 2,048-token prompt. The restored rows equal the dead rank's
  (prediction 8), so the KV is not the difference.
- **Likely cause, not verified on a GPU:** the slot collective sums the members in row order
  (`core/csrc/slot.cu`, "sums the members in row order in fp32"). The spare takes row 4
  (`executor.py`, `row=world`), so after REPLACE the replaced rank's partial is added fourth instead
  of third. The TP all-reduce has four producers per element, so a different fp32 order rounds
  differently in bf16, and session 1 flips a token 13 steps later. The divergence is deterministic,
  which fits. In earlier runs the TP all-reduce was pynccl, where the spare took the dead rank's
  place in the ring, and 8/8 sessions matched in every run.
- The fix this points to is summing in TP-rank order rather than row order, so a spare takes the
  dead rank's place in the sum. That is a product change for a later session, not made here.
- **The notice runs were not tested for equality.** The reference is `adapter_1` (201 tokens) and
  the notice swap comes at about token 420, so `after_fault_match` is false by construction for
  every notice session. The harness needs a no-fault reference of the notice length.

## Surprises

- **Stock with custom all-reduce on is 17 % faster than with it off** (982 against 837 tok/s), and
  that is what turned the adapter's 8 % lead in the build session into an 8 % deficit.
- **The first run of each fault arm pays about 0.55 s in its first step after resuming**:
  replace_between_1 (first token 0.68 s) and notice_1 (0.70 s), against 0.13–0.16 s in every
  other run. The shared-memory run showed the same in its notice_1 (0.829 s against 0.27, 0.25).
  The notice arm runs with a different `--max-model-len` from the REPLACE arms, so a first-use
  cost cached on disk per configuration would fit; it is not identified.
- **replace_between_1's tok/s is 660** against 850–863 in the other REPLACE runs. REPLACE throughput
  is not a prediction.
- **The setup's CUDA check misfired on driver 595**: it installed `cuda-compat-13-0`, with which
  torch saw no CUDA. Fixed in `bench/harness/adapter_port.sh` (keep the compat libraries only if
  CUDA works with them) before the first arm.

## Dollars

| What | Cost |
|---|---|
| Pod `3ocsxq7ebzoe4w`, 5× A100 SXM, 17:37:46Z–18:08:01Z (30 min) at $7.95/h | $4.01 |
| Pod `0j9kgael3im7si`, 5× H100 SXM, 22 s at $17.45/h, terminated unused | $0.03 |
| **Total** (cap $10) | **$4.04** |

The balance went from $23.68 to $20.17 ($3.51; it lags the last minutes). The weights and
`/dev/shm` were deleted, both pods were terminated, and no pods remain.
