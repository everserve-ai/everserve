# Kernel build session, 2026-10-05: spend and what it produced

A build session on one GPU pod, not a measurement run. Commands, prices and timestamps are in
`run.log`; the development runs' raw outputs are in `data/` (not in git).

## Spend

| What | Cost |
|---|---|
| Pod `rsdyb5sl4ftg67`: 5× A100-SXM4-80GB, NVLink, secure cloud US, 14:38:40Z–16:56:00Z (2 h 17 min) at $7.95/h | **$18.20** |

The cap was $40, and the laptop watchdog was set to stop the pod before $37. The balance went from
$41.97 to $24.04 ($17.93, which lags the final minutes). The weights and `/dev/shm` were deleted,
the pod was terminated, and no pods remain.

## What landed (each gated, pushed, CI green before the next)

| Commit | Landing | Evidence on the pod |
|---|---|---|
| `6f935a1` | one gather kernel per step for the mirror, in `core/csrc` | byte-for-byte against the torch path for every layout (19 cases); the mirror round trip bit for bit; adapter 903.5 tok/s against 906.8 with the mirror off |
| `c4f2c01` | the slot collective in `core/csrc`, peer-to-peer and host-memory variants; `hostslot.py` deleted | bit-equal to `member_sum` in both variants, the abort release, and a graph captured before a change replayed after it; 16–18 µs against pynccl's 20.2 µs at decode size; a REPLACE run recovered |
| `98cf261` | concurrent aborts | each `ncclCommAbort` 0.41–0.76 s with P2P on; the change 0.39–0.55 s |

## Found on the way

- **Disk-backed staging cannot be pinned.** A file on the container's overlay disk cannot be pinned
  with `cudaHostRegister`, with any flags. Decision 0011's "disk-backed directory works through the
  page cache" does not hold; the error now says so.
- **The first P2P kernel was slower than NCCL** (32–36 µs at decode size). The cause was every
  thread reading the control words from host memory; fixed by one read per block and a rarer abort
  check.
- **`nvidia-ml-py` defines `NVML_P2P_CAPS_INDEX_READ` as the tuple `(0,)`.** `choose_variant` passes
  NVML's value 0.
- **The fence wait is the CPU's lead over the GPU under async scheduling, not a cost.** See the
  pre-registration's evidence for prediction 5.
