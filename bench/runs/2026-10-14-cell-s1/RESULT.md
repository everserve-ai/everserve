# S1 result: a 5,000-node cell comes up, and a change costs its engine, not the cell

Pre-registration: `PREREGISTRATION.md`. Numbers: `results.json`. Baseline:
`../2026-10-14-cell-s0`. Same host and harness, at 4ce8921.

| | 1,250 nodes | 2,500 nodes | 5,000 nodes |
|---|---|---|---|
| GPUs / engines | 10,000 / 625 | 20,000 / 1,250 | 40,000 / 2,500 |
| linked within | 0.51 s | 0.52 s | 0.52 s |
| CPU to link | 0.27 s | 0.57 s | 1.27 s |
| resident memory once linked | 70 MB | 133 MB | 263 MB |
| daemons sent to, per change | 2 | 2 | 2 |
| bytes received, per change | 593–609 | 593–617 | 593–617 |
| report-to-commit p50 / max | 0.9 / 1.9 ms | 1.3 / 1.9 ms | 1.6 / 5.6 ms |
| refused connects while linking | 960 | 2,133 | 4,463 |
| one readiness / one node list | 7 / 9 ms | 14 / 30 ms | 33 / 91 ms |

The link time is read by a 500 ms poll, so "within 0.52 s" is a bound, not a measurement.

At 1,250 nodes, against S0: linking used 0.27 CPU-seconds instead of 240, and memory was 70 MB
instead of 2.46 GB. A change sent 609 bytes to 2 daemons, against 115 KB to 1,250 before, and
committed in 0.9 ms instead of 16.

## Against the predictions

1. **Held:** 2 daemons per change at every size.
2. **Held:** at most 617 bytes per change; the bound was 5 KB.
3. **Held:** every size linked and committed all 20 changes.
4. **Held:** each doubling cost 1.9–2.0x the memory and 2.1–2.2x the link CPU; the bound was 3x.
5. **Held:** p50 of 0.9–1.6 ms; the bound was 32 ms.

## What it shows that was not predicted

- **The node list grows faster than the cell.** 9, 30 and 91 ms: 3.3x and 3.0x per doubling. A
  list serializes every node with each engine's ranks on it. That is S2 (the feed) and S4 (paged
  lists) at work, and they come next.
- **The first connects still meet the backlog.** Jitter spreads the retries, not the first
  attempt: every fake daemon starts at once, and a listener backlog of 128 refuses most of them.
  Production listens with Linux's larger backlog, and real daemons do not all start in the same
  millisecond. The count is recorded, not judged.
- **The change has no cell-wide work left at these sizes.** Its p50 grew 1.7x over a 4x cell. The
  one engines lock (S3) is still there, but it does not show at one change at a time. A run with
  concurrent changes is the test for S3.

## What changes in the plan

The cell comes up to 5,000 nodes, 40,000 GPUs, on one coordinator on a laptop. Next:

1. S2 and S4, for the read side.
2. Concurrent changes in the harness, to measure S3.
3. S5 and S6, sharding above one coordinator, which past one cell is a matter of design rather
   than repair.
