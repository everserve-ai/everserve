# S3: do concurrent changes of different engines wait on each other?

Committed before the run. Baseline: `../2026-10-14-cell-s1` (one change at a time).

## Why

The coordinator holds every engine's view under one lock (`engines: Mutex<BTreeMap<..>>`). The
S1 record measured changes one at a time, which cannot show contention. If every change holds
the lock for most of its work, a burst of changes on different engines finishes in about the
sum of their times, and engines are not independent.

## What runs

The cell harness at the commit that adds its burst phase: after the 20 one-at-a-time changes,
100 SHRINK changes are reported at once, each on another engine. It records the burst's wall time
and each change's report-to-commit time (p50, max). The host is the same as S1's, and so is the
size sequence:

```text
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture cell_1250
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture cell_2500
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture cell_5000
```

## Predictions

1. Every change in every burst commits.
2. **The lock is held briefly:** a burst of 100 finishes in less than 10x the size's
   one-at-a-time p50 change time (sequential would be about 100x).
3. The burst's wall time does not grow by more than 2x from 1,250 to 5,000 nodes.

## What would change the plan

If 2 fails, S3 (a lock per engine) is built before anything else, and this run is its baseline.
If 2 and 3 hold, S3 is not needed at cell size: it is recorded as measured, not built, and the
review says so.
