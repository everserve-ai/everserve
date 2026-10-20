# S1: the synthetic cell again, with an engine's facts sent only to its audience

Committed before the run. The baseline is `../2026-10-14-cell-s0/RESULT.md`.

## What changed since S0

- An engine's policy, placement and instance state go only to its audience (`coord/audience.rs`).
- A new link gets only the facts of its own engines.
- A hello refreshes replication only for its engines.
- Daemon reconnects are jittered between half and all of the backoff wait.

## What runs

The same harness, at this commit, in release mode, on the same host, one size at a time:

```text
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture cell_1250
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture cell_2500
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture cell_5000
```

## Predictions

1. `daemons_sent_to_per_change` is 2 at every size: the engine's own nodes.
2. Bytes received per change stay under 5 KB at every size (S0: 110–115 KB at 1,250 nodes).
3. All three sizes link within the 10-minute deadline and commit all 20 changes.
4. Link time and resident memory grow roughly in proportion to the node count. Each doubling
   of the cell costs less than 3x the time and memory of the size before it.
5. The p50 change time stays within 2x of S0's 16 ms at every size.

## What would change the plan

If 3 fails, read where linking stops before anything else. If 4 fails, linking still has cell-wide
work, so find it before S2. If 5 fails while 1 holds, the change path itself has cell-wide work
under the engines lock, and S3 moves ahead of S2.
