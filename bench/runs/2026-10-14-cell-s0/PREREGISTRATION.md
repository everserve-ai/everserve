# S0: a synthetic cell at 1,250, 2,500 and 5,000 nodes

Committed before the run. The review it measures is
`docs/reviews/2026-10-14-control-plane-at-fleet-scale.md`.

## What runs

`controller/tests/cell.rs`, at b0ecf3a, in release mode:

```text
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture --test-threads 1 cell_1250 cell_2500 cell_5000
```

One coordinator process. N fake daemons connect over the coordinator's real link (Unix
sockets). Each node has eight devices, and each engine has two nodes, so 16 ranks. Every engine
is SHRINK-only. 20 changes run one at a time, each on another engine. Every change loses rank 0.

The host is one Apple M-series Mac (14 cores, 36 GiB, `ulimit -n` 1048576). The absolute times
are this host's. The claim is how each figure changes with the cell's size.

## What it records, per size

- `link_s`: the time until every node is linked and the last engine's view is whole.
- Resident memory, and CPU time, while linking and while the changes run.
- `change_s`: report-to-commit time, p50, p99 and max.
- `daemons_sent_to_per_change`: how many daemons got any frame in the 500 ms after a change was
  reported.
- `bytes_received_per_change`: those frames' bytes, length prefixes included.
- `readiness_s` and `node_views_s`: one fleet-wide read each, after the changes.

## Predictions

1. **S1 holds:** `daemons_sent_to_per_change` is the cell's node count (1,250, 2,500, 5,000), not
   the engine's two. If it is 2 at every size, S1 is wrong, and the review gets corrected.
2. Bytes per change grow in proportion to the node count, within a factor of 1.5 of 2x and 4x
   the 1,250-node figure.
3. The p50 change time grows with the node count. The fan-out runs under the one lock that holds
   every engine (S3).
4. `readiness_s` and `node_views_s` grow in proportion to the node count.
5. Every size links and commits all 20 changes. A failure to link 5,000 nodes in 10 minutes is a
   finding, not an error in the harness, unless the run's log shows it is one.

## What would change the plan

If 1 and 2 hold, S1 and S3 come next, as the review orders them. After that, the same harness is
run again with its bound asserted. If change time stays flat while the fan-out grows, S3 moves
behind S2.
