# S7: a cell's failover, at 1,250, 2,500 and 5,000 nodes

Committed before the run. The review (`docs/reviews/2026-10-14-control-plane-at-fleet-scale.md`,
S7) asks it because one coordinator holds a cell: while its successor takes over, no engine in
the cell can change.

## What runs

The cell harness at this commit. After its other phases, the coordinator stops serving. A
successor takes the same store under the next term, acts under it, and listens at the same
address. The fake daemons see their links end and link again, as daemons do (a jittered wait from
200 ms, doubling to 5 s after a refusal), and say hello with what they hold now: the epoch of each
commit, and the ranks SHRINK took. The harness records `relinked_s`: from the stop until every
daemon is linked to the successor and every engine's view holds as many members as the
predecessor's last did. Then `first_change_s`: the successor's first change. Same host, one size
at a time:

```text
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture --exact cell_1250
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture --exact cell_2500
cargo test -p everserve-controller --release --test cell -- --ignored --nocapture --exact cell_5000
```

This measures the coordinator's side only. A lease's expiry before a standby may take it
(Kubernetes Lease or etcd TTL) adds its own time on top, which is configured, not measured here.

## Predictions

1. At every size, every daemon relinks, every engine's view comes back whole, and the
   successor's first change commits.
2. `relinked_s` is under 10 s at 5,000 nodes.
3. `first_change_s` is under 50 ms at every size.

## What would change the plan

If 2 fails, find where relinking spends its time before anything else. That is the case for S5,
sharding a cell's engines across coordinators so a failover takes only a share of them. If 1
fails, it is a defect in resume and comes first.
