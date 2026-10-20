# drills

A drill is a failure mode with its PASS condition stated before it runs. The drills are the
pre-registered runs under `bench/runs/`, each with its `PREREGISTRATION.md` (the predictions),
`RESULT.md` and record; `bench/runs/README.md` is the index. The failure modes they cover:

- a worker killed between steps or mid-step, and a notice, on one node, every session
  token-identical (`bench/runs/2026-10-06-exact-replay-4gpu`);
- a node of a pipeline-parallel engine deleted under load and its rank taken by a spare
  (`bench/runs/2026-10-14-dynamo-proof`); the same rank taken twice, the second time from the
  spare that took it (`2026-10-14-dynamo-proof-pipe-leave`, `2026-10-14-dynamo-proof-two-losses`);
- a node crashed without notice, twice (`2026-10-15-dynamo-proof-crash-3`), and a deletion
  then a crash with the spare built from the core's replayed start-up calls
  (`2026-10-19-dynamo-proof-known-gap`);
- the operator's drains, cordons and a coordinator restart mid-drain
  (`bench/runs/2026-10-13-run3-drains-smoke`);
- the control plane at cell size and its failover (`bench/runs/2026-10-14-cell-*`).

Not yet drilled: a hung or slow rank, and a reclaim during a rejoin.

`scripts/test.sh engine <drill>` points here: GPU-backed tests are runs, not unit tests, and the
script refuses to run anything without a pre-registration.
