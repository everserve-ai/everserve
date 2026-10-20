# Landing 5 on the single-node path, 2026-10-07

A regression check of `dfcddfd` (the pipeline pieces: member ids are the engine's ranks, slot
rows are columns, the collective is built from a layout) on the path every earlier run used: one
node, TP4 and a spare, Qwen3-1.7B, on 5x RTX PRO 4500 Blackwell (`knoya0tw6d7mps`, $3.60/h,
13:18-13:28, about $0.60). `run.log` has the commands; raw outputs in `data/` (not in git).

- Kernel tests: 14 passed on Blackwell (sm_120), the first build there; the peer-to-peer case
  skipped without NVLink.
- `adapter_1`: the replacement set `0-1-R2-3` formed on ranks 0, 1, 3 and the spare in 0.21 s;
  no lost-rank event, no watchdog report; 891 tok/s.
- `replace_mid_1`: detected by the slot in 48 ms, pause 0.671 s, 8/8 sessions identical,
  every exactness check passing, 56/56 tables, epochs matched; the region's order after the
  change [0, 1, 4, 3] (the spare's row 4 in column 2); 0 step RPCs in flight at the end.

The single-node path behaves as before the landing. The pipeline family itself has no GPU
test yet: that needs two stages, which the two-node run's platform decides.
