# The commit-path battery, 2026-10-06: stopped at its second run

Pre-registered in `PREREGISTRATION.md` (tree `a0b3378`). One 5× H100 SXM pod (`p267lcynio2uiu`,
secure, IN, $17.45/h). Commands and timestamps in `run.log`; outputs in `data/` (not in git).

- **Setup** took 17 minutes against an estimate of 13; the CUDA toolkit install is slow on this
  host. On the owner's word the watchdog was moved from the budget (75 min) to the estimate plus
  12 minutes.
- **`stock_1`** completed: 464.75 tok/s over steps 10–200.
- **`adapter_1` failed at start.** Rank 0 crashed forming the shadow groups:
  `groups.py:171 form_shadow_groups`, then vLLM's `StatelessProcessGroup.create`, then
  `create_tcp_store`, then `OSError: [Errno 9] Bad file descriptor`.
  - The crash came about 6 minutes into formation, close to the 300 s store timeout. That reads
    as a member that never reached the rendezvous; the cause is not established.
  - This is the first GPU run of `56c5d41` (membership changes as sets of ranks), which changed
    the sets and the spare's member id (`R{rank}`). The spare does receive
    `EVERSERVE_SPARE_RANK`.
  - The worker and spare logs are in `data/out/adapter_1.log`.
- **Every remaining arm starts an adapter engine,** so the battery was stopped and the pod
  terminated. No prediction is scored.

## Spend

Pod `p267lcynio2uiu`, 11:07:50Z–11:41:27Z (33.6 min) at $17.45/h: **$9.78** (cap $30).
