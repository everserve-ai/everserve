# Decision 0023's executor side on the single-node path, 2026-10-07

The check of `61b8870` (the executor follows every change the daemon forwards, and refuses a
local join for a loss that is not its spare's) on the path every earlier run used: one node, TP4
and a spare, Qwen3-1.7B, 5× RTX PRO 4500 Blackwell at $3.60/h. Two attempts, about $0.84 in
all; `run.log` has every command, raw outputs in `data-attempt1/` and `data/` (not in git).

**The first attempt found a bug the new timing exposed.** The watchdogs' reports reached the
coordinator, which claimed the change and forwarded it; the executor's follower marked rank 2
lost from the coordinator's word before the executor's own wait had read the slot's mark, which
also skipped that path's kill of the process. The change completed and cleared the dead set.
Then the process sentinel fired, found rank 2 not marked, marked it again and reported it at the
new epoch, which the coordinator could not fold (it tried a second change and refused it as
REBUILD); the drill's next RPC raised. The monitor's wait of up to a second for the exit code is
the window, and a slow teardown opens it wider.

**The fix** (`01a1576`): a sentinel from a monitor generation before the current one records the
exit and reports at the epoch the rank was lost, so the coordinator folds it, and never marks the
rank again; the slot path fences a marked rank whichever signal marked it first.

**The second attempt passed:** detected by the coordinator's word in 34 ms (the forwarded claim
beat the executor's own read of the mark); the sentinel at 0.206 s, reported at epoch 0, folded
by the coordinator together with the watchdogs' two later reports; one change committed; pause
0.667 s; 8/8 sessions identical, every exactness check passing, 56/56 tables, epochs matched;
the no-fault run reported nothing.

So on one node three signals now race for the same death, and any order gives one change: the
slot's mark, the coordinator's forwarded claim, and the sentinel.
