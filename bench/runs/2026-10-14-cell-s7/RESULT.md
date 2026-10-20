# S7 result: a cell fails over in under half a second; its first change after waits

Pre-registration: `PREREGISTRATION.md` (f978535). Numbers: `results.json`. Same host as S1.

## First, a defect

The first run, at f978535, never relinked: 0 of 1,250 daemons in 10 minutes, after 206,165
refused connects. The predecessor's listener was dropped after the successor had bound the same
path. Its `Drop` removed the path unconditionally, so it deleted the successor's socket. Any
daemon or coordinator started again before the old one exits would meet the same. Fixed in
fea4d8c: a listener removes its socket only while it is still the inode it bound. The runs below
are on fea4d8c.

## The runs

| | 1,250 nodes | 2,500 nodes | 5,000 nodes |
|---|---|---|---|
| every daemon relinked, every engine whole | 0.31 s | 0.32 s | 0.44 s |
| the successor's first change | 2.7 ms | 93 ms | 352 ms |

This is the coordinator's side only. A lease's expiry before a standby may take it adds its
configured time on top.

## Against the predictions

1. **Held:** at every size, every daemon relinked, every engine's view came back whole, and the
   successor's first change committed.
2. **Held:** relinking took 0.44 s at 5,000 nodes; the bound was 10 s.
3. **Failed:** the first change took 93 ms at 2,500 nodes and 352 ms at 5,000; the bound was
   50 ms. At 1,250 it took 2.7 ms.

## What it means

A cell's failover is not a reason to shard a cell (S5): every engine is changeable again within
half a second at 40,000 GPUs, plus the lease's own expiry. The first change after it is slower
at larger sizes. One likely cause is the successor still working through what the hellos set off
(plans, replication targets, status writes) when the change arrives, but this run does not show
it. That is the next thing to find, before a bound is stated for it.

## Addendum: what the first change waits on (not pre-registered)

At 5,000 nodes, the harness also times a change on another engine two seconds after the cell
relinked: 3.6 ms, as before the failover, against 380 ms for the first change in the same run.
So the first change waits on the successor finishing what 5,000 hellos set off, not on anything
that lasts. In practice, a 5,000-node cell changes at full speed about 0.8 s after its
coordinator stops, plus the lease's expiry. That costs one slower change per failover, so it is
recorded and not fixed now.

## Correction to the defect's fix

fea4d8c's inode check held on this Mac, where the runs above were made, but not on Linux. CI's
80-node cell failed over with 0 of 80 daemons relinked: Linux reused the freed inode number for
the successor's socket, so the predecessor still took the path for its own. No inode check can
tell the two files apart. Since the next commit, a listener, the daemon and the store leave their
socket's path when they stop, and the next bind replaces a leftover. The figures above do not
depend on which fix is in, since both let the successor's socket stand on this host.
