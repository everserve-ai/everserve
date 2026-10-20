# Battery dry run, pass 14 (2026-10-09 17:37–18:53 UTC): the take's stall named, the backup scored

The resident shape of passes 12 and 13 on three fresh single-GPU L40S VMs on Nebius eu-north1
(node-a rank 0, node-b rank 1, Qwen3-1.7B at TP1 x PP2; node-c holding the whole model pinned
with one spare standing for any position). Main `1cd8c43`: the set-aside groups' releases
deferred until after the join, the take's identity-and-slot step stamped, `backup_coverage`
scored on the store's own table layout at the lesser of holder and replica step after the
sessions finish, the clock probe's server actually started. Within the user's $20.

## Result

| arm | sessions identical | restored rows mismatched | backup after the take | kill→resume | gap |
|---|---|---|---|---|---|
| stock restart | 0/8 | n/a | n/a | n/a | 378.8 s to the first token |
| node fault (cold spare node) | 8/8 | 0 (coverage 1.0, 1.67M rows) | 1.0 (rank 0), 1.0 (rank 1) | 73.62 s | 73.70 s |
| single death (warm spare node) | 8/8 | 0 (coverage 1.0, 1.61M rows) | 1.0 (rank 0), 1.0 (rank 1) | 8.21 s | 8.30 s |

`backup_coverage`: every KV head slot's table of each rank, every 16th slot, scored at the lesser
of holder and replica step (both at 3587 for rank 0 and 3585 for rank 1, read after the sessions
finished): 9,216 rows per rank, every one byte-identical at the replica. After a take, each rank's
rows are where a second loss would restore them from. The join answer reached the executor in
0.77 / 0.73 s; detection 12.5 / 7.8 ms; the first steps after the resume at the steady size.

Clocks, measured for the first time on this shape: node-b and node-c against node-a, -0.21 and
-0.21 ms at the start, -0.16 and -0.26 ms at the end, each ± 0.07 ms.

## The take, term by term

| term (spare's stamps) | node fault | single death |
|---|---|---|
| the take | 72.58 | 7.21 |
| identity and slot | 47.95 | 0.02 |
| the worker built as the rank | 0.23 | 0.22 |
| the modules (`shard_copy_s` less `weights_s`) | 4.31 | 1.83 |
| the weights from the host copy | 0.09 | 0.10 |
| vLLM's profiling run (`kv_config_s`) | 3.99 | 2.93 |
| the warm-up and graph capture | 15.89 | 1.99 |

**The 47.95 s was the kernels' build, not the releases.** With the releases deferred the
stall stayed, in the identity-and-slot step: the slot collective made at the take loads
`everserve_kernels` (`kernels.load()`), and torch built the extension there, its first use on
node-c. `everserve.kernels` says an install builds it ahead (`python -m everserve.kernels`); the
harness's setup never did, and on node-a and node-b the serving workers paid it at engine start,
unseen. Single death's take, with the build cached, spent 0.02 s on that step. Fixed: the install
builds them (`nodes.sh setup`, and the adapter's README names it a required install step), and a
standing spare loads them when it stands, so a node installed without the build shows it at
standing, never on a change.

**The 15.9 s was vLLM's Triton kernels compiling cold on node-c.** The take hit the staged AOT
artifact ("Directly load AOT compilation", `torch.compile took 0.89 s`); the runner's Triton
kernels compiled into `~/.triton/cache` (no `TRITON_CACHE_DIR` set), empty on the fresh node
before the take and 37 entries after; single death's warm-up on that cache took 1.99 s. Fixed:
`nodes.sh cache_sync` stages the serving node's `~/.triton/cache` beside the compile cache
(decision 0029's principle: compiled artifacts travel with the deploy). The load average on
node-c stayed under 1.5 of 8 vCPUs through both takes: not contention.

**What remains on a warm node: about 7 s,** the modules' construction (1.8 s), vLLM's profiling
run sizing the KV (2.9 s), the warm-up (2.0 s). The profiling run is the next cut (a recorded
size); the modules and the warm-up are the pre-attachment trade (designed, not built).

## The harness's own costs this pass

The clock probe's server, now actually started, held its ssh session open: the launch backgrounded
a `cd && nohup` list, whose subshell kept the session's streams while it waited on the server. The
battery waited 10 minutes at the start (sessions released by hand) and 31 at the end. Fixed: only
the redirected server is backgrounded; the old form held its caller until killed, the new returns
at once (checked locally).

## Cost

Deployed 17:36:57Z, released 18:53:00Z: 1.27 h x 3 x $1.55/h = $5.90 (about $3.10 of it the two
clock-probe holds). The user's $20: passes 13 and 14, $8.77.
