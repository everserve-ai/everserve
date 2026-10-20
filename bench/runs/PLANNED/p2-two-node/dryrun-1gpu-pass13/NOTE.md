# Battery dry run, pass 13 (2026-10-09 16:44–17:22 UTC): the resident shape on the pass-12 fixes

Three fresh single-GPU L40S VMs on Nebius eu-north1, the shape of pass 12: node-a rank 0 and
node-b rank 1 serving Qwen3-1.7B at TP1 x PP2, node-c holding the whole model pinned in host
memory with one spare standing for any position. Launched on the user's word ("i loaded up 20
bucks, lets get these fixes in and run the smoke when its ready"), main `634688b`: the worker
built at the take, the staging moving no fence, both ranks replicating again after a take (the
fence per rank, the resend, the taken rank's steps continuing), the take stamped term by term,
native stacks only for a stalled change, `backup_coverage` scored. Driver `battery-dry10.sh`; all
three VMs placed within 45 s this time.

## Result

| arm | sessions identical | restored rows mismatched | kill→resume | gap |
|---|---|---|---|---|
| stock restart | 0/8 | n/a | n/a | 378.3 s to the first token (312.5 s to the relaunch: vLLM's RPC timeout) |
| node fault | 8/8, every token | 0 (coverage 1.0, 1.57M rows) | 60.04 s | 60.13 s |
| single death | 8/8, every token | 0 (coverage 1.0, 1.92M rows) | 21.57 s | 21.68 s |

Exact on both arms again; the first steps after the resume 19 / 21 ms, then 13 / 14 ms, then
steady (pass 12's 0.5–0.6 s second step is gone). The join answer reached the executor in 0.70 and
0.87 s with no debugger attached (pass 12's 3.6 s was the harness's gdb), and the harness took
no native stacks on either fault arm ("committed at epoch 1; no native stacks taken").

## The take, term by term

| term (spare's stamps) | node fault | single death |
|---|---|---|
| the take, total | 59.08 | 20.41 |
| the worker built as the rank | 0.23 | 0.23 |
| the model's modules built (`shard_copy_s` less `weights_s`) | 4.29 | 1.88 |
| the weights from the host copy (`weights_s`) | 0.09 | 0.10 |
| the KV configuration (`kv_config_s`: vLLM's profiling run) | 3.73 | 3.25 |
| the KV allocation | 0.002 | 0.002 |
| the warm-up and graph capture (`warm_s`) | 2.94 | 14.81 |
| unstamped, before the worker's build | about 47.8 | about 0.1 |

- **The weights are not the cost.** The resident loader moves the shard in 0.1 s; a bulk copy is
  not worth building. The terms worth cutting are the modules' construction, the profiling run
  and the warm-up.
- **47.8 s before the worker's build on node fault (open).** Nothing logged between the take's
  start (17:08:12) and the worker's build (17:08:59). The step there is the collective taking
  rank 1's identity and making its slot, which started the releases of rank 0's position's
  groups (`ncclCommAbort` on a communicator whose peer, node-b, was dead): the likely hold on the
  device. A probe for the spare's stacks missed (it looked the process up by name, and vLLM
  retitles it; the daemon that supervises the spare is where its process should be reported),
  so the cause is not proven; the releases now start only after the spare has joined, and the
  step is stamped (`identity and slot`).
- **The warm-up's 14.8 s on single death.** vLLM's `warmup_kernels` ran 12.5 s in this spare
  process (Triton compiles of the V2 runner's kernels) against 0.15 s in node fault's; those
  kernels do not persist in Triton's disk cache on these VMs (pass 10), so a fresh process can
  recompile them; why node fault's did not is open.

## The backup after a take

The battery's check sampled tables by layer, but the store keeps one table per KV head slot
(8 per rank, every layer's row in each): "rank 0 of rig has no table 8 here", so the scored
figure is missing on both arms (fixed for the next pass). Run by hand against the live stores
during single death (every table, every 16th slot), 2 s before the spare logged its take done:

| rank | holder (epoch, step) | replica (epoch, step) | rows compared | coverage |
|---|---|---|---|---|
| 0 | node-a (1, 3292) | node-c (1, 3587) | 8,640 | 1.0 |
| 1 | node-c (1, 3585) | node-a (1, 3585) | 9,216 | 1.0 |

Both ranks replicate at the new epoch with every sampled row identical; the sample predates the
spare's own commits after the take, so the claim after the take is not yet scored.

## Clocks

No offset again: every probe refused. The harness's restart of the probe's server killed its own
shell (the pkill pattern matched the remote command line, which carried the server's launch), so
no server ran; fixed by passing the subcommand through a variable. Single-clock figures stand.

## Cost

Deployed 16:44:44Z, released 17:21:43Z: 0.62 h x 3 x $1.55/h = $2.87, of the user's $20.
