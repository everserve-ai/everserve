# The node-role and 0024 smoke on the tunnel: no run yet, 2026-10-07

The cheap smoke that was to prove the control paths before the cluster rental: the daemons' spares
(decision 0024), the node role on the headless node (decision 0022), a node lost (decision 0023),
over RunPod's pod-to-pod tunnel. Five launches today; none reached an arm with results. The raw
log is `run.log`; the first three-node run's outputs are in `data/` (not in git).

## What was asked for, and what RunPod had

| Launch (UTC) | Shape | Outcome |
|---|---|---|
| 15:15, 15:37 (tree 7f0c599) | node-a 3 GPUs (rank 0, spares for ranks 0 and 1), node-b 1 GPU | no 3-GPU networked pod in EU-SE-1 (20 asks over 20 min) nor in ten other data centers |
| 16:19, 16:33 (tree 5b4f21e) | node-a 2 GPUs (rank 0, the spare for rank 1), node-b 1 GPU | no 2-GPU networked pod in any of RunPod's 52 data centers, nine cheap card types plus the RTX 3090; each ask answered "no instances currently available" |
| 16:56 (tree 1e97313) | three 1-GPU nodes: a (rank 0), b (rank 1 under the node role), c (a daemon and the spare it launches) | pods in EU-SE-1 (A6000s), tunnel ok in 5 s; a harness fault (`nodes.sh` not executable) ended it at once |
| 16:58 (tree 1e97313) | the same | pods in EU-SE-1, tunnel ok, setup done on three nodes in 2 min, **the control plane up on three nodes over TLS** (coordinator on a, daemons linked on a, b, c, records exchanged), **node-b's half started under the node role**; then two refusals in vLLM's configuration (below) ended the adapter arm before the engine loaded |
| 17:26 (tree f93214e, the refusals fixed) | the same | EU-SE-1 had no second pod any more; EU-CZ-1 and US-GA-2 gave one pod each and no second; CA-MTL-1 gave three and its tunnel passed nothing for four minutes, the third dead tunnel there today |

Spend across the five: $0.56 (balance 4.92 → 4.36). Nothing is left running.

## What the one run that got past setup showed

- Three-node control plane over the tunnel: stores, daemons and the coordinator over TLS on
  three hosts, each daemon linked, every node's discovery record on every other node, in under a
  minute from `control`. The daemon on node-c launched its spare from the policy's
  `spareLaunch` at once (and relaunched it every 30 s after each refusal below).
- `python -m everserve_vllm.node_main` on node-b started and reached vLLM's configuration; the
  head died 24 s later, before node-b's workers could join the world, so nothing more is known
  about the node role from this run.

## Two refusals vLLM's configuration makes at the pin, fixed in f93214e

1. **The spare on a 1-GPU node.** `ParallelConfig` with no backend named and a world larger than
   the node's GPU count raises "World size (2) is larger than the number of available GPUs (1) in
   this node". The single-host 0024 shape (three GPUs on node-a) never met it. The shared engine
   configuration (`engine_args.py`, used by the daemon's spare and the node role) now names the
   `mp` backend, which is the engine's; vLLM then counts no GPUs.
2. **The head under `--nnodes 2`.** A multi-node world is allowed only under the `mp` backend, and
   the pipeline-parallel check reads `supports_pp` off the backend before a class path is
   resolved, so the drill's `--distributed-executor-backend everserve_vllm.executor.EverserveExecutor`
   is refused with "Pipeline Parallelism without Ray distributed executor or multiprocessing
   executor ... is not supported". The drill now builds the configuration under `mp` and sets the
   executor class on it before the engine is built (`LLMEngine.from_vllm_config`), which the
   engine honours as it is. Earlier single-node pipeline runs passed the path because `nnodes`
   was 1 there.

## Where this leaves the plan

The control paths this smoke was to prove remain unproven on real hosts: the carrier join end to
end, the node executor's sentinel into the coordinator and the head's follower, the group moved
to a spare node, and the node loss. The harness is ready for three single-GPU nodes
(`nodes.sh` with `NODES="a b c"`, `policies/three-node-tp1pp2-managed.json`), and the smoke
script reruns unchanged when RunPod has three networked 1-GPU pods in a data center whose tunnel
passes traffic: today that was EU-SE-1 for about half an hour. The platform for the rental is the
owner's call; `bench/runs/PLANNED/p2-two-node/` and `docs/MULTINODE.md` carry the plan.

# On Nebius AI Cloud, 2026-10-07 evening: the smoke ran, and what it found

Three single-GPU L40S VMs in eu-north1 on the project's VPC (`bench/harness/nebius.py`): node-a
(rank 0, the head), node-b (rank 1 under the node role), node-c (a daemon and the spare it
launches for rank 1). Qwen3-1.7B, TP1 x PP2, prompts of 2048 tokens (512 for the loss arms), 8
sessions. Eight launches between 18:30 and 23:45 UTC; the arms with results ran on the tree of
`9695149` (the commits of the evening, each named below). Raw logs in `data-nebius/`, the run log
in `run.log`. Spend about $35 across the evening, most of it setup and holds; nothing left running.

## What ran green

- **The control arm.** The engine served across node-a and node-b, node-b's half under our
  executor's node role (`everserve_vllm.node_main`), the pipeline traffic over our exchange
  (`everserve.comm.exchange`, `everserve_vllm.pipeline`), the mirror on tmpfs: 8 sessions of
  2048-token prompts, 1309–1328 tok/s over the VPC at decode (internal figure: the control arm
  measures nothing the product quotes). Rank 0 pre-formed its pipeline group with node-c's spare
  across hosts in 0.4–14 s (the long ones waited for the spare to come up).
- **Decision 0024's launch path on real hosts.** Node-c's daemon launched the spare from the
  policy at once; the spare created the node's slot region (the first process of the engine on a
  spare node), loaded as pipeline stage 1, skipped vLLM's warm-ups, pre-formed with rank 0,
  registered warm. The readiness view showed it, node-b as reclaimable with outcome `replace`,
  and the per-node `slots`.
- **Detection to commit across three hosts, without the head.** Node-b's worker killed by the
  harness: the node role reported it; the coordinator claimed and committed the change for rank
  1; node-c's spare staged the new membership `0-R1` at epoch 1. Node-b's daemon and worker
  killed together: the coordinator lost the daemon's link and claimed the change.

## The per-signal numbers (smoke figures, not battery figures)

| Signal | Predicted | Measured (three attempts) |
|---|---|---|
| worker SIGKILLed on node-b → the node role's report at its daemon | 30–80 ms | **1.0–1.8 s** (the sentinel fires at the process's exit, and a CUDA process takes that long to finish exiting; the slot watch's `/proc` state would see it in 10 ms, but a node of one member has no other process to watch it) |
| report → the coordinator's claim | — | **1 ms** |
| claim → node-c's spare has the membership staged | — | **1–41 ms** |
| node-b's daemon and worker killed → the coordinator's "daemon lost" | 1.5–2.0 s (heartbeats) | **37 ms** (the link's close; the heartbeat never needed) |
| the head's `RankLost` on the coordinator's word | 30–80 ms | seen once (attempt 2 of the worker kill), time not recorded: the drill writes its events only on success |
| pause, exactness | 1.5–4 s, 8/8 | **not reached**: no attempt completed its change |

## What stopped every loss arm

1. **The head's surviving worker (rank 0) died 82–86 s after the loss, with the change waiting
   on its boundary** (two attempts: `exit code -11`). It was in the pipeline's traffic with the
   dead peer; `9695149` makes the exchange refuse an aborted family group instead of calling into
   its freed communicator, and the crash remained, so the cause is elsewhere in that window — the
   stack dump the drill can take (`--stacks-after`) fired on nothing, because it keys on the
   drill's own kill and not the harness's marker; that is the first thing to fix for the next
   attempt.
2. **The head never received the forwarded change in the first attempt**: the coordinator
   logged "daemon node-a lost (link closed)" 32 ms after node-b's report, and the daemon's log
   of that moment was overwritten by the next launch (logs append now). Node-a's daemon and the
   coordinator share the host. **Root-caused by the controller's owner from these logs the
   same night:** node-a's daemon at 23:33:30 logged "coordinator link ended: frame of 169019393
   bytes exceeds 16777216" (0x0a130801: protobuf payload bytes read as a length prefix), and
   node-c's the same with 716309514 bytes on nearly every link at each launch. The daemon's
   link loop read frames inside a `tokio::select!` against the heartbeat tick and the outbound
   queue, and `wire::read_frame` is not cancel-safe: when another branch won mid-frame the
   bytes read so far were lost and the next read took payload for a length, so the link was
   closed as too large. On node-a the change's Deliver crossed a tick; on node-c the
   LaunchSpare crossed the spare's first report. The same shape sits in the coordinator's
   daemon link and the daemon's worker links, so the head's unrecorded RankLost may be this
   too. The fix (a cancel-safe buffered `FrameReader` in the three select loops, with a test
   reproducing the exact failure) lands on main separately; run 9 goes on a tree that has it.
   The signature in a daemon log: every link dying at the same millisecond offset, ten
   reconnects in two seconds. (The third attempt, in which the head also never saw the change,
   is void: a terminate of node-a issued by the stopped script was still executing underneath
   it, and node-a went away during the arm; Nebius deletes a VM minutes after the call returns.)

## Fixed on the way (all on main)

The spare-node region (`27e63bb`); `python3-dev` and `build-essential` for Triton's JIT, and the
mirror's data directory on tmpfs (`eff6d73`); a spare under pipeline parallelism (`d389ef0`: the
exchange, the broadcast, the stage-aware stub); the KV configurations by rank and the TP1
collective at bootstrap (`836f939`); the warm-up skip, the TP-group-of-one stand-ins
(`ca6e039`); the drill's armed marker, node-b's probe path (`eff6d73`); the refusal and the node
role's sweep (`9695149`). Harness lessons: Nebius reuses public addresses (throwaway host keys);
never edit a running bash script; a loss arm with `changes: 0` must fail (assertion to add); the
kill must land while the engine serves (the armed marker; 512-token prompts for the output room).

## Next

1. The stack dump keyed on the harness's kill marker; then the survivor's 85 s window.
2. Run 9 on a tree with the wire's cancel-safe frame reader (the owner of the daemon's fix).
3. The node role watching its workers' `/proc` state, so a worker's loss on a node of one is
   seen in tens of milliseconds, not at the process's exit.
4. The smoke's assertion: a loss arm passes only with a committed change, a completed resume and
   the head's `RankLost`/`node_lost` event.

## Run 9 (2026-10-08 00:19–00:45 UTC, tree `3581439` + the wire fix `9584ce1` + the spare-node row fix)

With the daemon's cancel-safe frame reader, **the head received the coordinator's word on every
attempt** (no "frame … exceeds" line in any daemon log: the fix's signature holds). Three
attempts of the worker-kill arm on the same VMs, each further than the last:

1. Kill → node-b's report **1.69 s** → claim **1 ms** → node-c's spare staged **41 ms** → the
   coordinator's commit **1.58 s** after the claim → the head's join **routed to node-c 68 ms**
   after the commit (`rig: join for rank 1 from node-a routed to node-c`) — decision 0024's
   carrier, end to end, for the first time. The spare never answered: its stack (py-spy) sat in
   `take_position → apply_change → set_members → _open → Region.handle`, waiting for the slot
   buffer of row 1 — the engine-node convention mapped the replacement to the "spare row"
   (`tp`), while on a spare node I had given the spare its column (0) as its row. Fixed: a
   replacement's row is its column on a spare node (`VllmCollective.spare_rows_at_columns`,
   set by `spare_main` where it created the region), with a test.
2. With that fix the spare waited correctly at its join queue, and the head never got the
   commit: the survivor's worker (rank 0) logged NCCL's "socketProgress: Connection closed by
   remote peer 10.0.0.2" 1.7 s after the kill, then **segfaulted inside NCCL's proxy thread**
   (`proxy.cc:506 ncclLocalOpAppend`) 82 s later, and the coordinator reported "rank 0 failed
   to apply the change: connection closed": the commit waited on rank 0's acknowledgement of
   the change, which its runtime gives after aborting the groups (`_abort_all`, each bounded by
   `ABORT_BOUND_S` = 10 s), and that acknowledgement never came. The same 82–86 s death was
   seen in run 8; the refusal of the aborted family group did not touch it, and now the place
   is known: NCCL's proxy, with the pipeline communicator aborted while its socket to the dead
   peer is still in its own timeout.

So the carrier, the three daemons, the commit and the join are proven on real hosts up to the
spare taking the position, and the one blocker left for a completed cross-node replacement is
on the survivor's side: how the pipeline communicator with a dead peer is released. The abort
that works for the tensor-parallel group (decision 0021's REPLACE runs on one node) kills the
process when the peer is a dead *socket*. Candidates, for the design discussion with the
controller's owner before the next run: not aborting the pipeline communicator at all and
letting the family swap at the boundary with the old one left untouched; or aborting it only
after the survivor's pending NCCL work has been drained; or NCCL's own socket timeout tuned
down so the proxy fails fast and cleanly. The survivors' stacks during the window are the
first thing to read (`--stacks-after` keyed on the kill marker, `3581439`; the dump fired on
nothing this run because the drill's step loop had already left for the change — to move the
timer's start to the executor's `rank_lost` event).

Spend for run 9 about $4; nothing left running.

## Run 10 (2026-10-08 01:22–02:4x UTC, tree `b20685d` → `00b5cfb` + fixes): the first completed cross-node replacements

Three L40S VMs again (node-a rank 0 and the head, node-b rank 1 under the node role, node-c
the daemon-owned spare for rank 1), decision 0025 on main, the same arms. The VMs were held
between attempts (`retry_wait`) and the fixes went out from the working tree, so one deploy
served seven attempts of the worker-kill arm and one of the node-loss arm. The watchdogs were
moved from 75 min to 3 h at the owner's word to keep going until green.

**Attempts of `replace_worker_1`, each one step further** (times on node-b's clock from the
third attempt on, see the clock correction below):

1. 0025's path worked at once: node-b reported the loss, the commit followed in 81 ms (run 9:
   82 s and a segfault), the join reached node-c 100 ms later. The spare's join died in its
   `on_apply` hook: a tensor-parallel group of one is a `LocalStandIn` with no `.group`.
   Guarded (`spare._point_stub_at`).
2. The spare could not attach to the new ring: `join.encode` used `dataclasses.asdict`, which
   recursed vLLM's queue `Handle` into a dict. A shallow field dict now; the answer's
   response-queue handle took the same path.
3. The spare took rank 1's position in 3 ms; the drill's restore check then read `r["rank"]`
   where the carrier's record has `ranks`. Drill fixed; the carrier's record now carries
   `n_replayed`, `wait_commit_s`, `boundary_s`, `resume_s` like the local one.
4. The restore check compared the spare's rows; the drill's probe install then failed on the
   spare: node-c's daemon launched it without the harness `PYTHONPATH`. Policy fixed.
5. **The survivor segfaulted 2 s after the kill**, and the core dump named it (the drill turns
   cores on; gdb on node-a): thread 1, the step thread, in `ncclSend(count=66, int64, peer=1)`,
   the exchange's header, → `ncclEnqueueCheck` → `groupLaunch` → `uploadProxyOps` →
   `ncclLocalOpAppend` (proxy.cc:506); the communicator it passed had `rank=-1`, `nRanks=-1`,
   `destroyFlag=1`: the refused family communicator, already released by its 0025 release
   thread, with no abort still running. The step had been in flight at the kill; with the
   peer dead its sends backed up and the thread was inside NCCL's host-side queue when the
   release freed it. Run 9's "82 s later in the proxy thread" was the same free, seen from
   the other side.
6. With one transfer in flight per communicator (`everserve.comm.exchange.one_in_flight`: the
   next send/recv/broadcast waits on the previous transfer's CUDA event first) the survivor
   segfaulted again, in `ncclBroadcast(count=16, int32, root=1)`, the sampled-token broadcast,
   same comm state. The refusal check ran *before* the wait; the wait ended with the abort
   (the send to the dead stage never completes); the enqueue then went onto the freed comm.
7. **Complete.** The exchange reads the refusal after the wait, bound to the group the call
   found; and the controller owner's guard (`everserve.comm.guard`, `00b5cfb`) wraps the check
   and the enqueue in the communicator's shared side while `ncclCommAbort` takes the exclusive
   side, so a free waits for a call in progress (wait → shared → check → enqueue; the wait
   stays outside, since the abort is what ends it). `rc=0`, loss arm complete, 0 leaked.

`replace_node_1` (node-b's daemon and worker killed together) then completed on the same tree
at the first attempt.

**The per-signal numbers of the completed arms** (the drill's record; kill stamped on node-b):

| signal | worker killed | node lost |
|---|---|---|
| loss reported (node role's `/proc` watch; node: link closed) | +0.00 s | +0.052 s (coordinator) |
| membership committed | +0.137 s | +0.096 s |
| spare's join answered (`join_s`) | 0.0028 s | 0.0031 s |
| survivor's boundary RPC: released | fabric 0.521 s, torch 1.027 s | fabric 0.331 s, torch 1.017 s |
| survivor's boundary RPC: leaked | 0 | 0 |
| survivor's boundary (`boundary_s`) | 0.50 s | 0.68 s |
| executor boundary (`boundary_s`, from the commit) | 0.94 s | 0.98 s |
| resume (`resume_s`) | 0.0085 s | 0.009 s |
| replaced (executor event) | +1.089 s | +1.083 s |
| pipeline steps refused on rank 0 while the change applied | 4 | — |

The torch abort (1.02 s) is the long pole of the boundary now; the fabric's abort is 0.3–0.5 s.

**Exactness in those two arms: not held, and why.** All 8 sessions matched the control for
730 (worker arm) / 781 (node arm) of 1792 tokens and diverged at the first token after the
resume. The policy has no replication target (`replicates_to: []`), so the spare restored 0
rows (`counts {}`), and `resume_running` resumed a request absent from the committed counts
at its *computed* length: the spare served its stage from empty rows (the drill's row check:
all 112 of the spare's tables mismatched, the survivor's 0). Fixed in `resume.py`: absent
means nothing committed, resume from zero, every position recomputed. Both arms rerun with it
below.

**Clock correction for runs 8 and 9.** The harness stamped the kill on the operator's machine
and then sent it over ssh; one round trip to a Nebius VM measures 1.7 s. Every "report
1.0–1.8 s after the kill" figure in runs 8 and 9 was that latency, not detection. From run
10's third attempt the kill is stamped on node-b's own clock, and the node role's `/proc`
watch reports at +0.00 s.

**Harness defects fixed on the way:** the drill's stack dumps were empty files (still to
fix); `collect` mirrors the nodes' `out` dirs instead of layering over stale local files; a
3 GB core in `out` makes `collect` take minutes (cores now moved aside on the VM); the hold
loop deletes a `retry` marker touched before it starts; a Monitor on `tail -f | grep` never
delivered a line, so the arms were watched with `until` loops instead.

**The rerun with the resume fix (`replace_worker_2`, `replace_node_2`, same VMs, tree
`00b5cfb` + the uncommitted fixes above): complete again, still not exact.** Both arms resumed
every request from zero (`resume_from 0`, `replay_positions` 1748 / 1387) and completed with
the same signature (commit +0.136 / +0.096 s, released fabric 0.52 s and torch 1.02 s, leaked
0, replaced +1.086 / +1.088 s). Per session against the control, identical prefix of 1792
tokens:

| arm | tree | sessions 0–7 identical prefix | tokens generated before the change |
|---|---|---|---|
| replace_worker_1 (attempt 7) | `00b5cfb` + gate, check, guard | 730 ×8 | 729 |
| replace_node_1 | same | 781 ×8 | 780 |
| replace_worker_2 | + resume from zero | 1236 ×8 | 1236 |
| replace_node_2 | + resume from zero | 875 ×8 | 875 |

So with the fix the sessions match the control for every token generated before the change and
diverge at the first token after it, in every session alike. That is the replay's shape, not a
bug in it: a full recompute of 1748 positions runs as a prefill (chunked at 4096 tokens), and
prefill kernels do not reproduce decode's numerics bit for bit, so the first greedy pick after
the change can differ. Decision 0013's exactness rests on the committed rows being restored and
only the short uncommitted tail replayed one position per step in decode shape, as the local
smokes did. Here nothing was restored because nothing was replicated: decision 0016's targets
are the policy's spare nodes that announced a replica address, the harness passes
`--mirror-replica` to every daemon and the policy names node-c, yet readiness reported
`replicates_to: []` for both ranks through the whole run. That was the one open item between
this run and an exact cross-node replacement, and the controller's owner found it the same
night: `refresh_replication` read only the policy's `spareNodes`, and the three-node policy
places its spare with `spares` alone (decision 0024), so every rank's targets were empty from
the first hello (node-c's replica address arrived fine). Their fix takes both fields (landing
behind this commit); with it rank 1's rows replicate to node-c, the spare restores them and
only the tail replays in decode shape, and the next run should read 1792/1792 on every
session. The spare's restore path, the join and the resume are proven.

`bench/harness/twonode/accept.py` now gives a loss arm its verdict (complete *and* exact against
the control), so a replaced-but-diverged arm never reads green again.

Trees: `b20685d` (0025, the run's start), `00b5cfb` (the guard), the fixes of this run committed
on top as one change. Spend for run 10: three L40S VMs 01:23–02:46 UTC, about $7; nothing left
running.

## Run 11 (2026-10-08 09:59–11:14 UTC, trees `eabf82b` → `d228022` + the spare's layout fix): the first exact cross-node replacements

The same three L40S VMs, held through every attempt (the owner's word: keep them up until green),
the controller owner landing store and coordinator fixes between attempts and the nodes rebuilt
in place. Every attempt completed its replacement; what each one found:

1. `eabf82b` (targets from both policy fields): readiness named node-c for rank 1 before the
   kill, the arm completed, the spare restored 0 rows. node-b's store: "rank 1 replica
   10.0.0.10:7402 link failed at step 0: frame of 235789572 bytes exceeds 16777216;
   reconnecting", every 250 ms for the run's life — the prefill burst as one frame against the
   wire's 16 MB cap. Store owner's fix `dd8ceb0`: a step crosses in parts under the cap.
2. `dd8ceb0`: the link held (no "exceeds" line), but the source dropped steps 64 behind
   ("steps behind; dropping step"), and the owner ruled a run with any such line unreadable for
   decision 0013: a dropped step's rows never reach the replica while the next step's counts
   do, so a restore serves stale slots as written. Also a harness miss: at 8 sessions the
   drill's generation (~20 s, slower with replication on the NIC) ended before the kill; 16
   sessions and 4000 KV blocks from here (store at 64000 slots, the spare's launch matched).
3. `254b595` (parts packed and pipelined) with 16 sessions: complete; the first cross-node
   restore ran — 607488 rows in 0.635 s, counts 1438 ×16, catch-up 12544 rows, 162 positions
   replayed in decode shape — and diverged at the first token after the change with the
   spare's restored rows mismatching the snapshot in every table; node-b's store had dropped
   1908 steps. The ruling, with data.
4. `d228022` (backpressure: a byte-bounded queue per target, the commit waits for room, no
   step dropped): no dropped step, `replicated_total` 218, `wait_seconds_total` 0.0004; complete,
   restore 548352 rows in 0.116 s; still diverged at the first token after the change. The row
   check named it: on the spare, head slots 4–7 of every layer mismatched and 0–3 matched, and
   the spare had registered with `heads: 4` where the engine's snapshot has 8. **The spare's
   stub tensor-parallel group spanned the whole world (2 = TP × PP) instead of its row (1), so
   under TP1 × PP2 the spare sharded attention for two ranks: half the KV heads, half the
   weights, a restore of half the slots.** Every cross-node divergence in runs 10 and 11 was
   this; the local TP2 smokes never saw it because there the world was the row. Fixed in
   `spare._install_stubs` (the TP and EP stubs over the stage's row), with a test on the layout.
5. **Green.** `replace_worker_3` and `replace_node_3`: complete and exact against the
   16-session control, identical prefix 1792/1792 on every session in both arms.

**The per-signal numbers of the exact arms** (kill stamped on node-b; 16 sessions, 512-token
prompts, Qwen3-1.7B TP1 × PP2 across node-a and node-b, the spare on node-c):

| signal | worker killed | node lost |
|---|---|---|
| loss reported to the head | +0.049 s | +0.031 s (link closed) |
| membership committed | +0.289 s | +0.258 s |
| spare's join answered | 0.041 s | 0.042 s |
| restore (rows / time, 8 head slots) | 1,073,408 / 0.196 s | 1,068,032 / 0.194 s |
| committed count at the restore (of 1646 computed) | 602 | 598 |
| catch-up from the watermark (rows / time) | 5,376 / 0.040 s | 3,584 / 0.041 s |
| survivor's releases (fabric / torch), leaked | 0.516 s / 1.019 s, 0 | 0.519 s / 1.023 s, 0 |
| executor boundary | 0.78 s | 0.80 s |
| resume | 0.013 s | 0.012 s |
| sessions resumed (replaced event) | +1.086 s | +1.070 s |
| positions replayed, one per step, decode shape | 884 | 870 |
| exactness vs control, 16 sessions | 1792/1792 ×16 | 1792/1792 ×16 |

The pause to the resume is ~1.1 s either way, the torch abort its long pole. The replay length
(~880 positions, ~20 s at this decode rate) is the replica's lag at the kill: replication ran
at ~3.5 steps/s against ~40 produced, which the store owner traced to a Nagle/delayed-ACK
stall on the TCP replica link (the frame's length and payload as two writes, request-reply);
their fix (one write per frame, `TCP_NODELAY`) is next, and with it the watermark should trail
by a step or two and the replay be the tail. Also theirs, landed: `fa49bdb`, a moved rank's
old holder told it replicates the rank nowhere (the dead worker's store had kept draining its
backlog at epoch 0 into a replica at epoch 1, refused every 40 ms).

**Drill note:** the row check (`kv_below_committed`) reports every table of the spare
mismatched even on an exact run, because it digests the snapshot's full slot set, above the
committed count too; it is not a restore verdict until it is scoped to the committed count.

Trees: `eabf82b` (the run's start), `dd8ceb0`, `254b595`, `d228022` (the store owner's), and
the spare's layout fix committed on top of `d228022` as this record's commit; `fa49bdb` landed
during the run, not on the nodes. The VMs are up at the end of this record, at the owner's
word, for the replay-length number on the `TCP_NODELAY` fix; spend so far for run 11: three
L40S VMs from 09:59 UTC, about $4.65 an hour.

**On the one-write frame and `TCP_NODELAY` (`ded0bfc`, the owner's word: take it, then release):
both arms green and exact again, and the replica now keeps up.** `replicated_total` on node-b
read 2142 at arming (218 on the stalled link); the restore found the replica at the engine's
step (watermark 2141 / 2103), restored 2.83 M / 2.80 M rows in 0.38 s with counts within a
step of the computed length (1582 / 1563 of ~1600), the catch-up had nothing to add, and the
replay was **1 position** in each arm, against 884 / 870 on the stalled link. The rest of the
signature held: loss reported +0.005 / +0.011 s, commit +0.39 / +0.40 s, join 7 ms, releases
0.52–0.53 s and 1.02 s with 0 leaked, boundary 0.64 s, resume 12–13 ms, sessions resumed at
+1.043 / +1.047 s, 1792/1792 on every one of 16 sessions. The restore is 0.19 s longer than on
the lagging replica for 2.6x the rows; the pause to the resume is 1.04 s either way, the torch
abort its long pole, and with the replay gone the time to the first new token after a loss is
that pause plus one step.

CI on `ded0bfc` first refused to start (GitHub's spending limit), re-ran green on all six jobs
after the owner's account change. VMs released at 11:42:59 UTC; run 11's spend: three L40S VMs
09:59–11:43 UTC, about $8. Nothing left running.
