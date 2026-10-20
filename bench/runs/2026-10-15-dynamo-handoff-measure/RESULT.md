# Result: the handoff measurement, attempt 1 (2026-10-15): its precondition not met

The run did not meet its own precondition. Engine A had no standing spare when notice 1 was
sent, and the harness's wait for cover passed without waiting. So no rank moved, the core was
never retired, and measurement 3 (the retire under load) was not taken. The pre-registration
(sha256 `1f387cce…b8d5` at `2305e95`) stands for the rerun unchanged. This record gives what
notice 1 did yield, and the two defects that stopped it, with their fixes.

**The run.** Images `dev-2305e95`, 4 × H100 in eu-north1, with the pre-registration's command.
- The cluster was up at 23:40Z, both engines served at 23:57:54Z, and notice 1 was sent at
  23:57:59Z.
- The harness stopped at 00:01:07Z on "the head did not change at epoch 2", by design. Notice 2
  was never sent.
- Cost: the cluster from about 23:40Z to the end of its take-down (started 00:01:40Z), about
  35–40 minutes of 4 × H100, so about $10–12, plus the build VM (under $1).

## What notice 1 yielded

**1, cold start** (`handoff_read.py starts`, on the coordinator's clock):

| start | node | role | to the answer | to registration |
|---|---|---|---|---|
| bring-up | e00vetp (rank 1's) | head | 54.2 s | no line (a head's registration logs none) |
| notice 1's successor | e00jtck (rank 0's) | standby | 35.8 s | 35.8 s |

Two starts, not three, so the decisions resting on measurement 1 wait for the rerun. The bring-up
start overlapped the node's own model load. The successor's start ran beside a serving rank on
its node.

**2, footprint**: read on the successor's node (`proof-0-nodes-9tm5l`) before the notice and
once the successor stood:

| | GPU memory used | processes |
|---|---|---|
| before | 2473 MiB of 81559 | `VLLM::Worker_PP0`, 2464 MiB |
| after | 2473 MiB | the same, nothing new |

The core opened no GPU context. The bring-up core's own node (`4kj6w`) also lists its worker
alone. This is what the pre-registration's decision 2 reads as "below the headroom" (16311 MiB at
`--gpu-memory-utilization 0.8`): **no change to the renderer.** It rests on one read, the run's
only successor. The rerun's reads confirm or overturn it.

**3, retire under load**: not taken. 12 of the 32 streams were first served by A, and all 32
finished with no migration, because no core was retired. The `ends-1.json` summary calls A "the
retired engine", but it was not retired.

What the coordinator did with a head node's drain and no spare (its log):
- The successor started at once, which is step 0.
- At the deadline minus `headHandoffMs`, the move of rank 1 was refused: "policy chose REBUILD
  for rank 1", since no spare stood. The drain was blocked and retried 7 times.
- Each retry left under 30 s, so "the head stays, and its node's loss restarts the engine". The
  head stayed and the engine was never down.
- The peer's 5514551 now records this as HeadHandoffSkipped, naming the refusal.

## The two defects, and their fixes

1. **The spare for rank 0 failed its warm-up, so A never had a spare.** The third node pod's spare
   crashed in vLLM's kernel warm-up: `GroupCoordinator` has no `use_cpu_custom_send_recv`, read
   in `isend_tensor_dict`. The pod restarted, which is why the spare never stood.
   - The cause: a spare warmed up "alone" (sends dropped) only past the first pipeline stage. A
     first-stage spare's warm-up therefore sent its activations through vLLM's own send, on the
     stand-in group the spare builds without `__init__`, which this vLLM pin's send reads.
   - Earlier proofs never had a spare for rank 0, only for rank 1, the last stage, which sends
     nothing forward.
   - Fixed in `spare.py`: every stage of a multi-stage engine warms up alone, and only a
     one-stage engine warms up as vLLM runs it. Tested.
2. **The harness's wait for cover passed without waiting.** It read the core's node as holding
   no rank, so it drained at once.
   - The rank lookup gives the right answer on the saved record (rank 1 on e00vetp), so the read
     at 23:57:55Z must have differed. The record file is rewritten in place by every read, and
     the two reads of it were separate.
   - Fixed: the node and its ranks now come from one read (`proof_verdict.py head-ranks`). A
     record placing no rank on the core's node stops the run and names the members it read, as
     a core under `headOnRankNodes` always runs on a rank's node. Tested.

## Next

The rerun takes the same pre-registration, on images with both fixes. It needs Cort's word for
its spend, about $10 again.

# Attempt 2 (2026-10-16): no notice sent; the core started where no spare stood for it

Images `dev-7cd3150`, with both of attempt 1's fixes, the same command and the same
pre-registration. Cort approved the spend ("lets do it"). The record is `run-2/`.

**The spare fix held.** The spare for rank 0 stood on e00z48, and its replacement set
`R0@e00z48:0-1` was formed and covered. The first-stage warm-up no longer crashes.

**The harness's guard held.** It found the core's node (e00scy) and its rank (1), waited for
rank 1 to be covered, and stopped at its 900 s bound (01:59:39Z) without sending a notice: "no set
planned for the loss of rank 1". No notice went into a known refusal.

**The gap: the core started on a node whose rank no spare stood for, and Covered did not say so.**
The peer traced it in the coordinator, and fixed both parts.
- The core started on e00scy, rank 1's node, because rank 1's pod was told to serve first, and the
  bring-up start went to the first rank told to serve. The cell's one spare pod stood for rank 0,
  its slot's rank, so no spare could take rank 1.
- The engine's Covered condition left the head node's ranks out ("the head's node holds [1], whose
  loss restarts the engine"), which is decision 0038's rule from before the planned handoff. So it
  read True with rank 1 spareless. The planner itself never excluded the head node's ranks: it
  had no spare that could take rank 1.
- The fixes (the peer's, after attempt 2):
  - the bring-up start fires only when rank 0's process is told to serve, so the core lands on
    rank 0's node, as vLLM's and Dynamo's leader is;
  - under `headOnRankNodes`, Covered counts the head node's ranks ("which a notice moves before
    the head").
  Other shapes keep the old rule. Only an unannounced loss of the head's node restarts the
  engine.

**What attempt 2 yielded:**
- **Bring-up start**: 54.0 s from StartHead to the core told it holds the head (attempt 1: 54.2 s).
  As before, it overlapped the node's own model load.
- **The core's host memory, a manual read** beside the harness (`run-2/core-idle-head.txt`, the
  head core while serving, at 01:44:46Z):

  | process | RSS |
  |---|---|
  | `dynamo.vllm` launcher | 1.73 GB |
  | `VLLM::EngineCore` | 1.26 GB |
  | the core, together | about 3.0 GB |
  | the rank's worker | 2.97 GB |
  | `node_main` | 1.11 GB |

  The pod sets no memory limit (cgroup `max`). The node's allocatable is 194 GB and 15.9 CPU. The
  CPU figures in the read are averages since each process started, not idle CPU. No standby stood,
  so no idle read was taken.
- No successor started, no footprint read was taken before and after, and there was no retire.

**Cost**: the cluster from about 01:28Z to its take-down (started 02:00:11Z), about 35–40 minutes
of 4 × H100, so about $10–12, plus the build VM (under $1).

**Next**: attempt 3 runs on images with the peer's fixes, under the same pre-registration. It
needs Cort's word for its spend. With one spare pod, only notice 1 can be covered: after it, the
head is on rank 1's node, and the drained node's pod stands again for rank 0. So the third start
needs a second spare pod, a change to the cell recorded as an amendment if it is made.

# Attempt 3 (2026-10-16): stopped before notice 1, the harness read "serving" too early

Images `dev-679c49f`, with the peer's fixes (the core starts on rank 0's node, and Covered counts
the head node's ranks), the same command and the same pre-registration. Cort approved the spend
("ok go"). The record is `run-3/`.

**What happened.**
- Both engines' pods were up at 02:44:23Z.
- At 02:51:51Z the harness took engine A as serving and sent its warm-up, which completed.
- One second later, `notice_head` read A's record: state `starting`, `members []`, `worldSize` 0,
  and Serving False (NoMembers).
- The guard from attempt 1 stopped the run there ("head on e00eb6…, holding no rank; members []")
  rather than drain a node it could not read, and the cluster was taken down.

**The cause, a harness defect.** It also explains attempt 1.
- For an engine whose core runs on its own nodes, the harness took the head's registration in the
  record as "serving". The core registers as head before its ranks join it.
- The warm-up through the shared frontend was most likely answered by engine B, which serves the
  same model.
- So the harness went on while A was still bringing its ranks up. Attempt 1's "holds rank(s) none"
  was this same early read, not two reads straddling the core's registration, as attempt 1's
  section supposed.
- Fixed: the harness now waits for `proof_verdict.py serving`. It requires the Serving condition
  True, every position of the world held, and a head registered, and its tests use attempt 3's
  starting record. Applied to the saved records, attempt 3's reads not serving (0 of 0 held) and
  attempt 1's record at its notice reads serving (2 of 2).

**What attempt 3 yielded**: nothing measured. No notice was sent.

**Cost**: the cluster from about 02:35Z to its take-down (started 02:52:25Z), about 30 minutes of
4 × H100 with the take-down, so about $8–9, plus the build VM (under $1).

**For attempt 4**: the same pre-registration and cell, with `--losses 1`. With one spare node,
notice 2 cannot be covered: after notice 1 the head is on rank 1's node, and the only spare stands
for rank 0. The harness would wait its full 900 s for cover before stopping, which is GPU time
spent on nothing. The pre-registration already allows notice 2 not being sent, so this changes
neither the bar nor the cell, only the time spent. It needs Cort's word for its spend.

# Attempt 4 (2026-10-16): no notice sent, rank 0's spare refused on a checkpoint digest bug

Images `dev-51d1383`, with the serving wait and the replay tests, under amendment 1
(`--losses 1`). Cort approved the spend ("ok go head with 4"). The record is `run-4/`.

**Every harness read held.**
- A served 2 of 2 positions at 10:42:07Z.
- The core was on rank 0's node, e00njd, which confirms the peer's fix again.
- The harness waited for rank 0 to be covered, and stopped at its 900 s bound (10:57:17Z) without
  sending a notice.

**The blocker, a product bug in the adapter's identity (mine).**
- Rank 0's spare on e00q3n was refused under decision 0034: `checkpoint_digest is "b18fa9…" here,
  "984d63…" on the position`.
- In every attempt (attempt 1's rank 1, attempt 2's rank 1, attempt 4's rank 0), the rank on the
  core's node reports `984d63…`, and every other process reports `b18fa9…`.
- The pods' Hugging Face caches (`run-4/model-cache-by-pod.txt`, read live) show why:
  - On the nodes without the core, the weights blob `blobs/fdf756…` (the file's sha256) is a link
    into `hub/blobs/bb/bb5ff7…`. That is the Xet downloader's store, named by Xet's own hash.
  - On the core's node, the core's own `dynamo.vllm` download wrote plain blobs.
  - `identity.content_sha256` followed every link and took the final 64-hex name as the file's
    sha256. So the same weights digested two ways, and the node that differed is the core's, the
    one a spare must stand for under `headOnRankNodes`.
- **Fixed** (`identity.content_sha256`): one link only, `snapshot → ../../blobs/<sha256>`; any
  other layout is read byte by byte. Tested with the Xet double link and with layouts outside the
  snapshot shape, and mutation-checked.
- **Checked on the real downloads** before attempt 5. The model was fetched twice on a CPU, by the
  Xet downloader and by the plain one. The old code gives `b18fa9…` and `984d63…`, the two values
  the H100 nodes reported. The fixed code gives `984d63…`, the true digest, for both.

**Also found before attempt 5, on a CPU: a drained node cannot be returned to service.**
- `uncordon` is refused while the node's spec holds its drain ("a node being drained is cordoned;
  a drain needs unschedulable"), and `undrain` does not clear a drain that went through. That holds
  for 60 s past the deadline in the handoff end to end.
- The harness's uncordon after a notice would have exited before attempt 5's final reads. It now
  brings a node back only when another notice follows, and stops saying why if it cannot.
- The node lifecycle is the peer's to decide.

**What attempt 4 yielded**: no measurement. The bring-up start was on rank 0's node, as designed.

**Cost**: the cluster from about 10:22Z to its take-down (started 10:57:46Z), about 45 minutes of
4 × H100 with the take-down, so about $13, plus the build VM (under $1).

# Attempt 5 (2026-10-16): the handoff ran to the take, which failed on two bugs

Images `dev-bf15eea` (product code as main's `14edc1c`), under amendment 2. Cort approved the spend
("yeah start the next run right away"). The record is `run-5/`.

**What worked on GPUs, for the first time:**
- A served 2 of 2 positions with no Degraded MembersDiffer at bring-up, so the digest fix holds
  across members.
- Rank 0's spare was accepted, and its set `R0@e00mdk:0-1` formed.
- The notice went out at 11:46:31Z.
- The successor started on e00pdc, rank 1's node.
- The REPLACE of rank 0 committed at epoch 1.
- The successor was told to take the head at epoch 2, and its bring-up joined both ranks in
  0.052 s.

**What failed:**
1. **The take crashed** (`run-5/successor-pod.log`, the adapter's, mine).
   - Rank 1's worker, taking the successor's join, raised `AttributeError: 'GPUModelRunner' object
     has no attribute 'requests'` in `worker_loop.drop_requests`.
   - The engine runs vLLM's V2 model runner, and the drop was written for V1's. The CPU tests'
     stand-in runner was V1's.
   - The change failed at 11:50:02Z ("the standby did not take the head in time"), and the engine
     went down.
   - **Fixed:** the drop uses V2's own path (`_remove_request` over `req_states`, pooling first,
     sorted), and V1's for V1. A vLLM-gated test holds both to the pinned classes, and the
     end-to-end stand-in is V2's.
2. **The old core ended with its node, before its retire** (mine).
   - 8 ms after rank 0's move committed, the coordinator logged "its head on e00x2z stopped: ; its
     standby takes it", and no retire was sent.
   - `node_main` ends a node whose ranks have all left, and its end stops the core it started.
   - **Fixed:** a node outlives the core it started (`node_main.outlive_core`): it stays up until
     that core ends (its retire) or the node is told to stop. Deterministic tests with real child
     processes; the old line fails them.
   - The CPU handoff end to end does not tell the two apart: on one machine the retire reaches the
     core before its node sees its rank leave. This is stated, not hidden.

**The harness read the retired engine's streams wrong.**
- The core's pod restarted after the engine went down, so its Dynamo registration was only in the
  log before the restart. `app_logs` read current logs alone, so `ends-1.json` reads "0 of 32 on
  the retired engine".
- Read from the collected logs, previous ones included: 19 of the 32 streams were first served by
  A's core, all 19 ended in the frontend's `{"code": 503, ..., "type": "service_unavailable"}`, none
  migrated, and the 13 on B finished.
- The replay also found the reader crashing on an error that is an object, not a string.
- Both fixed, with a replay case on run-5's own logs.

**What attempt 5 yielded** (first reads, as the pre-registration has it; measurement 3 is not one,
since the retire never ran as designed):

| | attempt 5 |
|---|---|
| bring-up start (to the answer) | 52.2 s |
| successor start (to the answer / to registration) | 35.5 s / 35.5 s |
| rank 0's move (`headHandoffMs` before the deadline to its commit) | 0.51 s |
| successor's GPU, before and once it stood | 3175 MiB and 3175 MiB, nothing new: 0 MiB |
| standby core idle, host memory (`run-5/core-idle-standby*.txt`) | 2.67 GB RSS (launcher 1.56 GB + EngineCore 1.11 GB) |
| standby core idle, CPU | its CPU time unchanged over 18 s, about 0 |

**Cost**: the cluster from about 11:28Z to its take-down (started 11:50:10Z, done about 12:02Z),
about 35 minutes of 4 × H100, so about $10, plus the build VM (under $1).

# Attempt 6 (2026-10-16): the take failed again, on two paths only the real executor runs

Images `dev-6278690`, under amendment 3, with attempt 5's fixes and the peer's fail-fast. Cort
approved the spend ("yes"). The record is `run-6/`.

**What held:**
- A served with no MembersDiffer, and rank 0 was covered by `R0@e00kgq`.
- The notice went out at 12:52:31Z, and the successor stood on e00g5y at 12:53:08Z.
- The REPLACE of rank 0 committed at epoch 1 (12:54:01.745Z).
- The fail-fast ended the failed head change at once: 12:54:01.865Z, not 2 minutes on.

**What failed:**
1. **The old core stopped itself at the commit.**
   - The drained node's daemon logged "head of proof departed: Stopped:" at 12:54:01.754Z, 8 ms
     after that node's commit. It was a clean goodbye with no failure recorded, so the executor
     shut down with `failure` unset.
   - The node itself restarted only at 12:54:14Z, so `outlive_core` held, and it is not the node
     that ended the core this time.
   - Ruled out, by reading the code:
     - the daemon telling the head to leave (a head's link is a client, never in the daemon's
       displaced workers);
     - the head's client acting on a commit (it only queues it);
     - the change follower marking the departing rank lost (it skips a departing rank).
   - Left: the executor's own half of a REPLACE, run on a core holding no rank for the first
     time.
   - The core's own log is lost. It was in the drained pod's first container, gone by two
     restarts, because the harness's notice did not keep the core pod's log. **Fixed:** every node
     pod's log is now followed into the record from the notice on.
2. **The take was refused.**
   - The standby's `initialize_from_config` raised "the KV cache built differs from the one
     allocated: ranks [0]". Rank 0 is the spare that took it.
   - The block count and block size are pinned in every process's arguments, so it is not those
     alone, and the refusal named no field.
   - **Changed:** the refusal now names each field that differs, with both values
     (`worker_loop.kv_differences`, tested).

**What attempt 6 yielded** (first reads):

| | attempt 6 |
|---|---|
| bring-up start (to the answer) | 66.9 s |
| successor start (to the answer / to registration) | 36.6 s / 36.6 s |
| rank 0's move | 0.40 s |

No stream measurement: the engine went down at the take.

**Cost**: the cluster from about 12:34Z to about 13:08Z, about 35 minutes of 4 × H100, so about
$10, plus the build VM (under $1).

**Next, proposed to Cort**: no attempt 7 on this cell until both failures have run under the real
executor where every log survives. That means one multi-GPU VM running the cell's processes
directly (the core, both ranks and the spare, no Kubernetes or Dynamo), minutes per try. The
proof itself stays on Kubernetes with Dynamo.

## Attempt 7 (2026-10-16)

Images `dev-5ddeb0f`, under amendment 4, with attempt 6's two failures fixed: change delivery to the
head's node, and the spare's start-up replay. Cort approved the spend ("ok"). The record is `run-7/`.

**What held:**
- **The replay, its first run on GPUs.**
  - The spare asked for rank 0's calls through its daemon at 17:18:01.800Z. The core wrote 3 calls
    on a queue made for it 4 ms later: the layout, the memory profiling and the KV configuration,
    with no model-length change.
  - The spare ran them on its worker. Its watch's check at its start was answered the same way, at
    17:18:29.97Z.
  - The spare attached in 38.8 s and stood, and rank 0 was covered by `R0@e00n07`.
- A served, the notice went out at 17:18:36Z, and the successor stood on e00f61 at 17:19:11.66Z.
- The REPLACE of rank 0 committed at epoch 1, 0.54 s after `headHandoffMs` before the deadline.
- **The old core heard rank 0's move claimed**, so amendment 4's first fix held. It reached its
  half of the change (`complete_change`), which attempt 6 never did.
- The coordinator gave the head to the standby at epoch 2 at once, and both ranks answered its join.

**What failed:** the take, on two adapter defects. No control-plane frame was lost: every ask,
offer, claim and join in the record reached its process.
1. **The old core failed completing rank 0's move.**
   - The error was `AttributeError: 'NoneType' object has no attribute 'max_chunk_bytes'`, at
     `_complete_over_carrier`, 17:20:07.008Z.
   - The new ring's chunk size was read off the old ring's shared-memory buffer. A core holding no
     rank has a ring whose readers are all on other nodes, so it has no such buffer.
   - Nothing on CPU runs the executor's half of a change on a core of this shape. The head's end to
     end runs a stand-in core.
   - **Fixed:** the chunk size comes from vLLM's setting (`VLLM_MQ_MAX_CHUNK_BYTES_MB`), as vLLM's
     own executor sizes its ring (`executor.chunk_bytes`). A vLLM-gated contract test makes a ring
     with only remote readers and checks the executor reads no buffer.
2. **The spare refused the successor's take.**
   - Because of 1, rank 0's spare never joined the old core.
   - The successor's take offered rank 0 to that spare through the bring-up path. A core offers
     there before it has made its start-up calls, so the offer names no digest.
   - The spare's last guard, added in `5ddeb0f`, refused it: "rank 0's spare was built from other
     start-up calls than the core offering it the position (the core's none, ...)". The engine
     went down.
   - **Fixed:** the offer names what offers the position (`joins.Offer.kind`, join version 7): a
     change, a core's bring-up, or a standby's take. Only a bring-up or a take may come without a
     digest:
     - at a take, the standby compares each joined worker's allocated configuration with the one
       it builds, and refuses the head on a difference, naming the field;
     - at a bring-up, the core sends every joined process its calls after the join.
   - A change's offer without its digest is still refused. The absence of a digest grants nothing
     (the peer's review). The guard test sends a change's offer with its digest stripped, and the
     earlier rule, an absent digest allowed, fails it.

The 17 streams on A's engine ended in a 503 as it went down. None migrated, because the engine was
down, not retired.

**What attempt 7 yielded** (first reads):

| | attempt 7 |
|---|---|
| bring-up start (to the answer) | 52.8 s |
| successor start (to the answer / to standing) | 34.2 s / 34.2 s |
| rank 0's move | 0.54 s |
| the spare's replay (the ask to the calls written) | 4 ms |
| the spare's memory profiling, before it stood | 4.49 s |

The profiling ran before the spare stood: a spare built for its rank pays it there, and it adds
nothing to a move. The battery's 2.9 s estimate was low on these H100s.

**Cost**: the cluster from about 17:01Z to 17:34Z, about 33 minutes of 4 × H100, so about $10,
plus the build VM (under $1).

**Next**: attempt 8 on images with both fixes, under the same pre-registration. It needs Cort's
word.

## Attempt 8 (2026-10-16)

Images `dev-7001f7e`, under amendment 5, with attempt 7's failures fixed and covered on CPU. Cort
approved the spend ("ok"). The record is `run-8/`.

**What held, for the first time:**
- **Rank 0's move, completed by the old core.** The REPLACE committed at epoch 1, 0.44 s after
  `headHandoffMs` before the deadline. The old core completed its half: its change record shows
  the spare's epoch at 1 and the survivor's at 1. It then retired through Dynamo's graceful
  shutdown.
- **The head change.** The standby on e00kf3 was told to take the head at epoch 2. Both ranks
  joined it, rank 0 from the spare that had joined the old core, and the coordinator committed
  epoch 2 (Head, lost []) at 20:13:30.812Z, 5.36 s after rank 0's move.

**What failed:**
1. **The successor completed its own take as a change.**
   - 13 s after the take, the successor heard the commit of epoch 2, its own take. Its step hook
     passed it to `complete_change` as a change, which found no claim for epoch 2 and raised.
   - The engine went down: "completing the change at epoch 2: RuntimeError: no claim recorded
     for epoch 2".
   - **Fixed:** the commit of the core's own take is passed over, and only that commit
     (`EverserveExecutor.own_take`): its take's epoch, a head change naming this core's node, no
     rank lost. That holds in `pending_change` and in `complete_change`'s own wait.
   - Any other commit at that epoch is still completed as a change, and fails loudly. The peer
     reviewed an epoch-only rule as broad enough to hide the next defect.
   - The vLLM-gated test feeds a successor at epoch 2:
     - its own take's commit, which is passed over;
     - two other commits at epoch 2, which are both completed;
     - a change at epoch 3.
2. **The old core left without its goodbye.**
   - Dynamo's graceful shutdown ended the core's process after its 5 s grace, and its head
     connection closed with no goodbye. The coordinator read the head as lost, not retired, and
     gave it to the standby as it does on a loss.
   - The take went through either way. The goodbye on a retire is still owed: not fixed yet.

**First reading of the retire's stream migration** (amendment 3's flag).
- The old core's shutdown broke its 10 streams at 20:13:30.656Z.
- The frontend scheduled a migration retry for each, and each found "No workers available for
  endpoint" (`frontend-migration-1.txt`).
- The successor registers with Dynamo only once its take completes, which committed 0.16 s
  later, and B serves another endpoint.
- So with the retire as built, A's streams have no worker to move to while the head changes. This
  is a design question for measurement 3 and is put to Cort; it is not a defect of this run.

**What attempt 8 yielded** (first reads):

| | attempt 8 |
|---|---|
| bring-up start (to the answer) | 52.8 s |
| successor start (to the answer / to standing) | 34.6 s / 34.6 s |
| rank 0's move | 0.44 s |
| the head change (rank 0's move to the epoch 2 commit) | 5.36 s, 5 s of it Dynamo's grace |

**Cost**: the cluster from about 19:54Z to 20:28Z, about 34 minutes of 4 × H100, so about $10,
plus the build VM (under $1).
