# Amendments to the handoff measurement's pre-registration

`PREREGISTRATION.md` and its sha256 (`1f387cce…b8d5`, at `2305e95`) stay as they are. Each
amendment is written before the attempt it applies to.

## 1. Attempt 4 runs one notice (`--losses 1`), written 2026-10-16

**The change.** The run's command takes `--losses 1` in place of `--losses 2`. Everything else in
the command, the cell and the images' source stays the same.

**Why.**
- The cell has one spare pod. It stands for its slot's rank, rank 0.
- With the peer's fix, the core starts on rank 0's node, so notice 1 moves rank 0 to that spare
  and the core to rank 1's node.
- Notice 2, on rank 1's node, then needs a spare for rank 1. Nothing in this cell stands for it:
  the drained node's pod stands again for its own slot's rank, rank 0.
- So notice 2 can never be covered here. The harness would wait its full 900 s for cover before
  stopping, which is GPU time spent on nothing (attempts 2 and 3 stopped on such waits and reads).
- A spare for rank 1 means either a second spare pod, or a spare that stands for any position.
  Cort has ruled that the realistic spare is the generic one ("our spare shouldnt already hold
  the position, just the model weights"). That is the next work item, measured in its own run,
  not added to this one.

**What it gives up.** Start 3, the second successor's start. Measurement 1 has two starts, the
bring-up's and notice 1's successor's, which the pre-registration's own fallback allows ("measurement
1 then has two starts, not three"). Measurement 3 has one notice's streams in place of two.

**What does not move.** No decision and no bar. Each measurement's decision stands as written, read
on attempt 4's own numbers. The earlier attempts' figures sit beside them as first reads.

**The check before the rental.** Every record check the harness acts on was run against every saved
record (`replay-before-attempt-4.txt`, `bench/harness/k8s/replay_checks.sh`), and each answer
matched what its run did.

**Cort's word** (2026-10-16): "ok go head with 4", given after the one-notice plan and its cost (about $8–10) were put to him.

## 2. Attempt 5 runs as amendment 1 has it, written 2026-10-16

**The change.** None to the command, the cell or the bar. Attempt 5 runs the same command as
attempt 4 (`--losses 1`, for amendment 1's reasons), on images with:
- the digest fix (one checkpoint digests alike whichever downloader filled the cache);
- the peer's check across members at bring-up (Degraded MembersDiffer), which the harness's wait
  to serve now reads, so a digest disagreement stops the run at bring-up;
- the peer's fix for a head node's drain, which is now done once its head has left.

**The checks before the rental.**
- The digest fix was checked on the model fetched twice on a CPU, by Hugging Face's Xet
  downloader and by the plain one. The old code gave the two values the H100 nodes reported, and
  the fixed code gives one.
- Every record check the harness acts on was replayed against every saved record, attempt 4's
  included (`replay-before-attempt-5.txt`), and each answer matched what its run did.

**Cort's word** (2026-10-16), after attempt 4: "yeah start the next run right away unless thers
other things we can test and catch ahead of time". The checks above are the things caught ahead
of time.

## 3. Attempt 6 runs as amendments 1 and 2 have it, written 2026-10-16

**The change.** None to the command, the cell or the bar. Attempt 6 runs on images with attempt 5's
fixes:
- the dead core's requests dropped on vLLM's V2 runner as well as V1's;
- a node outliving the core it started, so the old core leaves by its retire, not with its node;
- the harness reading a restarted pod's previous log, and an error that is an object.

**The checks before the rental.**
- Every name the carrier and the head's take read off vLLM is held to the pinned classes, in a
  vLLM-gated contract test (`test_carrier_vllm_contract.py`, `test_drop_requests_vllm.py`). It
  ran in CI's adapter-vllm job on `e15330d` (294 passed, its only skips the two GPU-only tests),
  and its old-code mutation fails it (the V2 drop, PR #4's method).
- Every record check the harness acts on was replayed against every saved record, attempt 5's
  included (`replay-before-attempt-6.txt`), and each answer matched what its run did.

**First exercises.** Rule 2 (no saved record holds these cases), so each is flagged in advance:
- **The retire's migration class.** No run has yet retired a core, so the read the decision of
  measurement 3 rests on (a stream ending in `backend.engine_shutdown` and migrating to B) has
  no real-record case. The run-5 replay covers the parser on the frontend's 503 object, not that
  class. Attempt 6 is its first exercise.
- **The node outliving its core on GPUs.** The CPU handoff end to end does not tell the race apart:
  on one machine the retire reaches the core before its node sees its rank leave. The
  deterministic tests cover `outlive_core` itself, and attempt 6 is the first exercise of the
  race it closes.

**Cort's word** (2026-10-16): "yes", to attempt 6 at about $10, the same setup as attempt 5.

## 4. Attempt 7 runs as amendments 1 and 2 have it, written 2026-10-16

**The change.** None to the command, the cell or the bar. Attempt 7 runs on images built at
`5ddeb0f` (`dev-5ddeb0f`), with attempt 6's two failures fixed:
- **The old core stopping at the commit.** It never heard rank 0's move claimed: its node held only
  the departing rank and was sent the commit alone, and the executor raises at a commit whose claim
  it never heard. Every change is now delivered to the head's node, whatever that node holds
  (`30e3bf5`), and an engine that ends on a change's failure says why in the head's goodbye
  (`9712aac`).
- **The take refused on the spare's KV configuration.** The spare derived its configuration from
  its own worker and left the layout the engine core sets after building unset. A spare is now
  replayed what the core sent its workers at its start, never deriving it (`5781484`, decision
  0038's start-up replay, from the construction audit), hardened in `5ddeb0f`:
  - a standing spare re-checks at each new head;
  - every join offer names its core's calls;
  - the core caps the queues it holds, and closes each on its reader leaving;
  - the spare profiles its memory as the engine's workers do.

**In place of the single-VM rig** that attempt 6's result proposed: Cort chose to trace the code
and reproduce each failure on CPU before renting ("i dont want to pay 40 bucks for this, is there
not a better way to debug"; "the first hting yuou havev to do is trace the code exactly line for
line"). Both failures were reproduced in the CPU handoff end to end on the real control plane and
fixed there:
- the old core's record of a commit without its claim (`b99b3eb`);
- the spare's replay through the real coordinator, at bring-up and after the take.

**The checks before the rental.**
- The replay's vLLM-gated tests ran in CI's adapter-vllm job at the pin (PR #6 on `5ddeb0f`: 325
  passed, its only skips the two GPU-only tests). They cover:
  - the engine core's start-up calls and their order;
  - a replay crossing vLLM's queue;
  - the queue closed on a reader that exits without closing.
- Main's CI on `5ddeb0f` passed all 12 jobs.
- Every record check the harness acts on was replayed against every saved record, attempt 6's
  included (`replay-before-attempt-7.txt`). The earlier records read exactly as before attempt 6,
  and attempt 6's record reads as its run went: rank 0's move committed at epoch 1, no head change,
  the engine down.

**First exercises.** Rule 2: no saved record holds these cases, so each is flagged in advance.
- **The spare's replay on GPUs.** Its ask through the coordinator, the core's queue across nodes,
  and the calls run on a real worker before its warm-up.
- **The take, the core's retire and the stream migration.** Attempt 6 stopped at the commit, so
  everything after it is unread on GPUs. That includes amendment 3's flagged retire migration
  class.

**Read beside the measurements, with no decision attached.** The spare's time from its start to
standing now includes vLLM's own memory profiling (about 2.9 s on attempt 6's battery,
2026-10-09). A spare built for one rank pays it before standing, so the move's gap should not
change. The record's spare logs give both times.

**Cort's word** (2026-10-16): "ok", to attempt 7 at about $8–12, after the audit, the replay's
expected impact, and what it does not prove were put to him.

## 5. Attempt 8 runs as amendments 1 and 2 have it, written 2026-10-16

**The change.** None to the command, the cell or the bar. Attempt 8 runs on images built at
`7001f7e` (`dev-7001f7e`), with attempt 7's two failures fixed and one found since:
- **The old core failing to complete rank 0's move.** The new ring's size was read off the old
  ring's shared-memory buffer, which a core holding no rank does not have. It now comes from vLLM's
  setting (`2768c43`).
- **The spare refusing the successor's take.** A join offer now names its kind: a change, a
  bring-up or a take. Only a bring-up or a take may come without its core's calls' digest; a digest
  any offer names is checked (`2768c43`, `2b9d5aa`).
- **Found by the CPU test written for these.** A ring or response queue a change replaces is now
  closed, its zmq context destroyed with no linger. vLLM sets no linger, and a dropped queue
  holding a message a gone reader never took blocked forever in the collector's close, whatever
  thread ran it (`7001f7e`).

**In place of a GPU run, the CPU checks Cort asked for first** ("ok", after "I'd like to build a
CPU test that does: the real executor as a core holding no ranks, completing a change, then a
successor taking over a spare that took a rank").
- **The real path, on CPU.** `test_carrier_change_vllm.py` runs the executor's own
  `complete_change` on a core holding no rank, over vLLM's real queues, with threads for the
  survivor, the leaving rank and the spare. With attempt 7's buffer read restored, it fails (PR
  #9).
- **The spare's join.** `test_spare_join.py` feeds `spare_main.join_position` a change's, a
  bring-up's and a take's offers. Attempt 7's guard fails its take tests.
- **CI.** Main's CI on `7001f7e` passed all 12 jobs (adapter-vllm: 334 passed).
- **The harness reads.** Every record check the harness acts on was replayed against every saved
  record, attempt 7's included (`replay-before-attempt-8.txt`). The earlier records read exactly
  as before attempt 7, and attempt 7's record reads as its run went: rank 0's move at epoch 1, no
  head change, the engine down.

**First exercises.** Rule 2: no saved record holds these cases, so each is flagged in advance.
- The executor's completion of a change on a core holding no rank, on GPUs.
- The successor's take of a spare that joined the old core, on GPUs.
- The core's retire and the stream migration (amendment 3's flag, still unread).

**Cort's word** (2026-10-16): "ok", to attempt 8 at about $10, on images built at `7001f7e`.
