# 0025: a change refuses its dead groups now and releases them beside the change, bounded

2026-10-08. Status: current, as amended below (2026-10-09, 2026-10-15).

**Context.** A survivor's change did three things in one call on the daemon link's thread:
point its handles at a refusing stand-in, abort the live communicators, and stage the new set;
the acknowledgement to the coordinator followed. The aborts each ran on a thread with a bound,
but torch's abort of its process groups ran on the caller with none, and the fabric family's
group, a second wrapper over vLLM's pipeline communicator, was never marked at all: the family's
communicator was aborted through vLLM's handle while the family's own group still answered the
pipeline. Nebius run 9 (`bench/runs/PLANNED/two-node-0024-smoke/NOTE.md`) showed what that
costs across hosts. Node-b's stage died; rank 0's pipeline socket closed in 1.7 s; the survivor's
next pipeline send went through the freed communicator and segfaulted in NCCL's proxy 82 s into
the change, with the coordinator still waiting for an acknowledgement that sat behind the aborts.
On one host the aborts return in under a second and the order never showed; across hosts, with
a peer that is a dead socket, it is the whole pause.

**Decision.** A change has two separate acts, and the acknowledgement sits between them.

*Refuse now.* On the daemon link's thread the collective refuses every group that can hold the
dead peer: the slot's abort word; the live group, the fabric family's current group and every
handle vLLM holds pointed at a refusing stand-in. A refused group raises in Python on any
collective, so a step blocked on the dead peer fails at its next call and nothing calls into the
communicator again, from any path. The new set is staged and the change is acknowledged. The
acknowledgement means "staged": it never waits for anything to be freed.

*Release beside the change.* Freeing what a refused group held is hygiene, not the change.
Each release (`ncclCommAbort` of one communicator; torch's abort of its process groups) starts on
its own thread at the refusal. Nothing waits for it: not the acknowledgement, not the step
thread, and not the boundary. The boundary swaps and reaps: it records each release that has
returned, with its time, and counts as leaked any still running past its bound measured from its
start (`releaseBoundMs` in the policy's engine entry, 10 s unless given); a leaked release is
never waited out and never called again. This replaced an earlier wording under which the
boundary waited for each release up to the bound before the swap: on Nebius run 11 that wait was
0.5 to 0.8 s of a 1.1 s replacement, for a result nothing on the swap depended on, since the
refusal and the guard already keep every thread out of a released communicator.

*A release frees only with no thread inside the communicator's host code.* The refusal stops
new calls; a call already inside when the refusal lands must leave before the free. Run 10 on
Nebius showed the step thread inside ``ncclSend`` on the refused pipeline communicator, backed
up behind sends to the dead peer, with the release's free running under it. Two things hold the
invariant. The exchange keeps one transfer in flight per communicator and waits on the previous
transfer's CUDA event before it enqueues, so a thread's long wait on a communicator is a device
wait, which the abort releases, and its host calls are short. And every call takes the
communicator's shared guard while the release takes its exclusive guard
(`everserve.comm.guard`), so the free waits for a call in progress rather than running under
it; a call that never leaves keeps the release running, which the boundary leaves behind and
counts, as any slow release.

The placement of the shared side is the whole point, and run 10's sixth attempt fixed it. The
wait on the previous transfer's event stays outside the guard: the abort is what ends that
wait, so the abort must not be waiting for it, or the release deadlocks and the step thread
stays in the wait for good. The refusal check sits inside the shared side, just before the
enqueue, never before the wait or before the acquire: a call that reads "not refused" while
holding the shared side holds the communicator alive until it leaves, because the refusal is
set before the release starts and the free waits for the shared side; a call that reads
"refused" raises. Checked before the wait, the abort can end the wait and the enqueue goes onto
the freed communicator (attempt 6); checked before the acquire, the whole release can run in
the gap. In order: wait, then under the shared side, check, then enqueue.

*Amended 2026-10-09 (passes 9 and 10 of the two-node dry run,
`bench/runs/PLANNED/p2-two-node/dryrun-1gpu-pass9` and `-pass10`).* The wait itself is not a blocking
device wait when a check is present. Blocking on the previous transfer's event made the abort what
ended the survivor's wait, as the paragraph above says, and the abort takes 0.25 to 0.5 s to free
the old communicator: on every join whose previous transfer was still pending when the peer died,
the survivor's step sat in that wait until the abort returned, reached its boundary that much late,
and the executor's new ring waited on it (ring_ready 229 to 256 ms on three joins, 1 ms on the one
whose transfer had completed). So with a check the wait polls the event beside the refusal: the
refusal ends the step at once, with nothing enqueued, and the abort frees the communicator behind
the step rather than in front of it; without a check (no group anyone could refuse) the device wait
stays. The order is unchanged: a completed wait, then under the shared side the check, then the
enqueue, so nothing is ever enqueued onto a freed communicator. The boundary still waits for no
release; it is now also not reached late by one.

*Transport errors are not the signal.* A socket to a dead host closes in seconds; an RDMA queue
pair can wait minutes. The survivor is released by the coordinator's word through the refusal,
not by the transport noticing, so the pause does not depend on the fabric.

**Consequences.**

- The commit of a change across hosts depends on the refusal, which is microseconds, and on the
  spare's staging; the aborts' time and whether they return at all are out of the critical path,
  and so is the boundary, which swaps without waiting for them.
- A leaked release holds its communicator's memory (about 66 MiB per set per GPU) until the
  process ends. The count is reported at every boundary (`released`, `leaked`). A survivor that
  accumulates leaks is retired by a planned change, not by surgery: the coordinator moves its
  position to a spare on notice and the daemon restarts the process, which membership already
  does cheaply. That rotation is not built; the count that triggers it is.
- Whether `ncclCommAbort` over a socket transport ever returns, and whether a refused group with
  a dead socket peer can be left unaborted without the proxy failing on its own, are empirical;
  run 10 on the three-node smoke records both with the native stack of any crash, and the
  default (abort on the release thread) stands unless the stack says otherwise.
- The node role's sentinel reported the kill after the CUDA process exited (1.0–1.8 s). The
  slot watch reads `/proc` for the exit flag and the pending `SIGKILL`
  (`everserve.comm.slotwatch.process_state`); the node role uses the same read for its workers,
  so a kill is reported in milliseconds. That is the adapter's landing, not this one.
- `Collective.refuse` replaces `abort` in the core; `PyNcclGroup.refuse` returns the abort to run
  and `refused` replaces `aborted` in the adapter; `groups.Releases` holds the running releases.

*Amended 2026-10-15: a leaked release retires its process, by notice, from the first.* A leak was
counted, and the rotation it should trigger was left unbuilt. A thread left inside
`ncclCommAbort` for the rest of a process's life is a leak without a bound, and a leak without a
bound is a defect, not a cost. Two facts set the remedy:
- The documented cause of an abort that never returns is a CUDA graph still holding the
  communicator. That cannot occur here: no graph captures a serving communicator (decision 0029's
  graph-safe op runs every NCCL call between graph pieces). A guard on the handles vLLM holds
  refuses any collective made inside a capture, so a graph that would hold one fails the engine's
  start instead.
- An abort can still stay inside NCCL for other reasons, for instance a proxy thread blocked on a
  socket. Its process is serving a live rank, so ending it would turn one stuck release into an
  unplanned loss.

So:
- **Degraded at the first leak, and moved by the coordinator.** At the boundary that first counts
  a release as leaked, the process reports it once (`ReleaseLeaked`: the release's name, how long
  it has run, and the process's count), and its daemon forwards it under the process's
  connection. The process reports; it decides nothing. The coordinator, which already decides
  cover, puts a Degraded reason on the engine. If a spare can take the position now, it starts a
  planned departure through the drain's path: a notice to the process's node (protocol 9's
  `Change.departing`). The spare takes the position at the notice's deadline, with no gap, and
  the process's node restarts it. That bounds the leak to one release's run plus one planned
  change.
- **The departure's own bound.** The notice falls under the policy's `noticeMinMs`. If no spare
  can take the position (none stands, or none is covered for it), the process stays serving,
  degraded and counted. The reason stays on the engine's record, and the coordinator tries again
  at each plan that covers the position, so the operator sees why it stays.
- **One departure.** A second leak from the same process while its departure is pending adds to
  the count and to the reason. It starts no second departure.
- **Named where an operator looks.** The coordinator counts each leak in
  `everserve_releases_leaked_total{engine}` and records an engine event, `ReleaseLeaked`, with
  the release's name, its process and how long it has run. The alert `EverserveReleaseLeaked`
  fires on the counter's increase. Its runbook entry (`docs/RUNBOOK.md`) says what a process left
  degraded and waiting for a spare means, and the two ways out: add spare capacity, or restart
  the process by hand and accept the loss.

The remaining consequences of the first bullet stand: the count is reported at every boundary,
and a leaked release's memory is held until the process ends, which now happens at the planned
change.
