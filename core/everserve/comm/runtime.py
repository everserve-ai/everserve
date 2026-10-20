# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The per-worker runtime: a membership epoch, the collective it governs, and a step guard.

The scheduler never sees any of this. Two paths drive one runtime, and which is which:

* The product path (decision 0013) is :meth:`Runtime.at_boundary`. An engine that owns its step
  loop (vLLM) runs its steps itself; the adapter calls :meth:`Runtime.at_boundary` between steps,
  the pending change is applied there, and the engine resumes from its committed position: the
  scheduler replays what the change cost from the committed counts. The adapter constructs the
  runtime with ``reconcile_on_recover=False``; no agreement or reconciliation collective runs on
  this path.
* The reference guard (decision 0004) is :meth:`Runtime.run_step` and the agreement all-reduce it
  ends with. The CPU spec in ``core/tests`` runs SHRINK on :mod:`everserve.reference` under it.
  Its members alone are :meth:`Runtime.run_step`, ``_agree``, ``_reconcile_accepted``,
  ``_abandon_attempt``, ``_wait_fault``, the attempt buffers and their quarantine, ``accepted``,
  ``profile``, and the ``fault_confirm_timeout_s`` and ``reconcile_on_recover`` arguments. They
  stay on this class rather than on a subclass in :mod:`everserve.reference`: the product path
  constructs the class with ``reconcile_on_recover``, the recovery both paths share
  (:meth:`Runtime.on_change`, the fault flag and epoch, the wait ``change_timeout_s`` bounds, the
  callback lists the adapter's lifecycle registers on) runs the reconciliation and the quarantine
  keyed from :meth:`Runtime.all_reduce`, so a split needs hooks in the collective call, the
  recovery and the dump, not a move of methods.

The protocol, shared by both paths up to the end of a step:

* Every collective the model issues goes through :meth:`Runtime.all_reduce`, which refuses to
  enqueue once a fault is flagged and otherwise hands the tensor to the :class:`Collective`.
* A side thread (the control channel) calls :meth:`Runtime.on_fault` with the outcome the
  controller chose. That flags the fault, bumps the epoch under the lock, refuses the current set
  (the slot kernel's abort word; an adapter's communicators go to a refusing stand-in and their
  release starts beside the change, decision 0025) and moves the collective to the new member
  set: a pointer swap to a pre-formed group for REPLACE and for a SHRINK whose survivor set was
  pre-formed, a shrink of the current group with abort otherwise. Either way a kernel stuck in a
  collective with the dead peer is released, and the change is acknowledged without waiting for
  anything to be freed.

Under the reference guard, :meth:`Runtime.run_step` wraps the model forward. If a fault is pending
it recovers first (under SHRINK the model absorbs the lost rank's units through its callbacks;
under REPLACE the spare restores and survivors do nothing), runs the forward, runs the commit
callbacks (the KV put), then the agreement: a one-element all-reduce of ones that can only sum to
the member count if every rank finished every collective of the step and its commit. Either every
survivor accepts the step or every survivor re-runs it. A re-run writes the same KV slots with the
same values, so it is idempotent, and with placement-independent arithmetic its output is
bit-identical.

The KV put precedes the agreement so that a step the survivors accept always has every rank's KV
committed. A rank killed after its put but before the agreement makes the survivors re-run, which
is safe; a put after the agreement would have left an accepted step with KV missing.

A single collective is not atomic under a crash: the agreement can complete on one survivor (the
dead rank's contribution arrived before it died) and fail on another. The survivors then disagree
on whether the step was accepted, and the one that accepted it would wait for the next step while
the others re-run this one. Recovery under the guard therefore ends with a reconciliation on the
new group: every member contributes its accepted-step count, and a member that is ahead replays
its last step through the collectives, output discarded, so the others can complete theirs.
``fn`` must be re-runnable for this: it writes the same state again, which the step guard already
requires.

A member with no step in progress when the fault arrives recovers on the fault thread at once
rather than at its next step. Its caller will not issue the next step until every member has
answered the current one, and the others are in recovery waiting for it; recovering only in
:meth:`Runtime.run_step` would deadlock exactly there. On the product path no step is ever in
progress for the runtime, so every change recovers on the fault thread, and
:meth:`Runtime.at_boundary` finds it applied.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import torch

from everserve.comm.collective import Collective, CollectiveMoved
from everserve.comm.handle import LocalStandIn
from everserve.outcome import Outcome

log = logging.getLogger(__name__)

T = TypeVar("T")
RecoverCallback = Callable[[int], Mapping[str, Any] | None]
"""Called with the dead rank during a SHRINK recovery; returns facts for the event log."""
AttemptCallback = Callable[[], None]
"""Called before every forward attempt, including re-runs; used to drop uncommitted state."""
CommitCallback = Callable[[], None]
"""Called after the forward and before the agreement; used to put this step's KV in the store."""


CHANGE_ABORT = "step aborted: the member set changed"
"""In the message of every error a step raises because the member set changed under it.

An error crosses an engine's worker RPC as its text alone, so this is how the engine's executor
tells a step a membership change ended from a step that failed on its own
(``everserve_vllm.losses.loss_behind``)."""


class StepAborted(Exception):
    """Raised inside a step once a fault is flagged. The step is re-run after recovery."""

    def __init__(self, detail: str = "") -> None:
        """The message is :data:`CHANGE_ABORT`, with ``detail`` after it when given."""
        super().__init__(f"{CHANGE_ABORT}: {detail}" if detail else CHANGE_ABORT)


@dataclass
class Event:
    """One entry in the runtime's event log."""

    kind: str
    at: float
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepProfile:
    """Cumulative step accounting."""

    steps: int = 0
    forward_s: float = 0.0
    commit_s: float = 0.0


@dataclass
class _Pending:
    """A membership change the side thread prepared and the step thread has not applied."""

    lost: tuple[int, ...]
    outcome: Outcome
    method: str


TRACE_KEPT = 4096
"""Transitions kept in memory and in the dump; the trace file holds every one."""


class Runtime:
    """One rank's membership state and its step guard."""

    took_epoch: int | None = None
    """The epoch a spare took its position at; None for a worker, and for a spare standing."""

    def _trace(self, kind: str, **fields: Any) -> None:
        """Log a transition with its thread and clock, and write it out at once.

        The log is the evidence when a rank hangs: a dump only at the end of a recovery leaves
        nothing behind for the rank that never got there. Each transition is one line appended
        to ``rank<N>.trace.jsonl`` beside the event log, so a step's transitions cost one write
        of their own size; the last :data:`TRACE_KEPT` ride in the dump too.
        """
        fields["thread"] = threading.current_thread().name
        fields["t"] = time.perf_counter()
        event = Event(kind, time.time(), fields)
        self.trace.append(event)
        if self._event_log is None:
            return
        line = json.dumps({"kind": f"trace:{kind}", "at": event.at, "fields": fields}, default=str)
        with self._dump_lock:
            self._event_log.mkdir(parents=True, exist_ok=True)
            with (self._event_log / f"rank{self.orig_rank}.trace.jsonl").open("a") as f:
                f.write(line + "\n")

    def __init__(
        self,
        rank: int,
        world: int,
        device: torch.device,
        collective: Collective,
        event_log: Path | None = None,
        change_timeout_s: float = 600.0,
        fault_confirm_timeout_s: float = 60.0,
        reconcile_on_recover: bool = True,
    ) -> None:
        """Create the runtime for original rank ``rank`` of ``world`` on ``device``.

        Args:
            rank: this process's original rank id; never changes.
            world: ranks at startup.
            device: where the model runs; CUDA streams are synchronised at step boundaries.
            collective: the collective this runtime governs.
            event_log: a directory for ``rank<N>.json`` event dumps, or None for no file.
            change_timeout_s: how long recovery waits for the side thread's change to finish,
                on both paths.
            fault_confirm_timeout_s: the reference guard's: how long a failed collective in
                :meth:`run_step` waits for the executor to confirm a fault before the failure is
                treated as a real error.
            reconcile_on_recover: the reference guard's: run the accepted-step reconciliation
                at the end of recovery. On by default. The adapter turns it off: its engine
                replays from committed counts and its spare does not take part in recovery
                collectives (decision 0013).
        """
        self.orig_rank = rank
        self.world = world
        self.device = device
        self.collective = collective
        self.alive: list[int] = list(range(world))
        self.epoch = 0
        self.lock = threading.Lock()
        self.fault: tuple[int, ...] | None = None
        self.recover_callbacks: list[RecoverCallback] = []
        self.attempt_callbacks: list[AttemptCallback] = []
        self.commit_callbacks: list[CommitCallback] = []
        self.events: list[Event] = []
        """What happened: faults, recoveries, re-runs, reconciliations."""
        self.trace: collections.deque[Event] = collections.deque(maxlen=TRACE_KEPT)
        """The last transitions with their thread and clock; every one is in the trace file."""
        self.profile = StepProfile()
        self._changed = threading.Event()
        self._pending: _Pending | None = None
        self._attempt_buffers: list[tuple[int, torch.Tensor]] = []
        self._quarantine: dict[int, list[torch.Tensor]] = {}
        self._event_log = event_log
        self._change_timeout_s = change_timeout_s
        self._fault_confirm_timeout_s = fault_confirm_timeout_s
        self._reconcile = reconcile_on_recover
        self.accepted = 0
        """Steps every member agreed on so far."""
        self._last_fn: Callable[[], Any] | None = None
        self._in_step = False
        self._recover_lock = threading.Lock()
        self._dump_lock = threading.Lock()

    @property
    def rank(self) -> int:
        """This rank's position among the live ranks right now."""
        return self.alive.index(self.orig_rank)

    @property
    def is_cuda(self) -> bool:
        """Whether step boundaries need a CUDA stream sync."""
        return self.device.type == "cuda"

    def identity(self) -> tuple[int, int]:
        """``(original rank, epoch)`` for the control channel."""
        return self.orig_rank, self.epoch

    # ---------------------------------------------------------------- collectives
    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """Sum ``t`` in place over the live members, unless a fault is pending.

        The lock covers only the check. A collective that blocks on a dead peer must not stop
        the side thread from flagging the fault and moving the collective.
        """
        with self.lock:
            if self.fault is not None:
                raise StepAborted()
            epoch = self.epoch
        self._attempt_buffers.append((epoch, t))
        try:
            self.collective.all_reduce(t, epoch=epoch)
        except CollectiveMoved:
            raise StepAborted() from None
        return t

    def sync(self) -> None:
        """Wait for the device stream at a step boundary."""
        if self.is_cuda:
            torch.cuda.current_stream(self.device).synchronize()

    # ---------------------------------------------------------------- fault path
    def on_fault(
        self, dead: int, sent_at: float | None = None, outcome: Outcome = Outcome.SHRINK
    ) -> None:
        """One rank is gone: :meth:`on_change` for a set of one."""
        self.on_change((dead,), sent_at, outcome)

    def on_change(
        self,
        lost: Sequence[int],
        sent_at: float | None = None,
        outcome: Outcome = Outcome.SHRINK,
        epoch: int | None = None,
        departing: Sequence[int] = (),
    ) -> None:
        """Side thread: flag the loss of ``lost``, then move the collective to the new member set.

        ``lost`` is one rank, or every rank of a node, in one change at one epoch
        (`docs/MULTINODE.md` section 2). ``outcome`` is what the controller chose (decision 0004).
        REPLACE moves to the pre-formed set with each lost rank's spare in its place; the world and
        the live set do not change. SHRINK moves to the survivor set, by pointer swap when it was
        pre-formed and by shrinking the current group with abort otherwise. REBUILD is not a
        runtime change: the adapter rebuilds the executor, so it is rejected here.

        ``epoch`` is the change's own, from the coordinator: a process that was not in the
        changes before it (a spare standing for any position, outside the membership) has not
        counted them, and moves to the change's epoch, not to its own count plus one (counting,
        a spare acknowledges epoch 1 for a change at epoch 2). Without it, the next epoch after
        this process's own.

        ``departing`` are the lost ranks leaving on notice (protocol 9): alive, they finish the
        step in flight with the rest and leave at the boundary, so a change whose every lost rank
        departs refuses nothing. The change says so itself; a notice remembered here would not
        survive a boundary between it and its change, nor reach a process that joined after it:
        a notice that arrives during another change's boundary is cleared by it, and the
        survivors then refuse the departing rank's change while it is mid-step, which never
        ends.

        Idempotent for ranks already known lost. Returns once the collective has moved, which is
        the acknowledgement: the old groups' release runs beside it and is finished at the
        boundary (decision 0025). The step thread applies the rest in :meth:`run_step`.

        Raises:
            ValueError: for REBUILD, or an empty set.
            KeyError: for REPLACE without a pre-formed replacement set.
        """
        if outcome is Outcome.REBUILD:
            raise ValueError("REBUILD is an executor action, not a runtime change")
        lost = tuple(dict.fromkeys(lost))
        if not lost:
            raise ValueError("a change names at least one rank")
        if self.is_cuda:
            torch.cuda.set_device(self.device)
        t0 = time.time()
        with self.lock:
            known = self.fault or ()
            if all(r not in self.alive or r in known for r in lost):
                self._trace("fault_ignored", lost=lost)
                return
            self.fault = lost
            self.epoch = epoch if epoch is not None else self.epoch + 1
            epoch = self.epoch
        self._trace("fault_flagged", lost=lost, epoch=epoch, in_step=self._in_step)
        if set(lost) <= set(departing):
            self._trace("planned_departure", lost=lost)
        else:
            self.collective.refuse()
        if outcome is Outcome.REPLACE:
            self.collective.change(self.collective.replacement_set(lost), epoch)
            method = "shadow"
        else:
            survivors = self.collective.survivor_set(lost)
            if self.collective.has_shadow(survivors):
                self.collective.change(survivors, epoch)
                method = "shadow"
            else:
                method = self.collective.shrink(lost, epoch).value
        self._pending = _Pending(lost, outcome, method)
        self.events.append(
            Event(
                outcome.value,
                t0,
                {
                    "lost": list(lost),
                    "sent_at": sent_at,
                    # from the change's delivery (the coordinator's clock) to this refusal: the
                    # daemon hop plus the two clocks' offset
                    "hop_s": round(t0 - sent_at, 6) if sent_at else None,
                    "ready_s": time.time() - t0,
                    "method": method,
                    "epoch": epoch,
                },
            )
        )
        self._changed.set()
        with self.lock:
            idle = not self._in_step
        self._trace("fault_staged", lost=lost, idle=idle)
        if idle:
            # idle between steps: recover here, or the members in recovery wait for us forever.
            # A step that was running sees the flag at its next collective or at its end.
            self._recover()

    def on_abandoned(self, epoch: int, detail: str) -> None:
        """The change at ``epoch`` failed after its delivery: the engine is down (decision 0032).

        A worker records it and does nothing else: the head fails the engine, and vLLM's shutdown
        of its workers is the one stop path. A second here would race it.
        """
        log.warning("the change at epoch %d was abandoned: %s", epoch, detail)
        self._trace("abandoned", epoch=epoch, detail=detail)

    def on_take(self, takes: Sequence[Any]) -> None:
        """A take for a process that stands for one position: nothing to take, said once."""
        log.warning(
            "rank %d stands for its own position; a take of %s is not its", self.orig_rank, takes
        )

    def on_cover_plan(self, plan: Any) -> None:
        """A cover plan for a process that forms no cover: not applied, said once."""
        log.info("cover plan %d not applied: no cover here", getattr(plan, "generation", 0))

    def on_displaced(self, epoch: int, why: str) -> None:
        """Displaced, with nothing to do about it here: the adapter's runtime leaves."""
        log.warning("rank %d displaced at epoch %d: %s", self.orig_rank, epoch, why)

    def on_notice(self, rank: int, deadline_ms: int) -> None:
        """A rank will leave at ``deadline_ms`` (Unix milliseconds).

        Recorded for the event log; the controller drives the staging and, at the deadline, calls
        :meth:`on_fault` with REPLACE so the engine never observes a fault.
        """
        self.events.append(Event("notice", time.time(), {"rank": rank, "deadline_ms": deadline_ms}))

    def _recover(self) -> None:
        self._trace("recover_wait")
        with self._recover_lock:
            if self.fault is None:
                self._trace("recover_skipped")
                return  # recovered already, on the other thread
            self._trace("recover_start")
            self._recover_locked()
            self._trace("recover_end")

    def _recover_locked(self) -> None:
        t0 = time.time()
        if not self._changed.wait(timeout=self._change_timeout_s):
            raise RuntimeError("membership change did not finish")
        pending = self._pending
        assert pending is not None
        info: dict[str, Any] = {"outcome": pending.outcome.value, "method": pending.method}
        if pending.outcome is Outcome.SHRINK:
            for dead in pending.lost:
                self.alive.remove(dead)
                for cb in self.recover_callbacks:
                    info.update(cb(dead) or {})
        self.sync()
        with self.lock:
            self.fault = None
            self._pending = None
            self._changed.clear()
        if self._reconcile:
            info["reconciled"] = self._reconcile_accepted()
        self.events.append(
            Event(
                "recover",
                t0,
                {
                    "lost": list(pending.lost),
                    "alive": list(self.alive),
                    "rank": self.rank,
                    "recover_s": time.time() - t0,
                    **info,
                },
            )
        )
        self.dump()

    def at_boundary(self) -> bool:
        """Apply a pending change at a step boundary, for an engine that owns its step loop.

        The product path (decision 0013). The engine's step cannot be re-run from outside (vLLM's
        model runner keeps request state across steps), so the adapter calls this between steps
        instead of wrapping the forward in :meth:`run_step`: the pending change is applied here
        and the engine resumes from its committed position, the scheduler replaying what the
        change cost. No agreement collective runs; the adapter constructs the runtime with
        ``reconcile_on_recover=False``. Returns whether a change was applied.
        """
        if self.fault is None:
            return False
        self._recover()
        return True

    def _reconcile_accepted(self) -> bool:
        """Agree on the accepted-step count on the new group; replay the last step if ahead.

        Every member contributes ``accepted``; the sum is ``n * accepted`` when all agree. A smaller
        sum means some member will re-run the step this one already accepted, so this one replays
        it through the collectives (forward, commit callbacks, agreement) with the output discarded.
        Returns whether a replay happened.
        """
        if isinstance(self.collective.current, LocalStandIn):
            return False  # no peers to reconcile with (a spare before it joins, or a test)
        n = len(self.alive)
        t = torch.full((1,), float(self.accepted), dtype=torch.float32, device=self.device)
        self._trace("reconcile_exchange", accepted=self.accepted, n=n, epoch=self.epoch)
        self.collective.all_reduce(t, epoch=self.epoch)
        self.sync()
        total = int(t.item())
        self._trace("reconcile_result", accepted=self.accepted, sum=total, n=n)
        if total >= n * self.accepted:
            return False
        if self._last_fn is None:
            raise RuntimeError("a member is behind but there is no step to replay")
        self.events.append(
            Event("reconcile", time.time(), {"accepted": self.accepted, "sum": total, "n": n})
        )
        for cb in self.attempt_callbacks:
            cb()
        self._last_fn()
        self.sync()
        for cb in self.commit_callbacks:
            cb()
        self._agree(n)
        self._attempt_buffers = []
        return True

    def _abandon_attempt(self) -> None:
        """Keep the failed attempt's collective buffers alive until the old group cannot write.

        A group that cannot abort (gloo) may still deliver a live peer's data into the output
        buffer of a collective that already failed. If that memory were freed and reused by the
        re-run, the late write would corrupt a fresh tensor, and an all-reduce would then spread
        the corruption to every rank identically. Buffers are keyed by the epoch of the group they
        were handed to and held until two epochs have passed, longer than any collective timeout.
        """
        for epoch, t in self._attempt_buffers:
            self._quarantine.setdefault(epoch, []).append(t)
        self._attempt_buffers = []
        for old in [e for e in self._quarantine if e < self.epoch - 1]:
            del self._quarantine[old]

    def _wait_fault(self, timeout_s: float) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self.fault is not None:
                return True
            time.sleep(0.01)
        return False

    def _agree(self, n: int) -> bool:
        """End-of-step agreement: every survivor accepts the step, or every survivor re-runs it.

        The reference guard's collective; :meth:`run_step` and the reconciliation issue it.
        """
        with self.lock:
            if self.fault is not None:
                raise StepAborted()
            epoch = self.epoch
        t = torch.ones(1, dtype=torch.float32, device=self.device)
        self._attempt_buffers.append((epoch, t))
        try:
            self.collective.all_reduce(t, epoch=epoch)
        except CollectiveMoved:
            raise StepAborted() from None
        self.sync()
        return int(t.item()) == n

    # ---------------------------------------------------------------- the step guard
    def run_step(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` as one step: recover first if needed, commit, re-run until all ranks agree.

        The reference guard (decision 0004): the CPU spec runs SHRINK on the reference cores under
        it, each step ending in the agreement all-reduce (:meth:`_agree`) and re-run until every
        survivor accepts it. No adapter calls it: an engine that owns its step loop applies a
        change with :meth:`at_boundary` instead.
        """
        attempts = 0
        self._last_fn = fn
        while True:
            # a recovery may be running on the fault thread (we were idle when the fault came):
            # it clears the fault before its reconciliation collective, so wait it out here or
            # this thread's first collective would share the new group with it
            with self._recover_lock:
                pass
            if self.fault is not None:
                self._recover()
            with self.lock:
                self._in_step = True
            self._trace("attempt", attempt=attempts, epoch=self.epoch, n=len(self.alive))
            for cb in self.attempt_callbacks:
                cb()
            n = len(self.alive)
            try:
                t0 = time.perf_counter()
                out = fn()
                self.sync()
                t1 = time.perf_counter()
                for cb in self.commit_callbacks:
                    cb()
                ok = self._agree(n)
                self.profile.forward_s += t1 - t0
                self.profile.commit_s += time.perf_counter() - t1
                self.profile.steps += 1
                if self.profile.steps % 50 == 0:
                    self.dump()
            except StepAborted:
                self._trace("attempt_aborted", attempt=attempts)
                self._abandon_attempt()
                attempts += 1
                continue
            except Exception as e:
                # A collective with a dead peer can fail on its own (gloo timeout, NCCL error).
                # It is a fault only if the executor confirms one; otherwise it is a real error.
                self._trace(
                    "attempt_failed", attempt=attempts, error=f"{type(e).__name__}: {e}"[:160]
                )
                self._abandon_attempt()
                if self._wait_fault(self._fault_confirm_timeout_s):
                    attempts += 1
                    continue
                raise
            if not ok:
                self._abandon_attempt()
                if not self._wait_fault(self._fault_confirm_timeout_s):
                    raise RuntimeError("step commit failed without a fault")
                attempts += 1
                continue
            self._attempt_buffers = []
            self.accepted += 1
            with self.lock:
                self._in_step = False
                landed = self.fault is not None
            self._trace("accepted", accepted=self.accepted, landed=landed)
            if landed:
                # the fault arrived after this step's last collective: the fault thread saw a
                # step in progress and left recovery to us
                self._recover()
            if attempts:
                self.events.append(Event("redo", time.time(), {"attempts": attempts}))
                self.dump()
            return out

    # ---------------------------------------------------------------- the event log
    def dump(self) -> None:
        """Write the events and profile to ``rank<N>.json`` in the event-log directory, if set."""
        if self._event_log is None:
            return
        with self._dump_lock:
            self._dump_locked()

    def _dump_locked(self) -> None:
        assert self._event_log is not None
        self._event_log.mkdir(parents=True, exist_ok=True)
        payload = (
            [asdict(e) for e in self.events]
            + [{"kind": f"trace:{e.kind}", "at": e.at, "fields": e.fields} for e in self.trace]
            + [{"kind": "profile", "fields": asdict(self.profile)}]
        )
        # Written beside and renamed over, one writer at a time: a rank killed mid-write leaves
        # the previous complete log, never a truncated one, so a reader can always parse it.
        final = self._event_log / f"rank{self.orig_rank}.json"
        tmp = final.with_suffix(".json.part")
        tmp.write_text(json.dumps(payload, default=str))
        os.replace(tmp, final)
