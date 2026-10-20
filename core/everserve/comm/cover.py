# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A process's part of its engine's cover plan (decision 0035).

The coordinator plans every group an engine pre-forms and delivers the plan to every process of
the engine. A group is its scope and the members it spans, whatever sets need it (decision
0035, amended): a process forms each of the plan's groups that span it once, in the background,
holds it under every set that needs it, releases a group no set of the plan needs, and reports
each group, by its spans: formed, failed or released, at the plan's generation. Coverage is the
coordinator's, read from those reports.

What a process holds is kept by :class:`Cover` beside the :class:`~everserve.comm.collective.
Collective` that owns the groups:

- A plan below or at the generation applied is ignored: generations only move forward.
- A plan for a newer epoch first lets go of every held group, the serving ones aside, that names
  as a rank a position whose holder changed between the two plans' placements: that group holds
  the position's old process. A process the change moved also does this at its commit
  (:meth:`Collective.rename_at_commit`); a spare standing for any position runs no commit, and
  this is how its groups follow.
- A group the plan names that the process already holds is reported formed again, at the new
  generation, in no time (``seconds`` 0): coverage counts only the current generation's reports,
  and after a commit a kept group's key has changed (:meth:`Collective.rename_at_commit`).
- A group the plan no longer names is released beside serving, as a refused group is (decision
  0025), except the group the process serves on: that is refused with the reason, whatever the
  plan says, so a planner's error never tears down the serving communicator.
- Between a change and its commit (:meth:`Cover.pause`, :meth:`Cover.resume`) a plan is only
  recorded: the group the change moves to must not be released before the boundary takes it, and
  the groups' keys change at the commit, so nothing is released, reported or formed until the
  commit has renamed them; the newest plan is then applied in full. Applied as it arrived, a
  plan between a change and its commit releases the group the change is moving to.
- After a commit, a plan for an epoch older than the one committed is recorded and nothing more:
  it was made for the placement the change ended, and the coordinator plans again for the new
  one. Applied, it would form the set the commit has just made serving, at that group's own port
  ("Address already in use").
- A group to form is formed on a thread of its own, so a formation that waits for a member that
  never comes never holds back another. A group meets on the control plane's links: its server
  makes its id, which the coordinator carries to the other members (``GroupId``); no member
  opens a port. A formation superseded before it starts is not started; one superseded while it
  formed is released once formed, and reported released only if the newer plan no longer names
  it; one that fails superseded is not reported.
- The collective swaps by set, so a group is held under each set that needs it, the same group
  each time. A set the plan no longer names lets go of its hold; the group goes only when no set
  of the plan needs it. At a commit the collective renames a taker's id to the rank it took, and
  the cover's own keys follow the same renames.
- A group that fails is reported failed once per generation that names it; trying again is the
  controller's (its backoff), never the process's.

How a group is formed and released is the engine adapter's (``form``, ``release``): the core
holds the decisions, not the communicators.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from everserve.comm.shadow import (
    Member,
    MemberSet,
    is_replacement,
    replaced_rank,
    replacement,
    set_key,
)
from everserve.contract.signals import (
    CoverPlan,
    GroupFailed,
    GroupFormed,
    GroupReleased,
    SetMember,
    SpareRef,
)

log = logging.getLogger(__name__)

DOMAIN = "domain"
FABRIC = "fabric"


@dataclass(frozen=True)
class PlannedGroup:
    """One group of a cover plan: its scope and the members it spans (decision 0035, amended).

    ``spans`` are the communicator's members in its rank order; ``server`` is the member that
    makes the group's id, which the control plane carries to the others. A group is the same
    group whatever sets need it, so a process forms it once.
    """

    scope: str
    spans: MemberSet
    server: Member

    @property
    def key(self) -> tuple[str, MemberSet]:
        """The group's identity: its scope and its spans."""
        return (self.scope, self.spans)


@dataclass(frozen=True)
class PlannedSet:
    """One set of a cover plan: its members and the groups it needs, by index into the plan's."""

    members: MemberSet
    groups: tuple[int, ...]
    purpose: str = ""


@dataclass(frozen=True)
class Plan:
    """An engine's cover plan at a generation, for the placement at a committed epoch."""

    engine: str
    generation: int
    epoch: int
    placement: Mapping[int, str] = field(default_factory=dict)
    groups: tuple[PlannedGroup, ...] = ()
    sets: tuple[PlannedSet, ...] = ()

    def sets_needing(self, index: int) -> list[MemberSet]:
        """The members of every set that needs the plan's group at ``index``."""
        return [s.members for s in self.sets if index in s.groups]


class Reports(Protocol):
    """Where a process's reports go: its daemon link."""

    def formed(self, generation: int, group: PlannedGroup, seconds: float) -> None:
        """``group`` is held, formed now (``seconds`` > 0) or kept (0)."""
        ...

    def failed(self, generation: int, group: PlannedGroup, error: str) -> None:
        """``group`` was not formed, or not released, with the reason."""
        ...

    def released(self, generation: int, scope: str, spans: MemberSet) -> None:
        """The group of ``scope`` over ``spans`` is no longer held."""
        ...


class Held(Protocol):
    """The groups a process holds, by scope and member set: its collective's.

    The collective swaps by set, so a group several sets need is held under each of them, the
    same group every time.
    """

    def held(self) -> dict[tuple[str, MemberSet], Any]:
        """Every group held now, by the set it is held under, the serving ones included."""
        ...

    def serving(self) -> set[tuple[str, MemberSet]]:
        """The sets the process serves on now, in each scope."""
        ...

    def keep(self, scope: str, members: MemberSet, group: Any) -> None:
        """Hold ``group`` under the set ``members`` in ``scope``."""
        ...

    def drop(self, scope: str, members: MemberSet) -> Any:
        """Stop holding the group under the set ``members`` in ``scope``; returns it."""
        ...

    def renames(self) -> Mapping[Member, Member]:
        """Every spare id renamed to the rank it took, at the commits so far."""
        ...


Form = Callable[[PlannedGroup, int], Any]
"""Form one group for this process at the plan's generation and return it (the adapter's: its
own stream, no device synchronize). Raises on a failed formation."""
Release = Callable[[str, MemberSet, Any], None]
"""Start the release of a group no longer held, beside serving (decision 0025)."""


class Cover:
    """This process's part of its engine's cover plan; see the module."""

    def __init__(
        self,
        mine: Callable[[PlannedGroup], bool],
        held: Held,
        form: Form,
        release: Release,
        reports: Reports,
        record: Path | None = None,
        stuck_after_s: float | None = None,
        on_stuck: Callable[[str, float], None] | None = None,
    ) -> None:
        """Apply plans to ``held`` for the process ``mine`` recognises in a group's spans.

        ``record`` is a file the cover's events are written to as they happen, for the run's
        record: each rank's plans, formations, failures and releases, with their generation.

        ``on_stuck`` is called once, with the formation's name and its run in seconds, for a
        formation still running ``stuck_after_s`` after it started: a member that died after the
        go and before it connected leaves NCCL's blocking initialisation waiting for good (NCCL
        2.29.7's bootstrap has no bound). Its thread is left behind as a leaked release is, and
        reported the same way (decision 0025, amended); the coordinator's own deadline fails
        the group and plans it afresh.
        """
        self._mine = mine
        self._held = held
        self._form = form
        self._release = release
        self._reports = reports
        self._record = record
        self._stuck_after_s = stuck_after_s
        self._on_stuck = on_stuck
        self._record_lock = threading.Lock()
        self._lock = threading.Lock()
        self.generation = -1
        """The generation applied, -1 before the first plan."""
        self.committed = 0
        """The epoch this process last committed: a plan for an older one forms nothing."""
        self._plan: Plan | None = None
        self._moved: set[int] = set()
        """Positions whose holder changed across the plans received since the last processed."""
        self._processing = threading.Lock()
        self._open = threading.Event()
        self._open.set()
        self._groups: dict[tuple[str, MemberSet], Any] = {}
        """Every group formed here and not released, by its scope and spans, renamed as the
        collective renames at each commit."""
        self._pending: dict[tuple[str, MemberSet], PlannedGroup] = {}
        """The entry each group is forming at: a later plan naming it again forms nothing."""
        self.events: list[dict[str, Any]] = []
        """What the cover did, in order, for the process's record."""

    # ---------------------------------------------------------------- plans
    def apply(self, plan: Plan) -> bool:
        """Apply ``plan`` as the module says, or record it during a change; False if ignored."""
        with self._lock:
            if plan.generation <= self.generation:
                log.info(
                    "cover plan %d ignored: generation %d applied", plan.generation, self.generation
                )
                return False
            self._moved.update(_moved(self._plan, plan))
            self.generation = plan.generation
            self._plan = plan
        if not self._open.is_set():
            log.info(
                "cover plan %d recorded during a change: applied at its commit", plan.generation
            )
            self._note("deferred", plan.generation, ("", ()))
            return True
        self._process()
        return True

    def _wanted(self, plan: Plan) -> dict[tuple[str, MemberSet], list[MemberSet]]:
        """Each of this process's groups in ``plan``, with the sets it is held under."""
        out: dict[tuple[str, MemberSet], list[MemberSet]] = {}
        for i, g in enumerate(plan.groups):
            if self._mine(g):
                out.setdefault(g.key, []).extend(plan.sets_needing(i))
        return out

    def _rename(self) -> None:
        """The registry's spans renamed as the collective's sets are, at the commits so far."""
        renames = self._held.renames()
        if not renames:
            return
        with self._lock:
            self._groups = {
                (scope, tuple(renames.get(m, m) for m in spans)): g
                for (scope, spans), g in self._groups.items()
            }

    def _process(self) -> None:
        """The newest plan against what is held: release, re-report and queue."""
        with self._processing:
            with self._lock:
                plan = self._plan
                if plan is not None and plan.epoch < self.committed:
                    log.info(
                        "cover plan %d at epoch %d waits: epoch %d committed",
                        plan.generation,
                        plan.epoch,
                        self.committed,
                    )
                    stale = plan
                    plan = None
                else:
                    stale = None
                    moved, self._moved = sorted(self._moved), set()
            if stale is not None:
                self._note("stale_epoch", stale.generation, ("", ()))
            if plan is None:
                return
            self._rename()
            mine = {g.key: g for g in plan.groups if self._mine(g)}
            wanted = self._wanted(plan)
            held = self._held.held()
            serving = {id(held[k]) for k in self._held.serving() if k in held}
            # a group naming a moved position as a rank holds that position's old process
            for key in [
                k
                for k, g in self._groups.items()
                if id(g) not in serving and lost_members(k[1], moved)
            ]:
                self._let_go(plan.generation, key)
            # a group no set of the plan needs is released, the one served on aside
            for key in [k for k in self._groups if k not in mine]:
                if id(self._groups[key]) in serving:
                    self._reports.failed(
                        plan.generation,
                        _unplanned(key),
                        "the process serves on this group; it is not released",
                    )
                    self._note("kept_serving", plan.generation, key)
                    continue
                self._let_go(plan.generation, key)
            # a set the plan no longer names lets go of its hold, the group itself aside; the set
            # served on is never let go, whatever the plan says
            want = {(k[0], s) for k, sets in wanted.items() for s in sets}
            held = self._held.held()
            served = self._held.serving()
            for alias in [a for a in held if a not in want]:
                if id(held[alias]) in serving:
                    known = any(g is held[alias] for g in self._groups.values())
                    if alias in served and not known:  # a known group's refusal is said above
                        self._reports.failed(
                            plan.generation,
                            _unplanned(alias),
                            "the process serves on this group; it is not released",
                        )
                        self._note("kept_serving", plan.generation, alias)
                    continue
                group = self._held.drop(*alias)
                if group is not None and not self._holds(group):
                    # a group held before any plan named it (formed at the engine's start)
                    self._release(alias[0], alias[1], group)
                    self._reports.released(plan.generation, alias[0], alias[1])
                    self._note("released", plan.generation, alias)
            for key, g in mine.items():
                group = self._groups.get(key) or self._adopt(key, wanted.get(key, ()))
                if group is not None:
                    self._hold(key, group, wanted.get(key, ()))
                    self._reports.formed(plan.generation, g, 0.0)
                    self._note("kept", plan.generation, key)
                    continue
                with self._lock:
                    forming = self._pending.get(key) == g
                    if not forming:
                        self._pending[key] = g
                if not forming:
                    self._start_formation(plan.generation, g)
            log.info(
                "cover plan %d at epoch %d applied: %d groups of mine in %d sets, %d held",
                plan.generation,
                plan.epoch,
                len(mine),
                len(want),
                len([k for k in mine if k in self._groups]),
            )

    def _start_formation(self, generation: int, g: PlannedGroup) -> None:
        """Form ``g`` on a thread of its own, watched for running past ``stuck_after_s``."""
        done = threading.Event()

        def run() -> None:
            try:
                self._formation(generation, g)
            finally:
                done.set()

        threading.Thread(target=run, daemon=True, name="EverserveCoverForm").start()
        if self._on_stuck is None or self._stuck_after_s is None:
            return
        t0 = time.monotonic()
        name = f"formation of {g.scope} {set_key(g.spans)}"

        def check() -> None:
            if done.is_set():
                return
            log.warning("%s still running after %.1f s: left behind", name, self._stuck_after_s)
            self._note("stuck", generation, g.key)
            try:
                self._on_stuck(name, time.monotonic() - t0)
            except Exception:  # noqa: BLE001 - a report never stops the cover
                log.exception("%s's leak could not be reported", name)

        timer = threading.Timer(self._stuck_after_s, check)
        timer.daemon = True
        timer.start()

    def _adopt(self, key: tuple[str, MemberSet], sets: Sequence[MemberSet]) -> Any:
        """A group already held under one of the sets that need it, taken into the registry.

        The one served on, formed at the engine's start, is such a group. None when there is none.
        """
        held = self._held.held()
        for s in sets:
            group = held.get((key[0], s))
            if group is not None:
                with self._lock:
                    self._groups[key] = group
                return group
        return None

    def _hold(self, key: tuple[str, MemberSet], group: Any, sets: Sequence[MemberSet]) -> None:
        """Hold ``group`` under every set that needs it."""
        held = self._held.held()
        for s in sets:
            if held.get((key[0], s)) is not group:
                self._held.keep(key[0], s, group)

    def _holds(self, group: Any) -> bool:
        """Whether ``group`` is still held, in the registry or under any set."""
        with self._lock:
            if any(g is group for g in self._groups.values()):
                return True
        return any(g is group for g in self._held.held().values())

    def forget(self, group: Any) -> bool:
        """Take ``group`` out of the registry: released elsewhere (a commit's stale set).

        Returns whether it was there.
        """
        with self._lock:
            keys = [k for k, g in self._groups.items() if g is group]
            for k in keys:
                del self._groups[k]
        return bool(keys)

    def current(self, key: tuple[str, MemberSet]) -> bool:
        """Whether the newest plan names ``key`` as one of this process's groups."""
        with self._lock:
            plan = self._plan
        return plan is not None and any(g.key == key and self._mine(g) for g in plan.groups)

    def _planned(self, g: PlannedGroup) -> bool:
        """Whether the newest plan names ``g`` itself as this process's."""
        with self._lock:
            plan = self._plan
        return plan is not None and g in plan.groups and self._mine(g)

    # ---------------------------------------------------------------- changes
    def pause(self) -> None:
        """A change is claimed: no group is formed until :meth:`resume` (its commit)."""
        self._open.clear()

    def resume(self, committed: int | None = None) -> None:
        """The change is committed at epoch ``committed``, or abandoned (None): formation goes on.

        The newest plan is applied, unless it is for an older epoch than the one committed.
        """
        if committed is not None:
            with self._lock:
                self.committed = max(self.committed, committed)
        self._open.set()
        self._process()

    def close(self) -> None:
        """Let any formation waiting on a change go on: each ends with its own bound."""
        self._open.set()

    # ---------------------------------------------------------------- formations
    def _formation(self, generation: int, g: PlannedGroup) -> None:
        try:
            self._form_one(generation, g)
        finally:
            with self._lock:
                if self._pending.get(g.key) == g:
                    del self._pending[g.key]

    def _form_one(self, generation: int, g: PlannedGroup) -> None:
        self._open.wait()
        if not self._planned(g):
            self._note("superseded", generation, g.key)
            return
        t0 = time.monotonic()
        try:
            group = self._form(g, generation)
        except Exception as e:  # noqa: BLE001 - every failure is reported, with its reason
            log.warning("forming %s %s failed: %s", g.scope, set_key(g.spans), e)
            if self._planned(g):
                self._note("failed", generation, g.key, error=str(e))
                self._reports.failed(generation, g, f"{type(e).__name__}: {e}")
            else:  # a newer plan gave it up while it formed: nothing to report
                self._note("failed_superseded", generation, g.key, error=str(e))
            return
        # unrounded: a formed group's time is never 0, which marks a kept one
        seconds = max(time.monotonic() - t0, 1e-9)
        if not self._planned(g):
            # a newer plan dropped it while it formed: never held, so released at once, and
            # reported released only if no newer plan names it
            self._release(g.scope, g.spans, group)
            if not self.current(g.key):
                self._reports.released(self.generation, g.scope, g.spans)
            self._note("formed_superseded", generation, g.key, seconds=round(seconds, 3))
            return
        with self._lock:
            self._groups[g.key] = group
            plan = self._plan
        self._hold(g.key, group, self._wanted(plan).get(g.key, ()) if plan else ())
        # the record first: a reader that has the report finds it in the record too
        self._note("formed", self.generation, g.key, seconds=round(seconds, 3))
        self._reports.formed(self.generation, g, seconds)

    # ---------------------------------------------------------------- helpers
    def _let_go(self, generation: int, key: tuple[str, MemberSet]) -> None:
        """Release the group ``key``: out of the registry and from under every set."""
        scope, spans = key
        with self._lock:
            group = self._groups.pop(key, None)
        if group is not None:
            for alias, g in self._held.held().items():
                if g is group:
                    self._held.drop(*alias)
            self._release(scope, spans, group)
        self._reports.released(generation, scope, spans)
        self._note("released", generation, key)

    def _note(self, kind: str, generation: int, key: tuple[str, MemberSet], **extra: Any) -> None:
        self.events.append(
            {
                "kind": kind,
                "generation": generation,
                "scope": key[0],
                "spans": set_key(key[1]),
                "at": time.time(),
                **extra,
            }
        )
        if self._record is not None:
            self._write_record()

    def _write_record(self) -> None:
        """Every event so far, written beside and renamed over: a reader never sees half."""
        assert self._record is not None
        with self._record_lock:
            self._record.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._record.with_suffix(".json.part")
            tmp.write_text(json.dumps(self.events, default=str))
            os.replace(tmp, self._record)


def _unplanned(key: tuple[str, MemberSet]) -> PlannedGroup:
    """A held group the plan does not name, as a group to report a refusal about."""
    scope, spans = key
    return PlannedGroup(scope, spans, spans[0] if spans else 0)


def _moved(previous: Plan | None, plan: Plan) -> list[int]:
    """The positions whose holder changed from ``previous`` to ``plan``, a plan at a newer epoch."""
    if previous is None or plan.epoch <= previous.epoch:
        return []
    return [r for r, holder in plan.placement.items() if previous.placement.get(r) != holder]


def lost_members(members: MemberSet, lost: Sequence[int]) -> bool:
    """Whether a group's members name, as a rank, a position whose process the change moved."""
    gone = set(lost)
    return any(isinstance(m, int) and m in gone for m in members)


# ---------------------------------------------------------------- the wire's plan and reports


def member_id(m: SetMember) -> Member:
    """A plan's member as a process keys it: a rank, or the spare's id naming its process."""
    if isinstance(m, SpareRef):
        return replacement(m.for_rank, m.node, m.device)
    return int(m)


def set_member(m: Member) -> SetMember:
    """A process's member id as the plan names it.

    Raises:
        ValueError: for a spare's id that does not name its process (``R1``, not
            ``R1@node-d:0``): such an id never comes from a plan.
    """
    if not is_replacement(m):
        return int(m)
    _, _, where = str(m)[1:].partition("@")
    node, _, device = where.rpartition(":")
    if not node:
        raise ValueError(f"{m!r} names no spare's process")
    return SpareRef(replaced_rank(m), node, int(device))


def plan_from(plan: CoverPlan) -> Plan:
    """The wire's plan as this process applies it."""
    return Plan(
        engine=plan.engine,
        generation=plan.generation,
        epoch=plan.epoch,
        placement=dict(plan.placement),
        groups=tuple(
            PlannedGroup(
                scope=g.scope,
                spans=tuple(member_id(m) for m in g.spans),
                server=member_id(g.server),
            )
            for g in plan.groups
        ),
        sets=tuple(
            PlannedSet(
                members=tuple(member_id(m) for m in s.members),
                groups=tuple(s.groups),
                purpose=s.purpose,
            )
            for s in plan.sets
        ),
    )


class LinkReports:
    """:class:`Reports` sent over the process's daemon link, as the member ``me()`` names."""

    def __init__(self, engine: str, me: Callable[[], Member], link: Any) -> None:
        """``me`` is read at each report: a spare's id becomes its rank at its commit."""
        self._engine = engine
        self._me = me
        self._link = link

    def _common(self, generation: int, scope: str, spans: MemberSet) -> dict[str, Any]:
        return {
            "engine": self._engine,
            "generation": generation,
            "spans": tuple(set_member(m) for m in spans),
            "scope": scope,
            "member": set_member(self._me()),
        }

    def formed(self, generation: int, group: PlannedGroup, seconds: float) -> None:
        """Report ``group`` held at ``generation``."""
        self._link.report(
            GroupFormed(seconds=seconds, **self._common(generation, group.scope, group.spans))
        )

    def failed(self, generation: int, group: PlannedGroup, error: str) -> None:
        """Report ``group`` not formed, or not released, with ``error``."""
        self._link.report(
            GroupFailed(error=error, **self._common(generation, group.scope, group.spans))
        )

    def released(self, generation: int, scope: str, spans: MemberSet) -> None:
        """Report the group no longer held."""
        self._link.report(GroupReleased(**self._common(generation, scope, spans)))
