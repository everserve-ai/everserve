# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The one object a model's collectives go through.

It owns the current group, the pre-formed shadow groups for every member set the policy allows,
the indirected handle external captured code reads, and the slot collective when the device has
one. A membership change is :meth:`Collective.change`: abort the slot, point at the pre-formed
group for the new member set, program the slot, resume. No rendezvous, no NCCL init and no graph
capture on the critical path (decision 0005). :meth:`Collective.shrink` is the fallback for a
member set that was not pre-formed.

Dispatch by size: a tensor the slot collective can take goes to the slot kernel, which is what a
FULL CUDA graph captures; anything larger goes to the current group, resolved at call time and
never captured with a communicator.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

from everserve.comm.cover import DOMAIN, FABRIC, lost_members
from everserve.comm.group import ShrinkableGroup, ShrinkMethod
from everserve.comm.handle import GroupHandle, LocalStandIn
from everserve.comm.layout import Scope
from everserve.comm.shadow import (
    Member,
    MemberSet,
    is_replacement,
    replaced_rank,
    replacement,
    set_key,
)
from everserve.comm.slot import SlotCollective

log = logging.getLogger(__name__)


class CollectiveMoved(Exception):
    """The collective changed epoch between a caller's check and its call; the step must retry."""


def ranks_of(members: MemberSet) -> list[int]:
    """The rank ids a member set spans; the spare takes the id of the position it stands in.

    Replacement sets are full-world sets in rank order with the spare at the replaced rank's
    position, so that position is the id. Survivor sets contain no spare.
    """
    return [m if isinstance(m, int) else i for i, m in enumerate(members)]


class GroupFamily:
    """The groups of one scope beside the collective's own: a current group and its shadows.

    The collective's own groups carry the tensor-parallel all-reduce inside a domain. An engine
    wider than a domain also sends and receives across the fabric, through a group of a second
    scope (:class:`~everserve.comm.layout.Scope`). A family holds that scope's current group and
    one pre-formed group per member set, keyed by the full member set as the collective's shadows
    are, and moves with the collective at a change: the same set, the same epoch, a pointer swap.
    What the group carries is the adapter's (vLLM's pipeline handle reads it at call time).
    """

    def __init__(
        self,
        scope: Scope,
        live: ShrinkableGroup | LocalStandIn,
        shadows: Mapping[MemberSet, ShrinkableGroup | LocalStandIn] | None = None,
    ) -> None:
        """Start on ``live``, with ``shadows`` pre-formed beside it, keyed by full member set."""
        self.scope = scope
        self.current: ShrinkableGroup | LocalStandIn = live
        self._shadows: dict[MemberSet, ShrinkableGroup | LocalStandIn] = dict(shadows or {})

    @property
    def shadow_sets(self) -> list[MemberSet]:
        """The member sets with a pre-formed group."""
        return list(self._shadows)

    def preform(self, shadows: Mapping[MemberSet, ShrinkableGroup | LocalStandIn]) -> None:
        """Add pre-formed groups after start."""
        self._shadows.update({tuple(s): g for s, g in shadows.items()})

    def drop(
        self, sets: Sequence[MemberSet]
    ) -> list[tuple[MemberSet, ShrinkableGroup | LocalStandIn]]:
        """Set aside the pre-formed groups for ``sets``; returns them. A set not held is ignored."""
        out = []
        for s in sets:
            g = self._shadows.pop(tuple(s), None)
            if g is not None:
                out.append((tuple(s), g))
        return out

    def items(self) -> list[tuple[MemberSet, ShrinkableGroup | LocalStandIn]]:
        """Every pre-formed group, by the member set it is kept under."""
        return list(self._shadows.items())

    def rename(self, renamed: Callable[[MemberSet], MemberSet]) -> None:
        """Key every pre-formed group by ``renamed(set)``, at a commit."""
        self._shadows = {renamed(s): g for s, g in self._shadows.items()}

    def has(self, members: MemberSet) -> bool:
        """Whether a change to ``members`` is a pointer swap for this scope."""
        return tuple(members) in self._shadows

    def swap(self, members: MemberSet) -> ShrinkableGroup | LocalStandIn:
        """Move to the pre-formed group for ``members``; returns it.

        Raises:
            KeyError: if ``members`` has no pre-formed group in this scope.
        """
        group = self._shadows[tuple(members)]
        self.current = group
        return group


class Collective:
    """Current group, shadow groups, the handle and the slot, for one process."""

    def __init__(
        self,
        me: Member,
        live: ShrinkableGroup | LocalStandIn,
        shadows: Mapping[MemberSet, dist.ProcessGroup | ShrinkableGroup | LocalStandIn],
        slot: SlotCollective | None = None,
        backend: str | None = None,
        store: dist.Store | None = None,
        fabric: GroupFamily | None = None,
        stands_for: Sequence[Member] = (),
    ) -> None:
        """Start on ``live`` with ``shadows`` pre-formed beside it.

        Args:
            me: this process's id in member sets: its rank, or the spare's id.
            live: the group collectives run on now; a stand-in for a spare with no peers yet.
            shadows: pre-formed groups by member set, as process groups or already wrapped.
                Each must contain ``me``.
            slot: the slot collective on this device, or None on a device without one.
            backend: the backend shadow groups use; taken from ``live`` when it is a group.
            store: the store a fallback shrink forms its group over; taken from ``live`` when
                it is a group.
            fabric: the groups that cross the fabric (the pipeline), or None for an engine
                inside one domain. They move with the collective at every change.
            stands_for: the further ids of a spare that stands for any of several positions
                (decision 0030): it pre-forms the sets of each, and :meth:`take_identity`
                names the one it takes.
        """
        self.stands_for: tuple[Member, ...] = tuple(dict.fromkeys((me, *stands_for)))
        for s in shadows:
            if not self._mine(s):
                raise ValueError(f"shadow set {set_key(s)} does not contain {self.stands_for}")
        self.me = me
        self.current: ShrinkableGroup | LocalStandIn = live
        self.members: MemberSet = tuple(live.ranks) if isinstance(live, ShrinkableGroup) else ()
        self.handle = GroupHandle(live.group if isinstance(live, ShrinkableGroup) else live)
        self.slot = slot
        self.epoch = 0
        self.backend = backend or (live.backend if isinstance(live, ShrinkableGroup) else "nccl")
        self.store = store or (live.store if isinstance(live, ShrinkableGroup) else None)
        self._shadows = dict(shadows)
        self.fabric = fabric
        self._lock = threading.Lock()
        self._renamed: dict[Member, Member] = {}
        """Every spare id a commit renamed to its rank (:meth:`rename_at_commit`): a group formed
        under an old id while the commit ran is kept under the new one."""

    # ---------------------------------------------------------------- member sets
    @property
    def shadow_sets(self) -> list[MemberSet]:
        """The member sets with a pre-formed group."""
        return list(self._shadows)

    def preform(
        self, shadows: Mapping[MemberSet, dist.ProcessGroup | ShrinkableGroup | LocalStandIn]
    ) -> None:
        """Add pre-formed groups after start, for a process that forms them once it has peers.

        Raises:
            ValueError: if a set does not contain this process.
        """
        for s in shadows:
            if not self._mine(s):
                raise ValueError(f"shadow set {set_key(s)} does not contain {self.stands_for}")
        with self._lock:
            self._shadows.update(shadows)

    def _mine(self, members: MemberSet) -> bool:
        return any(m in members for m in self.stands_for)

    def take_identity(
        self, me: Member
    ) -> list[tuple[str, MemberSet, ShrinkableGroup | LocalStandIn]]:
        """Become ``me``, one of the ids this process stands for; returns the groups set aside.

        A spare standing for several positions takes one at a change (decision 0030): the sets
        it pre-formed as another position's spare are no longer its own, and leave both scopes.
        Each comes back as (scope, set, group) for the caller to release (decision 0025).

        Raises:
            ValueError: for an id this process does not stand for.
        """
        if me not in self.stands_for:
            raise ValueError(
                f"{me} is not one of the ids this process stands for {self.stands_for}"
            )
        dropped: list[tuple[str, MemberSet, ShrinkableGroup | LocalStandIn]] = []
        with self._lock:
            for s in [s for s in self._shadows if me not in s]:
                dropped.append(("domain", s, self._shadows.pop(s)))
            if self.fabric is not None:
                gone = [s for s in self.fabric.shadow_sets if me not in s]
                dropped += [("fabric", s, g) for s, g in self.fabric.drop(gone)]
            self.me = me
            self.stands_for = (me,)
        return dropped

    def has_shadow(self, members: MemberSet) -> bool:
        """Whether a change to ``members`` is a pointer swap, in every scope."""
        members = tuple(members)
        return members in self._shadows and (self.fabric is None or self.fabric.has(members))

    def survivor_set(self, lost: Sequence[int]) -> MemberSet:
        """The current members without ``lost``."""
        return tuple(m for m in self.members if m not in lost)

    def replacement_set(self, lost: Sequence[int]) -> MemberSet:
        """The current members with each lost rank's spare in its place: the held set that is.

        A spare's id names its process when a cover plan formed the set (``R1@node-d:0``,
        decision 0035), so the set is found among those held, not composed from the rank: the
        one whose member at each lost position is a spare for that position and whose other
        members are the current ones. Without any held set of that shape, the composed set
        (``R1``), which a change to it then refuses as not pre-formed.

        Raises:
            KeyError: when two held sets fit, two spares for one position: the change must
                name the spare that takes it.
        """
        gone = set(lost)

        def fits(s: MemberSet) -> bool:
            return len(s) == len(self.members) and all(
                m == c
                if not (isinstance(c, int) and c in gone)
                else is_replacement(m) and replaced_rank(m) == c
                for m, c in zip(s, self.members, strict=True)
            )

        with self._lock:
            held = list(self._shadows)
            if self.fabric is not None:
                held += self.fabric.shadow_sets
        found = list(dict.fromkeys(s for s in held if fits(s)))
        if len(found) > 1:
            raise KeyError(
                f"several held sets replace ranks {sorted(gone)}: "
                f"{[set_key(s) for s in found]}; the change must name the spare that takes them"
            )
        if found:
            return found[0]
        return tuple(
            replacement(m) if isinstance(m, int) and m in lost else m for m in self.members
        )

    # ---------------------------------------------------------------- cover (decision 0035)
    def held(self) -> dict[tuple[str, MemberSet], Any]:
        """Every group held, by scope and member set, the serving ones included.

        A local stand-in is no group (a scope of one member, or a spare with no peers yet), so
        it is not listed: a plan never names it, and its absence is not a refusal.
        """
        with self._lock:
            out: dict[tuple[str, MemberSet], Any] = {
                (DOMAIN, s): g for s, g in self._shadows.items() if not isinstance(g, LocalStandIn)
            }
            if self.members and not isinstance(self.current, LocalStandIn):
                out[(DOMAIN, self.members)] = self.current
            if self.fabric is not None:
                out.update(
                    {
                        (FABRIC, s): g
                        for s, g in self.fabric.items()
                        if not isinstance(g, LocalStandIn)
                    }
                )
                if self.members and not isinstance(self.fabric.current, LocalStandIn):
                    out[(FABRIC, self.members)] = self.fabric.current
        return out

    def serving(self) -> set[tuple[str, MemberSet]]:
        """The groups this process serves on now: the current member set's, in each scope."""
        with self._lock:
            if not self.members:
                return set()
            scopes = (DOMAIN, FABRIC) if self.fabric is not None else (DOMAIN,)
            return {(scope, self.members) for scope in scopes}

    def keep(self, scope: str, members: MemberSet, group: Any) -> None:
        """Hold ``group``, newly formed, under ``members`` with every rename since applied.

        Raises:
            ValueError: for a set without this process, or a fabric group without a fabric.
        """
        with self._lock:
            members = tuple(self._renamed.get(m, m) for m in members)
            if not self._mine(members):
                raise ValueError(f"set {set_key(members)} does not contain {self.stands_for}")
            if scope == FABRIC:
                if self.fabric is None:
                    raise ValueError("a fabric group for an engine inside one domain")
                self.fabric.preform({members: group})
            else:
                self._shadows[members] = group

    def renames(self) -> dict[Member, Member]:
        """Every spare id renamed to the rank it took, at the commits so far."""
        with self._lock:
            return dict(self._renamed)

    def drop(self, scope: str, members: MemberSet) -> Any:
        """Stop holding the group of ``scope`` over ``members``; returns it, or None."""
        with self._lock:
            if scope == FABRIC:
                if self.fabric is None:
                    return None
                return next((g for _, g in self.fabric.drop([members])), None)
            return self._shadows.pop(tuple(members), None)

    def name_process(self, node: str, device: int) -> None:
        """Name this spare's ids by its process: ``R1`` becomes ``R1@node:device``.

        A cover plan names a spare by its node and the device it registered with, so two spares
        for one position are two sets (decision 0035); a process held to a plan keys its sets
        the same way. A serving rank's id is its rank and is not changed.

        Raises:
            ValueError: once groups are held under the old ids.
        """
        with self._lock:
            if self._shadows or (self.fabric is not None and self.fabric.shadow_sets):
                raise ValueError("a spare's ids are named before it holds any group")

            def named(m: Member) -> Member:
                if is_replacement(m) and "@" not in str(m):
                    return replacement(replaced_rank(m), node, device)
                return m

            self.me = named(self.me)
            self.stands_for = tuple(named(m) for m in self.stands_for)

    def rename_at_commit(self, lost: Sequence[int]) -> list[tuple[str, MemberSet, Any]]:
        """After a change to the current members: each taker becomes the rank it took.

        Every held set that names a lost position as a rank holds that position's old process,
        so it is set aside and returned, as (scope, set, group), for the caller to release
        (decision 0025). Then each spare id in the current members that stands for a lost rank
        is renamed to that rank in every key, the current members and this process's own id. A
        set naming another spare for that rank keeps its id: it is another process (decision
        0035).
        """
        gone = set(lost)
        with self._lock:
            takers = {
                m: replaced_rank(m)
                for m in self.members
                if is_replacement(m) and replaced_rank(m) in gone
            }
            stale: list[tuple[str, MemberSet, Any]] = []
            for s in [s for s in self._shadows if s != self.members and lost_members(s, gone)]:
                stale.append((DOMAIN, s, self._shadows.pop(s)))
            if self.fabric is not None:
                old = [
                    s
                    for s in self.fabric.shadow_sets
                    if s != self.members and lost_members(s, gone)
                ]
                stale += [(FABRIC, s, g) for s, g in self.fabric.drop(old)]
            self._renamed.update(takers)

            def renamed(s: MemberSet) -> MemberSet:
                return tuple(takers.get(m, m) for m in s)

            self._shadows = {renamed(s): g for s, g in self._shadows.items()}
            if self.fabric is not None:
                self.fabric.rename(renamed)
            self.members = renamed(self.members)
            self.me = takers.get(self.me, self.me)
            self.stands_for = tuple(takers.get(m, m) for m in self.stands_for)
        if takers:
            log.info(
                "renamed at commit: %s; %d stale groups set aside",
                {str(k): v for k, v in takers.items()},
                len(stale),
            )
        return stale

    # ---------------------------------------------------------------- collectives
    def all_reduce_into(
        self, inp: torch.Tensor, out: torch.Tensor, epoch: int | None = None
    ) -> None:
        """``out = sum over the members of inp``, in place; the slot kernel when it applies.

        With ``epoch`` the call is pinned to that membership: the group is chosen under the lock
        and the call refuses if the collective has moved since. A step's collectives must never
        run on a member set the step did not start under.

        Raises:
            CollectiveMoved: when ``epoch`` is given and the collective's epoch differs.
        """
        with self._lock:
            if epoch is not None and epoch != self.epoch:
                raise CollectiveMoved(f"epoch {epoch} asked, collective is at {self.epoch}")
            current = self.current
            slot = self.slot
        if slot is not None and slot.can_take(inp):
            slot.all_reduce_into(inp, out)
            return
        if out is not inp:
            out.copy_(inp)
        if not isinstance(current, LocalStandIn):
            current.all_reduce(out)

    def all_reduce(self, t: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        """Sum ``t`` in place over the members; see :meth:`all_reduce_into` for ``epoch``."""
        self.all_reduce_into(t, t, epoch)
        return t

    # ---------------------------------------------------------------- changes
    def refuse(self) -> None:
        """Refuse the current set from now. Safe to call twice.

        Members spinning in the slot kernel on a dead peer are released by its abort word.
        Nothing here is freed: the core holds no communicator. An adapter that does starts the
        release of every refused group beside the change and finishes it, bounded, at the
        boundary (decision 0025); the change never waits for it.
        """
        if self.slot is not None:
            self.slot.abort()

    def change(self, members: MemberSet, epoch: int) -> None:
        """Move to the pre-formed group for ``members`` at ``epoch``.

        Raises:
            KeyError: if ``members`` has no pre-formed group; see :meth:`shrink`.
        """
        members = tuple(members)
        shadow = self._shadows[members]
        group: ShrinkableGroup | LocalStandIn
        if isinstance(shadow, ShrinkableGroup | LocalStandIn):
            group = shadow  # a stand-in is a spare's "no peers yet"; never wrap it as a group
        else:
            group = ShrinkableGroup(
                shadow, ranks_of(members), self.backend, self.store, ShrinkMethod.SHADOW
            )
        self.change_to_group(members, group, epoch)

    def shrink(self, lost: Sequence[int], epoch: int) -> ShrinkMethod:
        """Fallback when the survivor set was not pre-formed: shrink the current group.

        Raises:
            RuntimeError: if there is no current group to shrink (a stand-in).
        """
        if not isinstance(self.current, ShrinkableGroup):
            raise RuntimeError("cannot shrink a stand-in group")
        if not isinstance(self.me, int):
            raise RuntimeError("a spare that has not joined cannot shrink")
        new = self.current.shrink(lost, self.me, epoch)
        self.change_to_group(self.survivor_set(lost), new, epoch)
        return new.method

    def change_to_group(
        self, members: MemberSet, group: ShrinkableGroup | LocalStandIn, epoch: int
    ) -> None:
        """Move to ``group`` over ``members`` at ``epoch``, whether or not it was pre-formed.

        The fabric family, if any, moves to its pre-formed group for ``members`` at the same
        time, so a step never runs its domain and fabric collectives on different member sets.

        Raises:
            KeyError: if the fabric family has no pre-formed group for ``members``.
        """
        members = tuple(members)
        with self._lock:
            if self.fabric is not None:
                self.fabric.swap(members)
            self.current = group
            self.members = members
            self.epoch = epoch
            self.handle.swap(group.group if isinstance(group, ShrinkableGroup) else group)
            if self.slot is not None:
                self.slot.set_members(ranks_of(members), epoch)
                self.slot.resume()
        log.info("membership %s at epoch %d", set_key(members), epoch)
