# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The engine's layout over its parallel dimensions: which members of a set each rank's groups span.

A member set names every position of the engine (`docs/MULTINODE.md` section 2). A rank's
tensor-parallel collectives run over the members of its stage; its pipeline send and receive run
over the members of its column. :class:`ParallelLayout` is the one place that mapping lives: the
runtime and the collective see member sets, the adapter asks the layout which members its groups
span, and a collective's scope (:class:`Scope`) says whether it stays inside a domain or crosses
the fabric (section 4).

The layout is vLLM's: with data and context parallel 1, rank ``r`` is at stage ``r // tp`` and
column ``r % tp``. A spare's id (``R<rank>``) sits at the replaced rank's position.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from everserve.comm.shadow import Member, MemberSet, is_replacement, replaced_rank


class Scope(Enum):
    """Where a collective's members are, which decides what can carry it."""

    DOMAIN = "domain"
    """Inside one domain (a node's NVLink or PCIe): the slot collective applies."""
    FABRIC = "fabric"
    """Across domains, over the fabric: a handle read at call time, never captured."""


def position(member: Member) -> int:
    """The rank position a member holds: its rank, or the rank a spare stands in for."""
    return replaced_rank(member) if is_replacement(member) else int(member)


@dataclass(frozen=True)
class ParallelLayout:
    """The engine's tensor-parallel and pipeline-parallel sizes, and the mapping they imply."""

    tensor_parallel: int
    pipeline_parallel: int = 1

    def __post_init__(self) -> None:
        """Both sizes are at least one."""
        if self.tensor_parallel < 1 or self.pipeline_parallel < 1:
            raise ValueError(f"a layout needs sizes of at least 1: {self}")

    @property
    def world(self) -> int:
        """Ranks in the engine."""
        return self.tensor_parallel * self.pipeline_parallel

    def stage(self, member: Member) -> int:
        """The pipeline stage a member's position is in."""
        return position(member) // self.tensor_parallel

    def column(self, member: Member) -> int:
        """The member's index within its stage: its tensor-parallel rank, and its slot row."""
        return position(member) % self.tensor_parallel

    def tensor_members(self, members: MemberSet, me: Member) -> MemberSet:
        """The members of ``me``'s stage in ``members``, in the set's order."""
        stage = self.stage(me)
        return tuple(m for m in members if self.stage(m) == stage)

    def pipeline_members(self, members: MemberSet, me: Member) -> MemberSet:
        """The members of ``me``'s column in ``members``, in the set's order."""
        column = self.column(me)
        return tuple(m for m in members if self.column(m) == column)

    def members_of(self, scope: Scope, members: MemberSet, me: Member) -> MemberSet:
        """The members ``me``'s collective of ``scope`` runs over, within ``members``."""
        if scope is Scope.DOMAIN:
            return self.tensor_members(members, me)
        return self.pipeline_members(members, me)
