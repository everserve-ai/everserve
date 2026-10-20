# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The deployment's policy: which outcomes it may use, and the choice for one event.

The policy document is the protobuf ``everserve.policy.v1.Policy`` (``proto/policy.proto``),
written as JSON by the operator and read by the controller. This module gives the runtime side a
typed view of it and the one pure decision the controller makes per event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from google.protobuf import json_format

from everserve.comm.shadow import MemberSet, member_sets
from everserve.outcome import Outcome
from everserve.proto import policy_pb2, signals_pb2

__all__ = [
    "Engine",
    "Loading",
    "Member",
    "Outcome",
    "Policy",
    "choose_outcome",
    "from_message",
    "load",
    "sets_to_preform",
]


_FROM_PB = {
    signals_pb2.OUTCOME_REPLACE: Outcome.REPLACE,
    signals_pb2.OUTCOME_SHRINK: Outcome.SHRINK,
    signals_pb2.OUTCOME_REBUILD: Outcome.REBUILD,
}
TO_PB = {v: k for k, v in _FROM_PB.items()}
"""Outcome -> wire enum value."""


class Guarantee(str, Enum):
    """What an engine promises its sessions across a change (decision 0030)."""

    EXACT = "exact"
    """Tokens and K/V rows equal an uninterrupted engine's: deterministic kernels, and a process
    on another compile cache than its position's is refused."""
    CONTINUOUS = "continuous"
    """Sessions continue from their committed positions; numerics are stated, not promised."""


class Loading(str, Enum):
    """How a spare comes to hold a rank's state."""

    PRELOADED = "preloaded"
    ON_NOTICE = "on_notice"


@dataclass(frozen=True)
class Member:
    """One GPU slot."""

    node: str
    device: int


@dataclass(frozen=True)
class SpareSlot:
    """One spare's place: the node and device it runs on and the rank it stands in for.

    ``rank`` is None for a spare that stands for any position of the engine (decision 0030).
    """

    node: str
    device: int
    rank: int | None


@dataclass(frozen=True)
class SpareLaunch:
    """How a node's daemon starts a spare process (decision 0024).

    ``{engine}``, ``{rank}``, ``{device}``, ``{node}`` and ``{socket}`` are filled in, in the
    command and the environment.
    """

    command: tuple[str, ...]
    env: Mapping[str, str]


@dataclass(frozen=True)
class Engine:
    """One engine the policy covers."""

    name: str
    members: tuple[Member, ...]
    """Each rank's node and device, in rank order, when the policy pins them; empty for an engine
    placed by the processes that register (decision 0041), whose placement is status."""
    tensor_parallel: int
    expert_parallel: int
    pipeline_parallel: int
    placement_independent_model: bool
    data_parallel: int = 1
    head_on_demand: bool = False
    """The head runs on capacity a spot reclaim never takes (decision 0041)."""
    spare_nodes: tuple[str, ...] = ()
    """The nodes that hold this engine's spares (`docs/MULTINODE.md` section 5); empty when
    the spares are on the engine's own nodes."""
    replica_nodes: tuple[str, ...] = ()
    """The nodes whose stores take this engine's rows (decision 0030); empty for the engine's
    own member nodes, each rank's rows going to its siblings."""
    guarantee: Guarantee = Guarantee.EXACT
    """The engine's guarantee class (decision 0030); the adapter compiles with deterministic
    kernels for the exact class only."""
    fabric_collectives: bool = False
    """Whether the engine's captured collectives can span the fabric (decision 0022)."""
    spares: tuple[SpareSlot, ...] = ()
    """Where the daemons place the engine's spares (decision 0024); empty when the engine
    launches its own."""
    spare_launch: SpareLaunch | None = None
    """How a daemon starts one of them; None when the engine launches its own."""
    release_bound_ms: int | None = None
    """How long a boundary waits for a refused group's release before leaving it behind
    (decision 0025); None for the adapter's default."""

    @property
    def spare_ranks(self) -> tuple[int, ...]:
        """The ranks the daemons' spares stand in for, in slot order."""
        return tuple(s.rank for s in self.spares if s.rank is not None)

    def spare_node(self, rank: int) -> str | None:
        """The node holding the daemons' spare for ``rank``, or None if none is placed."""
        return next((s.node for s in self.spares if s.rank == rank), None)

    @property
    def world_size(self) -> int:
        """Ranks in the engine: its shape's, tensor x pipeline x data parallel."""
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel


@dataclass(frozen=True)
class Policy:
    """A typed view of the policy document."""

    version: str
    outcomes_in_order: tuple[Outcome, ...]
    spares_per_engine: int
    spare_loading: Loading
    max_concurrent_losses: int
    notice_min_ms: int
    engines: tuple[Engine, ...] = field(default_factory=tuple)
    cover_planned: bool = False
    """The coordinator plans every engine's pre-formed groups and delivers them (decision 0035);
    a process then forms what its plan names, never the policy's sets."""
    cover_form_timeout_ms: int = 30_000
    """How long a member's formation of a planned group may run. The coordinator fails a group not
    formed by every member within it (plus the transit), and a process whose formation runs past it
    reports the thread left behind (decision 0025, amended)."""

    def engine(self, name: str) -> Engine:
        """The engine called ``name``.

        Raises:
            KeyError: if the policy has no such engine.
        """
        for e in self.engines:
            if e.name == name:
                return e
        raise KeyError(name)

    def notice_lead_s(self, wanted_s: float) -> float:
        """How far ahead a leaving process puts its notice's deadline: ``wanted_s``, or longer.

        The coordinator takes a notice whose deadline is at least :attr:`notice_min_ms` ahead as
        it receives it, and reads a shorter one as no notice: the rank is then lost when its
        process ends. The lead is never under the minimum and :data:`NOTICE_TRANSIT_S`, the
        notice's way from the process to the coordinator.
        """
        return max(wanted_s, self.notice_min_ms / 1e3 + NOTICE_TRANSIT_S)


NOTICE_TRANSIT_S = 0.5
"""What a notice's lead allows for its way up (the process's link, its daemon, the coordinator's
link): a deadline at the policy's minimum when sent would be under it when read."""


def from_message(msg: policy_pb2.Policy) -> Policy:
    """Build the typed view from the protobuf message.

    Raises:
        ValueError: for an unspecified outcome or loading rule, no outcomes at all, an engine
            whose pinned members do not number its world, or one that pins none without cover.
    """
    outcomes = []
    for o in msg.outcomes_in_order:
        if o not in _FROM_PB:
            raise ValueError(f"unspecified outcome in policy {msg.version!r}")
        outcomes.append(_FROM_PB[o])
    if not outcomes:
        raise ValueError(f"policy {msg.version!r} allows no outcome")
    if msg.spares.loading == policy_pb2.SparePool.LOADING_ON_NOTICE:
        loading = Loading.ON_NOTICE
    elif msg.spares.loading == policy_pb2.SparePool.LOADING_PRELOADED:
        loading = Loading.PRELOADED
    else:
        raise ValueError(f"policy {msg.version!r} has no spare loading rule")
    engines = tuple(_engine(msg.version, e) for e in msg.engines)
    for e in engines:
        if not e.members and not msg.cover.enabled:
            # its spares stand after bring-up, so only the sets cover plans as they stand can
            # replace a rank (decision 0035); the coordinator refuses the same policy
            raise ValueError(
                f"engine {e.name} pins no members, so the processes that register place it, and "
                "it is replaced only through the sets cover plans as its spares stand: enable cover"
            )
    return Policy(
        version=msg.version,
        outcomes_in_order=tuple(outcomes),
        spares_per_engine=msg.spares.per_engine,
        spare_loading=loading,
        max_concurrent_losses=msg.max_concurrent_losses,
        notice_min_ms=msg.notice_min_ms,
        engines=engines,
        cover_planned=msg.cover.enabled,
        cover_form_timeout_ms=msg.cover.form_timeout_ms or 30_000,  # unset (0): the default
    )


def _engine(version: str, e: policy_pb2.Engine) -> Engine:
    """One engine's typed view: an unset width counts as one.

    Raises:
        ValueError: for members that do not number the engine's world.
    """
    tp, pp, dp = (max(1, w) for w in (e.tensor_parallel, e.pipeline_parallel, e.data_parallel))
    if e.members and len(e.members) != tp * pp * dp:
        raise ValueError(
            f"policy {version!r}: engine {e.name!r} pins {len(e.members)} members for a world of "
            f"{tp * pp * dp} (tensor {tp} x pipeline {pp} x data {dp})"
        )
    return Engine(
        name=e.name,
        members=tuple(Member(m.node, m.device) for m in e.members),
        tensor_parallel=tp,
        expert_parallel=e.expert_parallel,
        pipeline_parallel=pp,
        data_parallel=dp,
        head_on_demand=e.head_on_demand,
        placement_independent_model=e.placement_independent_model,
        spare_nodes=tuple(e.spare_nodes),
        replica_nodes=tuple(e.replica_nodes),
        fabric_collectives=e.fabric_collectives,
        spares=tuple(
            SpareSlot(s.node, s.device, s.rank if s.HasField("rank") else None) for s in e.spares
        ),
        spare_launch=(
            SpareLaunch(tuple(e.spare_launch.command), dict(e.spare_launch.env))
            if e.spare_launch.command
            else None
        ),
        release_bound_ms=(e.release_bound_ms if e.HasField("release_bound_ms") else None),
        guarantee=(
            Guarantee.CONTINUOUS
            if e.guarantee == policy_pb2.GUARANTEE_CONTINUOUS
            else Guarantee.EXACT
        ),
    )


def from_bytes(data: bytes) -> Policy:
    """The typed view of an encoded ``Policy``, as the engine's link carries it.

    Raises:
        ValueError: for a document :func:`from_message` refuses, or bytes that are not one.
    """
    from google.protobuf.message import DecodeError

    msg = policy_pb2.Policy()
    try:
        msg.ParseFromString(data)
    except DecodeError as e:
        raise ValueError(f"an engine policy that does not decode: {e}") from e
    return from_message(msg)


def load(path: str | Path) -> Policy:
    """Read a policy document (protobuf JSON) from ``path``."""
    msg = json_format.Parse(Path(path).read_text(), policy_pb2.Policy())
    return from_message(msg)


def choose_outcome(
    policy: Policy, *, spare_ready: bool, placement_independent_model: bool
) -> Outcome:
    """The first outcome in the policy's order whose preconditions hold right now.

    REPLACE needs a ready spare. SHRINK needs a placement-independent model. REBUILD needs
    nothing. A policy that lists only outcomes whose preconditions fail falls back to REBUILD,
    because an engine that can rebuild must not die for lack of a preference.
    """
    for o in policy.outcomes_in_order:
        if o is Outcome.REPLACE and spare_ready:
            return o
        if o is Outcome.SHRINK and placement_independent_model:
            return o
        if o is Outcome.REBUILD:
            return o
    return Outcome.REBUILD


def sets_to_preform(
    policy: Policy, engine: Engine, spare_ranks: Sequence[int] = ()
) -> list[MemberSet]:
    """The member sets an engine pre-forms as shadow groups under this policy.

    Survivor sets are needed for SHRINK and for the fallback path. A replacement set is needed
    for REPLACE, one per rank in ``spare_ranks``: the ranks whose spare is launched with the
    engine, since a spare is built as one rank and stands in for that rank only. A set naming a
    spare that does not exist can never form: its members would wait out the rendezvous
    (`bench/runs/2026-10-06-commit-path-72b/NOTE.md`). Sets are in a fixed global order so
    every process agrees.

    Raises:
        ValueError: for a spare rank outside the engine's world, or without REPLACE in the
            policy's order.
    """
    out: list[MemberSet] = []
    world = engine.world_size
    if Outcome.SHRINK in policy.outcomes_in_order:
        out.extend(member_sets(world))
    spares = sorted(set(spare_ranks))
    if spares and Outcome.REPLACE not in policy.outcomes_in_order:
        raise ValueError(f"spares for ranks {spares} under a policy without REPLACE")
    bad = [r for r in spares if not 0 <= r < world]
    if bad:
        raise ValueError(f"spare ranks {bad} outside the engine's world of {world}")
    for r in spares:
        out.append(member_sets(world, losses=(), replacements=[(r,)])[0])
    return out
