# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Shadow groups: every member set a single loss can need, formed while the engine is healthy.

A fault then costs a pointer swap instead of a rendezvous. For a world of ``N`` ranks and one
spare, the sets are the ``N`` survivor sets of ``N-1`` and the replacement set with the spare in
the place of the rank it stands in for. For a loss of several ranks at once (a node's), the policy
names the loss, and the survivor set and the replacement set are pre-formed for the whole set.
Each set is formed on its own store, beside the live group, in a global order every process
agrees on, so a process forms exactly the sets it is in.

The torch formation here is what the CPU tests and the gloo path use. The adapter forms the same
sets over the engine's own NCCL wrapper.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta

import torch.distributed as dist

Member = int | str
MemberSet = tuple[Member, ...]


def replacement(rank: int, node: str = "", device: int = 0) -> str:
    """The id of the spare standing in for ``rank`` inside a member set, before it takes it.

    Distinct per rank, so a set that replaces several ranks at once (a node's) names each
    spare, and each spare finds its own position in the set. With ``node`` the id names the
    spare's process too (``R1@node-d:0``), as a cover plan's sets do (decision 0035): two spares
    standing in for one rank are two sets.
    """
    return f"R{rank}@{node}:{device}" if node else f"R{rank}"


def is_replacement(member: Member) -> bool:
    """Whether ``member`` is a spare's id."""
    return isinstance(member, str) and member.startswith("R")


def replaced_rank(member: Member) -> int:
    """The rank a spare's id stands in for.

    Raises:
        ValueError: for a member that is not a spare's id.
    """
    if not is_replacement(member):
        raise ValueError(f"{member!r} is not a replacement id")
    return int(str(member)[1:].split("@", 1)[0])


def member_sets(
    world: int,
    losses: Sequence[Sequence[int]] | None = None,
    replacements: Sequence[Sequence[int]] = (),
) -> list[MemberSet]:
    """The member sets to pre-form.

    Args:
        world: live ranks ``0..world-1``.
        losses: the sets of ranks that can be lost together, each getting a survivor set; every
            single rank by default. A node's ranks are one loss.
        replacements: the sets of ranks that have spares, each getting a replacement set with
            every rank's spare in its place.

    Returns:
        Survivor sets in loss order, then the replacement sets in order.
    """
    loss_sets = [(d,) for d in range(world)] if losses is None else [tuple(x) for x in losses]
    sets: list[MemberSet] = [tuple(r for r in range(world) if r not in loss) for loss in loss_sets]
    for rep in replacements:
        rep = tuple(rep)
        sets.append(tuple(replacement(r) if r in rep else r for r in range(world)))
    return sets


def check_formable(sets: Sequence[MemberSet], world: int, spare_ranks: Sequence[int]) -> None:
    """Every member of every set is a live rank or the id of a spare that exists.

    A set with a member nobody plays never forms: each process in it waits out the store's
    timeout at startup, and the engine never serves. This refuses such a set before any
    rendezvous, so a misconfiguration fails in a millisecond instead of five minutes.

    Raises:
        ValueError: naming the first set and member that no process plays.
    """
    spares = set(spare_ranks)
    for s in sets:
        for m in s:
            if is_replacement(m):
                if replaced_rank(m) not in spares:
                    raise ValueError(
                        f"set {set_key(s)} names {m}, but no spare is launched for rank "
                        f"{replaced_rank(m)} (spares: {sorted(spares)})"
                    )
            elif not 0 <= int(m) < world:
                raise ValueError(f"set {set_key(s)} names rank {m} outside the world of {world}")


@dataclass(frozen=True)
class Rendezvous:
    """One group to form: which member set it belongs to, who is in it, and where they meet."""

    members: MemberSet
    """The full member set the group is pre-formed for; the key it is kept under."""
    group: MemberSet
    """The members the group spans: those of one stage, or of one column, in set order."""
    host: str
    """The address of the group's first member, where the rendezvous store listens."""
    port: int


RENDEZVOUS_PORTS = (30000, 9000)
"""The port range engines' pre-formed groups rendezvous in: a base per engine, then a port per
set and scope above it."""


def rendezvous_base_port(engine: str) -> int:
    """The base port an engine's groups rendezvous on, the same in every process from the name.

    Every process that pre-forms the engine's groups, its workers and the spares the daemons
    launch for it (decision 0024), must meet on the same ports without being told them; the name
    is what they all have. Two engines on one host with colliding bases would collide at
    formation, which the store's timeout reports.
    """
    return RENDEZVOUS_PORTS[0] + zlib.crc32(engine.encode()) % RENDEZVOUS_PORTS[1]


def rendezvous_plan(
    sets: Sequence[MemberSet],
    me: Member,
    select: Callable[[MemberSet], MemberSet],
    hosts: Mapping[Member, str],
    base_port: int,
    slot: int = 0,
    slots: int = 1,
) -> list[Rendezvous]:
    """The groups of one scope that ``me`` pre-forms: over whom, and where they rendezvous.

    Every process of the engine computes its plan from the same ``sets`` in the same order, so
    set ``i``'s group of this scope meets on ``base_port + i * slots + slot`` wherever it is
    formed. ``select`` picks the members the scope's group spans out of a set (a stage's, or a
    column's); ``hosts`` gives each member's node address, and the store listens at the group's
    first member, which creates it. A group that does not contain ``me``, or that would be
    ``me`` alone, is not this process's to form.

    Raises:
        KeyError: for a member without a host.
    """
    out = []
    for i, members in enumerate(sets):
        group = tuple(select(tuple(members)))
        if me not in group or len(group) < 2:
            continue
        out.append(Rendezvous(tuple(members), group, hosts[group[0]], base_port + i * slots + slot))
    return out


def set_key(members: MemberSet) -> str:
    """A stable string name for a member set, e.g. ``"0-1-R-3"``."""
    return "-".join(str(m) for m in members)


def form_groups(
    me: Member,
    sets: Sequence[MemberSet],
    host: str,
    ports: Sequence[int],
    backend: str = "gloo",
    timeout: timedelta = timedelta(seconds=30),
) -> dict[MemberSet, dist.ProcessGroup]:
    """Form every set in ``sets`` containing ``me``; set ``i`` gets its own store at ``ports[i]``.

    The store for a set is hosted by the set's first member. Every process must call this with
    the same ``sets`` and ``ports`` in the same order.

    Raises:
        ValueError: if ``ports`` and ``sets`` differ in length.
    """
    if len(ports) != len(sets):
        raise ValueError(f"{len(ports)} ports for {len(sets)} sets")
    out: dict[MemberSet, dist.ProcessGroup] = {}
    for i, s in enumerate(sets):
        if me not in s:
            continue
        rank = s.index(me)
        store = dist.TCPStore(host, ports[i], len(s), rank == 0, timeout=timeout)
        prefixed = dist.PrefixStore(f"everserve/shadow/{i}", store)
        if backend == "gloo":
            out[s] = dist.ProcessGroupGloo(prefixed, rank, len(s), timeout)
        elif backend == "nccl":
            out[s] = dist.ProcessGroupNCCL(prefixed, rank, len(s), dist.ProcessGroupNCCL.Options())
        else:
            raise ValueError(f"unsupported backend {backend!r}")
    return out
