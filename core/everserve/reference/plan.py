# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Placement plans: who owns which unit of work, before and after a rank is lost.

A plan is a pure function of the world and the lost ranks, so every survivor computes the same
plan without being told. Units are the pieces of a model that can move between ranks without
changing any tensor shape: a KV head with its query heads, a fixed-size chunk of experts, a
fixed-width slice of an MLP. Keeping the units fixed is what makes the arithmetic after a loss
identical to the arithmetic before it; see ``everserve.reference.arith``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TypeVar

Unit = TypeVar("Unit", bound=Hashable)
"""Anything a rank can own: an int head index, a tuple of expert ids, a tagged pair."""

Owned = dict[int, list[Unit]]
"""Rank -> the units it owns, in a stable order."""

Cost = Callable[[int, "Owned[Unit]"], object]
"""Load of a rank under a plan. Any orderable value; lower is less loaded."""


def even_units(n_units: int, world: int) -> Owned[int]:
    """Split ``n_units`` unit indices evenly across ``world`` ranks, in rank order.

    Raises:
        ValueError: if the units do not divide evenly. An uneven split would give ranks
            different shapes, which the placement-independent arithmetic does not allow.
    """
    if world <= 0:
        raise ValueError(f"world must be positive, got {world}")
    if n_units % world:
        raise ValueError(f"{n_units} units do not divide evenly over {world} ranks")
    k = n_units // world
    return {r: list(range(r * k, (r + 1) * k)) for r in range(world)}


def expert_chunks(n_experts: int, world: int, chunk: int) -> Owned[tuple[int, ...]]:
    """Group experts into fixed-size chunks and give each rank its contiguous share.

    A chunk is the unit that moves on a loss. Every chunk has the same size, so an expert GEMM
    has the same shape no matter which rank runs it.

    Raises:
        ValueError: if the experts do not divide evenly into chunks per rank.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if world <= 0 or n_experts % world:
        raise ValueError(f"{n_experts} experts do not divide evenly over {world} ranks")
    per_rank = n_experts // world
    if per_rank % chunk:
        raise ValueError(f"{per_rank} experts per rank do not divide into chunks of {chunk}")
    return {
        r: [
            tuple(range(r * per_rank + i, r * per_rank + i + chunk))
            for i in range(0, per_rank, chunk)
        ]
        for r in range(world)
    }


def _count(rank: int, owned: Owned[Unit]) -> int:
    return len(owned[rank])


def after_loss(
    owned: Mapping[int, Sequence[Unit]],
    dead: int,
    cost: Cost[Unit] | None = None,
) -> tuple[Owned[Unit], Owned[Unit]]:
    """Reassign the lost rank's units to the survivors.

    Greedy: each of the dead rank's units, in its stored order, goes to the least-loaded
    survivor, ties broken by the lowest rank. Deterministic, so every survivor agrees.

    Args:
        owned: the plan before the loss.
        dead: the rank that is gone.
        cost: how loaded a rank is under a plan. Defaults to its unit count. A caller that
            places two kinds of unit can pass a cost that accounts for both.

    Returns:
        ``(new_owned, gained)``: the plan after the loss, and for each survivor the units it
        took on. ``gained`` has an entry for every survivor, empty when it took nothing.

    Raises:
        KeyError: if ``dead`` is not in the plan.
    """
    if dead not in owned:
        raise KeyError(f"rank {dead} is not in the plan")
    load: Cost[Unit] = cost or _count
    new: Owned[Unit] = {r: list(v) for r, v in owned.items() if r != dead}
    gained: Owned[Unit] = {r: [] for r in new}
    for unit in owned[dead]:
        taker = min(new, key=lambda r: (load(r, new), r))
        new[taker].append(unit)
        gained[taker].append(unit)
    return new, gained


def capacity_after_one_loss(
    owned: Mapping[int, Sequence[Unit]],
    cost: Cost[Unit] | None = None,
    count: Callable[[Sequence[Unit]], int] = len,
) -> int:
    """The most units any rank can hold after any single loss.

    This is what a rank reserves at startup, so absorbing its share after a loss needs no
    reallocation: a KV head slot per unit, with zeros in the slots it does not use yet.

    Args:
        owned: the plan before any loss.
        cost: passed to :func:`after_loss`.
        count: how to count a rank's units when more than one kind is placed together.
    """
    if not owned:
        return 0
    if len(owned) == 1:
        return max(count(v) for v in owned.values())
    worst = 0
    for dead in owned:
        new, _ = after_loss(owned, dead, cost)
        worst = max(worst, max(count(v) for v in new.values()))
    return worst
