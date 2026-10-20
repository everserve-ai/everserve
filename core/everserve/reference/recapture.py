# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The order in which CUDA graphs are recaptured after a membership change.

After a rank is replaced the survivors may have released their captured graphs and the newcomer
never had any. Serving continues eagerly, and graphs come back in small slices between steps.
Every rank must pick the same slice, because capture runs collectives in lock step, so the
choice is a pure function of what has been asked for and what is done.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TypeVar

Descriptor = TypeVar("Descriptor", bound=Hashable)
"""Whatever identifies one graph to capture: a (mode, batch size) pair in vLLM."""


def next_slice(
    wanted: Sequence[Descriptor],
    all_descriptors: Sequence[Descriptor],
    done: Iterable[Descriptor],
    n: int,
) -> list[Descriptor]:
    """Choose up to ``n`` descriptors to capture next.

    Descriptors serving has asked for come first, in the order they were asked for, then the
    rest in capture order. Nothing already captured is repeated. Deterministic for equal inputs,
    so every rank captures the same slice.

    Args:
        wanted: descriptors a dispatch wanted and could not serve from a graph, oldest first.
        all_descriptors: every descriptor the engine captures, in its capture order.
        done: descriptors already captured.
        n: the slice size. Zero or negative returns an empty slice.
    """
    captured = set(done)
    pick: list[Descriptor] = []
    for d in list(wanted) + list(all_descriptors):
        if len(pick) >= n:
            break
        if d not in captured and d not in pick:
            pick.append(d)
    return pick
