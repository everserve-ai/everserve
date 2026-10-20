# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The slot collective: the decode-size all-reduce captured CUDA graphs can keep.

A captured NCCL all-reduce bakes its communicator into the kernel's arguments, so no handle swap
can redirect it. The slot collective is our own kernel: it reads a device-visible membership slot
(member order, abort word, per-chunk sequence counters) on every replay, so a captured graph follows
a membership change without recapture. The kernels live in ``core/csrc`` in two variants selected
by topology: peer-to-peer device memory, and host-mapped memory for hosts without P2P
(decision 0006). This module is the interface the runtime programs against; a CPU process has no
implementation and the :class:`Collective` falls back to the wrapped group.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch


class SlotCollective(Protocol):
    """One rank's handle on the slot collective."""

    def can_take(self, t: torch.Tensor) -> bool:
        """Whether ``t`` is a shape and dtype the kernel reduces (decode sizes)."""
        ...

    def all_reduce_into(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """Reduce ``inp`` over the current members into ``out``, in place, graph-capturable."""
        ...

    def set_members(self, members: Iterable[int], epoch: int) -> None:
        """Program the slot: who participates from now on, and the epoch that fences it."""
        ...

    def abort(self) -> None:
        """Make every member spinning on a dead peer leave the kernel."""
        ...

    def resume(self) -> None:
        """Clear the abort word and the sequence counters after the membership is programmed."""
        ...

    @property
    def variant(self) -> str:
        """Which kernel this is, for the run report: ``"p2p"``, ``"host"``, or a double's name."""
        ...
