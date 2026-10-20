# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""An indirected group handle: captured graphs call through it instead of binding a communicator.

A captured graph (a CUDA graph, or any replayed list of operations over fixed buffers) that holds
a process group object dies with that group. A graph that holds a :class:`GroupHandle` and reduces
in place into a buffer it owns survives a membership change: swap the handle's group between
steps, replay, and the collective runs on the new group. The reduction must be in place, because
the next captured node reads the address it saw at capture time.

A :class:`LocalStandIn` lets a replacement capture its graphs before it has any peers: the
collective copies input to output and the result is discarded.
"""

from __future__ import annotations

import threading

import torch
import torch.distributed as dist


class LocalStandIn:
    """A group with no peers. Collectives through it are local copies."""

    __slots__ = ()


class GroupHandle:
    """The one slot a captured collective reads at call time.

    Thread-safe: the fault path swaps the group from a side thread while the step thread may be
    about to issue a collective. A swap takes effect at the next collective.
    """

    def __init__(self, group: dist.ProcessGroup | LocalStandIn | None = None) -> None:
        """Start with ``group``, or empty; a collective through an empty handle is an error."""
        self._group = group
        self._lock = threading.Lock()

    @property
    def group(self) -> dist.ProcessGroup | LocalStandIn:
        """The group collectives run on right now."""
        with self._lock:
            if self._group is None:
                raise RuntimeError("group handle is empty")
            return self._group

    def swap(self, group: dist.ProcessGroup | LocalStandIn) -> None:
        """Point the handle at ``group``. Captured graphs follow at their next collective."""
        with self._lock:
            self._group = group

    def all_reduce_into(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """``out = sum over the group of inp``, written into ``out`` in place.

        ``out`` must be a buffer the caller owns for the life of the graph; the handle never
        allocates.
        """
        g = self.group
        out.copy_(inp)
        if not isinstance(g, LocalStandIn):
            dist.all_reduce(out, group=g)
