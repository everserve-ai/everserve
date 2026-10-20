# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The committed-KV interface over the Python store: exact and synchronous.

:class:`LoopbackCommittedKV` is the reference tier's
:class:`everserve.state.committed.CommittedKV` (decision 0004): the CPU tests and SHRINK on CPU
run it over :class:`everserve.reference.kvcopy.KVCopy` and the Python store. The production
transport is :class:`everserve.state.mirror.MirrorCommittedKV`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from everserve.reference.kvcopy import KVCopy
from everserve.state.committed import RequestId, Unit
from everserve.state.kvcopy import HeadSlot


class LoopbackCommittedKV:
    """The interface over the Python store: exact and synchronous. Tests and CPU SHRINK."""

    def __init__(self, copy: KVCopy) -> None:
        """Wrap a registered or to-be-registered :class:`KVCopy`."""
        self._copy = copy
        self._committed: dict[RequestId, int] = {}
        self.last_epoch = 0
        self.last_step = 0

    def save(self, layer_name: str, kv_layer: torch.Tensor, slots: torch.Tensor) -> None:
        """See :meth:`CommittedKV.save`."""
        self._copy.save(layer_name, kv_layer, slots)

    def commit(self, epoch: int, step: int, committed_tokens: Mapping[RequestId, int]) -> None:
        """See :meth:`CommittedKV.commit`."""
        self._copy.commit()
        self._committed.update(committed_tokens)
        self.last_epoch, self.last_step = epoch, step

    def fence(self, step: int) -> None:
        """See :meth:`CommittedKV.fence`: the loopback commit is synchronous, nothing is pending."""

    def committed(self, request_id: RequestId) -> int:
        """See :meth:`CommittedKV.committed`."""
        return self._committed.get(request_id, 0)

    def stats(self) -> Mapping[str, float]:
        """See :meth:`CommittedKV.stats`: nothing is pipelined."""
        return {}

    def timeline(self) -> Sequence[Mapping[str, float]]:
        """See :meth:`CommittedKV.timeline`: nothing is asynchronous."""
        return []

    def evicted(self, request_id: RequestId, kept_tokens: int) -> None:
        """See :meth:`CommittedKV.evicted`."""
        if kept_tokens < 0:
            raise ValueError(f"kept_tokens must be non-negative, got {kept_tokens}")
        if request_id in self._committed:
            self._committed[request_id] = min(self._committed[request_id], kept_tokens)

    def finished(self, request_id: RequestId) -> None:
        """See :meth:`CommittedKV.finished`."""
        self._committed.pop(request_id, None)

    def clear(self) -> None:
        """See :meth:`CommittedKV.clear`."""
        self._copy.clear()

    def restore(
        self,
        dead_rank: int,
        moves: Sequence[tuple[Unit, HeadSlot, HeadSlot]],
        since_step: int = 0,
    ) -> int:
        """See :meth:`CommittedKV.restore`. The store is keyed by unit; the dead rank is unused."""
        self._copy.restore([(unit, mine) for unit, _theirs, mine in moves])
