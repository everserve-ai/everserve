# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The committed-KV interface: KV that outlives the rank that produced it.

One interface, implemented by :class:`everserve.state.mirror.MirrorCommittedKV` over the
``store/`` service in production (``proto/mirror.proto``) and by
:class:`everserve.reference.committed.LoopbackCommittedKV` over the Python store for exactness
tests and SHRINK on CPU (the reference tier, decision 0004). The adapter calls
:meth:`CommittedKV.save` per layer with the rows written this step, then
:meth:`CommittedKV.commit` and :meth:`CommittedKV.fence` once per step at the step's end (under
the reference guard, before its agreement collective): the commit of step N starts without
waiting and the fence waits for step N-1, so a step the survivors accept has every rank's KV
durable through the previous step (decision 0009).
:meth:`CommittedKV.evicted` and :meth:`CommittedKV.finished` keep the per-request committed
counts true when the scheduler preempts or finishes a request. The mapping onto vLLM's
connector lifecycle is in ``docs/CONTRACT.md``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

from everserve.state.kvcopy import HeadSlot

Unit = Hashable
RequestId = str


@runtime_checkable
class CommittedKV(Protocol):
    """What the runtime needs from any committed-KV transport.

    Runtime-checkable so an adapter can assert the transport it was handed has the full
    surface at startup rather than at the first fault.
    """

    def save(self, layer_name: str, kv_layer: torch.Tensor, slots: torch.Tensor) -> None:
        """Keep the rows written to ``kv_layer`` at ``slots`` this step."""
        ...

    def commit(self, epoch: int, step: int, committed_tokens: Mapping[RequestId, int]) -> None:
        """Make this step's rows durable.

        Tagged with the membership epoch and the per-request token counts whose KV is now
        complete.
        """
        ...

    def fence(self, step: int) -> None:
        """Wait until the store has acknowledged the commit of ``step`` (decision 0009).

        Called inside the step guard before the agreement for step ``step + 1``, so a step the
        survivors accept has every rank's rows durable through the previous one. A transport
        that commits synchronously has nothing to wait for. A failed commit surfaces here.
        """
        ...

    def committed(self, request_id: RequestId) -> int:
        """Tokens of ``request_id`` whose KV is durable; the scheduler recomputes from here."""
        ...

    def stats(self) -> Mapping[str, float]:
        """Pipeline figures for a run report: steps, waits, bytes. Empty when nothing is queued."""
        ...

    def timeline(self) -> Sequence[Mapping[str, float]]:
        """Per-step timestamps of the recent commits, for a run report to split the fence's wait.

        Each record carries ``step`` and ``perf_counter`` seconds: ``commit`` (queued on the step
        thread), ``dequeue`` (the sender took it), ``ready`` (the device copies had landed),
        ``sent`` (the store acknowledged), ``fence_start`` and ``fence_end`` (the step thread
        waited for it). Empty for a transport with nothing asynchronous.
        """
        ...

    def evicted(self, request_id: RequestId, kept_tokens: int) -> None:
        """The scheduler evicted ``request_id``'s blocks beyond ``kept_tokens`` (preemption).

        Called before the slots are overwritten, as vLLM's ``handle_preemptions`` is. The
        committed count drops to ``kept_tokens``; the rows at the freed slots now belong to
        whoever writes them next.
        """
        ...

    def finished(self, request_id: RequestId) -> None:
        """``request_id`` is done; forget it. Called once, before its blocks are freed."""
        ...

    def clear(self) -> None:
        """Drop anything saved but not committed, before a step is re-run."""
        ...

    def restore(
        self,
        dead_rank: int,
        moves: Sequence[tuple[Unit, HeadSlot, HeadSlot]],
        since_step: int = 0,
    ) -> int:
        """Write the lost rank's rows into this rank's cache.

        ``moves`` lists ``(unit, head slot on the dead rank, head slot here)`` for every unit this
        rank takes on: every unit with the identity mapping for a spare taking the rank's place, a
        subset for a survivor absorbing under SHRINK. A transport keyed by unit uses the first and
        third fields; one keyed by head slot uses the second and third.
        """
        ...
