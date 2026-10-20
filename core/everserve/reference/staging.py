# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Staging a doomed rank's KV to its replacement during a preemption notice.

A notice names a rank and a deadline. Until the deadline the rank keeps serving, and its KV is
copied to the replacement in rounds: the first round copies every block in use, later rounds copy
only the blocks that received tokens since the previous round. Blocks are the scheduler's block
ids, shared by every tensor-parallel rank, so the replacement's KV takes each block at the same
id and nothing has to be recomputed when the rank goes.

This module decides which blocks to copy and keeps the bookkeeping. Moving the bytes is the
adapter's job.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

BlockId = int
RequestId = str


@dataclass
class StagedRequest:
    """How far one request's KV has been staged."""

    tokens: int
    """Tokens whose KV the replacement holds."""
    blocks: tuple[BlockId, ...]
    """The block ids those tokens were staged from, in table order."""


@dataclass
class StagingWindow:
    """Bookkeeping for one notice, from receipt to the deadline."""

    rank: int
    deadline: float
    """Unix seconds at which the rank leaves."""
    rounds: int = 0
    blocks_copied: int = 0
    bytes_copied: int = 0
    staged: dict[RequestId, StagedRequest] = field(default_factory=dict)

    def seconds_left(self, now: float) -> float:
        """Time to the deadline at ``now`` (Unix seconds); negative once it has passed."""
        return self.deadline - now


def blocks_to_stage(
    block_tables: Mapping[RequestId, Sequence[BlockId]],
    num_computed: Mapping[RequestId, int],
    staged: dict[RequestId, StagedRequest],
    block_size: int,
) -> list[BlockId]:
    """Choose the blocks holding KV written since the previous round, and update ``staged``.

    A request that was preempted and recomputed may sit in different blocks at the same token
    count. Its table is compared with the one it was staged from, and staging restarts at the
    first block whose id changed. Requests no longer in ``num_computed`` are dropped from
    ``staged``.

    Args:
        block_tables: request id -> its block ids, in scheduler order.
        num_computed: request id -> tokens whose KV is in the cache now.
        staged: per-request staging state, updated in place.
        block_size: tokens per block.

    Returns:
        Sorted distinct block ids to copy this round.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    out: set[BlockId] = set()
    for rid, n in num_computed.items():
        table = tuple(block_tables[rid])
        prior = staged.get(rid)
        done_tokens = prior.tokens if prior else 0
        old = prior.blocks if prior else ()
        same = next(
            (i for i, (x, y) in enumerate(zip(old, table, strict=False)) if x != y),
            min(len(old), len(table)),
        )
        done_tokens = min(done_tokens, same * block_size, n)
        if n > done_tokens:
            first, last = done_tokens // block_size, (n - 1) // block_size
            out.update(table[first : last + 1])
        n_blocks = (n + block_size - 1) // block_size
        staged[rid] = StagedRequest(tokens=n, blocks=table[:n_blocks])
    for rid in list(staged):
        if rid not in num_computed:
            del staged[rid]
    return sorted(out)
