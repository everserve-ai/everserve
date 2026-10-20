# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The paged KV cache's layouts, and the rows of one head slot in them.

The engine-independent half of a KV connector reads and writes a rank's paged cache through
:class:`PagedKV`: the production transport (:mod:`everserve.state.mirror`) gathers each step's rows
by head slot and writes a lost rank's rows back on a restore, and the adapter recognises an
engine's cache tensors with it. The paged cache layouts are the ones engines use for one layer's
K and V:

* ``kv_first``:     ``[2, blocks, block_size, heads, head_dim]``
* ``blocks_first``: ``[blocks, 2, block_size, heads, head_dim]``
* ``combined``:     ``[blocks, heads, block_size, 2 * head_dim]`` (K and V side by side)

A slot id is ``block * block_size + offset``. A unit is one KV head and its query heads; a rank
holds each unit it owns in a head slot of its attention layer, and the slot it uses for an
absorbed unit can differ from the slot the lost rank used.

The loopback copy over the Python store, which the CPU tests run the same interface against, is
:class:`everserve.reference.kvcopy.KVCopy` (decision 0004).
"""

from __future__ import annotations

import re
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum

import torch

Unit = Hashable
HeadSlot = int
_LAYER_INDEX = re.compile(r"layers\.(\d+)\.")


class CacheLayout(str, Enum):
    """How one layer's paged K/V tensor is laid out."""

    KV_FIRST = "kv_first"
    BLOCKS_FIRST = "blocks_first"
    COMBINED = "combined"


@dataclass(frozen=True)
class PagedKV:
    """One layer's paged-cache shape, and how to read and write the rows of one head slot."""

    layout: CacheLayout
    num_blocks: int
    block_size: int
    head_dim: int

    @property
    def num_slots(self) -> int:
        """Total slot ids in the cache."""
        return self.num_blocks * self.block_size

    @classmethod
    def of(cls, t: torch.Tensor) -> PagedKV:
        """Recognise the layout of a layer's cache tensor.

        Raises:
            ValueError: for a shape that is none of the three layouts.
        """
        if t.dim() == 5 and t.shape[0] == 2:
            return cls(CacheLayout.KV_FIRST, t.shape[1], t.shape[2], t.shape[4])
        if t.dim() == 5 and t.shape[1] == 2:
            return cls(CacheLayout.BLOCKS_FIRST, t.shape[0], t.shape[2], t.shape[4])
        if t.dim() == 4 and t.shape[3] % 2 == 0:
            return cls(CacheLayout.COMBINED, t.shape[0], t.shape[2], t.shape[3] // 2)
        raise ValueError(f"unsupported KV cache layout {tuple(t.shape)}")

    def split(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slot ids -> (block, offset)."""
        slots = slots.long()
        return slots // self.block_size, slots % self.block_size

    def rows(self, t: torch.Tensor, slots: torch.Tensor, head: HeadSlot) -> torch.Tensor:
        """The K and V rows of head slot ``head`` at ``slots``, as a copy of shape ``[2, n, d]``."""
        b, o = self.split(slots)
        if self.layout is CacheLayout.KV_FIRST:
            return t[:, b, o, head]
        if self.layout is CacheLayout.BLOCKS_FIRST:
            return t[b, :, o, head].transpose(0, 1)
        x = t[b, head, o]
        return torch.stack([x[:, : self.head_dim], x[:, self.head_dim :]])

    def rows_of_heads(
        self, t: torch.Tensor, slots: torch.Tensor, heads: torch.Tensor
    ) -> torch.Tensor:
        """The K and V rows of head slots ``heads`` at ``slots``, as a view ``[h, n, 2, d]``.

        One gather for every head, so a layer costs one kernel on the step thread instead of one
        per head slot. The result may be non-contiguous; a copy makes it so on the device.
        """
        b, o = self.split(slots)
        if self.layout is CacheLayout.KV_FIRST:
            return t[:, b, o].index_select(2, heads).permute(2, 1, 0, 3)  # [2, n, h, d]
        if self.layout is CacheLayout.BLOCKS_FIRST:
            return t[b, :, o].index_select(2, heads).permute(2, 0, 1, 3)  # [n, 2, h, d]
        x = t[b, :, o].index_select(1, heads)  # [n, h, 2d]
        return x.view(x.shape[0], x.shape[1], 2, self.head_dim).permute(1, 0, 2, 3)

    def layout_strides(self, t: torch.Tensor) -> tuple[int, int, int, int, int]:
        """``t``'s element strides as (block, offset in block, head slot, K-or-V, element).

        The gather kernel addresses any of the three layouts with these five numbers; it reads the
        same rows :meth:`rows_of_heads` does.
        """
        st = t.stride()
        if self.layout is CacheLayout.KV_FIRST:  # [2, nb, bs, h, d]
            return st[1], st[2], st[3], st[0], st[4]
        if self.layout is CacheLayout.BLOCKS_FIRST:  # [nb, 2, bs, h, d]
            return st[0], st[2], st[3], st[1], st[4]
        return st[0], st[2], st[1], self.head_dim * st[3], st[3]  # [nb, h, bs, 2d]

    def write_rows(
        self, t: torch.Tensor, slots: torch.Tensor, head: HeadSlot, x: torch.Tensor
    ) -> None:
        """Write ``x`` of shape ``[2, n, d]`` into head slot ``head`` at ``slots``, in place."""
        b, o = self.split(slots)
        if self.layout is CacheLayout.KV_FIRST:
            t[:, b, o, head] = x
        elif self.layout is CacheLayout.BLOCKS_FIRST:
            t[b, :, o, head] = x.transpose(0, 1)
        else:
            t[b, head, o] = torch.cat([x[0], x[1]], -1)


def layer_index(name: str) -> int:
    """The layer number in a cache name such as ``model.layers.12.self_attn.attn``.

    Raises:
        ValueError: if the name carries no layer index.
    """
    m = _LAYER_INDEX.search(name)
    if m is None:
        raise ValueError(f"cannot find a layer index in cache name {name!r}")
    return int(m.group(1))
