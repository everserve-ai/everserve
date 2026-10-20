# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The loopback KV copy: a rank's step-committed rows over the Python store, and a unit's restore.

The reference tier's committed-KV transport (decision 0004). The CPU tests call
:meth:`KVCopy.save` for each layer with the rows written this step, :meth:`KVCopy.commit` once per
step, and :meth:`KVCopy.restore` when a survivor absorbs a unit from a lost rank under SHRINK. The
production transport, :class:`everserve.state.mirror.MirrorCommittedKV`, implements the same
:class:`everserve.state.committed.CommittedKV` interface over the node's store; the paged layouts
both read and write through are :class:`everserve.state.kvcopy.PagedKV`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from everserve.reference.store import Address, KVStoreClient
from everserve.state.kvcopy import HeadSlot, PagedKV, Unit, layer_index

log = logging.getLogger(__name__)


@dataclass
class CopyProfile:
    """Seconds spent in each phase, cumulative."""

    save_s: float = 0.0
    commit_s: float = 0.0
    put_s: float = 0.0
    steps: int = 0


@dataclass
class KVCopy:
    """Per-rank: gather this step's rows per unit, commit them, restore a lost unit.

    Args:
        address: the store to commit to.
        units: returns ``[(head_slot, unit), ...]`` for the units this rank holds right now.
            Called at every save, because the set grows when the rank absorbs a unit.
    """

    address: Address
    units: Callable[[], Sequence[tuple[HeadSlot, Unit]]]
    layers: list[tuple[str, torch.Tensor]] = field(default_factory=list, init=False)
    paged: PagedKV | None = field(default=None, init=False)
    profile: CopyProfile = field(default_factory=CopyProfile, init=False)
    _pending: dict[Unit, dict[int, torch.Tensor]] = field(default_factory=dict, init=False)
    _pending_slots: torch.Tensor | None = field(default=None, init=False)
    _layer_of: dict[str, int] = field(default_factory=dict, init=False)
    _store: KVStoreClient | None = field(default=None, init=False)

    def register(self, kv_caches: Mapping[str, torch.Tensor]) -> None:
        """Take the rank's per-layer cache tensors, in layer order, and open the store."""
        ordered = sorted((layer_index(n), n, t) for n, t in kv_caches.items())
        self.layers = [(n, t) for _, n, t in ordered]
        self._layer_of = {n: i for i, (n, _) in enumerate(self.layers)}
        first = self.layers[0][1]
        self.paged = PagedKV.of(first)
        log.info(
            "kv cache %s stride %s dtype %s layout %s",
            tuple(first.shape),
            first.stride(),
            first.dtype,
            self.paged.layout.value,
        )
        self._store = KVStoreClient(self.address)
        self._store.init(
            [u for _, u in self.units()],
            len(self.layers),
            self.paged.num_slots,
            self.paged.head_dim,
        )

    def clear(self) -> None:
        """Drop anything saved but not committed. Called before a step is re-run."""
        self._pending = {}
        self._pending_slots = None

    def save(self, layer_name: str, kv_layer: torch.Tensor, slots: torch.Tensor) -> None:
        """Keep the rows written to ``kv_layer`` at ``slots`` this step, for every unit held.

        Negative slot ids (padding) are kept out at commit time.
        """
        if self.paged is None:
            return
        t0 = time.perf_counter()
        layer = self._layer_of[layer_name]
        clamped = slots.clamp(min=0)
        for head, unit in self.units():
            self._pending.setdefault(unit, {})[layer] = self.paged.rows(kv_layer, clamped, head)
        self._pending_slots = slots
        self.profile.save_s += time.perf_counter() - t0

    def commit(self) -> None:
        """Send this step's rows to the store. A no-op when nothing was saved."""
        if self._pending_slots is None or self._store is None:
            return
        t0 = time.perf_counter()
        slots = self._pending_slots.cpu()
        keep = slots >= 0
        out: dict[Unit, tuple[torch.Tensor, torch.Tensor]] = {}
        for unit, per_layer in self._pending.items():
            x = torch.stack([per_layer[i] for i in range(len(self.layers))])
            out[unit] = (slots[keep], x.cpu()[:, :, keep])
        t1 = time.perf_counter()
        self._store.put(out)
        self.clear()
        self.profile.put_s += time.perf_counter() - t1
        self.profile.commit_s += time.perf_counter() - t0
        self.profile.steps += 1

    def restore(self, unit_slots: Sequence[tuple[Unit, HeadSlot]]) -> None:
        """Write each lost ``unit`` into this rank's cache at ``head_slot``, every layer."""
        if self.paged is None or self._store is None:
            raise RuntimeError("restore before register")
        device = self.layers[0][1].device
        for unit, head in unit_slots:
            slots, x = self._store.get(unit)
            slots = slots.to(device)
            for i, (_, t) in enumerate(self.layers):
                self.paged.write_rows(t, slots, head, x[i].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
