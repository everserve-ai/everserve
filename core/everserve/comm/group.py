# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A process group that can lose a member.

:class:`ShrinkableGroup` wraps a ``torch.distributed`` process group with the one operation the
engines lack: remove a dead rank and keep going. With NCCL 2.27+ and a torch that exposes
``shrink_group`` this is ``ncclCommShrink`` with the abort flag, which also releases a kernel stuck
in a collective with the dead peer. Otherwise the old group is aborted and a new one is formed over
a prefix of a shared store, keyed by the membership epoch so no two epochs collide.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

log = logging.getLogger(__name__)

STORE_PREFIX = "everserve/group"


class ShrinkMethod(str, Enum):
    """How a group came to exist."""

    INIT = "init"
    NCCL_SHRINK = "ncclCommShrink"
    RECREATE = "recreate"
    SHADOW = "shadow"
    """Pre-formed beside the live group; the change was a pointer swap."""


def _group_from_store(
    store: dist.Store, rank: int, world: int, backend: str, epoch: int, timeout: timedelta
) -> dist.ProcessGroup:
    prefixed = dist.PrefixStore(f"{STORE_PREFIX}/{epoch}", store)
    if backend == "gloo":
        return dist.ProcessGroupGloo(prefixed, rank, world, timeout)
    if backend == "nccl":
        return dist.ProcessGroupNCCL(prefixed, rank, world, dist.ProcessGroupNCCL.Options())
    raise ValueError(f"unsupported backend {backend!r}")


@dataclass
class ShrinkableGroup:
    """A process group over ``ranks`` (original rank ids, in group order) that can shrink.

    Attributes:
        group: the torch process group.
        ranks: original rank ids in group order; ``ranks.index(me)`` is a rank's position.
        backend: ``"nccl"`` or ``"gloo"``.
        store: the shared store a recreated group is formed over, or None when the group can
            only shrink through NCCL.
        method: how this group was made.
        first_timeout: collective timeout for the first (epoch 0) gloo group. Gloo cannot abort a
            collective, so a short timeout is what turns a dead peer into an error.
        later_timeout: timeout for groups formed after a shrink; long, so a recovered rank waits
            for a peer still timing out of the old group.
    """

    group: dist.ProcessGroup
    ranks: list[int]
    backend: str
    store: dist.Store | None = None
    method: ShrinkMethod = ShrinkMethod.INIT
    first_timeout: timedelta = timedelta(seconds=3)
    later_timeout: timedelta = timedelta(seconds=120)

    @classmethod
    def new_subgroup(cls, world: int, backend: str, device: torch.device) -> ShrinkableGroup:
        """A group over ranks ``0..world-1`` of the default process group."""
        kwargs = {"device_id": device} if backend == "nccl" else {}
        pg = dist.new_group(ranks=list(range(world)), backend=backend, **kwargs)
        return cls(pg, list(range(world)), backend)

    @classmethod
    def from_store(
        cls,
        store: dist.Store,
        rank: int,
        world: int,
        backend: str,
        epoch: int = 0,
        first_timeout: timedelta = timedelta(seconds=3),
        later_timeout: timedelta = timedelta(seconds=120),
    ) -> ShrinkableGroup:
        """A group formed directly over ``store`` without a default process group."""
        timeout = first_timeout if epoch == 0 else later_timeout
        pg = _group_from_store(store, rank, world, backend, epoch, timeout)
        return cls(
            pg, list(range(world)), backend, store, ShrinkMethod.INIT, first_timeout, later_timeout
        )

    @property
    def world_size(self) -> int:
        """Members in this group."""
        return len(self.ranks)

    def all_reduce(self, t: torch.Tensor) -> None:
        """Sum ``t`` in place across the group."""
        dist.all_reduce(t, group=self.group)

    def shrink(self, lost: Sequence[int], me: int, epoch: int) -> ShrinkableGroup:
        """Return the group without ``lost``. Every survivor calls this with the same arguments.

        Tries ``ncclCommShrink`` with abort when available and the group was not formed over a
        store; otherwise aborts the old group (NCCL) and forms a new one over the store at
        ``epoch``. The recreate path is also the fallback when the NCCL shrink fails.

        Args:
            lost: original rank ids that are gone.
            me: this process's original rank id.
            epoch: the membership epoch the new group belongs to; must differ from any earlier one.
        """
        lost = tuple(lost)
        for dead in lost:
            if dead not in self.ranks:
                raise ValueError(f"rank {dead} is not in the group {self.ranks}")
        ranks = [r for r in self.ranks if r not in lost]
        failure: str | None = None
        if self.backend == "nccl" and hasattr(dist, "shrink_group") and self.store is None:
            try:
                pg = dist.shrink_group(
                    [self.ranks.index(d) for d in lost],
                    group=self.group,
                    shrink_flags=c10d.SHRINK_ABORT,
                )
                return ShrinkableGroup(pg, ranks, self.backend, None, ShrinkMethod.NCCL_SHRINK)
            except Exception as e:  # noqa: BLE001 - fall back to recreate and record why
                failure = f"{type(e).__name__}: {e}"
                log.warning("ncclCommShrink failed, recreating the group: %s", failure)
        store = self.store if self.store is not None else c10d._get_default_store()
        if self.backend == "nccl":
            try:
                c10d._abort_process_group(self.group)
            except Exception as e:  # noqa: BLE001 - an abort that fails must not stop recovery
                log.warning("abort of the old group failed: %s", e)
        pg = _group_from_store(
            store, ranks.index(me), len(ranks), self.backend, epoch, self.later_timeout
        )
        return ShrinkableGroup(
            pg,
            ranks,
            self.backend,
            store,
            ShrinkMethod.RECREATE,
            self.first_timeout,
            self.later_timeout,
        )
