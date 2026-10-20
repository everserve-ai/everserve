# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Leave a broken ``torch.distributed`` world and join a new one at the same world size.

The re-form path at constant world size: survivors leave the default group that includes the
dead rank, and join a fresh rendezvous with a replacement in the dead rank's place. The engine's
own groups are rebuilt on top by the adapter.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

log = logging.getLogger(__name__)


def leave_world() -> list[str]:
    """Abort (NCCL) and destroy the default process group and every subgroup.

    Returns:
        The steps that ran, for the event log: ``"abort"``, ``"destroy"``, or what failed.
        A failure is recorded, not raised: a rank that cannot leave cleanly must still join.
    """
    done: list[str] = []
    if not dist.is_initialized():
        return done
    try:
        if dist.get_backend() == "nccl":
            c10d._abort_process_group()
            done.append("abort")
    except Exception as e:  # noqa: BLE001 - recorded for the event log
        done.append(f"abort failed: {e}")
        log.warning("abort of the default group failed: %s", e)
    try:
        dist.destroy_process_group()
        done.append("destroy")
    except Exception as e:  # noqa: BLE001 - recorded for the event log
        done.append(f"destroy failed: {e}")
        log.warning("destroy of the default group failed: %s", e)
    return done


def join_world(
    init_method: str,
    rank: int,
    world: int,
    backend: str,
    timeout: timedelta = timedelta(seconds=120),
) -> None:
    """Join a new default process group as ``rank`` of ``world`` at ``init_method``."""
    dist.init_process_group(
        backend, init_method=init_method, rank=rank, world_size=world, timeout=timeout
    )
