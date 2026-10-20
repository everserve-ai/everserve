# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Positions a change moved off a node: their processes released, the node done (0033 step 3).

A rank that leaves on a notice is alive: it drains its last commits to the store at the boundary,
since the process that takes its position catches up from them, and then has nothing to do. Its
process still holds its GPU, its old communicators and its daemon link. It releases itself in
that order, after the drain: it closes its daemon link (a worker's close is its leaving) and marks
its row of the slot region released (:func:`release_position`). The node's executor reads the
mark, the channel it already reads every poll, and kills the process (:func:`end_process`). Its
exit frees what it held, its GPU memory and its communicators, with nothing torn down inside it:
vLLM's graceful end (its death pipe, or SIGTERM) runs ``worker.shutdown()``, which destroys every
communicator of the process, and the peers of those have moved to the change's groups. Decision
0025 never waits on the release of a refused group's communicators, and a released worker would:
given its death pipe and then SIGTERM, it is still running 9 s later. After the death pipe,
vLLM's SIGTERM handler does nothing (it raises
only before a shutdown was requested).

A position the change took is *taken* when the process that took it runs on this node, and has
*left* otherwise (:func:`taker_node`). The node holds nothing once every worker it launched has
left and exited and no position is taken here (:func:`holds_nothing`); its process then ends.

Nothing here imports vLLM.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


def taker_node(rank: int, takes: Iterable[Any], engine: Any, here: str) -> str:
    """The node whose process took ``rank``.

    A spare standing for any position is named in the change's takes with its node (decision
    0030); a spare preloaded for one rank is placed by the policy (``engine.spare_node``). With
    neither, the spare is the engine's own on this node (the single-node shape).
    """
    for t in takes:
        if t.rank == rank:
            return t.node
    placed = engine.spare_node(rank) if engine is not None else None
    return placed or here


def release_position(region: Any, row: int, link: Any) -> None:
    """A leaving worker, after its drain: close its daemon link, then mark its row released.

    The link first: the node ends the process once it reads the mark, and a process ended with
    its link still open would leave the daemon to learn of it from a dropped socket.
    """
    if link is not None:
        link.close()
    region.mark_released(row)


@dataclass(frozen=True)
class Ended:
    """How a released process ended."""

    how: str
    """``killed`` by this node, or ``gone`` when it had already exited."""
    exitcode: int | None
    end_s: float


def end_process(proc: Any, death_writer: Any) -> Ended:
    """End a released worker process: SIGKILL, then its death pipe closed.

    ``proc`` is a :class:`multiprocessing.Process`; ``death_writer`` the parent's end of its death
    pipe (None when already closed), closed after the kill so the executor's own shutdown later
    finds nothing to close. See the module for why nothing graceful is tried.
    """
    t0 = time.monotonic()
    how = "gone"
    if proc.is_alive():
        proc.kill()
        how = "killed"
    proc.join()
    if death_writer is not None:
        death_writer.close()
    return Ended(how, proc.exitcode, round(time.monotonic() - t0, 4))


def holds_nothing(
    launched: Mapping[int, Any], left: Iterable[int], taken: Mapping[int, Any]
) -> bool:
    """Whether a node holds nothing of its engine.

    ``launched`` maps each rank whose worker this node started to its process. Every one must have
    left (its position is held on another node) and exited, and no position may be taken by a
    process on this node: the node's region is mapped by the spares holding them.
    """
    gone = set(left)
    return (
        not taken
        and bool(launched)
        and all(r in gone and not p.is_alive() for r, p in launched.items())
    )
