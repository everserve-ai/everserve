# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A rank lost on another node: the coordinator's word, and the pipeline peer's failed send.

An executor sees a death on its own node (decision 0021: the slot kernel, the stalled step, the
process sentinel). A death on another node it cannot see: its wait on that rank's queue reports
nothing when the writer dies. Decision 0023 makes the coordinator the authority: the daemon
forwards every ``Change`` the coordinator claims to the
engine's clients, and :func:`follow_changes` turns the ranks it names into losses the executor
consumes as it consumes a local one, marking them dead so the step in flight raises.

A worker on another node sees the death too, about two seconds later, when the dead rank's
sockets close and its pipeline send or receive fails. :func:`pipeline_peers` names the ranks that
failure is about, so the worker reports them through its daemon link as a second signal; the
coordinator folds the later report (decision 0021).

The two sightings race. A worker hears a change on its own daemon link and refuses the current
group at once, so a step it is running can fail on that refusal before this node's executor has
the coordinator's word (a sampled-token broadcast refused 0.4 s after the other node's loss, the
executor not yet told, would shut the engine down). :func:`loss_behind` reads such a failure as
the loss it is.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Collection, Mapping
from pathlib import Path

from everserve.comm.layout import ParallelLayout, position
from everserve.comm.runtime import CHANGE_ABORT
from everserve.comm.shadow import Member, MemberSet
from everserve.comm.slotwatch import process_state
from everserve.contract.daemon import ChangeHeard

log = logging.getLogger(__name__)


class RankLostError(RuntimeError):
    """A worker died during this step. The engine core is alive; complete the change and step on."""


def loss_behind(
    error: str,
    lost: Callable[[], bool],
    bound_s: float,
    poll_s: float = 0.02,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Whether a worker's failed step is a loss this executor knows, or comes to know in time.

    ``error`` is the failure's text as vLLM's worker RPC carries it: the worker loop sends a failed
    call's exception as ``str(exception)`` and nothing else (``WorkerProc.enqueue_output``,
    ``PRIVATE-INTERNALS.md`` A8a), so its text is the only channel. A step ended by a membership
    change opens its message with :data:`~everserve.comm.runtime.CHANGE_ABORT`; the start is
    matched, not any substring, so an unrelated error that quotes the marker is not read as one.
    The change the worker acted on reaches this executor too, through the coordinator's word or
    the slot region, so ``lost`` (the executor's own record, read again at each poll) turns true
    within ``bound_s``. Any other failure is not a loss and is answered at once. A change abort
    with no loss inside the bound is not one either: the step failed and nothing replaces a member.
    """
    if not error.startswith(CHANGE_ABORT):
        return False
    deadline = clock() + bound_s
    while not lost():
        if clock() >= deadline:
            return False
        sleep(poll_s)
    return True


def failed_step(
    step: bool,
    rank: int,
    error: str,
    lost: Callable[[], bool],
    dead: Callable[[], Collection[int]],
    bound_s: float,
    poll_s: float = 0.02,
) -> tuple[Exception, dict | None]:
    """The error a worker's failed RPC raises in the executor, and the event it records if any.

    A step RPC (``execute_model``, ``sample_tokens``) that a change ended raises
    :class:`RankLostError` once the loss is this executor's (:func:`loss_behind`), with a
    ``step_refused`` event, and the engine core steps on to the change. A change abort no loss
    follows inside ``bound_s`` fails the engine and says so. Anything else, and any failure outside
    a step, fails the engine as stock does.
    """
    if step and loss_behind(error, lost, bound_s, poll_s):
        event = {"kind": "step_refused", "rank": rank, "error": error, "at": time.time()}
        return RankLostError(f"worker(s) {sorted(dead())} lost; rank {rank}: {error}"), event
    if step and error.startswith(CHANGE_ABORT):
        return RuntimeError(
            f"Worker failed with error '{error}': a change ended the step and no loss reached "
            f"this executor within {bound_s} s"
        ), None
    return RuntimeError(f"Worker failed with error '{error}'"), None


def follow_changes(
    changes: queue.Queue[ChangeHeard],
    known: Callable[[int], bool],
    on_lost: Callable[[int, ChangeHeard], None],
    stop: threading.Event,
    poll_s: float = 0.5,
    on_change: Callable[[ChangeHeard], None] | None = None,
) -> None:
    """Consume every change the coordinator claims; a rank it names is lost unless ``known``.

    Runs until ``stop`` is set. Every change goes to ``on_change`` first, known ranks included:
    the claim is the record of the ranks it took and of those leaving on notice. A rank already
    known lost (seen on this node first, and reported by this executor) is not marked again,
    nor is a rank the change names departing (protocol 9): it is alive, leaves at the boundary,
    and is no loss. The rest are passed to ``on_lost`` with the change, in the order the change
    names them.
    """
    while not stop.is_set():
        try:
            heard = changes.get(timeout=poll_s)
        except queue.Empty:
            continue
        if on_change is not None:
            on_change(heard)
        for rank in heard.change.lost:
            if known(rank):
                continue
            if rank in heard.change.departing:
                log.info("rank %d leaves on notice (epoch %d)", rank, heard.change.epoch)
                continue
            log.warning(
                "rank %d lost by the coordinator's word (epoch %d, %s)",
                rank,
                heard.change.epoch,
                heard.change.outcome.value,
            )
            on_lost(rank, heard)


def pipeline_peers(layout: ParallelLayout, members: MemberSet, me: Member) -> list[int]:
    """The ranks ``me`` sends to and receives from: the other positions of its column."""
    return [position(m) for m in layout.pipeline_members(members, me) if m != me]


def exiting_ranks(
    pids: Mapping[int, int], skip: Collection[int], proc: Path = Path("/proc")
) -> list[int]:
    """The ranks among ``pids`` whose process is exiting, killed, a zombie or gone.

    Read as :func:`everserve.comm.slotwatch.process_state` reads a late member: a SIGKILLed
    CUDA process stays in exit for a second or two while the driver tears its context down, and
    its sentinel fires only at the end of that, so a node that watches its workers' ``/proc``
    sees the loss when it happens. ``skip`` holds the ranks already known lost or taken.
    """
    return [r for r, pid in pids.items() if r not in skip and process_state(pid, proc) == "dead"]
