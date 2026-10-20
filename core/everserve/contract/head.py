# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The engine's head on the control plane (decision 0032): registered, abandoned, and its end.

The head is the process that owns the engine's scheduler (vLLM's engine core, for the vLLM
adapter). Its daemon connection registers as the head (``ROLE_HEAD``), so the coordinator holds
the engine alive while it is connected and marks it down when it is not. Three things every
adapter's head decides, kept here apart from any engine:

- which process a stop is sent to (:func:`launcher_pid`): the one the operator launched, which
  tears the engine core down in order; the engine core itself when it runs in that process;
- why the head stopped (:func:`stop_cause`), said in its goodbye: ``FAILED`` with the failure the
  executor raised when one ended it (vLLM shuts the engine down in order on a failure, so a
  goodbye without a cause would read ``STOPPED``), ``STOPPED`` otherwise. A failure raised in
  vLLM's scheduler and not the executor
  reads ``STOPPED``: the engine core's exit path is not pinned;
- a change abandoned after its delivery (:class:`ChangeAbandoned`): the membership the survivors
  applied will never commit, so the engine fails (:func:`wait_commit` ends a wait for that
  commit on it).
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import time
from collections.abc import Callable
from typing import Any

from everserve.contract.signals import Abandoned, StopCause


class ChangeAbandoned(RuntimeError):
    """A change failed after its delivery: the engine is down (decision 0032)."""

    def __init__(self, abandoned: Abandoned) -> None:
        """The failure for ``abandoned``, its epoch kept."""
        self.epoch = abandoned.epoch
        super().__init__(f"the change at epoch {abandoned.epoch} was abandoned: {abandoned.detail}")


def launcher_pid(
    parent: Callable[[], Any] = multiprocessing.parent_process,
    pid: Callable[[], int] = os.getpid,
    ppid: Callable[[], int] = os.getppid,
) -> int:
    """The process a stop is sent to: the engine core's parent when vLLM spawned it, else this.

    vLLM starts its engine core as a ``multiprocessing`` child of the process the operator
    launched (the server, or a drill); a ``SIGTERM`` to the child alone makes the parent read a
    dead engine, not a stop. With multiprocessing off the engine core runs in the launcher.
    """
    return ppid() if parent() is not None else pid()


def stop_cause(failure: str | None) -> tuple[StopCause, str]:
    """The goodbye's cause and detail: the failure that ended the engine, or an orderly stop."""
    return (StopCause.FAILED, failure) if failure else (StopCause.STOPPED, "")


def wait_commit(
    committed: queue.Queue,
    abandoned: Callable[[], Abandoned | None],
    timeout_s: float,
    poll_s: float = 0.05,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """The next commit from ``committed``, unless the change is abandoned first.

    Raises:
        ChangeAbandoned: ``abandoned`` returned the change's abandonment before its commit.
        queue.Empty: neither within ``timeout_s``.
    """
    deadline = clock() + timeout_s
    while True:
        gone = abandoned()
        if gone is not None:
            raise ChangeAbandoned(gone)
        left = deadline - clock()
        if left <= 0:
            raise queue.Empty
        try:
            return committed.get(timeout=min(poll_s, left))
        except queue.Empty:
            continue
