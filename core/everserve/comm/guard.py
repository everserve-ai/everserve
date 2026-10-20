# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A shared/exclusive guard per communicator (decision 0025).

Calls into the communicator take the shared side and its release takes the exclusive side, so
the release never frees the communicator under a call in progress.

A call inside NCCL's host code when its communicator is freed takes the process down (the step
thread in ``ncclSend`` on the refused pipeline communicator, the release's free under it). The
refusal stops new calls; this guard makes the free wait for the ones already in.
An exclusive side that never comes, because a call never leaves, leaves the release running,
which the boundary counts as leaked and never waits out.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class Guard:
    """Any number of shared holders, or one exclusive holder."""

    def __init__(self) -> None:
        """Nobody holds it."""
        self._lock = threading.Lock()
        self._free = threading.Condition(self._lock)
        self._shared = 0

    @contextmanager
    def shared(self) -> Iterator[None]:
        """Hold the shared side for the block; waits while an exclusive holder has it."""
        with self._lock:
            self._shared += 1
        try:
            yield
        finally:
            with self._lock:
                self._shared -= 1
                if self._shared == 0:
                    self._free.notify_all()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the exclusive side for the block.

        Waits until every shared holder has left, and keeps new shared holders out until the
        block ends.
        """
        with self._lock:
            while self._shared:
                self._free.wait()
            yield


_guards: weakref.WeakKeyDictionary[Any, Guard] = weakref.WeakKeyDictionary()
_registry = threading.Lock()


def guard_of(comm: Any) -> Guard:
    """The guard of ``comm``, made on first use and gone with the communicator."""
    with _registry:
        guard = _guards.get(comm)
        if guard is None:
            guard = _guards[comm] = Guard()
        return guard
