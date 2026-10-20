# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The engine's head on the control plane (decision 0032): which process a stop reaches, why the
head says it stopped, and a change abandoned after its delivery ending the wait for its commit."""

import queue
import threading

import pytest

from everserve.contract.head import ChangeAbandoned, launcher_pid, stop_cause, wait_commit
from everserve.contract.signals import Abandoned, StopCause


def test_a_stop_reaches_the_process_that_launched_the_engine_core():
    spawned = launcher_pid(parent=lambda: object(), pid=lambda: 20, ppid=lambda: 10)
    in_process = launcher_pid(parent=lambda: None, pid=lambda: 20, ppid=lambda: 10)
    assert (spawned, in_process) == (10, 20)


def test_a_head_ended_by_a_failure_says_so_in_its_goodbye():
    """vLLM shuts the engine down in order on a failure; the goodbye must still say FAILED."""
    failure = "Worker failed with error 'pipeline refused its group'"
    assert stop_cause(failure) == (StopCause.FAILED, failure)
    assert stop_cause(None) == (StopCause.STOPPED, "")


def test_the_wait_for_a_commit_returns_the_commit():
    committed: queue.Queue = queue.Queue()
    threading.Timer(0.05, committed.put, args=("epoch 2",)).start()
    assert wait_commit(committed, lambda: None, 2.0, poll_s=0.01) == "epoch 2"


def test_a_change_abandoned_after_its_delivery_ends_the_wait_for_its_commit():
    gone: list[Abandoned] = []
    threading.Timer(0.05, gone.append, args=(Abandoned(epoch=2, detail="rank 0 departed"),)).start()
    with pytest.raises(ChangeAbandoned, match="epoch 2 was abandoned: rank 0 departed") as e:
        wait_commit(queue.Queue(), lambda: gone[0] if gone else None, 2.0, poll_s=0.01)
    assert e.value.epoch == 2


def test_neither_within_the_bound_is_the_queue_s_empty():
    with pytest.raises(queue.Empty):
        wait_commit(queue.Queue(), lambda: None, 0.05, poll_s=0.01)
