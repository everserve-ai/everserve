# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The pin the executor's reading of a failed step rests on: vLLM's worker loop carries a failed
call's exception as its text, so a change abort reaches the executor with its marker first
(``PRIVATE-INTERNALS.md`` A8a). A pin that changes that transport fails here, not silently in a
fault. Runs where vLLM is installed; the decision itself is tested in ``test_losses.py``."""

from types import SimpleNamespace

import pytest

from everserve.comm.runtime import CHANGE_ABORT, StepAborted

pytest.importorskip("vllm")

from vllm.v1.executor.multiproc_executor import WorkerProc  # noqa: E402

from everserve_vllm.executor import EverserveExecutor, RankLostError  # noqa: E402
from everserve_vllm.pipeline import MemberSetChanged  # noqa: E402


class Queue:
    def __init__(self):
        self.sent = []

    def enqueue(self, item):
        self.sent.append(item)


@pytest.mark.parametrize("error", [MemberSetChanged(), StepAborted("rank 1")])
def test_vllm_carries_a_change_abort_to_the_executor_with_its_marker_first(error):
    q = Queue()
    WorkerProc.enqueue_output(SimpleNamespace(worker_response_mq=q), error)
    [(status, text)] = q.sent
    assert status == WorkerProc.ResponseStatus.FAILURE
    assert text.startswith(CHANGE_ABORT)


def test_the_executor_reads_a_refused_step_as_the_loss_it_holds():
    ex = EverserveExecutor.__new__(EverserveExecutor)
    ex.dead, ex.departing, ex.events, ex.region = {1}, set(), [], None
    assert isinstance(ex._failed(True, 0, MemberSetChanged()), RankLostError)
    assert ex.events[-1]["kind"] == "step_refused"
