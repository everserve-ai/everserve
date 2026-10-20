# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A remote loss reaches the executor by the coordinator's word, and a failed pipeline send
names the right peers (decision 0023)."""

import queue
import threading
import time

import pytest

from everserve.comm.layout import ParallelLayout
from everserve.comm.runtime import CHANGE_ABORT, StepAborted
from everserve.comm.shadow import member_sets, replacement
from everserve.contract.daemon import ChangeHeard
from everserve.contract.signals import Change
from everserve.outcome import Outcome
from everserve_vllm.losses import (
    RankLostError,
    failed_step,
    follow_changes,
    loss_behind,
    pipeline_peers,
)
from everserve_vllm.pipeline import MemberSetChanged


def test_a_claimed_change_marks_its_ranks_lost_unless_already_known():
    changes: queue.Queue = queue.Queue()
    dead = {2}  # seen on this node first and reported by this executor
    lost, claimed = [], {}
    stop = threading.Event()
    t = threading.Thread(
        target=follow_changes,
        args=(changes, dead.__contains__, lambda r, h: lost.append((r, h.change.epoch)), stop),
        kwargs={
            "poll_s": 0.05,
            "on_change": lambda h: claimed.update({h.change.epoch: h.change.lost}),
        },
        daemon=True,
    )
    t.start()
    changes.put(ChangeHeard(Change(lost=(2,), outcome=Outcome.REPLACE, epoch=1), 10.0))
    changes.put(ChangeHeard(Change(lost=(4, 5, 6, 7), outcome=Outcome.REPLACE, epoch=2), 11.0))
    for _ in range(100):
        if len(lost) == 4:
            break
        threading.Event().wait(0.02)
    stop.set()
    t.join(2)
    assert lost == [(4, 2), (5, 2), (6, 2), (7, 2)]  # rank 2 was known; node-b's four were not
    assert claimed == {1: (2,), 2: (4, 5, 6, 7)}  # every claim is kept, the known rank's too
    assert not t.is_alive()


def test_the_pipeline_peers_are_the_other_positions_of_the_column():
    layout = ParallelLayout(tensor_parallel=4, pipeline_parallel=2)
    world = tuple(range(8))
    assert pipeline_peers(layout, world, 1) == [5]
    assert pipeline_peers(layout, world, 5) == [1]
    node_b_lost = member_sets(8, losses=(), replacements=[(4, 5, 6, 7)])[0]
    assert pipeline_peers(layout, node_b_lost, 1) == [5]  # the spare in rank 5's position
    assert pipeline_peers(layout, node_b_lost, replacement(5)) == [1]
    assert pipeline_peers(ParallelLayout(4), tuple(range(4)), 1) == []  # one stage: no peers


def test_a_worker_whose_process_is_exiting_is_named_at_once(tmp_path):
    from everserve.comm.slotwatch import PF_EXITING, SIGKILL_BIT
    from everserve_vllm.losses import exiting_ranks

    def proc(pid, state="S", flags=0, pending=0):
        d = tmp_path / str(pid)
        d.mkdir()
        rest = " ".join(["1", "1", "1", "0", "-1", str(flags)] + ["0"] * 40)
        (d / "stat").write_text(f"{pid} (VLLM::Worker_PP1) {state} {rest}\n")
        (d / "status").write_text(f"Name:\tx\nSigPnd:\t{pending:016x}\nShdPnd:\t0000000000000000\n")

    proc(10)  # alive
    proc(11, flags=PF_EXITING)  # the driver still tearing its context down
    proc(12, pending=SIGKILL_BIT)  # killed, the signal not taken yet
    pids = {2: 10, 3: 11, 4: 12, 5: 13}  # 13: gone
    assert exiting_ranks(pids, set(), tmp_path) == [3, 4, 5]
    assert exiting_ranks(pids, {3, 5}, tmp_path) == [4]  # already known lost or taken


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_a_step_a_change_ended_is_the_loss_once_its_word_arrives_within_the_bound():
    """A worker refuses its pipeline group on its daemon link's word before the executor has
    the coordinator's; the refusal is the loss."""
    clock = FakeClock()
    heard_at = 0.4  # the coordinator's word, after the worker's refusal
    text = str(MemberSetChanged())  # what vLLM's worker RPC carries: the exception's text
    assert loss_behind(text, lambda: clock.t >= heard_at, 10.0, 0.02, clock, clock.sleep)
    assert clock.t < 0.5


def test_a_change_abort_with_no_loss_in_the_bound_is_a_failure():
    clock = FakeClock()
    assert not loss_behind(str(StepAborted()), lambda: False, 10.0, 0.5, clock, clock.sleep)
    assert 10.0 <= clock.t <= 10.5


def test_a_failure_that_names_no_change_is_answered_at_once():
    clock = FakeClock()
    waited = []
    assert not loss_behind(
        "CUDA error: an illegal memory access", lambda: True, 10.0, 0.02, clock, waited.append
    )
    assert waited == []


def test_an_unrelated_error_that_quotes_the_marker_is_not_a_change_abort():
    waited = []
    quoted = f"KeyError: 'retry after {CHANGE_ABORT}'"
    assert not loss_behind(quoted, lambda: True, 10.0, 0.02, time.monotonic, waited.append)
    assert waited == []


def test_every_change_abort_opens_its_message_with_the_marker():
    """An exchange abort crossed vLLM's worker RPC as an empty string before it carried one."""
    assert str(StepAborted()) == CHANGE_ABORT
    assert str(StepAborted("rank 3")).startswith(CHANGE_ABORT)
    assert str(MemberSetChanged()).startswith(CHANGE_ABORT)


def test_the_executor_raises_a_refused_step_as_the_loss_with_its_event():
    dead: set[int] = set()
    threading.Timer(0.1, dead.add, args=(1,)).start()
    err, event = failed_step(
        True, 0, str(MemberSetChanged()), lambda: bool(dead), lambda: dead, 2.0
    )
    assert isinstance(err, RankLostError) and "[1]" in str(err)
    assert event is not None and event["kind"] == "step_refused" and event["rank"] == 0


def test_the_executor_fails_the_engine_on_a_change_abort_no_loss_follows_and_says_so():
    err, event = failed_step(True, 0, str(StepAborted()), lambda: False, set, 0.05)
    assert not isinstance(err, RankLostError) and event is None
    assert "no loss reached this executor" in str(err)


def test_the_executor_fails_the_engine_at_once_on_a_step_that_failed_on_its_own():
    err, event = failed_step(True, 0, "CUDA error", lambda: True, lambda: {1}, 10.0)
    assert not isinstance(err, RankLostError) and event is None


def test_a_failure_outside_a_step_is_never_read_as_a_loss():
    err, _ = failed_step(False, 0, str(MemberSetChanged()), lambda: True, lambda: {1}, 10.0)
    assert not isinstance(err, RankLostError)


def test_a_pipeline_transfer_a_change_refused_reports_no_peers_lost(monkeypatch):
    """The report is folded into the pipeline's own wrap of the four methods (A51 over A33), so
    it is on wherever the routing is."""
    import torch

    from everserve_vllm import pipeline

    reported = []
    monkeypatch.setattr(pipeline, "_report_peers_lost", reported.append)
    monkeypatch.setattr(pipeline, "_ORIGINALS", {})
    monkeypatch.setattr(pipeline, "_ORIGINAL_BROADCAST", None)
    monkeypatch.setattr(torch.distributed, "broadcast", torch.distributed.broadcast)

    class Group:
        unique_name, world_size = "pp:0", 3
        fails_with: Exception = MemberSetChanged()

        def make_sibling_device_group(self, group_desc=None):
            return None

    def fail(self, *args, **kwargs):
        raise self.fails_with

    for name in pipeline.PIPELINE_METHODS:
        setattr(Group, name, fail)
    pipeline.install(Group)
    g = Group()
    with pytest.raises(MemberSetChanged):
        g.isend_tensor_dict({})
    assert reported == []  # the change names its own loss; the peers are alive
    g.fails_with = RuntimeError("NCCL error: remote process exited")
    with pytest.raises(RuntimeError, match="NCCL"):
        g.isend_tensor_dict({})
    assert reported == ["isend_tensor_dict: RuntimeError: NCCL error: remote process exited"]


def test_a_rank_leaving_on_notice_is_claimed_but_never_marked_lost():
    changes: queue.Queue = queue.Queue()
    lost, claimed = [], []
    stop = threading.Event()
    t = threading.Thread(
        target=follow_changes,
        args=(changes, lambda r: False, lambda r, h: lost.append(r), stop),
        kwargs={"poll_s": 0.05, "on_change": lambda h: claimed.append(h.change.departing)},
        daemon=True,
    )
    t.start()
    # node-c drained while rank 3 failed: one change, rank 2 departing, rank 3 a fault
    change = Change(lost=(2, 3), outcome=Outcome.REPLACE, epoch=2, departing=(2,))
    changes.put(ChangeHeard(change, 10.0))
    for _ in range(100):
        if lost:
            break
        threading.Event().wait(0.02)
    stop.set()
    t.join(2)
    assert lost == [3] and claimed == [(2,)]
