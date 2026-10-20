# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The step guard on one process with fake groups: recover, re-run, agree, quarantine, log, and
the three outcomes as inputs."""

import json
import threading
import time
from dataclasses import dataclass

import pytest
import torch
from collective_doubles import FakeGroup

from everserve.comm.collective import Collective
from everserve.comm.group import ShrinkMethod
from everserve.comm.runtime import Runtime, StepAborted
from everserve.comm.shadow import replacement
from everserve.outcome import Outcome


def make(tmp_path, world=4, shadows=None, **kw):
    live = FakeGroup(ranks=list(range(world)))
    col = Collective(0, live, shadows or {})
    rt = Runtime(0, world, torch.device("cpu"), col, event_log=tmp_path, **kw)
    return rt, live


def test_healthy_step_returns_output_and_counts(tmp_path):
    rt, _ = make(tmp_path)
    out = rt.run_step(lambda: rt.all_reduce(torch.ones(2)))
    assert out.tolist() == [4.0, 4.0]
    assert rt.profile.steps == 1 and rt.events == []


def test_fault_while_idle_recovers_at_once_by_fallback_shrink(tmp_path):
    rt, _ = make(tmp_path)
    seen = {}
    rt.recover_callbacks.append(lambda dead: seen.setdefault("dead", dead) and {"gained": [7]})
    attempts = []
    rt.attempt_callbacks.append(lambda: attempts.append(len(rt.alive)))
    rt.on_fault(2)
    # idle between steps: the fault thread recovered immediately, the group moved and shrank
    assert rt.fault is None and rt.epoch == 1 and rt.alive == [0, 1, 3]
    assert rt.collective.members == (0, 1, 3) and rt.collective.epoch == 1
    out = rt.run_step(lambda: rt.all_reduce(torch.ones(1)))
    assert out.item() == 3.0
    assert rt.alive == [0, 1, 3] and rt.fault is None and seen["dead"] == 2
    assert attempts == [3]
    assert [e.kind for e in rt.events] == ["shrink", "recover"]
    assert rt.events[0].fields["method"] == "recreate"
    assert rt.events[1].fields["gained"] == [7]
    dumped = json.loads((tmp_path / "rank0.json").read_text())
    assert dumped[-1]["kind"] == "profile"


def test_shrink_uses_a_preformed_survivor_set_when_there_is_one(tmp_path):
    survivors = FakeGroup(ranks=[0, 1, 3], method=ShrinkMethod.SHADOW)
    rt, live = make(tmp_path, shadows={(0, 1, 3): survivors})
    rt.on_fault(2)
    assert rt.collective.current is survivors
    out = rt.run_step(lambda: rt.all_reduce(torch.ones(1)))
    assert out.item() == 3.0 and rt.alive == [0, 1, 3]
    assert rt.events[0].fields["method"] == "shadow"


def test_replace_keeps_the_world_and_skips_survivor_absorb(tmp_path):
    rep = FakeGroup(ranks=[0, 1, 2, 3], method=ShrinkMethod.SHADOW)
    rt, _ = make(tmp_path, shadows={(0, 1, replacement(2), 3): rep})
    absorbed = []
    rt.recover_callbacks.append(lambda dead: absorbed.append(dead))
    rt.on_fault(2, outcome=Outcome.REPLACE)
    assert rt.collective.members == (0, 1, replacement(2), 3)
    out = rt.run_step(lambda: rt.all_reduce(torch.ones(1)))
    assert out.item() == 4.0  # constant world size: the spare holds rank 2
    assert rt.alive == [0, 1, 2, 3] and rt.collective.current is rep
    assert absorbed == []
    assert [e.kind for e in rt.events] == ["replace", "recover"]
    assert rt.events[1].fields["outcome"] == "replace"


def test_replace_without_a_preformed_set_is_an_error(tmp_path):
    rt, _ = make(tmp_path)
    with pytest.raises(KeyError):
        rt.on_fault(2, outcome=Outcome.REPLACE)


def test_rebuild_is_not_a_runtime_change(tmp_path):
    rt, _ = make(tmp_path)
    with pytest.raises(ValueError):
        rt.on_fault(2, outcome=Outcome.REBUILD)


def test_fault_mid_step_aborts_rerun_and_quarantines_buffers(tmp_path):
    rt, _ = make(tmp_path)
    calls = {"n": 0}

    def step():
        calls["n"] += 1
        x = torch.ones(3)
        if calls["n"] == 1:
            rt.on_fault(1)  # the fault lands while this attempt is running
        rt.all_reduce(x)
        return x

    out = rt.run_step(step)
    assert calls["n"] == 2 and out.tolist() == [3.0, 3.0, 3.0]
    assert [e.kind for e in rt.events] == ["shrink", "recover", "redo"]
    assert rt.events[-1].fields["attempts"] == 1
    assert rt._quarantine == {}  # the abort came before any collective was issued
    assert rt._attempt_buffers == []


def test_collective_failure_without_a_fault_is_a_real_error(tmp_path):
    rt, live = make(tmp_path, fault_confirm_timeout_s=0.05)
    live.fail_times = 1
    with pytest.raises(RuntimeError, match="peer went away"):
        rt.run_step(lambda: rt.all_reduce(torch.ones(1)))


def test_collective_failure_followed_by_a_fault_is_retried(tmp_path):
    rt, live = make(tmp_path, fault_confirm_timeout_s=2.0)
    live.fail_times = 1
    threading.Timer(0.1, rt.on_fault, args=(3,)).start()
    out = rt.run_step(lambda: rt.all_reduce(torch.ones(1)))
    assert out.item() == 3.0 and rt.alive == [0, 1, 2]
    assert [e.kind for e in rt.events] == ["shrink", "recover", "redo"]
    assert list(rt._quarantine) == [0] and len(rt._quarantine[0]) == 1  # the epoch-0 buffer


def test_on_fault_is_idempotent(tmp_path):
    rt, _ = make(tmp_path)
    rt.on_fault(2)
    rt.on_fault(2)
    assert rt.epoch == 1 and [e.kind for e in rt.events] == ["shrink", "recover"]


def test_all_reduce_refuses_while_a_fault_is_pending(tmp_path):
    rt, _ = make(tmp_path)
    rt._in_step = True  # a step is running on the main thread: recovery waits for it
    rt.on_fault(1)
    assert rt.fault == (1,)
    with pytest.raises(StepAborted):
        rt.all_reduce(torch.ones(1))
    rt._in_step = False


def test_notice_is_logged(tmp_path):
    rt, _ = make(tmp_path)
    rt.on_notice(3, 1_700_000_000_000)
    assert rt.events[0].kind == "notice" and rt.events[0].fields["rank"] == 3


@dataclass
class SplitGroup(FakeGroup):
    """After a shrink, the first one-element all-reduce reports one member a step behind."""

    behind_once: bool = False

    def all_reduce(self, t):
        self.calls += 1
        if self.behind_once and t.numel() == 1:
            self.behind_once = False
            t.mul_(len(self.ranks)).sub_(1.0)
            return
        t.mul_(len(self.ranks))

    def shrink(self, lost, me, epoch):
        return SplitGroup(
            ranks=[r for r in self.ranks if r not in lost],
            method=ShrinkMethod.RECREATE,
            behind_once=True,
        )


def test_recovery_reconciles_an_accepted_step_a_peer_will_rerun(tmp_path):
    live = SplitGroup(ranks=[0, 1, 2, 3])
    rt = Runtime(0, 4, torch.device("cpu"), Collective(0, live, {}), event_log=tmp_path)
    forwards = []
    attempts = []
    commits = []
    rt.attempt_callbacks.append(lambda: attempts.append(1))
    rt.commit_callbacks.append(lambda: commits.append(1))

    def step():
        forwards.append(1)
        return rt.all_reduce(torch.ones(1))

    rt.run_step(step)  # accepted here; a peer is about to re-run it
    assert rt.accepted == 1 and len(forwards) == 1
    rt.on_fault(2)
    out = rt.run_step(step)
    # the recovery replayed the accepted step once (forward, commit, agreement), output discarded,
    # then the new step ran: two more forwards, two more commits, accepted counts only the new one
    assert len(forwards) == 3 and len(commits) == 3 and len(attempts) == 3
    assert rt.accepted == 2 and out.item() == 3.0
    kinds = [e.kind for e in rt.events]
    assert kinds == ["shrink", "recover", "reconcile"] or kinds == [
        "shrink",
        "reconcile",
        "recover",
    ]
    assert any(e.kind == "recover" and e.fields.get("reconciled") is True for e in rt.events)


def test_recovery_without_a_split_does_not_replay(tmp_path):
    rt, _ = make(tmp_path)
    forwards = []

    def step():
        forwards.append(1)
        return rt.all_reduce(torch.ones(1))

    rt.run_step(step)
    rt.on_fault(2)
    rt.run_step(step)
    assert len(forwards) == 2
    assert all(e.kind != "reconcile" for e in rt.events)
    assert any(e.kind == "recover" and e.fields.get("reconciled") is False for e in rt.events)


def test_reconciliation_can_be_turned_off(tmp_path):
    live = SplitGroup(ranks=[0, 1, 2, 3])
    rt = Runtime(
        0,
        4,
        torch.device("cpu"),
        Collective(0, live, {}),
        event_log=tmp_path,
        reconcile_on_recover=False,
    )
    forwards = []

    def step():
        forwards.append(1)
        return rt.all_reduce(torch.ones(1))

    rt.run_step(step)
    rt.on_fault(2)
    rt.run_step(step)
    assert len(forwards) == 2 and "reconciled" not in rt.events[-1].fields


@dataclass
class MoveOnFirstCallGroup(FakeGroup):
    """The fault lands between the runtime's check and the collective: the first all-reduce
    moves the collective (as the fault thread would) before reducing, which must be refused."""

    runtime: object = None
    moved: bool = False

    def all_reduce(self, t):
        self.calls += 1
        if not self.moved:
            self.moved = True
            self.runtime.on_fault(2)  # the fault thread's work, interleaved exactly here
        t.mul_(len(self.ranks))


def test_a_fault_between_the_check_and_the_collective_aborts_the_step(tmp_path):
    live = MoveOnFirstCallGroup(ranks=[0, 1, 2, 3])
    rt = Runtime(0, 4, torch.device("cpu"), Collective(0, live, {}), event_log=tmp_path)
    live.runtime = rt
    rt._in_step = True  # the step is running, so on_fault leaves recovery to the step thread
    attempts = []

    def step():
        attempts.append(len(rt.alive))
        return rt.all_reduce(torch.ones(1))

    out = rt.run_step(step)
    # the first attempt's collective was refused (the collective had moved under it), the step
    # recovered and re-ran on the survivors, and the result is the three-member sum
    assert attempts == [4, 3] and out.item() == 3.0
    assert rt.alive == [0, 1, 3] and rt.fault is None
    assert [e.kind for e in rt.events] == ["shrink", "recover", "redo"]
    assert live.calls == 1, "the stale call never reached the old group"


def test_a_step_waits_for_a_recovery_running_on_the_fault_thread(tmp_path):
    """Idle when the fault lands, the fault thread recovers; a step requested meanwhile must not
    start until that recovery, including its reconciliation collective, has finished."""
    gate = threading.Event()  # the reconciliation "waits for peers" until this is set

    @dataclass
    class SlowReconcileGroup(FakeGroup):
        def all_reduce(self, t):
            self.calls += 1
            if t.numel() == 1 and t.item() != 1.0:  # the accepted-step exchange, not the ones
                gate.wait(5)
            t.mul_(len(self.ranks))

        def shrink(self, lost, me, epoch):
            return SlowReconcileGroup(
                ranks=[r for r in self.ranks if r not in lost], method=ShrinkMethod.RECREATE
            )

    live = SlowReconcileGroup(ranks=[0, 1, 2, 3])
    rt = Runtime(0, 4, torch.device("cpu"), Collective(0, live, {}), event_log=tmp_path)
    rt.run_step(lambda: rt.all_reduce(torch.ones(1)))  # accepted = 1, so the exchange carries 1.0
    rt.accepted = 2  # make the exchange tensor distinguishable from the agreement's ones
    fault_thread = threading.Thread(target=rt.on_fault, args=(2,))
    fault_thread.start()
    time.sleep(0.1)
    forwards = []

    def step():
        forwards.append(1)
        return rt.all_reduce(torch.ones(1))

    result = {}
    step_thread = threading.Thread(target=lambda: result.setdefault("out", rt.run_step(step)))
    step_thread.start()
    time.sleep(0.2)
    assert forwards == [], "the step started while recovery was still reconciling"
    gate.set()
    fault_thread.join(5)
    step_thread.join(5)
    assert forwards == [1] and result["out"].item() == 3.0
    assert rt.alive == [0, 1, 3]
