# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The slot watch (decision 0021): a late member is reported at once if its process is exiting,
only after the hang bound if it is alive, and a mark is read only at its own epoch.

The kernel's half (naming the late row) is in ``test_slot_kernel.py``; here a test writes the late
word as the kernel would, and a fake ``/proc`` stands for the processes.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

from everserve.comm.slotkernel import _LATE, Region
from everserve.comm.slotwatch import PF_EXITING, SIGKILL_BIT, SlotWatch, process_state

LIVE, DEAD = 0, 2  # the watching row and the row it waits for


def _proc(root: Path, pid: int, state: str = "S", flags: int = 0, pending: int = 0) -> None:
    d = root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    rest = " ".join(["1", "1", "1", "0", "-1", str(flags)] + ["0"] * 40)
    (d / "stat").write_text(f"{pid} (VLLM::Worker (TP2)) {state} {rest}\n")
    (d / "status").write_text(f"Name:\tx\nSigPnd:\t{pending:016x}\nShdPnd:\t0000000000000000\n")


def test_a_process_is_dead_from_the_moment_it_is_killed_until_it_is_gone(tmp_path):
    _proc(tmp_path, 10)
    _proc(tmp_path, 11, flags=PF_EXITING | 0x40)
    _proc(tmp_path, 12, pending=SIGKILL_BIT)
    _proc(tmp_path, 13, state="Z")
    assert process_state(10, tmp_path) == "alive"
    assert process_state(11, tmp_path) == "dead"  # exiting: the driver may take seconds yet
    assert process_state(12, tmp_path) == "dead"  # killed, the signal not taken yet
    assert process_state(13, tmp_path) == "dead"
    assert process_state(14, tmp_path) == "dead"  # gone
    assert process_state(0, tmp_path) == "unknown"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
def test_this_process_is_alive():
    assert process_state(os.getpid()) == "alive"


@pytest.fixture
def region():
    shm = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    r = Region(str(Path(tempfile.mkdtemp(dir=shm)) / "slot"), create=True)
    r.set_members([0, 1, DEAD], [0, 1, DEAD])
    r.set_bound([0, 1, DEAD], 0.05)
    r.set_pid(DEAD, 4242)
    yield r
    r.close()


def _watch(region, proc, hang_s=10.0):
    reports = []
    w = SlotWatch(
        region,
        LIVE,
        report=lambda rank, why, epoch: reports.append((rank, why, epoch)),
        epoch=lambda: 3,
        hang_s=hang_s,
        proc=proc,
    )
    return w, reports


def test_a_late_member_whose_process_is_exiting_is_reported_at_once(region, tmp_path):
    _proc(tmp_path, 4242, flags=PF_EXITING)
    w, reports = _watch(region, tmp_path)
    seen: dict = {}
    w.check(100.0, seen)
    assert reports == []  # nobody is late
    region.ctrl[LIVE, _LATE] = 1 << DEAD  # the kernel names row 2
    w.check(100.1, seen)
    assert [(r, e) for r, _, e in reports] == [([DEAD], 3)]
    assert "exiting" in reports[0][1]
    assert region.lost_ranks(3) == {DEAD}
    assert region.lost_ranks(4) == set()  # after the change, the mark is about the old member
    w.check(100.2, seen)
    assert len(reports) == 1  # once per episode


def test_a_late_member_that_is_alive_is_slow_until_the_hang_bound(region, tmp_path):
    _proc(tmp_path, 4242)
    w, reports = _watch(region, tmp_path, hang_s=10.0)
    seen: dict = {}
    region.ctrl[LIVE, _LATE] = 1 << DEAD
    w.check(100.0, seen)
    w.check(105.0, seen)
    assert reports == []  # a stall, not a death
    region.ctrl[LIVE, _LATE] = 0  # it arrived
    w.check(105.5, seen)
    assert reports == [] and w.slow == [{"rank": DEAD, "late_s": 5.5, "at": 105.5}]
    region.ctrl[LIVE, _LATE] = 1 << DEAD
    w.check(200.0, seen)
    w.check(210.5, seen)
    assert [(r, e) for r, _, e in reports] == [([DEAD], 3)]
    assert "hung" in reports[0][1]


def test_the_late_row_is_reported_as_the_rank_in_its_position(region, tmp_path):
    spare_row = 3
    region.set_members([0, 1, spare_row], [0, 1, spare_row])  # the spare stands in for rank 2
    region.set_pid(spare_row, 5151)
    _proc(tmp_path, 5151, state="Z")
    w, reports = _watch(region, tmp_path)
    region.ctrl[LIVE, _LATE] = 1 << spare_row
    w.check(1.0, {})
    assert [r for r, _, _ in reports] == [[2]]


def test_a_report_carries_the_epoch_the_lateness_began_in(region, tmp_path):
    _proc(tmp_path, 4242)
    epoch = {"now": 0}
    reports = []
    w = SlotWatch(
        region,
        LIVE,
        report=lambda rank, why, e: reports.append((rank, e)),
        epoch=lambda: epoch["now"],
        proc=tmp_path,
    )
    seen: dict = {}
    region.ctrl[LIVE, _LATE] = 1 << DEAD
    w.check(1.0, seen)  # late at epoch 0, the process not yet exiting
    epoch["now"] = 1  # the change is applied on this worker while its kernel still waits
    _proc(tmp_path, 4242, flags=PF_EXITING)
    w.check(1.1, seen)
    assert reports == [([DEAD], 0)]  # stale to the coordinator, not a loss of the replacement
    assert region.lost_ranks(1) == set()


def test_a_node_lost_at_once_is_one_report_of_every_rank(region, tmp_path):
    region.set_members([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    for row, pid in ((2, 4242), (3, 4343), (4, 4444)):
        region.set_pid(row, pid)
        _proc(tmp_path, pid, flags=PF_EXITING)
    w, reports = _watch(region, tmp_path)
    region.ctrl[LIVE, _LATE] = (1 << 2) | (1 << 3) | (1 << 4)
    w.check(1.0, {})
    assert [(r, e) for r, _, e in reports] == [([2, 3, 4], 3)]  # one change for the set
    assert region.lost_ranks(3) == {2, 3, 4}


def test_the_hang_clock_waits_for_the_first_steps_after_a_change(region, tmp_path):
    _proc(tmp_path, 4242)  # the replacement, alive and restoring
    epoch, steps = {"now": 0}, {"n": 100}
    reports = []
    w = SlotWatch(
        region,
        LIVE,
        report=lambda ranks, why, e: reports.append(ranks),
        epoch=lambda: epoch["now"],
        progress=lambda: steps["n"],
        hang_s=10.0,
        proc=tmp_path,
    )
    seen: dict = {}
    w.check(0.0, seen)
    epoch["now"] = 1  # the change is applied; the replacement's first step waits on its restore
    region.ctrl[LIVE, _LATE] = 1 << DEAD
    w.check(1.0, seen)
    w.check(30.0, seen)
    assert reports == []  # past the hang bound, but no step has completed since the change
    steps["n"] += 2  # the first steps after the change have run: the clock runs again
    w.check(31.0, seen)
    assert reports == [[DEAD]]


def _stall_watch(region, proc, steps):
    region.set_in_flight(1)  # the executor has a step out
    reports = []
    w = SlotWatch(
        region,
        LIVE,
        report=lambda ranks, why, e: reports.append((ranks, why, e)),
        epoch=lambda: 3,
        progress=lambda: steps["n"],
        proc=proc,
    )
    return w, reports


def test_a_death_outside_the_slot_collective_is_found_by_the_stalled_step(region, tmp_path):
    # the survivors wait in a collective the slot does not own: the late word stays clear
    _proc(tmp_path, 4242, flags=PF_EXITING)
    steps = {"n": 7}
    w, reports = _stall_watch(region, tmp_path, steps)
    seen: dict = {}
    w.check(10.00, seen)  # a step ended at 10.00
    w.check(10.04, seen)
    assert reports == []  # within the 50 ms bound
    w.check(10.06, seen)
    assert [(r, e) for r, _, e in reports] == [([DEAD], 3)]
    assert "no step ended" in reports[0][1]
    assert region.lost_ranks(3) == {DEAD}
    region.ctrl[LIVE, _LATE] = 1 << DEAD  # the slot names it too, later: one report in all
    w.check(10.10, seen)
    assert len(reports) == 1


def test_a_stalled_step_with_every_member_alive_reports_nobody(region, tmp_path):
    _proc(tmp_path, 4242)
    for row, pid in ((1, 4141),):
        region.set_pid(row, pid)
        _proc(tmp_path, pid)
    steps = {"n": 7}
    w, reports = _stall_watch(region, tmp_path, steps)
    seen: dict = {}
    w.check(10.0, seen)
    for t in (10.1, 11.0, 30.0):  # a slow step, or an idle engine: alive members are not lost
        w.check(t, seen)
    assert reports == []


def test_steps_that_end_keep_the_stall_rule_quiet(region, tmp_path):
    _proc(tmp_path, 4242, flags=PF_EXITING)  # even a dead process is not looked for
    steps = {"n": 0}
    w, reports = _stall_watch(region, tmp_path, steps)
    seen: dict = {}
    for i in range(10):
        steps["n"] += 1
        w.check(10.0 + 0.03 * i, seen)
    assert reports == []


def test_a_member_that_exits_with_no_step_in_flight_is_not_reported(region, tmp_path):
    # an idle engine, or one shutting down: its workers exit, and nobody is lost
    _proc(tmp_path, 4242, flags=PF_EXITING)
    steps = {"n": 7}
    w, reports = _stall_watch(region, tmp_path, steps)
    region.set_in_flight(0)
    seen: dict = {}
    for t in (10.0, 10.1, 11.0, 20.0):
        w.check(t, seen)
    assert reports == []
    region.set_in_flight(1)  # a step goes out: the stall is timed from here, not from 10.0
    w.check(20.01, seen)
    w.check(20.04, seen)
    assert reports == []
    w.check(20.07, seen)
    assert [(r, e) for r, _, e in reports] == [([DEAD], 3)]
