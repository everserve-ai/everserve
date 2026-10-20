# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The members' step words: the stall gate on a node without the engine (decision 0021), the
region named for the engine, and the stop word the node's shutdown sets."""

import tempfile
from pathlib import Path

import pytest

from everserve.comm.slotkernel import Region, region_path
from everserve.comm.slotwatch import PF_EXITING, SlotWatch


@pytest.fixture
def region():
    shm = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    r = Region(str(Path(tempfile.mkdtemp(dir=shm)) / "slot"), create=True)
    r.set_members([0, 1], [0, 1])
    yield r
    r.close()


def test_a_step_is_in_flight_while_any_member_runs_one_or_died_inside_one(region):
    assert not region.in_flight()
    region.begin_step(1)
    assert region.in_flight()
    region.end_step(1)
    assert not region.in_flight() and region.steps(1) == 1
    region.begin_step(0)  # a member that dies here never answers: the step stays in flight
    assert region.in_flight()
    region.clear_watch([0])  # until the change clears its word
    assert not region.in_flight() and region.steps(0) == 0


def test_the_engine_s_count_and_the_members_words_are_one_gate(region):
    region.set_in_flight(2)
    assert region.in_flight()
    region.set_in_flight(0)
    assert not region.in_flight()


def test_the_region_is_named_for_the_engine():
    assert region_path("rig") == "/dev/shm/everserve_slot_rig"


def test_the_stop_word_ends_every_watch_at_its_next_look(region):
    w = SlotWatch(region, 0, report=lambda *a: None, epoch=lambda: 0, poll_s=0.001)
    w.start()
    region.set_stop()
    w._thread.join(timeout=2.0)
    assert not w._thread.is_alive()


def _proc(root: Path, pid: int, flags: int = 0) -> None:
    d = root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    rest = " ".join(["1", "1", "1", "0", "-1", str(flags)] + ["0"] * 40)
    (d / "stat").write_text(f"{pid} (VLLM::Worker) S {rest}\n")
    (d / "status").write_text("Name:\tx\nSigPnd:\t0000000000000000\nShdPnd:\t0000000000000000\n")


def test_a_report_names_the_engine_rank_and_the_region_the_position(region, tmp_path):
    # the second stage's region: positions 0 and 1 are engine ranks 2 and 3
    region.set_bound([0, 1], 0.05)
    region.set_pid(1, 4343)
    _proc(tmp_path, 4343, flags=PF_EXITING)
    reports = []
    w = SlotWatch(
        region,
        0,
        report=lambda ranks, why, e: reports.append((ranks, why, e)),
        epoch=lambda: 1,
        progress=lambda: 7,
        proc=tmp_path,
        first_rank=2,
    )
    region.begin_step(1)  # the member at position 1 received a step and never answers
    seen: dict = {}
    w.check(10.00, seen)
    w.check(10.06, seen)
    assert [(r, e) for r, _, e in reports] == [([3], 1)]
    assert "ranks [3]" in reports[0][1]
    assert region.lost_ranks(1) == {1}
