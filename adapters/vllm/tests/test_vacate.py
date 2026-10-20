# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Positions a change moved off a node: the leaving worker releases itself after its drain, the
node ends its process, and the node is done when it holds nothing (decision 0033, step 3)."""

import multiprocessing
import signal
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from everserve.comm.slotkernel import Region
from everserve.contract.signals import Take
from everserve_vllm import vacate, worker


@pytest.fixture
def region():
    shm = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    r = Region(str(Path(tempfile.mkdtemp(dir=shm)) / "slot"), create=True)
    yield r
    r.close()


def test_a_row_s_release_is_its_own_word_and_touches_nothing_else(region):
    assert region.released() == set() and not region.ends_released()
    region.mark_released(1)
    region.mark_released(3)
    assert region.released() == {1, 3}
    assert not region.in_flight() and not region.stopped() and not region.ends_released()
    region.set_ends_released()
    assert region.ends_released() and region.released() == {1, 3}


class Link:
    def __init__(self, log, region):
        self.log, self.region = log, region

    def close(self):
        self.log.append(("close", sorted(self.region.released())))


def _leaving(monkeypatch, region, log):
    kv = SimpleNamespace(drain=lambda: log.append(("drain", sorted(region.released()))))
    slot = SimpleNamespace(region=region, row=1)
    monkeypatch.setattr(
        worker,
        "STATE",
        SimpleNamespace(kv=kv, link=Link(log, region), collective=SimpleNamespace(slot=slot)),
    )
    return SimpleNamespace(worker=SimpleNamespace(rank=5))


def test_a_leaving_worker_drains_then_closes_its_link_then_marks_its_row(region, monkeypatch):
    region.set_ends_released()
    log = []
    out = worker.boundary_except(_leaving(monkeypatch, region, log), None, "t", (5,))
    assert out["released"] and out["rank"] == 5
    # the drain before anything (the taker catches up from it), the link before the mark
    assert log == [("drain", []), ("close", [])] and region.released() == {1}


def test_a_leaving_worker_stays_where_its_node_does_not_end_released_processes(region, monkeypatch):
    log = []
    out = worker.boundary_except(_leaving(monkeypatch, region, log), None, "t", (5,))
    assert not out["released"] and log == [("drain", [])] and region.released() == set()


def _as_a_released_worker(death, ready):
    """A vLLM worker past its death pipe: SIGTERM does nothing and the process stays."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.send("up")
    while True:
        time.sleep(0.05)


def _started(target):
    ctx = multiprocessing.get_context("spawn")
    death_reader, death_writer = ctx.Pipe(duplex=False)
    ready_reader, ready_writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(death_reader, ready_writer), daemon=True)
    proc.start()
    death_reader.close()  # the child holds the only reader
    assert ready_reader.poll(30) and ready_reader.recv() == "up"
    return proc, death_writer


def test_a_released_process_is_killed_at_once_whatever_it_does_with_signals():
    proc, death_writer = _started(_as_a_released_worker)
    ended = vacate.end_process(proc, death_writer)
    assert (ended.how, ended.exitcode) == ("killed", -signal.SIGKILL) and not proc.is_alive()
    assert ended.end_s < 2.0 and death_writer.closed


def test_a_released_process_already_gone_is_recorded_as_gone():
    proc, death_writer = _started(_as_a_released_worker)
    proc.kill()
    proc.join()
    assert vacate.end_process(proc, death_writer).how == "gone"


def test_the_taker_s_node_is_the_change_s_take_else_the_policy_s_spare_else_here():
    engine = SimpleNamespace(spare_node=lambda r: {1: "node-d"}.get(r))
    takes = (Take(0, "node-c", 3),)
    assert vacate.taker_node(0, takes, engine, "node-b") == "node-c"
    assert vacate.taker_node(1, takes, engine, "node-b") == "node-d"
    assert vacate.taker_node(2, takes, engine, "node-b") == "node-b"
    assert vacate.taker_node(2, (), None, "node-b") == "node-b"


class Proc:
    def __init__(self, alive):
        self.alive = alive

    def is_alive(self):
        return self.alive


def test_a_node_holds_nothing_once_every_worker_left_and_exited_and_nothing_is_taken_here():
    gone = {0: Proc(False), 1: Proc(False)}
    assert vacate.holds_nothing(gone, {0, 1}, {})
    assert not vacate.holds_nothing(gone, {0}, {})  # rank 1 has not left
    assert not vacate.holds_nothing({0: Proc(False), 1: Proc(True)}, {0, 1}, {})  # still ending
    assert not vacate.holds_nothing(gone, {0, 1}, {2: {"epoch": 1}})  # a spare here holds one
    assert not vacate.holds_nothing({}, set(), {})  # a node that launched nothing has not begun
