# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare stops on SIGTERM or SIGINT, standing or serving, and closes what it holds as it leaves.

The daemon stops its spare with SIGTERM. A spare with no handler for either signal is torn
rather than closed on SIGTERM, and still running 20 s after a SIGINT once it has taken a position.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from everserve_vllm import spare_main, worker

CHILD = textwrap.dedent(
    """
    import queue, sys, time
    from everserve_vllm.spare_main import until_stopped

    out = sys.argv[1]
    mode = sys.argv[2]

    def run():
        open(out, "a").write("running\\n")
        if mode == "standing":  # the main thread waits on its link's queue for the take
            queue.Queue().get()
        else:  # serving: the worker's loop, polling its broadcast queue
            while True:
                time.sleep(0.01)

    def close():
        open(out, "a").write("closing\\n")
        time.sleep(0.3)  # a second signal here must not cut the close short
        open(out, "a").write("closed\\n")

    sys.exit(until_stopped(run, close))
    """
)


def _started(out: Path) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        if out.exists() and "running" in out.read_text():
            return
        time.sleep(0.02)
    raise AssertionError("the child never ran")


@pytest.mark.parametrize("mode", ["standing", "serving"])
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_stop_ends_the_spare_where_it_waits_and_closes_it_once(tmp_path, mode, sig):
    out = tmp_path / "out"
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(out), mode])
    _started(out)
    child.send_signal(sig)
    time.sleep(0.1)
    child.send_signal(sig)  # a second stop, during the close
    assert child.wait(timeout=20) == 0
    assert out.read_text().splitlines() == ["running", "closing", "closed"]


def test_a_spare_whose_run_returns_closes_too_and_exits_zero():
    seen = []
    assert spare_main.until_stopped(lambda: seen.append("ran"), lambda: seen.append("closed")) == 0
    assert seen == ["ran", "closed"]


def test_an_exit_the_spare_asks_for_itself_is_not_taken_for_a_stop():
    def run():
        raise SystemExit("a spare for any position holds the model in host memory: --resident")

    closed = []
    with pytest.raises(SystemExit):
        spare_main.until_stopped(run, lambda: closed.append(True))
    assert closed == [True]


class _Recorder:
    def __init__(self, calls: list[str], name: str) -> None:
        self._calls, self._name = calls, name

    def stop(self) -> None:
        self._calls.append(f"{self._name}.stop")

    def shutdown(self) -> None:
        self._calls.append(f"{self._name}.shutdown")

    def close(self) -> None:
        self._calls.append(f"{self._name}.close")


def test_the_close_stops_the_watch_then_the_worker_then_the_mirror_and_the_link_last(monkeypatch):
    calls: list[str] = []

    class State:
        watch = _Recorder(calls, "watch")
        kv = _Recorder(calls, "kv")
        link = _Recorder(calls, "state_link")

    monkeypatch.setattr(worker, "STATE", State())
    held = spare_main.Held(proc=_Recorder(calls, "proc"), link=_Recorder(calls, "link"))
    spare_main.close(held)
    assert calls == ["watch.stop", "proc.shutdown", "kv.close", "link.close"]


def test_a_spare_stopped_while_standing_closes_the_link_it_registered_on(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(worker, "STATE", None)
    spare_main.close(spare_main.Held(link=_Recorder(calls, "link")))
    assert calls == ["link.close"]
