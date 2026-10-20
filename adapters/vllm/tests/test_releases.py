# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A release of an old group runs on its own thread; the registry reaps it without waiting and
keeps when it started and returned, so a boundary's record can place it against the join."""

import threading
import time

from everserve_vllm.groups import Releases


def test_a_release_is_reaped_when_it_returns_and_its_times_are_kept():
    gate = threading.Event()
    releases = Releases()
    releases.start("fabric", gate.wait)
    releases.start("nothing", None)
    before = time.time()
    assert releases.reap(bound_s=10.0) == [] and releases.running() == ["fabric"]
    times = releases.times()
    assert list(times) == ["fabric"] and len(times["fabric"]) == 1 <= before + 1
    gate.set()
    for _ in range(200):
        if not releases.running():
            break
        time.sleep(0.005)
    reports = releases.reap(bound_s=10.0)
    assert len(reports) == 1 and reports[0].startswith("fabric ") and reports[0].endswith("s")
    start, end = releases.times()["fabric"]
    assert start <= end <= time.time() and releases.done == reports and releases.leaked == 0


def test_a_release_past_its_bound_is_left_behind_and_counted():
    releases = Releases()
    releases.start("torch", threading.Event().wait)
    time.sleep(0.02)
    assert releases.reap(bound_s=0.01) == ["torch still running"]
    assert releases.leaked == 1 and releases.running() == []
    assert len(releases.times()["torch"]) == 1  # started, never returned
