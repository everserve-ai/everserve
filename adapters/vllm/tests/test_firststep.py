# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The first-steps probe: armed at a change, it records the next steps and where the thread was."""

import threading
import time

from everserve_vllm import firststep


def _wait_on_a_lock(seconds):
    threading.Event().wait(seconds)  # Python frames in threading.py: a wait the sampler classes


def test_the_probe_records_the_steps_after_a_change_and_classes_the_waits(monkeypatch):
    monkeypatch.setattr(firststep, "_PROBE", None)
    monkeypatch.setattr(firststep, "_DONE", [])
    firststep.step_begin()  # not armed: nothing recorded
    firststep.step_end()
    assert firststep.report() == []

    firststep.arm()
    for _ in range(3):  # the probe takes two steps and ignores the third
        firststep.step_begin()
        _wait_on_a_lock(0.1)
        firststep.step_end()
    (r,) = firststep.report()
    assert [s["begin_s"] < s["end_s"] for s in r["steps"]] == [True, True]
    assert r["steps"][1]["begin_s"] >= r["steps"][0]["end_s"]
    assert r["samples"] > 10
    assert r["classes_s"].get("queue_wait", 0) > 0.1, r["classes_s"]
    assert any("threading.py" in f for f, _ in r["top_frames"]), r["top_frames"]


def test_a_second_change_closes_the_first_probe(monkeypatch):
    monkeypatch.setattr(firststep, "_PROBE", None)
    monkeypatch.setattr(firststep, "_DONE", [])
    firststep.arm()
    firststep.step_begin()
    time.sleep(0.02)
    firststep.arm()  # a change before the first probe saw its second step
    reports = firststep.report()
    assert len(reports) == 2 and reports[0]["steps"] == [] and reports[1]["steps"] == []
