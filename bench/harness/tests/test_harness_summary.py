# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A trial's summary is scored against its run's pre-registration, named in the summary, never
against numbers held in the module."""

import json
from argparse import Namespace
from pathlib import Path

import pytest
import stock_restart
import summary

REPO = Path(__file__).resolve().parents[3]

PREDICTION_FILES = sorted((REPO / "bench/runs/PLANNED").glob("*/predictions.json"))


@pytest.mark.parametrize("path", PREDICTION_FILES, ids=lambda p: p.parent.name)
def test_every_predictions_file_names_a_pre_registration_that_exists(path):
    d = json.loads(path.read_text())
    assert (REPO / d["preregistration"]).is_file()
    for name, p in d["figures"].items():
        assert set(p) == {"n", "predicted", "kill"}, name


def test_a_figure_carries_the_loaded_pre_registration_s_prediction():
    summary.use_predictions(REPO / "bench/runs/PLANNED/p3-resident/predictions.json")
    f = summary.figure("node_fault_gap_s", 14.28)
    assert f["prediction"] == 8 and f["kill"].startswith("median > 40")
    assert summary.PREREGISTRATION.endswith("p3-resident/PREREGISTRATION.md")
    assert summary.figure("not_a_figure", 1)["prediction"] is None


def test_a_predictions_file_without_its_pre_registration_is_refused(tmp_path):
    bad = tmp_path / "predictions.json"
    bad.write_text(json.dumps({"figures": {}}))
    with pytest.raises(ValueError, match="names its preregistration"):
        summary.use_predictions(bad)


def test_the_stock_restart_splits_its_first_token_at_its_stamps(tmp_path):
    """A node fault trial in round numbers: the RPC timeout, the relaunch, the load, the prefill."""
    summary.use_predictions(REPO / "bench/runs/PLANNED/p3-resident/predictions.json")
    kill, result = tmp_path / "kill.json", tmp_path / "result.json"
    kill.write_text(json.dumps({"t_kill": 1000.0}))
    steps = [{"step": 0, "t_start": 1400.0, "t_end": 1402.5}]
    stamps = {
        "process": 1350.0,
        "vllm_imported": 1360.0,
        "engine_args": 1361.0,
        "engine_ready": 1386.0,
    }
    result.write_text(
        json.dumps({"steps": steps, "tokens": [[1]] * 8, "load_s": 25.0, "stamps": stamps})
    )
    a = Namespace(
        trial=2,
        kill=str(kill),
        result=str(result),
        clocks=None,
        exited="1310.0",
        relaunched="1316.0",
    )
    f = stock_restart.summarize(a)["figures"]
    got = {k: v["value"] for k, v in f.items() if k.startswith("stock_restart")}
    assert got == {
        "stock_restart_first_token_s": 402.5,
        "stock_restart_relaunch_latency_s": 316.0,
        "stock_restart_detection_s": 310.0,
        "stock_restart_supervisor_s": 6.0,
        "stock_restart_relaunch_to_ready_s": 84.0,
        "stock_restart_ready_to_first_token_s": 2.5,
        "stock_restart_engine_ready_s": 25.0,
    }
    assert f["stock_restart_relaunch_to_ready_s"]["parts"] == {
        "relaunch_to_process_s": 34.0,
        "imports_s": 10.0,
        "to_engine_args_s": 1.0,
        "engine_load_s": 25.0,
        "ready_to_first_step_s": 14.0,
    }
    a.exited = None  # a trial recorded before the exit was stamped
    f = stock_restart.summarize(a)["figures"]
    assert "stock_restart_detection_s" not in f and "stock_restart_supervisor_s" not in f


def test_a_stock_arm_s_own_kills_are_reported_apart_from_ours(tmp_path):
    """A stock headless killed at its own stop must not be scored as one of our kills."""
    summary.use_predictions(REPO / "bench/runs/PLANNED/p3-resident-run2/predictions.json")
    reads = []
    for node, exit_kind in (("a", "none"), ("b", "killed"), ("c", "clean")):
        p = tmp_path / f"written_since_stock_1_stopped_{node}.json"
        p.write_text(json.dumps({"exit": exit_kind}))
        reads.append(str(p))
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"steps": [], "tokens": []}))
    a = Namespace(
        arm="stock",
        trial=1,
        result=str(result),
        control="",
        stock="",
        store_metrics="",
        stopped=reads,
    )
    f = summary.summarize(a)["figures"]
    assert f["stops_killed"]["value"] == 0 and f["stops_killed"]["by_node"] == {"c": "clean"}
    assert f["stock_stops_killed"]["value"] == 1
    a.arm = "adapter"
    assert summary.summarize(a)["figures"]["stops_killed"]["value"] == 1


def test_a_kill_is_placed_in_node_a_s_step():
    steps = [
        {"step": 265, "t_start": 10.00, "t_end": 10.03},
        {"step": 266, "t_start": 10.03, "t_end": 10.06},
        {"step": 267, "t_start": 24.40, "t_end": 26.00},  # resumed after the change
    ]
    assert summary.kill_in_step(steps, 10.045) == {
        "step": 266,
        "into_s": 0.015,
        "of_s": 0.03,
        "fraction": 0.5,
    }
    # the step in flight was aborted, so not recorded
    assert summary.kill_in_step(steps, 10.0726) == {
        "after_step": 266,
        "since_end_s": 0.0126,
        "of_s": 0.03,
        "fraction": 0.42,
    }
    assert summary.kill_in_step(steps, None) is None and summary.kill_in_step([], 1.0) is None
