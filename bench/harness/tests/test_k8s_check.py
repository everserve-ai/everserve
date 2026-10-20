# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The Kubernetes test's checks pass the records they should and fail the ones they should not."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "k8s"))
import check  # noqa: E402

MEMBERS = {
    "0": ["a", "10.0.0.1"],
    "1": ["a", "10.0.0.1"],
    "2": ["b", "10.0.0.2"],
    "3": ["b", "10.0.0.2"],
}


def logs(tmp_path, recs):
    for pod, lines in recs.items():
        (tmp_path / f"{pod}.log").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return check.records(tmp_path)


def good():
    head = [{"asked": [0, 1], "host": "10.0.0.1"}, {"told": "worker"}, {"members": MEMBERS}]
    worker = [{"asked": [2, 3], "host": "10.0.0.2"}, {"told": "worker"}, {"members": MEMBERS}]
    spare = [{"asked": [2, 3], "host": "10.0.0.3"}, {"told": "spare"}, {"members": MEMBERS}]
    return {"probe-head-0": head, "probe-nodes-0": worker, "probe-nodes-1": spare}


def test_one_server_per_stage_and_every_rank_placed_by_its_server_s_host(tmp_path):
    recs = logs(tmp_path, good())
    check.decided(recs, 4, 2)
    check.placed(recs, 4)
    check.reserved(recs, "probe-nodes-0", [2, 3])


def test_two_servers_of_a_stage_fail(tmp_path):
    g = good()
    g["probe-nodes-1"][1] = {"told": "worker"}
    with pytest.raises(SystemExit):
        check.decided(logs(tmp_path, g), 4, 2)


def test_a_placement_missing_a_rank_or_naming_another_host_fails(tmp_path):
    g = good()
    g["probe-nodes-1"][2] = {"members": {k: v for k, v in MEMBERS.items() if k != "3"}}
    with pytest.raises(SystemExit):
        check.placed(logs(tmp_path, g), 4)
    g = good()
    g["probe-head-0"][2] = {"members": {**MEMBERS, "3": ["c", "10.0.0.3"]}}
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit):
        check.placed(logs(other, g), 4)


def test_a_probe_refused_for_its_step_passes_and_one_told_a_role_fails(tmp_path, capsys):
    why = 'probe is "other" here, "probe" on the engine'  # the e2e's own wording
    recs = logs(tmp_path, {"probe-odd-0": [{"asked": [2, 3]}, {"refused": why}]})
    check.refused(recs, "probe-odd-0")
    assert json.loads(capsys.readouterr().out) == {"refused": "probe-odd-0", "why": why}
    for bad in (
        [{"asked": [2, 3]}, {"told": "spare"}],  # never refused
        [{"asked": [2, 3]}, {"refused": 'the guarantee is "" here, "exact" in the policy'}],
        [{"asked": [2, 3]}, {"refused": why}, {"told": "worker"}],  # and told a role besides
    ):
        recs = logs(tmp_path, {"probe-odd-0": bad})
        with pytest.raises(SystemExit):
            check.refused(recs, "probe-odd-0")


def test_a_probe_that_stands_for_a_served_stage_passes_and_a_second_server_fails(tmp_path):
    recs = logs(tmp_path, good())
    check.stood(recs, "probe-nodes-1", [2, 3])
    with pytest.raises(SystemExit):
        check.stood(recs, "probe-nodes-0", [2, 3])


def test_a_refused_probe_is_no_part_of_the_engine_the_other_checks_read(tmp_path):
    """Replayed on a run's final logs (``bench/runs/2026-10-14-k8s-e2e``), "decided" and "placed"
    failed on probe-odd-0: refused, it neither stands nor hears a placement, and is no member of
    the engine."""
    recs = good()
    recs["probe-odd-0"] = [{"asked": [2, 3]}, {"refused": 'probe is "other" here'}]
    recs = logs(tmp_path, recs)
    check.decided(recs, 4, 2)
    check.placed(recs, 4)


def test_a_pod_s_exit_is_read_from_its_captured_record_wherever_kubernetes_keeps_it(tmp_path):
    def pod(**statuses):
        (tmp_path / "probe-odd-0.pod.json").write_text(
            json.dumps({"status": {"containerStatuses": [statuses]}})
        )

    pod(lastState={"terminated": {"exitCode": 2}}, state={"running": {}})  # restarted since
    check.exited(tmp_path, "probe-odd-0", 2)
    pod(state={"terminated": {"exitCode": 2}})  # waiting to restart
    check.exited(tmp_path, "probe-odd-0", 2)
    for statuses in ({"state": {"running": {}}}, {"lastState": {"terminated": {"exitCode": 1}}}):
        pod(**statuses)
        with pytest.raises(SystemExit):
            check.exited(tmp_path, "probe-odd-0", 2)
    (tmp_path / "probe-odd-0.pod.json").unlink()
    with pytest.raises(SystemExit):  # never captured: undecided, so failed
        check.exited(tmp_path, "probe-odd-0", 2)
