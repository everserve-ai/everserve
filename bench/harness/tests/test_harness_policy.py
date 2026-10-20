# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The drill policy's shape: members, spares and the concurrent losses a run asks for."""

import json

import policy as twonode_policy

SHAPE = [
    "policy.py",
    "--tp",
    "1",
    "--pp",
    "3",
    "--members",
    "a:1",
    "b:1",
    "c:1",
    "--spares",
    "d:1",
    "e:1",
    "--model",
    "/workspace/models/M",
    "--max-model-len",
    "4608",
    "--blocks",
    "2400",
    "--max-num-seqs",
    "8",
    "--policy-path",
    "bench/harness/policies/run3.json",
]


def _written(tmp_path, *extra):
    out = tmp_path / "policy.json"
    assert twonode_policy.main([*SHAPE, *extra, "--out", str(out)]) == 0
    return json.loads(out.read_text())


def test_spares_on_two_nodes_stand_in_for_the_last_two_ranks_in_order(tmp_path):
    p = _written(tmp_path)
    (engine,) = p["engines"]
    assert [m["node"] for m in engine["members"]] == ["node-a", "node-b", "node-c"]
    assert [(s["node"], s["device"], s["rank"]) for s in engine["spares"]] == [
        ("node-d", 0, 1),
        ("node-e", 0, 2),
    ]


def test_concurrent_losses_default_to_the_spares_and_a_run_may_ask_for_fewer(tmp_path):
    assert _written(tmp_path)["maxConcurrentLosses"] == 2
    assert _written(tmp_path, "--max-concurrent-losses", "1")["maxConcurrentLosses"] == 1


def test_a_run_may_have_the_coordinator_plan_its_cover(tmp_path):
    assert "cover" not in _written(tmp_path)
    assert _written(tmp_path, "--cover")["cover"] == {"enabled": True}
