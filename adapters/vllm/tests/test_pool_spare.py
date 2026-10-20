# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A pool spare (decision 0031): its command line, one standing per engine, the first take
deciding the engine, a second take refused, and the withdrawal from the others at the commit."""

import threading
import time
from types import SimpleNamespace

import pytest

from everserve.contract.signals import Take


def parse(*argv):
    from everserve_vllm.spare_main import parse as parse_args

    base = ["--rank", "", "--node", "node-d", "--policy", "p.json", "--discovery-dir", "/d"]
    base += ["--model", "m", "--tensor-parallel-size", "1", "--pipeline-parallel-size", "3"]
    return parse_args([*base, *argv])


def test_a_pool_spare_stands_for_its_engines_and_needs_any_position_and_residence():
    a = parse("--pool", "rig-pool", "--engines", "k3-a,k3-b", "--resident")
    assert a.pool_engines == ["k3-a", "k3-b"] and a.engine == "k3-a"
    with pytest.raises(SystemExit, match="names no engines"):
        parse("--pool", "rig-pool", "--resident")
    with pytest.raises(SystemExit, match="--resident"):
        parse("--pool", "rig-pool", "--engines", "k3-a")
    assert parse("--engine", "k3-a").pool_engines == ["k3-a"]  # a spare for one engine, as before
    with pytest.raises(SystemExit, match="--engine"):
        parse()


def _pool(monkeypatch):
    from everserve_vllm import collective_op, settings, spare_main, worker

    log = []
    standings = {}

    def stand(device, cfg, take, slot_choice, identity, collective):
        s = SimpleNamespace(link=f"link {cfg.engine}", collective=collective, take=take)
        standings[cfg.engine] = s
        return s, {"stands_for": ["R1@node-d:0"]}

    monkeypatch.setattr(worker, "stand", stand)
    monkeypatch.setattr(collective_op, "standing", lambda eligible: f"collective {eligible}")
    monkeypatch.setattr(collective_op, "adopt", lambda c: log.append(("adopt", c)))
    monkeypatch.setattr(worker, "adopt_cover", lambda e: log.append(("cover", e)))
    monkeypatch.setattr(settings, "update", lambda **kw: log.append(("settings", kw["region"])))
    monkeypatch.setattr(
        spare_main,
        "_take_position",
        lambda load, engine, spec, node, taken, standing, device, cfg, shard, startup_for, answer: (
            answer.put(f"runtime of {engine}") or (taken.rank, "pair", "proc", {"rank": taken.rank})
        ),
    )
    committed = []
    monkeypatch.setattr(worker, "on_next_commit", committed.append)
    monkeypatch.setattr(worker, "withdraw", lambda e, c, link: log.append(("withdraw", e, link)))
    engines = [
        spare_main.PoolEngine(n, None, (1, 2), SimpleNamespace(engine=n), None)
        for n in ("k3-a", "k3-b")
    ]
    return spare_main, engines, standings, log, committed


def test_the_first_take_decides_the_engine_and_the_others_are_withdrawn_at_its_commit(
    monkeypatch,
):
    spare_main, engines, standings, log, committed = _pool(monkeypatch)
    got = {}

    def link_thread():
        deadline = time.monotonic() + 5
        while "k3-b" not in standings:
            if time.monotonic() > deadline:
                return  # never stood: the test fails on what the link got, not in a hang
            time.sleep(0.01)
        got["runtime"] = standings["k3-b"].take(Take(2, "node-d", 0))  # k3-b's change names it

    t = threading.Thread(target=link_thread, daemon=True)
    t.start()
    out = spare_main.stand_and_take_pool(
        None, lambda: "cuda:0", engines, "node-d", "shard", lambda name, r: "calls"
    )
    t.join(5)
    assert out[:3] == ("k3-b", 2, "link k3-b") and got["runtime"] == "runtime of k3-b"
    assert log == [
        ("adopt", "collective (1, 2)"),
        ("cover", "k3-b"),
        ("settings", "/dev/shm/everserve_slot_k3-b"),
    ]
    assert len(committed) == 1  # withdrawn only once the take's commit has applied
    committed[0]()
    assert log[-1] == ("withdraw", "k3-a", "link k3-a")
    with pytest.raises(RuntimeError, match="took a position of k3-b"):
        standings["k3-a"].take(Take(1, "node-d", 0))


def test_a_spare_reads_a_policy_document_or_is_handed_the_engine_s_never_both():
    from everserve_vllm.spare_main import parse as parse_args

    base = ["--engine", "k", "--rank", "5", "--node", "gpu-3", "--model", "m"]
    a = parse_args([*base, "--given", "--node-devices", "0,1"])
    assert (a.given, a.policy, a.discovery_dir, a.node_devices) == (True, None, "", "0,1")
    for bad in ([], ["--given", "--policy", "p.json"]):
        with pytest.raises(SystemExit):
            parse_args([*base, *bad])
