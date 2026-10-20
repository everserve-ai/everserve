# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Each member's rendezvous host: a rank's node's daemon record, a spare's launching node."""

import pytest

from everserve.comm.shadow import replacement
from everserve.contract.signals import Member
from everserve.discovery import DaemonRecord
from everserve.policy import Engine
from everserve_vllm.worker import rendezvous_hosts

ENGINE = Engine(
    name="k3-a",
    members=tuple(Member("node-a", d) for d in range(4))
    + tuple(Member("node-b", d) for d in range(4)),
    tensor_parallel=4,
    expert_parallel=1,
    pipeline_parallel=2,
    placement_independent_model=False,
    spare_nodes=("node-c",),
)


def record(node, address):
    return DaemonRecord(node, address, (), 0, 0, "")


def test_ranks_take_their_node_s_host_and_spares_the_launching_node_s():
    records = {
        "node-a": record("node-a", "10.0.0.1:7400"),
        "node-b": record("node-b", "10.0.0.2:7400"),
        "node-c": record("node-c", "10.0.0.3:7400"),
    }
    hosts = rendezvous_hosts(ENGINE, records, {r: "node-c" for r in (4, 5, 6, 7)})
    assert [hosts[r] for r in range(8)] == ["10.0.0.1"] * 4 + ["10.0.0.2"] * 4
    assert {hosts[replacement(r)] for r in (4, 5, 6, 7)} == {"10.0.0.3"}
    # a single node, its daemon on a Unix socket: every host is loopback
    local = {"local": record("local", "/run/everserve/local.sock")}
    single = Engine("rig", tuple(Member("local", d) for d in range(4)), 4, 1, 1, False)
    assert set(rendezvous_hosts(single, local, {2: "local"}).values()) == {"127.0.0.1"}


def test_a_missing_record_is_refused_before_any_rendezvous():
    records = {"node-a": record("node-a", "10.0.0.1:7400"), "node-b": None, "node-c": None}
    with pytest.raises(RuntimeError, match="node-b"):
        rendezvous_hosts(ENGINE, records, {})
    records["node-b"] = record("node-b", "10.0.0.2:7400")
    with pytest.raises(RuntimeError, match="node-c"):
        rendezvous_hosts(ENGINE, records, {4: "node-c"})
    assert rendezvous_hosts(ENGINE, records, {})[7] == "10.0.0.2"


def test_the_bootstrap_every_process_derives_from_the_policy_agrees(tmp_path):
    from pathlib import Path

    from google.protobuf import json_format

    from everserve.comm.shadow import rendezvous_base_port, replacement
    from everserve.policy import load
    from everserve.proto import discovery_pb2
    from everserve_vllm.worker import bootstrap_from_policy

    policy_path = Path(__file__).parents[3] / "docs/examples/policy-two-node-managed.json"
    policy = load(policy_path)
    for node, host in (("node-a", "10.0.0.1"), ("node-b", "10.0.0.2"), ("node-c", "10.0.0.3")):
        rec = discovery_pb2.DaemonRecord(node=node, address=f"/run/{node}.sock", host=host)
        (tmp_path / f"{node}.json").write_text(json_format.MessageToJson(rec))
    spares = {s.rank: s.node for s in policy.engine("k3-a").spares}
    head = bootstrap_from_policy(policy, "k3-a", "node-a", str(tmp_path), spares)
    spare = bootstrap_from_policy(policy, "k3-a", "node-c", str(tmp_path), spares)
    # the same sets, in the same order, on the same ports: formed without being told
    assert head.sets == spare.sets and head.base_port == spare.base_port
    assert head.base_port == rendezvous_base_port("k3-a")
    assert head.layout.tensor_parallel == 4 and head.layout.pipeline_parallel == 2
    assert head.nodes == ("node-a",) * 4 + ("node-b",) * 4
    assert head.hosts[0] == "10.0.0.1" and head.hosts[replacement(6)] == "10.0.0.3"
    assert head.node == "node-a" and spare.node == "node-c"
    # the release bound from the engine entry ("releaseBoundMs": 10000 in the example), in
    # seconds, the same for every process
    assert head.release_bound_s == spare.release_bound_s == 10.0


def _placed(tmp_path):
    """The example's engine both ways: pinned with discovery, and shape-only with a placement."""
    from dataclasses import replace
    from pathlib import Path

    from google.protobuf import json_format

    from everserve.contract.signals import EnginePlacement, RankPlacement, SparePlacement
    from everserve.policy import load
    from everserve.proto import discovery_pb2

    path = Path(__file__).parents[3] / "docs/examples/policy-two-node-managed.json"
    pinned = load(path)
    hosts = {"node-a": "10.0.0.1", "node-b": "10.0.0.2", "node-c": "10.0.0.3"}
    for node, host in hosts.items():
        rec = discovery_pb2.DaemonRecord(node=node, address=f"/run/{node}.sock", host=host)
        (tmp_path / f"{node}.json").write_text(json_format.MessageToJson(rec))
    spec = pinned.engine("k3-a")
    # the same engine unpinned, beside the same policy-wide fields: the bootstrap itself, apart
    # from the loader, which takes an unpinned engine only under cover (whose sets are planned)
    unpinned = replace(spec, members=(), spares=(), spare_launch=None, replica_nodes=())
    shape = replace(pinned, engines=(unpinned,))
    placement = EnginePlacement(
        engine="k3-a",
        epoch=0,
        members=tuple(
            RankPlacement(r, m.node, None, hosts[m.node]) for r, m in enumerate(spec.members)
        ),
        spares=tuple(SparePlacement(s.rank, s.node, s.device, hosts[s.node]) for s in spec.spares),
    )
    return pinned, shape, placement, {s.rank: s.node for s in spec.spares}


def test_an_engine_bootstraps_alike_from_its_placement_and_from_a_policy_that_pins_it(tmp_path):
    from dataclasses import replace

    from everserve_vllm.worker import bootstrap_from_placement, bootstrap_from_policy

    pinned, shape, placement, spares = _placed(tmp_path)
    for node in ("node-a", "node-c"):
        by_policy = bootstrap_from_policy(pinned, "k3-a", node, str(tmp_path), spares)
        by_placement = bootstrap_from_placement(shape, "k3-a", node, placement)
        assert by_placement == replace(by_policy, discovery_dir="")


def test_a_spare_for_any_position_stands_for_every_rank_off_its_node(tmp_path):
    from dataclasses import replace

    from everserve.comm.shadow import replacement
    from everserve_vllm.worker import bootstrap_from_placement

    _, shape, placement, _ = _placed(tmp_path)
    anyone = replace(placement, spares=(replace(placement.spares[0], for_rank=None),))
    cfg = bootstrap_from_placement(shape, "k3-a", "node-a", anyone)
    assert {m for m in cfg.hosts if m not in range(8)} == {replacement(r) for r in range(8)}
    assert cfg.hosts[replacement(0)] == "10.0.0.3"


def test_a_placement_that_misses_a_rank_or_a_host_is_refused(tmp_path):
    from dataclasses import replace

    from everserve_vllm.worker import bootstrap_from_placement

    _, shape, placement, _ = _placed(tmp_path)
    with pytest.raises(ValueError, match="not each of its 8 once"):
        bootstrap_from_placement(
            shape, "k3-a", "node-a", replace(placement, members=placement.members[:7])
        )
    twice = placement.members[:7] + (placement.members[0],)
    with pytest.raises(ValueError, match="not each of its 8 once"):
        bootstrap_from_placement(shape, "k3-a", "node-a", replace(placement, members=twice))
    unaddressed = (replace(placement.members[0], host=""),) + placement.members[1:]
    with pytest.raises(ValueError, match=r"no host for \[0\]"):
        bootstrap_from_placement(shape, "k3-a", "node-a", replace(placement, members=unaddressed))
