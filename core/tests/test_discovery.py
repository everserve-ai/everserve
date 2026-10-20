# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
import pytest
from google.protobuf import json_format

from everserve.discovery import DaemonRecord, FileDiscovery
from everserve.proto import discovery_pb2


def write(dir_, node, **fields):
    msg = discovery_pb2.DaemonRecord(node=node, **fields)
    (dir_ / f"{node}.json").write_text(json_format.MessageToJson(msg))


def test_reads_records_by_node_in_name_order(tmp_path):
    write(tmp_path, "n2", address="/run/n2.sock", engines=["k3-a"], pid=7, started_at_ms=5)
    write(tmp_path, "n1", address="/run/n1.sock", mirror_address="/run/m1.sock")
    (tmp_path / "notes.txt").write_text("ignored")
    fd = FileDiscovery(tmp_path)
    recs = fd.daemons()
    assert [r.node for r in recs] == ["n1", "n2"]
    assert recs[1] == DaemonRecord("n2", "/run/n2.sock", ("k3-a",), 5, 7, "")
    assert fd.daemon("n1").mirror_address == "/run/m1.sock"
    assert fd.daemon("nope") is None


def test_missing_directory_is_empty_and_bad_record_is_an_error(tmp_path):
    assert FileDiscovery(tmp_path / "absent").daemons() == []
    (tmp_path / "n1.json").write_text("{not json")
    with pytest.raises(ValueError, match="not a daemon record"):
        FileDiscovery(tmp_path).daemons()


def test_a_record_s_host_is_where_other_nodes_reach_it():
    assert DaemonRecord("n", "/run/n.sock", (), 0, 0, "").host == "127.0.0.1"
    assert DaemonRecord("n", "10.1.84.61:7400", (), 0, 0, "").host == "10.1.84.61"
    assert DaemonRecord("n", "node-b.internal:7400", (), 0, 0, "").host == "node-b.internal"
    assert DaemonRecord("n", "[fd00::2]:7400", (), 0, 0, "").host == "fd00::2"
    assert DaemonRecord("n", "/run/n.sock", (), 0, 0, "", "10.1.2.3").host == "10.1.2.3"
    assert (
        DaemonRecord.from_message(
            discovery_pb2.DaemonRecord(node="n", address="/run/n.sock", host="10.1.2.4")
        ).host
        == "10.1.2.4"
    )


def test_under_the_pod_contract_the_local_daemon_comes_from_the_environment(tmp_path, monkeypatch):
    from everserve.discovery import local_daemon

    monkeypatch.setenv("EVERSERVE_DAEMON_SOCKET", "/run/everserve/node.sock")
    monkeypatch.setenv("EVERSERVE_HOST_IP", "10.4.2.7")
    r = local_daemon("gpu-node-3", tmp_path)  # no record published there, and none needed
    assert (r.node, r.address, r.mirror_address, r.host) == (
        "gpu-node-3",
        "/run/everserve/node.sock",
        "/run/everserve/mirror.sock",
        "10.4.2.7",
    )
    monkeypatch.setenv("EVERSERVE_MIRROR_SOCKET", "/var/run/store.sock")
    assert local_daemon("gpu-node-3").mirror_address == "/var/run/store.sock"
    monkeypatch.delenv("EVERSERVE_HOST_IP")
    with pytest.raises(ValueError, match="EVERSERVE_HOST_IP"):
        local_daemon("gpu-node-3")


def test_on_bare_metal_the_local_daemon_is_its_published_record(tmp_path, monkeypatch):
    from everserve.discovery import local_daemon

    monkeypatch.delenv("EVERSERVE_DAEMON_SOCKET", raising=False)
    assert local_daemon("node-a", tmp_path) is None and local_daemon("node-a") is None


def test_the_slot_region_lives_under_the_shared_memory_base(monkeypatch):
    from everserve.comm.slotkernel import region_path

    monkeypatch.delenv("EVERSERVE_SHM_DIR", raising=False)
    assert region_path("rig") == "/dev/shm/everserve_slot_rig"
    monkeypatch.setenv("EVERSERVE_SHM_DIR", "/dev/shm/everserve")
    assert region_path("rig") == "/dev/shm/everserve/everserve_slot_rig"
