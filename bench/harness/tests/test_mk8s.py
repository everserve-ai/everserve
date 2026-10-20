# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A rented cluster is always taken down, an interrupted take-down included, and a cluster being
deleted is never adopted by the next run."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1]))
import mk8s  # noqa: E402


class FakeNebius:
    """The CLI's mk8s calls over an in-memory cloud; deletions finish after ``ticks`` lists."""

    def __init__(self, clusters=(), groups=(), ticks=2):
        self.clusters = {c["metadata"]["id"]: c for c in clusters}
        self.groups = {g["metadata"]["id"]: g for g in groups}
        self.ticks = ticks
        self.calls = []
        self.parent = "project-e00test"

    def _age(self):
        for store in (self.groups, self.clusters):
            for rid, r in list(store.items()):
                if r["status"]["state"] == "DELETING":
                    r["left"] -= 1
                    if r["left"] <= 0:
                        del store[rid]

    def run(self, *args, parent=True, timeout_s=0, text=False):
        self.calls.append(args)
        kind, verb = args[1], args[2]
        if verb == "list":
            self._age()
            if kind == "cluster":
                return {"items": list(self.clusters.values())}
            owner = args[args.index("--parent-id") + 1]
            return {"items": [g for g in self.groups.values() if g["owner"] == owner]}
        if verb == "delete":
            rid = args[args.index("--id") + 1]
            store = self.clusters if kind == "cluster" else self.groups
            if store[rid]["status"]["state"] == "DELETING":
                raise RuntimeError("the resource is being deleted")
            store[rid]["status"]["state"] = "DELETING"
            store[rid]["left"] = self.ticks
            return {}
        raise AssertionError(f"unexpected call {args}")

    def subnet(self):
        return "vpcsubnet-test"

    cpu_limit = None

    def quota_limit(self, name):
        assert name == mk8s.NON_GPU_VCPU
        return self.cpu_limit


GPU = {"platform": "gpu-l40s-a", "preset": "1gpu-8vcpu-32gb", "nodes": 3}


def cluster(cid, state="RUNNING"):
    return {"metadata": {"id": cid, "name": "everserve-e2e"}, "status": {"state": state}}


def group(gid, owner, state="RUNNING"):
    return {"metadata": {"id": gid}, "owner": owner, "status": {"state": state}, "left": 2}


def test_a_take_down_interrupted_after_its_node_groups_finishes_the_cluster(monkeypatch):
    monkeypatch.setattr(mk8s.time, "sleep", lambda s: None)
    nb = FakeNebius([cluster("c1")], [group("g1", "c1", "DELETING"), group("g2", "c1")])
    out = mk8s.down(nb, "everserve-e2e")
    assert out == {"down": "everserve-e2e", "cluster": "c1"}
    deletes = [a for a in nb.calls if a[2] == "delete"]
    # the group already going is not asked again; the cluster goes once no group is left
    assert deletes == [
        ("mk8s", "node-group", "delete", "--id", "g2"),
        ("mk8s", "cluster", "delete", "--id", "c1"),
    ]
    assert nb.clusters["c1"]["status"]["state"] == "DELETING"


def test_a_cluster_already_being_deleted_is_left_to_finish(monkeypatch):
    nb = FakeNebius([cluster("c1", "DELETING")])
    nb.clusters["c1"]["left"] = 5
    assert mk8s.down(nb, "everserve-e2e")["already"] is True
    assert not [a for a in nb.calls if a[2] == "delete"]


def test_a_run_never_adopts_the_last_run_s_cluster_while_it_is_deleted(monkeypatch):
    monkeypatch.setattr(mk8s.time, "sleep", lambda s: None)
    nb = FakeNebius([cluster("old", "DELETING")])
    nb.clusters["old"]["left"] = 3
    created = {}

    def run(*args, parent=True, timeout_s=0, text=False):
        if args[1:3] == ("cluster", "create"):
            created["after"] = dict(nb.clusters)  # what was there when the create went out
            raise SystemExit("created")  # the rest of up() is not this test's concern
        return FakeNebius.run(nb, *args, parent=parent, timeout_s=timeout_s, text=text)

    nb.run = run
    a = SimpleNamespace(name="everserve-e2e", version="1.34", **GPU)
    try:
        mk8s.up(nb, a)
    except SystemExit:
        pass
    assert created["after"] == {}  # the old cluster was gone before the new one was asked for


def test_a_run_never_adopts_a_cluster_whose_node_groups_are_being_deleted(monkeypatch):
    monkeypatch.setattr(mk8s.time, "sleep", lambda s: None)
    # the last run's take-down, between its node groups' deletion and the cluster's
    nb = FakeNebius([cluster("old")], [group("g1", "old", "DELETING")], ticks=2)
    created = {}

    def run(*args, parent=True, timeout_s=0, text=False):
        if args[1:3] == ("cluster", "create"):
            created["after"] = dict(nb.clusters)
            raise SystemExit("created")
        if args[1:3] == ("cluster", "list") and not nb.groups and "old" in nb.clusters:
            nb.run_delete_cluster()  # the take-down goes on to the cluster once its groups are gone
        return FakeNebius.run(nb, *args, parent=parent, timeout_s=timeout_s, text=text)

    def delete_cluster():
        nb.clusters["old"]["status"]["state"] = "DELETING"
        nb.clusters["old"]["left"] = 1

    nb.run_delete_cluster = delete_cluster
    nb.run = run
    try:
        mk8s.up(nb, SimpleNamespace(name="everserve-e2e", version="1.34", **GPU))
    except SystemExit:
        pass
    assert created["after"] == {}  # waited out to the end, then created afresh


def test_a_watchdog_outlives_its_driver_stopped_as_a_whole(tmp_path):
    import os
    import signal
    import subprocess
    import time

    harness = Path(__file__).parents[1]
    record = tmp_path / "watchdog.txt"
    # a driver in a process group of its own starts a watchdog the way the drivers do, then
    # waits; the watchdog writes its session and group, then keeps waiting for its --at
    driver = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys, time; sys.path.insert(0, sys.argv[1]); import nebius\n"
            "if not nebius.detach():\n"
            "    open(sys.argv[2], 'w').write(f'{os.getsid(0)} {os.getpgrp()} {os.getpid()}')\n"
            "    time.sleep(30); os._exit(0)\n"
            "time.sleep(30)",
            str(harness),
            str(record),
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not record.exists() or not record.read_text():
            assert time.monotonic() < deadline, "the watchdog never ran"
            time.sleep(0.05)
        sid, pgid, pid = map(int, record.read_text().split())
        assert sid != driver.pid and pgid != driver.pid  # out of the driver's session and group
        os.killpg(driver.pid, signal.SIGKILL)  # the driver stopped as a whole, no trap run
        driver.wait(timeout=5)
        os.kill(pid, 0)  # the watchdog is still there
    finally:
        if record.exists() and record.read_text():
            try:
                os.kill(int(record.read_text().split()[2]), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cpu_nodes_a_region_has_no_quota_for_are_refused_before_anything_is_rented():
    """The cloud never refuses such a node group: its create waits until the CLI's timeout, with
    no reason (15 minutes in a region allowing no CPU VMs)."""
    import pytest

    nb = FakeNebius()
    nb.cpu_limit = 0
    a = SimpleNamespace(name="e", version="1.34", platform="cpu-d3", preset="8vcpu-32gb", nodes=3)
    with pytest.raises(RuntimeError, match="need 24 vCPUs .* allows 0"):
        mk8s.up(nb, a)
    assert nb.calls == []  # nothing was asked of the cloud
    nb.cpu_limit = 24
    mk8s.cpu_room(nb, a)  # room for exactly three
    nb.cpu_limit = 0
    mk8s.cpu_room(nb, SimpleNamespace(**GPU))  # GPU nodes count against GPU quotas


RECORDED_CREATE_ASYNC = Path(__file__).parent / "fixtures" / "nebius-node-group-create-async.txt"
"""`nebius mk8s node-group create --async`'s answer as the CLI gave it, recorded."""


def test_the_node_group_create_s_answer_is_read_as_the_text_the_cli_gives(tmp_path, monkeypatch):
    """--async answers with an operation's id as text, not JSON, and a harness reading JSON stops
    on it. The CLI here prints the answer recorded from the real one, through the harness's own
    call."""
    import nebius
    import pytest

    cli = tmp_path / "nebius"
    cli.write_text(f"#!/bin/sh\ncat {RECORDED_CREATE_ASYNC}\n")
    cli.chmod(0o755)
    monkeypatch.setattr(nebius, "CLI", str(cli))
    nb = nebius.Nebius("everserve", "project-e00test")
    args = ("mk8s", "node-group", "create", "--parent-id", "c1", "--async")
    assert nb.run(*args, parent=False, text=True) == RECORDED_CREATE_ASYNC.read_text()
    with pytest.raises(ValueError):  # read as JSON
        nb.run(*args, parent=False)


class Provisioning:
    """A node group the cloud is placing: PROVISIONING for ``ticks`` reads, then ``end``."""

    def __init__(self, ticks, end="RUNNING"):
        self.ticks, self.end = ticks, end

    def run(self, *args, parent=True, timeout_s=0, text=False):
        if args[1:3] == ("node-group", "list"):
            return {"items": [{"metadata": {"id": "ng1"}, "status": {"state": "PROVISIONING"}}]}
        if args[1:3] == ("node-group", "get"):
            self.ticks -= 1
            state = "PROVISIONING" if self.ticks > 0 else self.end
            return {"metadata": {"id": "ng1"}, "status": {"state": state, "node_count": 0}}
        raise AssertionError(f"unexpected call {args}")

    def project(self):
        return {"spec": {"region": "eu-north1"}}


def test_a_node_group_the_cloud_cannot_place_is_named_after_the_run_s_bound(monkeypatch):
    """3 x gpu-l40s can wait 30 minutes, quota free, the cloud silent; the bound names it."""
    import pytest

    monkeypatch.setattr(mk8s.time, "sleep", lambda s: None)
    a = SimpleNamespace(nodes=3, platform="gpu-l40s-a", preset="1gpu-8vcpu-32gb", nodes_within_s=0)
    with pytest.raises(RuntimeError, match=r"no capacity for 3 x gpu-l40s-a .* in eu-north1"):
        mk8s.nodes_running(Provisioning(ticks=10**6), "c1", a)
    a.nodes_within_s = 3600
    assert mk8s.nodes_running(Provisioning(ticks=3), "c1", a)["status"]["state"] == "RUNNING"


RECORDED_UNAVAILABLE = Path(__file__).parent / "fixtures" / "nebius-unavailable.txt"


RECORDED_NOT_FOUND = Path(__file__).parent / "fixtures" / "nebius-not-found.txt"


class Cli:
    """``subprocess.run`` of the CLI: each call's answer in turn, the real CLI's words."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        code, out, err = self.answers.pop(0)
        return SimpleNamespace(returncode=code, stdout=out, stderr=err)


def test_a_repeatable_call_the_api_refuses_for_a_moment_is_made_again(monkeypatch):
    """A DNS timeout to the API (the real CLI's words, recorded) on a release leaves a build VM
    up with the build exiting 0. A repeatable call is made again on a bound; then its error
    stands."""
    import nebius
    import pytest

    error = RECORDED_UNAVAILABLE.read_text().strip()
    slept = []
    nb = nebius.Nebius("everserve", "project-e00test", sleep=slept.append)
    cli = Cli([(1, "", error), (1, "", error), (0, '{"id": "vm1"}', "")])
    monkeypatch.setattr(nebius.subprocess, "run", cli)
    assert nb.run("compute", "instance", "get", "vm1", parent=False) == {"id": "vm1"}
    assert slept == list(nebius.RUN_WAITS_S[:2])  # two failures, two waits, then the answer

    cli.answers, slept[:] = [(1, "", error)] * 5, []
    with pytest.raises(RuntimeError, match="Unavailable"):
        nb.run("compute", "instance", "get", "vm1", parent=False)
    assert slept == list(nebius.RUN_WAITS_S)  # every wait spent, then the error stands


def test_a_create_is_made_once_and_another_failure_is_never_made_again(monkeypatch):
    import nebius
    import pytest

    unavailable = RECORDED_UNAVAILABLE.read_text().strip()
    nb = nebius.Nebius("everserve", "project-e00test", sleep=lambda s: None)
    cli = Cli([(1, "", unavailable)])
    monkeypatch.setattr(nebius.subprocess, "run", cli)
    with pytest.raises(RuntimeError, match="Unavailable"):
        nb.run("compute", "instance", "create", "--name", "vm1")
    assert len(cli.calls) == 1  # its answer lost, the run finds what it made by name
    cli.answers = [(1, "", RECORDED_NOT_FOUND.read_text().strip())]
    with pytest.raises(RuntimeError, match="NotFound"):
        nb.run("compute", "instance", "get", "vm1", parent=False)
    assert len(cli.calls) == 2  # not transient: once


def test_a_delete_whose_first_answer_was_lost_finds_it_deleted_and_is_done(monkeypatch):
    import nebius

    nb = nebius.Nebius("everserve", "project-e00test", sleep=lambda s: None)
    cli = Cli(
        [
            (1, "", RECORDED_UNAVAILABLE.read_text().strip()),
            (1, "", RECORDED_NOT_FOUND.read_text().strip()),
        ]
    )
    monkeypatch.setattr(nebius.subprocess, "run", cli)
    assert nb.run("compute", "instance", "delete", "vm1", parent=False) == {}
