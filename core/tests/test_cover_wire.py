# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The cover plan on the wire (decision 0035, amended): the plan, the reports and a group's id
through protobuf, the member ids a process keys groups by, and the daemon link delivering a plan
and a group's id to its process."""

import tempfile
import threading
import time
from pathlib import Path

import pytest
from test_daemon_link import FakeDaemon

from everserve.comm.cover import (
    LinkReports,
    PlannedGroup,
    PlannedSet,
    member_id,
    plan_from,
    set_member,
)
from everserve.contract.daemon import DaemonConnection, GroupIds
from everserve.contract.signals import (
    CoverGroup,
    CoverPlan,
    CoverSet,
    GroupFailed,
    GroupFormed,
    GroupId,
    GroupReady,
    GroupReleased,
    Holder,
    Member,
    SpareRef,
    from_message,
    to_message,
)
from everserve.proto import signals_pb2 as pb

D1 = SpareRef(1, "node-d", 0)
NCCL_ID = bytes(range(128))
PLAN = CoverPlan(
    engine="rig",
    generation=7,
    epoch=2,
    placement={0: Holder("node-a", 0), 1: Holder("node-b", 0), 2: Holder("node-c", 0)},
    groups=(
        CoverGroup("domain", (0, 1, 2), 0),
        CoverGroup("fabric", (D1, 2), D1),
    ),
    sets=(
        CoverSet((0, 1, 2), (0,), "current"),
        CoverSet((0, D1, 2), (1,), "REPLACE of ranks [1]"),
    ),
)


def round_trip(sig):
    return from_message(pb.Signal.FromString(to_message(sig).SerializeToString()))


@pytest.mark.parametrize(
    "sig",
    [
        PLAN,
        GroupFormed("rig", 7, (D1, 2), "fabric", 1.25, D1),
        GroupFormed("rig", 7, (0, 1, 2), "domain", 0.0, 2),
        GroupFailed("rig", 7, (0, D1), "domain", "TimeoutError: no id", 0),
        GroupReleased("rig", 8, (D1, 2), "fabric", 2),
        GroupId("rig", 7, "fabric", (D1, 2), NCCL_ID),
        GroupReady("rig", 7, (D1, 2), "fabric", D1),
    ],
)
def test_the_plan_every_report_and_a_group_s_id_survive_the_wire(sig):
    assert round_trip(sig) == sig


def test_a_group_s_id_is_never_shown():
    """It is carried only on the authenticated links: a log line of a signal holds no id."""
    assert NCCL_ID.hex() not in repr(GroupId("rig", 7, "fabric", (D1, 2), NCCL_ID))


def test_a_group_without_a_scope_is_refused():
    msg = to_message(PLAN)
    msg.cover_plan.groups[0].scope = 0
    with pytest.raises(ValueError, match="scope"):
        from_message(msg)


def test_a_spare_s_id_names_its_process_and_maps_back():
    assert member_id(D1) == "R1@node-d:0" and set_member("R1@node-d:0") == D1
    assert member_id(2) == 2 and set_member(2) == 2
    with pytest.raises(ValueError):
        set_member("R1")  # an id without its process never comes from a plan


def test_the_wire_s_plan_is_keyed_by_the_process_ids():
    plan = plan_from(PLAN)
    assert (plan.generation, plan.epoch) == (7, 2)
    assert plan.groups[1] == PlannedGroup("fabric", ("R1@node-d:0", 2), "R1@node-d:0")
    assert plan.sets[1] == PlannedSet((0, "R1@node-d:0", 2), (1,), "REPLACE of ranks [1]")
    assert plan.sets_needing(1) == [(0, "R1@node-d:0", 2)]


class Link:
    def __init__(self):
        self.sent = []

    def report(self, r):
        self.sent.append(r)


def test_reports_name_the_group_by_its_spans_and_the_process_as_it_is_now():
    link, me = Link(), {"id": "R1@node-d:0"}
    reports = LinkReports("rig", lambda: me["id"], link)
    g = plan_from(PLAN).groups[1]
    reports.formed(7, g, 1.5)
    me["id"] = 1  # its commit: the spare is rank 1 now
    reports.released(8, "fabric", (1, 2))
    assert link.sent == [
        GroupFormed("rig", 7, (D1, 2), "fabric", 1.5, D1),
        GroupReleased("rig", 8, (1, 2), "fabric", 1),
    ]


def test_a_group_s_id_is_the_newest_one_heard_and_a_wait_for_one_is_bounded():
    ids = GroupIds()
    ids.put(GroupId("rig", 7, "fabric", (D1, 2), b"a" * 128))
    ids.put(GroupId("rig", 9, "fabric", (D1, 2), b"b" * 128))
    ids.put(GroupId("rig", 8, "fabric", (D1, 2), b"c" * 128))  # older than held: not taken
    assert ids.wait("fabric", (D1, 2), 7, 1.0).id == b"b" * 128
    with pytest.raises(TimeoutError, match="no id for the fabric group"):
        ids.wait("fabric", (D1, 2), 10, 0.05)  # its server never made one at generation 10
    with pytest.raises(TimeoutError):
        ids.wait("domain", (0, 1), 0, 0.05)  # a group with no id heard at all

    # a waiter wakes when the id comes
    got = {}
    t = threading.Thread(target=lambda: got.update(gid=ids.wait("domain", (0, 1), 3, 5.0)))
    t.start()
    time.sleep(0.05)
    ids.put(GroupId("rig", 3, "domain", (0, 1), NCCL_ID))
    t.join(5)
    assert got["gid"].id == NCCL_ID


class Covered:
    """A process's runtime that forms cover: it keeps each plan it is handed."""

    orig_rank = 0

    def __init__(self):
        self.plans = []
        self.got = threading.Event()

    def identity(self):
        return 0, 0

    def on_cover_plan(self, plan):
        self.plans.append(plan)
        self.got.set()


def test_the_daemon_link_carries_a_plan_a_group_s_id_and_the_reports():
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    rt, holder = Covered(), {}
    t = threading.Thread(
        target=lambda: holder.update(link=DaemonConnection(path, "rig", Member("node-a", 0), rt))
    )
    t.start()
    daemon.accept()
    t.join()
    link = holder["link"]
    daemon.send(PLAN)
    assert rt.got.wait(5) and rt.plans == [PLAN]
    # a group's id from the server member, through the daemon: kept for the formation's wait
    gid = GroupId("rig", 7, "fabric", (D1, 2), NCCL_ID)
    daemon.send(gid)
    assert link.group_ids.wait("fabric", (D1, 2), 7, 5.0) == gid
    # this process as a server: its id goes up, and its reports
    link.report_group_id(GroupId("rig", 7, "domain", (0, 1, 2), NCCL_ID))
    assert daemon.signal() == GroupId("rig", 7, "domain", (0, 1, 2), NCCL_ID)
    link.report(GroupFormed("rig", 7, (0, 1, 2), "domain", 0.0, 0))
    assert daemon.signal() == GroupFormed("rig", 7, (0, 1, 2), "domain", 0.0, 0)
    link.close()
