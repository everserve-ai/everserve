# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The worker held to its engine's cover plans (decision 0035): a plan that comes before the cover,
the commit's rename and releases, and a bootstrap that takes no sets from the policy."""

import dataclasses
from pathlib import Path

from google.protobuf import json_format

from everserve.comm.collective import Collective
from everserve.comm.handle import LocalStandIn
from everserve.comm.shadow import replacement
from everserve.contract.signals import CoverPlan
from everserve.policy import load
from everserve.proto import discovery_pb2
from everserve_vllm import worker

D1 = replacement(1, "node-d", 0)
E2 = replacement(2, "node-e", 0)


def plan(generation, epoch=0):
    return CoverPlan("rig", generation, epoch, {}, ())


class Cover:
    def __init__(self):
        self.applied, self.resumed = [], 0
        self.forgotten = []

    def forget(self, group):
        self.forgotten.append(group)
        return True

    def apply(self, p):
        self.applied.append(p.generation)

    def resume(self, committed=None):
        self.resumed += 1
        self.committed = committed


def test_a_plan_before_the_cover_is_kept_the_newest_only_then_applied(monkeypatch):
    monkeypatch.setattr(worker, "COVERS", {})
    monkeypatch.setattr(worker, "_EARLY_PLANS", {})
    worker.cover_plan(plan(3))
    worker.cover_plan(plan(2))  # older: not kept over the newer
    assert worker._EARLY_PLANS["rig"].generation == 3
    cover = Cover()
    worker.COVERS["rig"] = cover
    worker.cover_plan(plan(4))
    assert cover.applied == [4]


def test_a_pool_spare_s_plans_go_to_their_engine_s_cover(monkeypatch):
    a, b = Cover(), Cover()
    monkeypatch.setattr(worker, "COVERS", {"k3-a": a, "k3-b": b})
    monkeypatch.setattr(worker, "_EARLY_PLANS", {})
    worker.cover_plan(CoverPlan("k3-b", 5, 0, {}, ()))
    worker.cover_plan(CoverPlan("k3-a", 2, 0, {}, ()))
    assert (a.applied, b.applied) == ([2], [5])


class Releases:
    def __init__(self):
        self.started = []

    def start(self, name, run):
        self.started.append(name)


class Refusable:
    def refuse(self):
        return lambda: None


def test_the_commit_renames_the_taker_releases_stale_groups_and_resumes_the_cover(monkeypatch):
    c = Collective(0, LocalStandIn(), {(0, D1, 2): Refusable(), (0, 1, E2): Refusable()})
    c.members = (0, D1, 2)  # rank 1 moved to node-d's spare
    c.releases = Releases()
    cover = Cover()
    monkeypatch.setattr(worker, "COVER", cover)
    assert worker.after_commit(c, [1]) == 1
    assert c.members == (0, 1, 2) and c.shadow_sets == [(0, 1, 2)]
    assert c.releases.started == ["stale domain 0-1-R2@node-e:0"] and cover.resumed == 1
    assert cover.committed == c.epoch  # the cover hears the epoch committed


def test_under_a_cover_plan_the_bootstrap_takes_no_sets_from_the_policy(tmp_path):
    policy_path = Path(__file__).parents[3] / "docs/examples/policy-two-node-managed.json"
    policy = load(policy_path)
    for node, host in (("node-a", "10.0.0.1"), ("node-b", "10.0.0.2"), ("node-c", "10.0.0.3")):
        rec = discovery_pb2.DaemonRecord(node=node, address=f"/run/{node}.sock", host=host)
        (tmp_path / f"{node}.json").write_text(json_format.MessageToJson(rec))
    spares = {s.rank: s.node for s in policy.engine("k3-a").spares}
    planned = dataclasses.replace(policy, cover_planned=True)
    cfg = worker.bootstrap_from_policy(planned, "k3-a", "node-a", str(tmp_path), spares)
    assert cfg.sets == () and cfg.cover_planned
    assert worker.bootstrap_from_policy(policy, "k3-a", "node-a", str(tmp_path), spares).sets


def test_a_stage_of_one_holds_its_stand_in_with_the_planned_pipeline_group():
    from everserve.comm.collective import GroupFamily
    from everserve.comm.layout import ParallelLayout, Scope
    from everserve_vllm.collective_op import VllmCollective

    c = VllmCollective(
        0,
        LocalStandIn(),
        {},
        None,
        spare_row=1,
        layout=ParallelLayout(1, 3),
        fabric=GroupFamily(Scope.FABRIC, LocalStandIn()),
    )
    c.members = (0, 1, 2)
    c.keep("fabric", (0, D1, 2), "pipeline group")
    assert c.has_shadow((0, D1, 2)) and c.replacement_set([1]) == (0, D1, 2)
    assert set(c.held()) == {("fabric", (0, D1, 2))}  # the stand-in is no group
    c.drop("fabric", (0, D1, 2))
    assert not c.has_shadow((0, D1, 2)) and c.shadow_sets == []


def test_the_set_a_staged_change_moves_to_is_served_on():
    from everserve.comm.collective import GroupFamily
    from everserve.comm.layout import ParallelLayout, Scope
    from everserve_vllm.collective_op import VllmCollective

    c = VllmCollective(
        0,
        LocalStandIn(),
        {},
        None,
        spare_row=1,
        layout=ParallelLayout(1, 3),
        fabric=GroupFamily(Scope.FABRIC, LocalStandIn()),
    )
    c.members = (0, 1, 2)
    c.keep("fabric", (0, D1, 2), "pipeline group")
    c.change(c.replacement_set([1]), 1)  # staged until the boundary
    assert ("fabric", (0, D1, 2)) in c.serving()


def test_a_stale_set_s_group_another_set_still_holds_is_kept_at_the_commit(monkeypatch):
    """Decision 0035, amended: a group several sets need is held under each. A set the commit
    finds stale (it names the lost rank 1) may share its group with a set that is not: the group
    stays, and goes only with the last set holding it."""
    shared = Refusable()
    c = Collective(
        0,
        LocalStandIn(),
        {(0, D1, 2): Refusable(), (0, 1, E2): shared, (0, D1, E2): shared},
    )
    c.members = (0, D1, 2)
    c.releases = Releases()
    cover = Cover()
    monkeypatch.setattr(worker, "COVER", cover)
    worker.after_commit(c, [1])
    # (0, 1, E2) names rank 1 and is set aside; its group is (0, D1, E2)'s too, renamed (0, 1, E2)
    # at the commit and still held: nothing released, nothing forgotten
    assert c.releases.started == [] and cover.forgotten == []
    assert c.held()[("domain", (0, 1, E2))] is shared


class Link:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Planned:
    """A planned group, released by refusing it beside serving."""

    def refuse(self):
        return lambda: None


class Closable(Cover):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


def test_a_pool_spare_withdraws_from_its_other_engine_link_cover_and_groups(monkeypatch):
    other = Closable()
    monkeypatch.setattr(worker, "COVERS", {"k3-b": other})
    monkeypatch.setattr(worker, "_EARLY_PLANS", {})
    held = Planned()
    c = Collective(D1, LocalStandIn(), {(0, D1, 2): held}, stands_for=(D1,))
    c.releases = Releases()
    link = Link()
    assert worker.withdraw("k3-b", c, link) == 1
    assert link.closed and other.closed and "k3-b" not in worker.COVERS
    assert c.releases.started == ["withdrawn domain 0-R1@node-d:0-2"]


def test_a_withdrawal_releases_a_group_two_sets_hold_once(monkeypatch):
    monkeypatch.setattr(worker, "COVERS", {})
    monkeypatch.setattr(worker, "_EARLY_PLANS", {})
    shared = Planned()
    c = Collective(D1, LocalStandIn(), {(0, D1, 2): shared, (0, D1, 3): shared}, stands_for=(D1,))
    c.releases = Releases()
    assert worker.withdraw("k3-b", c, Link()) == 1 and len(c.releases.started) == 1


def test_a_withdrawal_waits_for_the_take_s_commit_and_runs_once(monkeypatch):
    monkeypatch.setattr(worker, "_AFTER_COMMIT", [])
    ran = []
    worker.on_next_commit(lambda: ran.append("withdrawn"))
    assert ran == []
    while worker._AFTER_COMMIT:  # what apply_change does once the commit has applied
        worker._AFTER_COMMIT.pop(0)()
    assert ran == ["withdrawn"] and worker._AFTER_COMMIT == []
