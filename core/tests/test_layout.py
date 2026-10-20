# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The parallel layout: which members of a set a rank's groups span, the rendezvous plan each
scope's groups meet on, and a fabric family that moves with the collective at a change."""

import pytest
from collective_doubles import FakeGroup

from everserve.comm.collective import Collective, GroupFamily
from everserve.comm.handle import LocalStandIn
from everserve.comm.layout import ParallelLayout, Scope, position
from everserve.comm.shadow import member_sets, rendezvous_plan, replacement

TP4_PP2 = ParallelLayout(tensor_parallel=4, pipeline_parallel=2)
WORLD = tuple(range(8))
NODE_B_LOST = member_sets(8, losses=(), replacements=[(4, 5, 6, 7)])[0]  # the spare node's set


def test_positions_stages_and_columns_follow_the_engine_layout():
    assert position(5) == 5 and position(replacement(5)) == 5
    assert [TP4_PP2.stage(r) for r in WORLD] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [TP4_PP2.column(r) for r in WORLD] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert TP4_PP2.stage(replacement(6)) == 1 and TP4_PP2.column(replacement(6)) == 2
    assert TP4_PP2.world == 8
    with pytest.raises(ValueError):
        ParallelLayout(0, 1)


def test_a_rank_s_tensor_group_is_its_stage_and_its_pipeline_group_its_column():
    assert TP4_PP2.tensor_members(WORLD, 5) == (4, 5, 6, 7)
    assert TP4_PP2.pipeline_members(WORLD, 5) == (1, 5)
    # with node-b's four positions taken by spares, rank 1's column has the spare for rank 5
    assert TP4_PP2.pipeline_members(NODE_B_LOST, 1) == (1, replacement(5))
    assert TP4_PP2.tensor_members(NODE_B_LOST, replacement(6)) == tuple(
        replacement(r) for r in (4, 5, 6, 7)
    )
    assert TP4_PP2.members_of(Scope.DOMAIN, WORLD, 0) == (0, 1, 2, 3)
    assert TP4_PP2.members_of(Scope.FABRIC, WORLD, 0) == (0, 4)
    single = ParallelLayout(4)  # one stage: the pipeline group is the rank alone
    assert single.pipeline_members(tuple(range(4)), 2) == (2,)


def test_the_rendezvous_plan_meets_at_the_first_member_s_node_on_a_port_per_set_and_scope():
    sets = [WORLD, NODE_B_LOST]
    hosts = {r: ("10.0.0.1" if r < 4 else "10.0.0.2") for r in WORLD}
    hosts.update({replacement(r): "10.0.0.3" for r in (4, 5, 6, 7)})
    me = 1
    domain = rendezvous_plan(
        sets, me, lambda s: TP4_PP2.tensor_members(s, me), hosts, 30000, slot=0, slots=2
    )
    fabric = rendezvous_plan(
        sets, me, lambda s: TP4_PP2.pipeline_members(s, me), hosts, 30000, slot=1, slots=2
    )
    assert [(r.group, r.host, r.port) for r in domain] == [
        ((0, 1, 2, 3), "10.0.0.1", 30000),
        ((0, 1, 2, 3), "10.0.0.1", 30002),
    ]
    assert [(r.group, r.host, r.port) for r in fabric] == [
        ((1, 5), "10.0.0.1", 30001),
        ((1, replacement(5)), "10.0.0.1", 30003),
    ]
    # the spare for rank 5 forms only the column it is in; that column's store is on node-a
    r5 = replacement(5)
    spare = rendezvous_plan(sets, r5, lambda s: TP4_PP2.pipeline_members(s, r5), hosts, 30000, 1, 2)
    assert [(r.members, r.group, r.host, r.port) for r in spare] == [
        (NODE_B_LOST, (1, r5), "10.0.0.1", 30003)
    ]
    # a group of one process is not formed
    one_stage = [tuple(range(4))]
    alone = rendezvous_plan(
        one_stage, 1, lambda s: ParallelLayout(4).pipeline_members(s, 1), hosts, 1
    )
    assert alone == []
    with pytest.raises(KeyError):
        rendezvous_plan(sets, 1, lambda s: TP4_PP2.pipeline_members(s, 1), {}, 30000)


def test_the_fabric_family_moves_with_the_collective_and_gates_the_pointer_swap():
    live_tp = FakeGroup(group="tp-live", ranks=[0, 1, 2, 3])
    live_pp = FakeGroup(group="pp-live", ranks=[1, 5])
    rep_tp = FakeGroup(group="tp-rep", ranks=[0, 1, 2, 3])
    rep_pp = FakeGroup(group="pp-rep", ranks=[1, 5])
    fabric = GroupFamily(Scope.FABRIC, live_pp, {NODE_B_LOST: rep_pp})
    c = Collective(1, live_tp, {NODE_B_LOST: rep_tp}, fabric=fabric)
    assert c.has_shadow(NODE_B_LOST)
    c.change(NODE_B_LOST, epoch=1)
    assert c.current is rep_tp and fabric.current is rep_pp and c.members == NODE_B_LOST
    # a set pre-formed in one scope only is not a pointer swap, and a change to it fails whole
    other = member_sets(8, losses=(), replacements=[(5,)])[0]
    c2 = Collective(1, live_tp, {other: rep_tp}, fabric=GroupFamily(Scope.FABRIC, live_pp))
    assert not c2.has_shadow(other)
    with pytest.raises(KeyError):
        c2.change_to_group(other, rep_tp, epoch=1)
    assert c2.current is live_tp and c2.fabric.current is live_pp
    # a spare alone in its column starts the family on a stand-in
    spare = GroupFamily(Scope.FABRIC, LocalStandIn())
    assert isinstance(spare.current, LocalStandIn) and spare.shadow_sets == []


def test_the_rendezvous_base_port_follows_from_the_engine_s_name():
    from everserve.comm.shadow import RENDEZVOUS_PORTS, rendezvous_base_port

    lo, span = RENDEZVOUS_PORTS
    for name in ("k3-a", "rig", "x" * 50):
        p = rendezvous_base_port(name)
        assert lo <= p < lo + span and p == rendezvous_base_port(name)
    assert rendezvous_base_port("k3-a") != rendezvous_base_port("k3-b")
