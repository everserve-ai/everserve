# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The collective's side of the cover plan (decision 0035): the groups it holds by scope and set,
and at a commit, the stale groups set aside and each taker renamed to the rank it took."""

import pytest

from everserve.comm.collective import Collective, GroupFamily
from everserve.comm.cover import DOMAIN, FABRIC
from everserve.comm.handle import LocalStandIn
from everserve.comm.layout import Scope
from everserve.comm.shadow import replaced_rank, replacement

D1 = replacement(1, "node-d", 0)  # node-d's spare standing in for rank 1
E1 = replacement(1, "node-e", 0)  # node-e's, a second spare for rank 1
E2 = replacement(2, "node-e", 0)


class Group:
    """A held group, named so a test can tell which one came back."""

    def __init__(self, name):
        self.name = name


def g(name):
    return Group(name)


def collective(me, members, shadows, fabric=None):
    c = Collective(me, LocalStandIn(), shadows, fabric=fabric, stands_for=(me,))
    c.members = tuple(members)
    return c


def test_a_spare_s_id_names_its_process_and_still_its_rank():
    assert D1 == "R1@node-d:0" and replaced_rank(D1) == 1 and replaced_rank("R3") == 3
    assert replacement(1) == "R1"


def test_held_lists_every_group_by_scope_and_the_serving_ones():
    fabric = GroupFamily(Scope.FABRIC, g("fabric live"), {(0, D1, 2): g("fabric d1")})
    c = collective(0, (0, 1, 2), {(0, D1, 2): g("d1"), (0, 1, E2): g("e2")}, fabric)
    c.current = g("live")  # a serving group of several members
    held = c.held()
    assert set(held) == {
        (DOMAIN, (0, 1, 2)),
        (FABRIC, (0, 1, 2)),
        (DOMAIN, (0, D1, 2)),
        (DOMAIN, (0, 1, E2)),
        (FABRIC, (0, D1, 2)),
    }
    assert c.serving() == {(DOMAIN, (0, 1, 2)), (FABRIC, (0, 1, 2))}


def test_keep_and_drop_hold_and_let_go_a_group_in_its_scope():
    fabric = GroupFamily(Scope.FABRIC, LocalStandIn())
    c = collective(0, (0, 1, 2), {}, fabric)
    c.keep(DOMAIN, (0, D1, 2), g("d1"))
    c.keep(FABRIC, (0, D1, 2), g("fd1"))
    assert c.has_shadow((0, D1, 2))
    assert c.drop(FABRIC, (0, D1, 2)).name == "fd1" and not c.has_shadow((0, D1, 2))
    assert c.drop(DOMAIN, (0, D1, 2)).name == "d1" and c.drop(DOMAIN, (0, D1, 2)) is None
    with pytest.raises(ValueError):
        c.keep(DOMAIN, (1, 2), g("not mine"))


def test_a_commit_sets_aside_the_lost_rank_s_old_groups_and_renames_the_taker():
    # rank 1 moved to node-d's spare: the current set is the replacement set
    fabric = GroupFamily(
        Scope.FABRIC,
        g("fabric live"),
        {(0, D1, 2): g("f d1"), (0, 1, E2): g("f e2")},
    )
    c = collective(
        0,
        (0, D1, 2),
        {
            (0, D1, 2): g("d1"),  # the set moved to: kept, renamed
            (0, 1, E2): g("e2"),  # names rank 1's old process: stale
            (0, E1, 2): g("e1"),  # node-e's spare for rank 1: another process, kept as is
        },
        fabric,
    )
    stale = c.rename_at_commit([1])
    assert sorted((scope, s) for scope, s, _ in stale) == [
        (DOMAIN, (0, 1, E2)),
        (FABRIC, (0, 1, E2)),
    ]
    assert c.members == (0, 1, 2)
    assert set(s for s in c.shadow_sets) == {(0, 1, 2), (0, E1, 2)}
    assert fabric.shadow_sets == [(0, 1, 2)]
    assert (DOMAIN, (0, 1, 2)) in c.serving()


def test_the_taker_itself_becomes_the_rank():
    c = collective(D1, (0, D1, 2), {(0, D1, 2): g("d1")})
    c.rename_at_commit([1])
    assert c.me == 1 and c.stands_for == (1,) and c.members == (0, 1, 2)


def test_a_group_formed_under_the_old_id_during_the_commit_is_kept_under_the_new_one():
    c = collective(0, (0, D1, 2), {(0, D1, 2): g("d1")})
    c.rename_at_commit([1])
    c.keep(DOMAIN, (0, D1, E2), g("formed late"))
    assert (0, 1, E2) in c.shadow_sets


def test_a_commit_that_moved_no_spare_renames_nothing():
    c = collective(0, (0, 2), {(0, 2): g("survivors"), (0, 1, E2): g("e2")})
    stale = c.rename_at_commit([1])  # a shrink: rank 1 gone, no taker
    assert [s for _, s, _ in stale] == [(0, 1, E2)] and c.members == (0, 2)


def test_a_spare_held_to_a_plan_names_its_ids_by_its_process():
    c = collective(replacement(1), (0, 1, 2), {})
    c.stands_for = (replacement(1), replacement(2))
    c.name_process("node-d", 0)
    assert c.me == D1 and c.stands_for == (D1, replacement(2, "node-d", 0))
    serving = collective(0, (0, 1, 2), {})
    serving.name_process("node-a", 0)
    assert serving.me == 0  # a rank's id is its rank
    with pytest.raises(ValueError):
        collective(replacement(1), (0, 1, 2), {(0, "R1", 2): g("held")}).name_process("node-d", 0)


def test_a_local_stand_in_is_no_group_held():
    # tensor parallelism of one: the domain's "group" is a stand-in; the pipeline's is real
    fabric = GroupFamily(Scope.FABRIC, g("pipeline"), {(0, D1, 2): LocalStandIn()})
    c = collective(0, (0, 1, 2), {(0, 1, E2): LocalStandIn()}, fabric)
    assert set(c.held()) == {(FABRIC, (0, 1, 2))}


def test_the_replacement_set_is_the_held_set_naming_the_spare_s_process():
    c = collective(0, (0, 1, 2), {(0, D1, 2): g("d1"), (0, 1, E2): g("e2")})
    assert c.replacement_set([1]) == (0, D1, 2)
    assert c.replacement_set([2]) == (0, 1, E2)


def test_a_replacement_set_held_only_in_the_fabric_scope_is_found():
    # tensor parallelism of one: a cover plan forms the pipeline's group only
    fabric = GroupFamily(Scope.FABRIC, g("pipeline"), {(0, D1, 2): g("f d1")})
    c = collective(0, (0, 1, 2), {}, fabric)
    assert c.replacement_set([1]) == (0, D1, 2)


def test_two_spares_for_one_position_need_the_change_to_name_one():
    c = collective(0, (0, 1, 2), {(0, D1, 2): g("d1"), (0, E1, 2): g("e1")})
    with pytest.raises(KeyError, match="name the spare"):
        c.replacement_set([1])


def test_without_a_held_set_the_composed_set_is_named():
    c = collective(0, (0, 1, 2), {})
    assert c.replacement_set([1]) == (0, "R1", 2)
