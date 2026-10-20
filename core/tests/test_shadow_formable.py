# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A set naming a member no process plays is refused before any rendezvous.

A policy that pre-forms a replacement set per rank while one spare is launched leaves the other
sets waiting out the store timeout at startup, and the engine never comes up.
"""

import pytest

from everserve.comm.shadow import check_formable, member_sets

W = 4


def test_sets_whose_every_member_exists_pass():
    check_formable(member_sets(W, replacements=[(2,)]), W, spare_ranks=[2])
    check_formable(member_sets(W), W, spare_ranks=[])
    check_formable(member_sets(W, losses=[(0, 1)], replacements=[(0, 1)]), W, spare_ranks=[0, 1])


def test_a_set_naming_a_spare_nobody_launched_is_refused():
    with pytest.raises(ValueError, match="no spare is launched for rank 2"):
        check_formable(member_sets(W, replacements=[(2,)]), W, spare_ranks=[])
    with pytest.raises(ValueError, match="rank 0"):
        check_formable(member_sets(W, losses=(), replacements=[(0,), (2,)]), W, spare_ranks=[2])


def test_a_set_naming_a_rank_outside_the_world_is_refused():
    with pytest.raises(ValueError, match="outside the world"):
        check_formable([(0, 1, 2, 4)], W, spare_ranks=[])
