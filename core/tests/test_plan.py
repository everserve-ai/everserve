# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
import pytest

from everserve.reference.plan import (
    after_loss,
    capacity_after_one_loss,
    even_units,
    expert_chunks,
)


def test_even_units_splits_in_rank_order():
    assert even_units(8, 4) == {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}


def test_even_units_rejects_uneven_split():
    with pytest.raises(ValueError):
        even_units(6, 4)


def test_expert_chunks_are_fixed_size_and_contiguous():
    chunks = expert_chunks(128, 4, 4)
    assert len(chunks) == 4
    assert all(len(c) == 4 for per_rank in chunks.values() for c in per_rank)
    assert chunks[2][0] == (64, 65, 66, 67)
    assert sum(len(v) for v in chunks.values()) == 32


def test_expert_chunks_rejects_bad_divisions():
    with pytest.raises(ValueError):
        expert_chunks(128, 4, 5)
    with pytest.raises(ValueError):
        expert_chunks(130, 4, 4)


def test_after_loss_spreads_the_dead_ranks_chunks_evenly():
    chunks = expert_chunks(128, 4, 4)
    new, gained = after_loss(chunks, 2)
    assert 2 not in new
    assert sorted(len(v) for v in new.values()) == [10, 11, 11]
    assert sum(len(v) for v in gained.values()) == 8
    assert set(gained) == {0, 1, 3}


def test_after_loss_is_deterministic_and_ties_go_to_the_lowest_rank():
    owned = {0: ["a"], 1: ["b"], 2: ["c", "d"]}
    new, gained = after_loss(owned, 2)
    assert new == {0: ["a", "c"], 1: ["b", "d"]}
    assert gained == {0: ["c"], 1: ["d"]}
    assert after_loss(owned, 2) == (new, gained)


def test_after_loss_honours_a_custom_cost():
    owned = {0: ["a"], 1: ["b"], 2: ["c"]}
    # rank 1 is declared heavily loaded, so the unit must go to rank 0
    new, _ = after_loss(owned, 2, cost=lambda r, o: (10 if r == 1 else 0, len(o[r])))
    assert new[0] == ["a", "c"]


def test_after_loss_rejects_an_unknown_rank():
    with pytest.raises(KeyError):
        after_loss({0: [1], 1: [2]}, 5)


def test_capacity_after_one_loss_is_the_worst_case_over_every_loss():
    assert capacity_after_one_loss(even_units(4, 4)) == 2
    assert capacity_after_one_loss(even_units(8, 4)) == 3
    assert capacity_after_one_loss(even_units(4, 1)) == 4
    assert capacity_after_one_loss({}) == 0


def test_capacity_counts_only_the_requested_kind():
    owned = {
        r: [("a", u) for u in a] + [("m", b) for b in m]
        for r, (a, m) in enumerate(
            zip(even_units(8, 4).values(), even_units(28, 4).values(), strict=True)
        )
    }
    heads = capacity_after_one_loss(owned, count=lambda v: sum(k == "a" for k, _ in v))
    assert heads == 3
