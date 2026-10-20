# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The recapture slice order is deterministic, serves what decoding asked for first, never repeats
a descriptor and ends with every descriptor captured."""

from everserve.reference.recapture import next_slice


def test_wanted_first_then_capture_order_until_done():
    all_descs = [("FULL", n) for n in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)] + [
        ("PIECEWISE", n) for n in (512, 256, 8)
    ]
    wanted, done, order = [("FULL", 8)], set(), []
    while True:
        pick = next_slice(wanted, all_descs, done, 2)
        if not pick:
            break
        assert pick == next_slice(list(wanted), list(all_descs), set(done), 2)
        assert not (set(pick) & done)
        order += pick
        done.update(pick)
        if len(order) == 4:
            wanted.append(("PIECEWISE", 256))
    assert order[0] == ("FULL", 8)
    assert order.index(("PIECEWISE", 256)) == 4
    assert sorted(order) == sorted(all_descs) and len(order) == len(set(order))


def test_zero_slice_returns_nothing():
    assert next_slice([("FULL", 8)], [("FULL", 8)], set(), 0) == []


def test_wanted_duplicates_are_picked_once():
    assert next_slice([("a",), ("a",)], [("a",), ("b",)], set(), 5) == [("a",), ("b",)]
