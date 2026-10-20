# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""KVCopy against every paged layout: save a unit's rows on one rank, restore them into another
rank's cache at a different head slot, bit for bit."""

import pytest
import torch

from everserve.reference.kvcopy import KVCopy
from everserve.reference.store import start_server
from everserve.state.kvcopy import CacheLayout, PagedKV

NB, BS, H, D, L = 6, 4, 3, 8, 2


def make(layout):
    if layout == "kv_first":
        return torch.zeros(2, NB, BS, H, D, dtype=torch.bfloat16)
    if layout == "blocks_first":
        return torch.zeros(NB, 2, BS, H, D, dtype=torch.bfloat16)
    return torch.zeros(NB, BS, H, 2 * D, dtype=torch.bfloat16).permute(0, 2, 1, 3)


@pytest.mark.parametrize("layout", ["kv_first", "blocks_first", "combined"])
def test_layout_is_recognised(layout):
    p = PagedKV.of(make(layout))
    assert p.layout is CacheLayout(layout)
    assert (p.num_blocks, p.block_size, p.head_dim, p.num_slots) == (NB, BS, D, NB * BS)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        PagedKV.of(torch.zeros(3, 3, 3))


@pytest.mark.parametrize("layout", ["kv_first", "blocks_first", "combined"])
def test_rows_roundtrip_through_write_rows(layout):
    t = make(layout)
    p = PagedKV.of(t)
    slots = torch.tensor([0, 5, 23])
    x = torch.randn(2, 3, D).to(torch.bfloat16)
    p.write_rows(t, slots, 1, x)
    assert torch.equal(p.rows(t, slots, 1), x)
    assert torch.equal(p.rows(t, slots, 0), torch.zeros_like(x))


@pytest.mark.parametrize("layout", ["kv_first", "blocks_first", "combined"])
def test_rows_of_heads_is_the_per_head_rows_in_one_gather(layout):
    t = make(layout)
    p = PagedKV.of(t)
    t.normal_()
    slots = torch.tensor([0, 5, 9, p.num_slots - 1])
    heads = torch.tensor([1, 0])
    got = p.rows_of_heads(t, slots, heads)  # [h, n, 2, d]
    assert got.shape == (2, 4, 2, p.head_dim)
    for h, head in enumerate(heads.tolist()):
        assert torch.equal(got[h].transpose(0, 1), p.rows(t, slots, head))


@pytest.mark.parametrize("layout", ["kv_first", "blocks_first", "combined"])
def test_save_commit_restore_across_ranks_and_head_slots(layout):
    sp = start_server(("127.0.0.1", 0))
    try:
        a = {f"model.layers.{i}.self_attn.attn": make(layout) for i in range(L)}
        b = {f"model.layers.{i}.self_attn.attn": make(layout) for i in range(L)}
        rank_a = KVCopy(sp.address, lambda: [(0, 7)])  # rank A holds unit 7 in head slot 0
        rank_a.register(a)
        slots = torch.tensor([1, 5, 9, 22])
        written = {}
        for name, t in rank_a.layers:
            x = torch.randn(2, 4, D).to(torch.bfloat16)
            rank_a.paged.write_rows(t, slots, 0, x)
            rank_a.save(name, t, slots)
            written[name] = x
        rank_a.commit()
        assert rank_a.profile.steps == 1

        rank_b = KVCopy(sp.address, lambda: [(1, 3)])  # rank B absorbs unit 7 into head slot 2
        rank_b.register(b)
        rank_b.restore([(7, 2)])
        for name, t in rank_b.layers:
            assert torch.equal(rank_b.paged.rows(t, slots, 2), written[name])
            assert torch.equal(
                rank_b.paged.rows(t, slots, 1), torch.zeros(2, 4, D, dtype=torch.bfloat16)
            )
    finally:
        sp.stop()


def test_padding_slots_are_dropped_at_commit():
    sp = start_server(("127.0.0.1", 0))
    try:
        caches = {"model.layers.0.self_attn.attn": make("kv_first")}
        kc = KVCopy(sp.address, lambda: [(0, 1)])
        kc.register(caches)
        t = kc.layers[0][1]
        kc.paged.write_rows(t, torch.tensor([3]), 0, torch.ones(2, 1, D, dtype=torch.bfloat16))
        kc.save(kc.layers[0][0], t, torch.tensor([3, -1, -1]))
        kc.commit()
        slots, data = sp.client().get(1)
        assert slots.tolist() == [3]
        assert data.shape == (1, 2, 1, D)
    finally:
        sp.stop()


def test_clear_discards_an_uncommitted_step():
    sp = start_server(("127.0.0.1", 0))
    try:
        kc = KVCopy(sp.address, lambda: [(0, 1)])
        kc.register({"model.layers.0.self_attn.attn": make("kv_first")})
        kc.save(kc.layers[0][0], kc.layers[0][1], torch.tensor([2]))
        kc.clear()
        kc.commit()
        assert kc.profile.steps == 0
        assert sp.client().stats()["puts"] == 0
    finally:
        sp.stop()
