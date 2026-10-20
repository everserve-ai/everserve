# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The fault tests' collective doubles keep production's failure semantics: an abort releases only
the process that aborts, an aborted slot call returns with its output undefined, an aborted group
call raises, a shrink aborts the old group, a new epoch never reads an older one's words, and every
member sums in the same order."""

import tempfile
import threading
import time
from datetime import timedelta

import pytest
import torch
from collective_doubles import AbortableGroup, Mesh, SlotDouble

from everserve.comm.group import ShrinkableGroup, ShrinkMethod


@pytest.fixture
def meshes():
    d = tempfile.mkdtemp(prefix="mesh-")
    return [Mesh(d, r, connect_timeout_s=1.0) for r in range(3)]


def in_thread(fn):
    """Run ``fn`` on a thread; the result, or the exception it raised, once it ends."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except Exception as e:  # noqa: BLE001 - handed back to the test
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def bf16(v, n=8):
    return torch.full((n,), v, dtype=torch.bfloat16)


def test_slot_members_sum_to_the_same_bits(meshes):
    slots = [SlotDouble(m, m.me, range(3)) for m in meshes]
    outs = [torch.empty(8, dtype=torch.bfloat16) for _ in slots]
    runs = [
        in_thread(lambda s=s, o=o: s.all_reduce_into(bf16(0.1 * (s.row + 1)), o))
        for s, o in zip(slots, outs, strict=True)
    ]
    for t, box in runs:
        t.join(5)
        assert "error" not in box
    assert all(torch.equal(o, outs[0]) for o in outs)
    assert torch.allclose(outs[0].float(), torch.full((8,), 0.6), atol=1e-2)


def test_a_slot_abort_releases_only_its_own_process(meshes):
    # row 2 is dead: rows 0 and 1 wait on it; only row 0 gets its change
    slots = [SlotDouble(m, m.me, range(3)) for m in meshes[:2]]
    outs = [torch.zeros(8, dtype=torch.bfloat16) for _ in slots]
    waits = [
        in_thread(lambda s=s, o=o: s.all_reduce_into(bf16(1.0), o))
        for s, o in zip(slots, outs, strict=True)
    ]
    time.sleep(0.1)
    slots[0].abort()
    waits[0][0].join(5)
    assert not waits[0][0].is_alive(), "row 0 left on its own abort"
    assert "error" not in waits[0][1], "an aborted slot call returns, it does not raise"
    assert torch.isnan(outs[0].float()).all(), "its output is undefined"
    time.sleep(0.1)
    assert waits[1][0].is_alive(), "row 1 got no change and still waits"
    slots[1].abort()
    waits[1][0].join(5)
    assert not waits[1][0].is_alive()


def test_a_new_epoch_never_reads_an_older_ones_words(meshes):
    a, b = (SlotDouble(m, m.me, [0, 1]) for m in meshes[:2])
    out = torch.zeros(8, dtype=torch.bfloat16)
    # row 1 completed epoch 0's first round alone, then both moved to epoch 1
    t, _ = in_thread(lambda: b.all_reduce_into(bf16(5.0), torch.zeros(8, dtype=torch.bfloat16)))
    time.sleep(0.1)
    b.abort()
    t.join(5)
    for s in (a, b):
        s.set_members([0, 1], epoch=1)
        s.resume()
    runs = [
        in_thread(lambda: a.all_reduce_into(bf16(1.0), out)),
        in_thread(lambda: b.all_reduce_into(bf16(2.0), torch.zeros(8, dtype=torch.bfloat16))),
    ]
    for t, box in runs:
        t.join(5)
        assert "error" not in box
    assert torch.equal(out, bf16(3.0)), "row 1's epoch-0 words were not summed"


def test_the_slot_takes_what_the_kernel_takes(meshes):
    s = SlotDouble(meshes[0], 0, range(3))
    assert s.can_take(bf16(1.0, 4096))
    assert not s.can_take(torch.ones(8)), "float32 goes to the group, as in production"
    assert not s.can_take(bf16(1.0, 7)), "an odd count"
    assert not s.can_take(bf16(1.0, 64 * 1024 + 2)), "beyond MAXN"


def group(mesh, name="init"):
    base = ShrinkableGroup(None, [0, 1, 2], "gloo", None, ShrinkMethod.INIT, timedelta(seconds=3))
    return AbortableGroup(base, mesh, name)


def test_a_group_call_blocked_on_a_dead_peer_raises_when_the_group_is_aborted(meshes):
    g = group(meshes[0])
    t, box = in_thread(lambda: g.all_reduce(torch.ones(1)))
    time.sleep(0.1)
    assert t.is_alive(), "it waits on rows 1 and 2"
    g.abort()
    t.join(5)
    assert isinstance(box.get("error"), RuntimeError), box
    with pytest.raises(RuntimeError, match="aborted"):
        g.all_reduce(torch.ones(1))


def test_a_shrink_aborts_the_old_group_first(meshes, monkeypatch):
    g = group(meshes[0])
    t, box = in_thread(lambda: g.all_reduce(torch.ones(1)))
    time.sleep(0.1)
    seen = {}

    def formed(self, lost, me, epoch):
        t.join(5)
        seen["waiter released before the new group forms"] = not t.is_alive()
        return ShrinkableGroup(None, [0, 1], "gloo", None, ShrinkMethod.RECREATE)

    monkeypatch.setattr(ShrinkableGroup, "shrink", formed)
    new = g.shrink([2], 0, epoch=1)
    assert seen == {"waiter released before the new group forms": True}
    assert isinstance(box.get("error"), RuntimeError)
    assert isinstance(new, AbortableGroup) and new.ranks == [0, 1]
