# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Staging plan against a mock paged KV cache.

Two ranks' caches are simulated block by block while decode writes tokens. The doomed rank's cache
is staged round by round into a replacement copy. After the last round the replacement must equal
the doomed rank's cache on every block in use, and each round after the first must copy only the
blocks that changed.
"""

import random

import pytest
import torch

from everserve.reference.staging import StagedRequest, StagingWindow, blocks_to_stage

BS, NB = 16, 64


class MockCache:
    """Scheduler block tables plus a per-block KV tensor written token by token."""

    def __init__(self, n_req, prompt, seed=0):
        self.rng = random.Random(seed)
        free = list(range(NB))
        self.rng.shuffle(free)
        self.free = free
        self.tables = {str(i): [] for i in range(n_req)}
        self.n = {str(i): 0 for i in range(n_req)}
        self.kv = torch.zeros(NB, BS)
        for rid in self.tables:
            self.write(rid, prompt)

    def write(self, rid, k):
        for _ in range(k):
            pos = self.n[rid]
            if pos // BS >= len(self.tables[rid]):
                self.tables[rid].append(self.free.pop())
            b = self.tables[rid][pos // BS]
            self.kv[b, pos % BS] = hash((rid, pos)) % 1000 + 1
            self.n[rid] += 1

    def preempt(self, rid):
        """Free the request's blocks and recompute it into new ones."""
        n = self.n[rid]
        for b in self.tables[rid]:
            self.kv[b] = 0
            self.free.append(b)
        self.tables[rid], self.n[rid] = [], 0
        self.rng.shuffle(self.free)
        self.write(rid, n)

    def finish(self, rid):
        for b in self.tables.pop(rid):
            self.kv[b] = 0
            self.free.append(b)
        del self.n[rid]


def stage(src, dst, staged):
    ids = blocks_to_stage(src.tables, dict(src.n), staged, BS)
    dst[ids] = src.kv[ids]
    return ids


def in_use(c):
    return sorted(b for t in c.tables.values() for b in t)


def test_first_round_copies_every_block_in_use():
    c = MockCache(4, 37)
    dst, staged = torch.zeros(NB, BS), {}
    ids = stage(c, dst, staged)
    assert ids == in_use(c)
    assert torch.equal(dst[ids], c.kv[ids])
    assert all(isinstance(s, StagedRequest) for s in staged.values())


def test_rounds_copy_only_changed_blocks_and_end_equal():
    c = MockCache(8, 50, seed=1)
    dst, staged = torch.zeros(NB, BS), {}
    stage(c, dst, staged)
    for _ in range(40):
        before = dict(c.n)
        for rid in c.n:
            c.write(rid, 1)
        ids = stage(c, dst, staged)
        touched = {c.tables[rid][p // BS] for rid in c.n for p in range(before[rid], c.n[rid])}
        assert set(ids) == touched
        assert len(ids) <= len(c.n)
    u = in_use(c)
    assert torch.equal(dst[u], c.kv[u])


def test_preempted_request_is_restaged_and_finished_request_dropped():
    c = MockCache(4, 30, seed=2)
    dst, staged = torch.zeros(NB, BS), {}
    stage(c, dst, staged)
    for rid in c.n:
        c.write(rid, 5)
    c.preempt("1")
    c.finish("3")
    stage(c, dst, staged)
    assert "3" not in staged
    u = in_use(c)
    assert torch.equal(dst[u], c.kv[u])


def test_multi_token_steps_spanning_block_boundaries():
    c = MockCache(3, 15, seed=3)
    dst, staged = torch.zeros(NB, BS), {}
    stage(c, dst, staged)
    for k in (1, 2, 17, 3, 33):
        for rid in c.n:
            c.write(rid, k)
        stage(c, dst, staged)
        u = in_use(c)
        assert torch.equal(dst[u], c.kv[u])


def test_block_size_must_be_positive():
    with pytest.raises(ValueError):
        blocks_to_stage({}, {}, {}, 0)


def test_window_reports_time_left():
    w = StagingWindow(rank=2, deadline=100.0)
    assert w.seconds_left(40.0) == 60.0
    assert w.seconds_left(160.0) == -60.0
    assert w.rounds == 0 and w.staged == {}
