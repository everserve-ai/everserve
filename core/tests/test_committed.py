# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The loopback committed-KV transport keeps per-request committed counts true through commit,
eviction and finish."""

import pytest
import torch

from everserve.reference.committed import LoopbackCommittedKV
from everserve.reference.kvcopy import KVCopy
from everserve.reference.store import start_server
from everserve.state.committed import CommittedKV

NB, BS, H, D = 6, 4, 2, 8


@pytest.fixture
def ckv():
    sp = start_server(("127.0.0.1", 0))
    try:
        copy = KVCopy(sp.address, lambda: [(0, 7)])
        copy.register(
            {"model.layers.0.self_attn.attn": torch.zeros(2, NB, BS, H, D, dtype=torch.bfloat16)}
        )
        yield LoopbackCommittedKV(copy), sp
    finally:
        sp.stop()


def test_loopback_satisfies_the_protocol(ckv):
    kv, _ = ckv
    assert isinstance(kv, CommittedKV)


def test_commit_records_counts_epoch_and_step(ckv):
    kv, sp = ckv
    t = kv._copy.layers[0][1]
    kv.save(kv._copy.layers[0][0], t, torch.tensor([1, 2]))
    kv.commit(epoch=3, step=12, committed_tokens={"a": 40, "b": 7})
    assert (kv.committed("a"), kv.committed("b"), kv.committed("zzz")) == (40, 7, 0)
    assert (kv.last_epoch, kv.last_step) == (3, 12)
    assert sp.client().stats()["puts"] == 1


def test_evicted_lowers_the_count_and_never_raises_it(ckv):
    kv, _ = ckv
    kv.commit(epoch=1, step=1, committed_tokens={"a": 40})
    kv.evicted("a", kept_tokens=16)
    assert kv.committed("a") == 16
    kv.evicted("a", kept_tokens=100)
    assert kv.committed("a") == 16
    kv.evicted("unknown", kept_tokens=0)  # a request we never saw is not an error
    assert kv.committed("unknown") == 0
    with pytest.raises(ValueError):
        kv.evicted("a", kept_tokens=-1)


def test_finished_forgets_the_request(ckv):
    kv, _ = ckv
    kv.commit(epoch=1, step=1, committed_tokens={"a": 40, "b": 3})
    kv.finished("a")
    kv.finished("never-seen")
    assert kv.committed("a") == 0 and kv.committed("b") == 3


def test_clear_drops_only_uncommitted_rows(ckv):
    kv, sp = ckv
    name, t = kv._copy.layers[0]
    kv.save(name, t, torch.tensor([3]))
    kv.clear()
    kv.commit(epoch=1, step=1, committed_tokens={"a": 1})
    assert sp.client().stats()["puts"] == 0  # nothing pending survived clear
    assert kv.committed("a") == 1  # but the count bookkeeping did
