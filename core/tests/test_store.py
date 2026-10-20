# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from everserve.reference.store import KVStoreClient, start_server


@pytest.fixture
def store():
    sp = start_server(("127.0.0.1", 0))
    try:
        yield sp
    finally:
        sp.stop()


def test_put_get_roundtrip_is_idempotent(store):
    c = store.client()
    c.init([5], layers=3, slots=100, head_dim=4)
    x = torch.randn(3, 2, 2, 4).to(torch.bfloat16)
    c.put({5: (torch.tensor([7, 9]), x)})
    c.put({5: (torch.tensor([9]), x[:, :, 1:])})
    s, y = c.get(5)
    assert s.tolist() == [7, 9]
    assert torch.equal(y, x)
    assert c.stats()["puts"] == 2
    c.close()


def test_init_is_idempotent_and_units_are_independent(store):
    a, b = store.client(), KVStoreClient(store.address)
    a.init([1, 2], layers=1, slots=8, head_dim=2)
    b.init([2], layers=1, slots=8, head_dim=2)
    a.put({1: (torch.tensor([3]), torch.ones(1, 2, 1, 2, dtype=torch.bfloat16))})
    s1, _ = b.get(1)
    s2, _ = b.get(2)
    assert s1.tolist() == [3]
    assert s2.numel() == 0
    assert b.stats()["units"] == [1, 2]


def test_get_of_an_unknown_unit_is_an_error_not_a_hang(store):
    c = store.client()
    with pytest.raises(RuntimeError, match="KeyError"):
        c.get("never-initialised")
    assert c.stats()["puts"] == 0  # the connection is still usable


def test_server_start_failure_is_reported():
    with pytest.raises(RuntimeError):
        start_server(("127.0.0.1", 1), ready_timeout_s=0.5)  # privileged port: cannot bind
