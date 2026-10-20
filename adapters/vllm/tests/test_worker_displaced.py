# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A worker a later commit displaces leaves its position: released where its node ends released
processes, else ended with EXIT_STALE, so its node or framework starts it anew."""

import types

import pytest
import torch

from everserve.comm.collective import LocalStandIn
from everserve.contract.daemon import EXIT_STALE
from everserve_vllm import worker


class Region:
    def __init__(self, ends):
        self.ends, self.marked = ends, []

    def ends_released(self):
        return self.ends

    def mark_released(self, row):
        self.marked.append(row)


class Link:
    closed = False

    def close(self):
        self.closed = True


def runtime():
    return worker.AdapterRuntime(4, 8, torch.device("cpu"), LocalStandIn())


def with_slot(monkeypatch, region):
    link = Link()
    state = types.SimpleNamespace(
        collective=types.SimpleNamespace(slot=types.SimpleNamespace(region=region, row=1)),
        link=link,
    )
    monkeypatch.setattr(worker, "STATE", state)
    return link


def test_where_the_node_ends_released_processes_a_displaced_worker_releases_itself(monkeypatch):
    region = Region(ends=True)
    link = with_slot(monkeypatch, region)
    monkeypatch.setattr(worker.os, "_exit", lambda code: pytest.fail(f"exited {code}"))
    rt = runtime()
    rt.on_displaced(3, "epoch 3 places rank 4 elsewhere")
    assert link.closed and region.marked == [1]  # the link first, then the mark (vacate's order)
    assert any(e.kind == "displaced" and e.fields["epoch"] == 3 for e in rt.trace)


def test_elsewhere_a_displaced_worker_ends_itself_stale(monkeypatch):
    with_slot(monkeypatch, Region(ends=False))
    codes = []
    monkeypatch.setattr(worker.os, "_exit", codes.append)
    runtime().on_displaced(3, "epoch 3 places rank 4 elsewhere")
    assert codes == [EXIT_STALE]
