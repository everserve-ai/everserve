# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A planned group meets on the control plane's links (decision 0035, amended): its server makes
the group's id and reports it; every other member takes it from its own link, bounded."""

import ctypes
import gc
import os
import sys
import types

import pytest

from everserve.comm.cover import PlannedGroup
from everserve.contract.daemon import GroupIds
from everserve.contract.signals import GroupId, GroupReady, SpareRef
from everserve_vllm.groups import group_id_for, mint_group_id

NCCL_ID = bytes(range(128))
GROUP = PlannedGroup("fabric", (0, "R1@node-d:0"), 0)


WIRE = (0, SpareRef(1, "node-d", 0))


class Link:
    """A process's link. ``relay`` stands for the coordinator: once every member is ready, the
    id the server reported comes back to each member, the server included."""

    def __init__(self, relay=False):
        self.reported, self.ready = [], []
        self.relay = relay
        self.group_ids = GroupIds()

    def report(self, r):
        assert isinstance(r, GroupReady)
        self.ready.append(r)

    def report_group_id(self, gid):
        assert self.ready, "the id is made only after this member reported itself ready"
        self.reported.append(gid)
        if self.relay:
            self.group_ids.put(gid)


def test_the_server_reports_itself_ready_makes_the_id_and_joins_only_on_its_return():
    link = Link(relay=True)
    assert group_id_for(GROUP, 0, link, "rig", 4, 1.0, lambda: NCCL_ID) == (0, NCCL_ID)
    assert link.ready == [GroupReady("rig", 4, WIRE, "fabric", 0)]
    assert link.reported == [GroupId("rig", 4, "fabric", WIRE, NCCL_ID)]


def test_a_server_whose_id_never_comes_back_does_not_join():
    """The coordinator holds the id until every member is ready: a member that never is keeps
    the server out of NCCL's blocking initialisation too, and the formation times out."""
    link = Link(relay=False)
    with pytest.raises(TimeoutError):
        group_id_for(GROUP, 0, link, "rig", 4, 0.05, lambda: NCCL_ID)
    assert len(link.reported) == 1  # made and reported, never joined on


def test_a_member_takes_the_id_its_server_made_at_the_generation_or_later():
    link = Link()
    link.group_ids.put(GroupId("rig", 3, "fabric", (0, SpareRef(1, "node-d", 0)), b"x" * 128))
    with pytest.raises(TimeoutError):  # an id of an older generation is not this formation's
        group_id_for(GROUP, "R1@node-d:0", link, "rig", 4, 0.05, lambda: b"never")
    link.group_ids.put(GroupId("rig", 4, "fabric", (0, SpareRef(1, "node-d", 0)), NCCL_ID))
    assert group_id_for(GROUP, "R1@node-d:0", link, "rig", 4, 1.0, lambda: b"never") == (
        1,
        NCCL_ID,
    )
    assert link.reported == []  # only the server makes an id
    assert link.ready == [GroupReady("rig", 4, WIRE, "fabric", SpareRef(1, "node-d", 0))] * 2


def test_a_group_whose_server_is_not_its_rank_0_is_refused():
    bad = PlannedGroup("fabric", (0, "R1@node-d:0"), "R1@node-d:0")
    with pytest.raises(ValueError, match="not its rank 0"):
        group_id_for(bad, 0, Link(), "rig", 4, 1.0, lambda: NCCL_ID)


class FreedId(ctypes.Structure):
    """vLLM's ``ncclUniqueId``, whose memory is overwritten once the struct is freed.

    NCCL puts the id's magic in its first eight bytes. Probe 2 read an id through a temporary
    struct (``string_at(addressof(lib.ncclGetUniqueId()), 128)``): the struct was freed as soon as
    ``addressof`` returned, the allocator's pointer took the magic's place, and every member's
    join was refused for it. This struct lies over memory its library keeps, so reading it after
    it is freed is defined, and it blanks that memory when freed, as the allocator reused it.
    """

    _fields_ = [("internal", ctypes.c_byte * 128)]

    def __del__(self):
        ctypes.memset(ctypes.addressof(self), 0, ctypes.sizeof(self))


class Library:
    """vLLM's ``NCCLLibrary``: each id is new random bytes, as ``ncclGetUniqueId``'s are."""

    def __init__(self):
        self.memory: list[bytearray] = []
        self.written: list[bytes] = []

    def ncclGetUniqueId(self):
        self.written.append(os.urandom(128))
        self.memory.append(bytearray(self.written[-1]))
        return FreedId.from_buffer(self.memory[-1])


@pytest.fixture
def library(monkeypatch):
    lib = Library()
    name = "vllm.distributed.device_communicators.pynccl_wrapper"
    for parent in ("vllm", "vllm.distributed", "vllm.distributed.device_communicators"):
        monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    wrapper = types.ModuleType(name)
    wrapper.NCCLLibrary = lambda: lib
    monkeypatch.setitem(sys.modules, name, wrapper)
    return lib


def test_an_id_is_the_bytes_nccl_wrote_copied_while_its_struct_is_held(library):
    """Each id is its own and exactly what NCCL wrote, through a collection and allocation churn
    between the mint and its use."""
    first, second = mint_group_id(), mint_group_id()
    churn = [bytearray(os.urandom(128)) for _ in range(1000)]
    gc.collect()
    assert [first, second] == library.written and first != second
    del churn


def test_a_read_through_a_freed_temporary_is_caught(library):
    """What probe 2 did, against the same library: the bytes read are not the id NCCL wrote."""
    read = ctypes.string_at(ctypes.addressof(library.ncclGetUniqueId()), 128)
    assert read != library.written[-1] and read[:8] == bytes(8)
