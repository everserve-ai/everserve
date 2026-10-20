# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The join payload: self-describing bytes both halves read, declared data only, and errors a
build can name."""

import json
import pickle
from dataclasses import dataclass, field

import pytest

from everserve.contract.signals import CompileCache
from everserve_vllm import join
from everserve_vllm.join import TAG, VERSION, Answer, JoinFormatError, Offer, decode, encode


@dataclass
class Handle:
    """vLLM's ``Handle`` at the pinned commit, field for field: the type a decode rebuilds."""

    local_reader_ranks: list[int] = field(default_factory=list)
    buffer_handle: tuple[int, int, int, str] | None = None
    local_subscribe_addr: str | None = None
    local_notify_addr: str | None = None
    remote_subscribe_addr: str | None = None
    remote_addr_ipv6: bool = False


@pytest.fixture(autouse=True)
def vllm_handle(monkeypatch):
    monkeypatch.setattr(join, "handle_type", lambda: Handle)


RING = Handle([0, 1], (1, 1048576, 10, "psm_a1b2"), "ipc://a", "ipc://b", "tcp://10.1.2.3:41000")


def test_an_offer_and_an_answer_round_trip():
    offer = Offer(rank=5, epoch=2, tag="change-2-123.4", ring=RING)
    back = decode(encode(offer))
    assert back == offer and isinstance(back.ring, Handle)
    assert back.ring.buffer_handle == (1, 1048576, 10, "psm_a1b2")  # a tuple, as vLLM has it
    answer = Answer(
        rank=5,
        epoch=2,
        tag="change-2-123.4",
        status="READY",
        response=Handle(remote_subscribe_addr="tcp://10.1.2.4:42000"),
        applied={"moved": {"members": "0-1-2-3-4-R5-6-7", "rows": [0], "epoch": 2}},
        restore={"rows": 2750720, "counts": {"0-abc": 2149}},
        join_s=0.41,
        compile_cache=CompileCache("key", "digest"),
        attachment={"weights_s": 0.1, "take_s": 6.0},
    )
    assert decode(encode(answer)) == answer
    failed = Answer(rank=5, epoch=2, tag="t", status="FAILED", error="restore mismatch")
    assert decode(encode(failed)).error == "restore mismatch"


def test_a_record_crosses_as_data_whatever_it_held():
    applied = {"released": ("fabric 0-1-2", "torch"), "ranks": {2, 1}, "at": object()}
    back = decode(encode(Answer(rank=1, epoch=1, tag="t", status="READY", applied=applied)))
    assert back.applied["released"] == ["fabric 0-1-2", "torch"]
    assert back.applied["ranks"] == [1, 2] and isinstance(back.applied["at"], str)


def test_bytes_from_another_build_or_kind_are_refused_by_name():
    good = encode(Offer(rank=1, epoch=1, tag="t", ring=None))
    with pytest.raises(JoinFormatError, match="not a join payload"):
        decode(b"garbage")
    with pytest.raises(JoinFormatError, match="tagged"):
        decode(good.replace(TAG, b"someone-else", 1))
    with pytest.raises(JoinFormatError, match="version"):
        decode(good.replace(b"\x00%d\x00" % VERSION, b"\x004\x00", 1))  # a pickling build
    with pytest.raises(JoinFormatError, match="kind"):
        decode(good.replace(b"\x00offer\x00", b"\x00hello\x00", 1))


class _Runs:
    """A pickle that runs code when it is loaded: what a pickled body could carry."""

    ran: list = []

    def __reduce__(self):
        return (_Runs.ran.append, ("ran",))


def test_a_pickled_body_is_refused_and_never_run():
    header = b"%s\x00%d\x00offer\x00" % (TAG, VERSION)
    with pytest.raises(JoinFormatError, match="not JSON"):
        decode(header + pickle.dumps(_Runs()))
    assert _Runs.ran == []


def _offer_body(**changes):
    body = json.loads(encode(Offer(rank=1, epoch=1, tag="t", ring=RING)).split(b"\x00", 3)[3])
    body.update(changes)
    return b"%s\x00%d\x00offer\x00" % (TAG, VERSION) + json.dumps(body).encode()


def test_a_body_that_is_not_the_declared_fields_is_refused_by_name():
    with pytest.raises(JoinFormatError, match=r"does not: \['shell'\]"):
        decode(_offer_body(shell="rm -rf /"))
    with pytest.raises(JoinFormatError, match="rank"):
        decode(_offer_body(rank="1"))
    with pytest.raises(JoinFormatError, match="faulted"):
        decode(_offer_body(faulted=1))  # a number is no flag
    with pytest.raises(JoinFormatError, match="ring is not a queue handle"):
        decode(_offer_body(ring={"remote_subscribe_addr": "tcp://x", "extra": 1}))
    with pytest.raises(JoinFormatError, match="lacks fields"):
        body = b"%s\x00%d\x00offer\x00" % (TAG, VERSION) + json.dumps({"rank": 1}).encode()
        decode(body)


def test_an_offer_says_whether_the_rank_died():
    """A spare skips the catch-up restore for a rank that died (a catch-up after a fault copies
    no row and takes 3 to 9 s)."""
    assert decode(encode(Offer(rank=1, epoch=1, tag="t", ring=None, faulted=True))).faulted
    assert not decode(encode(Offer(rank=1, epoch=1, tag="t", ring=None))).faulted


def test_vllm_s_own_queue_handle_round_trips(monkeypatch):
    """The stand-in above is vLLM's Handle field for field; where vLLM is installed, the real one
    crosses too, so a vLLM whose Handle changed fails here, not at a join."""
    pytest.importorskip("vllm")
    monkeypatch.undo()
    real = join.handle_type()
    ring = real([0], (1, 4096, 2, "psm_x"), "ipc://a", "ipc://b", "tcp://10.0.0.1:1", False)
    back = decode(encode(Offer(rank=1, epoch=1, tag="t", ring=ring))).ring
    assert type(back) is real and back == ring
