# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The named-tensor exchange: the receiver learns the shapes from the wire, over send and recv."""

import pytest
import torch

from everserve.comm.exchange import (
    HEADER_WORDS,
    MAX_DIMS,
    MAX_TENSORS,
    Exchange,
    ExchangeError,
    encode_header,
)


class FakeComm:
    """Two members on one wire: what one sends, the other receives, in order."""

    def __init__(self):
        self.wire: list[tuple[int, torch.Tensor]] = []

    def send(self, tensor, dst):
        self.wire.append((dst, tensor.clone()))

    def recv(self, tensor, src):
        dst, sent = self.wire.pop(0)
        assert sent.shape == tensor.shape and sent.dtype == tensor.dtype
        tensor.copy_(sent)


class Comm:
    """A communicator the gate is exercised on; nothing crosses it."""


def test_named_tensors_cross_with_their_keys_shapes_and_dtypes():
    ex = Exchange()
    comm = FakeComm()
    sent = {
        "hidden_states": torch.randn(5, 16, dtype=torch.bfloat16),
        "residual": torch.randn(5, 16),
        "mask": torch.tensor([True, False]),
        "empty": torch.empty(0, 3, dtype=torch.int32),
    }
    ex.send_named(comm, 1, sent, "cpu")
    got = ex.recv_named(comm, 0, "cpu")
    assert list(got) == list(sent)
    for k in sent:
        assert got[k].dtype == sent[k].dtype and got[k].shape == sent[k].shape
        assert torch.equal(got[k], sent[k])
    assert comm.wire == []  # nothing left on the wire: the empty tensor sent no payload


def test_an_empty_dict_crosses_as_a_header_alone():
    ex = Exchange()
    comm = FakeComm()
    ex.send_named(comm, 1, {}, "cpu")
    assert len(comm.wire) == 1 and comm.wire[0][1].numel() == HEADER_WORDS
    assert ex.recv_named(comm, 0, "cpu") == {}


def test_what_the_protocol_cannot_carry_is_refused_by_name():
    with pytest.raises(ExchangeError, match="tensors"):
        encode_header({f"t{i}": torch.zeros(1) for i in range(MAX_TENSORS + 1)})
    with pytest.raises(ExchangeError, match="dimensions"):
        encode_header({"deep": torch.zeros([1] * (MAX_DIMS + 1))})
    with pytest.raises(ExchangeError, match="dtype"):
        encode_header({"c": torch.zeros(1, dtype=torch.complex64)})
    with pytest.raises(ExchangeError, match="not a tensor"):
        encode_header({"n": None})


def test_one_transfer_in_flight_per_communicator():
    """A transfer waits on the communicator's previous event, then leaves its own behind."""
    log = []

    class Event:
        def __init__(self):
            self.n = len(log)

        def record(self):
            log.append(("record", self.n))

        def synchronize(self):
            log.append(("wait", self.n))

    ex = Exchange(events=Event)
    a, b = Comm(), Comm()
    with ex.one_in_flight(a, "cpu"):
        log.append(("body", "a1"))
    with ex.one_in_flight(a, "cpu"):
        log.append(("body", "a2"))
    with ex.one_in_flight(b, "cpu"):
        log.append(("body", "b1"))
    assert log == [
        ("body", "a1"),
        ("record", 1),
        ("wait", 1),
        ("body", "a2"),
        ("record", 4),
        ("body", "b1"),
        ("record", 6),
    ]


def test_without_events_of_its_own_a_cpu_device_runs_the_body_as_it_is():
    ex = Exchange()
    log = []
    with ex.one_in_flight(Comm(), "cpu"):
        log.append("body")
    assert log == ["body"] and ex.stats["first_transfer_s"] is None


def test_the_check_runs_after_the_wait_and_a_refusal_keeps_the_body_off():
    """A group refused while the call waited is read before the enqueue, which never runs."""
    log = []
    refused = []

    class Event:
        def record(self):
            log.append("record")

        def synchronize(self):
            log.append("wait")
            refused.append(True)  # the abort that ended the wait refused the group

    def check():
        log.append("check")
        if refused:
            raise RuntimeError("refused")

    ex = Exchange(events=Event)
    comm = Comm()
    with ex.one_in_flight(comm, "cpu", check):
        log.append("body")
    with pytest.raises(RuntimeError, match="refused"):
        with ex.one_in_flight(comm, "cpu", check):
            log.append("body")
    assert log == ["check", "body", "record", "wait", "check"]


def test_the_check_and_the_enqueue_run_under_the_communicator_s_shared_guard(monkeypatch):
    """The wait is outside the guard; the check and the body are inside it."""
    from everserve.comm import exchange

    log = []

    class Event:
        def record(self):
            log.append("record")

        def synchronize(self):
            log.append("wait")

    class Shared:
        def __enter__(self):
            log.append("shared-in")

        def __exit__(self, *a):
            log.append("shared-out")

    monkeypatch.setattr(exchange, "guard_of", lambda c: type("G", (), {"shared": Shared})())
    ex = Exchange(events=Event)
    comm = Comm()
    for _ in range(2):
        with ex.one_in_flight(comm, "cpu", lambda: log.append("check")):
            log.append("body")
    assert log == [
        "shared-in",
        "check",
        "body",
        "record",
        "shared-out",
        "wait",
        "shared-in",
        "check",
        "body",
        "record",
        "shared-out",
    ]


def test_a_deeper_gate_waits_only_on_the_transfer_in_flight_furthest_back():
    """With two in flight, the third waits on the first; the second is still running."""
    log = []

    class Event:
        def __init__(self):
            self.n = len([x for x in log if x[0] == "record"])

        def record(self):
            log.append(("record", self.n))

        def synchronize(self):
            log.append(("wait", self.n))

    ex = Exchange(in_flight=2, events=Event)
    comm = Comm()
    for _ in range(3):
        with ex.one_in_flight(comm, "cpu"):
            log.append(("body", None))
    assert log == [
        ("body", None),
        ("record", 0),
        ("body", None),
        ("record", 1),
        ("wait", 0),
        ("body", None),
        ("record", 2),
    ]


def test_packed_tensors_cross_as_one_operation_by_a_layout_the_receiver_knows():
    ex = Exchange()
    comm = FakeComm()
    h = torch.arange(12, dtype=torch.float32).view(3, 4)
    r = -h
    ex.send_packed(comm, 1, {"hidden_states": h, "residual": r}, "cpu")
    assert ex.stats["sends"] == 1
    layout = [("hidden_states", torch.float32, (3, 4)), ("residual", torch.float32, (3, 4))]
    out = ex.recv_packed(comm, 0, layout, "cpu")
    assert ex.stats["recvs"] == 1
    assert torch.equal(out["hidden_states"], h) and torch.equal(out["residual"], r)
    into = torch.zeros(64, dtype=torch.float32)
    comm2 = FakeComm()
    ex.send_packed(comm2, 1, {"a": h, "b": r}, "cpu")
    out2 = ex.recv_packed(
        comm2, 0, [("a", torch.float32, (3, 4)), ("b", torch.float32, (3, 4))], "cpu", into=into
    )
    assert torch.equal(out2["b"], r) and out2["b"].data_ptr() >= into.data_ptr()
    with pytest.raises(ExchangeError, match="pack"):
        ex.send_packed(comm, 1, {"a": h, "b": torch.zeros(2)}, "cpu")


def test_a_communicator_s_first_transfer_is_timed_once():
    """From its first enqueue to the first wait on its event, then never again for that
    communicator; a second communicator (a new ring) gets its own reading."""

    class Event:
        def record(self):
            pass

        def synchronize(self):
            pass

    ex = Exchange(events=Event)
    a = Comm()
    with ex.one_in_flight(a, "cpu"):
        pass
    assert ex.stats["first_transfer_s"] is None  # nothing waited on yet
    with ex.one_in_flight(a, "cpu"):
        pass
    first = ex.stats["first_transfer_s"]
    assert first is not None and first >= 0
    with ex.one_in_flight(a, "cpu"):
        pass
    assert ex.stats["first_transfer_s"] == first  # the same communicator, read once
    b = Comm()
    for _ in range(2):
        with ex.one_in_flight(b, "cpu"):
            pass
    assert b in ex._pending and b not in ex._first  # a new ring's own reading, taken


def test_a_wait_on_a_dead_peer_s_transfer_ends_at_the_refusal_not_at_the_abort():
    """With a check, the wait polls the pending event against it: a transfer that never
    completes (its peer gone) ends the step when the group is refused, the device never
    waited on; a transfer that completes ends it on the next poll with the check read after."""
    log = []
    polls = {"n": 0}
    refused = []

    class Event:
        def __init__(self):
            self.done = False

        def record(self):
            log.append("record")

        def query(self):
            polls["n"] += 1
            if polls["n"] == 3:
                refused.append(True)  # the loss is read three polls in; the abort is far off
            return self.done

        def synchronize(self):
            raise AssertionError("a blocking device wait on a dead peer's transfer")

    def check():
        log.append("check")
        if refused:
            raise RuntimeError("refused")

    ex = Exchange(events=Event)
    comm = Comm()
    with ex.one_in_flight(comm, "cpu", check):
        log.append("body")
    with pytest.raises(RuntimeError, match="refused"):
        with ex.one_in_flight(comm, "cpu", check):
            log.append("never")
    assert log == ["check", "body", "record", "check", "check", "check"] and polls["n"] == 3
    # the transfer completes: the wait ends on the next poll and the check runs after it
    refused.clear()
    pending = ex._pending[comm]
    pending[0].done = True
    with ex.one_in_flight(comm, "cpu", check):
        log.append("body2")
    assert log[-3:] == ["check", "body2", "record"] and polls["n"] == 4
