# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare's start-up replay on the links, against a fake daemon (decision 0044).

A spare asks on a connection of its own whose first frame is the ask, and reads the core's offer
back on it; the core hears asks on its head connection and answers each asker alone.
"""

import threading

import pytest
from link_doubles import FakeDaemon

from everserve.contract.daemon import DaemonClient, ask_replay
from everserve.contract.signals import (
    EnginePolicy,
    Member,
    ProcessRef,
    ReplayAsk,
    ReplayOffer,
    from_message,
)
from everserve.proto import signals_pb2 as pb
from everserve.wire import read_frame

ASKER = ProcessRef("gpu-2", 7, 31)


def offer(rank):
    return ReplayOffer("k", rank, f"queue for {rank}", ASKER)


def asked(daemon, rank, timeout_s=5.0):
    """The spare's ask on its own connection, run beside the daemon; what it got or raised."""
    got = {}

    def run():
        try:
            got["offer"] = ask_replay(daemon.path, "k", rank, timeout_s, connect_timeout_s=5)
        except Exception as e:  # noqa: BLE001 - the test reads what it raised
            got["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, got


def test_a_spare_asks_on_a_connection_whose_first_frame_is_the_ask_and_reads_the_offer_on_it():
    daemon = FakeDaemon()
    t, got = asked(daemon, 3)
    first = daemon.accept()
    # registered as nothing: the ask is the first frame, its asker left for the daemon to name
    assert first == ReplayAsk("k", 3)
    daemon.send(EnginePolicy("k", b""))  # anything else on it is passed over
    daemon.send(offer(1))  # an offer for another rank too
    daemon.send(offer(3))
    t.join(5)
    assert got == {"offer": offer(3)}
    assert daemon.conn.recv(1) == b""  # and the connection is closed once the offer is read


def test_an_ask_no_core_answers_is_bounded_and_one_the_daemon_ends_is_named():
    daemon = FakeDaemon()
    t, got = asked(daemon, 0, timeout_s=0.3)
    daemon.accept()
    t.join(5)
    assert isinstance(got["error"], TimeoutError) and "rank 0" in str(got["error"])

    t, got = asked(daemon, 0)
    daemon.accept()
    daemon.conn.close()
    t.join(5)
    assert isinstance(got["error"], ConnectionError)


def test_the_core_hears_asks_on_its_head_connection_and_answers_the_asker_alone():
    daemon = FakeDaemon()
    reg = {}
    t = threading.Thread(target=lambda: reg.setdefault("reg", daemon.accept()))
    t.start()
    client = DaemonClient(daemon.path, "k", Member("gpu-0", 0), connect_timeout_s=5, head=True)
    t.join(5)
    daemon.send(ReplayAsk("k", 1, ASKER))
    ask = client.replay_asks.get(timeout=5)
    assert ask == ReplayAsk("k", 1, ASKER)
    client.offer_replay(ask, "the queue")
    sent = from_message(pb.Signal.FromString(read_frame(daemon.conn)))
    assert sent == ReplayOffer("k", 1, "the queue", ASKER)
    with pytest.raises(ValueError, match="names no asker"):
        client.offer_replay(ReplayAsk("k", 1), "the queue")
    client.close()
