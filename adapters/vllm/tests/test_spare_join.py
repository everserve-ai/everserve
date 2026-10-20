# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A standing spare's join, as its process answers the offers its link hears (``spare_main``).

Every offer comes from a core that made its start-up calls, and names their digest (decision
0044): a spare takes an offer naming the calls it was built from, and refuses by name one that
names other calls or none, taking nothing. The offers are encoded as the core encodes them.
"""

import queue
from types import SimpleNamespace

import pytest

from everserve_vllm import join as joins
from everserve_vllm import replay, spare, spare_main


class Link:
    """A spare's daemon link as the join reads it: offers in, answers out."""

    def __init__(self, *payloads):
        self.joins = queue.Queue()
        for p in payloads:
            self.joins.put(SimpleNamespace(epoch=2, payload=p))
        self.answers = []

    def join(self, epoch, payload):
        self.answers.append(joins.decode(payload))


BUILT = replay.Calls((("initialize_from_config", ([SimpleNamespace(kv_cache_tensors=[])],), {}),))
OTHER = replay.Calls(
    (("initialize_from_config", ([SimpleNamespace(kv_cache_tensors=["t0"])],), {}),)
)


def offer(startup=""):
    return joins.encode(joins.Offer(0, 2, "change-2", None, startup=startup))


@pytest.fixture
def taking(monkeypatch):
    took = []

    def take_position(proc, rank, ring, connect_ip, catch_up):
        took.append((rank, connect_ip, catch_up))
        return {"handle": None, "applied": {}, "restore": None, "join_s": 0.25}

    monkeypatch.setattr(spare, "take_position", take_position)
    monkeypatch.setattr(spare_main, "TAKEN", [])
    return took


def test_a_spare_takes_an_offer_naming_the_calls_it_was_built_from(taking):
    link = Link(offer(replay.digest(BUILT)))
    got = spare_main.join_position(link, 0, "proc", BUILT, "10.0.0.3", None, {})
    assert got["join_s"] == 0.25 and taking == [(0, "10.0.0.3", True)]
    [answer] = link.answers
    assert (answer.status, answer.rank, answer.tag) == ("READY", 0, "change-2")


@pytest.mark.parametrize(
    ("startup", "named"), [("", "none"), (replay.digest(OTHER), replay.digest(OTHER)[:12])]
)
def test_a_spare_refuses_an_offer_naming_other_calls_or_none_and_takes_nothing(
    taking, startup, named
):
    """A missing digest is a defect, never a reason to skip the check."""
    link = Link(offer(startup))
    with pytest.raises(replay.ReplayRefused, match=f"the core's {named}"):
        spare_main.join_position(link, 0, "proc", BUILT, "10.0.0.3", None, {})
    [answer] = link.answers
    assert answer.status == "FAILED" and f"the core's {named}" in answer.error
    assert taking == []


def test_an_offer_this_build_cannot_read_is_answered_and_the_next_one_taken(taking):
    link = Link(b"not a join", offer(replay.digest(BUILT)))
    spare_main.join_position(link, 0, "proc", BUILT, "10.0.0.3", None, {})
    assert [a.status for a in link.answers] == ["FAILED", "READY"] and len(taking) == 1
