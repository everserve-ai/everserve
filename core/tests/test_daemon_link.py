# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The worker's daemon link against a fake daemon over a Unix socket: register, change, ack."""

import tempfile
import threading
from pathlib import Path

import torch
from collective_doubles import FakeGroup
from link_doubles import FakeDaemon

from everserve.comm.collective import Collective
from everserve.comm.runtime import Runtime
from everserve.contract.daemon import DaemonClient, DaemonConnection
from everserve.contract.signals import (
    Change,
    Join,
    Member,
    MembershipCommitted,
    Notice,
    RankLost,
    Reason,
    Register,
    Role,
    from_message,
    to_message,
)
from everserve.outcome import Outcome


def test_register_change_ack_and_report(tmp_path):
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    rt = Runtime(0, 4, torch.device("cpu"), Collective(0, FakeGroup(ranks=[0, 1, 2, 3]), {}))
    link_holder = {}

    def connect():
        link_holder["link"] = DaemonConnection(path, "e1", Member("n1", 0), rt)

    t = threading.Thread(target=connect)
    t.start()
    reg = daemon.accept()
    t.join()
    assert reg == Register(engine="e1", rank=0, member=Member("n1", 0))

    daemon.send(Notice(rank=2, deadline_ms=1_700_000_000_000))
    daemon.send(Change(lost=(2,), outcome=Outcome.SHRINK, epoch=1))
    ack = daemon.ack()
    assert (ack.rank, ack.epoch, ack.error) == (0, 1, "")
    assert rt.fault is None and rt.alive == [0, 1, 3]  # idle: recovered on the link thread
    assert rt.collective.members == (0, 1, 3)
    assert [e.kind for e in rt.events] == ["notice", "shrink", "recover"]

    link_holder["link"].report_lost([1], "collective failed")
    assert daemon.signal() == RankLost(ranks=(1,), detail="collective failed")

    # a worker whose pod is going gives its own rank notice on its link (a spare that took one)
    link_holder["link"].notice(0, 1_700_000_005_000)
    assert daemon.signal() == Notice(rank=0, deadline_ms=1_700_000_005_000, reason=Reason.NOTICE)

    # a change the runtime cannot apply is acknowledged with the error, not silence
    daemon.send(Change(lost=(3,), outcome=Outcome.REPLACE, epoch=2))
    ack = daemon.ack()
    assert ack.error.startswith("KeyError")

    link_holder["link"].close()
    assert link_holder["link"].closed.wait(5)


class ReportsWhileApplying(Runtime):
    """A runtime whose change is applied while the slot watch reports a rank (decision 0021)."""

    link = None

    def on_change(self, lost, outcome=Outcome.REPLACE, epoch=None, departing=()):
        self.link.report_lost([2], "slot row 0: no words from rank 2", 0)
        super().on_change(lost, outcome=outcome)


def test_a_report_made_while_a_change_is_applied_is_read_by_its_kind(tmp_path):
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    rt = ReportsWhileApplying(
        0, 4, torch.device("cpu"), Collective(0, FakeGroup(ranks=[0, 1, 2, 3]), {})
    )
    holder = {}

    def connect():
        holder["link"] = DaemonConnection(path, "e1", Member("n1", 0), rt)

    t = threading.Thread(target=connect)
    t.start()
    daemon.accept()
    t.join()
    rt.link = holder["link"]
    daemon.send(Change(lost=(2,), outcome=Outcome.SHRINK, epoch=1))
    # the report goes out before the Ack; the daemon reads each frame by its kind
    assert daemon.signal() == RankLost(ranks=(2,), detail="slot row 0: no words from rank 2")
    ack = daemon.ack()
    assert (ack.rank, ack.epoch, ack.error) == (0, 1, "")
    holder["link"].close()


def test_a_spare_registers_as_a_spare_and_a_client_reports_and_hears_commits():
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    rt = Runtime(2, 4, torch.device("cpu"), Collective(2, FakeGroup(ranks=[0, 1, 2, 3]), {}))
    holder = {}
    t = threading.Thread(
        target=lambda: holder.setdefault(
            "spare", DaemonConnection(path, "e1", Member("n1", 4), rt, role=Role.SPARE)
        )
    )
    t.start()
    assert daemon.accept() == Register("e1", 2, Member("n1", 4), Role.SPARE)
    t.join()
    holder["spare"].close()

    daemon.sock.listen(1)
    t = threading.Thread(
        target=lambda: holder.setdefault("client", DaemonClient(path, "e1", Member("n1", 0)))
    )
    t.start()
    assert daemon.accept().role is Role.CLIENT
    t.join()
    client = holder["client"]
    client.report_lost([2], "sentinel")
    assert daemon.signal() == RankLost(ranks=(2,), detail="sentinel")
    client.notice(2, 1_700_000_000_000)
    assert daemon.signal() == Notice(rank=2, deadline_ms=1_700_000_000_000, reason=Reason.NOTICE)
    committed = MembershipCommitted(epoch=1, active_ranks=(0, 1, 2, 3), world_size=4)
    daemon.send(committed)
    heard = client.wait_committed(5)
    assert heard.committed == committed and heard.at > 0
    client.close()
    assert client.closed.wait(5)


def test_shadow_groups_can_be_added_after_start():
    c = Collective(0, FakeGroup(ranks=[0, 1, 2, 3]), {})
    assert not c.has_shadow((0, 1, "R", 3))
    c.preform({(0, 1, "R", 3): FakeGroup(ranks=[0, 1, 2, 3])})
    assert c.has_shadow((0, 1, "R", 3))
    import pytest

    with pytest.raises(ValueError):
        c.preform({(1, 2, 3): FakeGroup(ranks=[1, 2, 3])})


def test_a_client_hears_a_claimed_change_before_its_commit(tmp_path):
    """Decision 0023: the daemon forwards a Change to the engine's clients; the client keeps it."""
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    holder = {}

    def connect():
        holder["client"] = DaemonClient(path, "e1", Member("n1", 0))

    t = threading.Thread(target=connect)
    t.start()
    reg = daemon.accept()  # the registration is the first frame; accept reads it
    t.join()
    client = holder["client"]
    assert (reg.engine, reg.rank, reg.role) == ("e1", 0, Role.CLIENT)
    daemon.send(Change(lost=(2, 3), outcome=Outcome.REPLACE, epoch=1))
    heard = client.changes.get(timeout=5)
    assert heard.change == Change(lost=(2, 3), outcome=Outcome.REPLACE, epoch=1)
    client.close()


def test_a_join_goes_from_the_client_to_the_daemon_and_an_answer_back(tmp_path):
    """Decision 0024: the client sends a join for a rank; the daemon's forwarded answer lands in
    the client's queue; a worker's link queues a join for its position and answers with its own."""
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    holder = {}

    def connect():
        holder["client"] = DaemonClient(path, "e1", Member("n1", 0))

    t = threading.Thread(target=connect)
    t.start()
    daemon.accept()
    t.join()
    client = holder["client"]
    client.join(rank=4, epoch=1, payload=b"ring handle")
    assert daemon.signal() == Join(engine="e1", epoch=1, rank=4, payload=b"ring handle")
    daemon.send(Join(engine="e1", epoch=1, rank=4, payload=b"response handle"))
    answer = Join(engine="e1", epoch=1, rank=4, payload=b"response handle")
    assert client.joins.get(timeout=5) == answer
    client.close()

    # a worker's link: the join for its position is queued, and its answer carries its rank
    path = str(Path(tempfile.mkdtemp()) / "w.sock")
    daemon = FakeDaemon(path)
    rt = Runtime(2, 4, torch.device("cpu"), Collective(2, FakeGroup(ranks=[0, 1, 2, 3]), {}))
    link_holder = {}

    def connect_worker():
        link_holder["link"] = DaemonConnection(path, "e1", Member("n1", 2), rt)

    t = threading.Thread(target=connect_worker)
    t.start()
    daemon.accept()
    t.join()
    link = link_holder["link"]
    daemon.send(Join(engine="e1", epoch=1, rank=2, payload=b"ring handle"))
    assert link.joins.get(timeout=5) == Join(engine="e1", epoch=1, rank=2, payload=b"ring handle")
    link.join(epoch=1, payload=b"response handle")
    assert daemon.signal() == Join(engine="e1", epoch=1, rank=2, payload=b"response handle")
    link.close()


def test_a_registration_carries_this_builds_protocol():
    """Decision 0027: the daemon refuses another protocol by its number, so ours is in every
    registration, from the one definition in signals.proto."""
    from everserve.contract.signals import (
        PROTOCOL_VERSION,
        Member,
        Register,
    )
    from everserve.proto import signals_pb2

    assert PROTOCOL_VERSION == signals_pb2.PROTOCOL_VERSION
    assert PROTOCOL_VERSION >= 1
    msg = to_message(Register(engine="e", rank=1, member=Member("n", 0)))
    assert msg.register.protocol == PROTOCOL_VERSION
    assert from_message(msg).protocol == PROTOCOL_VERSION


def test_a_register_carries_the_compile_cache_the_process_loaded():
    """Decision 0029: the pair goes in the Register; None from a process that compiled nothing."""
    from everserve.contract.signals import CompileCache

    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    rt = Runtime(0, 4, torch.device("cpu"), Collective(0, FakeGroup(ranks=[0, 1, 2, 3]), {}))
    holder = {}
    pair = CompileCache(key="760834052d", digest="ab" * 32)

    def connect():
        holder["link"] = DaemonConnection(path, "e1", Member("n1", 0), rt, compile_cache=pair)

    t = threading.Thread(target=connect)
    t.start()
    reg = daemon.accept()
    t.join()
    assert reg.compile_cache == pair
    assert reg == Register(engine="e1", rank=0, member=Member("n1", 0), compile_cache=pair)


class Counts(Runtime):
    """A runtime that records the epoch each change moved it to."""

    def __init__(self, collective):
        super().__init__(2, 4, torch.device("cpu"), collective)


def test_a_change_moves_to_its_own_epoch_not_the_process_s_count_plus_one(tmp_path):
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    # a process that was in none of the changes before this one: its own count is 0
    rt = Counts(Collective(2, FakeGroup(ranks=[0, 1, 2, 3]), {}))
    holder = {}
    t = threading.Thread(
        target=lambda: holder.update(link=DaemonConnection(path, "e1", Member("n1", 0), rt))
    )
    t.start()
    daemon.accept()
    t.join()
    daemon.send(Change(lost=(3,), outcome=Outcome.SHRINK, epoch=2))
    ack = daemon.ack()
    assert (ack.epoch, ack.error) == (2, "") and rt.epoch == 2
    holder["link"].close()


def _linked(rt, **kw):
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    holder = {}
    t = threading.Thread(
        target=lambda: holder.update(link=DaemonConnection(path, "e1", Member("n1", 0), rt, **kw))
    )
    t.start()
    daemon.accept()
    t.join()
    return daemon, holder["link"]


def test_a_change_of_ranks_leaving_on_notice_refuses_nothing_and_a_fault_refuses():
    rt = Counts(Collective(2, FakeGroup(ranks=[0, 1, 2, 3]), {}))
    refused = []
    rt.collective.refuse = lambda: refused.append(rt.epoch)
    daemon, link = _linked(rt)
    # rank 3 leaves on notice: alive, it finishes the step with the rest, so nothing is refused
    daemon.send(Change(lost=(3,), outcome=Outcome.SHRINK, epoch=1, departing=(3,)))
    assert daemon.ack().epoch == 1 and refused == []
    daemon.send(Change(lost=(1,), outcome=Outcome.SHRINK, epoch=2))  # a fault
    assert daemon.ack().epoch == 2 and refused == [2]
    link.close()


def test_a_process_leaving_on_notice_hears_its_change_without_applying_or_acknowledging_it():
    rt = Counts(Collective(2, FakeGroup(ranks=[0, 1, 2, 3]), {}))  # this process is rank 2
    daemon, link = _linked(rt)
    daemon.send(Change(lost=(2,), outcome=Outcome.REPLACE, epoch=1, departing=(2,)))
    daemon.send(Change(lost=(3,), outcome=Outcome.SHRINK, epoch=2))
    ack = daemon.ack()  # the first acknowledgement is the second change's
    assert (ack.epoch, ack.error) == (2, "") and rt.epoch == 2
    link.close()


def test_a_spare_standing_for_a_departing_rank_applies_and_acknowledges_its_change():
    rt = Counts(Collective(2, FakeGroup(ranks=[0, 1, 2, 3]), {}))  # a spare preloaded for rank 2
    applied = []
    rt.on_change = lambda lost, **kw: applied.append((tuple(lost), kw["departing"]))
    daemon, link = _linked(rt, role=Role.SPARE)
    # the rank it stands for leaves on notice: the spare is the one arriving, not leaving
    daemon.send(Change(lost=(2,), outcome=Outcome.REPLACE, epoch=1, departing=(2,)))
    ack = daemon.ack()
    assert ack.error == "" and applied == [((2,), (2,))]
    link.close()
