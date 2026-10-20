# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A process a later commit displaces leaves; one the commit still places serves on.

A daemon sends a worker the commit that no longer places it on its node: a process whose node and
device are no longer at its rank has left (protocol 9). A late node's ranks, registered before a
commit they never heard, are told so this way.
"""

import tempfile
import threading
from pathlib import Path

from test_daemon_link import FakeDaemon

from everserve.contract.daemon import DaemonConnection
from everserve.contract.signals import (
    Change,
    Member,
    MembershipCommitted,
    Outcome,
    RankPlacement,
    Role,
)


class Serving:
    """A worker's runtime at ``rank``: it applies changes and records a displacement."""

    def __init__(self, rank=4, epoch=0, took_epoch=0):
        self.orig_rank, self.epoch, self.took_epoch = rank, epoch, took_epoch
        self.displaced: list[tuple[int, str]] = []
        self.left = threading.Event()

    def identity(self):
        return self.orig_rank, self.epoch

    def on_change(self, lost, sent_at=None, outcome=None, epoch=None, departing=()):
        self.epoch = epoch

    def on_notice(self, rank, deadline_ms):
        pass

    def on_abandoned(self, epoch, detail):
        pass

    def on_displaced(self, epoch, why):
        self.displaced.append((epoch, why))
        self.left.set()


def link(runtime, role=Role.WORKER):
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    holder = {}
    t = threading.Thread(
        target=lambda: holder.update(
            link=DaemonConnection(path, "rig", Member("node-b", 0), runtime, role=role)
        )
    )
    t.start()
    daemon.accept()  # its registration
    t.join()
    return daemon, holder["link"]


def commit(epoch, *placed):
    """A commit at ``epoch`` placing each ``(rank, node)``."""
    return MembershipCommitted(
        epoch=epoch,
        active_ranks=tuple(r for r, _ in placed),
        world_size=8,
        members=tuple(RankPlacement(r, n, 0, "h") for r, n in placed),
    )


def settled(daemon, runtime):
    """Whether the link read past every signal before this: a later one is answered first."""
    daemon.send(Change(lost=(7,), outcome=Outcome.REPLACE, epoch=runtime.epoch + 50))
    daemon.ack()  # its acknowledgement: everything sent before it was read
    return not runtime.left.is_set()


def test_a_commit_placing_the_rank_elsewhere_displaces_the_process():
    runtime = Serving()
    daemon, conn = link(runtime)
    daemon.send(commit(1, (4, "node-c"), (0, "node-a")))  # rank 4 moved to node-c
    assert runtime.left.wait(5)
    epoch, why = runtime.displaced[0]
    assert epoch == 1 and "elsewhere" in why
    conn.close()


def test_a_commit_placing_it_at_an_epoch_it_never_staged_displaces_it():
    runtime = Serving()
    daemon, conn = link(runtime)
    daemon.send(commit(1, (4, "node-b")))  # placed here, but no change for epoch 1 reached it
    assert runtime.left.wait(5)
    assert "never reached" in runtime.displaced[0][1]
    conn.close()


def test_a_commit_placing_it_at_an_epoch_it_staged_is_served_on():
    runtime = Serving()
    daemon, conn = link(runtime)
    daemon.send(Change(lost=(7,), outcome=Outcome.REPLACE, epoch=1))
    daemon.ack()  # its acknowledgement: epoch 1 staged
    runtime.epoch = 0  # the commit arrives before the process's own epoch moved
    daemon.send(commit(1, (4, "node-b")))
    assert settled(daemon, runtime) and runtime.displaced == []
    conn.close()


def test_old_news_an_empty_commit_and_a_standing_spare_displace_nobody():
    runtime = Serving(epoch=3)
    daemon, conn = link(runtime)
    daemon.send(commit(2, (4, "node-c")))  # before this process's epoch
    daemon.send(MembershipCommitted(epoch=5, active_ranks=(4,), world_size=8))  # no placements
    assert settled(daemon, runtime) and runtime.displaced == []
    conn.close()

    spare = Serving(took_epoch=None)
    daemon, conn = link(spare, role=Role.SPARE)
    daemon.send(commit(1, (4, "node-c")))  # a commit places no spare still standing
    assert settled(daemon, spare) and spare.displaced == []
    conn.close()


def test_a_leaked_release_is_reported_once_under_the_process_s_rank():
    """The process reports a release left behind past its bound, and decides nothing (0025)."""
    from everserve.contract.signals import ReleaseLeaked

    runtime = Serving(rank=4, epoch=2)
    daemon, conn = link(runtime)
    conn.report_leak("live", 12.3456, 1)
    assert daemon.signal() == ReleaseLeaked("rig", 4, "live", 12345, 1)
    conn.close()
