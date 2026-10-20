# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Decision 0032 over the daemon link, against a fake daemon: the head registers as the head with
the launcher's pid and says why it stops before it closes; an abandoned change reaches a worker's
runtime, which records it and changes nothing."""

import tempfile
import threading
from pathlib import Path

import torch
from test_daemon_link import FakeDaemon, FakeGroup

from everserve.comm.collective import Collective
from everserve.comm.runtime import Runtime
from everserve.contract.daemon import DaemonClient, DaemonConnection
from everserve.contract.signals import Abandoned, Goodbye, Member, Register, Role, StopCause


def connected(make):
    path = str(Path(tempfile.mkdtemp()) / "d.sock")
    daemon = FakeDaemon(path)
    holder = {}
    t = threading.Thread(target=lambda: holder.setdefault("link", make(path)))
    t.start()
    first = daemon.accept()
    t.join()
    return daemon, holder["link"], first


def test_the_head_registers_as_the_head_and_says_why_it_stops():
    daemon, client, reg = connected(
        lambda p: DaemonClient(p, "e1", Member("n1", 0), head=True, launcher_pid=4242)
    )
    assert reg == Register(
        engine="e1", rank=0, member=Member("n1", 0), role=Role.HEAD, launcher_pid=4242
    )
    client.goodbye(StopCause.FAILED, "Worker failed with error 'x'")
    assert daemon.signal() == Goodbye(cause=StopCause.FAILED, detail="Worker failed with error 'x'")
    assert client.closed.wait(5)


def test_a_driver_registers_as_a_client_with_no_launcher():
    _, client, reg = connected(lambda p: DaemonClient(p, "e1", Member("n1", 0), launcher_pid=4242))
    assert reg.role is Role.CLIENT and reg.launcher_pid == 0
    client.close()


def test_the_head_hears_an_abandoned_change():
    daemon, client, _ = connected(lambda p: DaemonClient(p, "e1", Member("n1", 0), head=True))
    daemon.send(Abandoned(epoch=3, detail="rank 0 departed before the commit"))
    assert client.abandoned.get(timeout=5) == Abandoned(
        epoch=3, detail="rank 0 departed before the commit"
    )
    client.close()


def test_a_worker_records_an_abandoned_change_and_changes_nothing():
    rt = Runtime(0, 2, torch.device("cpu"), Collective(0, FakeGroup(ranks=[0, 1]), {}))
    daemon, link, _ = connected(lambda p: DaemonConnection(p, "e1", Member("n1", 0), rt))
    daemon.send(Abandoned(epoch=3, detail="rank 0 departed"))
    for _ in range(500):
        if any(e.kind == "abandoned" for e in rt.trace):
            break
        threading.Event().wait(0.01)
    assert [e.kind for e in rt.trace] == ["abandoned"] and rt.events == []
    assert rt.alive == [0, 1] and rt.fault is None
    link.close()
