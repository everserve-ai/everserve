# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A worker node lost twice, end to end on the real control plane, in the Dynamo integration's
shape: this tree's coordinator and daemons, and the engine's processes speaking the adapter's and
the contract's own code (``loss_cell``). The head on node 0 holds rank 0; rank 1's node is lost,
first by deletion (its rank given notice and moved), then by a crash of the spare that took it
(killed, no notice); each time a spare that stood, built from the core's replayed start-up calls,
takes rank 1 at the next epoch, checked against their digest (decision 0044). Between the losses
another spare stands, as Grove's pod in place of the deleted one does. Slow: seconds, and it needs
the controller built (cargo build).
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import tempfile
from pathlib import Path

import pytest
from loss_cell import (
    BOUND_S,
    POLICY,
    ControlPlane,
    controller_binary,
    head,
    rank_node,
    spare_node,
    until,
)

pytestmark = pytest.mark.slow


class Cell:
    """The control plane, the engine's processes, and what each put on the test's queue."""

    def __init__(self, binary: str, nodes: list[str]):
        # Unix sockets: a short path (macOS allows 104 bytes)
        self.root = Path(tempfile.mkdtemp(prefix="wl-", dir="/tmp"))
        self.plane = ControlPlane(binary, self.root, nodes, POLICY)
        self.ctx = multiprocessing.get_context("spawn")
        self.events = self.ctx.Queue()
        self.stop = str(self.root / "stop")  # a file: a killed process holds no lock on it
        self.procs: dict[str, multiprocessing.Process] = {}
        self.seen: list[tuple] = []

    def start(self, name: str, target, *args) -> None:
        p = self.ctx.Process(target=target, args=args, name=name)
        p.start()
        self.procs[name] = p

    def of(self, kind: str) -> list[tuple]:
        """Every event of ``kind`` so far; a process that failed fails the test."""
        while True:
            try:
                self.seen.append(self.events.get_nowait())
            except queue.Empty:
                break
        failed = [e for e in self.seen if e[0] == "failed"]
        if failed:
            pytest.fail(f"a process failed: {failed}\n{self.plane.dump()}")
        return [e for e in self.seen if e[0] == kind]

    def until(self, what: str, ok, timeout_s: float = BOUND_S) -> None:
        try:
            until(what, ok, self.plane, timeout_s)
        except pytest.fail.Exception as e:
            self.of("")
            pytest.fail(f"{e.msg}\n== the processes' events\n{self.seen}")

    def close(self) -> None:
        Path(self.stop).touch()
        for p in self.procs.values():
            p.join(15)
            if p.is_alive():
                p.kill()
        self.plane.stop()


def test_rank_1_s_node_lost_by_deletion_then_by_a_crash_is_replaced_by_a_spare_each_time():
    binary = controller_binary()
    if binary is None:
        pytest.skip("everserve-controller binary not built")
    cell = Cell(binary, ["gpu-0", "gpu-1", "spare-0", "spare-1"])
    socks = cell.plane.socks
    try:
        # bring-up: the head with rank 0, then rank 1's node, then the spare, as the pods register
        cell.start("head", head, socks["gpu-0"], cell.events, cell.stop)
        cell.until("the head registered", lambda: cell.of("head"))
        leave, released = cell.ctx.Event(), cell.ctx.Event()
        cell.start(
            "gpu-1", rank_node, socks["gpu-1"], "gpu-1", 1, cell.events, cell.stop, leave, released
        )
        cell.until("rank 1 served", lambda: cell.of("serving"))
        assert [e[2] for e in cell.of("told")] == ["worker", "worker"]
        cell.start("spare-0", spare_node, socks["spare-0"], "spare-0", 1, cell.events, cell.stop)
        cell.until("the spare standing", lambda: cell.of("standing"))
        # built from the core's own start-up calls, asked through its daemon and the coordinator
        assert cell.of("replayed") == [("replayed", 1, "spare-0")]
        assert cell.of("replay") == [("replay", 1, "spare-0", "gpu-0")]
        cell.until(
            "rank 1 covered by spare-0", lambda: cell.plane.covered_by("spare-0", 0), timeout_s=90.0
        )

        # loss 1, a deletion: rank 1 given notice and moved to the spare at epoch 1
        leave.set()
        cell.until("the head heard epoch 1 commit", lambda: cell.of("commit"))
        released.set()  # the head's boundary releases the departing worker (over vLLM's ring)
        cell.until("the spare took rank 1", lambda: cell.of("took"))
        cell.until("rank 1's node left", lambda: cell.of("left"))
        assert cell.of("left") == [("left", 1, True, "gpu-1")]
        assert ("commit", 1, [1], "gpu-0") in cell.of("commit")
        assert ("answer", 1, 1, "READY", "", "gpu-0") in cell.of("answer")
        # the survivor and the spare moved into the new set at the boundary
        assert {(e[1], e[2]) for e in cell.of("moved")} == {(0, 1), (1, 1)}
        assert cell.plane.coordinator_says("membership committed at epoch 1 (Replace, lost [1])")

        # another spare stands, as Grove's pod in place of the deleted one does
        cell.start("spare-1", spare_node, socks["spare-1"], "spare-1", 1, cell.events, cell.stop)
        cell.until("the next spare standing", lambda: len(cell.of("standing")) == 2)
        assert len(cell.of("replayed")) == 2
        cell.until(
            "rank 1 covered by spare-1", lambda: cell.plane.covered_by("spare-1", 1), timeout_s=90.0
        )

        # loss 2, a crash: the process serving rank 1 killed, no notice
        os.kill(cell.procs["spare-0"].pid, signal.SIGKILL)
        cell.until("the next spare took rank 1", lambda: len(cell.of("took")) == 2)
        assert ("commit", 2, [1], "gpu-0") in cell.of("commit")
        assert ("answer", 1, 2, "READY", "", "gpu-0") in cell.of("answer")
        assert cell.plane.coordinator_says("membership committed at epoch 2 (Replace, lost [1])")
        # the engine never went down: the head stood through both
        assert "EngineDown" not in cell.plane.events() and "HeadLost" not in cell.plane.events()
    finally:
        cell.close()
