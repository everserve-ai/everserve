# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A head given its policy on its link asks for node 0's ranks before anything loads (decision
0041), and an engine its processes place bootstraps from the control plane's placement. Runs where
vLLM is installed."""

import json
import socket
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.protobuf import json_format

from everserve.comm.layout import ParallelLayout
from everserve.contract.signals import (
    EnginePlacement,
    EnginePolicy,
    RankPlacement,
    Role,
    RoleAssigned,
    from_message,
    to_message,
)
from everserve.proto import policy_pb2
from everserve.proto import signals_pb2 as pb
from everserve.wire import read_frame, write_frame

pytest.importorskip("vllm")

from everserve.contract.signals import SoftwareIdentity, StepConfig  # noqa: E402
from everserve_vllm import executor, settings  # noqa: E402
from everserve_vllm.executor import EverserveExecutor  # noqa: E402

IDENTITY = SoftwareIdentity("vllm", "v0.31.0", "c0ffee", step=StepConfig("d1", {"a": "1"}))

POLICY = {
    "version": "v1",
    "outcomesInOrder": ["OUTCOME_REPLACE"],
    "spares": {"loading": "LOADING_PRELOADED"},
    "cover": {"enabled": True},
    "engines": [{"name": "k", "tensorParallel": 4, "pipelineParallel": 2}],
}


class FakeDaemon:
    """Answers one undecided registration: the policy, the answer, then the placement."""

    def __init__(self, role):
        self.path = str(Path(tempfile.mkdtemp()) / "node.sock")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        self.sock.listen(1)
        self.asked = None
        self.conn = None
        threading.Thread(target=self._serve, args=(role,), daemon=True).start()

    def send(self, sig):
        write_frame(self.conn, to_message(sig).SerializeToString())

    def _serve(self, role):
        self.conn, _ = self.sock.accept()
        self.asked = from_message(pb.Signal.FromString(read_frame(self.conn)))
        doc = json_format.Parse(json.dumps(POLICY), policy_pb2.Policy()).SerializeToString()
        self.send(EnginePolicy("k", doc))
        self.send(RoleAssigned("k", 0, 0, role, self.asked.stage))


@pytest.fixture
def head(monkeypatch):
    def at(role):
        daemon = FakeDaemon(role)
        monkeypatch.setenv("EVERSERVE_DAEMON_SOCKET", daemon.path)
        monkeypatch.setenv("EVERSERVE_HOST_IP", "10.0.0.1")
        settings.update(engine="k", node="gpu-0", policy="", discovery_dir="")
        ex = EverserveExecutor.__new__(EverserveExecutor)
        ex.layout = ParallelLayout(4, 2)
        ex.parallel_config = SimpleNamespace(nnodes=2)
        ex.asking, ex.given = None, None
        # what it steps with, which its ask carries (resolved by vLLM in a real head)
        ex.vllm_config = "config"
        monkeypatch.setattr(executor, "software_identity", lambda c: IDENTITY)
        return ex, daemon

    yield at
    settings.update(engine="", node="", policy="", discovery_dir="")


def test_a_head_given_its_policy_asks_for_node_0_s_ranks_before_it_loads(head):
    ex, daemon = head(Role.WORKER)
    ex._ask()
    assert (daemon.asked.role, daemon.asked.stage) == (Role.UNDECIDED, (0, 1, 2, 3))
    assert daemon.asked.host == "10.0.0.1"
    assert ex._policy().version == "v1"  # the link's, no document named
    members = tuple(
        RankPlacement(r, f"gpu-{r // 4}", None, f"10.0.0.{1 + r // 4}") for r in range(8)
    )
    daemon.send(EnginePlacement("k", 0, members))
    assert ex.asking.engine_placement(timeout_s=5).members == members
    ex.asking.close()


def test_a_head_whose_ranks_are_another_s_does_not_serve(head):
    ex, _ = head(Role.SPARE)
    with pytest.raises(RuntimeError, match="another process's"):
        ex._ask()
