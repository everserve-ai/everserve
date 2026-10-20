# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The Kubernetes test's probe asks for its slot's stage and reports what it hears."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from google.protobuf import json_format

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

PROBE = Path(__file__).parents[1] / "k8s" / "stage_probe.py"
POLICY = {
    "version": "e2e-1",
    "outcomesInOrder": ["OUTCOME_REPLACE"],
    "spares": {"loading": "LOADING_PRELOADED"},
    "cover": {"enabled": True},
    "engines": [{"name": "k", "tensorParallel": 2, "pipelineParallel": 2}],
}


def test_a_probe_asks_for_its_stage_and_reports_its_policy_answer_and_placement():
    path = str(Path(tempfile.mkdtemp()) / "node.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    asked = {}

    def daemon():
        conn, _ = srv.accept()
        asked["reg"] = from_message(pb.Signal.FromString(read_frame(conn)))
        doc = json_format.Parse(json.dumps(POLICY), policy_pb2.Policy()).SerializeToString()
        for sig in (
            EnginePolicy("k", doc),
            RoleAssigned("k", 2, 0, Role.WORKER, (2, 3)),
            EnginePlacement(
                "k",
                0,
                tuple(RankPlacement(r, f"n{r // 2}", None, f"10.0.0.{r // 2}") for r in range(4)),
            ),
        ):
            write_frame(conn, to_message(sig).SerializeToString())
        asked["conn"] = conn

    threading.Thread(target=daemon, daemon=True).start()
    env = {
        **os.environ,
        "EVERSERVE_DAEMON_SOCKET": path,
        "EVERSERVE_HOST_IP": "10.0.0.1",
        "EVERSERVE_NODE": "n1",
    }
    p = subprocess.Popen(
        [
            sys.executable,
            str(PROBE),
            "--engine",
            "k",
            "--world",
            "4",
            "--nnodes",
            "2",
            "--slot",
            "0",
        ],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    lines = []
    for line in p.stdout:
        lines.append(json.loads(line))
        if "placement" in lines[-1]:
            break
    p.send_signal(signal.SIGTERM)
    rest = p.communicate(timeout=10)[0]
    lines += [json.loads(x) for x in rest.splitlines() if x]
    assert asked["reg"].role is Role.UNDECIDED and asked["reg"].stage == (2, 3)
    assert asked["reg"].host == "10.0.0.1"
    got = {k: v for line in lines for k, v in line.items()}
    assert got["asked"] == [2, 3] and got["policy"] == "e2e-1" and got["world"] == 4
    assert got["told"] == "worker"
    assert got["members"]["3"] == ["n1", "10.0.0.1"]
    assert got["ended"] == "stopped"
