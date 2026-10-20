# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A node asks what it is before it loads anything (decision 0041), or is placed by its operator."""

import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
from google.protobuf import json_format

from everserve.contract.daemon import EXIT_REFUSED, handed_over
from everserve.contract.signals import (
    EnginePlacement,
    EnginePolicy,
    Goodbye,
    RankPlacement,
    Role,
    RoleAssigned,
    SoftwareIdentity,
    SparePlacement,
    StepConfig,
    StopCause,
    from_message,
    to_message,
)
from everserve.policy import Guarantee
from everserve.proto import policy_pb2
from everserve.proto import signals_pb2 as pb
from everserve.wire import read_frame, write_frame
from everserve_vllm import identity, node_main

IDENTITY = SoftwareIdentity("vllm", "v0.31.0", "c0ffee", step=StepConfig("d1", {"a": "1"}))

ARGS = ["--model", "/m", "--tensor-parallel-size", "4", "--pipeline-parallel-size", "4"]
BASE = ["--engine", "k", "--node", "gpu-3", "--nnodes", "4", "--master-addr", "k-head.serving.svc"]
ASKS = ["--guarantee", "continuous"]  # an asking node resolves with its engine's guarantee
POLICY = {
    "version": "v7",
    "outcomesInOrder": ["OUTCOME_REPLACE"],
    "spares": {"loading": "LOADING_PRELOADED"},
    "cover": {"enabled": True},
    "engines": [
        {
            "name": "k",
            "tensorParallel": 4,
            "pipelineParallel": 4,
            "guarantee": "GUARANTEE_CONTINUOUS",
        }
    ],
}


PLACED = EnginePlacement(
    "k",
    0,
    tuple(RankPlacement(r, f"gpu-{r // 4}", None, f"10.0.0.{r // 4}") for r in range(16)),
    (SparePlacement(None, "gpu-3", 0, "10.0.0.3"),),
)


def encoded():
    import json

    return json_format.Parse(json.dumps(POLICY), policy_pb2.Policy()).SerializeToString()


def test_a_node_is_placed_or_asks_never_both():
    assert node_main.parse([*BASE, "--node-rank", "2", "--policy", "/p", *ARGS]).node_rank == 2
    a = node_main.parse([*BASE, "--slot", "5", *ASKS, *ARGS])  # given its policy on its link
    assert (a.slot, a.node_rank, a.discovery_dir, a.policy) == (5, None, "", "")
    for bad in (
        [],
        ["--node-rank", "1", "--slot", "0"],
        ["--slot", "-1"],
        ["--node-rank", "1"],  # a placed node reads its policy document
        ["--slot", "0"],  # an asking node names its guarantee
    ):
        with pytest.raises(SystemExit):
            node_main.parse([*BASE, *bad, *ARGS])


def test_slots_ask_for_the_non_head_nodes_in_turn_and_each_node_s_ranks_are_its_stage():
    # four nodes: the head is node 0; slots 0-2 are nodes 1-3 at bring-up, 3-5 stand for them
    assert [node_main.asked_node_rank(s, 4) for s in range(6)] == [1, 2, 3, 1, 2, 3]
    assert node_main.stage_of(2, 16, 4) == (8, 9, 10, 11)
    with pytest.raises(ValueError):
        node_main.stage_of(1, 12, 5)


class FakeDaemon:
    def __init__(self, role, refusal=""):
        self.path = str(Path(tempfile.mkdtemp()) / "node.sock")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        self.sock.listen(1)
        self.asked = None
        self.goodbye = None
        threading.Thread(target=self._answer, args=(role, refusal), daemon=True).start()

    def _answer(self, role, refusal):
        conn, _ = self.sock.accept()
        self.asked = from_message(pb.Signal.FromString(read_frame(conn)))
        if self.asked.role is Role.UNDECIDED:
            # as the daemon does: the engine's policy first, the answer after; a refused
            # process is given no policy, only its refusal
            if not refusal:
                write_frame(conn, to_message(EnginePolicy("k", encoded())).SerializeToString())
            told = RoleAssigned("k", self.asked.rank, 0, role, self.asked.stage, refusal)
            write_frame(conn, to_message(told).SerializeToString())
            if refusal:
                # the refused process says why; the daemon has nothing more for it
                self.goodbye = from_message(pb.Signal.FromString(read_frame(conn)))
                self.conn = conn
                return
            # and where the engine runs, once every rank of its world is placed; a node told to
            # stand as a spare may have ended its asking connection already
            try:
                write_frame(conn, to_message(PLACED).SerializeToString())
            except BrokenPipeError:
                pass
        self.conn = conn


@pytest.fixture
def pod(monkeypatch):
    def at(role, refusal=""):
        d = FakeDaemon(role, refusal)
        monkeypatch.setenv("EVERSERVE_DAEMON_SOCKET", d.path)
        monkeypatch.setenv("EVERSERVE_HOST_IP", "10.0.0.7")
        # the configuration an asking node resolves before it asks, and its identity, as vLLM
        # would give them in the pod
        monkeypatch.setattr(node_main, "resolve", lambda a, rank, g, record: ("config", rank, g))
        monkeypatch.setattr(identity, "software_identity", lambda c: IDENTITY)
        # the ranks' compiled sets in place (stage_compiled's own tests read the link)
        monkeypatch.setattr(node_main, "stage_compiled", lambda link, stage: None)
        return d

    return at


def test_a_node_told_to_serve_serves_the_node_rank_it_asked_for(pod, monkeypatch):
    daemon = pod(Role.WORKER)
    served = {}

    def serve(a, rank, engine, record, stop, health, vllm_config=None):
        served.update(rank=rank, engine=engine, config=vllm_config)

    monkeypatch.setattr(node_main, "serve", serve)
    assert node_main.main([*BASE, "--slot", "4", *ASKS, *ARGS]) == 0
    assert served["rank"] == 2
    assert served["engine"].guarantee.value == "continuous"  # the policy its link gave it
    # what it serves with is what it resolved before it asked, and the ask carried its step
    assert served["config"] == ("config", 2, Guarantee.CONTINUOUS)
    assert daemon.asked.identity.step == IDENTITY.step
    assert (daemon.asked.role, daemon.asked.rank) == (Role.UNDECIDED, 8)
    assert daemon.asked.stage == (8, 9, 10, 11)
    assert daemon.asked.member.node == "gpu-3"


def test_a_node_told_to_stand_runs_a_spare_per_rank_each_on_its_gpu_and_ends_with_one(
    pod, monkeypatch, tmp_path
):
    pod(Role.SPARE)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    commands = node_main.spare_commands(
        node_main.parse([*BASE, "--slot", "0", *ASKS, *ARGS]), (4, 5)
    )
    assert [gpu for _, gpu in commands] == ["0", "1"]
    cmd = commands[1][0]
    assert cmd[cmd.index("--rank") + 1] == "5" and cmd[-len(ARGS) :] == ARGS
    assert "--given" in cmd and "--policy" not in cmd  # handed on its standard input
    assert cmd[cmd.index("--node-devices") + 1] == "0,1"

    seen = tmp_path / "gpus"

    def spares(a, stage):
        # each notes its GPU; the last ends with an error, as a spare whose load failed
        out = []
        for i, rank in enumerate(stage):
            code = 3 if i == len(stage) - 1 else 0
            note = (
                f"import os, sys, time; open({str(seen)!r} + '-{rank}', 'w')"
                f".write(os.environ['CUDA_VISIBLE_DEVICES']);"
                f" open({str(seen)!r} + '-{rank}.in', 'wb').write(sys.stdin.buffer.read());"
                # its leave pipe: named on its command line, and open in it
                f" fd = int(sys.argv[sys.argv.index('--leave-fd') + 1]); os.fstat(fd);"
                f" open({str(seen)!r} + '-{rank}.leave', 'w').write('open');"
                f" time.sleep(0.5 if {code} else 60); sys.exit({code})"
            )
            out.append(([sys.executable, "-c", note], str(i)))
        return out

    monkeypatch.setattr(node_main, "spare_commands", spares)
    monkeypatch.setattr(node_main, "serve", lambda *_: pytest.fail("a spare does not serve"))
    assert (
        node_main.main([*BASE, "--slot", "0", *ASKS, *ARGS]) == 3
    )  # the pod restarts and asks again
    assert {Path(f"{seen}-{r}").read_text() for r in (4, 5, 6, 7)} == {"0", "1", "2", "3"}
    assert {Path(f"{seen}-{r}.leave").read_text() for r in (4, 5, 6, 7)} == {"open"}
    # each was handed the engine's policy and placement, as the pod's process heard them
    with open(f"{seen}-5.in", "rb") as handed:
        policy, placement = handed_over(handed)
    assert (policy.version, placement) == ("v7", PLACED)


def test_a_placed_node_asks_nothing_and_reads_its_policy_document(pod, monkeypatch, tmp_path):
    import json

    daemon = pod(Role.SPARE)
    (tmp_path / "p.json").write_text(json.dumps(POLICY))
    served = {}

    def serve(a, rank, engine, record, stop, health, vllm_config=None):
        served.update(rank=rank, engine=engine.name)

    monkeypatch.setattr(node_main, "serve", serve)
    argv = [*BASE, "--node-rank", "3", "--policy", str(tmp_path / "p.json"), *ARGS]
    assert node_main.main(argv) == 0
    assert served == {"rank": 3, "engine": "k"} and daemon.asked is None


class Link:
    """A daemon link's state as a process hears it."""

    def __init__(self, *args, **kw):
        self.closed = threading.Event()
        self.state = None


def _down():
    from everserve.contract.signals import InstancePhase, InstanceState

    return InstanceState("k", InstancePhase.FAILED, 1, 2, 3, reason="head failed: mirror")


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(node_main, "SUPERVISE_S", 0.01)
    monkeypatch.setattr(node_main, "DOWN", threading.Event())
    monkeypatch.setattr(node_main, "EXITED", threading.Event())


def test_a_node_whose_engine_goes_down_stops_and_exits_as_down(fresh):
    from everserve.contract.signals import InstancePhase, InstanceState

    link, stop = Link(), threading.Event()
    node_main.end_when_down(link, stop, grace_s=5)
    link.state = InstanceState("k", InstancePhase.DEGRADED, 1, 2, 3)  # serving short: not down
    assert not stop.wait(0.1)
    link.state = _down()
    assert stop.wait(2) and node_main.DOWN.is_set()
    node_main.EXITED.set()  # main returned in its grace: no forced exit


def test_a_node_that_cannot_stop_in_its_grace_is_ended(fresh, monkeypatch):
    ended, codes = threading.Event(), []

    def _exit(code):
        codes.append(code)
        ended.set()

    monkeypatch.setattr(node_main.os, "_exit", _exit)
    link, stop = Link(), threading.Event()
    node_main.end_when_down(link, stop, grace_s=0.05)
    link.state = _down()  # still loading, say: main never returns
    assert ended.wait(2) and codes == [node_main.EXIT_ENGINE_DOWN]


def test_a_spare_node_of_a_downed_engine_exits_as_down(fresh, monkeypatch, tmp_path):
    import json

    import everserve.contract.daemon as daemon_mod
    import everserve.discovery as discovery

    monkeypatch.setattr(node_main, "resolve", lambda a, rank, g, record: ("config", rank, g))
    monkeypatch.setattr(identity, "software_identity", lambda c: IDENTITY)
    monkeypatch.setattr(node_main, "stage_compiled", lambda link, stage: None)

    class Asking(Link):
        def engine_policy(self):
            raise AssertionError("a policy document is named")

        def assigned(self):
            self.state = _down()  # the engine goes down as the spare stands
            return RoleAssigned("k", 1, 0, Role.SPARE, (1,))

        def close(self):
            self.closed.set()

    def stand(a, stage, stop, health, handed):
        assert stop.wait(5)  # the watcher stopped the spares' supervision
        return 0

    monkeypatch.setattr(node_main, "stand", stand)
    monkeypatch.setattr(daemon_mod, "UndecidedConnection", Asking)
    monkeypatch.setattr(discovery, "local_daemon", lambda node, d: type("R", (), {"address": ""}))
    (tmp_path / "p.json").write_text(json.dumps(POLICY))
    argv = [*BASE, "--slot", "4", *ASKS, "--policy", str(tmp_path / "p.json"), *ARGS]
    assert node_main.main(argv) == node_main.EXIT_ENGINE_DOWN


def test_a_node_the_coordinator_refused_says_why_and_exits_refused(pod, monkeypatch):
    why = 'vllm.scheduler is "b08c" here, "77aa" on the engine'
    daemon = pod(Role.WORKER, refusal=why)  # WORKER is the refusal's default role: never read
    monkeypatch.setattr(node_main, "serve", lambda *_: pytest.fail("a refused node does not serve"))
    assert node_main.main([*BASE, "--slot", "4", *ASKS, *ARGS]) == EXIT_REFUSED
    deadline = time.monotonic() + 5
    while daemon.goodbye is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert daemon.goodbye == Goodbye(cause=StopCause.REFUSED, detail=why)


def test_a_node_started_with_another_guarantee_than_its_engine_s_never_serves(pod, monkeypatch):
    pod(Role.WORKER)  # the policy the link gives is continuous
    monkeypatch.setattr(node_main, "serve", lambda *_: pytest.fail("never served"))
    with pytest.raises(SystemExit, match="continuous in its policy"):
        node_main.main([*BASE, "--slot", "4", "--guarantee", "exact", *ARGS])


class CacheLink:
    """A spare node's link that answers its asks for compiled sets."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def cache(self, rank, dp=0, timeout_s=None):
        self.asked.append(rank)
        answer = self.answers[rank]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _staged(root, key, rank):
    from everserve.contract.compile_cache import rank_manifest, write_manifest

    art = root / "torch_compile_cache" / "torch_aot_compile" / key
    (art / f"rank_{rank}_0").mkdir(parents=True)
    (art / f"rank_{rank}_0" / "model").write_bytes(f"graph {rank}".encode())
    m = rank_manifest(art, rank)
    write_manifest(art, m)
    return m


def test_a_spare_node_stands_only_once_each_rank_s_compiled_set_is_in_place_and_checked(
    monkeypatch, tmp_path
):
    from everserve.contract.daemon import CacheUnavailable
    from everserve.contract.signals import CacheReady

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    key = "e" * 64
    m4, m5 = _staged(tmp_path, key, 4), _staged(tmp_path, key, 5)
    good = {4: CacheReady("k", 4, key, m4.digest), 5: CacheReady("k", 5, key, m5.digest)}
    link = CacheLink(good)
    assert node_main.stage_compiled(link, (4, 5)) is None and link.asked == [4, 5]
    # the daemon could not put a set in place: the node says why and stands for nothing
    refused = CacheLink({**good, 5: CacheUnavailable("no holder of rank 5's set in 600 s")})
    assert "no holder" in node_main.stage_compiled(refused, (4, 5))
    # a set that landed but digests otherwise than its position: never loaded
    other = CacheLink({**good, 4: CacheReady("k", 4, key, "f" * 64)})
    assert "as its position holds it" in node_main.stage_compiled(other, (4, 5))


@pytest.fixture
def leaving(monkeypatch):
    monkeypatch.setattr(node_main, "LEAVING", threading.Event())
    monkeypatch.setattr(node_main, "SERVING", [])
    return node_main.LEAVING


def test_a_node_not_serving_that_is_told_to_leave_stops_at_once(leaving):
    stop = threading.Event()
    node_main.leave_on_sigterm(stop, lead_s=5, within_s=60)
    assert not stop.wait(0.1)
    leaving.set()  # SIGTERM: a spare node, or one still loading
    assert stop.wait(2)


def test_a_serving_node_told_to_leave_gives_its_ranks_notice_and_stops_after_its_bound(leaving):
    noticed = []

    class Serving:
        def leave(self, lead_s):
            noticed.append(lead_s)
            return [4, 5]

    node_main.SERVING.append(Serving())
    stop = threading.Event()
    node_main.leave_on_sigterm(stop, lead_s=5, within_s=0.3)
    leaving.set()
    assert not stop.wait(0.1)  # still serving: its ranks move at the notice's deadline
    assert noticed == [5]
    assert stop.wait(2)  # they did not move within its bound: it stops (its loss is reported)


def test_a_standing_node_told_to_leave_passes_it_to_its_spares_and_stops_once_they_end(
    leaving, monkeypatch
):
    """A spare that took a rank serves it inside the standing pod; the pod's deletion alone would
    stop it at once, its rank lost rather than noticed. The leave goes to the spares."""

    import os

    class Spare:
        pid, code = 0, None

        def poll(self):
            return self.code

    a, b = Spare(), Spare()
    (ra, wa), (rb, wb) = os.pipe(), os.pipe()
    monkeypatch.setattr(node_main, "STANDING", [(a, wa), (b, wb)])
    stop = threading.Event()
    node_main.leave_on_sigterm(stop, lead_s=5, within_s=60)
    leaving.set()
    assert not stop.wait(0.3)  # the spares still run: one may be serving its rank's last steps
    assert os.read(ra, 8) == os.read(rb, 8) == node_main.LEAVE_BYTE  # each told on its own pipe
    a.code = b.code = 0
    assert stop.wait(2)
    for fd in (ra, wa, rb, wb):
        os.close(fd)
