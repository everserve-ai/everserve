# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The real control plane and an engine's processes, for the worker-loss end to end.

This tree's coordinator and a daemon per node, each a process of the built controller, and the
engine's processes in the Dynamo integration's shape: the head on node 0 holding rank 0, a node
holding rank 1, and a spare standing for rank 1. Each process speaks the adapter's and the
contract's own Python code; only what needs a GPU is stood in for (vLLM's queues, a worker's
collective, the spare's take of its position on the device). Each runs in a process of its own
and puts what it did on the test's queue, ending each event with its node. Each stops once the
file ``stop`` exists: a process the test kills may die waiting, and a shared lock it held then
would never be released, as a multiprocessing ``Event``'s is not.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from everserve.contract.signals import (
    KvFormat,
    Member,
    ParallelWidths,
    SoftwareIdentity,
    StepConfig,
)

ROOT = Path(__file__).resolve().parents[3]
ENGINE = "k"
WORLD = 2
BOUND_S = 30.0
POLICY = """{
  "version": "e2e",
  "outcomesInOrder": ["OUTCOME_REPLACE"],
  "spares": {"perEngine": 1, "loading": "LOADING_PRELOADED"},
  "cover": {"enabled": true},
  "engines": [{
    "name": "k", "tensorParallel": 1, "pipelineParallel": 2, "dataParallel": 1,
    "expertParallel": 1, "guarantee": "GUARANTEE_CONTINUOUS"
  }]
}"""
"""The engine's policy as the Dynamo renderer's is: shape only, placed by what registers."""


def controller_binary() -> str | None:
    """The controller built from this tree, if it is (EVERSERVE_CONTROLLER_BIN, else target/).

    A binary named and missing fails the test rather than skipping it: a CI job whose build left
    none would otherwise pass with nothing tested.
    """
    env = os.environ.get("EVERSERVE_CONTROLLER_BIN")
    if env:
        if not Path(env).exists():
            pytest.fail(f"EVERSERVE_CONTROLLER_BIN names {env}, which does not exist")
        return env
    for profile in ("debug", "release"):
        candidate = ROOT / "target" / profile / "everserve-controller"
        if candidate.exists():
            return str(candidate)
    return None


def identity(serves_changes: bool = False) -> SoftwareIdentity:
    """What every process of the engine runs: every compared field stated, as one build's."""
    return SoftwareIdentity(
        engine="vllm",
        engine_commit="db9527a4",
        adapter_commit="e2e",
        torch_version="2.9.0",
        cuda_version="12.8",
        nccl_version="2.27.3",
        model="m",
        checkpoint_digest="e2e-checkpoint",
        model_dtype="bfloat16",
        kv_format=KvFormat("bfloat16", 16, "e2e-layout"),
        widths=ParallelWidths(tensor=1, pipeline=2),
        step=StepConfig("e2e-step", {"max_num_seqs": "256", "block_size": "16"}),
        guarantee="continuous",
        serves_changes=serves_changes,
    )


def started() -> Any:
    """The calls the engine's core sent its workers at its start, as the spare is replayed them."""
    from everserve_vllm import replay

    kv = SimpleNamespace(num_blocks=1024, kv_cache_tensors=[], kv_cache_groups=[])
    return replay.Calls((("initialize_from_config", ([kv, kv],), {}),))


@dataclass
class Handle:
    """vLLM's queue handle as the join's codec carries it: a dataclass of fields."""

    name: str


def install_fake_queues() -> None:
    """vLLM's message queue module, as far as the join codec reads its handle type."""
    shm = types.ModuleType("vllm.distributed.device_communicators.shm_broadcast")
    shm.Handle = Handle
    for name in ("vllm", "vllm.distributed", "vllm.distributed.device_communicators"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[shm.__name__] = shm


def worker_link(sock: str, node: str, rank: int, role: Any, events) -> Any:
    """``rank``'s process's link to its node's daemon, held to its engine's cover plans.

    Its collective is the adapter's own (``VllmCollective``) and its cover the core's
    (``everserve.comm.cover.Cover``), applying each plan the daemon forwards: every group a plan
    names that spans this process is formed by the real protocol, ``groups.group_id_for`` (every
    member reports itself ready, the coordinator relays the server's id only then), and only the
    NCCL communicator the id would join is stood in for. ``link.boundary()`` is what the
    executor's boundary call runs on a worker after a commit (``worker.apply_change``): the
    change applied, the collective moved, the cover resumed (``worker.after_commit``). The
    executor sends that call over vLLM's ring, which no process here has, so the head calls it on
    its own rank's worker and a spare in its take, as ``spare.take_position`` does.
    """
    import torch

    from everserve.comm.collective import GroupFamily
    from everserve.comm.cover import Cover, LinkReports
    from everserve.comm.handle import LocalStandIn
    from everserve.comm.layout import ParallelLayout, Scope
    from everserve.comm.shadow import replacement
    from everserve.contract.daemon import DaemonConnection
    from everserve.contract.signals import Role
    from everserve_vllm import worker
    from everserve_vllm.collective_op import VllmCollective
    from everserve_vllm.groups import group_id_for

    spare = role is Role.SPARE
    # as collective_op builds it for a rank of a TP 1 x PP 2 engine: its pipeline's scope a family
    layout = ParallelLayout(1, WORLD)
    c = VllmCollective(
        replacement(rank) if spare else rank,
        LocalStandIn(),
        {},
        spare_row=layout.tensor_parallel,
        layout=layout,
        fabric=GroupFamily(Scope.FABRIC, LocalStandIn()),
    )
    c.members = tuple(range(layout.world))
    if spare:
        c.name_process(node, 0)  # as the worker names a spare's ids, before it holds a group
    runtime = worker.AdapterRuntime(rank, WORLD, torch.device("cpu"), c)
    runtime.on_displaced = lambda epoch, why: events.put(("displaced", rank, epoch, why, node))
    link = DaemonConnection(sock, ENGINE, Member(node, 0), runtime, role=role, identity=identity())

    def boundary() -> None:
        # what the executor's boundary call runs on a worker (worker.apply_change), minus the
        # device: the change applied, the collective moved, the cover resumed
        if runtime.at_boundary() | (c.apply_staged() is not None):
            worker.after_commit(c, worker._CHANGE_LOST)
            events.put(("moved", rank, c.epoch, node))

    link.boundary = boundary

    def form(g: Any, generation: int) -> Any:
        me = next(m for m in c.stands_for if m in g.spans)
        if len(g.spans) == 1:
            return LocalStandIn()
        group_id_for(g, me, link, ENGINE, generation, BOUND_S, lambda: os.urandom(128))
        return LocalStandIn()

    cover = Cover(
        lambda g: any(m in g.spans for m in c.stands_for),
        c,
        form,
        lambda scope, spans, group: None,
        LinkReports(ENGINE, lambda: c.me, link),
    )
    worker.COVERS[ENGINE] = worker.COVER = cover
    early = worker._EARLY_PLANS.pop(ENGINE, None)
    if early is not None:
        cover.apply(worker.plan_from(early))
    return link


def ask(sock: str, node: str, rank: int, events) -> Any:
    """This process's ask for its stage, as a node's pod asks with its slot (decision 0041)."""
    from everserve.contract.daemon import UndecidedConnection

    asking = UndecidedConnection(sock, ENGINE, (rank,), Member(node, 0), identity=identity())
    told = asking.assigned(timeout_s=BOUND_S)
    events.put(("told", rank, told.role.value, node))
    return asking, told


def head(sock: str, events, stop) -> None:
    """The engine's head on node 0, holding rank 0, as Dynamo's vLLM worker with Everserve's
    executor runs it: it asks for node 0's rank, its rank's worker registers, and its executor
    registers as the head. It answers each spare's ask for its start-up calls, and after each
    commit offers the taken rank its join, naming the calls' digest (decision 0044)."""
    from everserve.contract.daemon import DaemonClient
    from everserve.contract.signals import Role
    from everserve_vllm import join as joins
    from everserve_vllm import replay

    install_fake_queues()
    node = "gpu-0"
    try:
        asking, _ = ask(sock, node, 0, events)
        link = worker_link(sock, node, 0, Role.WORKER, events)
        client = DaemonClient(
            sock, ENGINE, Member(node, 0), head=True, launcher_pid=os.getpid(),
            identity=identity(serves_changes=True),
        )  # fmt: skip
        events.put(("head", 0, node))
        calls = started()
        took: dict[int, tuple[int, ...]] = {}
        while not os.path.exists(stop):
            try:
                asked = client.replay_asks.get(timeout=0.05)
                # the queue's handle stands for the queue the calls go on (test_replay_vllm reads
                # them across vLLM's own); here it names their digest
                client.offer_replay(asked, f"calls {replay.digest(calls)}")
                events.put(("replay", asked.rank, asked.asker.node, node))
            except queue.Empty:
                pass
            try:
                claimed = client.changes.get_nowait()
                took[claimed.change.epoch] = tuple(claimed.change.lost)
            except queue.Empty:
                pass
            try:
                heard = client.committed.get_nowait()
            except queue.Empty:
                continue
            epoch = heard.committed.epoch
            events.put(("commit", epoch, list(took.get(epoch, ())), node))
            link.boundary()  # its own rank's worker, at the step boundary after the commit
            for rank in took.get(epoch, ()):
                offer = joins.Offer(
                    rank, epoch, f"change-{epoch}", Handle(f"ring {epoch}"),
                    startup=replay.digest(calls),
                )  # fmt: skip
                client.join(rank, epoch, joins.encode(offer))
                answer = joins.decode(client.joins.get(timeout=BOUND_S).payload)
                events.put(("answer", rank, epoch, answer.status, answer.error, node))
        link.close()
        asking.close()
        client.close()
    except Exception as e:  # noqa: BLE001 - the test reads why the head stopped
        events.put(("failed", 0, f"{type(e).__name__}: {e}", node))


def rank_node(sock: str, node: str, rank: int, events, stop, leave, released) -> None:
    """A node told to serve ``rank``: its worker registers and serves until ``leave``, when it
    gives its rank notice and leaves once a commit no longer places it there, as a node pod being
    deleted does (``node_main.leave_on_sigterm``): the departing process applies nothing, and the
    head's boundary releases it once the change committed. That release travels over vLLM's ring;
    here the test sets ``released`` once the head heard the commit."""
    install_fake_queues()
    try:
        asking, told = ask(sock, node, rank, events)
        link = worker_link(sock, node, rank, told.role, events)
        events.put(("serving", rank, node))
        while not os.path.exists(stop) and not leave.wait(0.05):
            pass
        if leave.is_set():
            lead_s = link.notice_lead_s(1.0)
            link.notice(rank, int((time.time() + lead_s) * 1000))
            events.put(("left", rank, released.wait(BOUND_S), node))
        link.close()
        asking.close()
    except Exception as e:  # noqa: BLE001 - the test reads why the node stopped
        events.put(("failed", rank, f"{type(e).__name__}: {e}", node))


def spare_node(sock: str, node: str, rank: int, events, stop) -> None:
    """A node told to stand as ``rank``'s spare: the spare asks the engine's core for the rank's
    start-up calls, stands, and takes the position at the first join offered it, checked against
    the calls (``spare_main.join_position``); only its take on the device is stood in for. It then
    serves the rank until it is killed or stopped."""
    from everserve.contract.daemon import ask_replay
    from everserve.contract.signals import Role
    from everserve_vllm import replay, spare, spare_main

    install_fake_queues()
    try:
        asking, told = ask(sock, node, rank, events)
        if told.role is not Role.SPARE:
            raise RuntimeError(f"told {told.role.value}, not to stand as a spare")
        offer = ask_replay(sock, ENGINE, rank, BOUND_S)
        built_from = started()
        if offer.handle != f"calls {replay.digest(built_from)}":
            raise RuntimeError(f"replayed other calls than the core's: {offer.handle}")
        events.put(("replayed", rank, node))
        link = worker_link(sock, node, rank, Role.SPARE, events)
        events.put(("standing", rank, node))

        def take_position(proc, rank, ring, connect_ip, catch_up):
            link.boundary()  # the spare moves into the new set, as the real take does
            return {"handle": Handle(f"out {node}"), "applied": {}, "restore": None, "join_s": 0.0}

        spare.take_position = take_position
        spare_main.join_position(link, rank, "proc", built_from, "127.0.0.1", None, {})
        events.put(("took", rank, node))
        while not os.path.exists(stop):
            time.sleep(0.05)
        link.close()
        asking.close()
    except Exception as e:  # noqa: BLE001 - the test reads why the spare stopped
        events.put(("failed", rank, f"{type(e).__name__}: {e}", node))


class ControlPlane:
    """The coordinator and a daemon per node, each a process of this tree's controller.

    The coordinator serves its operator API over TLS as a deployment's does (decision 0039),
    certificates from the controller's own ``certs``: the engine's conditions and events are read
    there, as an operator reads them (:meth:`engine`).
    """

    def __init__(self, binary: str, root: Path, nodes: list[str], policy: str) -> None:
        self.binary, self.root, self.procs, self.logs = binary, root, [], {}
        written = root / "policy.json"
        written.write_text(policy)
        coord = f"unix:{root / 'coord.sock'}"
        certs = root / "certs"
        subprocess.run(
            [binary, "certs", "--out", str(certs), "--operator", "e2e"],
            check=True, capture_output=True,
        )  # fmt: skip
        addr = f"127.0.0.1:{free_port()}"
        self.operator_env = {
            "EVERSERVE_API": addr,
            "EVERSERVE_API_CA": str(certs / "ca.crt"),
            "EVERSERVE_API_CERT": str(certs / "operator-e2e.crt"),
            "EVERSERVE_API_KEY": str(certs / "operator-e2e.key"),
        }
        self._start(
            "coordinator",
            [binary, "coord", "--policy", str(written), "--listen", coord,
             "--lease", str(root / "coord.lease"), "--api-addr", addr, "--api-operators", "e2e",
             "--tls-cert", str(certs / "coordinator.crt"),
             "--tls-key", str(certs / "coordinator.key"), "--tls-ca", str(certs / "ca.crt")],
        )  # fmt: skip
        self.socks = {}
        for n in nodes:
            (root / n).mkdir()
            self.socks[n] = str(root / n / "node.sock")
            self._start(
                n,
                [binary, "node", "--coordinator", coord, "--socket", self.socks[n],
                 "--node", n, "--discovery-dir", str(root / n), "--notice-source", "none"],
            )  # fmt: skip

    def _start(self, name: str, cmd: list[str]) -> None:
        log = self.root / f"{name}.log"
        self.logs[name] = log
        env = {**os.environ, "EVERSERVE_LOG": "info"}
        f = log.open("w")
        self.procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env))

    def _get(self, path: str) -> Any:
        """``GET /v1/<path>`` on the operator API, as an operator's client reads it."""
        import json
        import ssl
        import urllib.request

        env = self.operator_env
        tls = ssl.create_default_context(cafile=env["EVERSERVE_API_CA"])
        tls.load_cert_chain(env["EVERSERVE_API_CERT"], env["EVERSERVE_API_KEY"])
        tls.check_hostname = False  # the coordinator's certificate names its service, not the IP
        url = f"https://{env['EVERSERVE_API']}/v1/{path}"
        with urllib.request.urlopen(url, context=tls, timeout=BOUND_S) as answer:
            return json.loads(answer.read())

    def condition(self, kind: str) -> tuple[str, str]:
        """The engine's condition ``kind``: its truth and its reason, empty while unread."""
        try:
            status = self._get(f"engines/{ENGINE}").get("status") or {}
        except OSError:
            return "", ""  # the API not listening yet
        for c in status.get("conditions") or []:
            if c.get("type") == kind:
                return c.get("status", ""), c.get("reason", "")
        return "", ""

    def covered_by(self, spare: str, epoch: int) -> bool:
        """Whether the engine reads every loss covered under a plan for ``epoch`` with a set
        naming the spare on node ``spare``, so a reading left from before a change never counts."""
        if self.condition("Covered") != ("True", "EveryLossCovered"):
            return False
        try:
            cover = (self._get(f"engines/{ENGINE}").get("status") or {}).get("cover") or {}
        except OSError:
            return False
        return cover.get("epoch") == epoch and any(
            f"@{spare}:" in s.get("set", "") for s in cover.get("sets") or []
        )

    def events(self) -> list[str]:
        """The engine's event reasons, oldest first."""
        return [e["reason"] for e in self._get(f"engines/{ENGINE}/events")]

    def coordinator_log(self) -> str:
        return self.logs["coordinator"].read_text(errors="replace")

    def coordinator_says(self, text: str) -> bool:
        return text in self.coordinator_log()

    def dump(self) -> str:
        return "\n".join(
            f"== {name}\n{path.read_text(errors='replace')[-6000:]}"
            for name, path in self.logs.items()
        )

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


def free_port() -> int:
    """A TCP port nothing on this host listens on, for a listener started next."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def until(what: str, ok, plane: ControlPlane, timeout_s: float = BOUND_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not ok():
        if time.monotonic() > deadline:
            pytest.fail(f"{what}: never, within {timeout_s} s\n{plane.dump()}")
        time.sleep(0.1)
