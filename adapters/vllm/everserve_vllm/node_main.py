# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A node without the engine: ``python -m everserve_vllm.node_main`` (decision 0022's node role).

In place of ``vllm serve --headless``, whose executor is stock. It builds the engine
configuration the head runs with this node's rank among the nodes, starts
:class:`everserve_vllm.node.NodeExecutor`, which launches the node's workers into vLLM's
multi-node world, and serves until stopped (SIGTERM or SIGINT) or until every position it held
has moved to another node and its workers have been released (:mod:`everserve_vllm.vacate`).

A node's rank is given (``--node-rank``: the operator placed it, on bare metal or a rig) or asked
for (``--slot``: one pod of a framework's group, decision 0041). An asking node registers with its
node's daemon undecided for its node's ranks, before it loads anything, and is told to serve them
or to stand as their spare: one spare process (:mod:`everserve_vllm.spare_main`) per rank, each on
its own GPU. Slot ``s`` asks for node rank ``1 + s mod (nnodes - 1)``: the head is node 0, the
group's first ``nnodes - 1`` pods are the engine's other nodes at bring-up, and the rest stand
for them in turn, as does a pod the group recreates for one whose ranks a spare took.

An asking node, serving or standing, ends once its engine is down (the instance's state, heard
on its link, is failed), with :data:`EXIT_ENGINE_DOWN`: the coordinator starts a downed engine
anew once none of its serving processes is left, and the framework's new pod asks again.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from everserve.contract.signals import InstancePhase
from everserve_vllm import engine_args

if TYPE_CHECKING:
    from everserve.discovery import DaemonRecord
    from everserve.policy import Engine
    from everserve_vllm.health import Health

log = logging.getLogger(__name__)

SUPERVISE_S = 1.0
"""How often a spare node's process looks at its spares."""

EXIT_ENGINE_DOWN = 3
"""The exit status of a process that ended because its engine is down (decision 0032).

A node of a downed engine holds nothing worth keeping: the engine's world is gone with its head,
and its workers would wait on it forever, the pod standing. The coordinator starts the engine
anew only once no
process of the old incarnation is registered, so the node ends, its framework starts the pod
again, and the new pod asks as at bring-up."""

DOWN_GRACE_S = 30.0
"""How long a node of a downed engine is given to stop its workers before it exits regardless."""


def end_when_down(link: Any, stop: threading.Event, grace_s: float = DOWN_GRACE_S) -> None:
    """Watch ``link``'s instance state; once the engine is down, stop, then exit if still here.

    ``link`` hears the instance's state from the node's daemon (protocol 11) for the process's
    life. ``stop`` ends the serve loop or a spare's supervision; a process still loading (whose
    construction waits on the dead head's world) or one whose shutdown hangs is ended after
    ``grace_s``, with :data:`EXIT_ENGINE_DOWN`.
    """

    def watch() -> None:
        while not link.closed.wait(SUPERVISE_S):
            state = link.state
            if state is not None and state.phase is InstancePhase.FAILED:
                log.error(
                    "engine %s is down%s: this node ends, and its pod asks again",
                    state.engine,
                    f": {state.reason}" if state.reason else "",
                )
                DOWN.set()
                stop.set()
                if not EXITED.wait(grace_s):
                    log.error("still here %.0f s after the engine went down: exiting", grace_s)
                    logging.shutdown()
                    os._exit(EXIT_ENGINE_DOWN)
                return

    threading.Thread(target=watch, daemon=True, name="EverserveEngineDown").start()


LEAVING = threading.Event()
"""Set by SIGTERM: this pod is being deleted, and its node leaves (:func:`leave_on_sigterm`)."""
SERVING: list[Any] = []
"""The node executor while it serves: what a leaving node gives notice through."""
STANDING: list[tuple[Any, int]] = []
"""The spare processes while this node stands, each with the write end of its leave pipe: a
leaving node writes the leave there (:func:`tell_leave`)."""
NOTICE_LEAD_S = 5.0
"""How far ahead a leaving node's notice puts its ranks' change: the change runs at the
deadline, the node serving until then."""
LEAVE_WITHIN_S = 60.0
"""How long a leaving node waits for its ranks to move before it stops regardless (its loss is
then reported as a worker's departure): within the pod's grace period, which the renderer sets
with room past it."""


LEAVE_BYTE = b"L"
"""What a standing node writes on a spare's leave pipe when its pod is going."""


def tell_leave(standing: list[tuple[Any, int]]) -> list[Any]:
    """Write the leave on every running spare's pipe; the spares, in order.

    A pipe this process made and handed each spare at its start (``spare_main --leave-fd``), not
    a signal: a signal is shared with every library the spare loads, and a leave sent as SIGUSR1
    is taken by the plugin's stack dump (faulthandler) and never reaches the spare.
    """
    procs = []
    for p, fd in standing:
        procs.append(p)
        if p.poll() is None:
            try:
                os.write(fd, LEAVE_BYTE)
            except OSError as e:  # its end closed: the spare is ending already
                log.info("spare %d's leave pipe is closed: %s", p.pid, e)
    return procs


def leave_on_sigterm(stop: threading.Event, lead_s: float, within_s: float) -> None:
    """Once :data:`LEAVING` is set, this node leaves.

    A serving node gives its ranks notice and stops once they moved, or after ``within_s``. A
    standing node passes the leave on to its spares (:func:`tell_leave`): one that took its rank
    serves it, and gives it notice as a serving node does. A node still loading stops at once.
    """

    def leave() -> None:
        LEAVING.wait()
        if STANDING:
            procs = tell_leave(STANDING)
            log.info("leaving: %d spares told; stopping once they ended", len(procs))
            end = time.monotonic() + within_s + 5.0  # their own bound, and their close
            while any(p.poll() is None for p in procs) and time.monotonic() < end:
                time.sleep(0.2)
            stop.set()
            return
        if not SERVING:
            stop.set()
            return
        ranks = SERVING[0].leave(lead_s)
        log.info(
            "leaving: %d ranks noticed; stopping once they moved, %.0f s at most",
            len(ranks),
            within_s,
        )
        # the serve loop returns once the node holds nothing; past the bound, it stops anyway
        if not stop.wait(within_s):
            log.warning("leaving: ranks %s not moved in %.0f s: stopping", ranks, within_s)
            stop.set()

    threading.Thread(target=leave, daemon=True, name="EverserveLeave").start()


DOWN = threading.Event()
"""Set once this process's engine was heard down: :func:`main` exits with
:data:`EXIT_ENGINE_DOWN`."""
EXITED = threading.Event()
"""Set as :func:`main` returns: a down watcher's grace ends early."""


def settings_guarantee() -> str | None:
    """The guarantee the pod's environment states (EVERSERVE_GUARANTEE), or None."""
    from everserve_vllm import settings

    return settings.current().guarantee or None


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line: this node, the policy and discovery, and the node's placement.

    Every other argument is vLLM's engine arguments, the head's own (:mod:`everserve_vllm.
    engine_args`), kept in ``engine_argv`` and passed to vLLM's parser untouched.
    """
    ap = argparse.ArgumentParser(prog="everserve_vllm.node_main")
    ap.add_argument("--engine", required=True, help="the engine's name in the policy")
    ap.add_argument("--node", required=True, help="this node's name in the policy")
    ap.add_argument(
        "--policy",
        default="",
        help="the policy document (bare metal, a rig); an asking node is given its engine's on its "
        "link (protocol 11)",
    )
    ap.add_argument(
        "--discovery-dir",
        default="",
        help="where daemons publish their records (bare metal); unused under the pod contract",
    )
    ap.add_argument("--nnodes", type=int, required=True, help="nodes the engine spans")
    placed = ap.add_mutually_exclusive_group(required=True)
    placed.add_argument("--node-rank", type=int, help="this node's rank among them, as placed")
    placed.add_argument(
        "--slot",
        type=int,
        help="this pod's index in its group: the node asks the control plane what it is",
    )
    ap.add_argument("--master-addr", required=True, help="the head's reachable address")
    ap.add_argument("--master-port", type=int, default=29501)
    ap.add_argument(
        "--guarantee",
        choices=["exact", "continuous"],
        default=settings_guarantee(),
        help="the engine's guarantee class (policy Engine.guarantee): an asking node resolves its "
        "configuration with it before it asks, and holds the policy it is given to it",
    )
    ap.add_argument(
        "--notice-lead-s",
        type=float,
        default=NOTICE_LEAD_S,
        help="on SIGTERM, a serving node's ranks move this far ahead",
    )
    ap.add_argument(
        "--leave-within-s",
        type=float,
        default=LEAVE_WITHIN_S,
        help="on SIGTERM, how long a serving node waits for its ranks to move before it stops",
    )
    ap.add_argument(
        "--health-port",
        type=int,
        default=None,
        help="serve /live and /ready on this port for the pod's probes (everserve_vllm.health)",
    )
    a, a.engine_argv = ap.parse_known_args(argv)
    ident = engine_args.identity(a.engine_argv)
    a.model, a.tp, a.pp = ident.model, ident.tp, ident.pp
    if a.slot is not None and (a.slot < 0 or a.nnodes < 2):
        ap.error("--slot needs a non-negative index and an engine on two nodes or more")
    if a.slot is not None and a.guarantee is None:
        ap.error(
            "an asking node (--slot) names its engine's guarantee: --guarantee, or "
            "EVERSERVE_GUARANTEE"
        )
    if a.node_rank is not None and not a.policy:
        ap.error("a placed node (--node-rank) reads its --policy")
    return a


def stage_of(node_rank: int, world: int, nnodes: int) -> tuple[int, ...]:
    """The ranks node ``node_rank`` of ``nnodes`` holds, in a world of ``world``.

    Raises:
        ValueError: when the ranks do not fill the nodes evenly.
    """
    if world % nnodes:
        raise ValueError(f"{world} ranks do not fill {nnodes} nodes evenly")
    per = world // nnodes
    return tuple(range(node_rank * per, (node_rank + 1) * per))


def asked_node_rank(slot: int, nnodes: int) -> int:
    """The node rank slot ``slot`` asks for: the head's is never asked by a node."""
    return 1 + slot % (nnodes - 1)


def devices() -> list[str]:
    """This pod's GPUs, as the CUDA runtime numbers them for a child."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return [d.strip() for d in visible.split(",") if d.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def spare_commands(a: argparse.Namespace, stage: tuple[int, ...]) -> list[tuple[list[str], str]]:
    """Each spare's command for ``stage`` and the GPU it runs on, one per rank.

    Raises:
        SystemExit: when the pod has fewer GPUs than the stage has ranks.
    """
    gpus = devices()
    if len(gpus) < len(stage):
        raise SystemExit(f"{len(stage)} ranks to stand for and {len(gpus)} GPUs in this pod")
    out = []
    for gpu, rank in zip(gpus, stage, strict=False):
        cmd = [sys.executable, "-m", "everserve_vllm.spare_main", "--engine", a.engine]
        cmd += ["--rank", str(rank), "--node", a.node]
        # the policy document where one is named; else the engine's policy and placement, as this
        # process heard them on its link, handed over on the spare's standard input
        cmd += ["--policy", a.policy] if a.policy else ["--given"]
        cmd += ["--node-devices", ",".join(gpus[: len(stage)])]
        cmd += ["--notice-lead-s", str(a.notice_lead_s), "--leave-within-s", str(a.leave_within_s)]
        cmd += ["--discovery-dir", a.discovery_dir, *a.engine_argv]
        out.append((cmd, gpu))
    return out


CACHE_WAIT_S = 900.0
"""How long a spare node waits for its ranks' compiled sets: past the daemon's own bound (10
minutes, which answers with the reason), so the daemon's answer is the one heard."""


def stage_compiled(link: Any, stage: tuple[int, ...], wait_s: float = CACHE_WAIT_S) -> str | None:
    """Have each rank's compiled set put in place on this node and check it (decision 0043).

    The node's daemon has its store fetch each rank's set from where the rank is served, once
    its serving process published it; this process then recomputes the rank's digest over what
    landed (:func:`everserve_vllm.staging.verify`). None when every rank's set is in place, else
    why not: the spares then load nothing, and never compile in its place.
    """
    from everserve.contract.daemon import CacheUnavailable
    from everserve_vllm import staging

    for rank in stage:
        try:
            ready = link.cache(rank, timeout_s=wait_s)
            staging.verify(ready.key, rank, ready.digest)
        except (CacheUnavailable, staging.StagingRefused, TimeoutError) as e:
            return f"rank {rank}'s compiled set: {e}"
        log.info(
            "rank %d's compiled set is in place: %s/%s", rank, ready.key[:12], ready.digest[:12]
        )
    return None


def stand(
    a: argparse.Namespace,
    stage: tuple[int, ...],
    stop: threading.Event,
    health: Health | None,
    handed: bytes = b"",
) -> int:
    """Stand as ``stage``'s spare: one spare process per rank, until one ends or a stop.

    A spare that takes its rank serves it in its own process; the pod's process supervises them
    and ends with the first that ends, so the pod restarts and asks again. ``handed`` is the
    engine's policy and placement for each spare's standard input (:func:`everserve.contract.
    daemon.hand_over`), where no policy document is named.
    """
    procs = {}
    pipes = []
    for rank, (cmd, gpu) in zip(stage, spare_commands(a, stage), strict=True):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        # the spare's leave pipe: its read end handed over at the start, its write end kept here
        read_fd, write_fd = os.pipe()
        p = subprocess.Popen(
            [*cmd, "--leave-fd", str(read_fd)],
            env=env,
            stdin=subprocess.PIPE if handed else None,
            pass_fds=(read_fd,),
        )
        os.close(read_fd)
        pipes.append(write_fd)
        if handed:
            assert p.stdin is not None
            p.stdin.write(handed)
            p.stdin.close()
        procs[rank] = p
        STANDING.append((p, write_fd))
    log.info("standing for ranks %s: spares %s", list(stage), [p.pid for p in procs.values()])
    if health is not None:
        health.set_ready()  # a spare node gates nothing a router reads: up is its processes up
    try:
        while not stop.wait(SUPERVISE_S):
            if health is not None:
                health.beat()
            ended = [(r, p) for r, p in procs.items() if p.poll() is not None]
            if ended and LEAVING.is_set():
                continue  # leaving: the leave ends the stand once every spare has
            if ended:
                r, p = ended[0]
                log.warning("spare %d for rank %d ended (%d)", p.pid, r, p.returncode)
                return p.returncode or 1
        return 0
    finally:
        STANDING.clear()
        for fd in pipes:
            os.close(fd)
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()


def main(argv: list[str] | None = None) -> int:
    """Configure, then serve the node's ranks or stand as their spare until a signal."""
    from everserve.contract.daemon import EXIT_REFUSED, Refused, UndecidedConnection, hand_over
    from everserve.contract.signals import Member, Role
    from everserve.discovery import local_daemon
    from everserve.policy import Guarantee
    from everserve_vllm.health import Health

    a = parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s node %(message)s")
    record = local_daemon(a.node, a.discovery_dir or None)
    if record is None:
        raise SystemExit(f"no daemon record for node {a.node} in {a.discovery_dir}")
    stop = threading.Event()
    # SIGTERM is the pod's deletion (an eviction, a drain, a spot reclaim): a serving node gives
    # its ranks notice and leaves once they moved (leave_on_sigterm); SIGINT stops at once
    signal.signal(signal.SIGTERM, lambda *_: LEAVING.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    leave_on_sigterm(stop, a.notice_lead_s, a.leave_within_s)
    # live while it loads (the startup probe bounds that), ready once its workers are up and
    # the serve loop turns, live after only while the loop makes progress
    health = Health(a.health_port) if a.health_port is not None else None
    asking = None
    vllm_config = None
    try:
        node_rank = a.node_rank
        policy = None
        if a.slot is not None:
            node_rank = asked_node_rank(a.slot, a.nnodes)
            stage = stage_of(node_rank, a.tp * a.pp, a.nnodes)
            # what this node will step with, resolved before it asks (nothing loads to resolve it),
            # so that the ask carries it and a node resolving otherwise than its engine is refused
            vllm_config = resolve(a, node_rank, Guarantee(a.guarantee), record)
            from everserve_vllm.identity import software_identity

            # before anything loads; the connection is the stage's liveness until this process ends
            asking = UndecidedConnection(
                record.address,
                a.engine,
                stage,
                Member(a.node, 0),
                identity=software_identity(vllm_config),
            )
            # the answer first: a refused process is closed on without the policy, which an
            # accepted one heard before its answer
            try:
                told = asking.assigned()
            except Refused as refused:
                log.error("engine %s refused this node: %s", a.engine, refused.refusal)
                asking.refused(refused)
                asking = None
                return EXIT_REFUSED
            policy = asking.engine_policy() if not a.policy else None
            log.info("slot %d, node rank %d: told %s", a.slot, node_rank, told.role.value)
            if policy is not None and policy.engine(a.engine).guarantee is not Guarantee(
                a.guarantee
            ):
                # the configuration this node resolved is not the engine's: say so, never serve it
                raise SystemExit(
                    f"engine {a.engine} is {policy.engine(a.engine).guarantee.value} in its "
                    f"policy; this node was started --guarantee {a.guarantee}"
                )
            # the link is this process's for its life: its engine's state comes on it
            end_when_down(asking, stop)
            if told.role is Role.SPARE:
                handed = b""
                if not a.policy:
                    # a spare bootstraps from where the engine runs, which comes once every rank
                    # of its world is placed; it is handed what this process heard
                    placement = asking.engine_placement()
                    given = asking.policy_signal
                    assert given is not None  # heard before the answer
                    handed = hand_over(given, placement)
                # each rank's compiled set, staged from where its position is served and checked,
                # before any spare loads: a spare that compiled for itself would be refused
                why = stage_compiled(asking, stage)
                if why is not None:
                    log.error("engine %s: this node cannot stand: %s", a.engine, why)
                    asking.refused(Refused(why))
                    asking = None
                    return EXIT_REFUSED
                code = stand(a, stage, stop, health, handed)
                return EXIT_ENGINE_DOWN if DOWN.is_set() else code
        if policy is None:
            from everserve.policy import load as load_policy

            policy = load_policy(a.policy)
        serve(a, node_rank, policy.engine(a.engine), record, stop, health, vllm_config)
        return EXIT_ENGINE_DOWN if DOWN.is_set() else 0
    finally:
        EXITED.set()
        if asking is not None:
            asking.close()
        if health is not None:
            health.close()


def resolve(a: argparse.Namespace, node_rank: int, guarantee: Any, record: DaemonRecord) -> Any:
    """The configuration node ``node_rank`` runs with.

    The engine's arguments, the adapter's requirements for ``guarantee``, and this node's place
    among the engine's nodes; the process's settings exported for its workers alike.
    """
    from everserve_vllm import settings

    # the workers' output queues bind this node's address, where the head reads them
    if not settings.current().host_ip:
        os.environ["VLLM_HOST_IP"] = record.host  # vLLM's own setting, read by vLLM
    # this process's settings and its workers' alike (vLLM starts them with this environment)
    settings.export(
        graph_safe=True,
        engine=a.engine,
        node=a.node,
        policy=a.policy,
        discovery_dir=a.discovery_dir,
    )
    return engine_args.config(
        a.engine_argv,
        headless=True,
        guarantee=guarantee,
        nnodes=a.nnodes,
        node_rank=node_rank,
        master_addr=a.master_addr,
        master_port=a.master_port,
    )


def serve(
    a: argparse.Namespace,
    node_rank: int,
    engine: Engine,
    record: DaemonRecord,
    stop: threading.Event,
    health: Health | None,
    vllm_config: Any = None,
) -> None:
    """Serve node ``node_rank``'s ranks, of ``engine``: its workers in vLLM's multi-node world.

    ``vllm_config`` is the configuration the node resolved before it asked; a placed node
    resolves it here.
    """
    from everserve_vllm.node import NodeExecutor

    if vllm_config is None:
        vllm_config = resolve(a, node_rank, engine.guarantee, record)
    executor = NodeExecutor(vllm_config, engine_policy=engine)

    def turned() -> None:
        if health is not None:
            health.beat()
            health.set_ready()

    SERVING.append(executor)
    try:
        executor.serve(stop, on_turn=turned)
    finally:
        SERVING.clear()
        executor.shutdown()


if __name__ == "__main__":
    sys.exit(main())
