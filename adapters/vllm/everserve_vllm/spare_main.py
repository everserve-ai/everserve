# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare a daemon launches: ``python -m everserve_vllm.spare_main`` (decision 0024).

The policy's ``spare_launch`` names this command with the engine's arguments, since a warm spare
loads its rank's shard before any join exists. It builds the same engine configuration the head
runs, loads as the rank it stands in for, allocates its KV from the same configuration the
engine's workers got (derived here from its own KV spec and memory, as the engine core derives
it, with the same block count), warms and captures its graphs, pre-forms its sets from the
policy and the nodes' discovery records (:func:`everserve_vllm.worker.bootstrap_from_policy`:
nobody sends a daemon's spare anything before the join), registers with its node's daemon as
a spare, and waits on that link for the executor's join. On a node that only holds spares it
also creates the node's slot region (:func:`prepare_region`). The join's payload is the adapter's
(:mod:`everserve_vllm.join`); the daemons and the coordinator carry it unread.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from everserve_vllm import engine_args, pipeline
from everserve_vllm.identity import software_identity

log = logging.getLogger(__name__)


REGION_WAIT_LOG_S = 30.0
"""How often a spare waiting for its node's slot region says so."""


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line: the position, the policy and discovery, the engine's arguments."""
    ap = argparse.ArgumentParser(prog="everserve_vllm.spare_main")
    ap.add_argument("--engine", default="", help="the engine's name in the policy")
    ap.add_argument(
        "--pool",
        default="",
        help="a pool spare (decision 0031): stand for a position of every engine in --engines "
        "and take the first one a change names; needs an empty --rank and --resident",
    )
    ap.add_argument(
        "--engines",
        default="",
        help="the pool's engines, comma-separated, as its spec lists them ({engines})",
    )
    ap.add_argument(
        "--rank",
        type=lambda x: int(x) if x else None,
        required=True,
        help="the rank this spare stands in for; empty for a spare that stands for any position "
        "of the engine (decision 0030), which must be --resident",
    )
    ap.add_argument("--node", required=True, help="this node's name in the policy")
    given = ap.add_mutually_exclusive_group(required=True)
    given.add_argument("--policy", help="the policy document (a rig, bare metal)")
    given.add_argument(
        "--given",
        action="store_true",
        help="read the engine's policy and placement from standard input, as the node process "
        "that launched this one heard them on its link (protocol 11)",
    )
    ap.add_argument(
        "--node-devices",
        default="",
        help="the GPUs of this node's spares, comma-separated, which the slot variant is decided "
        "over; else the policy's spare devices on the node",
    )
    ap.add_argument(
        "--discovery-dir",
        default="",
        help="where daemons publish their records (bare metal); unused under the pod contract",
    )
    ap.add_argument("--hang-s", type=float, default=10.0)
    ap.add_argument(
        "--leave-fd",
        type=int,
        default=None,
        help="the read end of the leave pipe the standing node's process handed over: a byte on "
        "it, or its end, is the pod going",
    )
    ap.add_argument(
        "--notice-lead-s",
        type=float,
        default=5.0,
        help="once it serves a position, when its pod goes its rank moves this far ahead",
    )
    ap.add_argument(
        "--leave-within-s",
        type=float,
        default=60.0,
        help="once it serves a position, when its pod goes how long it waits for its rank to "
        "move before it stops regardless",
    )
    ap.add_argument(
        "--resident",
        action="store_true",
        help="hold the whole model in pinned host memory and build the rank's shard from it at "
        "the attachment (decisions 0028 and 0030), instead of loading the shard from disk at start",
    )
    # every other argument is vLLM's engine arguments, the head's own (everserve_vllm.
    # engine_args), passed to vLLM's parser untouched
    a, a.engine_argv = ap.parse_known_args(argv)
    ident = engine_args.identity(a.engine_argv)
    a.model, a.tp, a.pp = ident.model, ident.tp, ident.pp
    a.pool_engines = pool_engines(a)
    if not a.engine:
        a.engine = a.pool_engines[0]  # the process's settings until a take names its engine
    return a


def pool_engines(a: argparse.Namespace) -> list[str]:
    """The engines this spare stands for: the pool's, or the one it was launched for.

    Raises:
        SystemExit: a pool without engines, or without an empty --rank and --resident; a spare
            with neither --engine nor --pool.
    """
    if not a.pool:
        if not a.engine:
            raise SystemExit("a spare needs --engine, or --pool with --engines")
        return [a.engine]
    engines = [e for e in a.engines.split(",") if e]
    if not engines:
        raise SystemExit(f"pool {a.pool} names no engines (--engines)")
    if a.rank is not None or not a.resident:
        raise SystemExit("a pool spare stands for any position: an empty --rank and --resident")
    return engines


def wait_for_records(
    discovery: Any,
    nodes: Iterable[str],
    stop: Callable[[], bool] = lambda: False,
    poll_s: float = 0.2,
) -> None:
    """Wait until every node in ``nodes`` has published its daemon record.

    A daemon launches its spares when it starts, and the other nodes' daemons may publish
    later; the rendezvous names each member's host from its node's record, so the spare waits
    for them instead of exiting on a missing record for its supervisor to relaunch it. Until
    ``stop`` says otherwise.
    """
    waiting = sorted(set(nodes))
    said = time.monotonic()
    while waiting and not stop():
        waiting = [n for n in waiting if discovery.daemon(n) is None]
        if not waiting:
            return
        if time.monotonic() - said >= REGION_WAIT_LOG_S:
            log.info("waiting for the daemon records of %s", waiting)
            said = time.monotonic()
        time.sleep(poll_s)


def wait_for_region(path: str, stop: Callable[[], bool] = lambda: False) -> None:
    """Wait until this node's executor has created the slot region at ``path``.

    The node's daemon launches its spares when it starts; the executor on the node, which creates
    the region every process of the engine on the node maps, may start later, and a process maps
    the region at its first collective, which loading can run. Until ``stop`` says otherwise.
    """
    said = time.monotonic()
    while not os.path.exists(path) and not stop():
        if time.monotonic() - said >= REGION_WAIT_LOG_S:
            log.info("waiting for the node's slot region at %s", path)
            said = time.monotonic()
        time.sleep(0.2)


def prepare_region(
    spec: Any,
    node: str,
    rank: int,
    path: str,
    floor_s: float,
    cache: Path | None = None,
    devices: Sequence[str] = (),
) -> tuple[int, Any]:
    """This spare's row in its node's region, and the region made where this node has no executor.

    The first process of the engine on a node creates the node's region. On a node that serves
    the engine that is the executor, and a spare there waits for it and takes the row past the
    stage's columns, the one row an engine node keeps for a spare. On a node the policy gives
    no member of the engine, only spares, the spare creates the region itself: atomically, so
    two spares starting at once make one (:meth:`Region.create`), with the stage's rows and the
    detection floor, and the variant decided over ``devices``, else the policy's spare devices
    on the node, as the executor decides over its GPUs; such a spare's row is its column, so a
    whole group moved to the node has distinct rows. Returns the row and the variant choice this
    process made, or None where the executor's stands.
    """
    from dataclasses import asdict

    from everserve.comm.layout import ParallelLayout
    from everserve.comm.slotkernel import Region, decide_variant
    from everserve.contract.signals import SlotChoice

    tp = spec.tensor_parallel
    if any(m.node == node for m in spec.members):
        wait_for_region(path)
        return tp, None
    layout = ParallelLayout(tp, spec.pipeline_parallel)
    devices = list(devices) or [str(s.device) for s in spec.spares if s.node == node]
    choice = decide_variant(devices, cache=cache)
    region = Region.create(path, choice.variant, replace=False)
    try:
        if region.variant != choice.variant:
            log.warning(
                "the region on %s is %s, decided by an earlier spare; this one decided %s",
                node,
                region.variant,
                choice.variant,
            )
        region.set_members(range(tp), range(tp))
        region.set_bound(range(tp + 1), floor_s)
    finally:
        region.close()
    return layout.column(rank), SlotChoice(**asdict(choice))


def spare_row(spec: Any, node: str, rank: int) -> int:
    """The slot row a spare on ``node`` takes for ``rank``.

    The row past the stage's columns on a node that serves the engine, the rank's column on a
    node of spares only (:func:`prepare_region`).
    """
    from everserve.comm.layout import ParallelLayout

    if any(m.node == node for m in spec.members):
        return spec.tensor_parallel
    return ParallelLayout(spec.tensor_parallel, spec.pipeline_parallel).column(rank)


def check_pair(rank: int, expected: Any, attached: Any) -> None:
    """Refuse a take whose spare compiled other kernels than the position (decisions 0029, 0030).

    ``expected`` is the position's pair as its serving rank last reported it
    (``Take.compile_cache``); None when the position never reported one, and nothing is
    held. A difference fails the change: the acknowledgement names both pairs and nothing is
    committed onto a spare that would run other kernels than the rank it replaces.

    Raises:
        RuntimeError: the pairs differ.
    """
    if expected is None:
        return
    mine = (getattr(attached, "key", None), getattr(attached, "digest", None))
    theirs = (expected.key, expected.digest)
    if mine != theirs:
        raise RuntimeError(
            f"rank {rank}'s compile pair is key {theirs[0]} digest {theirs[1]}; this spare "
            f"attached with key {mine[0]} digest {mine[1]}"
        )


def stand_and_take(
    load: Callable[[int], Any],
    placeholder: Callable[[], Any],
    engine: str,
    spec: Any,
    node: str,
    cfg: Any,
    shard: Any,
    startup_for: Callable[[int], Any],
    slot_choice: Any,
    stood_on: Callable[[Any], None] = lambda link: None,
    identity: Any = None,
) -> tuple[int, Any, Any, Any, dict[str, Any]]:
    """A spare that stands for any position: stand, then take the one a change names.

    Builds no vLLM worker until the take: a worker's model runner fixes its pipeline stage
    when it is built (whether it allocates the intermediate tensors a later stage receives),
    so a worker built as one rank cannot serve another's stage (built as rank 0 and given rank
    1, it fails its profiling run on ``assert intermediate_tensors is not None``). Standing
    needs only the device and vLLM's
    groups pointed at a placeholder rank (``placeholder``, returning the device), from which
    the collective is made; it forms the replacement sets of every eligible position and
    registers as standing for any (:func:`everserve_vllm.worker.stand`).

    A change that names this spare in ``take`` reaches the daemon link's thread, which hands
    the rank here and waits: this thread, vLLM's, takes
    the position's identity and slot row, builds the shard from the host copy, warms it
    (:func:`everserve_vllm.spare.attach`) and binds the rank's runtime and KV; the link's
    thread then restores the rank's rows and acknowledges as that rank. At the take the worker
    is built as the rank (``load(rank)``). Returns the rank, the link the join arrives on, the
    compile pair the join's answer reports (decision 0029, held at the take for a spare like
    this one), the worker, and the take's record for the answer. ``stood_on`` gets the daemon
    link as soon as the spare registers on it, so a stop while standing closes it.
    """
    from everserve_vllm import worker

    asked: queue.Queue[Any] = queue.Queue()
    answer: queue.Queue[Any] = queue.Queue()

    def take(t: Any) -> Any:  # on the link's thread: the work runs on this one
        asked.put(t)
        got = answer.get()
        if isinstance(got, BaseException):
            raise RuntimeError(f"taking rank {t.rank} failed: {type(got).__name__}: {got}") from got
        return got

    device = placeholder()
    standing, stood = worker.stand(device, cfg, take, slot_choice, identity)
    stood_on(standing.link)  # the link a stop closes, from the moment it is registered
    log.info("standing for any of ranks %s: %s", [str(m) for m in stood["stands_for"]], stood)
    taken = asked.get()
    rank, pair, proc, record = _take_position(
        load, engine, spec, node, taken, standing, device, cfg, shard, startup_for, answer
    )
    return rank, standing.link, pair, proc, record


def _take_position(
    load: Callable[[int], Any],
    engine: str,
    spec: Any,
    node: str,
    taken: Any,
    standing: Any,
    device: Any,
    cfg: Any,
    shard: Any,
    startup_for: Callable[[int], Any],
    answer: queue.Queue[Any],
) -> tuple[int, Any, Any, dict[str, Any]]:
    """Take the position ``taken`` names, on this thread; the link's thread waits on ``answer``.

    See :func:`stand_and_take`. Returns the rank, the compile pair, the worker and the take's
    record.
    """
    from everserve_vllm import collective_op, spare, worker

    rank = taken.rank
    t = time.time()
    try:
        t_ids = time.time()
        dropped = collective_op.take(rank, spare_row(spec, node, rank))
        identity_s = time.time() - t_ids
        t_build = time.time()
        proc = load(rank)  # the worker, built as the rank it takes
        build_s = time.time() - t_build
        w = proc.worker.worker
        attachment = spare.attach(proc, engine, rank, startup_for(rank), shard=shard)
        pair = worker.compiled_with(
            w.vllm_config, getattr(getattr(w, "model_runner", None), "model", None), rank
        )
        check_pair(rank, taken.compile_cache, pair)
        runtime = worker.bind_taken(standing, rank, device, cfg, spare.SpareRuntime)
    except BaseException as e:
        answer.put(e)
        raise
    log.info(
        "took rank %d in %.2f s (identity and slot %.2f s, the worker built in %.2f s; %d other "
        "positions' sets set aside): %s",
        rank,
        time.time() - t,
        identity_s,
        build_s,
        dropped,
        attachment,
    )
    answer.put(runtime)
    record = {
        **asdict(attachment),
        "take_s": round(time.time() - t, 3),
        "identity_s": round(identity_s, 3),
        "build_s": round(build_s, 3),
    }
    return rank, pair, proc, record


@dataclass(frozen=True)
class PoolEngine:
    """One engine of a pool spare's (decision 0031).

    Its policy entry, its positions eligible here, its bootstrap and its node's slot choice.
    """

    name: str
    spec: Any
    eligible: tuple[int, ...]
    cfg: Any
    slot_choice: Any


def stand_and_take_pool(
    load: Callable[[int], Any],
    placeholder: Callable[[], Any],
    engines: Sequence[PoolEngine],
    node: str,
    shard: Any,
    startup_for: Callable[[str, int], Any],
    stood_on: Callable[[Any], None] = lambda link: None,
    identity: Any = None,
) -> tuple[str, int, Any, Any, Any, dict[str, Any]]:
    """A pool spare: stand for every engine of the pool at once, take the first position named.

    Decision 0031: one process, one registration per engine, each on a collective of its own
    (:func:`everserve_vllm.collective_op.standing`) holding that engine's cover. The first
    change to name this spare in a take decides the engine: its collective and cover become the
    process's, its region the slot's, and the position is taken as a spare for one engine
    takes it (:func:`stand_and_take`). A later take for another engine is refused, naming the
    engine taken. Once the take's commit is applied, the other engines' registrations close and
    their groups are released (:func:`everserve_vllm.worker.withdraw`), which each engine reads
    as a withdrawal from its pool. Returns the engine, then as :func:`stand_and_take`.
    """
    from everserve.comm.slotkernel import region_path
    from everserve_vllm import collective_op, settings, worker

    asked: queue.Queue[Any] = queue.Queue()
    answer: queue.Queue[Any] = queue.Queue()
    chosen: list[str] = []
    lock = threading.Lock()

    def take_for(engine: str) -> Callable[[Any], Any]:
        def take(t: Any) -> Any:  # on that engine's link's thread
            with lock:
                if chosen and chosen[0] != engine:
                    raise RuntimeError(f"this spare took a position of {chosen[0]} already")
                chosen.append(engine)
            asked.put((engine, t))
            got = answer.get()
            if isinstance(got, BaseException):
                raise RuntimeError(
                    f"taking rank {t.rank} of {engine} failed: {type(got).__name__}: {got}"
                ) from got
            return got

        return take

    device = placeholder()
    standings: dict[str, Any] = {}
    for e in engines:
        c = collective_op.standing(e.eligible)
        standing, stood = worker.stand(
            device, e.cfg, take_for(e.name), e.slot_choice, identity, collective=c
        )
        stood_on(standing.link)
        standings[e.name] = standing
        log.info("standing for %s, any of ranks %s: %s", e.name, list(e.eligible), stood)
    name, taken = asked.get()
    e = next(x for x in engines if x.name == name)
    standing = standings[name]
    # the engine's from now: its collective, its cover, its node's region for the slot
    collective_op.adopt(standing.collective)
    worker.adopt_cover(name)
    settings.update(region=region_path(name), spare_any=tuple(e.eligible))
    rank, pair, proc, record = _take_position(
        load,
        name,
        e.spec,
        node,
        taken,
        standing,
        device,
        e.cfg,
        shard,
        lambda r: startup_for(name, r),
        answer,
    )
    others = [(n, s) for n, s in standings.items() if n != name]

    def withdraw_others() -> None:
        for n, s in others:
            worker.withdraw(n, s.collective, s.link)

    worker.on_next_commit(withdraw_others)  # never before the take's commit: a close is read
    log.info(
        "took rank %d of %s; withdrawing from %s at its commit", rank, name, [n for n, _ in others]
    )
    return name, rank, standing.link, pair, proc, record


def pool_engine(
    policy: Any,
    name: str,
    a: argparse.Namespace,
    launched: Any,
    prepare: Callable[[Any, str], tuple[int, Any]],
) -> PoolEngine:
    """One engine of the pool as this spare stands for it.

    Its positions eligible on this node, its node's region prepared, its bootstrap.

    Raises:
        SystemExit: no position of the engine is eligible for a spare on this node.
    """
    from everserve.comm.slotkernel import region_path
    from everserve.discovery import FileDiscovery
    from everserve_vllm import worker

    spec = policy.engine(name)
    eligible = tuple(worker.eligible_ranks(spec, a.node))
    if not eligible:
        raise SystemExit(f"no rank of {name} is eligible for a spare on {a.node}")
    _, slot_choice = prepare(spec, region_path(name))
    wait_for_records(
        FileDiscovery(a.discovery_dir),
        {m.node for m in spec.members} | set(worker.spare_nodes_of(spec).values()),
    )
    cfg = worker.bootstrap_from_policy(
        policy,
        name,
        a.node,
        a.discovery_dir,
        worker.spare_nodes_of(spec),
        event_dir=launched.event_dir,
        hang_s=a.hang_s,
    )
    return PoolEngine(name, spec, eligible, cfg, slot_choice)


def stale_for(offer: Any, built_from: Any) -> str:
    """Why a spare built from ``built_from`` must refuse ``offer``, or empty when it need not.

    An offer names the digest of the calls its core sent its workers at its start; a spare built
    from other calls (another core's, its engine restarted while it stood) has a cache the
    engine's workers were not built with. An offer that names none is refused as well: every
    offer comes from a core that made its calls, so a missing digest is a defect, never a
    reason to skip the check.
    """
    from everserve_vllm import replay

    held = replay.digest(built_from)
    if offer.startup == held:
        return ""
    named = offer.startup[:12] or "none"
    return (
        f"rank {offer.rank}'s spare was built from other start-up calls than the core offering "
        f"it the position (the core's {named}, the spare's {held[:12]}): it leaves to be built "
        "anew"
    )


def join_position(
    link: Any,
    rank: int,
    proc: Any,
    built_from: Any,
    connect_ip: str,
    pair: Any,
    attached: dict[str, Any],
) -> dict[str, Any]:
    """Take ``rank``'s position at the first offer that names it; the take's record.

    An offer this build cannot read is answered FAILED and the next awaited. An offer a spare
    built from ``built_from`` must refuse (:func:`stale_for`) is answered FAILED, and the spare
    leaves. Otherwise the position is taken (``spare.take_position``) and answered READY, a
    change's offer and a core's bring-up or take alike: a successor's take offers a spare its
    position when the spare never joined the core before it.

    Raises:
        replay.ReplayRefused: for an offer from a core whose calls are not the ones held.
        Exception: whatever the take raised, after its FAILED answer.
    """
    from everserve_vllm import join as joins
    from everserve_vllm import replay, spare, worker

    while True:
        sig = link.joins.get()
        try:
            offer = joins.decode(sig.payload)
        except joins.JoinFormatError as e:
            link.join(
                sig.epoch, joins.encode(joins.Answer(rank, sig.epoch, "", "FAILED", error=str(e)))
            )
            log.error("a join this build cannot read: %s", e)
            continue
        if not isinstance(offer, joins.Offer):
            continue
        stale = stale_for(offer, built_from)
        if stale:
            # the last guard: a spare that missed its engine's new core is never its rank's
            link.join(
                offer.epoch,
                joins.encode(joins.Answer(rank, offer.epoch, offer.tag, "FAILED", error=stale)),
            )
            raise replay.ReplayRefused(stale)
        try:
            took = spare.take_position(
                proc, rank, offer.ring, connect_ip=connect_ip, catch_up=not offer.faulted
            )
        except Exception as e:  # noqa: BLE001 - the executor must learn why, whatever it was
            log.exception("taking rank %d's position failed", rank)
            link.join(
                offer.epoch,
                joins.encode(
                    joins.Answer(
                        rank, offer.epoch, offer.tag, "FAILED", error=f"{type(e).__name__}: {e}"
                    )
                ),
            )
            raise
        link.join(
            offer.epoch,
            joins.encode(
                joins.Answer(
                    rank,
                    offer.epoch,
                    offer.tag,
                    "READY",
                    response=took["handle"],
                    applied=took["applied"],
                    restore=took["restore"],
                    join_s=took["join_s"],
                    compile_cache=pair,
                    attachment=attached,
                )
            ),
        )
        log.info("took rank %d's position at epoch %d in %.3f s", rank, offer.epoch, took["join_s"])
        TAKEN.append((rank, link))
        # the other positions' groups, refused at the take, released now that the take is done
        n = worker.STATE.collective.release_set_aside() if worker.STATE is not None else 0
        if n:
            log.info("released %d groups set aside at the take", n)
        return took


@dataclass
class Held:
    """What a spare holds that its stop must close: its worker process and its daemon link."""

    proc: Any = None
    link: Any = None


def until_stopped(run: Callable[[], None], close: Callable[[], None]) -> int:
    """Run ``run`` until it returns or a stop arrives, then ``close``; the exit status.

    A stop is ``SIGTERM`` (the daemon stopping its spare, a node shutting down) or ``SIGINT``.
    It ends ``run`` where it waits, standing on its link or serving in the worker's loop, by
    raising ``SystemExit`` once in the main thread, and ``close`` runs as the process leaves,
    as vLLM's own worker process does around its loop (``WorkerProc.worker_main``). A spare
    without it dies of a ``SIGTERM`` with its link and its worker torn, never closed, and a
    ``SIGINT`` leaves one that has taken a position running.
    """
    stopping = threading.Event()

    def stop(signum: int, frame: Any) -> None:
        if stopping.is_set():
            return  # once: a second signal must not interrupt the close
        stopping.set()
        log.info("stopping on signal %d", signum)
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        run()
        return 0
    except SystemExit as e:
        if not stopping.is_set():
            raise
        return int(e.code or 0)
    finally:
        stopping.set()
        close()


TAKEN: list[tuple[int, Any]] = []
"""The rank this process took and the link it serves it on, once it took one."""


def moved(link: Any, rank: int, node: str, epoch: int) -> bool:
    """Whether a placement after ``epoch`` places ``rank`` off ``node``: the change took it."""
    p = link.placement
    return (
        p is not None
        and p.epoch > epoch
        and not any(m.rank == rank and m.node == node for m in p.members)
    )


def leave_taken(
    link: Any, rank: int, node: str, lead_s: float, within_s: float, poll_s: float = 0.2
) -> bool:
    """Give ``rank`` notice on ``link`` to leave ``lead_s`` from now; wait for the change.

    The lead is never under the minimum of the policy the link heard. True once a later placement
    puts ``rank`` off ``node``; False past ``within_s``, when the process stops regardless and its
    loss is reported as a worker's departure.
    """
    lead_s = link.notice_lead_s(lead_s)
    placed = link.placement
    epoch = placed.epoch if placed is not None else 0
    link.notice(rank, int((time.time() + lead_s) * 1000))
    log.info("leaving: rank %d noticed to leave in %.1f s", rank, lead_s)
    end = time.monotonic() + within_s
    while not moved(link, rank, node, epoch):
        if time.monotonic() > end:
            log.warning("leaving: rank %d not moved in %.0f s: stopping", rank, within_s)
            return False
        time.sleep(poll_s)
    log.info("leaving: rank %d moved: stopping", rank)
    return True


def leave_on_pipe(fd: int, node: str, lead_s: float, within_s: float) -> None:
    """Once the leave arrives on ``fd``, leave: notice the rank taken and stop once it moved.

    ``fd`` is the leave pipe the standing node's process handed this spare at its start
    (``node_main.tell_leave``): a byte on it, or its end (that process gone), is the pod going.
    A spare that took its rank serves it in this process, outside any node executor: without
    this, its pod's deletion stops it at once and the rank is lost when the process ends, the
    streams waiting on the loss (a 3.7 s gap where a notice gives 0.7 s). The stop is a
    ``SIGTERM`` to this process, which :func:`until_stopped`
    turns into the closing exit.
    """

    def leave() -> None:
        try:
            got = os.read(fd, 1)
        except OSError as e:
            got = b""
            log.warning("leave pipe unreadable (%s): leaving", e)
        log.info("leaving: %s", "told by the node" if got else "the node's process is gone")
        if TAKEN:
            rank, link = TAKEN[0]
            leave_taken(link, rank, node, lead_s, within_s)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=leave, daemon=True, name="EverserveLeave").start()


def close(held: Held) -> None:
    """Close what the spare holds, as a worker process's exit does: the worker, then its link.

    The worker process's own shutdown (its queues, the worker, vLLM's groups) when one was
    built; the slot watch and the mirror's sender before it, the daemon link last, which the
    daemon reads as the process leaving.
    """
    from everserve_vllm import worker

    state = worker.STATE
    if state is not None and state.watch is not None:
        state.watch.stop()
    if held.proc is not None:
        held.proc.shutdown()
    if state is not None:
        state.kv.close()
    link = held.link or (state.link if state is not None else None)
    if link is not None:
        link.close()
    log.info("stopped")


def main(argv: list[str] | None = None) -> int:
    """Load, warm, pre-form, register, then take the position when the join comes; until stopped."""
    a = parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s spare %(message)s")
    held = Held()
    if a.leave_fd is not None:  # a standing node's spare; one a daemon launched has none
        leave_on_pipe(a.leave_fd, a.node, a.notice_lead_s, a.leave_within_s)
    return until_stopped(lambda: run(a, held), lambda: close(held))


def run(a: argparse.Namespace, held: Held) -> None:
    """The spare's life: stand, take the position at its change, serve it."""
    from everserve.comm.slotkernel import region_path
    from everserve.contract.daemon import (
        DaemonConnection,  # noqa: F401 - the link's type, for readers
        ask_replay,
    )
    from everserve.discovery import FileDiscovery, local_daemon
    from everserve.policy import load as load_policy
    from everserve_vllm import replay, settings, spare, worker

    # what the daemon launched this process with (the policy's spareLaunch env), once; the
    # command line names what the environment leaves out
    launched = settings.current()
    settings.update(
        graph_safe=True,
        spare=True,
        layout=launched.layout or (a.tp, a.pp),
        region=region_path(a.engine),  # this node's, by the engine
    )
    from everserve.contract.daemon import handed_over

    if a.given:
        if a.pool:
            raise SystemExit("a pool spare reads its pool's policy document: --policy")
        policy, placement = handed_over(sys.stdin.buffer)
    else:
        policy, placement = load_policy(a.policy), None
    spec = policy.engine(a.engine)
    eligible: list[int] = []
    if a.rank is None:
        if not a.resident:
            raise SystemExit("a spare for any position holds the model in host memory: --resident")
        eligible = worker.eligible_ranks(spec, a.node)
        if not eligible:
            raise SystemExit(f"no rank of {a.engine} is eligible for a spare on {a.node}")
    # the rank the process is built as until a change names one (any position: the first)
    built_as = a.rank if a.rank is not None else eligible[0]
    settings.update(spare_any=tuple(eligible), spare_rank=built_as)
    record = local_daemon(a.node, a.discovery_dir or None)
    if record is None:
        raise SystemExit(f"no daemon record for node {a.node} in {a.discovery_dir}")
    # the queue the executor reads this spare's output from binds this node's address, so a
    # head on another node reaches it; on one node the record's host is loopback
    if not launched.host_ip:
        os.environ["VLLM_HOST_IP"] = record.host  # vLLM's own setting, read by vLLM
    t0 = time.time()
    vllm_config = engine_args.config(a.engine_argv, guarantee=spec.guarantee)
    lock = multiprocessing.get_context("spawn").Lock()
    # the region before loading: on an engine node its executor creates it before its workers
    # start loading, so the wait ends before the engine is warm and the spare still loads beside
    # it; on a spare node this spare creates it
    row, slot_choice = prepare_region(
        spec,
        a.node,
        built_as,
        settings.current().region,
        launched.detect_floor_ms / 1e3,
        cache=Path.home() / ".cache" / "everserve" / "slot-variant.json",
        devices=[d for d in a.node_devices.split(",") if d],
    )
    # a spare node: every replacement sits at its column there
    settings.update(row=row, spare_node=slot_choice is not None)
    shard = None
    if a.resident:
        from everserve_vllm import resident

        # the whole model, pinned, before the device holds anything: the term this mode adds to
        # a change is the shard's copy from here, measured at the attachment
        shard = resident.hold(a.model)
        log.info("resident: %d bytes pinned in %.1f s", shard.nbytes, shard.stage_s)

    read: dict[tuple[str, int], replay.Calls] = {}

    def startup_for(engine: str, rank: int) -> replay.Calls:
        """Rank ``rank``'s start-up calls from ``engine``'s core, asked once (decision 0044).

        Asked through this node's daemon: what the core sent the rank's worker, run on this one.
        A spare standing for any position asks at its take, for the rank it takes.
        """
        if (engine, rank) not in read:

            def ask(bound: float) -> Any:
                return ask_replay(record.address, engine, rank, bound)

            read[(engine, rank)] = replay.ask(ask, rank)
        return read[(engine, rank)]

    if placement is not None:
        # where the engine runs, as the control plane placed it (protocol 11)
        cfg = worker.bootstrap_from_placement(
            policy, a.engine, a.node, placement, event_dir=launched.event_dir, hang_s=a.hang_s
        )
    else:
        wait_for_records(
            FileDiscovery(a.discovery_dir),
            {m.node for m in spec.members} | set(worker.spare_nodes_of(spec).values()),
        )
        cfg = worker.bootstrap_from_policy(
            policy,
            a.engine,
            a.node,
            a.discovery_dir,
            worker.spare_nodes_of(spec),
            event_dir=launched.event_dir,
            hang_s=a.hang_s,
        )
    pair = None
    if a.pool:
        pool = [
            pool_engine(
                policy,
                name,
                a,
                launched,
                prepare=lambda spec, path: prepare_region(
                    spec,
                    a.node,
                    built_as,
                    path,
                    launched.detect_floor_ms / 1e3,
                    cache=Path.home() / ".cache" / "everserve" / "slot-variant.json",
                ),
            )
            for name in a.pool_engines
        ]

        def pool_placeholder() -> Any:
            import torch

            torch.cuda.set_device(0)
            spare.point_groups(vllm_config, built_as, 0)
            return torch.device("cuda", 0)

        def load_pool(rank: int) -> Any:
            return spare.load(vllm_config, rank, lock, local_rank=0, resident=True)

        def stood_pool(link: Any) -> None:
            held.link = link

        engine, rank, link, pair, proc, attached = stand_and_take_pool(
            load_pool,
            pool_placeholder,
            pool,
            a.node,
            shard,
            startup_for,
            stood_on=stood_pool,
            identity=software_identity(vllm_config),
        )
        held.proc, held.link = proc, link
        built_from = startup_for(engine, rank)
        log.info(
            "took rank %d of %s for pool %s, %.1f s after start",
            rank,
            engine,
            a.pool,
            time.time() - t0,
        )
    elif a.rank is None:

        def placeholder() -> Any:
            # vLLM's groups at the placeholder rank, so the collective can be made; no worker
            import torch

            torch.cuda.set_device(0)
            spare.point_groups(vllm_config, built_as, 0)
            return torch.device("cuda", 0)

        def load_as(rank: int) -> Any:
            return spare.load(vllm_config, rank, lock, local_rank=0, resident=True)

        def stood(link: Any) -> None:
            held.link = link

        rank, link, pair, proc, attached = stand_and_take(
            load_as,
            placeholder,
            a.engine,
            spec,
            a.node,
            cfg,
            shard,
            lambda r: startup_for(a.engine, r),
            slot_choice,
            stood_on=stood,
            identity=software_identity(vllm_config),
        )
        held.proc, held.link = proc, link
        built_from = startup_for(a.engine, rank)
        log.info("took rank %d, %.1f s after start", rank, time.time() - t0)
    else:
        rank = a.rank
        # before the load: the core holds the queue for a bounded wait, and the calls are data
        startup = built_from = startup_for(a.engine, rank)
        log.info("rank %d's start-up calls read from the core", rank)
        proc = spare.load(vllm_config, rank, lock, local_rank=0, resident=a.resident)
        held.proc = proc
        attachment = spare.attach(proc, a.engine, rank, startup, shard=shard)
        log.info("attached in %.1f s: %s", time.time() - t0, attachment)
        attached = asdict(attachment)
        started = spare.register(attachment, proc, cfg, slot_choice)  # warm first, by construction
        assert worker.STATE is not None
        log.info("registered as the spare for rank %d: %s", rank, started)
        link = worker.STATE.link
        held.link = link
    assert worker.STATE is not None
    worker.STATE.collective.on_apply = spare._point_stub_at
    pipeline.set_runner(proc.worker.model_runner)  # the packed exchange after the join
    pipeline.hook_worker(proc.worker)
    join_position(link, rank, proc, built_from, record.host, pair, attached)
    spare.serve(proc)


if __name__ == "__main__":
    sys.exit(main())
