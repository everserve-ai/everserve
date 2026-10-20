# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Groups over vLLM's NCCL wrapper for the live group and every pre-formed replacement set.

The :class:`~everserve.comm.collective.Collective` holds them all.

The live group wraps the communicator vLLM already made for its tensor-parallel group. A shadow
group of the policy's pre-formed sets is a fresh ``PyNcclCommunicator`` over a
``StatelessProcessGroup`` (a TCP store, no torch default group). A cover plan's group
(:func:`form_planned`, decision 0035, amended) is one over an id its server makes and the control
plane's links carry: no port. Either is formed beside the live one while the engine is healthy
(harness week five: NCCL allows the overlapping communicators with ``NCCL_P2P_DISABLE=1``; 66 MiB
per set per GPU).

REPLACE never shrinks: a change is a pointer swap to a pre-formed set, so :meth:`PyNcclGroup.shrink`
raises.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from everserve.comm.group import ShrinkableGroup, ShrinkMethod
from everserve.comm.guard import guard_of
from everserve.comm.handle import LocalStandIn
from everserve.comm.layout import position
from everserve.comm.shadow import Member, MemberSet, Rendezvous, set_key

log = logging.getLogger(__name__)


class RefusedComm:
    """Stands in for a communicator whose member set is gone.

    Swapped in for vLLM's NCCL handles when they are refused. Marking a refused handle
    ``disabled`` would make vLLM fall back to its torch process group, which still contains the
    dead rank and faults the CUDA context; this refuses instead, so the interrupted step fails
    fast in Python and the boundary installs the replacement set's communicator.
    """

    disabled = False
    available = True

    def __init__(self, replaced: Any) -> None:
        """Refuse on behalf of ``replaced`` (kept for its device and world size)."""
        self.replaced = replaced
        self.device = getattr(replaced, "device", None)
        self.world_size = getattr(replaced, "world_size", 0)
        self.rank = getattr(replaced, "rank", 0)

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("collective refused: the member set changed (awaiting the boundary)")

    all_reduce = all_gather = all_gatherv = reduce_scatter = reduce_scatterv = _refuse
    broadcast = send = recv = _refuse

    def destroy(self) -> None:
        """Nothing to free: the replaced handle was released. vLLM calls this at shutdown."""


class UncapturedComm:
    """vLLM's handle to a serving communicator, refusing any collective inside a graph's capture.

    ``ncclCommAbort`` waits for every CUDA graph captured on a communicator to be freed, so a
    graph holding a serving communicator would hang the abort a loss runs (probe 5's first run
    hung that way). No graph is meant to: the graph-safe op runs every NCCL call between graph
    pieces (decision 0029). This makes that hold by construction. It is put on vLLM's handles
    before vLLM captures (:func:`everserve_vllm.worker.guard_engine_groups`) and on every handle a
    change points vLLM at, so a capture that reaches NCCL fails the engine's start, loudly,
    instead of hanging an abort later. Everything else is the wrapped communicator's. It costs one
    ``cudaStreamIsCapturing`` per call (probe 5: within 3 % of the unguarded call, on H100).
    """

    def __init__(self, wrapped: Any) -> None:
        """Guard ``wrapped`` (a ``PyNcclCommunicator``)."""
        object.__setattr__(self, "wrapped", wrapped)

    def __getattr__(self, name: str) -> Any:
        """Every attribute not a collective is the wrapped communicator's."""
        return getattr(self.wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Every write, vLLM's ``disabled`` among them, goes to the wrapped communicator."""
        setattr(self.wrapped, name, value)

    def _guarded(name: str) -> Any:  # noqa: N805 - builds the class's methods
        def call(self: UncapturedComm, *args: Any, **kwargs: Any) -> Any:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"a {name} on a serving communicator inside a CUDA graph's capture: its "
                    "abort would wait on the graph (decisions 0025, 0029)"
                )
            return getattr(self.wrapped, name)(*args, **kwargs)

        call.__name__ = name
        return call

    all_reduce = _guarded("all_reduce")
    all_gather = _guarded("all_gather")
    all_gatherv = _guarded("all_gatherv")
    reduce_scatter = _guarded("reduce_scatter")
    reduce_scatterv = _guarded("reduce_scatterv")
    broadcast = _guarded("broadcast")
    send = _guarded("send")
    recv = _guarded("recv")
    del _guarded


def unguarded(comm: Any) -> Any:
    """The communicator itself, under an :class:`UncapturedComm` or not."""
    return comm.wrapped if isinstance(comm, UncapturedComm) else comm


RELEASE_BOUND_S = 10.0
"""How long a release may run from its start before it counts as leaked (decision 0025)."""


class Releases:
    """The releases of refused groups, each on its own thread from the refusal.

    A release is ``ncclCommAbort`` of one communicator, or torch's abort of its process groups.
    Nothing waits for one: not the refusal, not the acknowledgement, not the boundary, which
    swaps the groups and reaps (:meth:`reap`) what has returned by then. A release still
    running past its bound from its start is counted as leaked and never waited out (decision
    0025). Waited for, the fabric's abort costs the boundary 0.5–0.8 s of a 1.08 s pause, and
    nothing on the swap depends on it, since the refusal keeps every call off the refused group
    and the guard keeps the free off a call.
    """

    def __init__(self) -> None:
        """Nothing running, nothing done, nothing leaked."""
        self.on_leak: Callable[[str, float, int], None] | None = None
        """Called once per release counted leaked, with its name, its run in seconds and the
        count so far: the process reports it (decision 0025, amended), and decides nothing."""
        self._running: list[tuple[str, float, threading.Thread, list[float]]] = []
        self.done: list[str] = []
        """Every release that returned, with its time, or that was left behind past its bound."""
        self.leaked = 0
        """Releases left running past the bound; what they hold is never freed."""
        self._times: dict[str, list[float]] = {}
        """Each release's wall-clock start and, once returned, end (``time.time()``)."""

    def start(self, name: str, run: Callable[[], None] | None) -> None:
        """Run ``run`` now on its own thread; None is nothing to release."""
        if run is None:
            return
        took: list[float] = []
        t0 = time.perf_counter()
        stamps = self._times.setdefault(name, [])
        stamps.append(time.time())

        def timed() -> None:
            run()
            took.append(time.perf_counter() - t0)
            stamps.append(time.time())

        thread = threading.Thread(target=timed, daemon=True, name=f"everserve-release-{name}")
        thread.start()
        self._running.append((name, t0, thread, took))

    def reap(self, bound_s: float | None = None) -> list[str]:
        """Record every release that has returned, and count one past its bound as leaked.

        Never waits: a release within its bound stays running and is reaped at a later call.
        The bound is :data:`RELEASE_BOUND_S` when none is given. Returns this call's reports.
        """
        bound = RELEASE_BOUND_S if bound_s is None else bound_s
        now = time.perf_counter()
        reports, still = [], []
        for entry in self._running:
            name, t0, _thread, took = entry
            if took:
                reports.append(f"{name} {took[0]:.3f}s")
            elif now - t0 > bound:
                self.leaked += 1
                reports.append(f"{name} still running")
                log.warning("release %s left behind past its bound (%d leaked)", name, self.leaked)
                if self.on_leak is not None:
                    try:
                        self.on_leak(name, now - t0, self.leaked)
                    except Exception:  # noqa: BLE001 - a report never stops the reaping
                        log.exception("release %s's leak could not be reported", name)
            else:
                still.append(entry)
        self._running = still
        self.done.extend(reports)
        return reports

    def times(self) -> dict[str, list[float]]:
        """Each release's wall-clock start and end (the end missing while it runs), by name."""
        return {name: list(stamps) for name, stamps in self._times.items()}

    def running(self) -> list[str]:
        """The releases still within their bound and not returned, by name."""
        return [name for name, _t0, _thread, took in self._running if not took]


def abort_of(comm: Any) -> Callable[[], None]:
    """``ncclCommAbort`` of ``comm`` (a ``PyNcclCommunicator``), for a release thread.

    Runs under the communicator's exclusive guard (:func:`everserve.comm.guard.guard_of`): a
    call still inside the communicator's host code holds the shared side, and the free waits
    for it to leave rather than running under it.
    """

    def run() -> None:
        # a new thread starts on device 0: abort on the communicator's own device, as
        # vLLM's PyNcclCommunicator.destroy does (pynccl.py:257)
        device = getattr(comm, "device", None)
        with (
            guard_of(comm).exclusive(),
            torch.cuda.device(device) if device is not None else contextlib.nullcontext(),
        ):
            comm.nccl.ncclCommAbort(comm.comm)

    return run


class PyNcclGroup(ShrinkableGroup):
    """A :class:`ShrinkableGroup` whose ``group`` is a vLLM ``PyNcclCommunicator``."""

    refused: bool = False
    """Set by :meth:`refuse`: the member set changed, and the communicator is never used again."""

    @classmethod
    def over(
        cls, comm: Any, ranks: Sequence[int], method: ShrinkMethod = ShrinkMethod.INIT
    ) -> PyNcclGroup:
        """Wrap ``comm`` (a ``PyNcclCommunicator``) spanning original ranks ``ranks``."""
        return cls(group=comm, ranks=list(ranks), backend="nccl", store=None, method=method)

    def all_reduce(self, t: torch.Tensor) -> None:
        """Sum ``t`` in place across the group (NCCL allows one buffer for send and receive).

        Raises:
            RuntimeError: after :meth:`refuse`.
        """
        if self.refused:
            raise RuntimeError("collective refused: the member set changed")
        self.group.all_reduce(t, out_tensor=t)

    def refuse(self) -> Callable[[], None] | None:
        """Refuse collectives from now on; returns the communicator's abort, for a release thread.

        None when the group was refused already or has no communicator. Callers that reach the
        communicator through vLLM must first be pointed at a :class:`RefusedComm`
        (:func:`everserve_vllm.collective_op` does this); the refused handle is never used again.

        ``ncclCommAbort`` can block until CUDA graphs that captured the communicator are freed.
        No graph captures this one (the graph-safe op keeps NCCL out of every graph), but the
        abort runs on a release thread so a surprise cannot hang a step beyond its bound.
        """
        if self.refused:
            return None
        self.refused = True
        comm = self.group
        return abort_of(comm) if getattr(comm, "comm", None) is not None else None

    def shrink(self, dead: int, me: int, epoch: int) -> ShrinkableGroup:
        """REPLACE never shrinks; the runtime moves to a pre-formed set instead.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError("a pynccl group does not shrink; REPLACE swaps to a pre-formed set")


def form_shadow_groups(
    me: Member,
    plan: Sequence[Rendezvous],
    device: torch.device,
    timeout_s: float = 300.0,
) -> tuple[dict[MemberSet, PyNcclGroup], dict[str, float]]:
    """Form the groups of ``plan`` (:func:`everserve.comm.shadow.rendezvous_plan`) for ``me``.

    Each group rendezvouses on its first member's node at its own port; every process of the
    set meets there, and the group's ranks are its members' positions in the engine.

    Returns:
        The groups by the full member set they are pre-formed for, and the seconds each took by
        group name.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    out: dict[MemberSet, PyNcclGroup] = {}
    took: dict[str, float] = {}
    for r in plan:
        t0 = time.time()
        pg = StatelessProcessGroup.create(
            host=r.host,
            port=r.port,
            rank=r.group.index(me),
            world_size=len(r.group),
            store_timeout=int(timeout_s),
        )
        comm = PyNcclCommunicator(pg, device=device)
        ranks = [position(m) for m in r.group]
        out[tuple(r.members)] = PyNcclGroup.over(comm, ranks, ShrinkMethod.SHADOW)
        took[set_key(r.group)] = round(time.time() - t0, 3)
        t1 = time.time()
        warm_group(comm, r.group.index(me), len(r.group), device)
        torch.cuda.synchronize(device)
        took[f"{set_key(r.group)} warm"] = round(time.time() - t1, 3)
    torch.cuda.synchronize(device)
    log.info("shadow groups formed for %s: %s", me, took)
    return out, took


def form_planned(
    group: Any,
    me: Member,
    device: torch.device,
    stream: torch.cuda.Stream,
    link: Any,
    engine: str,
    generation: int,
    timeout_s: float = 300.0,
) -> PyNcclGroup | LocalStandIn:
    """Form one group of a cover plan for ``me`` beside serving (decision 0035, amended).

    ``group`` is an :class:`everserve.comm.cover.PlannedGroup`: its ``spans`` are the
    communicator's rank order, and its ``server``, rank 0 of the group, makes the group's NCCL id.
    The group meets on the control plane's links, not on a port: the server reports the id on
    ``link`` (``report_group_id``) and the coordinator carries it to the other members' daemons,
    whose processes wait for it on their own link (``link.group_ids``), at ``generation`` or
    later. Every member then joins with it (vLLM's ``PyNcclCommunicator.from_unique_id_bytes``,
    for peers that share the id out of band). The joins' warm-up (vLLM's, on the current stream)
    and the group's run on ``stream``, the cover's own, which is then synchronised alone: the
    serving stream and the device are never waited on. A group spanning only ``me`` is a local
    stand-in.

    Raises:
        ValueError: for a group without ``me``, or whose server is not its rank 0.
        TimeoutError: a member that is not its server heard no id within ``timeout_s``.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    spans = list(group.spans)
    if me not in spans:
        raise ValueError(f"{set_key(tuple(spans))} does not span {me}")
    if len(spans) == 1:
        return LocalStandIn()
    rank, uid = group_id_for(group, me, link, engine, generation, timeout_s, mint_group_id)
    with torch.cuda.stream(stream):
        comm = PyNcclCommunicator.from_unique_id_bytes(uid, rank, len(spans), device)
        warm_group(comm, rank, len(spans), device)
    stream.synchronize()
    return PyNcclGroup.over(comm, [position(m) for m in spans], ShrinkMethod.SHADOW)


def group_id_for(
    group: Any,
    me: Member,
    link: Any,
    engine: str,
    generation: int,
    timeout_s: float,
    mint: Callable[[], bytes],
) -> tuple[int, bytes]:
    """``me``'s rank in ``group`` and the group's id, once every member is ready (0035).

    Every member, its server included, first reports itself ready at ``generation``
    (``GroupReady``). The server then makes the id and reports it. The coordinator relays the id
    only once every member is ready, and its arrival is each member's go: the server too joins on
    its own id coming back, never on making it. So no member enters NCCL's blocking
    initialisation before every other has shown it is alive. A duplicate of an id already heard
    changes nothing: the id is read once, by this formation.

    Raises:
        ValueError: for a group whose server is not its rank 0.
        TimeoutError: no id at ``generation`` or later reached this member within ``timeout_s``:
            a member never became ready, or the server never made one.
    """
    from everserve.comm.cover import set_member
    from everserve.contract.signals import GroupId, GroupReady

    spans = list(group.spans)
    if group.server != spans[0]:
        raise ValueError(
            f"{set_key(tuple(spans))}: its server {group.server} is not its rank 0 ({spans[0]})"
        )
    rank = spans.index(me)
    wire = tuple(set_member(m) for m in spans)
    link.report(GroupReady(engine, generation, wire, group.scope, set_member(me)))
    if rank == 0:
        link.report_group_id(GroupId(engine, generation, group.scope, wire, mint()))
    return rank, link.group_ids.wait(group.scope, wire, generation, timeout_s).id


def mint_group_id() -> bytes:
    """A new NCCL unique id, as the bytes ``from_unique_id_bytes`` takes."""
    import ctypes

    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

    uid = NCCLLibrary().ncclGetUniqueId()
    return ctypes.string_at(ctypes.addressof(uid), ctypes.sizeof(uid))


def warm_group(comm: Any, rank: int, world: int, device: torch.device) -> None:
    """Connect a formed group's paths with one tiny operation of every kind the engine issues.

    Its first real step after a swap then costs what any step costs.

    NCCL connects a communicator's transports lazily, per kind of operation, at the first one
    (the first step on a new ring takes 0.07–0.63 s against 8 ms steady). A pre-formed group is
    therefore formed and warmed: an all-reduce and a
    broadcast (the stage groups' and the token broadcast's paths), and on a pair a send and a
    receive each way (the pipeline exchange's path). All on the current stream, a few bytes.
    """
    if world < 2:
        return
    x = torch.ones(8, dtype=torch.float32, device=device)
    comm.all_reduce(x)
    comm.broadcast(x, 0)
    if world == 2:
        other = 1 - rank
        if rank == 0:
            comm.send(x, other)
            comm.recv(x, other)
        else:
            comm.recv(x, other)
            comm.send(x, other)
