# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A worker's connection to the node daemon.

The daemon is the Rust process in ``controller/``. A worker connects, registers with its engine
and rank, then serves the daemon's signals for the life of the process: a :class:`Change` moves
the runtime (:meth:`Participant.on_change` with the chosen outcome) and is acknowledged once the
collective has moved; a :class:`Notice` is recorded. The worker may also report a peer it saw
die. Runs on a daemon thread so a signal lands even while the main thread is blocked in a
collective.

:class:`DaemonClient` is the other kind of connection: the executor or a drill driver. It reports
a lost rank, delivers a notice, and hears every committed membership with the time it arrived.
"""

from __future__ import annotations

import faulthandler
import logging
import queue
import socket
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, Protocol

from everserve.contract.signals import (
    Abandoned,
    CacheReady,
    CacheRefused,
    CacheWanted,
    Change,
    CompileCache,
    CoverPlan,
    DumpStacks,
    EnginePlacement,
    EnginePolicy,
    Goodbye,
    GroupFailed,
    GroupFormed,
    GroupId,
    GroupReady,
    GroupReleased,
    InstanceState,
    Join,
    Member,
    MembershipCommitted,
    Notice,
    RankLost,
    Reason,
    Register,
    ReleaseLeaked,
    ReplayAsk,
    ReplayOffer,
    Role,
    RoleAssigned,
    SetMember,
    Signal,
    SlotChoice,
    SoftwareIdentity,
    StopCause,
    UnknownSignal,
    from_message,
    to_message,
)
from everserve.discovery import HOST_IP_ENV
from everserve.outcome import Outcome
from everserve.policy import Policy
from everserve.policy import from_bytes as policy_from_bytes
from everserve.proto import signals_pb2 as pb
from everserve.wire import frame, read_frame, read_stream_frame, write_frame

log = logging.getLogger(__name__)


class Participant(Protocol):
    """What a process's connection applies the daemon's signals to: the engine-side runtime.

    The engine's runtime (``everserve.comm.runtime.Runtime`` and the adapter's subclasses) is one;
    the contract names everything it calls. A runtime that does not stand for any position, form
    cover or get displaced answers those hooks by saying so (the base runtime's defaults);
    nothing is probed for.
    """

    orig_rank: int
    took_epoch: int | None
    """The epoch at which a spare took its position; None while it stands, and for a worker."""

    def identity(self) -> tuple[int, int]:
        """This process's rank and the epoch it last applied."""
        ...

    def on_change(
        self,
        lost: Sequence[int],
        sent_at: float | None = None,
        outcome: Outcome = ...,
        epoch: int | None = None,
        departing: Sequence[int] = (),
    ) -> None:
        """Apply a change that lost ``lost``, at the change's ``epoch``; returns once moved.

        ``departing`` are the lost ranks leaving on notice: a change of only those refuses nothing.
        """
        ...

    def on_notice(self, rank: int, deadline_ms: int) -> None:
        """Record that ``rank`` leaves at ``deadline_ms``."""
        ...

    def on_abandoned(self, epoch: int, detail: str) -> None:
        """The change at ``epoch`` was abandoned after its delivery (decision 0032)."""
        ...

    def on_take(self, takes: Sequence[Any]) -> None:
        """A spare standing for any position takes one of ``takes``, before the change."""
        ...

    def on_cover_plan(self, plan: CoverPlan) -> None:
        """Each cover plan the daemon forwards (decision 0035), applied at once."""
        ...

    def on_displaced(self, epoch: int, why: str) -> None:
        """Displaced at ``epoch``: ``why`` says whether no longer placed, or placed unstaged."""
        ...


def _connect(path: str, timeout_s: float) -> socket.socket:
    """A Unix socket connected to ``path``, retried while the daemon starts."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            sock.connect(path)
            return sock
        except OSError:
            if time.monotonic() >= deadline:
                sock.close()
                raise
            time.sleep(0.05)


def _next_signal(sock: socket.socket) -> Signal:
    """The next signal from the daemon this build knows, wherever it falls among the frames.

    A frame of a kind added after this build (a newer daemon's, decision 0040), or of one its
    schema names but its codec does not decode, is passed over rather than ending the link.

    Raises:
        EOFError, OSError: when the connection ends.
    """
    while True:
        msg = pb.Signal.FromString(read_frame(sock))
        if msg.WhichOneof("kind") is None:
            log.debug("passing over a signal of a kind this build does not know")
            continue
        try:
            return from_message(msg)
        except UnknownSignal as e:
            # its schema knows the kind, its codec does not: never the end of the link
            log.debug("passing over %s", e)


def dump_stacks(asked: DumpStacks) -> None:
    """Write every thread's stack to this process's log (its standard error), as the daemon asked.

    The daemon asks on the process's own link (protocol 11), never by a signal to its pid.
    """
    log.warning("every thread's stack, as the daemon asked: %s", asked.reason or "no reason given")
    sys.stderr.flush()
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


class HearsEngine:
    """What a link last heard of its engine from its daemon (protocol 11): the coordinator's.

    Every link takes :class:`EnginePolicy` and :class:`EnginePlacement` wherever they fall among
    its frames and keeps the last of each, so a process waits for its first with
    :meth:`engine_policy` and :meth:`engine_placement` and reads the current one after.
    """

    closed: threading.Event
    """Set when the link ends."""
    _sock: socket.socket

    def __init__(self) -> None:
        """Nothing heard yet."""
        self.closed = threading.Event()
        self._heard_lock = threading.Lock()
        self._policy: Policy | None = None
        self._policy_signal: EnginePolicy | None = None
        self._policy_heard = threading.Event()
        self._placement: EnginePlacement | None = None
        self._placement_heard = threading.Event()
        self._state: InstanceState | None = None
        self._state_heard = threading.Event()

    def _serve_signals(self, what: str, take: Callable[[Any], bool]) -> None:
        """Read the daemon's signals until the link ends; ``closed`` is set when it has.

        What every link hears alike (a stacks request, the engine's policy, placement and
        state) is taken here; ``take`` takes the link's own kinds and says whether it did.
        """
        try:
            while True:
                sig = _next_signal(self._sock)
                if self._heard_common(sig) or take(sig):
                    continue
                log.warning("%s ignoring signal %s", what, type(sig).__name__)
        except (EOFError, OSError) as e:
            log.info("%s closed: %s", what, e)
        finally:
            self.closed.set()

    def _heard_common(self, sig: Any) -> bool:
        """Take a signal every link hears alike; False for any other."""
        if isinstance(sig, DumpStacks):
            dump_stacks(sig)
        elif isinstance(sig, EnginePolicy):
            self._heard_policy(sig)
        elif isinstance(sig, EnginePlacement):
            self._heard_placement(sig)
        elif isinstance(sig, InstanceState):
            self._heard_state(sig)
        else:
            return False
        return True

    def _heard_policy(self, sig: EnginePolicy) -> None:
        try:
            policy = policy_from_bytes(sig.policy)
        except ValueError:
            log.exception(
                "engine %s's policy from the daemon refused; the last is kept", sig.engine
            )
            return
        with self._heard_lock:
            before = self._policy
            self._policy = policy
            self._policy_signal = sig
        if before is None or before.version != policy.version:
            log.info("engine %s's policy %s, from the daemon", sig.engine, policy.version)
        self._policy_heard.set()

    def _heard_placement(self, sig: EnginePlacement) -> None:
        with self._heard_lock:
            if self._placement is not None and sig.epoch < self._placement.epoch:
                log.warning(
                    "engine %s's placement at epoch %d after %d: kept the later",
                    sig.engine,
                    sig.epoch,
                    self._placement.epoch,
                )
                return
            self._placement = sig
        self._placement_heard.set()

    def _heard_state(self, sig: InstanceState) -> None:
        with self._heard_lock:
            before = self._state
            self._state = sig
        if before is None or before.phase is not sig.phase:
            log.info(
                "engine %s is %s (%d of %d positions held, epoch %d)%s",
                sig.engine,
                sig.phase.value,
                sig.positions_held,
                sig.world,
                sig.epoch,
                f": {sig.reason}" if sig.reason else "",
            )
        self._state_heard.set()

    @property
    def state(self) -> InstanceState | None:
        """The instance's last state heard, or None before the first (decision 0041)."""
        with self._heard_lock:
            return self._state

    @property
    def policy(self) -> Policy | None:
        """The last policy heard, or None before the first."""
        with self._heard_lock:
            return self._policy

    @property
    def policy_signal(self) -> EnginePolicy | None:
        """The last policy heard, as it came: for a process to hand to those it launches."""
        with self._heard_lock:
            return self._policy_signal

    @property
    def placement(self) -> EnginePlacement | None:
        """The latest placement heard, or None before the first."""
        with self._heard_lock:
            return self._placement

    def notice_lead_s(self, wanted_s: float) -> float:
        """How far ahead this process's notice goes: ``wanted_s``, or the policy's minimum.

        Never under the minimum of the policy heard (:meth:`everserve.policy.Policy.
        notice_lead_s`); ``wanted_s`` before one is heard.
        """
        policy = self.policy
        return policy.notice_lead_s(wanted_s) if policy is not None else wanted_s

    def engine_policy(self, timeout_s: float | None = None) -> Policy:
        """The last policy heard, waiting for the first.

        Raises:
            ConnectionError: if the link ends before one.
            TimeoutError: if none comes within ``timeout_s``.
        """
        self._wait(self._policy_heard, "the policy", timeout_s)
        policy = self.policy
        assert policy is not None  # set before the event
        return policy

    def engine_placement(self, timeout_s: float | None = None) -> EnginePlacement:
        """The latest placement heard, waiting for the first: every rank of the world placed.

        Raises:
            ConnectionError: if the link ends before one.
            TimeoutError: if none comes within ``timeout_s``.
        """
        self._wait(self._placement_heard, "the placement", timeout_s)
        placement = self.placement
        assert placement is not None  # set before the event
        return placement

    def _wait(self, heard: threading.Event, what: str, timeout_s: float | None) -> None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while not heard.wait(0.2):
            if self.closed.is_set():
                raise ConnectionError(f"the daemon closed the link before it sent {what}")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"no {what.split(' ', 1)[1]} from the daemon in {timeout_s} s")


def hand_over(policy: EnginePolicy, placement: EnginePlacement) -> bytes:
    """The engine's policy and placement as a launching process heard them, for its children.

    Two frames of the link's own encoding; a child reads them with :func:`handed_over`.
    """
    return b"".join(frame(to_message(sig).SerializeToString()) for sig in (policy, placement))


def handed_over(stream: BinaryIO) -> tuple[Policy, EnginePlacement]:
    """The engine's policy and placement a launching process handed over (:func:`hand_over`).

    Raises:
        ValueError: for anything but a policy and then a placement.
    """
    policy = from_message(pb.Signal.FromString(read_stream_frame(stream)))
    placement = from_message(pb.Signal.FromString(read_stream_frame(stream)))
    if not isinstance(policy, EnginePolicy) or not isinstance(placement, EnginePlacement):
        raise ValueError(
            f"handed {type(policy).__name__} and {type(placement).__name__}, "
            "not the engine's policy and placement"
        )
    return policy_from_bytes(policy.policy), placement


def ask_replay(
    path: str, engine: str, rank: int, timeout_s: float, connect_timeout_s: float = 30.0
) -> ReplayOffer:
    """Ask the engine's core what it sent ``rank`` at its start; its offer (decision 0044).

    Through this node's daemon at ``path``, on a connection of its own whose only ask is this
    one, registered as nothing: a spare needs the calls to warm, and registers only once warm.
    The daemon names the connection as the asker, the coordinator hands the ask to the core, and
    the core's offer comes back here.

    Raises:
        ConnectionError: if the daemon closes the connection before an offer.
        TimeoutError: if none comes within ``timeout_s``: no core holds the head, or the ask or
            its offer was dropped; asking again is safe.
    """
    sock = _connect(path, connect_timeout_s)
    try:
        write_frame(sock, to_message(ReplayAsk(engine=engine, rank=rank)).SerializeToString())
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no replay offer for rank {rank} of {engine} in {timeout_s} s")
            sock.settimeout(left)
            try:
                sig = _next_signal(sock)
            except TimeoutError:
                # the socket's bound is what was left: a frame read in part is never resumed
                raise TimeoutError(
                    f"no replay offer for rank {rank} of {engine} in {timeout_s} s"
                ) from None
            except EOFError:
                raise ConnectionError(
                    f"the daemon closed the connection before the core's replay for rank {rank}"
                ) from None
            if isinstance(sig, ReplayOffer) and sig.engine == engine and sig.rank == rank:
                return sig
            log.info("asking for rank %d's replay: passing over %s", rank, type(sig).__name__)
    finally:
        _shut(sock)


def own_host() -> str:
    """The address this process's node is reached at, as the pod contract states it; else empty."""
    import os

    return os.environ.get(HOST_IP_ENV, "")


class GroupIds:
    """The group ids a process's link has heard (decision 0035, amended): the newest per group.

    The daemon hands every process of the engine on its node the id of every group with a member
    there, at registration and as each is made; a process forms only the groups that span it, so
    the others are kept and never read. A formation waits here for its group's id, bounded.
    """

    def __init__(self) -> None:
        """No id heard yet."""
        self._ids: dict[tuple[str, tuple[SetMember, ...]], GroupId] = {}
        self._cond = threading.Condition()

    def put(self, gid: GroupId) -> None:
        """Keep ``gid`` unless an id at a newer generation is held for its group."""
        key = (gid.scope, tuple(gid.spans))
        with self._cond:
            held = self._ids.get(key)
            if held is None or gid.generation >= held.generation:
                self._ids[key] = gid
                self._cond.notify_all()

    def wait(
        self, scope: str, spans: tuple[SetMember, ...], generation: int, timeout_s: float
    ) -> GroupId:
        """The id of the group over ``spans`` in ``scope`` made at ``generation`` or later.

        Raises:
            TimeoutError: if none comes within ``timeout_s``: its server never made one.
        """
        key = (scope, tuple(spans))
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                gid = self._ids.get(key)
                if gid is not None and gid.generation >= generation:
                    return gid
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(
                        f"no id for the {scope} group over {list(spans)} at generation "
                        f"{generation} or later within {timeout_s:.0f} s"
                    )
                self._cond.wait(left)


def _shut(sock: socket.socket) -> None:
    """Shut down, then close: a close alone does not wake a thread blocked in ``recv`` on Linux."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # already closed by the peer
    sock.close()


class DaemonConnection(HearsEngine):
    """One worker's link to the node daemon over a Unix domain socket."""

    def __init__(
        self,
        path: str,
        engine: str,
        member: Member,
        runtime: Participant,
        connect_timeout_s: float = 30.0,
        role: Role = Role.WORKER,
        slot: SlotChoice | None = None,
        compile_cache: CompileCache | None = None,
        any_rank: bool = False,
        identity: SoftwareIdentity | None = None,
    ) -> None:
        """Connect to the daemon at ``path``, register, and start serving its signals.

        Args:
            path: the daemon's Unix socket.
            engine: the engine this worker belongs to.
            member: this worker's node and device.
            runtime: the runtime the daemon's changes are applied to.
            connect_timeout_s: how long to retry the connection while the daemon starts.
            role: ``SPARE`` for a pre-loaded spare standing in for ``runtime.orig_rank``.
            slot: the slot-collective choice this process made for its node's region, when it
                created the region (a spare on a node that only holds spares); the readiness
                view can then say which node's decision it shows.
            compile_cache: the compiled-kernel cache this process loaded and warmed with
                (decision 0029), which the coordinator holds the engine's members to; None
                from a process that compiled nothing.
            any_rank: a spare that stands for any position of the engine (decision 0030): its
                rank is unused until a change names it in ``take``, which reaches
                ``runtime.on_take`` before the change itself.
            identity: what the process runs (decision 0034); a destination without one is
                refused.
        """
        super().__init__()
        self._runtime = runtime
        self._engine = engine
        self._role = role
        self._member = member
        self._staged: set[int] = set()
        """The epochs whose change this process staged: a commit placing it at another is one it
        missed."""
        self.joins: queue.Queue[Join] = queue.Queue()
        """Join payloads from the engine's executor for this process's position (decision 0024)."""
        self.group_ids = GroupIds()
        """The ids of the groups with a member on this node (decision 0035, amended)."""
        self._sock = _connect(path, connect_timeout_s)
        self._lock = threading.Lock()
        self._send(
            to_message(
                Register(
                    engine=engine,
                    rank=runtime.orig_rank,
                    member=member,
                    role=role,
                    slot=slot,
                    compile_cache=compile_cache,
                    any_rank=any_rank,
                    identity=identity,
                    host=own_host(),
                )
            )
        )
        threading.Thread(target=self._serve, daemon=True, name="everserve-daemon-link").start()

    def _send(self, msg: pb.Signal) -> None:
        with self._lock:
            write_frame(self._sock, msg.SerializeToString())

    def report_lost(self, ranks: Sequence[int], detail: str = "", epoch: int = 0) -> None:
        """Tell the daemon this worker saw ``ranks`` die at membership ``epoch``.

        A failed collective, for instance, or the slot kernel's late push from a rank whose process
        is exiting (decision 0021). It may go at any time, a change in flight included: every frame
        to the daemon is a ``Signal``, the Ack too.
        """
        self._send(to_message(RankLost(ranks=tuple(ranks), detail=detail, epoch=epoch)))

    def notice(self, rank: int, deadline_ms: int) -> None:
        """This worker's ``rank`` leaves at ``deadline_ms`` (Unix milliseconds): its pod is going.

        A spare that took a position serves it on this link, with no node executor's client
        beside it: its notice goes here, and the daemon sends it up as any notice.
        """
        self._send(to_message(Notice(rank=rank, deadline_ms=deadline_ms, reason=Reason.NOTICE)))

    @property
    def engine(self) -> str:
        """The engine this link registered for."""
        return self._engine

    def report(self, report: GroupFormed | GroupFailed | GroupReleased | GroupReady) -> None:
        """Tell the daemon what this process did with a group of its cover plan (decision 0035)."""
        self._send(to_message(report))

    def report_leak(self, release: str, run_s: float, leaked_total: int) -> None:
        """Tell the coordinator this process left ``release`` behind past its bound (0025).

        Once per leak; the daemon forwards it under this process's connection, and the coordinator
        decides what follows.
        """
        rank, _ = self._runtime.identity()
        self._send(
            to_message(ReleaseLeaked(self._engine, rank, release, int(run_s * 1000), leaked_total))
        )

    def report_group_id(self, gid: GroupId) -> None:
        """Send the id this process made as its group's server.

        The control plane carries it to the group's other members (decision 0035, amended).
        """
        self._send(to_message(gid))

    def join(self, epoch: int, payload: bytes) -> None:
        """Answer the executor's join with this process's own payload (decision 0024).

        Routed by the coordinator to the engine's clients; the control plane never reads it.
        """
        rank, _ = self._runtime.identity()
        self._send(to_message(Join(engine=self._engine, epoch=epoch, rank=rank, payload=payload)))

    def _serve(self) -> None:
        self._serve_signals("daemon connection", self._take)

    def _take(self, sig: Any) -> bool:
        if isinstance(sig, Change):
            self._apply(sig)
        elif isinstance(sig, MembershipCommitted):
            self._heard_commit(sig)
        elif isinstance(sig, Notice):
            self._runtime.on_notice(sig.rank, sig.deadline_ms)
        elif isinstance(sig, Join):
            self.joins.put(sig)
        elif isinstance(sig, Abandoned):
            self._runtime.on_abandoned(sig.epoch, sig.detail)
        elif isinstance(sig, CoverPlan):
            self._runtime.on_cover_plan(sig)  # at once; groups form on the cover's thread
        elif isinstance(sig, GroupId):
            self.group_ids.put(sig)
        else:
            return False
        return True

    def _heard_commit(self, commit: MembershipCommitted) -> None:
        """A committed membership the daemon sent this process: whether it still serves in it.

        A daemon sends a worker the commit that no longer places it on its node (a process
        whose node and device are no longer at its rank has left). The process has been
        displaced, so it leaves (``on_displaced``) and its node restarts it to ask anew. A
        commit placing it at an epoch whose change it never staged is one it missed. Its groups
        are an earlier epoch's, and it leaves the same way. The protocol rules that out (a
        commit places only the nodes that acknowledged it), so it is the process's own check.
        A commit stating no placements displaces nobody, and one at or before this process's
        epoch is old news. A spare still standing is placed by no commit.
        """
        rank, epoch = self._runtime.identity()
        standing = self._role is Role.SPARE and self._runtime.took_epoch is None
        if not commit.members or commit.epoch <= epoch or standing:
            return
        here = (self._member.node, self._member.device)
        placed = any(
            m.rank == rank and m.node == here[0] and m.device in (None, here[1])
            for m in commit.members
        )
        if placed and commit.epoch in self._staged:
            return
        why = (
            f"epoch {commit.epoch} places rank {rank} elsewhere"
            if not placed
            else f"epoch {commit.epoch} places rank {rank} here, but its change never reached it"
        )
        log.warning("rank %d at epoch %d is stale: %s", rank, epoch, why)
        self._runtime.on_displaced(commit.epoch, why)

    def _departs(self, change: Change) -> bool:
        """Whether this process is the one leaving in ``change``: it serves a departing rank.

        A spare standing for a rank carries that rank as its identity before it takes it, and is
        the one arriving, never the one leaving: a spare that read its rank's departure as its
        own would neither apply nor acknowledge the change, and the change would fail. Once it
        has taken its position it serves, and may leave in turn.
        """
        rank, _ = self._runtime.identity()
        standing = self._role is Role.SPARE and self._runtime.took_epoch is None
        return rank in change.departing and not standing

    def _apply(self, change: Change) -> None:
        if self._departs(change):
            rank, _ = self._runtime.identity()
            # this process leaves on notice (protocol 9): the change is delivered to it outside
            # its barrier, so it stages nothing and acknowledges nothing; it finishes its step,
            # and the head's boundary drains it and releases it
            log.info(
                "rank %d leaves at epoch %d: its change heard, not applied", rank, change.epoch
            )
            return
        try:
            # a stamped change carries when the coordinator sent it (its clock): the daemon
            # hop shows against it, with the two clocks' offset; an unstamped one is as before
            stamp = {"sent_at": change.delivered_at_us / 1e6} if change.delivered_at_us else {}
            if change.take:
                self._runtime.on_take(change.take)  # a spare standing for any position takes first
            self._runtime.on_change(
                change.lost,
                outcome=change.outcome,
                epoch=change.epoch,
                departing=change.departing,
                **stamp,
            )
            self._staged.add(change.epoch)
            error = ""
        except Exception as e:  # noqa: BLE001 - the daemon must learn why, whatever it was
            log.exception("membership change failed")
            error = f"{type(e).__name__}: {e}"
        # read after the change: a spare standing for any position acknowledges as the rank
        # it took, failed or not
        rank, epoch = self._runtime.identity()
        self._send(pb.Signal(ack=pb.Ack(rank=rank, epoch=epoch, error=error)))

    def close(self) -> None:
        """Close the link. The daemon treats it as the worker leaving."""
        _shut(self._sock)


@dataclass(frozen=True)
class Heard:
    """A committed membership and when this client received it (``time.time()``)."""

    committed: MembershipCommitted
    at: float


@dataclass(frozen=True)
class ChangeHeard:
    """A change the coordinator claimed and when this client received it (``time.time()``)."""

    change: Change
    at: float


class DaemonClient(HearsEngine):
    """An executor's or a driver's connection to the node daemon.

    A driver or a tool is a client (``ROLE_CLIENT``). An engine's executor is its head
    (``ROLE_HEAD``, decision 0032): the engine is down when this connection ends, and a head says
    why with :meth:`goodbye` before it closes.
    """

    def __init__(
        self,
        path: str,
        engine: str,
        member: Member,
        connect_timeout_s: float = 30.0,
        slot: SlotChoice | None = None,
        head: bool = False,
        launcher_pid: int = 0,
        identity: SoftwareIdentity | None = None,
    ) -> None:
        """Connect to the daemon at ``path`` for ``engine`` and start listening.

        ``slot`` is the engine's slot-collective choice, reported in the readiness view. ``head``
        registers the connection as the engine's head, with ``launcher_pid`` the process a stop is
        sent to (this process's own when the engine core runs in it). ``identity`` is what the
        process runs (decision 0034).
        """
        super().__init__()
        self._sock = _connect(path, connect_timeout_s)
        self._lock = threading.Lock()
        self.committed: queue.Queue[Heard] = queue.Queue()
        self.changes: queue.Queue[ChangeHeard] = queue.Queue()
        """Every ``Change`` the coordinator claimed for this engine, as the daemon forwarded it:
        the ranks it names are lost from that moment (decision 0023)."""
        self.joins: queue.Queue[Join] = queue.Queue()
        """Join answers from the processes that took positions (decision 0024)."""
        self.abandoned: queue.Queue[Abandoned] = queue.Queue()
        """Changes that failed after their delivery: the engine is down (decision 0032)."""
        self.replay_asks: queue.Queue[ReplayAsk] = queue.Queue()
        """Spares asking the core for what it sent their rank at its start (decision 0044)."""
        self._engine = engine
        self._send(
            Register(
                engine=engine,
                rank=0,
                member=member,
                role=Role.HEAD if head else Role.CLIENT,
                slot=slot,
                launcher_pid=launcher_pid if head else 0,
                identity=identity,
                host=own_host(),
            )
        )
        threading.Thread(target=self._listen, daemon=True, name="everserve-daemon-client").start()

    def _send(self, sig: Register | RankLost | Notice | Join | Goodbye | ReplayOffer) -> None:
        with self._lock:
            write_frame(self._sock, to_message(sig).SerializeToString())

    def report_lost(self, ranks: Sequence[int], detail: str = "", epoch: int = 0) -> None:
        """Report that ``ranks`` are gone, as seen at membership ``epoch``.

        The coordinator drives one change for the whole set, and folds a report of a rank that has
        changed since ``epoch`` (decision 0021).
        """
        self._send(RankLost(ranks=tuple(ranks), detail=detail, epoch=epoch))

    def join(self, rank: int, epoch: int, payload: bytes) -> None:
        """Send the executor's join payload to the process in ``rank``'s position (0024).

        The coordinator routes it to the node holding the rank, whose daemon hands it to that
        process; its answer arrives in :attr:`joins`. The control plane never reads the bytes.
        """
        self._send(Join(engine=self._engine, epoch=epoch, rank=rank, payload=payload))

    def offer_replay(self, ask: ReplayAsk, handle: str) -> None:
        """Answer ``ask`` with the queue made for its asker alone (decision 0044).

        ``handle`` is the adapter's, which the control plane carries unread to the asker's
        connection.

        Raises:
            ValueError: for an ask its daemon did not name the asker of.
        """
        if ask.asker is None:
            raise ValueError(f"an ask for rank {ask.rank} names no asker: its daemon names it")
        self._send(
            ReplayOffer(
                engine=ask.engine,
                rank=ask.rank,
                handle=handle,
                to=ask.asker,
            )
        )

    def goodbye(self, cause: StopCause, detail: str = "") -> None:
        """Say why the head stops, then close: without it, the head reads as lost (0032)."""
        try:
            self._send(Goodbye(cause=cause, detail=detail))
        finally:
            self.close()

    def notice(self, rank: int, deadline_ms: int) -> None:
        """Deliver a notice: ``rank`` leaves at ``deadline_ms`` (Unix milliseconds)."""
        self._send(Notice(rank=rank, deadline_ms=deadline_ms, reason=Reason.NOTICE))

    def wait_committed(self, timeout_s: float | None = None) -> Heard:
        """The next committed membership heard.

        Raises:
            queue.Empty: if none arrives within ``timeout_s``.
        """
        return self.committed.get(timeout=timeout_s)

    def _listen(self) -> None:
        self._serve_signals("daemon client connection", self._take)

    def _take(self, sig: Any) -> bool:
        if isinstance(sig, MembershipCommitted):
            self.committed.put(Heard(sig, time.time()))
        elif isinstance(sig, Change):
            # the coordinator claimed a change: the ranks it names are lost from now, which an
            # executor cannot see for a rank on another node (decision 0023)
            self.changes.put(ChangeHeard(sig, time.time()))
        elif isinstance(sig, Join):
            self.joins.put(sig)
        elif isinstance(sig, Abandoned):
            self.abandoned.put(sig)
        elif isinstance(sig, ReplayAsk):
            self.replay_asks.put(sig)
        else:
            return False
        return True

    def close(self) -> None:
        """Close the connection."""
        _shut(self._sock)


class Refused(RuntimeError):  # noqa: N818 - the coordinator's word, not an error's name
    """The coordinator refused this process (decision 0034's guards): why, in its own words.

    Permanent for this process: what it resolved will not change by asking again. It says so in
    a goodbye (:data:`~everserve.contract.signals.StopCause.REFUSED`) and exits with
    :data:`EXIT_REFUSED`.
    """

    def __init__(self, refusal: str) -> None:
        """``refusal`` names each field that differs from the engine's."""
        super().__init__(refusal)
        self.refusal = refusal


class CacheUnavailable(RuntimeError):  # noqa: N818 - the daemon's word, not an error's name
    """This node could not put a rank's compiled set in place for its spare (decision 0043)."""


EXIT_REFUSED = 2
"""The exit status of a process a guard refused: permanent, unlike a step down's (75)."""

EXIT_STALE = 4
"""The exit status of a process a later commit displaced: its rank serves elsewhere, or its groups
are an epoch it never staged. Not a fault: restarted, it asks anew at the current epoch."""


class UndecidedConnection(HearsEngine):
    """A process asking its node's daemon what it is before it loads anything (decision 0041).

    A framework's group pod's process registers undecided for its stage (every rank its answer
    covers, which move together) and is told :class:`RoleAssigned`: serve the stage as its worker
    at the answer's epoch, or stand as the stage's spare. The connection then stays open and
    silent for the process's life, its liveness: the daemon releases the answer when it ends
    before the stage's workers register on their own connections. The daemon may ask it for
    its stacks.
    """

    def __init__(
        self,
        path: str,
        engine: str,
        stage: Sequence[int],
        member: Member,
        connect_timeout_s: float = 30.0,
        identity: SoftwareIdentity | None = None,
    ) -> None:
        """Connect to the daemon at ``path`` and register undecided for ``stage`` of ``engine``.

        Raises:
            ValueError: for an empty stage.
        """
        if not stage:
            raise ValueError("an undecided registration names at least one rank")
        super().__init__()
        self._sock = _connect(path, connect_timeout_s)
        self._lock = threading.Lock()
        self._answer: queue.Queue[RoleAssigned] = queue.Queue(maxsize=1)
        self._caches: queue.Queue[CacheReady | CacheRefused] = queue.Queue()
        self._engine = engine
        with self._lock:
            reg = Register(
                engine=engine,
                rank=stage[0],
                member=member,
                role=Role.UNDECIDED,
                identity=identity,
                stage=tuple(stage),
                host=own_host(),
            )
            write_frame(self._sock, to_message(reg).SerializeToString())
        threading.Thread(target=self._listen, daemon=True, name="everserve-undecided").start()

    def assigned(self, timeout_s: float | None = None) -> RoleAssigned:
        """The daemon's answer, once it comes.

        Raises:
            Refused: when the coordinator refused this process: its role means nothing then.
            ConnectionError: if the connection ends before an answer.
            TimeoutError: if none comes within ``timeout_s``.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            try:
                answer = self._answer.get(timeout=0.2)
                if answer.refusal:
                    raise Refused(answer.refusal)
                return answer
            except queue.Empty:
                if self.closed.is_set() and self._answer.empty():
                    raise ConnectionError(
                        "the daemon closed the connection before it answered"
                    ) from None
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"no answer from the daemon in {timeout_s} s") from None

    def _listen(self) -> None:
        self._serve_signals("undecided connection", self._take)

    def _take(self, sig: Any) -> bool:
        if isinstance(sig, RoleAssigned):
            if self._answer.full():
                log.warning("a second answer, at epoch %d, ignored", sig.epoch)
            else:
                self._answer.put(sig)
        elif isinstance(sig, CacheReady | CacheRefused):
            self._caches.put(sig)
        else:
            return False
        return True

    def cache(self, rank: int, dp: int = 0, timeout_s: float | None = None) -> CacheReady:
        """Ask this node for rank ``rank``'s compiled set, and wait until it is in place (0043).

        The daemon has its store fetch the set from the holders the placement names, once the
        rank's serving process published it, within its own bound; asks are answered in order.

        Raises:
            CacheUnavailable: the daemon could not put the set in place, its reason given.
            ConnectionError: if the connection ends before an answer.
            TimeoutError: if none comes within ``timeout_s``.
        """
        with self._lock:
            ask = CacheWanted(engine=self._engine, rank=rank, dp=dp)
            write_frame(self._sock, to_message(ask).SerializeToString())
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            try:
                answer = self._caches.get(timeout=0.2)
            except queue.Empty:
                if self.closed.is_set() and self._caches.empty():
                    raise ConnectionError(
                        f"the daemon closed the connection before rank {rank}'s compiled set"
                    ) from None
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"rank {rank}'s compiled set not in place in {timeout_s} s"
                    ) from None
                continue
            if answer.rank != rank:
                log.warning("an answer for rank %d's compiled set, asked for %d", answer.rank, rank)
                continue
            if isinstance(answer, CacheRefused):
                raise CacheUnavailable(answer.reason)
            return answer

    def refused(self, refusal: Refused) -> None:
        """Say a guard refused this process, then close (the engine's events name why)."""
        try:
            with self._lock:
                goodbye = Goodbye(cause=StopCause.REFUSED, detail=refusal.refusal)
                write_frame(self._sock, to_message(goodbye).SerializeToString())
        except OSError:
            pass  # the daemon closed first, as it does after a refusal: its event is recorded
        finally:
            self.close()

    def close(self) -> None:
        """Close the connection: the daemon releases what it answered, if nothing serves it yet."""
        _shut(self._sock)
