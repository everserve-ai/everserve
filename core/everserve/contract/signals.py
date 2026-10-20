# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The signal protocol between the controller and an adapter.

These are the only messages the controller sends into an engine process and the only ones an
adapter sends out. The controller never touches engine state; an adapter never decides
membership. Everything here is plain data with no engine or torch dependency.

The wire form is ``proto/signals.proto``; :func:`to_message` and :func:`from_message` convert
through one registry of kinds, and a test keeps the two in step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from everserve.outcome import Outcome
from everserve.proto import signals_pb2 as pb

PROTOCOL_VERSION: int = int(pb.PROTOCOL_VERSION)
"""The protocol this build speaks, from the one definition in ``signals.proto`` (decision 0027)."""

PROTOCOL_OLDEST: int = int(pb.PROTOCOL_OLDEST)
"""The oldest protocol this build serves (decision 0040): a link opens when each side serves the
other, and speaks the lower version."""


class Reason(str, Enum):
    """Why a rank is leaving or has left."""

    NOTICE = "notice"
    """A preemption notice: the rank is still healthy and will leave at ``deadline_ms``."""
    FAULT = "fault"
    """The rank is gone or unreachable now."""
    OPERATOR = "operator"
    """An operator asked for the change."""
    RETIRED = "retired"
    """The coordinator retires the process: a communicator's release still runs past its
    bound, holding its memory until the process ends (decision 0025, amended). Healthy, the
    rank is taken by a spare at the deadline with no gap."""


@dataclass(frozen=True)
class Notice:
    """A rank will leave at a known time.

    Raised by the controller when a cloud preemption signal arrives. The adapter has until
    ``deadline_ms`` (Unix milliseconds) to commit state and swap to a shadow group.
    """

    rank: int
    deadline_ms: int
    reason: Reason = Reason.NOTICE


@dataclass(frozen=True)
class RankLost:
    """One or more ranks are gone now: one rank, or every rank of a node, reported once.

    Raised by an adapter when a collective fails, or by the controller when a node vanishes.
    ``epoch`` is the membership epoch the reporter saw the ranks in; the coordinator folds a
    report of a rank that has changed since (decision 0021).
    """

    ranks: tuple[int, ...]
    reason: Reason = Reason.FAULT
    detail: str = ""
    epoch: int = 0


@dataclass(frozen=True)
class Member:
    """One GPU slot in the fleet."""

    node: str
    device: int


@dataclass(frozen=True)
class RankJoining:
    """A spare is ready to take ``rank`` at constant world size."""

    rank: int
    member: Member


@dataclass(frozen=True)
class Change:
    """The controller's decision for a set of lost ranks: move to the new member set at ``epoch``.

    Under REPLACE the set is the members with each lost rank's spare in its place; under SHRINK
    it is the members without the lost ranks.
    """

    lost: tuple[int, ...]
    outcome: Outcome
    epoch: int
    delivered_at_us: int = 0
    """When the coordinator sent it, on its clock, microseconds since the Unix epoch."""
    take: tuple[Take, ...] = ()
    """Under REPLACE, which spare standing for any position takes which lost rank (decision
    0030); a spare preloaded for one rank takes that rank and is not listed."""
    departing: tuple[int, ...] = ()
    """The lost ranks leaving on notice, not by fault: still running, they hear this change
    outside its barrier, and drain and release (protocol 9). The head awaits their drain."""


@dataclass(frozen=True)
class RankPlacement:
    """Where ``rank`` runs: its node, device and the node's reachable address (protocols 9, 11).

    ``device`` is None for a rank placed only by its assignment, before its worker registers (an
    :class:`EnginePlacement` at bring-up); a committed membership's members always carry one.
    ``host`` is empty from a process that did not state one (the harness driver's, found by
    discovery).
    """

    rank: int
    node: str
    device: int | None
    host: str = ""
    compile_cache: CompileCache | None = None
    """The rank's compile cache, as its serving process last reported it (decision 0029)."""
    cache_holders: tuple[str, ...] = ()
    """The nodes whose stores hold that compiled set."""


@dataclass(frozen=True)
class SparePlacement:
    """Where a standing spare runs: for ``for_rank``, or for any position when None (0030)."""

    for_rank: int | None
    node: str
    device: int
    host: str = ""


@dataclass(frozen=True)
class ProcessRef:
    """One connection of one daemon: the node, the daemon's instance, the connection's serial.

    A restarted daemon is another instance, so a reply addressed to a connection of the old one is
    dropped rather than delivered to whatever holds that serial now.
    """

    node: str
    instance: int
    serial: int


@dataclass(frozen=True)
class EnginePlacement:
    """Where every rank of the engine and every standing spare runs, by host (protocol 11).

    Sent to every connection of the engine once every rank of its world is placed (assigned or
    registered), and again at every change: a process bootstraps from it in place of a policy that
    pins members and a discovery directory, and takes the last it received.
    """

    engine: str
    epoch: int
    members: tuple[RankPlacement, ...]
    spares: tuple[SparePlacement, ...] = ()


@dataclass(frozen=True)
class Take:
    """A spare standing for any position, by node and device, takes ``rank`` (decision 0030)."""

    rank: int
    node: str
    device: int
    compile_cache: CompileCache | None = None
    """The position's compile cache as its serving rank last reported it (decision 0029); the
    spare refuses the change, naming both pairs, when its own differs after it attaches."""


class Role(str, Enum):
    """What a connection to the node daemon is."""

    WORKER = "worker"
    """A serving worker: receives changes and acknowledges them."""
    SPARE = "spare"
    """A pre-loaded spare standing in for ``rank``; serves as ``rank`` once a REPLACE commits."""
    CLIENT = "client"
    """The executor or a drill driver: reports losses, delivers notices, hears commits."""
    HEAD = "head"
    """The engine's head, on its executor's connection (decision 0032): a client in everything a
    client does, and the engine is down when this connection ends."""
    UNDECIDED = "undecided"
    """A framework's group pod's process asking, before it loads anything, whether it serves its
    stage or stands as the stage's spare (decision 0041); the daemon answers with
    :class:`RoleAssigned`."""


class StopCause(str, Enum):
    """Why an engine's head stopped (decision 0032)."""

    STOPPED = "stopped"
    """In order, on request."""
    FAILED = "failed"
    """In order, after a failure the executor raised that was not a loss."""
    LOST = "lost"
    """Its connection ended without a goodbye; set by the daemon, never sent by a head."""
    REFUSED = "refused"
    """A guard refused the process (its step config or its store's size, decision 0034): it says
    why in the goodbye's detail and exits, and its engine's events show it."""


@dataclass(frozen=True)
class Goodbye:
    """A head's last message before it closes its connection (decision 0032)."""

    cause: StopCause
    detail: str = ""


@dataclass(frozen=True)
class Abandoned:
    """A change that failed after its delivery: the engine is down, and ``epoch`` is spent.

    The processes that applied the change hold a membership that will not commit (decision 0032).
    """

    epoch: int
    detail: str = ""


@dataclass(frozen=True)
class AckAt:
    """An acknowledgement a commit waited on, and when its daemon received it (its clock, us)."""

    rank: int
    received_at_us: int
    epoch: int = 0
    error: str = ""
    """Why the process failed the change; empty when it applied it."""


@dataclass(frozen=True)
class SlotChoice:
    """Which slot-collective variant an engine runs, and how that was decided.

    ``variant`` is ``"p2p"`` or ``"host"``; ``how`` is ``"override"``, ``"nvml"``, ``"cache"`` or
    ``"probe"`` (``everserve.comm.slotkernel.decide_variant``).
    """

    variant: str
    how: str
    detail: str = ""
    seconds: float = 0.0


@dataclass(frozen=True)
class CompileCache:
    """The compiled-kernel cache a process loaded and warmed with (decision 0029)."""

    key: str
    """vLLM's ``torch_compile_cache`` key."""
    digest: str
    """The digest over every file under it, the autotune results included."""


@dataclass(frozen=True)
class KvFormat:
    """An engine's KV layout for one position (decision 0034), not the store's row format."""

    dtype: str
    """The cache tensors' element type, as torch names it (``float8_e4m3fn``, ``bfloat16``)."""
    block_size: int
    layout_hash: str = ""
    """sha256 over the attention backend's name, the cache layout, ``num_kv_heads`` on the rank,
    ``head_size`` and the stage's layer count; empty for a process holding no position."""


@dataclass(frozen=True)
class ParallelWidths:
    """An engine's parallel widths (the wire's ``ParallelLayout``)."""

    tensor: int = 1
    pipeline: int = 1
    data: int = 1
    expert: int = 1
    context: int = 1
    """vLLM's decode context-parallel width."""


@dataclass(frozen=True)
class StepConfig:
    """What a process steps with, as it resolved its configuration (decision 0034's step guard).

    ``fields`` are the settings that shape its collectives and compiled graph, ``digest`` sha256
    over their sorted ``name=value`` lines.
    """

    digest: str
    fields: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftwareIdentity:
    """What a process runs, kept apart from where it runs (decision 0034).

    The coordinator compares the engine and adapter commits, the adapter's tree being clean, NCCL,
    the checkpoint's digest, the quantization, the model's dtype, the widths and the KV format
    (CUDA too under the exact guarantee class) before any position moves; the rest is reported.
    An empty compared field never matches: a destination that cannot say what it runs is refused.
    """

    engine: str
    engine_commit: str
    adapter_commit: str
    adapter_dirty: bool = False
    torch_version: str = ""
    cuda_version: str = ""
    driver_version: str = ""
    nccl_version: str = ""
    model: str = ""
    model_revision: str = ""
    checkpoint_digest: str = ""
    """sha256 over the sorted (file name, content sha256) of the checkpoint's weights and config:
    every weight file, never a shard, and none of the repository's other files."""
    quantization: str = ""
    model_dtype: str = ""
    kv_format: KvFormat | None = None
    widths: ParallelWidths | None = None
    step: StepConfig | None = None
    """What the process steps with (``everserve_vllm.identity.step_config``): compared with its
    engine's at registration, the undecided ask included; an empty one never matches."""
    guarantee: str = ""
    """The engine's guarantee class the process was started for (``exact`` or ``continuous``):
    held to the engine's policy at the ask and at a spare's admission; empty never matches."""
    serves_changes: bool = False
    """Whether the process's engine core completes membership changes in its own step loop
    (``everserve_vllm.serving``). The coordinator changes nothing on an engine whose head did not
    say so: an engine that only looks protected is refused before a loss, not at it."""


@dataclass(frozen=True)
class Register:
    """The first message on a connection to the node daemon.

    An engine's client may carry the engine's slot-collective choice, which the readiness view
    reports.
    """

    engine: str
    rank: int
    member: Member
    role: Role = Role.WORKER
    slot: SlotChoice | None = None
    protocol: int = PROTOCOL_VERSION
    """The protocol this build speaks (decision 0027); the daemon refuses another."""
    protocol_oldest: int = PROTOCOL_OLDEST
    """The oldest protocol the registering process serves (decision 0040); 0 from a build before
    it, read as ``protocol`` alone."""
    compile_cache: CompileCache | None = None
    """What the process runs with, reported after its warm-up (decision 0029); the coordinator
    holds every process of an engine to its first's. None from a process that compiled nothing."""
    any_rank: bool = False
    """A spare standing for any position of the engine (decision 0030): ``rank`` is unused and
    the compile cache is reported at the join, once it has attached."""
    launcher_pid: int = 0
    """A head's launcher, the process a stop is sent to (decision 0032); zero for other roles."""
    identity: SoftwareIdentity | None = None
    """What the process runs (decision 0034); every process of this build sends it."""
    stage: tuple[int, ...] = ()
    """An undecided registration's stage: every rank its answer covers, ``rank`` the first;
    empty, ``rank`` alone (decision 0041)."""
    host: str = ""
    """The address this process's node is reached at, where its groups rendezvous (protocol 11);
    empty from a process that has none to state (the harness driver's, found by discovery)."""


@dataclass(frozen=True)
class RoleAssigned:
    """The daemon's answer to an undecided registration (decision 0041).

    ``role`` is ``WORKER``, the one process to serve the stage at ``epoch``, or ``SPARE``, a spare
    preloaded for the stage's shards. ``stage`` is the ranks the answer covers, as asked.
    """

    engine: str
    rank: int
    epoch: int
    role: Role
    stage: tuple[int, ...] = ()
    refusal: str = ""
    """Set when the coordinator refused the process: why, each differing field named. ``role``
    then means nothing (its default is ``WORKER``), so a refused answer is never read as a role
    (:meth:`everserve.contract.daemon.UndecidedConnection.assigned` raises it)."""


@dataclass(frozen=True)
class EnginePolicy:
    """The engine's policy, from the coordinator through the process's daemon.

    ``policy`` is an encoded ``Policy`` narrowed to the engine: its entry and the policy-wide
    fields. Sent on every connection after its registration, an undecided one's before its answer,
    and again whenever it changes or the daemon relinks; a process takes the last it received.
    """

    engine: str
    policy: bytes


class InstancePhase(str, Enum):
    """An instance's state in its framework's terms (decision 0041, obligation 2)."""

    STARTING = "starting"
    """Bring-up: not every position of the world is held yet."""
    SERVING = "serving"
    """Every position held, nothing changing."""
    RECOVERING = "recovering"
    """A change is in flight: a replacement, a drain's move, a shrink."""
    DEGRADED = "degraded"
    """Serving with positions lost: its capacity is down."""
    FAILED = "failed"
    """The engine is down (decision 0032)."""


@dataclass(frozen=True)
class InstanceState:
    """The instance's state, from the coordinator through the process's daemon, on every change."""

    engine: str
    phase: InstancePhase
    positions_held: int
    world: int
    epoch: int
    reason: str = ""


class UnknownSignal(ValueError):  # noqa: N818 - a kind of signal, not an error's name
    """A signal of a kind this build's codec does not decode, though its schema names it."""


@dataclass(frozen=True)
class DumpStacks:
    """The daemon asks the process to write every thread's stack to its own log.

    Asked on the process's link, never by a signal to its pid, which a daemon in another pod
    cannot see. Nothing is answered yet: an answer (``StacksDumped``) is not built.
    """

    reason: str = ""
    request_id: int = 0
    """The daemon's: an answer (``StacksDumped``) carries it back to whoever asked."""


@dataclass(frozen=True)
class Join:
    """An opaque join payload between an engine's executor and the process taking a position.

    From the executor to ``rank``'s process, and that process's answer back (decision 0024). The
    control plane carries ``payload`` unread; the adapters on both ends define it.
    """

    engine: str
    epoch: int
    rank: int
    payload: bytes


@dataclass(frozen=True)
class MembershipCommitted:
    """The engine is serving again on ``active_ranks``.

    ``epoch`` increases by one on every committed change and fences stale messages.
    """

    epoch: int
    active_ranks: tuple[int, ...]
    world_size: int
    pause_ms: int = 0
    lost: tuple[int, ...] = field(default_factory=tuple)
    reported_at_us: int = 0
    """The change's moments on the coordinator's clock, microseconds since the Unix epoch: the
    report that moved it, the delivery, the last acknowledgement, the commit."""
    delivered_at_us: int = 0
    acknowledged_at_us: int = 0
    committed_at_us: int = 0
    acks: tuple[AckAt, ...] = ()
    """Every acknowledgement the commit waited on, with when its daemon received it."""
    members: tuple[RankPlacement, ...] = ()
    """The whole committed membership: where every active rank runs from this epoch on
    (protocol 9). A listener that hears only commits derives what it holds from it, whatever
    commits it missed."""

    def __post_init__(self) -> None:
        """Reject a membership that names a rank outside the world."""
        for r in (*self.active_ranks, *self.lost):
            if not 0 <= r < self.world_size:
                raise ValueError(f"rank {r} outside world of {self.world_size}")
        if set(self.active_ranks) & set(self.lost):
            raise ValueError("a rank cannot be both active and lost")


@dataclass(frozen=True)
class SpareRef:
    """A spare before it takes a position: the rank it stands in for, and its node and device."""

    for_rank: int
    node: str
    device: int


SetMember = int | SpareRef
"""A member of a cover plan's set: a serving rank, or a spare (decision 0035)."""


@dataclass(frozen=True)
class CoverGroup:
    """One group of a cover plan (decision 0035, amended): its scope and its spans.

    A group is the same group whatever sets need it. ``scope`` is ``"domain"`` or ``"fabric"``;
    ``spans`` the members its communicator spans, in its rank order; ``server`` the member that
    makes the group's id (:class:`GroupId`), which the control plane's links carry to the others:
    no member opens a port for it.
    """

    scope: str
    spans: tuple[SetMember, ...]
    server: SetMember


@dataclass(frozen=True)
class CoverSet:
    """One set of a cover plan: its members, and the groups it needs.

    ``groups`` are indices into the plan's ``groups``. A swap to the set points each scope at
    its group spanning the process.
    """

    members: tuple[SetMember, ...]
    groups: tuple[int, ...]
    purpose: str = ""


@dataclass(frozen=True)
class Holder:
    """The process holding a position: its node and the device it registered with."""

    node: str
    device: int


@dataclass(frozen=True)
class CoverPlan:
    """An engine's cover plan at ``generation``, for the placement at the committed ``epoch``.

    ``placement`` is each rank's holder at ``epoch``: a plan at a newer epoch whose holder of a
    rank differs tells every process its groups naming that rank hold the old process.
    """

    engine: str
    generation: int
    epoch: int
    placement: dict[int, Holder]
    groups: tuple[CoverGroup, ...]
    sets: tuple[CoverSet, ...] = ()


@dataclass(frozen=True)
class GroupFormed:
    """``member`` holds the group of ``scope`` over ``spans`` at ``generation``.

    Formed in ``seconds``, or kept from before (0).
    """

    engine: str
    generation: int
    spans: tuple[SetMember, ...]
    scope: str
    seconds: float
    member: SetMember


@dataclass(frozen=True)
class GroupFailed:
    """``member`` did not form the group, or refused to release it, with ``error``."""

    engine: str
    generation: int
    spans: tuple[SetMember, ...]
    scope: str
    error: str
    member: SetMember


@dataclass(frozen=True)
class GroupReleased:
    """``member`` no longer holds the group of ``scope`` over ``spans``."""

    engine: str
    generation: int
    spans: tuple[SetMember, ...]
    scope: str
    member: SetMember


@dataclass(frozen=True)
class GroupReady:
    """``member`` holds the plan at ``generation`` and waits for the group's id (0035, amended).

    The coordinator relays the group's id only once every member, its server included, is ready
    at the generation: the id is each member's go, so no member joins before every other has
    shown it is alive. A member lost after the go is the window NCCL's blocking initialisation
    leaves, bounded by the formation's deadline (decision 0025, amended).
    """

    engine: str
    generation: int
    spans: tuple[SetMember, ...]
    scope: str
    member: SetMember


@dataclass(frozen=True)
class GroupId:
    """A group's id, the NCCL unique id its members form it with (decision 0035, amended).

    Made by the group's server member and reported on its link; carried by the control plane to
    the daemons of the other members' nodes, which hand it to their engine's processes. Carried
    only on the authenticated links and the node's local sockets, and never logged. A group made
    again after a failure has a new id at a newer generation: a member takes the newest.
    """

    engine: str
    generation: int
    scope: str
    spans: tuple[SetMember, ...]
    id: bytes = field(repr=False)


@dataclass(frozen=True)
class ReleaseLeaked:
    """This process left a release behind past its bound (decision 0025, amended).

    Sent once per leaked release, at the boundary that counts it, or by a planned group's
    formation still running past the policy's ``cover.formTimeoutMs`` (``release`` then names
    the formation). The daemon forwards it under the process's own connection; the coordinator
    counts it and moves the position once a spare can take it.
    """

    engine: str
    rank: int
    release: str
    run_ms: int
    leaked_total: int
    """This process's count of releases left behind, this one included."""


@dataclass(frozen=True)
class CacheWanted:
    """A spare's process asks its node for its rank's compiled set, before it loads (0043).

    Sent on the undecided connection once it is told to stand. The daemon has its store fetch
    the set the rank's serving process published, from the holders the placement names, and
    answers :class:`CacheReady` or :class:`CacheRefused`, one ask at a time, in order.
    """

    engine: str
    rank: int
    dp: int = 0


@dataclass(frozen=True)
class CacheReady:
    """The rank's set is in place on this node: the pair it was fetched as."""

    engine: str
    rank: int
    key: str
    digest: str


@dataclass(frozen=True)
class CacheRefused:
    """The rank's set could not be put in place on this node, and why.

    No holder answered within the bound, the fetch would pass the node's compile cache budget,
    or a file did not match its manifest.
    """

    engine: str
    rank: int
    reason: str


@dataclass(frozen=True)
class ReplayAsk:
    """A spare asks the engine's core for what the core sent ``rank`` at its start (0044).

    The spare sends it with ``asker`` unset; its daemon names the spare's connection before
    passing it up, and the coordinator hands it to the head. The core answers that spare alone
    with :class:`ReplayOffer`.
    """

    engine: str
    rank: int
    asker: ProcessRef | None = None


@dataclass(frozen=True)
class ReplayOffer:
    """The core's answer to one :class:`ReplayAsk`: a queue made for the asker alone.

    ``handle`` is the adapter's, opaque to the control plane: the queue the spare reads the
    core's start-up calls for ``rank`` from. ``to`` is the asker.
    """

    engine: str
    rank: int
    handle: str
    to: ProcessRef


Signal = (
    Notice
    | RankLost
    | RankJoining
    | MembershipCommitted
    | Change
    | Register
    | Join
    | Goodbye
    | Abandoned
    | CoverPlan
    | GroupFormed
    | GroupFailed
    | GroupReleased
    | GroupReady
    | GroupId
    | ReleaseLeaked
    | RoleAssigned
    | DumpStacks
    | EnginePolicy
    | EnginePlacement
    | InstanceState
    | CacheWanted
    | CacheReady
    | CacheRefused
    | ReplayAsk
    | ReplayOffer
)


# ---------------------------------------------------------------------------- the wire codec
#
# Each kind of signal is one entry of _KINDS: the member of the proto's ``Signal.kind`` oneof it
# travels as, the dataclass it is here, and the two directions. The field helpers above a kind's
# pair convert the messages several kinds share. A field the proto has and a pair lacks fails
# ``core/tests/test_signals_coverage.py``, which fills every field of every kind.


def _enum_tables(to_pb: dict[Any, int]) -> tuple[dict[Any, int], dict[int, Any]]:
    return to_pb, {v: k for k, v in to_pb.items()}


_REASON_TO_PB, _REASON_FROM_PB = _enum_tables(
    {
        Reason.NOTICE: pb.REASON_NOTICE,
        Reason.FAULT: pb.REASON_FAULT,
        Reason.OPERATOR: pb.REASON_OPERATOR,
        Reason.RETIRED: pb.REASON_RETIRED,
    }
)
_OUTCOME_TO_PB, _OUTCOME_FROM_PB = _enum_tables(
    {
        Outcome.REPLACE: pb.OUTCOME_REPLACE,
        Outcome.SHRINK: pb.OUTCOME_SHRINK,
        Outcome.REBUILD: pb.OUTCOME_REBUILD,
    }
)
_ROLE_TO_PB, _ROLE_FROM_PB = _enum_tables(
    {
        Role.WORKER: pb.ROLE_WORKER,
        Role.SPARE: pb.ROLE_SPARE,
        Role.CLIENT: pb.ROLE_CLIENT,
        Role.HEAD: pb.ROLE_HEAD,
        Role.UNDECIDED: pb.ROLE_UNDECIDED,
    }
)
_PHASE_TO_PB, _PHASE_FROM_PB = _enum_tables(
    {
        InstancePhase.STARTING: pb.INSTANCE_STARTING,
        InstancePhase.SERVING: pb.INSTANCE_SERVING,
        InstancePhase.RECOVERING: pb.INSTANCE_RECOVERING,
        InstancePhase.DEGRADED: pb.INSTANCE_DEGRADED,
        InstancePhase.FAILED: pb.INSTANCE_FAILED,
    }
)
_CAUSE_TO_PB, _CAUSE_FROM_PB = _enum_tables(
    {
        StopCause.STOPPED: pb.STOP_CAUSE_STOPPED,
        StopCause.FAILED: pb.STOP_CAUSE_FAILED,
        StopCause.LOST: pb.STOP_CAUSE_LOST,
        StopCause.REFUSED: pb.STOP_CAUSE_REFUSED,
    }
)
_VARIANT_TO_PB, _VARIANT_FROM_PB = _enum_tables(
    {"p2p": pb.SlotChoice.VARIANT_P2P, "host": pb.SlotChoice.VARIANT_HOST}
)
_HOW_TO_PB, _HOW_FROM_PB = _enum_tables(
    {
        "override": pb.SlotChoice.HOW_OVERRIDE,
        "nvml": pb.SlotChoice.HOW_NVML,
        "cache": pb.SlotChoice.HOW_CACHE,
        "probe": pb.SlotChoice.HOW_PROBE,
    }
)
_SCOPE_TO_PB, _SCOPE_FROM_PB = _enum_tables({"domain": pb.SCOPE_DOMAIN, "fabric": pb.SCOPE_FABRIC})


def _enum(table: dict[int, Any], value: int, absent: str) -> Any:
    """``value``'s entry in ``table``.

    Raises:
        ValueError: ``absent``, for the unspecified value or one this build does not know.
    """
    if value not in table:
        raise ValueError(absent)
    return table[value]


# ------------------------------------------------------------------------------ shared fields


def _member_to_pb(m: Member) -> pb.Member:
    return pb.Member(node=m.node, device=m.device)


def _member_from_pb(m: pb.Member) -> Member:
    return Member(node=m.node, device=m.device)


def _compile_cache_to_pb(c: CompileCache | None) -> pb.CompileCache | None:
    return None if c is None else pb.CompileCache(key=c.key, digest=c.digest)


def _compile_cache_from_pb(holder: Any) -> CompileCache | None:
    """The ``compile_cache`` of ``holder``, None where it carries none."""
    if not holder.HasField("compile_cache"):
        return None
    return CompileCache(holder.compile_cache.key, holder.compile_cache.digest)


def _process_ref_to_pb(p: ProcessRef | None) -> pb.ProcessRef | None:
    return None if p is None else pb.ProcessRef(node=p.node, instance=p.instance, serial=p.serial)


def _process_ref_from_pb(p: pb.ProcessRef) -> ProcessRef:
    return ProcessRef(node=p.node, instance=p.instance, serial=p.serial)


def _slot_to_pb(s: SlotChoice | None) -> pb.SlotChoice | None:
    if s is None:
        return None
    return pb.SlotChoice(
        variant=_VARIANT_TO_PB[s.variant], how=_HOW_TO_PB[s.how], detail=s.detail, seconds=s.seconds
    )


def _slot_from_pb(s: pb.SlotChoice) -> SlotChoice:
    return SlotChoice(
        variant=_VARIANT_FROM_PB[s.variant],
        how=_HOW_FROM_PB[s.how],
        detail=s.detail,
        seconds=s.seconds,
    )


def _rank_placement_to_pb(p: RankPlacement) -> pb.RankPlacement:
    return pb.RankPlacement(
        rank=p.rank,
        node=p.node,
        device=p.device,
        host=p.host,
        compile_cache=_compile_cache_to_pb(p.compile_cache),
        cache_holders=list(p.cache_holders),
    )


def _rank_placement_from_pb(p: pb.RankPlacement) -> RankPlacement:
    """A rank's placement; its device None where absent, never device 0 (protocol 11)."""
    return RankPlacement(
        p.rank,
        p.node,
        p.device if p.HasField("device") else None,
        p.host,
        _compile_cache_from_pb(p),
        tuple(p.cache_holders),
    )


def _set_member_to_pb(m: SetMember) -> pb.SetMember:
    if isinstance(m, SpareRef):
        return pb.SetMember(spare=pb.SpareRef(for_rank=m.for_rank, node=m.node, device=m.device))
    return pb.SetMember(rank=m)


def _set_member_from_pb(m: pb.SetMember) -> SetMember:
    if m.WhichOneof("kind") == "spare":
        return SpareRef(m.spare.for_rank, m.spare.node, m.spare.device)
    return m.rank


def _scope_from_pb(scope: int) -> str:
    return _enum(_SCOPE_FROM_PB, scope, "a cover set without a scope")


def _identity_to_pb(i: SoftwareIdentity) -> pb.SoftwareIdentity:
    kv = i.kv_format
    w = i.widths
    return pb.SoftwareIdentity(
        engine=i.engine,
        engine_commit=i.engine_commit,
        adapter_commit=i.adapter_commit,
        adapter_dirty=i.adapter_dirty,
        torch_version=i.torch_version,
        cuda_version=i.cuda_version,
        driver_version=i.driver_version,
        nccl_version=i.nccl_version,
        model=i.model,
        model_revision=i.model_revision,
        checkpoint_digest=i.checkpoint_digest,
        quantization=i.quantization,
        model_dtype=i.model_dtype,
        kv_format=(
            pb.KvFormat(dtype=kv.dtype, block_size=kv.block_size, layout_hash=kv.layout_hash)
            if kv is not None
            else None
        ),
        layout=(
            pb.ParallelLayout(
                tensor=w.tensor,
                pipeline=w.pipeline,
                data=w.data,
                expert=w.expert,
                context=w.context,
            )
            if w is not None
            else None
        ),
        step=(
            pb.StepConfig(digest=i.step.digest, fields=dict(i.step.fields))
            if i.step is not None
            else None
        ),
        guarantee=i.guarantee,
        serves_changes=i.serves_changes,
    )


def _identity_from_pb(m: pb.SoftwareIdentity) -> SoftwareIdentity:
    return SoftwareIdentity(
        engine=m.engine,
        engine_commit=m.engine_commit,
        adapter_commit=m.adapter_commit,
        adapter_dirty=m.adapter_dirty,
        torch_version=m.torch_version,
        cuda_version=m.cuda_version,
        driver_version=m.driver_version,
        nccl_version=m.nccl_version,
        model=m.model,
        model_revision=m.model_revision,
        checkpoint_digest=m.checkpoint_digest,
        quantization=m.quantization,
        model_dtype=m.model_dtype,
        kv_format=(
            KvFormat(m.kv_format.dtype, m.kv_format.block_size, m.kv_format.layout_hash)
            if m.HasField("kv_format")
            else None
        ),
        widths=(
            ParallelWidths(
                m.layout.tensor, m.layout.pipeline, m.layout.data, m.layout.expert, m.layout.context
            )
            if m.HasField("layout")
            else None
        ),
        step=(StepConfig(m.step.digest, dict(m.step.fields)) if m.HasField("step") else None),
        guarantee=m.guarantee,
        serves_changes=m.serves_changes,
    )


# ------------------------------------------------------------------------------- each kind


def _notice_to_pb(s: Notice) -> pb.Notice:
    return pb.Notice(rank=s.rank, deadline_ms=s.deadline_ms, reason=_REASON_TO_PB[s.reason])


def _notice_from_pb(n: pb.Notice) -> Notice:
    reason = _enum(_REASON_FROM_PB, n.reason, "notice without a reason")
    return Notice(rank=n.rank, deadline_ms=n.deadline_ms, reason=reason)


def _rank_lost_to_pb(s: RankLost) -> pb.RankLost:
    return pb.RankLost(
        ranks=list(s.ranks), reason=_REASON_TO_PB[s.reason], detail=s.detail, epoch=s.epoch
    )


def _rank_lost_from_pb(r: pb.RankLost) -> RankLost:
    reason = _enum(_REASON_FROM_PB, r.reason, "rank_lost without a reason")
    return RankLost(ranks=tuple(r.ranks), reason=reason, detail=r.detail, epoch=r.epoch)


def _rank_joining_to_pb(s: RankJoining) -> pb.RankJoining:
    return pb.RankJoining(rank=s.rank, member=_member_to_pb(s.member))


def _rank_joining_from_pb(j: pb.RankJoining) -> RankJoining:
    return RankJoining(rank=j.rank, member=_member_from_pb(j.member))


def _membership_to_pb(s: MembershipCommitted) -> pb.MembershipCommitted:
    return pb.MembershipCommitted(
        epoch=s.epoch,
        active_ranks=list(s.active_ranks),
        world_size=s.world_size,
        pause_ms=s.pause_ms,
        lost=list(s.lost),
        reported_at_us=s.reported_at_us,
        delivered_at_us=s.delivered_at_us,
        acknowledged_at_us=s.acknowledged_at_us,
        committed_at_us=s.committed_at_us,
        acks=[
            pb.Ack(rank=a.rank, received_at_us=a.received_at_us, epoch=a.epoch, error=a.error)
            for a in s.acks
        ],
        members=[_rank_placement_to_pb(p) for p in s.members],
    )


def _membership_from_pb(m: pb.MembershipCommitted) -> MembershipCommitted:
    return MembershipCommitted(
        epoch=m.epoch,
        active_ranks=tuple(m.active_ranks),
        world_size=m.world_size,
        pause_ms=m.pause_ms,
        lost=tuple(m.lost),
        reported_at_us=m.reported_at_us,
        delivered_at_us=m.delivered_at_us,
        acknowledged_at_us=m.acknowledged_at_us,
        committed_at_us=m.committed_at_us,
        acks=tuple(AckAt(a.rank, a.received_at_us, a.epoch, a.error) for a in m.acks),
        members=tuple(_rank_placement_from_pb(p) for p in m.members),
    )


def _change_to_pb(s: Change) -> pb.Change:
    return pb.Change(
        lost=list(s.lost),
        outcome=_OUTCOME_TO_PB[s.outcome],
        epoch=s.epoch,
        delivered_at_us=s.delivered_at_us,
        take=[
            pb.Take(
                rank=t.rank,
                node=t.node,
                device=t.device,
                compile_cache=_compile_cache_to_pb(t.compile_cache),
            )
            for t in s.take
        ],
        departing=list(s.departing),
    )


def _change_from_pb(c: pb.Change) -> Change:
    if c.outcome == pb.OUTCOME_UNSPECIFIED:
        raise ValueError("change without an outcome")
    if c.outcome not in _OUTCOME_FROM_PB:
        # an outcome added after this build, to a process that predates it: its reader passes
        # it over and the link lives, rather than end on a schema newer than its own
        raise UnknownSignal(f"a change of outcome {c.outcome}, unknown to this build")
    return Change(
        lost=tuple(c.lost),
        outcome=_OUTCOME_FROM_PB[c.outcome],
        epoch=c.epoch,
        delivered_at_us=c.delivered_at_us,
        take=tuple(Take(t.rank, t.node, t.device, _compile_cache_from_pb(t)) for t in c.take),
        departing=tuple(c.departing),
    )


def _register_to_pb(s: Register) -> pb.Register:
    return pb.Register(
        engine=s.engine,
        rank=s.rank,
        member=_member_to_pb(s.member),
        role=_ROLE_TO_PB[s.role],
        slot=_slot_to_pb(s.slot),
        protocol=s.protocol,
        protocol_oldest=s.protocol_oldest,
        compile_cache=_compile_cache_to_pb(s.compile_cache),
        any_rank=s.any_rank,
        launcher_pid=s.launcher_pid,
        identity=_identity_to_pb(s.identity) if s.identity is not None else None,
        stage=list(s.stage),
        host=s.host,
    )


def _register_from_pb(r: pb.Register) -> Register:
    return Register(
        engine=r.engine,
        rank=r.rank,
        member=_member_from_pb(r.member),
        role=_ROLE_FROM_PB[r.role],
        slot=_slot_from_pb(r.slot) if r.HasField("slot") else None,
        protocol=r.protocol,
        protocol_oldest=r.protocol_oldest,
        compile_cache=_compile_cache_from_pb(r),
        any_rank=r.any_rank,
        launcher_pid=r.launcher_pid,
        identity=_identity_from_pb(r.identity) if r.HasField("identity") else None,
        stage=tuple(r.stage),
        host=r.host,
    )


def _join_to_pb(s: Join) -> pb.Join:
    return pb.Join(engine=s.engine, epoch=s.epoch, rank=s.rank, payload=s.payload)


def _join_from_pb(j: pb.Join) -> Join:
    return Join(engine=j.engine, epoch=j.epoch, rank=j.rank, payload=bytes(j.payload))


def _goodbye_to_pb(s: Goodbye) -> pb.Goodbye:
    return pb.Goodbye(cause=_CAUSE_TO_PB[s.cause], detail=s.detail)


def _goodbye_from_pb(g: pb.Goodbye) -> Goodbye:
    return Goodbye(cause=_enum(_CAUSE_FROM_PB, g.cause, "goodbye without a cause"), detail=g.detail)


def _abandoned_to_pb(s: Abandoned) -> pb.Abandoned:
    return pb.Abandoned(epoch=s.epoch, detail=s.detail)


def _abandoned_from_pb(a: pb.Abandoned) -> Abandoned:
    return Abandoned(epoch=a.epoch, detail=a.detail)


def _cover_plan_to_pb(s: CoverPlan) -> pb.CoverPlan:
    return pb.CoverPlan(
        engine=s.engine,
        generation=s.generation,
        epoch=s.epoch,
        placement={r: pb.Holder(node=h.node, device=h.device) for r, h in s.placement.items()},
        groups=[
            pb.CoverGroup(
                scope=_SCOPE_TO_PB[g.scope],
                spans=[_set_member_to_pb(m) for m in g.spans],
                server=_set_member_to_pb(g.server),
            )
            for g in s.groups
        ],
        sets=[
            pb.CoverSet(
                members=[_set_member_to_pb(m) for m in c.members],
                groups=list(c.groups),
                purpose=c.purpose,
            )
            for c in s.sets
        ],
    )


def _cover_plan_from_pb(c: pb.CoverPlan) -> CoverPlan:
    return CoverPlan(
        engine=c.engine,
        generation=c.generation,
        epoch=c.epoch,
        placement={r: Holder(h.node, h.device) for r, h in c.placement.items()},
        groups=tuple(
            CoverGroup(
                scope=_scope_from_pb(g.scope),
                spans=tuple(_set_member_from_pb(m) for m in g.spans),
                server=_set_member_from_pb(g.server),
            )
            for g in c.groups
        ),
        sets=tuple(
            CoverSet(
                members=tuple(_set_member_from_pb(m) for m in s.members),
                groups=tuple(s.groups),
                purpose=s.purpose,
            )
            for s in c.sets
        ),
    )


def _group_to_pb(s: GroupFormed | GroupFailed | GroupReleased | GroupReady) -> dict[str, Any]:
    """The fields every group report carries, as keyword arguments to its message."""
    return {
        "engine": s.engine,
        "generation": s.generation,
        "spans": [_set_member_to_pb(m) for m in s.spans],
        "scope": _SCOPE_TO_PB[s.scope],
        "member": _set_member_to_pb(s.member),
    }


def _group_from_pb(g: Any) -> dict[str, Any]:
    """The fields every group report carries, as keyword arguments to its dataclass."""
    return {
        "engine": g.engine,
        "generation": g.generation,
        "spans": tuple(_set_member_from_pb(m) for m in g.spans),
        "scope": _scope_from_pb(g.scope),
        "member": _set_member_from_pb(g.member),
    }


def _group_formed_to_pb(s: GroupFormed) -> pb.GroupFormed:
    return pb.GroupFormed(seconds=s.seconds, **_group_to_pb(s))


def _group_formed_from_pb(g: pb.GroupFormed) -> GroupFormed:
    return GroupFormed(seconds=g.seconds, **_group_from_pb(g))


def _group_failed_to_pb(s: GroupFailed) -> pb.GroupFailed:
    return pb.GroupFailed(error=s.error, **_group_to_pb(s))


def _group_failed_from_pb(g: pb.GroupFailed) -> GroupFailed:
    return GroupFailed(error=g.error, **_group_from_pb(g))


def _group_released_to_pb(s: GroupReleased) -> pb.GroupReleased:
    return pb.GroupReleased(**_group_to_pb(s))


def _group_released_from_pb(g: pb.GroupReleased) -> GroupReleased:
    return GroupReleased(**_group_from_pb(g))


def _group_ready_to_pb(s: GroupReady) -> pb.GroupReady:
    return pb.GroupReady(**_group_to_pb(s))


def _group_ready_from_pb(g: pb.GroupReady) -> GroupReady:
    return GroupReady(**_group_from_pb(g))


def _group_id_to_pb(s: GroupId) -> pb.GroupId:
    return pb.GroupId(
        engine=s.engine,
        generation=s.generation,
        scope=_SCOPE_TO_PB[s.scope],
        spans=[_set_member_to_pb(m) for m in s.spans],
        id=s.id,
    )


def _group_id_from_pb(g: pb.GroupId) -> GroupId:
    return GroupId(
        engine=g.engine,
        generation=g.generation,
        scope=_scope_from_pb(g.scope),
        spans=tuple(_set_member_from_pb(m) for m in g.spans),
        id=bytes(g.id),
    )


def _release_leaked_to_pb(s: ReleaseLeaked) -> pb.ReleaseLeaked:
    return pb.ReleaseLeaked(
        engine=s.engine,
        rank=s.rank,
        release=s.release,
        run_ms=s.run_ms,
        leaked_total=s.leaked_total,
    )


def _release_leaked_from_pb(r: pb.ReleaseLeaked) -> ReleaseLeaked:
    return ReleaseLeaked(r.engine, r.rank, r.release, r.run_ms, r.leaked_total)


def _role_assigned_to_pb(s: RoleAssigned) -> pb.RoleAssigned:
    return pb.RoleAssigned(
        engine=s.engine,
        rank=s.rank,
        epoch=s.epoch,
        role=_ROLE_TO_PB[s.role],
        stage=list(s.stage),
        refusal=s.refusal,
    )


def _role_assigned_from_pb(a: pb.RoleAssigned) -> RoleAssigned:
    return RoleAssigned(
        engine=a.engine,
        rank=a.rank,
        epoch=a.epoch,
        role=_enum(_ROLE_FROM_PB, a.role, "role_assigned without a role"),
        stage=tuple(a.stage),
        refusal=a.refusal,
    )


def _dump_stacks_to_pb(s: DumpStacks) -> pb.DumpStacks:
    return pb.DumpStacks(reason=s.reason, request_id=s.request_id)


def _dump_stacks_from_pb(d: pb.DumpStacks) -> DumpStacks:
    return DumpStacks(reason=d.reason, request_id=d.request_id)


def _engine_policy_to_pb(s: EnginePolicy) -> pb.EnginePolicy:
    return pb.EnginePolicy(engine=s.engine, policy=s.policy)


def _engine_policy_from_pb(p: pb.EnginePolicy) -> EnginePolicy:
    return EnginePolicy(engine=p.engine, policy=bytes(p.policy))


def _placement_to_pb(p: EnginePlacement) -> pb.EnginePlacement:
    return pb.EnginePlacement(
        engine=p.engine,
        epoch=p.epoch,
        members=[_rank_placement_to_pb(m) for m in p.members],
        spares=[
            pb.SparePlacement(for_rank=s.for_rank, node=s.node, device=s.device, host=s.host)
            for s in p.spares
        ],
    )


def _placement_from_pb(e: pb.EnginePlacement) -> EnginePlacement:
    return EnginePlacement(
        engine=e.engine,
        epoch=e.epoch,
        members=tuple(_rank_placement_from_pb(p) for p in e.members),
        spares=tuple(
            SparePlacement(
                for_rank=s.for_rank if s.HasField("for_rank") else None,
                node=s.node,
                device=s.device,
                host=s.host,
            )
            for s in e.spares
        ),
    )


def _instance_state_to_pb(s: InstanceState) -> pb.InstanceState:
    return pb.InstanceState(
        engine=s.engine,
        phase=_PHASE_TO_PB[s.phase],
        positions_held=s.positions_held,
        world=s.world,
        epoch=s.epoch,
        reason=s.reason,
    )


def _instance_state_from_pb(st: pb.InstanceState) -> InstanceState:
    return InstanceState(
        engine=st.engine,
        phase=_enum(_PHASE_FROM_PB, st.phase, "instance_state without a phase"),
        positions_held=st.positions_held,
        world=st.world,
        epoch=st.epoch,
        reason=st.reason,
    )


def _cache_wanted_to_pb(s: CacheWanted) -> pb.CacheWanted:
    return pb.CacheWanted(engine=s.engine, rank=s.rank, dp=s.dp)


def _cache_wanted_from_pb(w: pb.CacheWanted) -> CacheWanted:
    return CacheWanted(engine=w.engine, rank=w.rank, dp=w.dp)


def _cache_ready_to_pb(s: CacheReady) -> pb.CacheReady:
    return pb.CacheReady(engine=s.engine, rank=s.rank, key=s.key, digest=s.digest)


def _cache_ready_from_pb(r: pb.CacheReady) -> CacheReady:
    return CacheReady(engine=r.engine, rank=r.rank, key=r.key, digest=r.digest)


def _cache_refused_to_pb(s: CacheRefused) -> pb.CacheRefused:
    return pb.CacheRefused(engine=s.engine, rank=s.rank, reason=s.reason)


def _cache_refused_from_pb(f: pb.CacheRefused) -> CacheRefused:
    return CacheRefused(engine=f.engine, rank=f.rank, reason=f.reason)


def _replay_ask_to_pb(s: ReplayAsk) -> pb.ReplayAsk:
    return pb.ReplayAsk(engine=s.engine, rank=s.rank, asker=_process_ref_to_pb(s.asker))


def _replay_ask_from_pb(a: pb.ReplayAsk) -> ReplayAsk:
    return ReplayAsk(
        engine=a.engine,
        rank=a.rank,
        asker=_process_ref_from_pb(a.asker) if a.HasField("asker") else None,
    )


def _replay_offer_to_pb(s: ReplayOffer) -> pb.ReplayOffer:
    return pb.ReplayOffer(
        engine=s.engine, rank=s.rank, handle=s.handle, to=_process_ref_to_pb(s.to)
    )


def _replay_offer_from_pb(o: pb.ReplayOffer) -> ReplayOffer:
    return ReplayOffer(engine=o.engine, rank=o.rank, handle=o.handle, to=_process_ref_from_pb(o.to))


# ------------------------------------------------------------------------------- the registry


@dataclass(frozen=True)
class _Kind:
    """One member of the proto's ``Signal.kind`` oneof and the dataclass it carries."""

    field: str
    type: type
    to_pb: Callable[[Any], Any]
    from_pb: Callable[[Any], Any]


_KINDS: tuple[_Kind, ...] = (
    _Kind("notice", Notice, _notice_to_pb, _notice_from_pb),
    _Kind("rank_lost", RankLost, _rank_lost_to_pb, _rank_lost_from_pb),
    _Kind("rank_joining", RankJoining, _rank_joining_to_pb, _rank_joining_from_pb),
    _Kind("membership_committed", MembershipCommitted, _membership_to_pb, _membership_from_pb),
    _Kind("change", Change, _change_to_pb, _change_from_pb),
    _Kind("register", Register, _register_to_pb, _register_from_pb),
    _Kind("join", Join, _join_to_pb, _join_from_pb),
    _Kind("goodbye", Goodbye, _goodbye_to_pb, _goodbye_from_pb),
    _Kind("abandoned", Abandoned, _abandoned_to_pb, _abandoned_from_pb),
    _Kind("cover_plan", CoverPlan, _cover_plan_to_pb, _cover_plan_from_pb),
    _Kind("group_formed", GroupFormed, _group_formed_to_pb, _group_formed_from_pb),
    _Kind("group_failed", GroupFailed, _group_failed_to_pb, _group_failed_from_pb),
    _Kind("group_released", GroupReleased, _group_released_to_pb, _group_released_from_pb),
    _Kind("group_ready", GroupReady, _group_ready_to_pb, _group_ready_from_pb),
    _Kind("group_id", GroupId, _group_id_to_pb, _group_id_from_pb),
    _Kind("release_leaked", ReleaseLeaked, _release_leaked_to_pb, _release_leaked_from_pb),
    _Kind("role_assigned", RoleAssigned, _role_assigned_to_pb, _role_assigned_from_pb),
    _Kind("dump_stacks", DumpStacks, _dump_stacks_to_pb, _dump_stacks_from_pb),
    _Kind("engine_policy", EnginePolicy, _engine_policy_to_pb, _engine_policy_from_pb),
    _Kind("engine_placement", EnginePlacement, _placement_to_pb, _placement_from_pb),
    _Kind("instance_state", InstanceState, _instance_state_to_pb, _instance_state_from_pb),
    _Kind("cache_wanted", CacheWanted, _cache_wanted_to_pb, _cache_wanted_from_pb),
    _Kind("cache_ready", CacheReady, _cache_ready_to_pb, _cache_ready_from_pb),
    _Kind("cache_refused", CacheRefused, _cache_refused_to_pb, _cache_refused_from_pb),
    _Kind("replay_ask", ReplayAsk, _replay_ask_to_pb, _replay_ask_from_pb),
    _Kind("replay_offer", ReplayOffer, _replay_offer_to_pb, _replay_offer_from_pb),
)
"""Every kind this build encodes and decodes, in the oneof's order. A kind the schema names and
this leaves out (``ack``, ``stacks_dumped``) is sent only, or by no process yet; its reader
passes it over as :class:`UnknownSignal`."""

_KIND_OF_TYPE: dict[type, _Kind] = {k.type: k for k in _KINDS}
_KIND_OF_FIELD: dict[str, _Kind] = {k.field: k for k in _KINDS}


def to_message(sig: Signal) -> pb.Signal:
    """The protobuf ``Signal`` for ``sig``."""
    kind = _KIND_OF_TYPE[type(sig)]
    return pb.Signal(**{kind.field: kind.to_pb(sig)})


def from_message(msg: pb.Signal) -> Signal:
    """The dataclass for a protobuf ``Signal``.

    Raises:
        UnknownSignal: for a kind the schema names that this codec does not decode.
        ValueError: for an empty signal or an unspecified reason.
    """
    field_name = msg.WhichOneof("kind")
    if field_name is None:
        raise ValueError("empty signal")
    kind = _KIND_OF_FIELD.get(field_name)
    if kind is None:
        # a kind the schema names that this codec does not decode: its reader passes it over
        raise UnknownSignal(f"a {field_name} signal, which this build does not decode")
    return kind.from_pb(getattr(msg, field_name))
