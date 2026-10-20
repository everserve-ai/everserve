from everserve.proto import signals_pb2 as _signals_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EngineState(_message.Message):
    __slots__ = ("engine", "epoch", "workers", "spares", "world_size", "any_spares", "lost", "departed", "slot", "plan_generation", "held", "standing", "serving", "plan", "undecided", "heads", "notices")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKERS_FIELD_NUMBER: _ClassVar[int]
    SPARES_FIELD_NUMBER: _ClassVar[int]
    WORLD_SIZE_FIELD_NUMBER: _ClassVar[int]
    ANY_SPARES_FIELD_NUMBER: _ClassVar[int]
    LOST_FIELD_NUMBER: _ClassVar[int]
    DEPARTED_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    PLAN_GENERATION_FIELD_NUMBER: _ClassVar[int]
    HELD_FIELD_NUMBER: _ClassVar[int]
    STANDING_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    UNDECIDED_FIELD_NUMBER: _ClassVar[int]
    HEADS_FIELD_NUMBER: _ClassVar[int]
    NOTICES_FIELD_NUMBER: _ClassVar[int]
    engine: str
    epoch: int
    workers: _containers.RepeatedScalarFieldContainer[int]
    spares: _containers.RepeatedScalarFieldContainer[int]
    world_size: int
    any_spares: _containers.RepeatedScalarFieldContainer[int]
    lost: _containers.RepeatedScalarFieldContainer[int]
    departed: _containers.RepeatedScalarFieldContainer[int]
    slot: _signals_pb2.SlotChoice
    plan_generation: int
    held: _containers.RepeatedCompositeFieldContainer[HeldGroup]
    standing: _containers.RepeatedCompositeFieldContainer[WorkerEvent]
    serving: _containers.RepeatedCompositeFieldContainer[WorkerEvent]
    plan: _signals_pb2.CoverPlan
    undecided: _containers.RepeatedCompositeFieldContainer[Undecided]
    heads: _containers.RepeatedCompositeFieldContainer[WorkerEvent]
    notices: _containers.RepeatedCompositeFieldContainer[_signals_pb2.Notice]
    def __init__(self, engine: _Optional[str] = ..., epoch: _Optional[int] = ..., workers: _Optional[_Iterable[int]] = ..., spares: _Optional[_Iterable[int]] = ..., world_size: _Optional[int] = ..., any_spares: _Optional[_Iterable[int]] = ..., lost: _Optional[_Iterable[int]] = ..., departed: _Optional[_Iterable[int]] = ..., slot: _Optional[_Union[_signals_pb2.SlotChoice, _Mapping]] = ..., plan_generation: _Optional[int] = ..., held: _Optional[_Iterable[_Union[HeldGroup, _Mapping]]] = ..., standing: _Optional[_Iterable[_Union[WorkerEvent, _Mapping]]] = ..., serving: _Optional[_Iterable[_Union[WorkerEvent, _Mapping]]] = ..., plan: _Optional[_Union[_signals_pb2.CoverPlan, _Mapping]] = ..., undecided: _Optional[_Iterable[_Union[Undecided, _Mapping]]] = ..., heads: _Optional[_Iterable[_Union[WorkerEvent, _Mapping]]] = ..., notices: _Optional[_Iterable[_Union[_signals_pb2.Notice, _Mapping]]] = ...) -> None: ...

class Undecided(_message.Message):
    __slots__ = ("rank", "connection")
    RANK_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_FIELD_NUMBER: _ClassVar[int]
    rank: int
    connection: ConnectionRef
    def __init__(self, rank: _Optional[int] = ..., connection: _Optional[_Union[ConnectionRef, _Mapping]] = ...) -> None: ...

class HeldGroup(_message.Message):
    __slots__ = ("member", "spans", "scope", "generation")
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    member: _signals_pb2.SetMember
    spans: _containers.RepeatedCompositeFieldContainer[_signals_pb2.SetMember]
    scope: _signals_pb2.Scope
    generation: int
    def __init__(self, member: _Optional[_Union[_signals_pb2.SetMember, _Mapping]] = ..., spans: _Optional[_Iterable[_Union[_signals_pb2.SetMember, _Mapping]]] = ..., scope: _Optional[_Union[_signals_pb2.Scope, str]] = ..., generation: _Optional[int] = ...) -> None: ...

class Device(_message.Message):
    __slots__ = ("index", "uuid", "memory_bytes", "domain")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    CLASS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    index: int
    uuid: str
    memory_bytes: int
    domain: str
    def __init__(self, index: _Optional[int] = ..., uuid: _Optional[str] = ..., memory_bytes: _Optional[int] = ..., domain: _Optional[str] = ..., **kwargs) -> None: ...

class DaemonHello(_message.Message):
    __slots__ = ("node", "engines", "devices", "mirror_replica_address", "heartbeat_ms", "protocol", "advertise_host", "term", "protocol_oldest")
    NODE_FIELD_NUMBER: _ClassVar[int]
    ENGINES_FIELD_NUMBER: _ClassVar[int]
    DEVICES_FIELD_NUMBER: _ClassVar[int]
    MIRROR_REPLICA_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_MS_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    ADVERTISE_HOST_FIELD_NUMBER: _ClassVar[int]
    TERM_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_OLDEST_FIELD_NUMBER: _ClassVar[int]
    node: str
    engines: _containers.RepeatedCompositeFieldContainer[EngineState]
    devices: _containers.RepeatedCompositeFieldContainer[Device]
    mirror_replica_address: str
    heartbeat_ms: int
    protocol: int
    advertise_host: str
    term: int
    protocol_oldest: int
    def __init__(self, node: _Optional[str] = ..., engines: _Optional[_Iterable[_Union[EngineState, _Mapping]]] = ..., devices: _Optional[_Iterable[_Union[Device, _Mapping]]] = ..., mirror_replica_address: _Optional[str] = ..., heartbeat_ms: _Optional[int] = ..., protocol: _Optional[int] = ..., advertise_host: _Optional[str] = ..., term: _Optional[int] = ..., protocol_oldest: _Optional[int] = ...) -> None: ...

class Heartbeat(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Bye(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WorkerEvent(_message.Message):
    __slots__ = ("engine", "kind", "rank", "world_size", "device", "compile_cache", "any_rank", "pid", "launcher_pid", "cause", "detail", "epoch", "identity", "connection", "host")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_UNSPECIFIED: _ClassVar[WorkerEvent.Kind]
        WORKER_REGISTERED: _ClassVar[WorkerEvent.Kind]
        WORKER_DEPARTED: _ClassVar[WorkerEvent.Kind]
        SPARE_REGISTERED: _ClassVar[WorkerEvent.Kind]
        SPARE_DEPARTED: _ClassVar[WorkerEvent.Kind]
        HEAD_REGISTERED: _ClassVar[WorkerEvent.Kind]
        HEAD_DEPARTED: _ClassVar[WorkerEvent.Kind]
        UNDECIDED_DEPARTED: _ClassVar[WorkerEvent.Kind]
    KIND_UNSPECIFIED: WorkerEvent.Kind
    WORKER_REGISTERED: WorkerEvent.Kind
    WORKER_DEPARTED: WorkerEvent.Kind
    SPARE_REGISTERED: WorkerEvent.Kind
    SPARE_DEPARTED: WorkerEvent.Kind
    HEAD_REGISTERED: WorkerEvent.Kind
    HEAD_DEPARTED: WorkerEvent.Kind
    UNDECIDED_DEPARTED: WorkerEvent.Kind
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    WORLD_SIZE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    COMPILE_CACHE_FIELD_NUMBER: _ClassVar[int]
    ANY_RANK_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    LAUNCHER_PID_FIELD_NUMBER: _ClassVar[int]
    CAUSE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    engine: str
    kind: WorkerEvent.Kind
    rank: int
    world_size: int
    device: int
    compile_cache: _signals_pb2.CompileCache
    any_rank: bool
    pid: int
    launcher_pid: int
    cause: _signals_pb2.StopCause
    detail: str
    epoch: int
    identity: _signals_pb2.SoftwareIdentity
    connection: ConnectionRef
    host: str
    def __init__(self, engine: _Optional[str] = ..., kind: _Optional[_Union[WorkerEvent.Kind, str]] = ..., rank: _Optional[int] = ..., world_size: _Optional[int] = ..., device: _Optional[int] = ..., compile_cache: _Optional[_Union[_signals_pb2.CompileCache, _Mapping]] = ..., any_rank: bool = ..., pid: _Optional[int] = ..., launcher_pid: _Optional[int] = ..., cause: _Optional[_Union[_signals_pb2.StopCause, str]] = ..., detail: _Optional[str] = ..., epoch: _Optional[int] = ..., identity: _Optional[_Union[_signals_pb2.SoftwareIdentity, _Mapping]] = ..., connection: _Optional[_Union[ConnectionRef, _Mapping]] = ..., host: _Optional[str] = ...) -> None: ...

class LostReport(_message.Message):
    __slots__ = ("engine", "lost")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    LOST_FIELD_NUMBER: _ClassVar[int]
    engine: str
    lost: _signals_pb2.RankLost
    def __init__(self, engine: _Optional[str] = ..., lost: _Optional[_Union[_signals_pb2.RankLost, _Mapping]] = ...) -> None: ...

class NoticeReport(_message.Message):
    __slots__ = ("engine", "notice")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    NOTICE_FIELD_NUMBER: _ClassVar[int]
    engine: str
    notice: _signals_pb2.Notice
    def __init__(self, engine: _Optional[str] = ..., notice: _Optional[_Union[_signals_pb2.Notice, _Mapping]] = ...) -> None: ...

class NodeNotice(_message.Message):
    __slots__ = ("deadline_ms", "reason", "source", "detail")
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    deadline_ms: int
    reason: _signals_pb2.Reason
    source: str
    detail: str
    def __init__(self, deadline_ms: _Optional[int] = ..., reason: _Optional[_Union[_signals_pb2.Reason, str]] = ..., source: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class Deliver(_message.Message):
    __slots__ = ("request_id", "engine", "change", "to_ranks", "to_spares", "take")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    TO_RANKS_FIELD_NUMBER: _ClassVar[int]
    TO_SPARES_FIELD_NUMBER: _ClassVar[int]
    TAKE_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    engine: str
    change: _signals_pb2.Change
    to_ranks: _containers.RepeatedScalarFieldContainer[int]
    to_spares: _containers.RepeatedScalarFieldContainer[int]
    take: _containers.RepeatedCompositeFieldContainer[_signals_pb2.Take]
    def __init__(self, request_id: _Optional[int] = ..., engine: _Optional[str] = ..., change: _Optional[_Union[_signals_pb2.Change, _Mapping]] = ..., to_ranks: _Optional[_Iterable[int]] = ..., to_spares: _Optional[_Iterable[int]] = ..., take: _Optional[_Iterable[_Union[_signals_pb2.Take, _Mapping]]] = ...) -> None: ...

class Delivered(_message.Message):
    __slots__ = ("request_id", "engine", "epoch", "acks", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    ACKS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    engine: str
    epoch: int
    acks: _containers.RepeatedCompositeFieldContainer[_signals_pb2.Ack]
    error: str
    def __init__(self, request_id: _Optional[int] = ..., engine: _Optional[str] = ..., epoch: _Optional[int] = ..., acks: _Optional[_Iterable[_Union[_signals_pb2.Ack, _Mapping]]] = ..., error: _Optional[str] = ...) -> None: ...

class Committed(_message.Message):
    __slots__ = ("engine", "membership", "outcome", "lost", "take")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    MEMBERSHIP_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    LOST_FIELD_NUMBER: _ClassVar[int]
    TAKE_FIELD_NUMBER: _ClassVar[int]
    engine: str
    membership: _signals_pb2.MembershipCommitted
    outcome: _signals_pb2.Outcome
    lost: _containers.RepeatedScalarFieldContainer[int]
    take: _containers.RepeatedCompositeFieldContainer[_signals_pb2.Take]
    def __init__(self, engine: _Optional[str] = ..., membership: _Optional[_Union[_signals_pb2.MembershipCommitted, _Mapping]] = ..., outcome: _Optional[_Union[_signals_pb2.Outcome, str]] = ..., lost: _Optional[_Iterable[int]] = ..., take: _Optional[_Iterable[_Union[_signals_pb2.Take, _Mapping]]] = ...) -> None: ...

class ForwardNotice(_message.Message):
    __slots__ = ("engine", "notice")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    NOTICE_FIELD_NUMBER: _ClassVar[int]
    engine: str
    notice: _signals_pb2.Notice
    def __init__(self, engine: _Optional[str] = ..., notice: _Optional[_Union[_signals_pb2.Notice, _Mapping]] = ...) -> None: ...

class ReplicateTargets(_message.Message):
    __slots__ = ("engine", "rank", "targets")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    targets: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., targets: _Optional[_Iterable[str]] = ...) -> None: ...

class ExpectReplication(_message.Message):
    __slots__ = ("engine", "rank", "from_node", "epoch")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    FROM_NODE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    from_node: str
    epoch: int
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., from_node: _Optional[str] = ..., epoch: _Optional[int] = ...) -> None: ...

class DaemonToCoord(_message.Message):
    __slots__ = ("hello", "worker_event", "lost_report", "notice_report", "delivered", "node_notice", "engine_slot", "heartbeat", "bye", "join_report", "stacks_sent", "cover_report", "role_asked", "group_id", "release_leaked", "replay_ask", "replay_offer")
    HELLO_FIELD_NUMBER: _ClassVar[int]
    WORKER_EVENT_FIELD_NUMBER: _ClassVar[int]
    LOST_REPORT_FIELD_NUMBER: _ClassVar[int]
    NOTICE_REPORT_FIELD_NUMBER: _ClassVar[int]
    DELIVERED_FIELD_NUMBER: _ClassVar[int]
    NODE_NOTICE_FIELD_NUMBER: _ClassVar[int]
    ENGINE_SLOT_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    BYE_FIELD_NUMBER: _ClassVar[int]
    JOIN_REPORT_FIELD_NUMBER: _ClassVar[int]
    STACKS_SENT_FIELD_NUMBER: _ClassVar[int]
    COVER_REPORT_FIELD_NUMBER: _ClassVar[int]
    ROLE_ASKED_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASE_LEAKED_FIELD_NUMBER: _ClassVar[int]
    REPLAY_ASK_FIELD_NUMBER: _ClassVar[int]
    REPLAY_OFFER_FIELD_NUMBER: _ClassVar[int]
    hello: DaemonHello
    worker_event: WorkerEvent
    lost_report: LostReport
    notice_report: NoticeReport
    delivered: Delivered
    node_notice: NodeNotice
    engine_slot: EngineSlot
    heartbeat: Heartbeat
    bye: Bye
    join_report: JoinReport
    stacks_sent: StacksSent
    cover_report: CoverReport
    role_asked: RoleAsked
    group_id: _signals_pb2.GroupId
    release_leaked: ReleaseLeakedAt
    replay_ask: _signals_pb2.ReplayAsk
    replay_offer: _signals_pb2.ReplayOffer
    def __init__(self, hello: _Optional[_Union[DaemonHello, _Mapping]] = ..., worker_event: _Optional[_Union[WorkerEvent, _Mapping]] = ..., lost_report: _Optional[_Union[LostReport, _Mapping]] = ..., notice_report: _Optional[_Union[NoticeReport, _Mapping]] = ..., delivered: _Optional[_Union[Delivered, _Mapping]] = ..., node_notice: _Optional[_Union[NodeNotice, _Mapping]] = ..., engine_slot: _Optional[_Union[EngineSlot, _Mapping]] = ..., heartbeat: _Optional[_Union[Heartbeat, _Mapping]] = ..., bye: _Optional[_Union[Bye, _Mapping]] = ..., join_report: _Optional[_Union[JoinReport, _Mapping]] = ..., stacks_sent: _Optional[_Union[StacksSent, _Mapping]] = ..., cover_report: _Optional[_Union[CoverReport, _Mapping]] = ..., role_asked: _Optional[_Union[RoleAsked, _Mapping]] = ..., group_id: _Optional[_Union[_signals_pb2.GroupId, _Mapping]] = ..., release_leaked: _Optional[_Union[ReleaseLeakedAt, _Mapping]] = ..., replay_ask: _Optional[_Union[_signals_pb2.ReplayAsk, _Mapping]] = ..., replay_offer: _Optional[_Union[_signals_pb2.ReplayOffer, _Mapping]] = ...) -> None: ...

class ReleaseLeakedAt(_message.Message):
    __slots__ = ("connection", "leak")
    CONNECTION_FIELD_NUMBER: _ClassVar[int]
    LEAK_FIELD_NUMBER: _ClassVar[int]
    connection: ConnectionRef
    leak: _signals_pb2.ReleaseLeaked
    def __init__(self, connection: _Optional[_Union[ConnectionRef, _Mapping]] = ..., leak: _Optional[_Union[_signals_pb2.ReleaseLeaked, _Mapping]] = ...) -> None: ...

class RoleAsked(_message.Message):
    __slots__ = ("request_id", "engine", "rank", "connection", "stage", "host", "identity")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    engine: str
    rank: int
    connection: ConnectionRef
    stage: _containers.RepeatedScalarFieldContainer[int]
    host: str
    identity: _signals_pb2.SoftwareIdentity
    def __init__(self, request_id: _Optional[int] = ..., engine: _Optional[str] = ..., rank: _Optional[int] = ..., connection: _Optional[_Union[ConnectionRef, _Mapping]] = ..., stage: _Optional[_Iterable[int]] = ..., host: _Optional[str] = ..., identity: _Optional[_Union[_signals_pb2.SoftwareIdentity, _Mapping]] = ...) -> None: ...

class ConnectionRef(_message.Message):
    __slots__ = ("instance", "serial", "pid")
    INSTANCE_FIELD_NUMBER: _ClassVar[int]
    SERIAL_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    instance: int
    serial: int
    pid: int
    def __init__(self, instance: _Optional[int] = ..., serial: _Optional[int] = ..., pid: _Optional[int] = ...) -> None: ...

class RoleAnswer(_message.Message):
    __slots__ = ("request_id", "engine", "rank", "epoch", "role", "refusal")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    REFUSAL_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    engine: str
    rank: int
    epoch: int
    role: _signals_pb2.Role
    refusal: str
    def __init__(self, request_id: _Optional[int] = ..., engine: _Optional[str] = ..., rank: _Optional[int] = ..., epoch: _Optional[int] = ..., role: _Optional[_Union[_signals_pb2.Role, str]] = ..., refusal: _Optional[str] = ...) -> None: ...

class CoverReport(_message.Message):
    __slots__ = ("formed", "failed", "released", "ready")
    FORMED_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    RELEASED_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    formed: _signals_pb2.GroupFormed
    failed: _signals_pb2.GroupFailed
    released: _signals_pb2.GroupReleased
    ready: _signals_pb2.GroupReady
    def __init__(self, formed: _Optional[_Union[_signals_pb2.GroupFormed, _Mapping]] = ..., failed: _Optional[_Union[_signals_pb2.GroupFailed, _Mapping]] = ..., released: _Optional[_Union[_signals_pb2.GroupReleased, _Mapping]] = ..., ready: _Optional[_Union[_signals_pb2.GroupReady, _Mapping]] = ...) -> None: ...

class JoinReport(_message.Message):
    __slots__ = ("engine", "join", "from_client")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    JOIN_FIELD_NUMBER: _ClassVar[int]
    FROM_CLIENT_FIELD_NUMBER: _ClassVar[int]
    engine: str
    join: _signals_pb2.Join
    from_client: bool
    def __init__(self, engine: _Optional[str] = ..., join: _Optional[_Union[_signals_pb2.Join, _Mapping]] = ..., from_client: bool = ...) -> None: ...

class EngineSlot(_message.Message):
    __slots__ = ("engine", "slot")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    engine: str
    slot: _signals_pb2.SlotChoice
    def __init__(self, engine: _Optional[str] = ..., slot: _Optional[_Union[_signals_pb2.SlotChoice, _Mapping]] = ...) -> None: ...

class CoordToDaemon(_message.Message):
    __slots__ = ("term", "deliver", "committed", "forward_notice", "replicate_targets", "launch_spare", "forward_join", "expect_replication", "stacks", "abandon", "cover_plan", "forward_change", "role_answer", "engine_policy", "engine_placement", "instance_state", "group_id", "replay_ask", "replay_offer")
    TERM_FIELD_NUMBER: _ClassVar[int]
    DELIVER_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_FIELD_NUMBER: _ClassVar[int]
    FORWARD_NOTICE_FIELD_NUMBER: _ClassVar[int]
    REPLICATE_TARGETS_FIELD_NUMBER: _ClassVar[int]
    LAUNCH_SPARE_FIELD_NUMBER: _ClassVar[int]
    FORWARD_JOIN_FIELD_NUMBER: _ClassVar[int]
    EXPECT_REPLICATION_FIELD_NUMBER: _ClassVar[int]
    STACKS_FIELD_NUMBER: _ClassVar[int]
    ABANDON_FIELD_NUMBER: _ClassVar[int]
    COVER_PLAN_FIELD_NUMBER: _ClassVar[int]
    FORWARD_CHANGE_FIELD_NUMBER: _ClassVar[int]
    ROLE_ANSWER_FIELD_NUMBER: _ClassVar[int]
    ENGINE_POLICY_FIELD_NUMBER: _ClassVar[int]
    ENGINE_PLACEMENT_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_STATE_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    REPLAY_ASK_FIELD_NUMBER: _ClassVar[int]
    REPLAY_OFFER_FIELD_NUMBER: _ClassVar[int]
    term: int
    deliver: Deliver
    committed: Committed
    forward_notice: ForwardNotice
    replicate_targets: ReplicateTargets
    launch_spare: LaunchSpare
    forward_join: ForwardJoin
    expect_replication: ExpectReplication
    stacks: Stacks
    abandon: Abandon
    cover_plan: _signals_pb2.CoverPlan
    forward_change: ForwardChange
    role_answer: RoleAnswer
    engine_policy: _signals_pb2.EnginePolicy
    engine_placement: _signals_pb2.EnginePlacement
    instance_state: _signals_pb2.InstanceState
    group_id: _signals_pb2.GroupId
    replay_ask: _signals_pb2.ReplayAsk
    replay_offer: _signals_pb2.ReplayOffer
    def __init__(self, term: _Optional[int] = ..., deliver: _Optional[_Union[Deliver, _Mapping]] = ..., committed: _Optional[_Union[Committed, _Mapping]] = ..., forward_notice: _Optional[_Union[ForwardNotice, _Mapping]] = ..., replicate_targets: _Optional[_Union[ReplicateTargets, _Mapping]] = ..., launch_spare: _Optional[_Union[LaunchSpare, _Mapping]] = ..., forward_join: _Optional[_Union[ForwardJoin, _Mapping]] = ..., expect_replication: _Optional[_Union[ExpectReplication, _Mapping]] = ..., stacks: _Optional[_Union[Stacks, _Mapping]] = ..., abandon: _Optional[_Union[Abandon, _Mapping]] = ..., cover_plan: _Optional[_Union[_signals_pb2.CoverPlan, _Mapping]] = ..., forward_change: _Optional[_Union[ForwardChange, _Mapping]] = ..., role_answer: _Optional[_Union[RoleAnswer, _Mapping]] = ..., engine_policy: _Optional[_Union[_signals_pb2.EnginePolicy, _Mapping]] = ..., engine_placement: _Optional[_Union[_signals_pb2.EnginePlacement, _Mapping]] = ..., instance_state: _Optional[_Union[_signals_pb2.InstanceState, _Mapping]] = ..., group_id: _Optional[_Union[_signals_pb2.GroupId, _Mapping]] = ..., replay_ask: _Optional[_Union[_signals_pb2.ReplayAsk, _Mapping]] = ..., replay_offer: _Optional[_Union[_signals_pb2.ReplayOffer, _Mapping]] = ...) -> None: ...

class ForwardChange(_message.Message):
    __slots__ = ("engine", "change", "to_ranks")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    TO_RANKS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    change: _signals_pb2.Change
    to_ranks: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, engine: _Optional[str] = ..., change: _Optional[_Union[_signals_pb2.Change, _Mapping]] = ..., to_ranks: _Optional[_Iterable[int]] = ...) -> None: ...

class Abandon(_message.Message):
    __slots__ = ("engine", "epoch", "detail")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    engine: str
    epoch: int
    detail: str
    def __init__(self, engine: _Optional[str] = ..., epoch: _Optional[int] = ..., detail: _Optional[str] = ...) -> None: ...

class LaunchSpare(_message.Message):
    __slots__ = ("engine", "rank", "device", "command", "env", "pool", "engines")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    POOL_FIELD_NUMBER: _ClassVar[int]
    ENGINES_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    device: int
    command: _containers.RepeatedScalarFieldContainer[str]
    env: _containers.ScalarMap[str, str]
    pool: str
    engines: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., device: _Optional[int] = ..., command: _Optional[_Iterable[str]] = ..., env: _Optional[_Mapping[str, str]] = ..., pool: _Optional[str] = ..., engines: _Optional[_Iterable[str]] = ...) -> None: ...

class ForwardJoin(_message.Message):
    __slots__ = ("engine", "join", "to_clients")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    JOIN_FIELD_NUMBER: _ClassVar[int]
    TO_CLIENTS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    join: _signals_pb2.Join
    to_clients: bool
    def __init__(self, engine: _Optional[str] = ..., join: _Optional[_Union[_signals_pb2.Join, _Mapping]] = ..., to_clients: bool = ...) -> None: ...

class Stacks(_message.Message):
    __slots__ = ("request_id", "engine", "rank", "device", "head")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    engine: str
    rank: int
    device: int
    head: bool
    def __init__(self, request_id: _Optional[int] = ..., engine: _Optional[str] = ..., rank: _Optional[int] = ..., device: _Optional[int] = ..., head: bool = ...) -> None: ...

class StacksSent(_message.Message):
    __slots__ = ("request_id", "pid", "error", "answered", "text")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ANSWERED_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    pid: int
    error: str
    answered: bool
    text: str
    def __init__(self, request_id: _Optional[int] = ..., pid: _Optional[int] = ..., error: _Optional[str] = ..., answered: bool = ..., text: _Optional[str] = ...) -> None: ...

class LeaseRecord(_message.Message):
    __slots__ = ("holder", "term", "address", "renewed_at_ms")
    HOLDER_FIELD_NUMBER: _ClassVar[int]
    TERM_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    RENEWED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    holder: str
    term: int
    address: str
    renewed_at_ms: int
    def __init__(self, holder: _Optional[str] = ..., term: _Optional[int] = ..., address: _Optional[str] = ..., renewed_at_ms: _Optional[int] = ...) -> None: ...
