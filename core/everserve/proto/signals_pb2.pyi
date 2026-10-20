from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Protocol(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROTOCOL_UNSPECIFIED: _ClassVar[Protocol]
    PROTOCOL_VERSION: _ClassVar[Protocol]

class ProtocolOldest(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROTOCOL_OLDEST_UNSPECIFIED: _ClassVar[ProtocolOldest]
    PROTOCOL_OLDEST: _ClassVar[ProtocolOldest]

class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REASON_UNSPECIFIED: _ClassVar[Reason]
    REASON_NOTICE: _ClassVar[Reason]
    REASON_FAULT: _ClassVar[Reason]
    REASON_OPERATOR: _ClassVar[Reason]
    REASON_RETIRED: _ClassVar[Reason]

class Outcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OUTCOME_UNSPECIFIED: _ClassVar[Outcome]
    OUTCOME_REPLACE: _ClassVar[Outcome]
    OUTCOME_SHRINK: _ClassVar[Outcome]
    OUTCOME_REBUILD: _ClassVar[Outcome]

class Role(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ROLE_WORKER: _ClassVar[Role]
    ROLE_SPARE: _ClassVar[Role]
    ROLE_CLIENT: _ClassVar[Role]
    ROLE_HEAD: _ClassVar[Role]
    ROLE_UNDECIDED: _ClassVar[Role]

class StopCause(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STOP_CAUSE_UNSPECIFIED: _ClassVar[StopCause]
    STOP_CAUSE_STOPPED: _ClassVar[StopCause]
    STOP_CAUSE_FAILED: _ClassVar[StopCause]
    STOP_CAUSE_LOST: _ClassVar[StopCause]
    STOP_CAUSE_REFUSED: _ClassVar[StopCause]

class InstancePhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INSTANCE_PHASE_UNSPECIFIED: _ClassVar[InstancePhase]
    INSTANCE_STARTING: _ClassVar[InstancePhase]
    INSTANCE_SERVING: _ClassVar[InstancePhase]
    INSTANCE_RECOVERING: _ClassVar[InstancePhase]
    INSTANCE_DEGRADED: _ClassVar[InstancePhase]
    INSTANCE_FAILED: _ClassVar[InstancePhase]

class Scope(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SCOPE_UNSPECIFIED: _ClassVar[Scope]
    SCOPE_DOMAIN: _ClassVar[Scope]
    SCOPE_FABRIC: _ClassVar[Scope]
PROTOCOL_UNSPECIFIED: Protocol
PROTOCOL_VERSION: Protocol
PROTOCOL_OLDEST_UNSPECIFIED: ProtocolOldest
PROTOCOL_OLDEST: ProtocolOldest
REASON_UNSPECIFIED: Reason
REASON_NOTICE: Reason
REASON_FAULT: Reason
REASON_OPERATOR: Reason
REASON_RETIRED: Reason
OUTCOME_UNSPECIFIED: Outcome
OUTCOME_REPLACE: Outcome
OUTCOME_SHRINK: Outcome
OUTCOME_REBUILD: Outcome
ROLE_WORKER: Role
ROLE_SPARE: Role
ROLE_CLIENT: Role
ROLE_HEAD: Role
ROLE_UNDECIDED: Role
STOP_CAUSE_UNSPECIFIED: StopCause
STOP_CAUSE_STOPPED: StopCause
STOP_CAUSE_FAILED: StopCause
STOP_CAUSE_LOST: StopCause
STOP_CAUSE_REFUSED: StopCause
INSTANCE_PHASE_UNSPECIFIED: InstancePhase
INSTANCE_STARTING: InstancePhase
INSTANCE_SERVING: InstancePhase
INSTANCE_RECOVERING: InstancePhase
INSTANCE_DEGRADED: InstancePhase
INSTANCE_FAILED: InstancePhase
SCOPE_UNSPECIFIED: Scope
SCOPE_DOMAIN: Scope
SCOPE_FABRIC: Scope

class Member(_message.Message):
    __slots__ = ("node", "device")
    NODE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    node: str
    device: int
    def __init__(self, node: _Optional[str] = ..., device: _Optional[int] = ...) -> None: ...

class Notice(_message.Message):
    __slots__ = ("rank", "deadline_ms", "reason")
    RANK_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    rank: int
    deadline_ms: int
    reason: Reason
    def __init__(self, rank: _Optional[int] = ..., deadline_ms: _Optional[int] = ..., reason: _Optional[_Union[Reason, str]] = ...) -> None: ...

class RankLost(_message.Message):
    __slots__ = ("ranks", "reason", "detail", "epoch")
    RANKS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    ranks: _containers.RepeatedScalarFieldContainer[int]
    reason: Reason
    detail: str
    epoch: int
    def __init__(self, ranks: _Optional[_Iterable[int]] = ..., reason: _Optional[_Union[Reason, str]] = ..., detail: _Optional[str] = ..., epoch: _Optional[int] = ...) -> None: ...

class RankJoining(_message.Message):
    __slots__ = ("rank", "member")
    RANK_FIELD_NUMBER: _ClassVar[int]
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    rank: int
    member: Member
    def __init__(self, rank: _Optional[int] = ..., member: _Optional[_Union[Member, _Mapping]] = ...) -> None: ...

class MembershipCommitted(_message.Message):
    __slots__ = ("epoch", "active_ranks", "world_size", "pause_ms", "lost", "reported_at_us", "delivered_at_us", "acknowledged_at_us", "committed_at_us", "acks", "members")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_RANKS_FIELD_NUMBER: _ClassVar[int]
    WORLD_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAUSE_MS_FIELD_NUMBER: _ClassVar[int]
    LOST_FIELD_NUMBER: _ClassVar[int]
    REPORTED_AT_US_FIELD_NUMBER: _ClassVar[int]
    DELIVERED_AT_US_FIELD_NUMBER: _ClassVar[int]
    ACKNOWLEDGED_AT_US_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_AT_US_FIELD_NUMBER: _ClassVar[int]
    ACKS_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    active_ranks: _containers.RepeatedScalarFieldContainer[int]
    world_size: int
    pause_ms: int
    lost: _containers.RepeatedScalarFieldContainer[int]
    reported_at_us: int
    delivered_at_us: int
    acknowledged_at_us: int
    committed_at_us: int
    acks: _containers.RepeatedCompositeFieldContainer[Ack]
    members: _containers.RepeatedCompositeFieldContainer[RankPlacement]
    def __init__(self, epoch: _Optional[int] = ..., active_ranks: _Optional[_Iterable[int]] = ..., world_size: _Optional[int] = ..., pause_ms: _Optional[int] = ..., lost: _Optional[_Iterable[int]] = ..., reported_at_us: _Optional[int] = ..., delivered_at_us: _Optional[int] = ..., acknowledged_at_us: _Optional[int] = ..., committed_at_us: _Optional[int] = ..., acks: _Optional[_Iterable[_Union[Ack, _Mapping]]] = ..., members: _Optional[_Iterable[_Union[RankPlacement, _Mapping]]] = ...) -> None: ...

class RankPlacement(_message.Message):
    __slots__ = ("rank", "node", "device", "host", "compile_cache", "cache_holders")
    RANK_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    COMPILE_CACHE_FIELD_NUMBER: _ClassVar[int]
    CACHE_HOLDERS_FIELD_NUMBER: _ClassVar[int]
    rank: int
    node: str
    device: int
    host: str
    compile_cache: CompileCache
    cache_holders: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, rank: _Optional[int] = ..., node: _Optional[str] = ..., device: _Optional[int] = ..., host: _Optional[str] = ..., compile_cache: _Optional[_Union[CompileCache, _Mapping]] = ..., cache_holders: _Optional[_Iterable[str]] = ...) -> None: ...

class SparePlacement(_message.Message):
    __slots__ = ("for_rank", "node", "device", "host")
    FOR_RANK_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    for_rank: int
    node: str
    device: int
    host: str
    def __init__(self, for_rank: _Optional[int] = ..., node: _Optional[str] = ..., device: _Optional[int] = ..., host: _Optional[str] = ...) -> None: ...

class EnginePlacement(_message.Message):
    __slots__ = ("engine", "epoch", "members", "spares")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    SPARES_FIELD_NUMBER: _ClassVar[int]
    engine: str
    epoch: int
    members: _containers.RepeatedCompositeFieldContainer[RankPlacement]
    spares: _containers.RepeatedCompositeFieldContainer[SparePlacement]
    def __init__(self, engine: _Optional[str] = ..., epoch: _Optional[int] = ..., members: _Optional[_Iterable[_Union[RankPlacement, _Mapping]]] = ..., spares: _Optional[_Iterable[_Union[SparePlacement, _Mapping]]] = ...) -> None: ...

class Change(_message.Message):
    __slots__ = ("lost", "outcome", "epoch", "delivered_at_us", "take", "departing")
    LOST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    DELIVERED_AT_US_FIELD_NUMBER: _ClassVar[int]
    TAKE_FIELD_NUMBER: _ClassVar[int]
    DEPARTING_FIELD_NUMBER: _ClassVar[int]
    lost: _containers.RepeatedScalarFieldContainer[int]
    outcome: Outcome
    epoch: int
    delivered_at_us: int
    take: _containers.RepeatedCompositeFieldContainer[Take]
    departing: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, lost: _Optional[_Iterable[int]] = ..., outcome: _Optional[_Union[Outcome, str]] = ..., epoch: _Optional[int] = ..., delivered_at_us: _Optional[int] = ..., take: _Optional[_Iterable[_Union[Take, _Mapping]]] = ..., departing: _Optional[_Iterable[int]] = ...) -> None: ...

class Take(_message.Message):
    __slots__ = ("rank", "node", "device", "compile_cache")
    RANK_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    COMPILE_CACHE_FIELD_NUMBER: _ClassVar[int]
    rank: int
    node: str
    device: int
    compile_cache: CompileCache
    def __init__(self, rank: _Optional[int] = ..., node: _Optional[str] = ..., device: _Optional[int] = ..., compile_cache: _Optional[_Union[CompileCache, _Mapping]] = ...) -> None: ...

class CacheWanted(_message.Message):
    __slots__ = ("engine", "rank", "dp")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    DP_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    dp: int
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., dp: _Optional[int] = ...) -> None: ...

class CacheReady(_message.Message):
    __slots__ = ("engine", "rank", "key", "digest")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    key: str
    digest: str
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., key: _Optional[str] = ..., digest: _Optional[str] = ...) -> None: ...

class CacheRefused(_message.Message):
    __slots__ = ("engine", "rank", "reason")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    reason: str
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...

class Register(_message.Message):
    __slots__ = ("engine", "rank", "member", "role", "slot", "protocol", "compile_cache", "any_rank", "launcher_pid", "identity", "protocol_oldest", "stage", "host")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    COMPILE_CACHE_FIELD_NUMBER: _ClassVar[int]
    ANY_RANK_FIELD_NUMBER: _ClassVar[int]
    LAUNCHER_PID_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_OLDEST_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    member: Member
    role: Role
    slot: SlotChoice
    protocol: int
    compile_cache: CompileCache
    any_rank: bool
    launcher_pid: int
    identity: SoftwareIdentity
    protocol_oldest: int
    stage: _containers.RepeatedScalarFieldContainer[int]
    host: str
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., member: _Optional[_Union[Member, _Mapping]] = ..., role: _Optional[_Union[Role, str]] = ..., slot: _Optional[_Union[SlotChoice, _Mapping]] = ..., protocol: _Optional[int] = ..., compile_cache: _Optional[_Union[CompileCache, _Mapping]] = ..., any_rank: bool = ..., launcher_pid: _Optional[int] = ..., identity: _Optional[_Union[SoftwareIdentity, _Mapping]] = ..., protocol_oldest: _Optional[int] = ..., stage: _Optional[_Iterable[int]] = ..., host: _Optional[str] = ...) -> None: ...

class SoftwareIdentity(_message.Message):
    __slots__ = ("engine", "engine_commit", "adapter_commit", "adapter_dirty", "torch_version", "cuda_version", "driver_version", "nccl_version", "model", "model_revision", "checkpoint_digest", "quantization", "model_dtype", "kv_format", "layout", "step", "guarantee", "serves_changes")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    ENGINE_COMMIT_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_COMMIT_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_DIRTY_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    CUDA_VERSION_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    NCCL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MODEL_REVISION_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    MODEL_DTYPE_FIELD_NUMBER: _ClassVar[int]
    KV_FORMAT_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    GUARANTEE_FIELD_NUMBER: _ClassVar[int]
    SERVES_CHANGES_FIELD_NUMBER: _ClassVar[int]
    engine: str
    engine_commit: str
    adapter_commit: str
    adapter_dirty: bool
    torch_version: str
    cuda_version: str
    driver_version: str
    nccl_version: str
    model: str
    model_revision: str
    checkpoint_digest: str
    quantization: str
    model_dtype: str
    kv_format: KvFormat
    layout: ParallelLayout
    step: StepConfig
    guarantee: str
    serves_changes: bool
    def __init__(self, engine: _Optional[str] = ..., engine_commit: _Optional[str] = ..., adapter_commit: _Optional[str] = ..., adapter_dirty: bool = ..., torch_version: _Optional[str] = ..., cuda_version: _Optional[str] = ..., driver_version: _Optional[str] = ..., nccl_version: _Optional[str] = ..., model: _Optional[str] = ..., model_revision: _Optional[str] = ..., checkpoint_digest: _Optional[str] = ..., quantization: _Optional[str] = ..., model_dtype: _Optional[str] = ..., kv_format: _Optional[_Union[KvFormat, _Mapping]] = ..., layout: _Optional[_Union[ParallelLayout, _Mapping]] = ..., step: _Optional[_Union[StepConfig, _Mapping]] = ..., guarantee: _Optional[str] = ..., serves_changes: bool = ...) -> None: ...

class StepConfig(_message.Message):
    __slots__ = ("digest", "fields")
    class FieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    digest: str
    fields: _containers.ScalarMap[str, str]
    def __init__(self, digest: _Optional[str] = ..., fields: _Optional[_Mapping[str, str]] = ...) -> None: ...

class KvFormat(_message.Message):
    __slots__ = ("dtype", "block_size", "layout_hash")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_HASH_FIELD_NUMBER: _ClassVar[int]
    dtype: str
    block_size: int
    layout_hash: str
    def __init__(self, dtype: _Optional[str] = ..., block_size: _Optional[int] = ..., layout_hash: _Optional[str] = ...) -> None: ...

class ParallelLayout(_message.Message):
    __slots__ = ("tensor", "pipeline", "data", "expert", "context")
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    EXPERT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    tensor: int
    pipeline: int
    data: int
    expert: int
    context: int
    def __init__(self, tensor: _Optional[int] = ..., pipeline: _Optional[int] = ..., data: _Optional[int] = ..., expert: _Optional[int] = ..., context: _Optional[int] = ...) -> None: ...

class Goodbye(_message.Message):
    __slots__ = ("cause", "detail")
    CAUSE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    cause: StopCause
    detail: str
    def __init__(self, cause: _Optional[_Union[StopCause, str]] = ..., detail: _Optional[str] = ...) -> None: ...

class Abandoned(_message.Message):
    __slots__ = ("epoch", "detail")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    detail: str
    def __init__(self, epoch: _Optional[int] = ..., detail: _Optional[str] = ...) -> None: ...

class CompileCache(_message.Message):
    __slots__ = ("key", "digest")
    KEY_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    key: str
    digest: str
    def __init__(self, key: _Optional[str] = ..., digest: _Optional[str] = ...) -> None: ...

class SlotChoice(_message.Message):
    __slots__ = ("variant", "how", "detail", "seconds")
    class Variant(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        VARIANT_UNSPECIFIED: _ClassVar[SlotChoice.Variant]
        VARIANT_P2P: _ClassVar[SlotChoice.Variant]
        VARIANT_HOST: _ClassVar[SlotChoice.Variant]
    VARIANT_UNSPECIFIED: SlotChoice.Variant
    VARIANT_P2P: SlotChoice.Variant
    VARIANT_HOST: SlotChoice.Variant
    class How(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        HOW_UNSPECIFIED: _ClassVar[SlotChoice.How]
        HOW_OVERRIDE: _ClassVar[SlotChoice.How]
        HOW_NVML: _ClassVar[SlotChoice.How]
        HOW_CACHE: _ClassVar[SlotChoice.How]
        HOW_PROBE: _ClassVar[SlotChoice.How]
    HOW_UNSPECIFIED: SlotChoice.How
    HOW_OVERRIDE: SlotChoice.How
    HOW_NVML: SlotChoice.How
    HOW_CACHE: SlotChoice.How
    HOW_PROBE: SlotChoice.How
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    HOW_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    SECONDS_FIELD_NUMBER: _ClassVar[int]
    variant: SlotChoice.Variant
    how: SlotChoice.How
    detail: str
    seconds: float
    def __init__(self, variant: _Optional[_Union[SlotChoice.Variant, str]] = ..., how: _Optional[_Union[SlotChoice.How, str]] = ..., detail: _Optional[str] = ..., seconds: _Optional[float] = ...) -> None: ...

class Signal(_message.Message):
    __slots__ = ("notice", "rank_lost", "rank_joining", "membership_committed", "change", "register", "ack", "join", "goodbye", "abandoned", "cover_plan", "group_formed", "group_failed", "group_released", "role_assigned", "dump_stacks", "engine_policy", "engine_placement", "instance_state", "stacks_dumped", "group_id", "cache_wanted", "cache_ready", "cache_refused", "release_leaked", "group_ready", "replay_ask", "replay_offer")
    NOTICE_FIELD_NUMBER: _ClassVar[int]
    RANK_LOST_FIELD_NUMBER: _ClassVar[int]
    RANK_JOINING_FIELD_NUMBER: _ClassVar[int]
    MEMBERSHIP_COMMITTED_FIELD_NUMBER: _ClassVar[int]
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    REGISTER_FIELD_NUMBER: _ClassVar[int]
    ACK_FIELD_NUMBER: _ClassVar[int]
    JOIN_FIELD_NUMBER: _ClassVar[int]
    GOODBYE_FIELD_NUMBER: _ClassVar[int]
    ABANDONED_FIELD_NUMBER: _ClassVar[int]
    COVER_PLAN_FIELD_NUMBER: _ClassVar[int]
    GROUP_FORMED_FIELD_NUMBER: _ClassVar[int]
    GROUP_FAILED_FIELD_NUMBER: _ClassVar[int]
    GROUP_RELEASED_FIELD_NUMBER: _ClassVar[int]
    ROLE_ASSIGNED_FIELD_NUMBER: _ClassVar[int]
    DUMP_STACKS_FIELD_NUMBER: _ClassVar[int]
    ENGINE_POLICY_FIELD_NUMBER: _ClassVar[int]
    ENGINE_PLACEMENT_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_STATE_FIELD_NUMBER: _ClassVar[int]
    STACKS_DUMPED_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    CACHE_WANTED_FIELD_NUMBER: _ClassVar[int]
    CACHE_READY_FIELD_NUMBER: _ClassVar[int]
    CACHE_REFUSED_FIELD_NUMBER: _ClassVar[int]
    RELEASE_LEAKED_FIELD_NUMBER: _ClassVar[int]
    GROUP_READY_FIELD_NUMBER: _ClassVar[int]
    REPLAY_ASK_FIELD_NUMBER: _ClassVar[int]
    REPLAY_OFFER_FIELD_NUMBER: _ClassVar[int]
    notice: Notice
    rank_lost: RankLost
    rank_joining: RankJoining
    membership_committed: MembershipCommitted
    change: Change
    register: Register
    ack: Ack
    join: Join
    goodbye: Goodbye
    abandoned: Abandoned
    cover_plan: CoverPlan
    group_formed: GroupFormed
    group_failed: GroupFailed
    group_released: GroupReleased
    role_assigned: RoleAssigned
    dump_stacks: DumpStacks
    engine_policy: EnginePolicy
    engine_placement: EnginePlacement
    instance_state: InstanceState
    stacks_dumped: StacksDumped
    group_id: GroupId
    cache_wanted: CacheWanted
    cache_ready: CacheReady
    cache_refused: CacheRefused
    release_leaked: ReleaseLeaked
    group_ready: GroupReady
    replay_ask: ReplayAsk
    replay_offer: ReplayOffer
    def __init__(self, notice: _Optional[_Union[Notice, _Mapping]] = ..., rank_lost: _Optional[_Union[RankLost, _Mapping]] = ..., rank_joining: _Optional[_Union[RankJoining, _Mapping]] = ..., membership_committed: _Optional[_Union[MembershipCommitted, _Mapping]] = ..., change: _Optional[_Union[Change, _Mapping]] = ..., register: _Optional[_Union[Register, _Mapping]] = ..., ack: _Optional[_Union[Ack, _Mapping]] = ..., join: _Optional[_Union[Join, _Mapping]] = ..., goodbye: _Optional[_Union[Goodbye, _Mapping]] = ..., abandoned: _Optional[_Union[Abandoned, _Mapping]] = ..., cover_plan: _Optional[_Union[CoverPlan, _Mapping]] = ..., group_formed: _Optional[_Union[GroupFormed, _Mapping]] = ..., group_failed: _Optional[_Union[GroupFailed, _Mapping]] = ..., group_released: _Optional[_Union[GroupReleased, _Mapping]] = ..., role_assigned: _Optional[_Union[RoleAssigned, _Mapping]] = ..., dump_stacks: _Optional[_Union[DumpStacks, _Mapping]] = ..., engine_policy: _Optional[_Union[EnginePolicy, _Mapping]] = ..., engine_placement: _Optional[_Union[EnginePlacement, _Mapping]] = ..., instance_state: _Optional[_Union[InstanceState, _Mapping]] = ..., stacks_dumped: _Optional[_Union[StacksDumped, _Mapping]] = ..., group_id: _Optional[_Union[GroupId, _Mapping]] = ..., cache_wanted: _Optional[_Union[CacheWanted, _Mapping]] = ..., cache_ready: _Optional[_Union[CacheReady, _Mapping]] = ..., cache_refused: _Optional[_Union[CacheRefused, _Mapping]] = ..., release_leaked: _Optional[_Union[ReleaseLeaked, _Mapping]] = ..., group_ready: _Optional[_Union[GroupReady, _Mapping]] = ..., replay_ask: _Optional[_Union[ReplayAsk, _Mapping]] = ..., replay_offer: _Optional[_Union[ReplayOffer, _Mapping]] = ...) -> None: ...

class ReleaseLeaked(_message.Message):
    __slots__ = ("engine", "rank", "release", "run_ms", "leaked_total")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    RUN_MS_FIELD_NUMBER: _ClassVar[int]
    LEAKED_TOTAL_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    release: str
    run_ms: int
    leaked_total: int
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., release: _Optional[str] = ..., run_ms: _Optional[int] = ..., leaked_total: _Optional[int] = ...) -> None: ...

class InstanceState(_message.Message):
    __slots__ = ("engine", "phase", "positions_held", "world", "epoch", "reason")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_HELD_FIELD_NUMBER: _ClassVar[int]
    WORLD_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    engine: str
    phase: InstancePhase
    positions_held: int
    world: int
    epoch: int
    reason: str
    def __init__(self, engine: _Optional[str] = ..., phase: _Optional[_Union[InstancePhase, str]] = ..., positions_held: _Optional[int] = ..., world: _Optional[int] = ..., epoch: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...

class EnginePolicy(_message.Message):
    __slots__ = ("engine", "policy")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    POLICY_FIELD_NUMBER: _ClassVar[int]
    engine: str
    policy: bytes
    def __init__(self, engine: _Optional[str] = ..., policy: _Optional[bytes] = ...) -> None: ...

class DumpStacks(_message.Message):
    __slots__ = ("reason", "request_id")
    REASON_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    reason: str
    request_id: int
    def __init__(self, reason: _Optional[str] = ..., request_id: _Optional[int] = ...) -> None: ...

class StacksDumped(_message.Message):
    __slots__ = ("request_id", "text")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    text: str
    def __init__(self, request_id: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...

class RoleAssigned(_message.Message):
    __slots__ = ("engine", "rank", "epoch", "role", "stage", "refusal")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    REFUSAL_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    epoch: int
    role: Role
    stage: _containers.RepeatedScalarFieldContainer[int]
    refusal: str
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., epoch: _Optional[int] = ..., role: _Optional[_Union[Role, str]] = ..., stage: _Optional[_Iterable[int]] = ..., refusal: _Optional[str] = ...) -> None: ...

class Join(_message.Message):
    __slots__ = ("engine", "epoch", "rank", "payload")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    engine: str
    epoch: int
    rank: int
    payload: bytes
    def __init__(self, engine: _Optional[str] = ..., epoch: _Optional[int] = ..., rank: _Optional[int] = ..., payload: _Optional[bytes] = ...) -> None: ...

class Ack(_message.Message):
    __slots__ = ("rank", "epoch", "error", "received_at_us")
    RANK_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_AT_US_FIELD_NUMBER: _ClassVar[int]
    rank: int
    epoch: int
    error: str
    received_at_us: int
    def __init__(self, rank: _Optional[int] = ..., epoch: _Optional[int] = ..., error: _Optional[str] = ..., received_at_us: _Optional[int] = ...) -> None: ...

class SpareRef(_message.Message):
    __slots__ = ("for_rank", "node", "device")
    FOR_RANK_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    for_rank: int
    node: str
    device: int
    def __init__(self, for_rank: _Optional[int] = ..., node: _Optional[str] = ..., device: _Optional[int] = ...) -> None: ...

class SetMember(_message.Message):
    __slots__ = ("rank", "spare")
    RANK_FIELD_NUMBER: _ClassVar[int]
    SPARE_FIELD_NUMBER: _ClassVar[int]
    rank: int
    spare: SpareRef
    def __init__(self, rank: _Optional[int] = ..., spare: _Optional[_Union[SpareRef, _Mapping]] = ...) -> None: ...

class CoverGroup(_message.Message):
    __slots__ = ("scope", "spans", "server")
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    SERVER_FIELD_NUMBER: _ClassVar[int]
    scope: Scope
    spans: _containers.RepeatedCompositeFieldContainer[SetMember]
    server: SetMember
    def __init__(self, scope: _Optional[_Union[Scope, str]] = ..., spans: _Optional[_Iterable[_Union[SetMember, _Mapping]]] = ..., server: _Optional[_Union[SetMember, _Mapping]] = ...) -> None: ...

class GroupId(_message.Message):
    __slots__ = ("engine", "generation", "scope", "spans", "id")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    engine: str
    generation: int
    scope: Scope
    spans: _containers.RepeatedCompositeFieldContainer[SetMember]
    id: bytes
    def __init__(self, engine: _Optional[str] = ..., generation: _Optional[int] = ..., scope: _Optional[_Union[Scope, str]] = ..., spans: _Optional[_Iterable[_Union[SetMember, _Mapping]]] = ..., id: _Optional[bytes] = ...) -> None: ...

class Holder(_message.Message):
    __slots__ = ("node", "device")
    NODE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    node: str
    device: int
    def __init__(self, node: _Optional[str] = ..., device: _Optional[int] = ...) -> None: ...

class CoverPlan(_message.Message):
    __slots__ = ("engine", "generation", "epoch", "placement", "groups", "sets")
    class PlacementEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: Holder
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[Holder, _Mapping]] = ...) -> None: ...
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_FIELD_NUMBER: _ClassVar[int]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    SETS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    generation: int
    epoch: int
    placement: _containers.MessageMap[int, Holder]
    groups: _containers.RepeatedCompositeFieldContainer[CoverGroup]
    sets: _containers.RepeatedCompositeFieldContainer[CoverSet]
    def __init__(self, engine: _Optional[str] = ..., generation: _Optional[int] = ..., epoch: _Optional[int] = ..., placement: _Optional[_Mapping[int, Holder]] = ..., groups: _Optional[_Iterable[_Union[CoverGroup, _Mapping]]] = ..., sets: _Optional[_Iterable[_Union[CoverSet, _Mapping]]] = ...) -> None: ...

class CoverSet(_message.Message):
    __slots__ = ("members", "groups", "purpose")
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    PURPOSE_FIELD_NUMBER: _ClassVar[int]
    members: _containers.RepeatedCompositeFieldContainer[SetMember]
    groups: _containers.RepeatedScalarFieldContainer[int]
    purpose: str
    def __init__(self, members: _Optional[_Iterable[_Union[SetMember, _Mapping]]] = ..., groups: _Optional[_Iterable[int]] = ..., purpose: _Optional[str] = ...) -> None: ...

class GroupReady(_message.Message):
    __slots__ = ("engine", "generation", "spans", "scope", "member")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    engine: str
    generation: int
    spans: _containers.RepeatedCompositeFieldContainer[SetMember]
    scope: Scope
    member: SetMember
    def __init__(self, engine: _Optional[str] = ..., generation: _Optional[int] = ..., spans: _Optional[_Iterable[_Union[SetMember, _Mapping]]] = ..., scope: _Optional[_Union[Scope, str]] = ..., member: _Optional[_Union[SetMember, _Mapping]] = ...) -> None: ...

class GroupFormed(_message.Message):
    __slots__ = ("engine", "generation", "spans", "scope", "seconds", "member")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SECONDS_FIELD_NUMBER: _ClassVar[int]
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    engine: str
    generation: int
    spans: _containers.RepeatedCompositeFieldContainer[SetMember]
    scope: Scope
    seconds: float
    member: SetMember
    def __init__(self, engine: _Optional[str] = ..., generation: _Optional[int] = ..., spans: _Optional[_Iterable[_Union[SetMember, _Mapping]]] = ..., scope: _Optional[_Union[Scope, str]] = ..., seconds: _Optional[float] = ..., member: _Optional[_Union[SetMember, _Mapping]] = ...) -> None: ...

class GroupFailed(_message.Message):
    __slots__ = ("engine", "generation", "spans", "scope", "error", "member")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    engine: str
    generation: int
    spans: _containers.RepeatedCompositeFieldContainer[SetMember]
    scope: Scope
    error: str
    member: SetMember
    def __init__(self, engine: _Optional[str] = ..., generation: _Optional[int] = ..., spans: _Optional[_Iterable[_Union[SetMember, _Mapping]]] = ..., scope: _Optional[_Union[Scope, str]] = ..., error: _Optional[str] = ..., member: _Optional[_Union[SetMember, _Mapping]] = ...) -> None: ...

class GroupReleased(_message.Message):
    __slots__ = ("engine", "generation", "spans", "scope", "member")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    engine: str
    generation: int
    spans: _containers.RepeatedCompositeFieldContainer[SetMember]
    scope: Scope
    member: SetMember
    def __init__(self, engine: _Optional[str] = ..., generation: _Optional[int] = ..., spans: _Optional[_Iterable[_Union[SetMember, _Mapping]]] = ..., scope: _Optional[_Union[Scope, str]] = ..., member: _Optional[_Union[SetMember, _Mapping]] = ...) -> None: ...

class ProcessRef(_message.Message):
    __slots__ = ("node", "instance", "serial")
    NODE_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_FIELD_NUMBER: _ClassVar[int]
    SERIAL_FIELD_NUMBER: _ClassVar[int]
    node: str
    instance: int
    serial: int
    def __init__(self, node: _Optional[str] = ..., instance: _Optional[int] = ..., serial: _Optional[int] = ...) -> None: ...

class ReplayAsk(_message.Message):
    __slots__ = ("engine", "rank", "asker")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    ASKER_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    asker: ProcessRef
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., asker: _Optional[_Union[ProcessRef, _Mapping]] = ...) -> None: ...

class ReplayOffer(_message.Message):
    __slots__ = ("engine", "rank", "handle", "to")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    handle: str
    to: ProcessRef
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., handle: _Optional[str] = ..., to: _Optional[_Union[ProcessRef, _Mapping]] = ...) -> None: ...
