from everserve.proto import signals_pb2 as _signals_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Guarantee(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GUARANTEE_UNSPECIFIED: _ClassVar[Guarantee]
    GUARANTEE_EXACT: _ClassVar[Guarantee]
    GUARANTEE_CONTINUOUS: _ClassVar[Guarantee]
GUARANTEE_UNSPECIFIED: Guarantee
GUARANTEE_EXACT: Guarantee
GUARANTEE_CONTINUOUS: Guarantee

class SparePool(_message.Message):
    __slots__ = ("per_engine", "loading", "allow_mismatched", "max_degrade_percent")
    class Loading(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        LOADING_UNSPECIFIED: _ClassVar[SparePool.Loading]
        LOADING_PRELOADED: _ClassVar[SparePool.Loading]
        LOADING_ON_NOTICE: _ClassVar[SparePool.Loading]
    LOADING_UNSPECIFIED: SparePool.Loading
    LOADING_PRELOADED: SparePool.Loading
    LOADING_ON_NOTICE: SparePool.Loading
    PER_ENGINE_FIELD_NUMBER: _ClassVar[int]
    LOADING_FIELD_NUMBER: _ClassVar[int]
    ALLOW_MISMATCHED_FIELD_NUMBER: _ClassVar[int]
    MAX_DEGRADE_PERCENT_FIELD_NUMBER: _ClassVar[int]
    per_engine: int
    loading: SparePool.Loading
    allow_mismatched: bool
    max_degrade_percent: int
    def __init__(self, per_engine: _Optional[int] = ..., loading: _Optional[_Union[SparePool.Loading, str]] = ..., allow_mismatched: bool = ..., max_degrade_percent: _Optional[int] = ...) -> None: ...

class Engine(_message.Message):
    __slots__ = ("name", "members", "tensor_parallel", "expert_parallel", "pipeline_parallel", "data_parallel", "placement_independent_model", "spare_nodes", "fabric_collectives", "spares", "spare_launch", "release_bound_ms", "replica_nodes", "guarantee", "head_on_demand")
    NAME_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    TENSOR_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    EXPERT_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_INDEPENDENT_MODEL_FIELD_NUMBER: _ClassVar[int]
    SPARE_NODES_FIELD_NUMBER: _ClassVar[int]
    FABRIC_COLLECTIVES_FIELD_NUMBER: _ClassVar[int]
    SPARES_FIELD_NUMBER: _ClassVar[int]
    SPARE_LAUNCH_FIELD_NUMBER: _ClassVar[int]
    RELEASE_BOUND_MS_FIELD_NUMBER: _ClassVar[int]
    REPLICA_NODES_FIELD_NUMBER: _ClassVar[int]
    GUARANTEE_FIELD_NUMBER: _ClassVar[int]
    HEAD_ON_DEMAND_FIELD_NUMBER: _ClassVar[int]
    name: str
    members: _containers.RepeatedCompositeFieldContainer[_signals_pb2.Member]
    tensor_parallel: int
    expert_parallel: int
    pipeline_parallel: int
    data_parallel: int
    placement_independent_model: bool
    spare_nodes: _containers.RepeatedScalarFieldContainer[str]
    fabric_collectives: bool
    spares: _containers.RepeatedCompositeFieldContainer[SpareSlot]
    spare_launch: SpareLaunch
    release_bound_ms: int
    replica_nodes: _containers.RepeatedScalarFieldContainer[str]
    guarantee: Guarantee
    head_on_demand: bool
    def __init__(self, name: _Optional[str] = ..., members: _Optional[_Iterable[_Union[_signals_pb2.Member, _Mapping]]] = ..., tensor_parallel: _Optional[int] = ..., expert_parallel: _Optional[int] = ..., pipeline_parallel: _Optional[int] = ..., data_parallel: _Optional[int] = ..., placement_independent_model: bool = ..., spare_nodes: _Optional[_Iterable[str]] = ..., fabric_collectives: bool = ..., spares: _Optional[_Iterable[_Union[SpareSlot, _Mapping]]] = ..., spare_launch: _Optional[_Union[SpareLaunch, _Mapping]] = ..., release_bound_ms: _Optional[int] = ..., replica_nodes: _Optional[_Iterable[str]] = ..., guarantee: _Optional[_Union[Guarantee, str]] = ..., head_on_demand: bool = ...) -> None: ...

class SpareSlot(_message.Message):
    __slots__ = ("node", "device", "rank")
    NODE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    node: str
    device: int
    rank: int
    def __init__(self, node: _Optional[str] = ..., device: _Optional[int] = ..., rank: _Optional[int] = ...) -> None: ...

class SpareLaunch(_message.Message):
    __slots__ = ("command", "env")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    command: _containers.RepeatedScalarFieldContainer[str]
    env: _containers.ScalarMap[str, str]
    def __init__(self, command: _Optional[_Iterable[str]] = ..., env: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Policy(_message.Message):
    __slots__ = ("version", "outcomes_in_order", "spares", "max_concurrent_losses", "notice_min_ms", "engines", "cover")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    OUTCOMES_IN_ORDER_FIELD_NUMBER: _ClassVar[int]
    SPARES_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENT_LOSSES_FIELD_NUMBER: _ClassVar[int]
    NOTICE_MIN_MS_FIELD_NUMBER: _ClassVar[int]
    ENGINES_FIELD_NUMBER: _ClassVar[int]
    COVER_FIELD_NUMBER: _ClassVar[int]
    version: str
    outcomes_in_order: _containers.RepeatedScalarFieldContainer[_signals_pb2.Outcome]
    spares: SparePool
    max_concurrent_losses: int
    notice_min_ms: int
    engines: _containers.RepeatedCompositeFieldContainer[Engine]
    cover: CoverPolicy
    def __init__(self, version: _Optional[str] = ..., outcomes_in_order: _Optional[_Iterable[_Union[_signals_pb2.Outcome, str]]] = ..., spares: _Optional[_Union[SparePool, _Mapping]] = ..., max_concurrent_losses: _Optional[int] = ..., notice_min_ms: _Optional[int] = ..., engines: _Optional[_Iterable[_Union[Engine, _Mapping]]] = ..., cover: _Optional[_Union[CoverPolicy, _Mapping]] = ...) -> None: ...

class CoverPolicy(_message.Message):
    __slots__ = ("enabled", "max_groups_per_gpu", "form_on_critical_path", "form_timeout_ms")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    MAX_GROUPS_PER_GPU_FIELD_NUMBER: _ClassVar[int]
    FORM_ON_CRITICAL_PATH_FIELD_NUMBER: _ClassVar[int]
    FORM_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    max_groups_per_gpu: int
    form_on_critical_path: bool
    form_timeout_ms: int
    def __init__(self, enabled: bool = ..., max_groups_per_gpu: _Optional[int] = ..., form_on_critical_path: bool = ..., form_timeout_ms: _Optional[int] = ...) -> None: ...
