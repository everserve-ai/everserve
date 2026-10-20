from everserve.proto import coord_pb2 as _coord_pb2
from everserve.proto import signals_pb2 as _signals_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NodeSpec(_message.Message):
    __slots__ = ("node", "unschedulable", "drain_by_ms", "reason")
    NODE_FIELD_NUMBER: _ClassVar[int]
    UNSCHEDULABLE_FIELD_NUMBER: _ClassVar[int]
    DRAIN_BY_MS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    node: str
    unschedulable: bool
    drain_by_ms: int
    reason: str
    def __init__(self, node: _Optional[str] = ..., unschedulable: bool = ..., drain_by_ms: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...

class ResourceSpec(_message.Message):
    __slots__ = ("generation", "written_by", "written_at_ms", "node", "pool")
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    WRITTEN_BY_FIELD_NUMBER: _ClassVar[int]
    WRITTEN_AT_MS_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    POOL_FIELD_NUMBER: _ClassVar[int]
    generation: int
    written_by: str
    written_at_ms: int
    node: NodeSpec
    pool: PoolSpec
    def __init__(self, generation: _Optional[int] = ..., written_by: _Optional[str] = ..., written_at_ms: _Optional[int] = ..., node: _Optional[_Union[NodeSpec, _Mapping]] = ..., pool: _Optional[_Union[PoolSpec, _Mapping]] = ...) -> None: ...

class PoolSpec(_message.Message):
    __slots__ = ("name", "engines", "wants", "device_class", "domain", "reason", "command", "env")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    ENGINES_FIELD_NUMBER: _ClassVar[int]
    WANTS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_CLASS_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    name: str
    engines: _containers.RepeatedScalarFieldContainer[str]
    wants: int
    device_class: str
    domain: str
    reason: str
    command: _containers.RepeatedScalarFieldContainer[str]
    env: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., engines: _Optional[_Iterable[str]] = ..., wants: _Optional[int] = ..., device_class: _Optional[str] = ..., domain: _Optional[str] = ..., reason: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., env: _Optional[_Mapping[str, str]] = ...) -> None: ...

class EngineStatus(_message.Message):
    __slots__ = ("engine", "plan_generation", "change", "assignments", "members")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    PLAN_GENERATION_FIELD_NUMBER: _ClassVar[int]
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    plan_generation: int
    change: ChangeLog
    assignments: _containers.RepeatedCompositeFieldContainer[Assignment]
    members: _containers.RepeatedCompositeFieldContainer[KeptMember]
    def __init__(self, engine: _Optional[str] = ..., plan_generation: _Optional[int] = ..., change: _Optional[_Union[ChangeLog, _Mapping]] = ..., assignments: _Optional[_Iterable[_Union[Assignment, _Mapping]]] = ..., members: _Optional[_Iterable[_Union[KeptMember, _Mapping]]] = ...) -> None: ...

class KeptMember(_message.Message):
    __slots__ = ("rank", "node")
    RANK_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    rank: int
    node: str
    def __init__(self, rank: _Optional[int] = ..., node: _Optional[str] = ...) -> None: ...

class Assignment(_message.Message):
    __slots__ = ("rank", "epoch", "node", "connection", "host")
    RANK_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    rank: int
    epoch: int
    node: str
    connection: _coord_pb2.ConnectionRef
    host: str
    def __init__(self, rank: _Optional[int] = ..., epoch: _Optional[int] = ..., node: _Optional[str] = ..., connection: _Optional[_Union[_coord_pb2.ConnectionRef, _Mapping]] = ..., host: _Optional[str] = ...) -> None: ...

class ChangeLog(_message.Message):
    __slots__ = ("change", "deliveries", "world_size", "shrunk", "from_nodes", "cover", "taking", "committed")
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    DELIVERIES_FIELD_NUMBER: _ClassVar[int]
    WORLD_SIZE_FIELD_NUMBER: _ClassVar[int]
    SHRUNK_FIELD_NUMBER: _ClassVar[int]
    FROM_NODES_FIELD_NUMBER: _ClassVar[int]
    COVER_FIELD_NUMBER: _ClassVar[int]
    TAKING_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_FIELD_NUMBER: _ClassVar[int]
    change: _signals_pb2.Change
    deliveries: _containers.RepeatedCompositeFieldContainer[NodeDelivery]
    world_size: int
    shrunk: _containers.RepeatedScalarFieldContainer[int]
    from_nodes: _containers.RepeatedScalarFieldContainer[str]
    cover: str
    taking: _containers.RepeatedCompositeFieldContainer[_signals_pb2.Take]
    committed: _coord_pb2.Committed
    def __init__(self, change: _Optional[_Union[_signals_pb2.Change, _Mapping]] = ..., deliveries: _Optional[_Iterable[_Union[NodeDelivery, _Mapping]]] = ..., world_size: _Optional[int] = ..., shrunk: _Optional[_Iterable[int]] = ..., from_nodes: _Optional[_Iterable[str]] = ..., cover: _Optional[str] = ..., taking: _Optional[_Iterable[_Union[_signals_pb2.Take, _Mapping]]] = ..., committed: _Optional[_Union[_coord_pb2.Committed, _Mapping]] = ...) -> None: ...

class NodeDelivery(_message.Message):
    __slots__ = ("node", "to_ranks", "to_spares", "take")
    NODE_FIELD_NUMBER: _ClassVar[int]
    TO_RANKS_FIELD_NUMBER: _ClassVar[int]
    TO_SPARES_FIELD_NUMBER: _ClassVar[int]
    TAKE_FIELD_NUMBER: _ClassVar[int]
    node: str
    to_ranks: _containers.RepeatedScalarFieldContainer[int]
    to_spares: _containers.RepeatedScalarFieldContainer[int]
    take: _containers.RepeatedCompositeFieldContainer[_signals_pb2.Take]
    def __init__(self, node: _Optional[str] = ..., to_ranks: _Optional[_Iterable[int]] = ..., to_spares: _Optional[_Iterable[int]] = ..., take: _Optional[_Iterable[_Union[_signals_pb2.Take, _Mapping]]] = ...) -> None: ...

class ResourceStatus(_message.Message):
    __slots__ = ("engine",)
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    engine: EngineStatus
    def __init__(self, engine: _Optional[_Union[EngineStatus, _Mapping]] = ...) -> None: ...

class SpecSnapshot(_message.Message):
    __slots__ = ("resources", "term", "statuses")
    class ResourcesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ResourceSpec
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ResourceSpec, _Mapping]] = ...) -> None: ...
    class StatusesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ResourceStatus
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ResourceStatus, _Mapping]] = ...) -> None: ...
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    TERM_FIELD_NUMBER: _ClassVar[int]
    STATUSES_FIELD_NUMBER: _ClassVar[int]
    resources: _containers.MessageMap[str, ResourceSpec]
    term: int
    statuses: _containers.MessageMap[str, ResourceStatus]
    def __init__(self, resources: _Optional[_Mapping[str, ResourceSpec]] = ..., term: _Optional[int] = ..., statuses: _Optional[_Mapping[str, ResourceStatus]] = ...) -> None: ...
