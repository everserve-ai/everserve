from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Open(_message.Message):
    __slots__ = ("engine", "rank", "protocol", "protocol_oldest")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_OLDEST_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    protocol: int
    protocol_oldest: int
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., protocol: _Optional[int] = ..., protocol_oldest: _Optional[int] = ...) -> None: ...

class Opened(_message.Message):
    __slots__ = ("rank_dir", "stage_prefix")
    RANK_DIR_FIELD_NUMBER: _ClassVar[int]
    STAGE_PREFIX_FIELD_NUMBER: _ClassVar[int]
    rank_dir: str
    stage_prefix: str
    def __init__(self, rank_dir: _Optional[str] = ..., stage_prefix: _Optional[str] = ...) -> None: ...

class StagedStep(_message.Message):
    __slots__ = ("stage", "base_offset", "layer_stride", "head_stride", "layers", "heads", "table_stride", "slots", "dtype", "shape")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    BASE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    LAYER_STRIDE_FIELD_NUMBER: _ClassVar[int]
    HEAD_STRIDE_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    HEADS_FIELD_NUMBER: _ClassVar[int]
    TABLE_STRIDE_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    stage: str
    base_offset: int
    layer_stride: int
    head_stride: int
    layers: _containers.RepeatedScalarFieldContainer[int]
    heads: _containers.RepeatedScalarFieldContainer[int]
    table_stride: int
    slots: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, stage: _Optional[str] = ..., base_offset: _Optional[int] = ..., layer_stride: _Optional[int] = ..., head_stride: _Optional[int] = ..., layers: _Optional[_Iterable[int]] = ..., heads: _Optional[_Iterable[int]] = ..., table_stride: _Optional[int] = ..., slots: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class Prepare(_message.Message):
    __slots__ = ("engine", "rank", "tables", "dtype", "shape", "slots")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    TABLES_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    tables: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    slots: int
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., tables: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., slots: _Optional[int] = ...) -> None: ...

class Commit(_message.Message):
    __slots__ = ("engine", "rank", "epoch", "step", "staged", "request_ids", "committed_tokens")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    STAGED_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    epoch: int
    step: int
    staged: StagedStep
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    committed_tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., epoch: _Optional[int] = ..., step: _Optional[int] = ..., staged: _Optional[_Union[StagedStep, _Mapping]] = ..., request_ids: _Optional[_Iterable[str]] = ..., committed_tokens: _Optional[_Iterable[int]] = ...) -> None: ...

class CommitAck(_message.Message):
    __slots__ = ("step",)
    STEP_FIELD_NUMBER: _ClassVar[int]
    step: int
    def __init__(self, step: _Optional[int] = ...) -> None: ...

class RestoreRequest(_message.Message):
    __slots__ = ("engine", "rank", "since_step")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    SINCE_STEP_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    since_step: int
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., since_step: _Optional[int] = ...) -> None: ...

class SlotRange(_message.Message):
    __slots__ = ("start", "count")
    START_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    start: int
    count: int
    def __init__(self, start: _Optional[int] = ..., count: _Optional[int] = ...) -> None: ...

class TableRef(_message.Message):
    __slots__ = ("layer", "path", "dtype", "shape", "written")
    LAYER_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    WRITTEN_FIELD_NUMBER: _ClassVar[int]
    layer: int
    path: str
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    written: _containers.RepeatedCompositeFieldContainer[SlotRange]
    def __init__(self, layer: _Optional[int] = ..., path: _Optional[str] = ..., dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., written: _Optional[_Iterable[_Union[SlotRange, _Mapping]]] = ...) -> None: ...

class RestoreReply(_message.Message):
    __slots__ = ("tables", "request_ids", "committed_tokens", "step")
    TABLES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    tables: _containers.RepeatedCompositeFieldContainer[TableRef]
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    committed_tokens: _containers.RepeatedScalarFieldContainer[int]
    step: int
    def __init__(self, tables: _Optional[_Iterable[_Union[TableRef, _Mapping]]] = ..., request_ids: _Optional[_Iterable[str]] = ..., committed_tokens: _Optional[_Iterable[int]] = ..., step: _Optional[int] = ...) -> None: ...

class Evicted(_message.Message):
    __slots__ = ("engine", "request_id", "kept_tokens")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    KEPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    request_id: str
    kept_tokens: int
    def __init__(self, engine: _Optional[str] = ..., request_id: _Optional[str] = ..., kept_tokens: _Optional[int] = ...) -> None: ...

class Finished(_message.Message):
    __slots__ = ("engine", "request_ids")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, engine: _Optional[str] = ..., request_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class Replicate(_message.Message):
    __slots__ = ("engine", "rank", "targets", "node")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    targets: _containers.RepeatedScalarFieldContainer[str]
    node: str
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., targets: _Optional[_Iterable[str]] = ..., node: _Optional[str] = ...) -> None: ...

class Expect(_message.Message):
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

class RowsQuery(_message.Message):
    __slots__ = ("engine", "rank", "table", "slots")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    TABLE_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    table: int
    slots: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., table: _Optional[int] = ..., slots: _Optional[_Iterable[int]] = ...) -> None: ...

class SlotRow(_message.Message):
    __slots__ = ("slot", "written_step", "data", "crc32c")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    WRITTEN_STEP_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CRC32C_FIELD_NUMBER: _ClassVar[int]
    slot: int
    written_step: int
    data: bytes
    crc32c: int
    def __init__(self, slot: _Optional[int] = ..., written_step: _Optional[int] = ..., data: _Optional[bytes] = ..., crc32c: _Optional[int] = ...) -> None: ...

class RowsReply(_message.Message):
    __slots__ = ("epoch", "step", "rows")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    step: int
    rows: _containers.RepeatedCompositeFieldContainer[SlotRow]
    def __init__(self, epoch: _Optional[int] = ..., step: _Optional[int] = ..., rows: _Optional[_Iterable[_Union[SlotRow, _Mapping]]] = ...) -> None: ...

class ReplicatedTable(_message.Message):
    __slots__ = ("table", "dtype", "shape", "slots", "data", "crc32c")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CRC32C_FIELD_NUMBER: _ClassVar[int]
    table: int
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    slots: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    crc32c: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, table: _Optional[int] = ..., dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., slots: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ..., crc32c: _Optional[_Iterable[int]] = ...) -> None: ...

class Replicated(_message.Message):
    __slots__ = ("engine", "rank", "epoch", "step", "tables", "request_ids", "committed_tokens")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    TABLES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    epoch: int
    step: int
    tables: _containers.RepeatedCompositeFieldContainer[ReplicatedTable]
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    committed_tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., epoch: _Optional[int] = ..., step: _Optional[int] = ..., tables: _Optional[_Iterable[_Union[ReplicatedTable, _Mapping]]] = ..., request_ids: _Optional[_Iterable[str]] = ..., committed_tokens: _Optional[_Iterable[int]] = ...) -> None: ...

class ReplicaStateQuery(_message.Message):
    __slots__ = ("engine", "rank", "protocol", "node", "protocol_oldest")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_OLDEST_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    protocol: int
    node: str
    protocol_oldest: int
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., protocol: _Optional[int] = ..., node: _Optional[str] = ..., protocol_oldest: _Optional[int] = ...) -> None: ...

class ReplicaState(_message.Message):
    __slots__ = ("epoch", "step", "held", "caught_up", "fence_epoch")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    HELD_FIELD_NUMBER: _ClassVar[int]
    CAUGHT_UP_FIELD_NUMBER: _ClassVar[int]
    FENCE_EPOCH_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    step: int
    held: bool
    caught_up: bool
    fence_epoch: int
    def __init__(self, epoch: _Optional[int] = ..., step: _Optional[int] = ..., held: bool = ..., caught_up: bool = ..., fence_epoch: _Optional[int] = ...) -> None: ...

class CaughtUp(_message.Message):
    __slots__ = ("engine", "rank", "epoch", "step")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    engine: str
    rank: int
    epoch: int
    step: int
    def __init__(self, engine: _Optional[str] = ..., rank: _Optional[int] = ..., epoch: _Optional[int] = ..., step: _Optional[int] = ...) -> None: ...

class CommittedQuery(_message.Message):
    __slots__ = ("engine", "request_ids")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    engine: str
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, engine: _Optional[str] = ..., request_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class CommittedReply(_message.Message):
    __slots__ = ("request_ids", "committed_tokens")
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    committed_tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, request_ids: _Optional[_Iterable[str]] = ..., committed_tokens: _Optional[_Iterable[int]] = ...) -> None: ...

class CacheFile(_message.Message):
    __slots__ = ("path", "size", "sha256")
    PATH_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    path: str
    size: int
    sha256: str
    def __init__(self, path: _Optional[str] = ..., size: _Optional[int] = ..., sha256: _Optional[str] = ...) -> None: ...

class CacheFetch(_message.Message):
    __slots__ = ("key", "rank", "dp", "digest", "holders")
    KEY_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    DP_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    HOLDERS_FIELD_NUMBER: _ClassVar[int]
    key: str
    rank: int
    dp: int
    digest: str
    holders: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, key: _Optional[str] = ..., rank: _Optional[int] = ..., dp: _Optional[int] = ..., digest: _Optional[str] = ..., holders: _Optional[_Iterable[str]] = ...) -> None: ...

class CacheManifestQuery(_message.Message):
    __slots__ = ("key", "rank", "dp", "digest")
    KEY_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    DP_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    key: str
    rank: int
    dp: int
    digest: str
    def __init__(self, key: _Optional[str] = ..., rank: _Optional[int] = ..., dp: _Optional[int] = ..., digest: _Optional[str] = ...) -> None: ...

class CacheManifest(_message.Message):
    __slots__ = ("key", "rank", "dp", "digest", "files")
    KEY_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    DP_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    key: str
    rank: int
    dp: int
    digest: str
    files: _containers.RepeatedCompositeFieldContainer[CacheFile]
    def __init__(self, key: _Optional[str] = ..., rank: _Optional[int] = ..., dp: _Optional[int] = ..., digest: _Optional[str] = ..., files: _Optional[_Iterable[_Union[CacheFile, _Mapping]]] = ...) -> None: ...

class CacheRead(_message.Message):
    __slots__ = ("key", "rank", "dp", "digest", "path", "offset")
    KEY_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    DP_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    key: str
    rank: int
    dp: int
    digest: str
    path: str
    offset: int
    def __init__(self, key: _Optional[str] = ..., rank: _Optional[int] = ..., dp: _Optional[int] = ..., digest: _Optional[str] = ..., path: _Optional[str] = ..., offset: _Optional[int] = ...) -> None: ...

class CacheChunk(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class MirrorRequest(_message.Message):
    __slots__ = ("commit", "restore", "committed", "evicted", "finished", "open", "prepare", "replicate", "replicated", "replica_state", "expect", "rows", "caught_up", "cache_fetch", "cache_manifest", "cache_read")
    COMMIT_FIELD_NUMBER: _ClassVar[int]
    RESTORE_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_FIELD_NUMBER: _ClassVar[int]
    EVICTED_FIELD_NUMBER: _ClassVar[int]
    FINISHED_FIELD_NUMBER: _ClassVar[int]
    OPEN_FIELD_NUMBER: _ClassVar[int]
    PREPARE_FIELD_NUMBER: _ClassVar[int]
    REPLICATE_FIELD_NUMBER: _ClassVar[int]
    REPLICATED_FIELD_NUMBER: _ClassVar[int]
    REPLICA_STATE_FIELD_NUMBER: _ClassVar[int]
    EXPECT_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    CAUGHT_UP_FIELD_NUMBER: _ClassVar[int]
    CACHE_FETCH_FIELD_NUMBER: _ClassVar[int]
    CACHE_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    CACHE_READ_FIELD_NUMBER: _ClassVar[int]
    commit: Commit
    restore: RestoreRequest
    committed: CommittedQuery
    evicted: Evicted
    finished: Finished
    open: Open
    prepare: Prepare
    replicate: Replicate
    replicated: Replicated
    replica_state: ReplicaStateQuery
    expect: Expect
    rows: RowsQuery
    caught_up: CaughtUp
    cache_fetch: CacheFetch
    cache_manifest: CacheManifestQuery
    cache_read: CacheRead
    def __init__(self, commit: _Optional[_Union[Commit, _Mapping]] = ..., restore: _Optional[_Union[RestoreRequest, _Mapping]] = ..., committed: _Optional[_Union[CommittedQuery, _Mapping]] = ..., evicted: _Optional[_Union[Evicted, _Mapping]] = ..., finished: _Optional[_Union[Finished, _Mapping]] = ..., open: _Optional[_Union[Open, _Mapping]] = ..., prepare: _Optional[_Union[Prepare, _Mapping]] = ..., replicate: _Optional[_Union[Replicate, _Mapping]] = ..., replicated: _Optional[_Union[Replicated, _Mapping]] = ..., replica_state: _Optional[_Union[ReplicaStateQuery, _Mapping]] = ..., expect: _Optional[_Union[Expect, _Mapping]] = ..., rows: _Optional[_Union[RowsQuery, _Mapping]] = ..., caught_up: _Optional[_Union[CaughtUp, _Mapping]] = ..., cache_fetch: _Optional[_Union[CacheFetch, _Mapping]] = ..., cache_manifest: _Optional[_Union[CacheManifestQuery, _Mapping]] = ..., cache_read: _Optional[_Union[CacheRead, _Mapping]] = ...) -> None: ...

class MirrorReply(_message.Message):
    __slots__ = ("commit_ack", "restore", "committed", "done", "opened", "replica_state", "rows", "cache_manifest", "cache_chunk", "error")
    COMMIT_ACK_FIELD_NUMBER: _ClassVar[int]
    RESTORE_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    OPENED_FIELD_NUMBER: _ClassVar[int]
    REPLICA_STATE_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    CACHE_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    CACHE_CHUNK_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    commit_ack: CommitAck
    restore: RestoreReply
    committed: CommittedReply
    done: Done
    opened: Opened
    replica_state: ReplicaState
    rows: RowsReply
    cache_manifest: CacheManifest
    cache_chunk: CacheChunk
    error: str
    def __init__(self, commit_ack: _Optional[_Union[CommitAck, _Mapping]] = ..., restore: _Optional[_Union[RestoreReply, _Mapping]] = ..., committed: _Optional[_Union[CommittedReply, _Mapping]] = ..., done: _Optional[_Union[Done, _Mapping]] = ..., opened: _Optional[_Union[Opened, _Mapping]] = ..., replica_state: _Optional[_Union[ReplicaState, _Mapping]] = ..., rows: _Optional[_Union[RowsReply, _Mapping]] = ..., cache_manifest: _Optional[_Union[CacheManifest, _Mapping]] = ..., cache_chunk: _Optional[_Union[CacheChunk, _Mapping]] = ..., error: _Optional[str] = ...) -> None: ...

class Done(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
