from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DaemonRecord(_message.Message):
    __slots__ = ("node", "address", "engines", "started_at_ms", "pid", "mirror_address", "host")
    NODE_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ENGINES_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    MIRROR_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    node: str
    address: str
    engines: _containers.RepeatedScalarFieldContainer[str]
    started_at_ms: int
    pid: int
    mirror_address: str
    host: str
    def __init__(self, node: _Optional[str] = ..., address: _Optional[str] = ..., engines: _Optional[_Iterable[str]] = ..., started_at_ms: _Optional[int] = ..., pid: _Optional[int] = ..., mirror_address: _Optional[str] = ..., host: _Optional[str] = ...) -> None: ...
