# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The two halves of the join, as bytes the control plane carries unread (decision 0024).

The executor offers a position's new process the ring to read scheduler output from; the process
answers with the queue the executor reads its output from and what it did to take the position.
Both are vLLM's queue handles and this adapter's records, nothing the coordinator or a daemon
understands, so they travel as an opaque payload. The bytes are self-describing: a tag, a version
and a kind in front of a JSON body, so a process on another build answers with an error it can
name rather than failing on a shape nobody can read. The control plane will never check them;
this module does.

The body is data and only data. Each message's fields are declared here and checked on the way
in: an unknown or missing field is refused by name, a queue handle is rebuilt field by field as
vLLM's ``Handle`` (its fields checked against vLLM's own), and the compile pair as the contract's
``CompileCache``. Nothing a payload carries can run as code in the process that reads it: the body
was pickled before version 5, and a process that took its bytes from the control plane would have
run whatever the bytes named.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

from everserve.contract.signals import CompileCache

TAG = b"everserve-vllm-join"
VERSION = 8
"""2: the answer carries the compile pair of a spare that stood for any position; 3: and its
attachment record; 4: the offer says whether the position's rank died (``Offer.faulted``);
5: the body is declared JSON, no longer a pickle; 6: the offer names the core's start-up calls
(``Offer.startup``); 7: and what offers the position (``Offer.kind``); 8: no kind, every offer
is a change's and names its calls (the head's take, the only offer that named none, is gone)."""


class JoinFormatError(ValueError):
    """A payload this build cannot read: another tag, version, kind or shape than it expects."""


@dataclass(frozen=True)
class Offer:
    """The executor's half: take ``rank``'s position at ``epoch`` on this ring."""

    rank: int
    epoch: int
    tag: str
    """Names the boundary, as the survivors' boundary RPC does, so answers are matched to it."""
    ring: Any
    """The new broadcast ring's handle (``MessageQueue.export_handle()``)."""
    faulted: bool = False
    """The rank died rather than left on a notice: it committed nothing after the take's restore,
    which waited for its rows held and caught up, so the spare skips the catch-up restore. A
    catch-up after a fault copies 0 rows and still takes 3 to 9 s, held behind the new holder's
    resync of the rank on its store."""
    startup: str = ""
    """The digest of the calls the offering core sent its workers at its start
    (:func:`everserve_vllm.replay.digest`), required on every offer. A spare built from another
    core's calls, or offered none, refuses the offer by name rather than serve with a cache the
    engine's workers were not built with."""


@dataclass(frozen=True)
class Answer:
    """The process's half: it took the position, or why it could not."""

    rank: int
    epoch: int
    tag: str
    status: str
    """``READY``, or ``FAILED`` with ``error``."""
    response: Any = None
    """The handle of the queue the executor reads this process's output from."""
    applied: dict[str, Any] = field(default_factory=dict)
    """What the boundary moved on this process (``worker.apply_change``'s record)."""
    restore: dict[str, Any] | None = None
    """The catch-up restore's record, for the run's report."""
    join_s: float = 0.0
    error: str = ""
    attachment: dict[str, Any] | None = None
    """How the spare attached to the position, term by term (``spare.Attachment`` and, for a
    spare that took the position at the change, the take's own stamps), for the change record."""
    compile_cache: CompileCache | None = None
    """The compile pair a spare that stood for any position attached with (decision 0029, held
    at the join for such a spare, decision 0030); None from a spare registered with its pair."""


_HANDLES = {"ring", "response"}
_RECORDS = {"applied", "restore", "attachment"}
_SCALARS: dict[str, type | tuple[type, ...]] = {
    "rank": int,
    "epoch": int,
    "tag": str,
    "status": str,
    "faulted": bool,
    "startup": str,
    "join_s": (int, float),
    "error": str,
}


def handle_type() -> type:
    """The queue handle type of vLLM, which both halves hand back to it."""
    from vllm.distributed.device_communicators.shm_broadcast import Handle

    return Handle


def encode(message: Offer | Answer) -> bytes:
    """The payload for ``message``: the tag, the version, the kind, then the JSON body."""
    kind = b"offer" if isinstance(message, Offer) else b"answer"
    body: dict[str, Any] = {}
    for f in dataclasses.fields(message):
        value = getattr(message, f.name)
        if f.name in _HANDLES:
            body[f.name] = None if value is None else handle_out(value)
        elif f.name == "compile_cache":
            body[f.name] = None if value is None else {"key": value.key, "digest": value.digest}
        elif f.name in _RECORDS:
            body[f.name] = None if value is None else _record(value)
        else:
            body[f.name] = value
    return b"%s\x00%d\x00%s\x00" % (TAG, VERSION, kind) + json.dumps(body).encode()


def decode(payload: bytes) -> Offer | Answer:
    """The message in ``payload``.

    Raises:
        JoinFormatError: for another tag, a version this build does not read, a kind it does not
            know, or a body that is not the kind's declared fields; the message names what it
            found.
    """
    try:
        tag, version, kind, body = payload.split(b"\x00", 3)
    except ValueError:
        raise JoinFormatError(f"not a join payload: {payload[:40]!r}") from None
    if tag != TAG:
        raise JoinFormatError(f"payload tagged {tag!r}, not {TAG!r}")
    if version != str(VERSION).encode():
        raise JoinFormatError(f"join payload version {version!r}; this build reads {VERSION}")
    if kind not in (b"offer", b"answer"):
        raise JoinFormatError(f"unknown join payload kind {kind!r}")
    cls = Offer if kind == b"offer" else Answer
    try:
        raw = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise JoinFormatError(f"join {kind.decode()} body is not JSON: {e}") from None
    if not isinstance(raw, dict):
        raise JoinFormatError(f"join {kind.decode()} body is {type(raw).__name__}, not an object")
    declared = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(declared))
    if unknown:
        raise JoinFormatError(f"join {kind.decode()} has fields this build does not: {unknown}")
    missing = sorted(
        n
        for n, f in declared.items()
        if n not in raw
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise JoinFormatError(f"join {kind.decode()} lacks fields: {missing}")
    return cls(**{name: _field_in(kind.decode(), name, value) for name, value in raw.items()})


def _field_in(kind: str, name: str, value: Any) -> Any:
    if name in _HANDLES:
        return None if value is None else handle_in(kind, name, value)
    if name == "compile_cache":
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {"key", "digest"}:
            raise JoinFormatError(f"join {kind} compile_cache is not a key and a digest: {value!r}")
        return CompileCache(key=str(value["key"]), digest=str(value["digest"]))
    if name in _RECORDS:
        if value is not None and not isinstance(value, dict):
            raise JoinFormatError(f"join {kind} {name} is {type(value).__name__}, not a record")
        return value
    want = _SCALARS[name]
    # bool is an int to Python and to JSON's readers: a flag is never a number, nor the reverse
    if not isinstance(value, want) or (isinstance(value, bool) and want is not bool):
        raise JoinFormatError(f"join {kind} {name} is {value!r}")
    return value


def handle_out(handle: Any) -> dict[str, Any]:
    """The queue ``Handle`` of vLLM as JSON's data, field by field (the replay's offer too)."""
    if not dataclasses.is_dataclass(handle) or isinstance(handle, type):
        raise TypeError(f"a queue handle is vLLM's Handle, not {type(handle).__name__}")
    return {f.name: _record(getattr(handle, f.name)) for f in dataclasses.fields(handle)}


def handle_in(kind: str, name: str, value: Any) -> Any:
    """The ``Handle`` of vLLM rebuilt from :func:`handle_out`'s data, its fields checked.

    Raises:
        JoinFormatError: naming ``kind`` and ``name``, for data that is not a handle of this build.
    """
    cls = handle_type()
    names = {f.name for f in dataclasses.fields(cls)}
    if not isinstance(value, dict) or set(value) != names:
        found = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise JoinFormatError(f"join {kind} {name} is not a queue handle of this build: {found}")
    args = dict(value)
    if isinstance(args.get("buffer_handle"), list):
        args["buffer_handle"] = tuple(args["buffer_handle"])  # vLLM's shape: a tuple
    return cls(**args)


def _record(value: Any) -> Any:
    """A report as JSON's data: tuples and sets as lists, any other object as its text."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_record(v) for v in value), key=repr)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _record(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return str(value)
