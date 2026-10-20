# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Framing for the daemon connection: a 4-byte big-endian length, then one protobuf message.

Used over a Unix domain socket on a node and over mTLS TCP across nodes. Both ends are the same
code; the Rust side frames identically (``wire/src/frame.rs``).
"""

from __future__ import annotations

import socket
import struct
from typing import BinaryIO

MAX_FRAME = 16 * 1024 * 1024
"""A frame larger than this is a protocol error, not a message."""

_LEN = struct.Struct(">I")


class FrameError(Exception):
    """The peer sent something that is not a frame."""


def write_frame(sock: socket.socket, payload: bytes) -> None:
    """Send one frame."""
    if len(payload) > MAX_FRAME:
        raise FrameError(f"frame of {len(payload)} bytes exceeds {MAX_FRAME}")
    sock.sendall(_LEN.pack(len(payload)) + payload)


def _read_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock: socket.socket) -> bytes:
    """Receive one frame's payload.

    Raises:
        EOFError: when the peer closed the connection.
        FrameError: when the length prefix is not a frame length.
    """
    (n,) = _LEN.unpack(_read_exactly(sock, _LEN.size))
    if n > MAX_FRAME:
        raise FrameError(f"frame of {n} bytes exceeds {MAX_FRAME}")
    return _read_exactly(sock, n)


def frame(payload: bytes) -> bytes:
    """One frame's bytes, for a stream that is not a socket (a child's standard input)."""
    if len(payload) > MAX_FRAME:
        raise FrameError(f"frame of {len(payload)} bytes exceeds {MAX_FRAME}")
    return _LEN.pack(len(payload)) + payload


def read_stream_frame(stream: BinaryIO) -> bytes:
    """One frame's payload from a stream that is not a socket.

    Raises:
        EOFError: when the stream ends first.
        FrameError: when the length prefix is not a frame length.
    """

    def exactly(n: int) -> bytes:
        data = stream.read(n)
        if len(data) < n:
            raise EOFError("stream ended")
        return data

    (n,) = _LEN.unpack(exactly(_LEN.size))
    if n > MAX_FRAME:
        raise FrameError(f"frame of {n} bytes exceeds {MAX_FRAME}")
    return exactly(n)
