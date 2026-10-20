# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A daemon's end of a process link, for the tests of the contract's connections.

The tests speak the real wire (framed protobuf signals over a Unix socket); only the daemon is
stood in for, by whatever the test sends and reads.
"""

import socket
import tempfile
from pathlib import Path

from everserve.contract.signals import from_message, to_message
from everserve.proto import signals_pb2 as pb
from everserve.wire import read_frame, write_frame


class FakeDaemon:
    """A listening socket a process connects to; the test reads its frames and sends its own."""

    def __init__(self, path: str | None = None) -> None:
        """Listen at ``path``, or at a socket in a fresh temporary directory."""
        self.path = path or str(Path(tempfile.mkdtemp()) / "d.sock")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        self.sock.listen(1)
        self.conn = None
        self.received = []

    def accept(self):
        """Accept the process's connection; its first signal."""
        self.conn, _ = self.sock.accept()
        return from_message(pb.Signal.FromString(read_frame(self.conn)))

    def send(self, sig) -> None:
        write_frame(self.conn, to_message(sig).SerializeToString())

    def _frame(self):
        return pb.Signal.FromString(read_frame(self.conn))

    def signal(self):
        """The next signal the process sent."""
        return from_message(self._frame())

    def ack(self):
        """The next acknowledgement; signals read on the way are kept in ``received``."""
        while True:
            f = self._frame()
            if f.WhichOneof("kind") == "ack":
                return f.ack
            self.received.append(from_message(f))

    def ended(self, timeout_s: float = 5.0) -> bool:
        """Whether the process closed its end, without its having said anything."""
        self.conn.settimeout(timeout_s)
        return self.conn.recv(1) == b""
