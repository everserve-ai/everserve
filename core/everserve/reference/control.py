# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The control channel between the executor and each worker's runtime.

The executor learns that a worker died from its process sentinel. It has to tell every surviving
worker, and the message must land even when the worker's main thread is blocked in a collective
with the dead peer. So each worker runs a :class:`FaultListener` on a side thread, and the executor
holds one :class:`FaultClient` per worker. Loopback only; the key authenticates the pipe, nothing
more.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from multiprocessing.connection import Client, Connection, Listener

log = logging.getLogger(__name__)

DEFAULT_AUTH = b"everserve-control"
Address = tuple[str, int]


class FaultListener:
    """Accepts fault notices for one worker and forwards them to ``on_fault(dead, sent_at)``."""

    def __init__(
        self,
        address: Address,
        on_fault: Callable[[int, float | None], None],
        identity: Callable[[], tuple[int, int]],
        auth: bytes = DEFAULT_AUTH,
    ) -> None:
        """Bind ``address`` and start serving on a daemon thread.

        Args:
            address: loopback host and port to listen on.
            on_fault: called on the listener's thread with the dead rank and the sender's clock.
            identity: returns ``(original rank, epoch)`` for pings.
            auth: the channel key.
        """
        self._on_fault = on_fault
        self._identity = identity
        self._listener = Listener(address, authkey=auth, backlog=16)
        self.address: Address = self._listener.address
        threading.Thread(target=self._accept, daemon=True, name="everserve-fault-listener").start()

    def _accept(self) -> None:
        while True:
            try:
                conn = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: Connection) -> None:
        try:
            while True:
                msg = conn.recv()
                if msg[0] == "fault":
                    self._on_fault(msg[1], msg[2] if len(msg) > 2 else None)
                    conn.send(("ok",))
                elif msg[0] == "ping":
                    rank, epoch = self._identity()
                    conn.send(("pong", rank, epoch))
                else:
                    conn.send(("error", f"unknown message {msg[0]!r}"))
        except (EOFError, OSError):
            return

    def close(self) -> None:
        """Stop accepting. Open connections end when their clients close."""
        self._listener.close()


class FaultClient:
    """The executor's end of one worker's control channel."""

    def __init__(
        self,
        address: Address,
        auth: bytes = DEFAULT_AUTH,
        connect_timeout_s: float = 10.0,
        reply_timeout_s: float = 60.0,
    ) -> None:
        """Connect, retrying until the worker's listener is up or ``connect_timeout_s`` passes."""
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self._conn = Client(address, authkey=auth)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        self._reply_timeout_s = reply_timeout_s
        self._lock = threading.Lock()

    def _rpc(self, *msg: object) -> tuple:
        with self._lock:
            self._conn.send(msg)
            if not self._conn.poll(self._reply_timeout_s):
                raise TimeoutError(
                    f"worker did not acknowledge {msg[0]!r} in {self._reply_timeout_s}s"
                )
            reply = self._conn.recv()
        if reply[0] == "error":
            raise RuntimeError(reply[1])
        return reply

    def fault(self, dead: int, sent_at: float | None = None) -> None:
        """Tell the worker that ``dead`` is gone; returns once the worker has shrunk its group.

        The shrink is a collective over the survivors, so a worker cannot acknowledge until every
        other survivor has been told. To notify a whole group use :func:`broadcast_fault`, which
        sends to all before waiting on any; calling this method on survivors one by one deadlocks.
        """
        self.send_fault(dead, sent_at)
        self.wait_ack()

    def send_fault(self, dead: int, sent_at: float | None = None) -> None:
        """Send the fault notice without waiting. Pair with :meth:`wait_ack`."""
        self._conn.send(("fault", dead, time.time() if sent_at is None else sent_at))

    def wait_ack(self) -> None:
        """Wait for the acknowledgement of the last :meth:`send_fault`."""
        if not self._conn.poll(self._reply_timeout_s):
            raise TimeoutError(f"worker did not acknowledge the fault in {self._reply_timeout_s}s")
        reply = self._conn.recv()
        if reply[0] == "error":
            raise RuntimeError(reply[1])

    def ping(self) -> tuple[int, int]:
        """The worker's ``(original rank, epoch)``."""
        _, rank, epoch = self._rpc("ping")
        return rank, epoch

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()


def broadcast_fault(
    clients: Iterable[FaultClient], dead: int, sent_at: float | None = None
) -> None:
    """Tell every survivor that ``dead`` is gone, then wait for all of them.

    Every notice is sent before any acknowledgement is awaited, because each worker's shrink is a
    collective that completes only when all survivors take part.
    """
    clients = list(clients)
    at = time.time() if sent_at is None else sent_at
    for c in clients:
        c.send_fault(dead, at)
    for c in clients:
        c.wait_ack()
