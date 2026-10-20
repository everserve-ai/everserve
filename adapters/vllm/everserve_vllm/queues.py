# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Closing a vLLM ``MessageQueue`` this adapter is done with.

vLLM makes each queue its own zmq context and sets no linger on its sockets: a queue it keeps for
the engine's life never closes. This adapter replaces queues at a change and drops each spare's
replay queue once read. A dropped queue's context is terminated whenever Python collects it, and
a terminating context with a socket that still holds a message for a peer that will never read it
(a rank gone, a spare that left) waits forever, in whatever thread the collection ran. So a
queue is closed when it is done with, its context destroyed with no linger, never left to the
collector.
"""

from __future__ import annotations

from typing import Any


def close_queue(mq: Any) -> None:
    """Close ``mq``'s sockets now, nothing kept to be sent.

    Its zmq context, with every socket on it (the remote and local ones, the local readers'
    wake-up), is destroyed with no linger.

    A queue with no socket (a stand-in) is left as it is; closing twice is a no-op.
    """
    for name in ("remote_socket", "local_socket"):
        sock = getattr(mq, name, None)
        if sock is not None:
            ctx = getattr(sock, "context", None)
            if ctx is not None and not ctx.closed:
                ctx.destroy(linger=0)
                return
            if not sock.closed:
                sock.close(linger=0)
