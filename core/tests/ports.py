# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Free loopback ports for a test's processes to bind, shared by the tests that need them.

A listener the test starts itself binds port 0 and reports the port it took (``start_server``,
``torch.distributed.TCPStore``): nothing can take that port first. This module is for the rest,
ports that processes bind later and that others must know beforehand (a worker's control
listener, a gloo rendezvous). Such a port is free when chosen, not when bound; the ports of one
call are held together while chosen, so they are distinct.
"""

from __future__ import annotations

import socket


def free_ports(n: int) -> list[int]:
    """``n`` distinct loopback ports that were free when chosen."""
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def free_port() -> int:
    """One loopback port that was free when chosen."""
    return free_ports(1)[0]
