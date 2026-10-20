# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Reading the daemons' records (``proto/everserve/proto/discovery.proto``).

The read side only: a daemon publishes its own record. The file backend is a directory of
``<node>.json`` records in protobuf JSON, written atomically by the daemon, so a reader sees a whole
record or none. The Kubernetes backend arrives with the coordinator and yields the same records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from google.protobuf import json_format

from everserve.proto import discovery_pb2


@dataclass(frozen=True)
class DaemonRecord:
    """One node daemon, as published."""

    node: str
    address: str
    engines: tuple[str, ...]
    started_at_ms: int
    pid: int
    mirror_address: str
    advertised_host: str = ""
    """The address the daemon advertises for this node (``--advertise-host``), empty on a
    single node."""

    @property
    def host(self) -> str:
        """The address other nodes reach this node at.

        The advertised host when the daemon published one; else the host of a TCP ``address``;
        else loopback, which is what a Unix socket path means: a node reached only from itself.
        Used for the rendezvous of groups that cross nodes (`docs/MULTINODE.md` section 4):
        the store listens on the node of the group's first member.
        """
        if self.advertised_host:
            return self.advertised_host
        a = self.address
        if a.startswith("/") or "/" in a and ":" not in a:
            return "127.0.0.1"
        if a.startswith("["):  # [v6]:port
            return a[1 : a.index("]")]
        host, _, port = a.rpartition(":")
        return host if host and port.isdigit() else "127.0.0.1"

    @classmethod
    def from_message(cls, msg: discovery_pb2.DaemonRecord) -> DaemonRecord:
        """Build from the wire message."""
        return cls(
            node=msg.node,
            address=msg.address,
            engines=tuple(msg.engines),
            started_at_ms=msg.started_at_ms,
            pid=msg.pid,
            mirror_address=msg.mirror_address,
            advertised_host=msg.host,
        )


class FileDiscovery:
    """A directory of daemon records."""

    def __init__(self, directory: str | Path) -> None:
        """Read records from ``directory``; it need not exist yet."""
        self._dir = Path(directory)

    def daemons(self) -> list[DaemonRecord]:
        """Every published daemon, by node name.

        Raises:
            ValueError: if a file in the directory is not a daemon record.
        """
        out = []
        if not self._dir.exists():
            return out
        for path in sorted(self._dir.glob("*.json")):
            try:
                msg = json_format.Parse(path.read_text(), discovery_pb2.DaemonRecord())
            except json_format.ParseError as e:
                raise ValueError(f"{path} is not a daemon record: {e}") from e
            out.append(DaemonRecord.from_message(msg))
        return out

    def daemon(self, node: str) -> DaemonRecord | None:
        """The daemon for ``node``, or None when it is not published."""
        return next((r for r in self.daemons() if r.node == node), None)


DAEMON_SOCKET_ENV = "EVERSERVE_DAEMON_SOCKET"
"""The node daemon's socket where the deployment states it (Kubernetes: the DaemonSet's fixed
host path, mounted into the engine's pods), in place of a discovery directory."""
MIRROR_SOCKET_ENV = "EVERSERVE_MIRROR_SOCKET"
"""The node's store socket, stated beside the daemon's."""
HOST_IP_ENV = "EVERSERVE_HOST_IP"
"""The address other nodes reach this process's node at (Kubernetes: the pod's IP, from the
downward API's ``status.podIP``)."""


def local_daemon(node: str, discovery_dir: str | Path | None = None) -> DaemonRecord | None:
    """The record of the daemon on this process's node.

    Where the deployment states the daemon's socket (``EVERSERVE_DAEMON_SOCKET``, the Kubernetes
    pod contract: the DaemonSet's daemon at a fixed host path), the record is built from the
    environment: the daemon's and the store's sockets, and the address this node is reached at
    (``EVERSERVE_HOST_IP``). Otherwise the daemon's published record in ``discovery_dir`` (bare
    metal), or None when it has not published one.

    Raises:
        ValueError: when the environment states the daemon's socket but not the address this node
            is reached at, which a group crossing nodes rendezvouses on.
    """
    import os

    socket = os.environ.get(DAEMON_SOCKET_ENV, "")
    if socket:
        host = os.environ.get(HOST_IP_ENV, "")
        if not host:
            raise ValueError(
                f"{DAEMON_SOCKET_ENV} is set without {HOST_IP_ENV}: this node's address"
            )
        mirror = os.environ.get(MIRROR_SOCKET_ENV, "") or str(Path(socket).parent / "mirror.sock")
        return DaemonRecord(
            node=node,
            address=socket,
            engines=(),
            started_at_ms=0,
            pid=0,
            mirror_address=mirror,
            advertised_host=host,
        )
    if discovery_dir is None:
        return None
    return FileDiscovery(discovery_dir).daemon(node)
