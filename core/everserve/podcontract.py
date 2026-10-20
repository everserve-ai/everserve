# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The Kubernetes pod contract: what an engine's pod mounts and sets to reach its node's daemon.

On Kubernetes the node daemon runs as a DaemonSet (decision 0037), its socket in a fixed directory
of the host. An engine's pod, whichever framework's controller creates it (decision 0041), mounts
that directory and the node's shared-memory directory from the host, and states its node and its
address from the downward API. :func:`everserve.discovery.local_daemon` builds the daemon's record
from them, and the slot region lives under the shared-memory directory, which every pod of the
node mounts, so they share it.

The pieces are Kubernetes objects as dictionaries, for a renderer to place in a pod template.
"""

from __future__ import annotations

import os

from everserve.discovery import DAEMON_SOCKET_ENV, HOST_IP_ENV

DAEMON_DIR = "/run/everserve"
"""The host directory holding the node daemon's socket and its store's, mounted at the same path."""
DAEMON_SOCKET = f"{DAEMON_DIR}/node.sock"
SHM_DIR = "/dev/shm/everserve"
"""The host's tmpfs directory the node's pods share the slot region through."""
SHM_DIR_ENV = "EVERSERVE_SHM_DIR"
"""Where the slot region lives (:func:`everserve.comm.slotkernel.region_path`); /dev/shm unset."""
COMPILE_CACHE_DIR = "/var/lib/everserve/vllm-cache"
"""The host directory holding the node's compiled kernels (decision 0043): the engine's processes
compile into it and load from it (it is their ``VLLM_CACHE_ROOT``), and the node's store serves a
rank's set from it to another node's store and writes the set it fetches for a spare. On the host,
so a pod started again on its node finds its kernels compiled. The chart pins its DaemonSet's and
store's paths to this."""
COMPILE_CACHE_ENV = "VLLM_CACHE_ROOT"
"""vLLM's own setting for where its caches live: the compile cache is
``$VLLM_CACHE_ROOT/torch_compile_cache``."""
NODE_GID = 64000
"""The node's group: the daemon, the store and every engine process of the node share it, so a
process of one pod reaches the sockets and maps the files another made. The chart's ``node.group``,
pinned to this by the chart's test."""
SHARED_MODE = 0o660
"""Every file an engine process makes in the shared directories: its group reads and writes it."""
NODE_ENV = "EVERSERVE_NODE"
"""The node's name in the policy: the Kubernetes Node's name."""


def make_shared(path: str, size: int) -> None:
    """Make the file ``path``, ``size`` bytes, for every process of the node's group.

    Its mode is :data:`SHARED_MODE` whatever the process's umask; the directory's setgid bit gives
    it the node's group. An existing file is resized, and its mode set alike.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, SHARED_MODE)
    try:
        os.fchmod(fd, SHARED_MODE)
        os.ftruncate(fd, size)
    finally:
        os.close(fd)


def security_context() -> dict:
    """An engine pod's security context: never root, and in the node's group."""
    return {"runAsNonRoot": True, "supplementalGroups": [NODE_GID]}


def env() -> list[dict]:
    """The pod's environment under the contract."""
    return [
        {"name": DAEMON_SOCKET_ENV, "value": DAEMON_SOCKET},
        {"name": SHM_DIR_ENV, "value": SHM_DIR},
        {"name": COMPILE_CACHE_ENV, "value": COMPILE_CACHE_DIR},
        {"name": HOST_IP_ENV, "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
        {"name": NODE_ENV, "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
    ]


def volumes() -> list[dict]:
    """The pod's volumes: the directories exist only where the DaemonSet runs.

    A pod scheduled to a node without the daemon fails to start rather than serving unprotected.
    """
    return [
        {"name": "everserve-daemon", "hostPath": {"path": DAEMON_DIR, "type": "Directory"}},
        {"name": "everserve-shm", "hostPath": {"path": SHM_DIR, "type": "Directory"}},
        {
            "name": "everserve-compile-cache",
            "hostPath": {"path": COMPILE_CACHE_DIR, "type": "Directory"},
        },
    ]


def mounts() -> list[dict]:
    """The container's mounts of :func:`volumes`, at the host's paths."""
    return [
        {"name": "everserve-daemon", "mountPath": DAEMON_DIR},
        {"name": "everserve-shm", "mountPath": SHM_DIR},
        {"name": "everserve-compile-cache", "mountPath": COMPILE_CACHE_DIR},
    ]
