# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What a process of the engine was launched with: one typed object, read from the environment once.

The adapter's processes are started by vLLM (the engine's workers), by a node's daemon (a spare,
from the policy's ``spareLaunch``) or by an operator (``node_main``), and the only channel all
three share is the environment. This module is its one reader: :func:`current` reads it at the
process's first use and keeps the result; every module of the adapter asks :func:`current` and
none reads ``os.environ`` itself. An entry point that knows more than its environment said (a
spare that has taken a rank) records it with :func:`update`; one that starts processes of its own
writes what they need with :func:`export`, which sets both this process's settings and the
environment the children inherit.

Process configuration that belongs to other libraries stays theirs to set (``TORCH_NCCL_*`` by
the plugin's ``register``, ``CUDA_VISIBLE_DEVICES`` and ``VLLM_HOST_IP`` by the launchers);
the two the adapter needs to know are read here with the rest.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass

ENV_ENABLE = "EVERSERVE_GRAPH_SAFE"
ENV_REGION = "EVERSERVE_SLOT_REGION"
ENV_ROW = "EVERSERVE_SLOT_ROW"
ENV_LAYOUT = "EVERSERVE_LAYOUT"
ENV_SPARE = "EVERSERVE_SPARE"
ENV_SPARE_NODE = "EVERSERVE_SPARE_NODE"
ENV_SPARE_RANK = "EVERSERVE_SPARE_RANK"
ENV_SPARE_ANY = "EVERSERVE_SPARE_ANY"

ENV = {
    "engine": "EVERSERVE_ENGINE",
    "guarantee": "EVERSERVE_GUARANTEE",
    "node": "EVERSERVE_NODE",
    "discovery_dir": "EVERSERVE_DISCOVERY_DIR",
    "policy": "EVERSERVE_POLICY",
    "spare_gpu": "EVERSERVE_SPARE_GPU",
    "spare_rank": ENV_SPARE_RANK,
    "event_dir": "EVERSERVE_EVENT_DIR",
    "slot_variant": "EVERSERVE_SLOT_VARIANT",
    "detect_steps": "EVERSERVE_DETECT_STEPS",
    "detect_floor_ms": "EVERSERVE_DETECT_FLOOR_MS",
    "hang_s": "EVERSERVE_HANG_S",
}
"""The launcher's names, by setting."""


@dataclass(frozen=True)
class Settings:
    """Everything a process of the engine reads from how it was launched."""

    graph_safe: bool = False
    """The graph-safe all-reduce and the adapter's hooks are on in this process."""
    engine: str = ""
    """The engine's name in the policy; empty, the executor behaves as stock plus death handling."""
    guarantee: str = ""
    """The engine's guarantee class as its launcher states it (``exact`` or ``continuous``): what
    the adapter's flags and a node's configuration are resolved for, and what the process's
    identity says, which the control plane holds to the engine's policy. Empty, unstated."""
    node: str = "local"
    discovery_dir: str = ""
    policy: str = ""
    event_dir: str = ""
    spare_gpu: str | None = None
    """The executor's own spare's GPU (the single-node shape); None, no spare is started."""
    slot_variant: str = ""
    """A slot-collective variant forced by the operator; empty, the node decides."""
    detect_steps: float = 3.0
    """Step periods the slot kernel waits for a member before naming it lost."""
    detect_floor_ms: float = 30.0
    """The detection bound's floor."""
    hang_s: float = 10.0
    """How long a live member may stay late before the slot watch reports it hung."""
    spare: bool = False
    """This process is a spare: its collective starts on a local stand-in."""
    spare_rank: int | None = None
    """The rank a spare stands for, or took."""
    spare_any: tuple[int, ...] = ()
    """The ranks a spare that stands for any position is eligible for (decision 0030)."""
    spare_node: bool = False
    """A spare on a node with no member of the engine: a replacement's row is its column."""
    layout: tuple[int, int] | None = None
    """``(tp, pp)`` as the executor launched the engine; None, one stage of the whole world."""
    region: str = ""
    """The node's slot region, or empty for none."""
    row: int | None = None
    """This process's slot row when it is not its rank's column (a spare)."""
    visible_devices: tuple[str, ...] = ()
    """``CUDA_VISIBLE_DEVICES`` as launched: the GPUs the engine's workers map."""
    host_ip: str = ""
    """``VLLM_HOST_IP``: this node's address, where queues a remote head reads bind."""


def read(environ: Mapping[str, str]) -> Settings:
    """The settings ``environ`` names; the only place the adapter reads its environment."""

    def num(name: str, default: float) -> float:
        value = environ.get(name, "")
        return float(value) if value else default

    def opt_int(name: str) -> int | None:
        value = environ.get(name, "")
        return int(value) if value else None

    layout = environ.get(ENV_LAYOUT, "")
    return Settings(
        graph_safe=environ.get(ENV_ENABLE) == "1",
        engine=environ.get(ENV["engine"], ""),
        guarantee=environ.get(ENV["guarantee"], ""),
        node=environ.get(ENV["node"], "") or "local",
        discovery_dir=environ.get(ENV["discovery_dir"], ""),
        policy=environ.get(ENV["policy"], ""),
        event_dir=environ.get(ENV["event_dir"], ""),
        spare_gpu=environ.get(ENV["spare_gpu"]) or None,
        slot_variant=environ.get(ENV["slot_variant"], ""),
        detect_steps=num(ENV["detect_steps"], 3.0),
        detect_floor_ms=num(ENV["detect_floor_ms"], 30.0),
        hang_s=num(ENV["hang_s"], 10.0),
        spare=environ.get(ENV_SPARE) == "1",
        spare_rank=opt_int(ENV_SPARE_RANK),
        spare_any=tuple(int(r) for r in environ.get(ENV_SPARE_ANY, "").split(",") if r),
        spare_node=environ.get(ENV_SPARE_NODE) == "1",
        layout=tuple(int(x) for x in layout.split(",")) if layout else None,  # type: ignore[arg-type]
        region=environ.get(ENV_REGION, ""),
        row=opt_int(ENV_ROW),
        visible_devices=tuple(
            g.strip() for g in environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()
        ),
        host_ip=environ.get("VLLM_HOST_IP", ""),
    )


def environ_of(s: Settings) -> dict[str, str]:
    """The environment that :func:`read` turns back into ``s``, for a child process."""
    out = {
        ENV_ENABLE: "1" if s.graph_safe else "",
        ENV["engine"]: s.engine,
        ENV["guarantee"]: s.guarantee,
        ENV["node"]: s.node,
        ENV["discovery_dir"]: s.discovery_dir,
        ENV["policy"]: s.policy,
        ENV["event_dir"]: s.event_dir,
        ENV["spare_gpu"]: s.spare_gpu or "",
        ENV["slot_variant"]: s.slot_variant,
        ENV["detect_steps"]: repr(s.detect_steps),
        ENV["detect_floor_ms"]: repr(s.detect_floor_ms),
        ENV["hang_s"]: repr(s.hang_s),
        ENV_SPARE: "1" if s.spare else "",
        ENV_SPARE_RANK: "" if s.spare_rank is None else str(s.spare_rank),
        ENV_SPARE_ANY: ",".join(str(r) for r in s.spare_any),
        ENV_SPARE_NODE: "1" if s.spare_node else "",
        ENV_LAYOUT: ",".join(str(x) for x in s.layout) if s.layout else "",
        ENV_REGION: s.region,
        ENV_ROW: "" if s.row is None else str(s.row),
    }
    return out


_CURRENT: Settings | None = None
_LOCK = threading.Lock()


def current() -> Settings:
    """This process's settings, read from its environment at the first call and kept."""
    global _CURRENT
    if _CURRENT is None:
        with _LOCK:
            if _CURRENT is None:
                _CURRENT = read(os.environ)
    return _CURRENT


def reload() -> Settings:
    """Read the environment again: a process entry whose environment was set after startup."""
    global _CURRENT
    with _LOCK:
        _CURRENT = read(os.environ)
    return _CURRENT


def update(**changes: object) -> Settings:
    """This process's settings with ``changes``; children started later do not see them."""
    global _CURRENT
    with _LOCK:
        _CURRENT = dataclasses.replace(_CURRENT or read(os.environ), **changes)  # type: ignore[arg-type]
    return _CURRENT


def export(**changes: object) -> Settings:
    """:func:`update`, and the environment set so processes started from here on read the same."""
    s = update(**changes)
    os.environ.update({k: v for k, v in environ_of(s).items() if v})
    for k, v in environ_of(s).items():
        if not v:
            os.environ.pop(k, None)
    return s
