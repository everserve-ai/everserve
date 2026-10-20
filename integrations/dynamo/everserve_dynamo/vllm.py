# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Dynamo's vLLM backend: the processes an instance's pods run.

The head is Dynamo's own vLLM worker (``python3 -m dynamo.vllm``), which registers the instance
with Dynamo and runs the engine core, with Everserve's executor and the adapter's requirements as
vLLM flags (:mod:`everserve_vllm.flags`, computed in the pod from the vLLM installed there). Every
other node runs the adapter's node role (:mod:`everserve_vllm.node_main`), which registers with its
node's daemon undecided and serves the rank it is told or stands as a spare (decision 0041). Both
take the instance's engine arguments in vLLM's spelling, so every process builds one configuration.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence

from everserve_vllm.engine_args import identity
from everserve_vllm.settings import ENV, ENV_ENABLE

IMAGE_UID = 1000
"""The user Dynamo's vLLM runtime image runs as (``dynamo``, ``USER dynamo``), stated by number,
which a pod that may never run as root needs: Kubernetes cannot check a name."""
LEAVE_WITHIN_S = 60
"""How long a node pod being deleted waits for its ranks to move to their spares (node_main's
--leave-within-s), notice and change included."""
GRACE_S = LEAVE_WITHIN_S + 30
"""The node pods' terminationGracePeriodSeconds: the leave's bound and room for the process's
own shutdown after it, so Kubernetes never kills a node that is still handing its ranks over."""
HEALTH_PORT = 8090
"""Where a node's process serves /live and /ready (:mod:`everserve_vllm.health`)."""


def flag(engine_args: Sequence[str], name: str) -> str | None:
    """The value ``engine_args`` give flag ``name`` (``--name V`` or ``--name=V``), or None."""
    args = list(engine_args)
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a[len(name) + 1 :]
    return None


KV_BLOCKS = "--num-gpu-blocks-override"
BLOCK_SIZE = "--block-size"


def store_slots(engine_args: Sequence[str]) -> int:
    """The token slots of the engine's KV cache: its blocks times its block size.

    A node's store holds a row for every slot the engine can write (the chart's
    ``node.store.slots``), so the engine states its KV cache rather than have vLLM size it from
    the GPU's free memory at start, a number nothing outside the engine knows before it runs. A
    store sized otherwise refuses the engine at its first registration (a chart default of 4,096
    slots against an engine of about 6 million).

    Raises:
        ValueError: when the engine arguments do not state both, or state either below 1.
    """
    blocks, size = flag(engine_args, KV_BLOCKS), flag(engine_args, BLOCK_SIZE)
    if blocks is None or size is None:
        raise ValueError(
            f"engineArgs must state the KV cache: {KV_BLOCKS} N and {BLOCK_SIZE} B (the node "
            "store's slots are N x B, and vLLM otherwise sizes the cache only once it runs)"
        )
    try:
        n, b = int(blocks), int(size)
    except ValueError:
        raise ValueError(f"{KV_BLOCKS} {blocks!r} and {BLOCK_SIZE} {size!r}: integers") from None
    if n < 1 or b < 1:
        raise ValueError(f"{KV_BLOCKS} {n} and {BLOCK_SIZE} {b}: each at least 1")
    return n * b


def widths(engine_args: Sequence[str]) -> tuple[int, int]:
    """The tensor- and pipeline-parallel widths the engine arguments name."""
    i = identity(list(engine_args))
    return i.tp, i.pp


HEAD_ADDITIONAL_CONFIG = {"consolidator_endpoints": None}
"""What Dynamo's vLLM worker adds to vLLM's ``additional_config`` before its engine starts
(``dynamo/vllm/main.py``: the KV block manager's consolidator endpoints, None without it).

vLLM folds ``additional_config`` into its configuration's hash, and so into its compile cache's
key and the step config the control plane compares: a node that resolved the engine's arguments
without it differs from its head in that alone, and is refused. Every process of the instance
takes it in its engine arguments (:func:`engine_args`), the head
too, where Dynamo's own assignment then leaves it unchanged. With Dynamo's KV block manager
enabled the endpoints are the head's own and no node could state them: not supported here."""

ADDITIONAL_CONFIG = "--additional-config"
_ASSIGNED = re.compile(r"""additional_config\[\s*["'](\w+)["']\s*\]\s*=""")
NOT_RENDERED = {"benchmark"}
"""Keys Dynamo's worker adds only in a mode this integration never launches it in
(``--benchmark-mode``)."""


def unaccounted_head_additions(source: str) -> set[str]:
    """The keys Dynamo's worker ``source`` adds to ``additional_config`` that nothing here renders.

    Those neither :data:`HEAD_ADDITIONAL_CONFIG` nor :data:`NOT_RENDERED` accounts for. The
    engine image's build runs it over the ``dynamo.vllm.main`` it installed, and refuses the
    image if any is left: a Dynamo pin that adds another key would otherwise part its head from
    its nodes, which the step guard would refuse at the first bring-up.
    """
    return set(_ASSIGNED.findall(source)) - set(HEAD_ADDITIONAL_CONFIG) - NOT_RENDERED


def main() -> int:
    """The image's check: Dynamo's worker as installed adds nothing the renderer does not."""
    import inspect

    import dynamo.vllm.main as worker

    left = unaccounted_head_additions(inspect.getsource(worker))
    if left:
        sys.stderr.write(f"dynamo.vllm adds {sorted(left)} to additional_config, unrendered\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


def engine_args(args: Sequence[str]) -> list[str]:
    """The instance's engine arguments for every process, the head's included.

    :data:`HEAD_ADDITIONAL_CONFIG` is merged into any ``--additional-config`` the operator gave,
    which keeps its own keys.
    """
    out = list(args)
    given: dict = {}
    for i, arg in enumerate(out):
        if arg == ADDITIONAL_CONFIG and i + 1 < len(out):
            given = json.loads(out[i + 1])
            del out[i : i + 2]
            break
        if arg.startswith(ADDITIONAL_CONFIG + "="):
            given = json.loads(arg.split("=", 1)[1])
            del out[i]
            break
    merged = {**HEAD_ADDITIONAL_CONFIG, **given}
    return [*out, ADDITIONAL_CONFIG, json.dumps(merged, sort_keys=True, separators=(",", ":"))]


def env(engine: str, guarantee: str) -> list[dict]:
    """The adapter's settings every process of the engine reads."""
    return [
        {"name": ENV_ENABLE, "value": "1"},
        {"name": ENV["engine"], "value": engine},
        # the one statement of the class every process of the instance resolves for, and says
        {"name": ENV["guarantee"], "value": guarantee},
    ]


def head(engine_args: Sequence[str], nnodes: int, rendezvous: str, port: int) -> dict:
    """The head's container command: Dynamo's worker, as node 0 of ``nnodes``.

    The adapter's flags are expanded by the container's shell, so they come from the image, for
    the guarantee the pod's environment states (:func:`env`).
    """
    script = 'exec python3 -m dynamo.vllm "$@" $(python3 -m everserve_vllm.flags --head)'
    return {
        "command": ["/bin/sh", "-c", script, "dynamo.vllm"],
        "args": [
            *engine_args,
            "--nnodes",
            str(nnodes),
            "--node-rank",
            "0",
            "--master-addr",
            rendezvous,
            "--master-port",
            str(port),
        ],
    }


def node(
    engine: str,
    node_env: str,
    slot_env: str,
    engine_args: Sequence[str],
    nnodes: int,
    rendezvous: str,
    port: int,
) -> dict:
    """A GPU node's container command: the node role, asking what it is by its pod's slot.

    It resolves its configuration before it asks, for the guarantee the pod's environment states
    as the head's flags are (:func:`env`), so the ask carries what it will step with.
    """
    return {
        "command": ["python3", "-m", "everserve_vllm.node_main"],
        "args": [
            "--engine",
            engine,
            "--node",
            f"$({node_env})",
            "--slot",
            f"$({slot_env})",
            "--nnodes",
            str(nnodes),
            "--master-addr",
            rendezvous,
            "--master-port",
            str(port),
            "--health-port",
            str(HEALTH_PORT),
            "--leave-within-s",
            str(LEAVE_WITHIN_S),
            *engine_args,
        ],
    }


def probes() -> dict:
    """A node's probes, on its own health port.

    Dynamo's operator probes its system port, which the node role does not serve, and a probe the
    template names replaces the operator's. Startup allows the load (an hour at most); after it,
    /live fails once the serve loop stops turning, and /ready gates nothing a router reads.
    """

    def get(path: str) -> dict:
        return {"httpGet": {"path": path, "port": HEALTH_PORT}}

    return {
        "startupProbe": {**get("/live"), "periodSeconds": 10, "failureThreshold": 360},
        "livenessProbe": {**get("/live"), "periodSeconds": 10, "failureThreshold": 3},
        "readinessProbe": {**get("/ready"), "periodSeconds": 5},
    }
