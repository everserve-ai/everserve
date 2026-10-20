# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The vLLM adapter for Everserve.

Loaded by vLLM through the ``vllm.general_plugins`` entry point in every process (API server,
engine core, each worker). ``register`` is the only thing vLLM calls. It registers the
out-of-tree model classes, lets a serving engine core complete membership changes in its own step
loop (:mod:`everserve_vllm.serving`), and, when the launcher set ``EVERSERVE_GRAPH_SAFE=1``,
routes the tensor-parallel all-reduce through the graph-safe op
(:mod:`everserve_vllm.collective_op`). The connector is selected by engine arguments, and the
executor too, except that ``mp`` names Everserve's in a process its launcher made an engine's
(:mod:`everserve_vllm.backend`): ``python -m everserve_vllm.flags --head`` prints them. Every
patch the plugin puts on vLLM is listed and installed by :mod:`everserve_vllm.patches`.

A vLLM process that no Everserve launcher made an engine's (``EVERSERVE_ENGINE`` unset) is
left as stock vLLM: the plugin registers its classes and changes nothing else, neither the
process's signals nor its environment.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import signal

try:
    __version__ = importlib.metadata.version("everserve-vllm")
except importlib.metadata.PackageNotFoundError:  # imported from a checkout, not installed
    __version__ = "0+unknown"

LOGGERS = ("everserve", "everserve_vllm")
"""The packages whose loggers this process's log carries."""


def log_through_vllm() -> None:
    """Carry our log records on vLLM's handlers where nothing else configured logging.

    The records are this package's and Everserve core's. vLLM configures its own logger
    (``vllm``) in every process it starts and leaves the root logger bare, so a record of ours
    below WARNING reaches no handler in the engine core or a worker, and the head's every INFO
    line (its node's ranks, its start, its step config) is missing from its log. A process that
    configured logging itself (``node_main`` and ``spare_main`` call ``basicConfig``) keeps its
    own. Idempotent.
    """
    if logging.getLogger().handlers:
        return
    vllm_log = logging.getLogger("vllm")
    if not vllm_log.handlers:
        return
    for name in LOGGERS:
        lg = logging.getLogger(name)
        if lg.handlers:
            continue
        for h in vllm_log.handlers:
            lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False


STACKS_SIGNAL = signal.SIGUSR1
"""What dumps every thread's stack in a process of an Everserve engine (``faulthandler``, at C
level): the daemon's stack request and an operator's ``kill -USR1``. No other meaning may share
it: faulthandler's handler replaces any Python handler for it once the process is prepared
(:func:`prepare_engine_process`)."""


def prepare_engine_process() -> None:
    """What every process of an Everserve engine has set before its workers are made.

    The stacks signal (there is no ptrace on rented hosts, so a stack dump is the way in), and
    NCCL's own error handling off: a survivor must not tear itself down when its NCCL peer dies,
    since the runtime aborts the communicator and swaps it. Workers inherit the environment from
    the engine core that spawns them, so this runs in the core before the first spawn.
    """
    import faulthandler

    try:
        faulthandler.register(STACKS_SIGNAL, all_threads=True)
    except (AttributeError, ValueError, RuntimeError):
        pass
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")


def register() -> None:
    """Register the model classes and, in an engine's own process, Everserve's hooks.

    Imports vLLM lazily so this package imports cleanly where vLLM is not installed. Only a
    process an Everserve launcher made an engine's (``EVERSERVE_ENGINE`` set) is prepared
    (:func:`prepare_engine_process`) and hooked; any other vLLM process stays stock.
    """
    import vllm.logger  # noqa: F401 - configures vLLM's handlers, which this process's log uses

    from everserve_vllm import settings

    log_through_vllm()
    if settings.current().engine:
        prepare_engine_process()
    from everserve_vllm.models import register_models

    register_models()
    from everserve_vllm import patches

    # the step hook (A47) and the executor resolution (A48) in every process; the graph-safe
    # all-reduce (A12), the pipeline's routing (A33, A34, A50, A51) and the log-prob warm-up
    # (A41) in a process whose launcher set EVERSERVE_GRAPH_SAFE
    patches.install()
