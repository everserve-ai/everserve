# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The executor of a head another command line launches: vLLM's ``mp``, made Everserve's.

A head launched by its own command line (NVIDIA Dynamo's ``python -m dynamo.vllm``, decision 0041)
names its executor by a flag. Named by its qualified name, Everserve's executor is refused for a
pipeline-parallel engine: vLLM 0.31's check of the engine's arguments reads ``supports_pp`` from
the backend as given, and a name has none (``engine/arg_utils.py``, ``_check_feature_supported``).
So the head is given ``--distributed-executor-backend mp``, which the check takes, and this hook,
installed at plugin load (``everserve_vllm.register``, the registry's A48 in
:mod:`everserve_vllm.patches`), resolves ``mp`` to
:class:`everserve_vllm.executor.EverserveExecutor` in a process its launcher made an engine's
(``EVERSERVE_ENGINE`` with ``EVERSERVE_GRAPH_SAFE``). Every other process resolves ``mp`` as
stock. Ledger A48.

Its exit, upstream: the check asks the class a qualified backend names for ``supports_pp`` (as it
asks a class given directly), a one-line change to vLLM; the head is then given Everserve's
executor by name, and this wrap is retired.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

log = logging.getLogger(__name__)


def install() -> bool:
    """Wrap ``Executor.get_class`` once; False where vLLM's executor is not importable.

    The registry's A48 (:data:`everserve_vllm.patches.EXECUTOR_CLASS`).
    """
    from everserve_vllm import patches

    return patches.EXECUTOR_CLASS.apply()


def resolving(get_class: Any) -> Any:
    """``get_class`` with ``mp`` resolved to Everserve's executor in an engine's process.

    Marked ``everserve_backend``, so the registry sees it on; the original is its
    ``__wrapped__``.
    """
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    @functools.wraps(get_class)
    def resolved(vllm_config: Any) -> type:
        cls = get_class(vllm_config)
        if cls is MultiprocExecutor and ours():
            from everserve_vllm.executor import EverserveExecutor

            log.info("the engine's mp executor is Everserve's")
            return EverserveExecutor
        return cls

    resolved.everserve_backend = True  # type: ignore[attr-defined]
    return resolved


def ours() -> bool:
    """Whether this process's launcher made it an engine's: an engine named, graph-safe."""
    from everserve_vllm import settings

    s = settings.current()
    return bool(s.engine) and s.graph_safe
