# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A head another command line launches is given ``mp``, and an engine's process resolves it to
Everserve's executor (A48); every other process resolves it as stock. Needs vLLM."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.v1.executor.abstract import Executor  # noqa: E402
from vllm.v1.executor.multiproc_executor import MultiprocExecutor  # noqa: E402

from everserve_vllm import backend, settings  # noqa: E402


def config(name):
    return SimpleNamespace(parallel_config=SimpleNamespace(distributed_executor_backend=name))


def test_mp_is_everserve_s_executor_in_an_engine_s_process_and_stock_elsewhere():
    from everserve_vllm.executor import EverserveExecutor

    assert backend.install() and backend.install()  # once, however often the plugin loads
    try:
        settings.update(engine="proof", graph_safe=True)
        assert Executor.get_class(config("mp")) is EverserveExecutor
        settings.update(engine="", graph_safe=True)
        assert Executor.get_class(config("mp")) is MultiprocExecutor
        settings.update(engine="proof", graph_safe=False)
        assert Executor.get_class(config("mp")) is MultiprocExecutor
    finally:
        settings.update(engine="", graph_safe=False)
