# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
import os
from importlib.metadata import entry_points

import pytest


def test_plugin_entry_point_is_registered():
    eps = entry_points(group="vllm.general_plugins")
    names = {ep.name: ep.value for ep in eps}
    assert names.get("everserve") == "everserve_vllm:register"


def test_package_imports_without_vllm():
    import everserve_vllm

    assert everserve_vllm.__version__


def test_a_vllm_process_that_is_not_an_engine_s_is_left_as_stock(monkeypatch):
    pytest.importorskip("vllm")
    import everserve_vllm
    from everserve_vllm import settings

    prepared = []
    monkeypatch.setattr(everserve_vllm, "prepare_engine_process", lambda: prepared.append(True))
    monkeypatch.delenv("EVERSERVE_ENGINE", raising=False)
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    settings.reload()
    everserve_vllm.register()
    assert prepared == [], "nothing was prepared in a process no launcher made an engine's"
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ
    monkeypatch.setenv("EVERSERVE_ENGINE", "k")
    settings.reload()
    everserve_vllm.register()
    assert prepared == [True]


def test_preparing_an_engine_s_process_sets_what_its_workers_inherit(monkeypatch):
    import everserve_vllm

    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    monkeypatch.delenv("TORCH_NCCL_ENABLE_MONITORING", raising=False)
    everserve_vllm.prepare_engine_process()
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "0"
    assert os.environ["TORCH_NCCL_ENABLE_MONITORING"] == "0"
