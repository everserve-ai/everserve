# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The patch registry: every permanent patch goes on through it, in order and under its
condition, comes off through it, and is listed while it is on; the ledger's process-wide rows
are all in it and its docstring names the instance patches it does not install."""

import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from everserve_vllm import collective_op, patches, pipeline, serving, settings, warmup

LEDGER = Path(__file__).resolve().parents[1] / "PRIVATE-INTERNALS.md"


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture
def vllm(monkeypatch):
    """Stand-ins for every vLLM module the registry resolves, and the records cleared."""

    class EngineCore:
        def __init__(self):
            self.step_fn = self.step

        def step(self):
            return {}, True

    class EngineCoreProc(EngineCore):
        def _process_engine_step(self):
            return self.step_fn()

    class MultiprocExecutor:
        pass

    class Executor:
        @staticmethod
        def get_class(vllm_config):
            return MultiprocExecutor

    class GroupCoordinator:
        unique_name, world_size, rank_in_group = "pp:0", 2, 0

        def all_reduce(self, input_):
            return "stock all_reduce"

        def send_tensor_dict(self, tensor_dict, dst=None, **kw):
            return "stock send"

        def recv_tensor_dict(self, src=None, **kw):
            return "stock recv"

        def isend_tensor_dict(self, tensor_dict, dst=None, **kw):
            return "stock isend"

        def irecv_tensor_dict(self, src=None, **kw):
            return "stock irecv"

        def make_sibling_device_group(self, group_desc=None):
            return "stock sibling"

    def warmup_kernels(model_runner, execute_model, sample_tokens):
        return "stock warm"

    def kernel_warmup(worker):
        return "stock kernel warm"

    registered = []
    mods = {
        "vllm": _module("vllm"),
        "vllm.v1": _module("vllm.v1"),
        "vllm.v1.engine": _module("vllm.v1.engine"),
        "vllm.v1.engine.core": _module(
            "vllm.v1.engine.core", EngineCore=EngineCore, EngineCoreProc=EngineCoreProc
        ),
        "vllm.v1.executor": _module("vllm.v1.executor"),
        "vllm.v1.executor.abstract": _module("vllm.v1.executor.abstract", Executor=Executor),
        "vllm.v1.executor.multiproc_executor": _module(
            "vllm.v1.executor.multiproc_executor", MultiprocExecutor=MultiprocExecutor
        ),
        "vllm.distributed": _module("vllm.distributed"),
        "vllm.distributed.parallel_state": _module(
            "vllm.distributed.parallel_state", GroupCoordinator=GroupCoordinator
        ),
        "vllm.utils": _module("vllm.utils"),
        "vllm.utils.torch_utils": _module(
            "vllm.utils.torch_utils",
            direct_register_custom_op=lambda *a, **k: registered.append(a[0]),
        ),
        "vllm.v1.worker": _module("vllm.v1.worker"),
        "vllm.v1.worker.gpu_worker": _module(
            "vllm.v1.worker.gpu_worker", warmup_kernels=warmup_kernels, kernel_warmup=kernel_warmup
        ),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(patches, "_OWNERS", {})
    monkeypatch.setattr(patches, "_BEFORE", {})
    monkeypatch.setattr(pipeline, "_ORIGINALS", {})
    monkeypatch.setattr(pipeline, "_ORIGINAL_BROADCAST", None)
    monkeypatch.setattr(collective_op, "_ORIG_ALL_REDUCE", None)
    monkeypatch.setattr(collective_op, "_LIB", None)
    monkeypatch.setattr(torch.distributed, "broadcast", torch.distributed.broadcast)
    monkeypatch.setattr(settings, "_CURRENT", settings.Settings(engine="rig", graph_safe=True))
    return SimpleNamespace(
        core=mods["vllm.v1.engine.core"],
        abstract=mods["vllm.v1.executor.abstract"],
        group=GroupCoordinator,
        gpu_worker=mods["vllm.v1.worker.gpu_worker"],
        registered=registered,
        originals={
            "get_class": Executor.get_class,
            "all_reduce": GroupCoordinator.all_reduce,
            **{n: getattr(GroupCoordinator, n) for n in pipeline.PIPELINE_METHODS},
            "make_sibling_device_group": GroupCoordinator.make_sibling_device_group,
            "broadcast": torch.distributed.broadcast,
            "warmup_kernels": warmup_kernels,
            "kernel_warmup": kernel_warmup,
        },
    )


def _targets(v):
    """Every patched attribute as it stands now, by the originals' names."""
    out = {
        "get_class": v.abstract.Executor.get_class,
        "all_reduce": v.group.all_reduce,
        "broadcast": torch.distributed.broadcast,
        "warmup_kernels": v.gpu_worker.warmup_kernels,
        "kernel_warmup": v.gpu_worker.kernel_warmup,
    }
    for n in (*pipeline.PIPELINE_METHODS, "make_sibling_device_group"):
        out[n] = getattr(v.group, n)
    return out


def test_install_puts_every_patch_on_in_order_and_uninstall_puts_each_original_back(vllm):
    v = vllm
    assert patches.installed() == []
    on = patches.install()
    assert [p.ids for p in on] == [
        ("A47",),
        ("A48",),
        ("A12",),
        ("A33", "A34", "A50", "A51"),
        ("A41",),
    ]
    assert on == list(patches.PATCHES) and patches.installed() == on
    # each target is patched
    assert isinstance(vars(v.core.EngineCore)["step_fn"], serving.ServedStep)
    assert serving.installed()
    now = _targets(v)
    for name, original in v.originals.items():
        if name == "kernel_warmup":
            assert now[name] is original  # A41 wraps warmup_kernels alone
        else:
            assert now[name] is not original, name
    assert v.abstract.Executor.get_class.everserve_backend
    assert v.registered == ["all_reduce_into"] and collective_op._ORIG_ALL_REDUCE is not None
    assert pipeline._ORIGINALS["send_tensor_dict"] is v.originals["send_tensor_dict"]
    assert warmup.marks(v.gpu_worker, "warmup_kernels") == {warmup.LOGPROBS}
    # a second install is a no-op: the same objects stay on, the op is not registered again
    assert patches.install() == on
    assert _targets(v) == now and v.registered == ["all_reduce_into"]
    # uninstall restores each original
    patches.uninstall()
    assert patches.installed() == []
    assert "step_fn" not in vars(v.core.EngineCore) and not serving.installed()
    for name, original in v.originals.items():
        assert _targets(v)[name] is original, name
    assert pipeline._ORIGINALS == {} and pipeline._ORIGINAL_BROADCAST is None
    assert collective_op._ORIG_ALL_REDUCE is None
    patches.uninstall()  # idempotent
    assert _targets(v)["get_class"] is v.originals["get_class"]
    # and it all goes on again
    assert patches.install() == on and _targets(v)["all_reduce"] is not v.originals["all_reduce"]


def test_a_process_that_is_not_graph_safe_gets_the_two_every_process_patches_only(vllm):
    settings._CURRENT = settings.Settings(engine="rig", graph_safe=False)
    assert [p.ids for p in patches.install()] == [("A47",), ("A48",)]
    assert _targets(vllm)["all_reduce"] is vllm.originals["all_reduce"]
    assert torch.distributed.broadcast is vllm.originals["broadcast"]
    assert _targets(vllm)["warmup_kernels"] is vllm.originals["warmup_kernels"]
    patches.uninstall()
    assert patches.installed() == []


def test_where_vllm_is_not_installed_nothing_goes_on(monkeypatch):
    monkeypatch.setattr(patches, "_OWNERS", {})
    monkeypatch.setattr(settings, "_CURRENT", settings.Settings(engine="rig", graph_safe=True))
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.core",
        "vllm.v1.executor",
        "vllm.v1.executor.abstract",
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_worker",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    assert patches.install() == [] and patches.installed() == []
    patches.uninstall()


def test_the_pipeline_patch_carries_the_failure_report_in_one_wrap(vllm):
    """A51 is folded into A33's wrap: one function on the class, the report around the route."""
    reported = []

    def failing(self, *args, **kwargs):
        raise RuntimeError("NCCL error: remote process exited")

    vllm.group.isend_tensor_dict = failing
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "_report_peers_lost", reported.append)
        patches.install()
        g = vllm.group()
        with pytest.raises(RuntimeError, match="NCCL"):
            g.isend_tensor_dict({})
        assert reported == ["isend_tensor_dict: RuntimeError: NCCL error: remote process exited"]
        assert vllm.group.isend_tensor_dict.__wrapped__.__name__ == "isend_tensor_dict"
        patches.uninstall()


def test_every_process_wide_ledger_row_is_in_the_registry_and_every_instance_row_is_named():
    rows = []
    section = ""
    for line in LEDGER.read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif section == "Patches" and line.startswith("| A"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append((re.split(r",\s*", cells[0]), cells[1], cells[3]))
    assert rows
    in_registry = {i for p in patches.PATCHES for i in p.ids}
    for ids, anchor, scope in rows:
        registry = scope.startswith("process-wide") or "every worker process" in scope
        for i in ids:
            if registry:
                assert i in in_registry, f"{i} ({scope}) is not in patches.PATCHES"
                assert "`patches.py:" in anchor, f"{i}'s Patches row does not anchor patches.py"
            else:
                assert re.search(rf"\b{i}\b", patches.__doc__), f"{i} ({scope}) not named"
