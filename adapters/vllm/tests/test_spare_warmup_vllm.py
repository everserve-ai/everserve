# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare's kernel warm-up on vLLM's own groups at the pin: the groups the spare builds, our
pipeline routing installed on vLLM's ``GroupCoordinator``, and a warm-up step that sends its
activations on, as the first stage of a pipeline does. Needs vLLM."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

import torch  # noqa: E402
from vllm.distributed import parallel_state as ps  # noqa: E402
from vllm.distributed.parallel_state import GroupCoordinator  # noqa: E402
from vllm.v1.worker import gpu_worker  # noqa: E402

from everserve_vllm import pipeline, settings, spare  # noqa: E402

GROUPS = ("_WORLD", "_TP", "_ETP", "_EP", "_PP", "_PCP", "_DCP", "_DP", "_EPLB", "_ENGRAM_DP")


@pytest.fixture
def spare_groups(monkeypatch):
    """vLLM's process-wide groups and our routing on its class, as a spare's are; all undone after.

    The groups are the spare's own stand-ins (:func:`spare.point_groups`) for rank 0 of a world
    of two pipeline stages, and the routing is :func:`pipeline.install` on vLLM's class, as a
    worker's process installs it.
    """
    for name in (*GROUPS, "_NODE_COUNT"):
        if hasattr(ps, name):
            monkeypatch.setattr(ps, name, getattr(ps, name))
    monkeypatch.setattr(ps, "_groups", dict(ps._groups))
    for name in (*pipeline.PIPELINE_METHODS, "make_sibling_device_group"):
        monkeypatch.setattr(GroupCoordinator, name, getattr(GroupCoordinator, name))
    monkeypatch.setattr(torch.distributed, "broadcast", torch.distributed.broadcast)
    monkeypatch.setattr(pipeline, "_ORIGINALS", {})
    monkeypatch.setattr(pipeline, "_ORIGINAL_BROADCAST", None)
    pipeline.install(GroupCoordinator)
    monkeypatch.setattr(settings, "_CURRENT", settings.Settings(layout=(1, 2)))
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))
    spare.point_groups(config, rank=0, local_rank=0)
    for name in ("kernel_warmup", "warmup_kernels"):
        if hasattr(gpu_worker, name):
            monkeypatch.setattr(gpu_worker, name, getattr(gpu_worker, name))
    return ps.get_pp_group()


def test_the_first_stage_s_warm_up_sends_nothing_through_vllm_s_own_send(spare_groups):
    """A spare has no stage after it to send to, and its stand-in group has no transport.

    The first stage's warm-up steps run with the stage alone, and their sends are dropped. Before
    the fix, the first stage ran vLLM's warm-up as is, and its send reached vLLM's own
    ``isend_tensor_dict`` on a group built without ``__init__``. At this pin that raised
    ``AttributeError: ... use_cpu_custom_send_recv``, and the spare for rank 0 died in its
    warm-up.
    """
    pp = spare_groups
    assert isinstance(pp, GroupCoordinator) and pp.is_first_rank and pp.world_size == 2
    sent = []

    def warmup_kernels(model_runner, execute_model, sample_tokens):
        execute_model("scheduler output")

    def step(scheduler_output):
        hidden = {"hidden_states": torch.zeros(2, 4), "residual": torch.zeros(2, 4)}
        sent.append(ps.get_pp_group().isend_tensor_dict(hidden))

    gpu_worker.warmup_kernels = warmup_kernels
    spare._warm_pipeline_stage()
    gpu_worker.warmup_kernels(object(), step, None)
    assert len(sent) == 1 and all(isinstance(h, pipeline.Done) for h in sent[0])
    assert pipeline._LOCAL is None  # the stage is alone for the warm-up only
