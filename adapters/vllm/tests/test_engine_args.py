# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Every process of an engine takes vLLM's own engine arguments, and the adapter's requirements
over them."""

import json

import pytest

from everserve_vllm.engine_args import CONNECTOR, identity


def test_the_model_and_widths_are_read_from_vllm_s_own_arguments_without_vllm():
    i = identity(["--model", "/m", "-tp", "4", "--pipeline-parallel-size=2", "--seed", "0"])
    assert (i.model, i.tp, i.pp) == ("/m", 4, 2)
    assert identity(["--model", "q"]) == identity(["--model", "q", "-tp", "1", "-pp", "1"])
    with pytest.raises(SystemExit, match="no model"):
        identity(["-tp", "2"])


def test_a_node_or_spare_builds_its_configuration_under_the_head_s_usage_context(monkeypatch):
    # vLLM's batch defaults follow the usage context: a node under another one steps with other
    # buffers than its head, and the stages' first sampled-token broadcast never matches
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("vLLM infers its device from a GPU it sees")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    from everserve_vllm import engine_args

    seen = {}

    def create(self, usage_context=None, headless=False):
        seen.update(
            usage_context=usage_context,
            headless=headless,
            backend=self.distributed_executor_backend,
        )

    monkeypatch.setattr(EngineArgs, "create_engine_config", create)
    engine_args.config(["--model", "m", "-pp", "2"], headless=True)
    # vllm serve's and dynamo.vllm's context alike
    assert seen == {
        "usage_context": UsageContext.OPENAI_API_SERVER,
        "headless": True,
        "backend": "mp",
    }


def test_the_flags_a_head_takes_are_the_adapter_s_requirements_in_vllm_s_spelling():
    pytest.importorskip("vllm")
    from everserve_vllm.flags import flags

    head = flags("exact", head=True)
    assert head[:2] == ["--distributed-executor-backend", "mp"]  # resolved to ours (A48)
    from everserve_vllm.collective_op import OP

    compiled = json.loads(head[head.index("--compilation-config") + 1])
    assert OP in compiled["splitting_ops"]  # the graph-safe all-reduce splits the graph
    assert compiled["inductor_compile_config"]["deterministic"] is True  # exact
    assert json.loads(head[head.index("--kv-transfer-config") + 1]) == CONNECTOR
    assert "--disable-custom-all-reduce" in head
    assert all(" " not in f for f in head)  # one shell word each, unquoted
    continuous = flags("continuous")
    assert "--distributed-executor-backend" not in continuous
    assert "inductor_compile_config" not in json.loads(
        continuous[continuous.index("--compilation-config") + 1]
    )
