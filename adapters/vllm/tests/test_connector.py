# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The vLLM connector class itself; skipped where vLLM is not installed."""

from types import SimpleNamespace

import pytest

vllm = pytest.importorskip("vllm")


def test_the_scheduler_metadata_carries_the_step_record():
    from everserve_vllm.connector import EverserveConnector, EverserveMeta

    out = SimpleNamespace(
        num_scheduled_tokens={"a": 1, "b": 2048},
        scheduled_new_reqs=[SimpleNamespace(req_id="b", num_computed_tokens=0)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a"], num_computed_tokens=[2048]),
        preempted_req_ids={"c"},
    )
    meta = EverserveConnector.build_connector_meta(None, out)
    assert isinstance(meta, EverserveMeta)
    assert dict(meta.record.committed) == {"a": 2049, "b": 2048}
    assert meta.record.preempted == ("c",)
