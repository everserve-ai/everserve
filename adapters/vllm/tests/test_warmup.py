# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The log-prob kernel is warmed at every block size a request can reach, on every rank."""

import sys
import types
from types import SimpleNamespace

from everserve_vllm import warmup


def test_one_count_per_block_size_up_to_the_engine_s_bound_skipping_vllm_s_own():
    # blocks: 0 -> 1, 1 -> 2, 3 -> 4, 7 -> 8 (vLLM's five log-probs: six entries, block 8), 15 -> 16
    assert warmup.logprob_counts(20) == [0, 1, 3, 15]
    assert warmup.logprob_counts(5) == [0, 1, 3]
    assert warmup.logprob_counts(0) == [0]
    assert warmup.logprob_counts(7, covered=0) == [1, 3, 7]
    wide = warmup.logprob_counts(10_000)
    assert wide[-1] == warmup.MAX_TOPK_BLOCK - 1 and len(wide) == 10


def test_the_install_runs_vllm_s_pass_then_one_per_count_and_restores_the_params(monkeypatch):
    class SamplingParams:
        def __init__(self, logprobs=5):
            self.logprobs = logprobs

        @staticmethod
        def for_sampler_warmup():
            return SamplingParams()

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    base = SamplingParams.for_sampler_warmup
    passes = []

    def warmup_kernels(model_runner, execute_model, sample_tokens):
        passes.append(SamplingParams.for_sampler_warmup().logprobs)
        return "warm"

    gw = types.ModuleType("gpu_worker")
    gw.warmup_kernels = warmup_kernels
    warmup.install(gw, logprobs=True)
    warmup.install(gw, logprobs=True)  # idempotent: one wrapper
    assert gw.warmup_kernels.__name__ == "warmup_kernels"
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_logprobs=20))
    )
    assert gw.warmup_kernels(runner, "execute", "sample") == "warm"
    assert passes == [5, 0, 1, 3, 15]
    assert SamplingParams.for_sampler_warmup is base
    # an engine that allows no log-probs beyond the sampled token's still warms that block
    passes.clear()
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_logprobs=0))
    )
    gw.warmup_kernels(runner, "execute", "sample")
    assert passes == [5, 0]


def test_a_spare_s_stage_alone_warm_up_wraps_the_log_prob_passes(monkeypatch):
    """The plugin's wrap is installed first; the spare's stage-alone wrap goes around it."""
    from everserve_vllm import pipeline, spare

    class SamplingParams:
        def __init__(self, logprobs=5):
            self.logprobs = logprobs

        @staticmethod
        def for_sampler_warmup():
            return SamplingParams()

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = SamplingParams
    ps = types.ModuleType("parallel_state")
    ps.get_pp_group = lambda: SimpleNamespace(is_first_rank=False, rank_in_group=1, world_size=2)
    distributed = types.ModuleType("vllm.distributed")
    distributed.parallel_state = ps
    vllm.distributed = distributed
    gw = types.ModuleType("gpu_worker")
    seen = []

    def warmup_kernels(model_runner, execute_model, sample_tokens):
        seen.append((SamplingParams.for_sampler_warmup().logprobs, pipeline._LOCAL is model_runner))

    gw.warmup_kernels = warmup_kernels
    v1 = types.ModuleType("vllm.v1")
    v1w = types.ModuleType("vllm.v1.worker")
    for name, mod in {
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.parallel_state": ps,
        "vllm.v1": v1,
        "vllm.v1.worker": v1w,
        "vllm.v1.worker.gpu_worker": gw,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    warmup.install(gw, logprobs=True)
    spare._warm_pipeline_stage()
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_logprobs=3))
    )
    gw.warmup_kernels(runner, lambda so: None, None)
    assert seen == [(5, True), (0, True), (1, True), (3, True)]
