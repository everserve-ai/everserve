# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The sampler's log-prob kernel warmed at every count a request may ask for (ledger A41).

vLLM's kernel warm-up (``v1/worker/gpu/warmup.py``) drives the sampler once, with
``SamplingParams.for_sampler_warmup()``: five log-probs. The log-prob kernel
(``_topk_log_softmax_kernel``, ``v1/worker/gpu/sample/logprob.py``) is specialized on its top-k
block, the next power of two above the count of entries it emits (the sampled token plus the
requested log-probs), so a request at another count compiles another specialization at its
first step: 0.13 s on a serving rank's first step with log-probs and on the spare's first step
after a resume alike, the one kernel vLLM's JIT monitor still reports "during inference". Here
the warm-up runs once more
per block size up to the engine's ``max_logprobs``, on every rank (the plugin installs it in
each worker process; a spare's stage-alone warm-up wraps it too), so no process compiles the
kernel in inference.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Any

log = logging.getLogger(__name__)

MAX_TOPK_BLOCK = 1024
"""The kernel's widest block (``logprob.py``'s ``_MAX_TOPK_BLOCK``): counts past it share it."""
COVERED = 5
"""The log-prob count vLLM's own warm-up runs with (``SamplingParams.for_sampler_warmup``)."""


def _block(count: int) -> int:
    """The kernel's top-k block for ``count`` log-probs: the sampled token plus the count."""
    entries = count + 1
    block = 1
    while block < entries:
        block *= 2
    return min(block, MAX_TOPK_BLOCK)


def logprob_counts(max_logprobs: int, covered: int = COVERED) -> list[int]:
    """The log-prob counts to warm beyond vLLM's own: one per block size a request can reach.

    Counts ``k`` with ``k + 1`` a power of two, from 0 (the sampled token alone) up to
    ``max_logprobs`` (the engine's bound on a request), skipping the block ``covered`` already
    compiled and stopping at the widest block.
    """
    seen = {_block(covered)}
    counts: list[int] = []
    k = 0
    while k <= max_logprobs:
        b = _block(k)
        if b not in seen:
            seen.add(b)
            counts.append(k)
        if b >= MAX_TOPK_BLOCK:
            break
        k = 2 * (k + 1) - 1
    return counts


LOGPROBS = "_everserve_logprobs"
"""Marks the wrap that adds the log-prob passes to ``warmup_kernels`` (A41, every rank)."""
STAGE_ALONE = "_everserve_stage_alone"
"""Marks the wrap that runs an entry point on the spare's stage alone (A38, the spare)."""
ENTRY_POINTS = ("kernel_warmup", "warmup_kernels")
"""The warm-up entry points bound in the worker module (``compile_or_warm_up_model`` calls
both)."""


def _ours(fn: Any) -> bool:
    return bool(getattr(fn, LOGPROBS, False) or getattr(fn, STAGE_ALONE, False))


def marks(gpu_worker: Any, name: str) -> set[str]:
    """The markers on ``gpu_worker.<name>``, through every wrap of ours down to vLLM's own."""
    out: set[str] = set()
    fn = getattr(gpu_worker, name, None)
    while _ours(fn):
        out.update(m for m in (LOGPROBS, STAGE_ALONE) if getattr(fn, m, False))
        fn = fn.__wrapped__
    return out


def _base(fn: Any) -> Any:
    """The entry point as vLLM bound it, under every wrap of ours."""
    while _ours(fn):
        fn = fn.__wrapped__
    return fn


def install(
    gpu_worker: Any, logprobs: bool = False, stage_alone: bool = False, timed: Any = None
) -> None:
    """Wrap the warm-up entry points as asked, over what is on them already.

    ``logprobs`` wraps ``warmup_kernels`` with the log-prob passes (A41, the registry's, every
    rank); ``stage_alone`` wraps both entry points to run on the spare's stage alone (A38, the
    spare's), outside the passes whichever call came first, ``timed`` being the spare's timing
    block by the entry point's name. Idempotent: a wrap already on stays, one asked for again is
    not added; a wrap to add is put on with the ones already there, rebuilt over vLLM's own in
    that fixed order.
    """
    for name in ENTRY_POINTS:
        fn = getattr(gpu_worker, name, None)
        if fn is None:
            continue
        on = marks(gpu_worker, name)
        wanted = set(on)
        if logprobs and name == "warmup_kernels":
            wanted.add(LOGPROBS)
        if stage_alone:
            wanted.add(STAGE_ALONE)
        if wanted == on:
            continue
        out = _base(fn)
        if LOGPROBS in wanted:
            out = _with_logprob_passes(out)
        if STAGE_ALONE in wanted:
            out = _on_stage_alone(name, out, timed)
        setattr(gpu_worker, name, out)


def uninstall(gpu_worker: Any) -> None:
    """Put vLLM's own entry points back: every wrap of ours, the spare's with the registry's."""
    for name in ENTRY_POINTS:
        fn = getattr(gpu_worker, name, None)
        if fn is not None and _ours(fn):
            setattr(gpu_worker, name, _base(fn))


def _with_logprob_passes(original: Any) -> Any:
    """``warmup_kernels``: vLLM's pass, then one per uncovered log-prob count.

    Each extra pass is the same scheduler-realistic steps with ``for_sampler_warmup`` returning
    that count; the warm-up removes its requests at its end, so passes compose.
    """

    @functools.wraps(original)
    def warmup_kernels(model_runner: Any, execute_model: Any, sample_tokens: Any) -> Any:
        from vllm import SamplingParams

        out = original(model_runner, execute_model, sample_tokens)
        bound = getattr(getattr(model_runner.vllm_config, "model_config", None), "max_logprobs", 0)
        counts = logprob_counts(int(bound or 0))
        if not counts:
            return out
        base = SamplingParams.for_sampler_warmup

        def at(count: int) -> Any:
            def for_sampler_warmup() -> Any:
                p = base()
                p.logprobs = count
                return p

            return staticmethod(for_sampler_warmup)

        log.info("warming the log-prob kernel at %s log-probs (max %s)", counts, bound)
        try:
            for count in counts:
                SamplingParams.for_sampler_warmup = at(count)
                original(model_runner, execute_model, sample_tokens)
        finally:
            SamplingParams.for_sampler_warmup = staticmethod(base)
        return out

    setattr(warmup_kernels, LOGPROBS, True)
    return warmup_kernels


def _on_stage_alone(name: str, original: Any, timed: Any) -> Any:
    """``original``, the entry point ``name``, run on the spare's stage alone (A38).

    Both entry points drive the worker's ``execute_model`` with synthetic steps; on a stage past
    the first every such step would wait on the stage before, which a spare has no peer for, and
    on the first the send would go to vLLM's own on the spare's stand-in group, which has no
    transport. So the stage runs alone (:func:`everserve_vllm.pipeline.local_stage`: a receive
    yields the step's tensors made here, a send is dropped), ``warmup_kernels``'s
    ``execute_model`` wrapped to keep the step's scheduler output
    (:func:`everserve_vllm.pipeline.stepping`), and the runner's kernels compile as they would
    on a serving rank. An engine of one stage warms up as vLLM runs it.
    """

    @functools.wraps(original)
    def warmup(*args: Any, **kwargs: Any) -> Any:
        block = timed(f"{name}_s") if timed is not None else contextlib.nullcontext()
        with block:
            return _on_stage(name, original, *args, **kwargs)

    setattr(warmup, STAGE_ALONE, True)
    return warmup


def _on_stage(name: str, original: Any, *args: Any, **kwargs: Any) -> Any:
    from vllm.distributed import parallel_state as ps

    from everserve_vllm import pipeline

    pp = ps.get_pp_group()
    if pp.world_size == 1:
        return original(*args, **kwargs)
    if name == "warmup_kernels":
        runner, execute_model, *rest = args
        args = (runner, pipeline.stepping(execute_model), *rest)
    else:
        runner = args[0].model_runner
    log.info("spare: %s on pipeline stage %d alone", name, pp.rank_in_group)
    with pipeline.local_stage(runner):
        return original(*args, **kwargs)
