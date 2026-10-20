# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Every permanent patch the adapter puts on vLLM or torch, in one place.

A patch here is process-wide: a class attribute or a module global replaced for the life of the
process. :func:`install` puts each on, in the order below and under its condition (``register``
calls it in every vLLM process); :func:`uninstall` puts each original back; :func:`installed`
lists what is on. The modules keep their logic (the wrappers, the descriptor) and the originals
where their wrappers read them (``pipeline._ORIGINALS``, ``pipeline._ORIGINAL_BROADCAST``,
``collective_op._ORIG_ALL_REDUCE``); this module does the assignments and keeps the record of
what they were put on.

* A47: ``vllm.v1.engine.core.EngineCore.step_fn``, every process;
* A48: ``vllm.v1.executor.abstract.Executor.get_class``, every process;
* A12: ``vllm.distributed.parallel_state.GroupCoordinator.all_reduce``, graph-safe;
* A33, A34, A50, A51: ``GroupCoordinator.send_tensor_dict``, ``recv_tensor_dict``,
  ``isend_tensor_dict``, ``irecv_tensor_dict``, ``make_sibling_device_group``, and
  ``torch.distributed.broadcast``, graph-safe;
* A41: ``vllm.v1.worker.gpu_worker.warmup_kernels``, graph-safe.

"Graph-safe" is ``settings.current().graph_safe``: the launcher set ``EVERSERVE_GRAPH_SAFE=1``.
A51's wrap (a failed transfer reports the column's peers lost) is folded into A33's, so each
pipeline method is wrapped once with both behaviours, the report around the exchange.

Instance patches, not installed here (each is one object's, scoped by its owner; the ledger's
Patches section has the row): A25 and A26 (the spare's stand-in groups, ``spare.point_groups``),
A28 and A42 (the spare's load), A36 (the spare's FlashInfer tactics), A38 (the spare's warm-up
on its stage alone, put over A41's wrap by ``warmup.install`` from ``spare._warm_pipeline_stage``),
A43 (the spare's dummy runs), A39 (``execute_model`` on the worker), A58 (the RPC target's
steps), A11a (the replaying scheduler), A19 and A29 (the worker's queues), A45 (the handle's
death writer), A18 and A15, A49 (the groups' communicators).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from everserve_vllm import backend, collective_op, pipeline, serving, settings, warmup

log = logging.getLogger(__name__)

EVERY_PROCESS = "every process"
GRAPH_SAFE = "graph-safe"

_OWNERS: dict[str, Any] = {}
"""What each patch is on (the module or class), by the patch's name, while it is on."""
_BEFORE: dict[str, Any] = {}
"""What a class held before a patch whose module keeps no original (A47), for the restore."""
ABSENT = object()


class Patch:
    """One permanent patch: its ledger rows, target, scope, and how it goes on and comes off.

    Each fills in four hooks: ``_resolve`` finds the owner in vLLM (None where vLLM lacks it),
    ``_present`` reads whether the patch is on that owner, ``_put`` and ``_restore`` do the
    assignments.
    """

    name: str
    ids: tuple[str, ...]
    target: str
    scope: str
    why: str
    upstream: str

    def _resolve(self) -> Any:
        raise NotImplementedError

    def _present(self, owner: Any) -> bool:
        raise NotImplementedError

    def _put(self, owner: Any) -> None:
        raise NotImplementedError

    def _restore(self, owner: Any) -> None:
        raise NotImplementedError

    def apply(self, owner: Any = None) -> bool:
        """Put the patch on ``owner`` (resolved from vLLM when not given) unless it is on already.

        False where vLLM lacks the target (this package without the engine extra).
        """
        if owner is None:
            owner = self._resolve()
            if owner is None:
                return False
        if not self._present(owner):
            self._put(owner)
        _OWNERS[self.name] = owner
        return True

    def remove(self) -> None:
        """Put the original back on what the patch was put on; nothing where it is not on."""
        owner = _OWNERS.pop(self.name, None)
        if owner is not None and self._present(owner):
            self._restore(owner)

    def is_present(self, owner: Any = None) -> bool:
        """Whether the patch is on ``owner`` (resolved from vLLM when not given)."""
        if owner is None:
            owner = self._resolve()
        return owner is not None and self._present(owner)

    def __repr__(self) -> str:
        """The ledger rows and the target."""
        return f"Patch({', '.join(self.ids)} on {self.target})"


class StepFn(Patch):
    """A47: :class:`~everserve_vllm.serving.ServedStep` on ``EngineCore.step_fn``."""

    name = "step_fn"
    ids = ("A47",)
    target = "vllm.v1.engine.core.EngineCore.step_fn"
    scope = EVERY_PROCESS
    why = (
        "a serving engine core completes membership changes in its own step loop; vLLM loads "
        "the plugin inside EngineCore.__init__ before step_fn is set, so a data descriptor on "
        "the class takes that engine core's own assignment"
    )
    upstream = "an executor given its engine core after construction, or a step-loop hook"

    def _resolve(self) -> Any:
        try:
            from vllm.v1.engine import core
        except ImportError:
            return None
        return core

    def _present(self, core: Any) -> bool:
        return isinstance(vars(core.EngineCore).get("step_fn"), serving.ServedStep)

    def _put(self, core: Any) -> None:
        _BEFORE[self.name] = vars(core.EngineCore).get("step_fn", ABSENT)
        core.EngineCore.step_fn = serving.ServedStep(core)

    def _restore(self, core: Any) -> None:
        before = _BEFORE.pop(self.name, ABSENT)
        if before is ABSENT:
            del core.EngineCore.step_fn
        else:
            core.EngineCore.step_fn = before


class ExecutorClass(Patch):
    """A48: ``Executor.get_class`` resolves ``mp`` to Everserve's executor in an engine's."""

    name = "executor_class"
    ids = ("A48",)
    target = "vllm.v1.executor.abstract.Executor.get_class"
    scope = EVERY_PROCESS
    why = (
        "a head another command line launches names its executor by a flag, and a qualified "
        "name is refused for a pipeline-parallel engine (supports_pp read off the name)"
    )
    upstream = "the check asking the class a qualified name resolves to for supports_pp"

    def _resolve(self) -> Any:
        try:
            from vllm.v1.executor import abstract
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor  # noqa: F401
        except ImportError:
            return None
        return abstract

    def _present(self, abstract: Any) -> bool:
        return bool(getattr(abstract.Executor.get_class, "everserve_backend", False))

    def _put(self, abstract: Any) -> None:
        abstract.Executor.get_class = staticmethod(backend.resolving(abstract.Executor.get_class))

    def _restore(self, abstract: Any) -> None:
        abstract.Executor.get_class = staticmethod(abstract.Executor.get_class.__wrapped__)


class AllReduce(Patch):
    """A12: the TP all-reduce through ``everserve::all_reduce_into``."""

    name = "all_reduce"
    ids = ("A12",)
    target = "vllm.distributed.parallel_state.GroupCoordinator.all_reduce"
    scope = GRAPH_SAFE
    why = (
        "an in-place reduce into a buffer the traced code owns, a PIECEWISE split point, so no "
        "graph holds a communicator and a captured graph follows a membership change"
    )
    upstream = "an all-reduce the executor can supply per group"

    def _resolve(self) -> Any:
        try:
            from vllm.distributed.parallel_state import GroupCoordinator
        except ImportError:
            return None
        return GroupCoordinator

    def _present(self, group_cls: Any) -> bool:
        return collective_op._ORIG_ALL_REDUCE is not None

    def _put(self, group_cls: Any) -> None:
        collective_op.register_op()  # our own library; it stays registered across a restore
        group_cls.all_reduce = collective_op.routed_all_reduce(group_cls.all_reduce)

    def _restore(self, group_cls: Any) -> None:
        group_cls.all_reduce = collective_op.release_all_reduce()


class Pipeline(Patch):
    """A33, A34, A50, A51: the pipeline methods and the broadcast over the fabric family."""

    name = "pipeline"
    ids = ("A33", "A34", "A50", "A51")
    target = (
        "vllm.distributed.parallel_state.GroupCoordinator.send_tensor_dict, recv_tensor_dict, "
        "isend_tensor_dict, irecv_tensor_dict, make_sibling_device_group; "
        "torch.distributed.broadcast"
    )
    scope = GRAPH_SAFE
    why = (
        "vLLM's pipeline traffic goes over torch groups formed once over the world, which a dead "
        "rank cannot leave and a spare outside the world has none of; PPHandler broadcasts "
        "through torch's module-level function with the group it made at init"
    )
    upstream = (
        "a pipeline transport the executor can supply, PPHandler taking a group object with a "
        "broadcast, a transport failure callback on the group"
    )

    def _resolve(self) -> Any:
        try:
            from vllm.distributed.parallel_state import GroupCoordinator
        except ImportError:
            return None
        return GroupCoordinator

    def _present(self, group_cls: Any) -> bool:
        return bool(pipeline._ORIGINALS)

    def _put(self, group_cls: Any) -> None:
        methods, broadcast = pipeline.routed(group_cls)
        for name, fn in methods.items():
            setattr(group_cls, name, fn)
        torch.distributed.broadcast = broadcast

    def _restore(self, group_cls: Any) -> None:
        methods, broadcast = pipeline.release()
        for name, fn in methods.items():
            setattr(group_cls, name, fn)
        torch.distributed.broadcast = broadcast


class LogprobWarmup(Patch):
    """A41: the sampler's log-prob kernel warmed at every count a request may ask for."""

    name = "logprob_warmup"
    ids = ("A41",)
    target = "vllm.v1.worker.gpu_worker.warmup_kernels"
    scope = GRAPH_SAFE
    why = (
        "the log-prob kernel is specialized on its top-k block, so a request at a count outside "
        "vLLM's one warm-up pass compiles a kernel at its first step"
    )
    upstream = "a warm-up that covers every block size up to max_logprobs"

    def _resolve(self) -> Any:
        try:
            from vllm.v1.worker import gpu_worker
        except ImportError:
            return None
        return gpu_worker

    def _present(self, gpu_worker: Any) -> bool:
        return warmup.LOGPROBS in warmup.marks(gpu_worker, "warmup_kernels")

    def _put(self, gpu_worker: Any) -> None:
        warmup.install(gpu_worker, logprobs=True)

    def _restore(self, gpu_worker: Any) -> None:
        warmup.uninstall(gpu_worker)  # the spare's stage-alone wrap (A38) comes off with it


STEP_FN = StepFn()
EXECUTOR_CLASS = ExecutorClass()
ALL_REDUCE = AllReduce()
PIPELINE = Pipeline()
LOGPROB_WARMUP = LogprobWarmup()

PATCHES: tuple[Patch, ...] = (STEP_FN, EXECUTOR_CLASS, ALL_REDUCE, PIPELINE, LOGPROB_WARMUP)
"""Every permanent patch, in install order."""


def install() -> list[Patch]:
    """Put on every patch whose condition this process meets; what is on after. Idempotent."""
    graph_safe = settings.current().graph_safe
    for p in PATCHES:
        if p.scope == GRAPH_SAFE and not graph_safe:
            continue
        p.apply()
    on = installed()
    log.debug("patches on: %s", on)
    return on


def uninstall() -> None:
    """Put every original back, last patch first. Idempotent."""
    for p in reversed(PATCHES):
        p.remove()


def installed() -> list[Patch]:
    """The patches on in this process, in install order: for a diagnostic."""
    return [p for p in PATCHES if p.name in _OWNERS and p._present(_OWNERS[p.name])]
