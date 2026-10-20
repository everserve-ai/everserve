# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Membership changes in a serving engine: stock ``vllm serve``, the engine core's own step loop.

A change completes on the engine core's thread, between steps (``EverserveExecutor.
complete_change``): it needs the core's scheduler and batch queue. In a serving engine nothing of
ours runs that loop: ``EngineCoreProc.run_busy_loop`` calls ``self.step_fn`` and, if it raises,
the engine core dies and the API server answers every later request with ``EngineDeadError``.
:func:`install` (the registry's A47, :mod:`everserve_vllm.patches`) puts a data descriptor on
``EngineCore.step_fn`` once, in the engine core's process (vLLM loads general plugins there,
``core.py`` A47): the stock constructor's own
``self.step_fn = ...`` goes through it, and an engine whose executor drives changes keeps
:func:`with_changes` over the stock step instead. Nothing changes for an engine on a stock
executor.

The hook is on the assignment, not around the constructor, because vLLM loads the plugins inside
``EngineCore.__init__``, before it builds the executor and sets ``step_fn``. In an engine core
process started fresh (Dynamo's head), the plugin is installed while the first engine core is
being constructed: a wrap of ``__init__`` reaches only the instances made after it, so that
engine core would keep the stock step and die on its first notice. A data descriptor on the
class takes every assignment made after it is installed, that one included.

:func:`with_changes` does what the harness driver's loop does (``bench/harness/adapter_drill.py``):
a step
that a loss ended (:class:`~everserve_vllm.losses.RankLostError`) completes the change, which waits
for its commit, and returns no output; after a step that ran, a commit that arrived without a
death (a notice, a drain) completes too. The change's replay puts the requests' committed
positions back on the scheduler, so the next step continues them. A change that cannot complete
raises, and the engine fails as stock does, with the reason.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

Step = Callable[[], tuple[Any, bool]]


def with_changes(executor: Any, core: Any, step: Step) -> Step:
    """``step`` with the executor's membership changes completed around it, on its thread."""
    from everserve_vllm.losses import RankLostError

    def changed(heard: Any = None) -> None:
        record = executor.complete_change(core, heard)
        summary = {k: v for k, v in record.items() if k not in ("boundaries", "spare", "spares")}
        log.info("membership change completed: %s", summary)

    def failed(e: BaseException, where: str) -> None:
        # the engine ends on this, and its head's goodbye says why (decision 0032): a cause not
        # kept here leaves the goodbye "Stopped" with no detail, and the failure (say, "no claim
        # recorded for epoch 1") has to be dug out of the logs
        fail = getattr(executor, "_fail", None)
        if fail is not None:
            fail(f"{where}: {type(e).__name__}: {e}")

    @functools.wraps(step)
    def stepped() -> tuple[Any, bool]:
        try:
            out = step()
        except RankLostError as e:
            log.warning("a step ended on a loss: %s", e)
            try:
                changed()
            except Exception as change_error:
                failed(change_error, "completing a change")
                raise
            return {}, False
        except Exception as e:
            failed(e, "a step")
            raise
        heard = executor.pending_change()
        if heard is not None:
            try:
                changed(heard)
            except Exception as e:
                failed(e, f"completing the change at epoch {heard.committed.epoch}")
                raise
        return out

    return stepped


class UnsupportedEngineCore(RuntimeError):
    """This vLLM's engine core is not the shape the step-loop hook needs: refuse to serve.

    An engine whose hook silently stopped applying would serve normally and die on its first loss,
    which nothing would show before production. The engine refuses at startup instead, naming
    what it did not find (A47).
    """


def check_shape(core_module: Any, engine_core: Any) -> None:
    """Raise :class:`UnsupportedEngineCore` unless the engine core steps through ``step_fn``.

    The hook replaces the instance's ``step_fn``; that changes the engine's steps only while the
    busy loop calls ``self.step_fn`` (``EngineCoreProc._process_engine_step``) and the instance
    holds a scheduler for the change's replay.
    """
    missing = [n for n in ("step_fn", "scheduler") if not hasattr(engine_core, n)]
    proc = getattr(core_module, "EngineCoreProc", None)
    loop = getattr(proc, "_process_engine_step", None) if proc is not None else None
    if loop is None or "step_fn" not in getattr(getattr(loop, "__code__", None), "co_names", ()):
        missing.append("EngineCoreProc._process_engine_step calling self.step_fn")
    if missing:
        raise UnsupportedEngineCore(
            "everserve: this vLLM's engine core lacks what membership changes need: "
            f"{', '.join(missing)}; this adapter supports the vLLM in adapters/vllm/PIN"
        )


def install() -> bool:
    """Put :class:`ServedStep` on ``EngineCore`` once: a change-driving engine serves changes.

    The registry's A47 (:data:`everserve_vllm.patches.STEP_FN`). False where vLLM's engine core
    is not importable (this package without the engine extra).
    """
    from everserve_vllm import patches

    return patches.STEP_FN.apply()


class StepHookMissing(RuntimeError):
    """The head's engine core would step without membership changes: refuse to serve.

    Its first loss or notice would end the engine with every stream in flight, while everything
    before it looked protected. Raised where the head's executor is built, before it registers.
    """


def installed() -> bool:
    """Whether this process's engine core serves changes: :class:`ServedStep` is on it.

    vLLM loads the plugins before it builds the executor (``core.py``), so from the executor's
    construction on, this is what the engine core's step will be.
    """
    from everserve_vllm import patches

    return patches.STEP_FN.is_present()


def require_installed() -> None:
    """Raise :class:`StepHookMissing` unless this process's engine core serves changes."""
    if not installed():
        raise StepHookMissing(
            "Everserve's step hook was not installed: the engine would serve without membership "
            "changes. The everserve_vllm plugin did not load in the engine core's process "
            "(vLLM's general plugins: check VLLM_PLUGINS names 'everserve' where it is set)"
        )


class ServedStep:
    """``EngineCore.step_fn``: the step the engine core set, with changes served around it.

    A data descriptor, so the constructor's ``self.step_fn = ...`` comes here in place of the
    instance's dictionary, however late in that constructor the plugin was installed.
    """

    SLOT = "_everserve_step_fn"

    def __init__(self, core_module: Any) -> None:
        """``core_module`` is vLLM's ``vllm.v1.engine.core``, whose busy loop the check reads."""
        self.core_module = core_module

    def __get__(self, engine_core: Any, owner: type | None = None) -> Any:
        """The step this engine core was given, served where its executor drives changes."""
        if engine_core is None:
            return self
        try:
            return engine_core.__dict__[self.SLOT]
        except KeyError:
            raise AttributeError("step_fn") from None

    def __set__(self, engine_core: Any, step: Step) -> None:
        """Keep ``step``; over a change-driving executor, the served step, after the shape check."""
        engine_core.__dict__[self.SLOT] = step
        executor = getattr(engine_core, "model_executor", None)
        serve_steps = getattr(executor, "serve_steps", None)
        if callable(serve_steps):
            check_shape(self.core_module, engine_core)
            engine_core.__dict__[self.SLOT] = serve_steps(engine_core, step)
            log.info("the engine core serves membership changes (%s)", type(executor).__name__)
