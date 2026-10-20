# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A serving engine core completes membership changes in its own step loop: stock
``vllm serve``, no drill."""

import sys
import types

import pytest

from everserve_vllm import serving
from everserve_vllm.losses import RankLostError


class Executor:
    """Records the changes it is asked to complete."""

    def __init__(self, pending=None, raises=None):
        self.completed = []
        self.pending = list(pending or [])
        self.raises = raises
        self.failure = None

    def complete_change(self, core, heard=None):
        self.completed.append((core, heard))
        if self.raises is not None:
            raise self.raises
        return {"epoch_committed": 1, "boundaries": {}, "ranks": [1]}

    def _fail(self, failure):
        if self.failure is None:
            self.failure = failure

    def pending_change(self):
        return self.pending.pop(0) if self.pending else None

    def serve_steps(self, core, step):
        return serving.with_changes(self, core, step)


def test_a_step_a_loss_ended_completes_the_change_and_returns_no_output():
    ex = Executor()

    def step():
        raise RankLostError("worker(s) [1] lost")

    assert serving.with_changes(ex, "core", step)() == ({}, False)
    assert ex.completed == [("core", None)]  # waits for the commit itself


def test_a_commit_without_a_death_completes_after_the_step_that_ran():
    ex = Executor(pending=["commit at epoch 1"])
    out = ({0: "outputs"}, True)
    stepped = serving.with_changes(ex, "core", lambda: out)
    assert stepped() == out  # the step's outputs still go to the clients
    assert ex.completed == [("core", "commit at epoch 1")]
    assert stepped() == out and len(ex.completed) == 1  # nothing pending: nothing to complete


def test_any_other_failure_fails_the_engine_as_stock_does():
    ex = Executor()

    def step():
        raise RuntimeError("Worker failed with error 'CUDA error'")

    with pytest.raises(RuntimeError, match="CUDA error"):
        serving.with_changes(ex, "core", step)()
    assert ex.completed == []
    assert ex.failure == "a step: RuntimeError: Worker failed with error 'CUDA error'"


def test_a_change_that_fails_its_completion_says_why_in_the_engine_s_failure():
    """The head's goodbye carries the executor's failure: without it, it reads "Stopped" with no
    detail."""
    from types import SimpleNamespace

    heard = SimpleNamespace(committed=SimpleNamespace(epoch=1))
    ex = Executor(pending=[heard], raises=RuntimeError("no claim recorded for epoch 1"))
    with pytest.raises(RuntimeError, match="no claim recorded"):
        serving.with_changes(ex, "core", lambda: ({}, True))()
    assert ex.failure == (
        "completing the change at epoch 1: RuntimeError: no claim recorded for epoch 1"
    )
    lost = Executor(raises=RuntimeError("no join answer from [1]"))

    def step():
        raise RankLostError("worker(s) [1] lost")

    with pytest.raises(RuntimeError, match="no join answer"):
        serving.with_changes(lost, "core", step)()
    assert lost.failure == "completing a change: RuntimeError: no join answer from [1]"


@pytest.fixture
def engine_core_module(monkeypatch):
    """vLLM's ``vllm.v1.engine.core`` as far as the hook reads it: an EngineCore whose
    constructor builds its executor and sets ``step_fn``."""

    class EngineCore:
        def __init__(self, executor):
            self.model_executor = executor
            self.scheduler = "scheduler"
            self.step_fn = self.step

        def step(self):
            return {0: "outputs"}, True

    class EngineCoreProc(EngineCore):
        def _process_engine_step(self):
            outputs, model_executed = self.step_fn()
            return model_executed

    core = types.ModuleType("vllm.v1.engine.core")
    core.EngineCore = EngineCore
    core.EngineCoreProc = EngineCoreProc
    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core)
    return core


def test_the_hook_wraps_an_engine_on_our_executor_and_leaves_a_stock_one_alone(
    engine_core_module,
):
    assert serving.install() and serving.install()  # once, however often vLLM loads plugins
    ours = engine_core_module.EngineCore(Executor(pending=["commit"]))
    assert ours.step_fn() == ({0: "outputs"}, True)
    assert ours.model_executor.completed == [(ours, "commit")]
    stock = engine_core_module.EngineCore(object())
    assert stock.step_fn == stock.step  # a stock executor's engine is untouched


def test_the_engine_core_whose_constructor_loads_the_plugin_is_served_itself(engine_core_module):
    """vLLM loads general plugins inside ``EngineCore.__init__``, before the executor and
    ``step_fn``: in a fresh engine core process (Dynamo's head) the plugin is installed during the
    first engine core's construction, and that engine core is the one that serves. Under a hook on
    ``__init__`` it keeps the stock step and dies on the first notice."""

    class FreshEngineCore(engine_core_module.EngineCore):
        def __init__(self, executor):
            serving.install()  # load_general_plugins(), core.py's first statement
            super().__init__(executor)

    engine_core_module.EngineCore = FreshEngineCore
    first = FreshEngineCore(Executor(pending=["commit"]))
    assert first.step_fn() == ({0: "outputs"}, True)
    assert first.model_executor.completed == [(first, "commit")]


def test_a_failing_step_on_the_served_engine_core_completes_the_change_in_its_loop(
    engine_core_module,
):
    """The busy loop's ``self.step_fn()`` (``_process_engine_step``) reaches the served step: a
    step ended by a notice completes the change and the loop goes on, the engine alive."""

    class Leaving(Executor):
        def serve_steps(self, core, step):
            def lost():
                raise RankLostError("worker(s) [1] leaving on notice")

            return serving.with_changes(self, core, lost)

    serving.install()
    proc = engine_core_module.EngineCoreProc(Leaving())
    assert proc._process_engine_step() is False  # no output this step, no exception
    assert proc.model_executor.completed == [(proc, None)]


def test_a_head_whose_engine_core_would_step_without_changes_refuses_to_start(engine_core_module):
    """The plugin not loaded in the engine core's process: the head's executor refuses where it
    is built, naming why, rather than serve an engine that looks protected and is not."""
    assert not serving.installed()
    with pytest.raises(serving.StepHookMissing, match="step hook was not installed"):
        serving.require_installed()
    serving.install()
    assert serving.installed()
    serving.require_installed()


def test_without_vllm_the_hook_installs_nothing(monkeypatch):
    # import fails as it does without the extra; the submodules too, since a test that imported
    # vLLM's engine core earlier leaves them cached, and a cached module is found before its parent
    for name in ("vllm", "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.core"):
        monkeypatch.setitem(sys.modules, name, None)
    assert serving.install() is False


def test_an_engine_core_of_another_shape_refuses_to_start_rather_than_serve_unprotected(
    engine_core_module,
):
    """A vLLM whose busy loop no longer steps through ``step_fn`` would serve normally and die on
    its first loss; the engine refuses at startup instead, naming what it did not find."""

    class Loop:
        def _process_engine_step(self):
            return self.step_with_new_name()

    engine_core_module.EngineCoreProc = Loop
    serving.install()
    with pytest.raises(serving.UnsupportedEngineCore, match="calling self.step_fn"):
        engine_core_module.EngineCore(Executor())
    engine_core_module.EngineCore(object())  # a stock executor's engine is not ours to refuse


def test_the_pinned_vllm_s_engine_core_steps_through_step_fn():
    """Where vLLM is installed, its busy loop is the shape the hook needs (A47)."""
    pytest.importorskip("vllm")
    from vllm.v1.engine import core

    loop = core.EngineCoreProc._process_engine_step
    assert "step_fn" in loop.__code__.co_names
