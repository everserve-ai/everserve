# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The adapter's log records reach the process's log in a process only vLLM configured."""

import logging

import pytest

import everserve_vllm


class Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


CHILDREN = ["everserve_vllm.identity", "everserve_vllm.node", "everserve.contract.daemon"]
"""The loggers the tests write to, below :data:`everserve_vllm.LOGGERS`."""


@pytest.fixture
def loggers():
    """The root, vLLM's and our loggers fresh, as a process starts them; put back after.

    A process that loaded the plugin earlier in the session (the engine image's run) left ours
    carrying vLLM's handlers and not propagating: every one is reset, not only its handlers.
    """
    names = ["", "vllm", *everserve_vllm.LOGGERS, *CHILDREN]
    saved = {
        n: (
            list(logging.getLogger(n).handlers),
            logging.getLogger(n).level,
            logging.getLogger(n).propagate,
            logging.getLogger(n).disabled,
        )
        for n in names
    }
    disabled_below = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    for n in names:
        lg = logging.getLogger(n)
        lg.handlers, lg.propagate, lg.disabled = [], True, False
        lg.setLevel(logging.WARNING if n == "" else logging.NOTSET)
    yield
    logging.disable(disabled_below)
    for n, (handlers, level, propagate, disabled) in saved.items():
        lg = logging.getLogger(n)
        lg.handlers, lg.propagate, lg.disabled = handlers, propagate, disabled
        lg.setLevel(level)


def can_say(name: str) -> str:
    """Why ``name``'s INFO record would reach no handler; empty when it would reach one."""
    lg = logging.getLogger(name)
    if logging.root.manager.disable >= logging.INFO:
        return f"logging.disable({logging.root.manager.disable})"
    if lg.disabled:
        return f"{name} disabled"
    if not lg.isEnabledFor(logging.INFO):
        return f"{name} at effective level {lg.getEffectiveLevel()}"
    return ""


def test_in_a_process_only_vllm_configured_our_info_lines_reach_its_handler(loggers):
    logging.getLogger().handlers = []  # pytest's capture is the root's only handler here
    vllm_handler = Collect()
    logging.getLogger("vllm").addHandler(vllm_handler)
    everserve_vllm.log_through_vllm()
    everserve_vllm.log_through_vllm()  # idempotent: one handler, one line
    assert [can_say(n) for n in CHILDREN] == ["", "", ""]
    logging.getLogger("everserve_vllm.identity").info("step config abc: x=1")
    logging.getLogger("everserve.contract.daemon").info("engine k is serving")
    assert vllm_handler.lines == ["step config abc: x=1", "engine k is serving"]


def test_a_process_that_configured_its_own_log_keeps_it(loggers):
    logging.getLogger().handlers = []
    own, vllm_handler = Collect(), Collect()
    logging.getLogger().addHandler(own)  # node_main's basicConfig, say
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("vllm").addHandler(vllm_handler)
    everserve_vllm.log_through_vllm()
    assert can_say("everserve_vllm.node") == ""
    logging.getLogger("everserve_vllm.node").info("node role started")
    assert own.lines == ["node role started"] and vllm_handler.lines == []


def test_nothing_changes_where_vllm_configured_nothing(loggers):
    logging.getLogger().handlers = []
    everserve_vllm.log_through_vllm()
    assert all(not logging.getLogger(n).handlers for n in everserve_vllm.LOGGERS)
