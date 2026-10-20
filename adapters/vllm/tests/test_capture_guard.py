# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""No CUDA graph holds a serving communicator: a capture that reaches NCCL is refused.

``ncclCommAbort`` waits for every graph captured on its communicator to be freed; probe 5's first
run hung in such an abort. The guard makes the graph-safe op's promise (decision 0029) hold by
construction, and costs one capture check per call.
"""

import sys
import types

import pytest
import torch

from everserve_vllm import collective_op, worker
from everserve_vllm.groups import RefusedComm, UncapturedComm, unguarded


class Comm:
    """vLLM's ``PyNcclCommunicator``, as much of it as the guard touches."""

    def __init__(self):
        self.disabled, self.calls = False, []
        self.comm, self.nccl = "handle", "lib"

    def all_reduce(self, t, out_tensor=None):
        self.calls.append("all_reduce")
        return t

    def send(self, t, dst):
        self.calls.append(("send", dst))


def capturing(monkeypatch, value):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: value)


def test_outside_a_capture_every_call_and_attribute_is_the_communicator_s(monkeypatch):
    capturing(monkeypatch, False)
    comm = Comm()
    g = UncapturedComm(comm)
    assert g.all_reduce("t") == "t" and g.send("t", 1) is None
    assert comm.calls == ["all_reduce", ("send", 1)]
    assert g.comm == "handle" and g.nccl == "lib"  # read through
    g.disabled = True  # vLLM's own write goes to the communicator
    assert comm.disabled is True and unguarded(g) is comm and unguarded(comm) is comm


def test_inside_a_capture_a_collective_is_refused(monkeypatch):
    capturing(monkeypatch, True)
    comm = Comm()
    g = UncapturedComm(comm)
    with pytest.raises(RuntimeError, match="inside a CUDA graph's capture"):
        g.all_reduce("t")
    with pytest.raises(RuntimeError, match="send"):
        g.send("t", 1)
    assert comm.calls == []


def parallel_state(monkeypatch, **handles):
    ps = types.ModuleType("vllm.distributed.parallel_state")
    for name, pn in handles.items():
        setattr(
            ps,
            name,
            types.SimpleNamespace(device_communicator=types.SimpleNamespace(pynccl_comm=pn)),
        )
    dist = types.ModuleType("vllm.distributed")
    dist.parallel_state = ps
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", dist)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", ps)
    return ps


def test_every_handle_vllm_holds_is_guarded_once_and_a_refused_one_left_alone(monkeypatch):
    tp, pp = Comm(), Comm()
    refused = RefusedComm(Comm())
    ps = parallel_state(monkeypatch, _TP=tp, _PP=pp, _EP=refused)
    assert worker.guard_engine_groups() == ["_TP", "_PP"]
    assert isinstance(ps._TP.device_communicator.pynccl_comm, UncapturedComm)
    assert unguarded(ps._TP.device_communicator.pynccl_comm) is tp
    assert ps._EP.device_communicator.pynccl_comm is refused
    assert worker.guard_engine_groups() == []  # never a guard around a guard


def test_a_refusal_aborts_the_communicator_itself_not_its_guard(monkeypatch):
    tp, pp = Comm(), Comm()
    ps = parallel_state(monkeypatch, _TP=UncapturedComm(tp), _PP=UncapturedComm(pp))
    rest = collective_op._refuse_engine_groups(skip=(tp,))  # the live group's own release
    assert rest == [("_PP", pp)]  # the TP handle is the live group's: skipped, not aborted twice
    assert isinstance(ps._TP.device_communicator.pynccl_comm, RefusedComm)
    assert ps._TP.device_communicator.pynccl_comm.replaced is tp
