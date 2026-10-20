# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The NCCL probe's control flow, checked without a GPU before a rental is spent on it.

On a nonblocking parent, ``ncclCommSplit`` returns ``ncclSuccess`` at once and fills in the
child's handle when the split completes: a probe that reads the return code as the split's end
issues the child's calls before the child exists. The rounds must issue the same calls on both
ranks, or the real pair hangs.
"""

import contextlib
import ctypes
import json
import sys
import types
import weakref
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "k8s"))
import nccl_probe  # noqa: E402

PARENT = 0x1000
CHILD = 0x2000


class Nccl:
    """NCCL as probe 3 found it. The split returns 0 with no child. Its handle and success come
    after ``polls`` reads of the state, or never for a member that does not come.

    ``parent_state`` is what the parent reads meanwhile.
    """

    def __init__(self, polls: int | None, parent_state: int = nccl_probe.NCCL_SUCCESS):
        self.polls, self.parent_state = polls, parent_state
        self.reads: list[int] = []
        self.child: ctypes.c_void_p | None = None

    def split_blocking(self, parent, rank):
        self.child = ctypes.c_void_p()
        return self.child, nccl_probe.NCCL_SUCCESS

    def state(self, comm) -> int:
        self.reads.append(comm.value)
        if self.polls is not None and len(self.reads) >= self.polls:
            self.child.value = CHILD  # NCCL's thread completes the split
        if comm.value == CHILD:
            return nccl_probe.NCCL_SUCCESS
        return self.parent_state


def test_the_split_is_done_only_when_the_child_exists_and_reads_success():
    nccl = Nccl(polls=5)
    child, read = nccl_probe.split(nccl, ctypes.c_void_p(PARENT), 0, deadline_s=5.0)
    assert child.value == CHILD
    assert read["rc"] == nccl_probe.NCCL_SUCCESS and read["state"] == nccl_probe.NCCL_SUCCESS
    assert read["child"] and read["polls"] >= 5
    assert nccl.reads[-1] == CHILD  # the last read is the child's own state, not the parent's


def test_a_split_whose_member_never_comes_is_in_progress_at_its_deadline():
    child, read = nccl_probe.split(Nccl(polls=None), ctypes.c_void_p(PARENT), 0, deadline_s=0.05)
    assert not read["child"] and child.value is None
    assert read["state"] == nccl_probe.NCCL_IN_PROGRESS  # the caller then aborts the parent


def test_a_parent_error_ends_the_wait():
    nccl = Nccl(polls=None, parent_state=6)  # ncclRemoteError: a member died mid-split
    _child, read = nccl_probe.split(nccl, ctypes.c_void_p(PARENT), 0, deadline_s=5.0)
    assert read["state"] == 6 and read["s"] < 1.0


def test_each_arm_leads_a_round_in_turn_and_both_ranks_agree():
    orders = [nccl_probe.arm_order(r) for r in range(8)]
    assert [o[0] for o in orders[:4]] == ["V", "P", "C", "W"]
    assert all(sorted(o) == sorted(nccl_probe.ARMS) for o in orders)
    assert orders[4:] == orders[:4]  # a function of the round alone: the same on every rank


def test_a_summary_is_the_rounds_median_and_spread():
    s = nccl_probe.summary([10.0, 12.0, 11.0, 30.0, 9.0])
    assert s == {"median": 11.0, "iqr": 2.0, "min": 9.0, "max": 30.0}


class Graph:
    """A captured graph. Weakly referenced, so a test can see whether one outlives its use."""

    def replay(self):
        pass


class Torch:
    """As much of torch as the rounds use: buffers, a stream, graphs."""

    bfloat16 = "bf16"

    def __init__(self):
        cuda = types.SimpleNamespace()
        cuda.set_device = lambda _d: None
        cuda.synchronize = lambda: None
        cuda.Stream = lambda: types.SimpleNamespace(cuda_stream=7, synchronize=lambda: None)
        self.graphs: list[weakref.ref] = []

        def graph():
            g = Graph()
            self.graphs.append(weakref.ref(g))
            return g

        cuda.CUDAGraph = graph
        self.capturing = False

        @contextlib.contextmanager
        def graph(_g, stream=None):
            self.capturing = True
            yield
            self.capturing = False

        cuda.graph = graph
        cuda.is_current_stream_capturing = lambda: self.capturing
        self.cuda = cuda

    @staticmethod
    def zeros(n, device=None, dtype=None):
        return types.SimpleNamespace(data_ptr=lambda: 0x9000 + n, numel=lambda: n)


class RoundsNccl(Nccl):
    """NCCL for the rounds: every call recorded by communicator and size, and the parent's
    answered in progress, as a nonblocking communicator's are."""

    def __init__(self):
        super().__init__(polls=1)
        self.calls: list[tuple[int, int]] = []

    def version(self):
        return 22907

    def start(self, world, uid, rank, blocking=False):
        return ctypes.c_void_p(PARENT), nccl_probe.NCCL_IN_PROGRESS

    def wait(self, comm, deadline_s, poll_s=0.01):
        return nccl_probe.NCCL_SUCCESS, 0.0, 1

    def all_reduce(self, send, recv, count, dtype, op, comm, stream):
        self.calls.append((comm.value, count))
        return nccl_probe.NCCL_IN_PROGRESS if comm.value == PARENT else nccl_probe.NCCL_SUCCESS

    def aborted(self, comm):
        # NCCL's abort waits for every graph captured on the communicator to be freed: one alive
        # here hangs the real probe (probe 5's first run)
        self.live_at_abort.append(sum(r() is not None for r in self.torch.graphs))
        return 0, 0.0


def run_rounds(monkeypatch, rank):
    vllm_comm = ctypes.c_void_p(0x3000)
    pynccl = types.ModuleType("vllm.distributed.device_communicators.pynccl")
    pynccl.PyNcclCommunicator = types.SimpleNamespace(
        from_unique_id_bytes=lambda uid, r, world, dev: types.SimpleNamespace(comm=vllm_comm)
    )
    for name in ("vllm", "vllm.distributed", "vllm.distributed.device_communicators"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, pynccl.__name__, pynccl)
    nccl = RoundsNccl()
    nccl.torch, nccl.live_at_abort = Torch(), []
    out = nccl_probe.rounds(nccl, nccl.torch, rank, [b"a" * 128, b"b" * 128], n_rounds=3)
    assert nccl.live_at_abort == [0, 0], "a graph outlived its use into an abort"
    assert all(a.get("rc") == 0 for a in out["aborts"].values())
    return nccl.calls, out


def test_both_ranks_issue_the_same_calls_and_every_switch_is_warmed(monkeypatch, capsys):
    calls_0, out = run_rounds(monkeypatch, 0)
    printed = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [p["round"] for p in printed] == [0, 1, 2]  # each round out as it ends
    assert set(printed[0]["us"]) == set(nccl_probe.ARMS)
    calls_1, _ = run_rounds(monkeypatch, 1)
    assert calls_0 == calls_1  # a mismatch would hang the real pair
    assert out["all_success"] and out["split"]["child"]
    per_round = nccl_probe.EAGER_CALLS * (nccl_probe.BATCHES + 1)  # the batches, one discarded
    capture = 20 + nccl_probe.GRAPH_CALLS  # the warm calls and the graph's own
    for comm in (PARENT, CHILD):
        for n in nccl_probe.DECODE:
            assert calls_0.count((comm, n)) == capture + 3 * per_round
    for n in nccl_probe.DECODE:  # V's and W's calls: W, eager only, goes to V's communicator
        assert calls_0.count((0x3000, n)) == capture + 3 * per_round + 3 * per_round
    assert "replay" not in out["arms"]["W"]["896"]  # never captured: the guard would refuse
    for arm in nccl_probe.ARMS:
        assert len(out["arms"][arm]["896"]["eager"]["rounds"]) == 3
    for arm in nccl_probe.REPLAYED:
        assert len(out["arms"][arm]["896"]["replay"]["rounds"]) == 3
