# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The graph-safe op's size dispatch and the two-half change, against fakes (no vLLM, no GPU)."""

from dataclasses import dataclass, field

import pytest
import torch

from everserve.comm.group import ShrinkableGroup
from everserve.comm.handle import LocalStandIn
from everserve.comm.shadow import replacement, set_key
from everserve_vllm.collective_op import VllmCollective, all_reduce_into

R = replacement(2)


class FakeSlot:
    """Takes up to ``limit`` elements; records every call."""

    def __init__(self, limit=8):
        self.limit = limit
        self.calls = []

    def can_take(self, t):
        return t.numel() <= self.limit

    def all_reduce_into(self, inp, out):
        self.calls.append(("reduce", inp.numel()))
        out.copy_(inp * 4)

    def set_members(self, members, epoch):
        self.calls.append(("members", list(members), epoch))

    def abort(self):
        self.calls.append(("abort",))

    def resume(self):
        self.calls.append(("resume",))


@dataclass
class FakeGroup(ShrinkableGroup):
    group: object = None
    ranks: list = field(default_factory=lambda: [0, 1, 2, 3])
    backend: str = "fake"
    reduced: int = 0
    refused: bool = False

    def all_reduce(self, t):
        self.reduced += 1
        t.mul_(len(self.ranks))

    def refuse(self):
        self.refused = True
        return None  # nothing to free in a fake


def make(**kw):
    live = FakeGroup()
    rep = FakeGroup(group="replacement-comm")
    slot = FakeSlot()
    c = VllmCollective(0, live, {(0, 1, R, 3): rep}, slot, spare_row=4, **kw)
    return c, live, rep, slot


def test_decode_sizes_go_to_the_slot_and_larger_ones_to_the_current_group():
    c, live, _rep, slot = make()
    small, out = torch.ones(8), torch.zeros(8)
    all_reduce_into(c, small, out)
    assert slot.calls == [("reduce", 8)] and torch.equal(out, torch.full((8,), 4.0))
    big, out = torch.ones(9), torch.zeros(9)
    all_reduce_into(c, big, out)
    assert live.reduced == 1 and torch.equal(out, torch.full((9,), 4.0))
    assert torch.equal(big, torch.ones(9)), "the input is not reduced in place"


def test_a_stand_in_copies_large_tensors():
    c = VllmCollective(R, LocalStandIn(), {}, FakeSlot(), spare_row=4)
    x, out = torch.arange(10.0), torch.zeros(10)
    all_reduce_into(c, x, out)
    assert torch.equal(out, x)


def test_a_change_is_staged_and_applied_at_the_boundary():
    applied = []
    c, live, rep, slot = make(on_apply=applied.append)
    c.members = (0, 1, 2, 3)
    c.refuse()
    assert ("abort",) in slot.calls and live.refused
    c.change(c.replacement_set((2,)), epoch=1)
    # staged: collectives still go where they went, the slot is untouched
    assert c.current is live and c.staged is not None
    assert all(call[0] in ("abort",) for call in slot.calls)
    moved = c.apply_staged()
    assert moved == {"members": "0-1-R2-3", "rows": [0, 1, 4, 3], "epoch": 1}
    assert c.current is rep and c.members == (0, 1, R, 3) and c.epoch == 1
    assert slot.calls[-2:] == [("resume",), ("members", [0, 1, 4, 3], 1)]
    assert applied == [rep]
    assert c.apply_staged() is None


def test_a_refused_group_and_its_stand_in_refuse_every_collective():
    import pytest

    from everserve_vllm.groups import PyNcclGroup, RefusedComm

    class FakeComm:
        device = None
        world_size = 4
        comm = object()
        disabled = False

        def __init__(self):
            self.aborted = 0
            outer = self

            class Nccl:
                def ncclCommAbort(self, comm):  # noqa: N802 - NCCL's name
                    outer.aborted += 1

            self.nccl = Nccl()

        def all_reduce(self, t, out_tensor=None):
            return t

    comm = FakeComm()
    g = PyNcclGroup.over(comm, [0, 1, 2, 3])
    g.all_reduce(torch.ones(2))
    release = g.refuse()
    assert g.refuse() is None  # once only
    release()
    assert comm.aborted == 1 and not comm.disabled  # never disabled: vLLM would fall back
    with pytest.raises(RuntimeError, match="refused"):
        g.all_reduce(torch.ones(2))
    stand_in = RefusedComm(comm)
    assert stand_in.world_size == 4 and not stand_in.disabled
    with pytest.raises(RuntimeError, match="refused"):
        stand_in.all_gather(torch.ones(2), torch.ones(1))


class SlowComm:
    """A communicator whose ncclCommAbort takes ``seconds``."""

    device = None
    disabled = False

    def __init__(self, seconds):
        import time

        self.comm = object()
        outer = self

        class Nccl:
            aborted = 0

            def ncclCommAbort(self, comm):  # noqa: N802 - NCCL's name
                time.sleep(outer.seconds)
                self.aborted += 1

        self.seconds = seconds
        self.nccl = Nccl()


def test_neither_the_refusal_nor_the_boundary_waits_for_a_release(monkeypatch):
    import time

    from everserve_vllm import collective_op, groups

    live = groups.PyNcclGroup.over(SlowComm(0.3), [0, 1, 2, 3])
    others = [("_EP", SlowComm(0.3)), ("_DP", SlowComm(0.3))]
    monkeypatch.setattr(collective_op, "_refuse_engine_groups", lambda skip=(): others)
    c = VllmCollective(0, live, {}, FakeSlot(), spare_row=4)
    t0 = time.perf_counter()
    c.refuse()
    took = time.perf_counter() - t0
    assert took < 0.1, f"the refusal took {took:.2f} s: it waited for a release"
    assert live.refused and c.releases.done == []
    with pytest.raises(RuntimeError, match="refused"):
        live.all_reduce(torch.ones(2))
    c.change_to_group((0, 1, R, 3), FakeGroup(group="replacement-comm"), epoch=1)
    t0 = time.perf_counter()
    c.apply_staged()
    assert time.perf_counter() - t0 < 0.1, "the boundary waited for a release"
    assert c.releases.done == [] and c.releases.running() == ["live", "_EP", "_DP"]
    time.sleep(0.4)  # the releases return beside the serving steps; the next reap records them
    c.releases.reap()
    assert [a.split()[0] for a in c.releases.done] == ["live", "_EP", "_DP"]
    assert all(a.endswith("s") and float(a.split()[1][:-1]) >= 0.29 for a in c.releases.done)
    assert c.releases.leaked == 0 and c.releases.running() == []


def test_a_release_past_the_bound_is_left_behind_and_counted(monkeypatch):
    import time

    from everserve_vllm import collective_op, groups

    monkeypatch.setattr(groups, "RELEASE_BOUND_S", 0.2)
    stuck = groups.PyNcclGroup.over(SlowComm(5.0), [0, 1, 2, 3])
    monkeypatch.setattr(collective_op, "_refuse_engine_groups", lambda skip=(): [])
    c = VllmCollective(0, stuck, {}, FakeSlot(), spare_row=4)
    reported = []
    c.releases.on_leak = lambda name, run_s, total: reported.append((name, run_s, total))
    c.refuse()
    c.change_to_group((0, 1, R, 3), FakeGroup(group="replacement-comm"), epoch=1)
    t0 = time.perf_counter()
    c.apply_staged()
    assert time.perf_counter() - t0 < 0.1
    # within its bound at the first boundary: still running, not yet leaked
    assert c.releases.done == [] and c.releases.leaked == 0 and reported == []
    time.sleep(0.3)
    # past its bound at the next: left behind and counted, never waited out, reported once
    c.change_to_group((0, 1, R, 3), FakeGroup(group="replacement-comm"), epoch=2)
    c.apply_staged()
    assert c.releases.done == ["live still running"] and c.releases.leaked == 1
    assert [(n, t) for n, _, t in reported] == [("live", 1)] and reported[0][1] > 0.2
    c.change_to_group((0, 1, R, 3), FakeGroup(group="replacement-comm"), epoch=3)
    c.apply_staged()
    assert c.releases.done == ["live still running"] and c.releases.leaked == 1
    assert len(reported) == 1  # once per leak, not at every boundary after


def test_torch_s_abort_is_a_bounded_release_too(monkeypatch):
    import time

    from torch.distributed import distributed_c10d as c10d

    from everserve_vllm import collective_op, groups

    monkeypatch.setattr(groups, "RELEASE_BOUND_S", 0.2)
    monkeypatch.setattr(c10d, "is_initialized", lambda: True)
    monkeypatch.setattr(c10d, "_abort_process_group", lambda: time.sleep(5.0))
    monkeypatch.setattr(collective_op, "_refuse_engine_groups", lambda skip=(): [])
    c, _live, _rep, _slot = make()
    t0 = time.perf_counter()
    c.refuse()
    assert time.perf_counter() - t0 < 0.1
    c.change_to_group((0, 1, R, 3), FakeGroup(group="replacement-comm"), epoch=1)
    c.apply_staged()
    assert c.releases.done == [] and c.releases.running() == ["torch"]
    time.sleep(0.3)
    c.releases.reap()
    assert c.releases.done == ["torch still running"] and c.releases.leaked == 1


def test_a_release_waits_for_a_call_inside_the_communicator(monkeypatch):
    """The free must not run under the step thread's ncclSend (decision 0025)."""
    import threading
    import time

    from everserve.comm.guard import guard_of
    from everserve_vllm import collective_op, groups

    comm = SlowComm(0.0)
    live = groups.PyNcclGroup.over(comm, [0, 1, 2, 3])
    monkeypatch.setattr(collective_op, "_refuse_engine_groups", lambda skip=(): [])
    c = VllmCollective(0, live, {}, FakeSlot(), spare_row=4)
    inside = threading.Event()
    leave = threading.Event()

    def send():  # the step thread, inside a call on the communicator
        with guard_of(comm).shared():
            inside.set()
            leave.wait()

    step = threading.Thread(target=send)
    step.start()
    inside.wait()
    c.refuse()
    time.sleep(0.2)
    assert comm.nccl.aborted == 0, "the free ran under a call in progress"
    leave.set()
    step.join()
    c.change_to_group((0, 1, R, 3), FakeGroup(group="replacement-comm"), epoch=1)
    c.apply_staged()
    time.sleep(0.2)  # the free runs beside the boundary, once the call has left
    c.releases.reap()
    assert comm.nccl.aborted == 1 and c.releases.leaked == 0

    # a call that never leaves: the release is left behind and counted, never waited out
    monkeypatch.setattr(groups, "RELEASE_BOUND_S", 0.2)
    comm2 = SlowComm(0.0)
    c2 = VllmCollective(
        0, groups.PyNcclGroup.over(comm2, [0, 1, 2, 3]), {}, FakeSlot(), spare_row=4
    )
    inside2 = threading.Event()

    def never_leaves():
        with guard_of(comm2).shared():
            inside2.set()
            threading.Event().wait()

    threading.Thread(target=never_leaves, daemon=True).start()
    inside2.wait()
    c2.refuse()
    c2.change_to_group((0, 1, R, 3), FakeGroup(group="replacement-comm"), epoch=1)
    c2.apply_staged()
    time.sleep(0.3)
    c2.releases.reap()
    assert c2.releases.done == ["live still running"] and c2.releases.leaked == 1
    assert comm2.nccl.aborted == 0


def test_the_fabric_family_s_group_is_refused_with_the_live_one(monkeypatch):
    """The family's group object is marked refused with the live one; unmarked, the pipeline
    sends through its freed communicator (decision 0025)."""
    from everserve.comm.collective import GroupFamily
    from everserve.comm.layout import Scope
    from everserve_vllm import collective_op

    seen = {}

    def refuse_engine_groups(skip=()):
        seen["skip"] = skip
        return []

    monkeypatch.setattr(collective_op, "_refuse_engine_groups", refuse_engine_groups)
    pp_live = FakeGroup(group="pp-comm", ranks=[0, 2])
    pp_next = FakeGroup(group="pp-comm-next", ranks=[0, R])
    fabric = GroupFamily(Scope.FABRIC, pp_live, {(0, 1, R, 3): pp_next})
    c, live, _rep, _slot = make(fabric=fabric)
    c.refuse()
    assert live.refused and pp_live.refused, "the family's current group was not refused"
    assert seen["skip"] == (None, "pp-comm")
    c.change((0, 1, R, 3), epoch=1)
    c.apply_staged()
    assert fabric.current is pp_next and not pp_next.refused


def test_slot_rows_are_the_stage_s_columns_under_a_layout():
    from everserve.comm.layout import ParallelLayout

    layout = ParallelLayout(tensor_parallel=4, pipeline_parallel=2)
    world = tuple(range(8))
    c = VllmCollective(5, FakeGroup(ranks=[4, 5, 6, 7]), {}, None, spare_row=4, layout=layout)
    assert c.rows_of(world) == [0, 1, 2, 3]  # stage 1's members, by column
    c.members = world
    assert c.rows_of(c.replacement_set((6,))) == [0, 1, 4, 3]  # the spare's row in column 2
    assert c.rows_of(c.replacement_set((1,))) == [0, 1, 2, 3]  # another stage's loss: no row here
    spare = VllmCollective(replacement(6), LocalStandIn(), {}, None, spare_row=4, layout=layout)
    spare.members = world
    assert spare.rows_of(spare.replacement_set((6,))) == [0, 1, 4, 3]


def test_the_fabric_family_is_swapped_at_the_boundary_and_gates_the_staging():
    from everserve.comm.collective import GroupFamily
    from everserve.comm.layout import ParallelLayout, Scope

    layout = ParallelLayout(tensor_parallel=4, pipeline_parallel=2)
    world = tuple(range(8))
    live_tp, rep_tp = FakeGroup(ranks=[4, 5, 6, 7]), FakeGroup(group="tp-rep", ranks=[4, 5, 6, 7])
    live_pp, rep_pp = FakeGroup(ranks=[1, 5]), FakeGroup(group="pp-rep", ranks=[1, 5])
    target = tuple(replacement(r) if r == 6 else r for r in world)
    fabric = GroupFamily(Scope.FABRIC, live_pp, {target: rep_pp})
    applied, applied_fabric = [], []
    c = VllmCollective(
        5,
        live_tp,
        {target: rep_tp},
        None,
        spare_row=4,
        on_apply=applied.append,
        layout=layout,
        fabric=fabric,
        on_apply_fabric=applied_fabric.append,
    )
    c.members = world
    assert c.has_shadow(target)
    c.change(target, epoch=1)
    assert fabric.current is live_pp  # staged: nothing moves before the boundary
    assert c.apply_staged()["members"] == set_key(target)
    assert c.current is rep_tp and fabric.current is rep_pp
    assert applied == [rep_tp] and applied_fabric == [rep_pp]
    # a set the fabric has no group for cannot be staged, so no step runs on half a change
    other = tuple(replacement(r) if r == 7 else r for r in world)
    c.preform({other: rep_tp})
    assert not c.has_shadow(other)
    with pytest.raises(KeyError):
        c.change(other, epoch=2)
    assert c.staged is None


def test_ensure_makes_the_collective_once_when_no_collective_has_run(monkeypatch):
    from everserve_vllm import collective_op

    made = []
    monkeypatch.setattr(collective_op, "_COLLECTIVE", None)
    monkeypatch.setattr(collective_op, "_make", lambda name: made.append(name) or object())
    first = collective_op.ensure()
    assert collective_op.ensure("tp:0") is first and collective_op.collective() is first
    assert made == ["tp:0"]  # an engine of tensor parallelism one never calls the op


def test_on_a_spare_node_a_replacement_s_row_is_its_column():
    from everserve.comm.collective import LocalStandIn
    from everserve.comm.layout import ParallelLayout
    from everserve.comm.shadow import replacement

    layout = ParallelLayout(2, 2)
    on_engine_node = VllmCollective(
        replacement(3), LocalStandIn(), {}, None, spare_row=2, layout=layout
    )
    assert on_engine_node.rows_of((0, 1, 2, replacement(3))) == [
        0,
        2,
    ]  # the engine node's spare row
    on_spare_node = VllmCollective(
        replacement(3),
        LocalStandIn(),
        {},
        None,
        spare_row=2,
        layout=layout,
        spare_rows_at_columns=True,
    )
    assert on_spare_node.rows_of((0, 1, replacement(2), replacement(3))) == [0, 1]  # the columns


def test_the_policy_s_release_bound_decides_when_a_release_counts_as_leaked(monkeypatch):
    """The bound set at bootstrap, not the module's default, is the one a release runs past."""
    import time

    from everserve_vllm import collective_op, groups

    stuck = groups.PyNcclGroup.over(SlowComm(5.0), [0, 1, 2, 3])
    monkeypatch.setattr(collective_op, "_refuse_engine_groups", lambda skip=(): [])
    c = VllmCollective(0, stuck, {}, FakeSlot(), spare_row=4)
    c.release_bound_s = 0.2  # the policy's, through Bootstrap.release_bound_s
    c.refuse()
    c.change_to_group((0, 1, R, 3), FakeGroup(group="replacement-comm"), epoch=1)
    c.apply_staged()
    assert c.releases.done == [] and c.releases.leaked == 0
    time.sleep(0.3)
    c.releases.reap(c.release_bound_s)
    assert c.releases.done == ["live still running"] and c.releases.leaked == 1
    assert groups.RELEASE_BOUND_S == 10.0  # the default stood


def test_the_compilation_config_splits_at_the_op_and_sets_inductor_s_deterministic_mode(
    monkeypatch,
):
    """Decision 0029: no reduction's tiles decided by timing on any launch of an engine."""
    import sys
    import types

    from everserve_vllm import collective_op

    fake = types.ModuleType("vllm.config.compilation")

    class CompilationConfig:
        _attention_ops = ["vllm::unified_attention", "vllm::unified_attention_with_output"]

    fake.CompilationConfig = CompilationConfig
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", types.ModuleType("vllm.config"))
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", fake)
    cfg = collective_op.compilation_config()
    assert cfg["splitting_ops"][-1] == collective_op.OP
    assert cfg["splitting_ops"][:2] == CompilationConfig._attention_ops
    assert cfg["inductor_compile_config"] == {
        "deterministic": True,
        "combo_kernels": False,  # vLLM would turn them on with a benchmark deterministic mode bans
        "benchmark_combo_kernel": False,
    }
    assert cfg["inductor_compile_config"] is not collective_op.INDUCTOR  # a copy per engine
    # the guarantee class decides it (decision 0030): exact compiles deterministically, as an
    # engine whose policy names no class does; continuous keeps vLLM's defaults
    from everserve.policy import Guarantee

    exact = collective_op.compilation_config(Guarantee.EXACT)
    assert exact["inductor_compile_config"] == cfg["inductor_compile_config"]
    cont = collective_op.compilation_config(Guarantee.CONTINUOUS)
    assert "inductor_compile_config" not in cont and cont["splitting_ops"] == cfg["splitting_ops"]


def test_a_formed_group_is_warmed_with_one_operation_of_every_kind():
    """A pair runs an all-reduce, a broadcast and a send and receive each way; a wider group
    the collectives only; a group of one nothing."""
    from everserve_vllm.groups import warm_group

    class Comm:
        def __init__(self):
            self.calls = []

        def all_reduce(self, t, *a, **k):
            self.calls.append("all_reduce")

        def broadcast(self, t, src, *a, **k):
            self.calls.append(f"broadcast<{src}")

        def send(self, t, dst, *a, **k):
            self.calls.append(f"send>{dst}")

        def recv(self, t, src, *a, **k):
            self.calls.append(f"recv<{src}")

    c = Comm()
    warm_group(c, 0, 2, torch.device("cpu"))
    assert c.calls == ["all_reduce", "broadcast<0", "send>1", "recv<1"]
    c = Comm()
    warm_group(c, 1, 2, torch.device("cpu"))
    assert c.calls == ["all_reduce", "broadcast<0", "recv<0", "send>0"]  # the pair's mirror order
    c = Comm()
    warm_group(c, 2, 4, torch.device("cpu"))
    assert c.calls == ["all_reduce", "broadcast<0"]
    c = Comm()
    warm_group(c, 0, 1, torch.device("cpu"))
    assert c.calls == []
