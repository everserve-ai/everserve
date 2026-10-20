# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The node role's bookkeeping of a change, the step word a worker keeps, and the entry point's
arguments. The executor class itself imports vLLM; its change is checked where vLLM is installed,
and the two-node harness exercises the rest."""

import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from everserve.comm.layout import ParallelLayout
from everserve.comm.slotkernel import Region
from everserve_vllm import worker
from everserve_vllm.node_main import parse


@pytest.fixture
def region():
    shm = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    r = Region(str(Path(tempfile.mkdtemp(dir=shm)) / "slot"), create=True)
    yield r
    r.close()


class FakeWorker:
    """vLLM's RPC target: the step methods are looked up on it by name; each notes ``region``."""

    def __init__(self, region=None):
        self.region = region
        self.calls = []

    def _note(self, call):
        self.calls.append((call, self.region.in_flight() if self.region else None))

    def execute_model(self, scheduler_output):
        self._note(("execute_model", scheduler_output))
        return "hidden"

    def sample_tokens(self, grammar_output):
        self._note(("sample_tokens", grammar_output))
        raise RuntimeError("a failed step still ends")


def bare(cls):
    """An executor of ``cls`` with no workers, its start-up record as ``_init_executor`` sets it."""
    from everserve_vllm import replay

    ex = cls.__new__(cls)
    ex.startup = replay.Recorder()
    return ex


def test_a_worker_keeps_its_step_word_while_it_runs_a_step_rpc(region, monkeypatch):
    region.set_members([0, 1], [0, 1])
    slot = SimpleNamespace(region=region, row=1)
    monkeypatch.setattr(worker, "STATE", SimpleNamespace(collective=SimpleNamespace(slot=slot)))
    w = FakeWorker(region)
    worker.mark_steps(w)
    assert w.execute_model("out") == "hidden"
    with pytest.raises(RuntimeError):
        w.sample_tokens("grammar")
    assert [seen for _, seen in w.calls] == [True, True]  # in flight while each ran
    assert not region.in_flight() and region.steps(1) == 2


def test_without_a_slot_region_no_word_is_kept(monkeypatch):
    monkeypatch.setattr(worker, "STATE", SimpleNamespace(collective=SimpleNamespace(slot=None)))
    w = FakeWorker()
    worker.mark_steps(w)
    assert w.execute_model("out") == "hidden" and w.calls == [(("execute_model", "out"), None)]


def test_a_taken_position_of_this_stage_goes_to_the_spare_s_row():
    layout = ParallelLayout(2, 2)
    rows = [0, 1]
    assert worker.take_rows(layout, 2, rows, [3]) == [1] and rows == [0, 2]
    assert worker.take_rows(layout, 2, rows, [0, 1]) == [] and rows == [0, 2]  # the other stage
    rows = [0, 1]
    assert worker.take_rows(layout, 0, rows, [0, 1]) == [0, 1] and rows == [2, 2]  # a whole stage


def test_a_change_moves_the_spare_s_row_in_and_clears_the_rows_that_left(region):
    pytest.importorskip("vllm")
    from everserve_vllm.node import NodeExecutor

    tp = 2
    n = bare(NodeExecutor)
    n.region, n.layout, n.slot_rows = region, ParallelLayout(tp, 2), [0, 1]
    region.set_members(n.slot_rows, n.slot_rows)
    region.set_members([tp], [tp])
    n.local_world_size = tp
    n.parallel_config = SimpleNamespace(node_rank_within_dp=1)  # the second stage: ranks 2, 3
    n.workers = []
    n.taken, n.claimed, n.dead, n._fenced, n.events = {}, {1: (3,)}, {3}, {3}, []
    n.departing = set()
    n.claimed_takes, n.left, n.node, n.engine_policy = {}, {}, "node-b", None  # a spare here
    n.epoch = 0
    n.start_worker_monitor = lambda: n.events.append({"kind": "monitor"})
    region.begin_step(1)  # rank 3 died inside a step
    region.set_abort([1], 1)
    out = n.node_change(SimpleNamespace(committed=SimpleNamespace(epoch=1, members=()), at=0.0))
    assert out["took"] == [3] and n.epoch == 1
    assert n.slot_rows == [0, 2]  # the executor's record; each member writes its own row's order
    assert region.order(0) == [0, 1] and region.order(2) == [2]
    assert not region.in_flight() and int(region.ctrl[1, 1]) == 0  # its step and abort words
    assert n.dead == set() and n._fenced == set() and n.taken == {3: {"epoch": 1}}
    assert [e["kind"] for e in n.events] == ["monitor", "changed"]


def test_a_position_moved_to_another_node_has_left_not_been_taken_here(region):
    pytest.importorskip("vllm")
    from everserve.contract.signals import Take
    from everserve_vllm.node import NodeExecutor

    n = bare(NodeExecutor)
    n.region, n.layout, n.slot_rows = region, ParallelLayout(2, 1), [0, 1]
    n.local_world_size = 2
    n.parallel_config = SimpleNamespace(node_rank_within_dp=0)
    n.workers = []
    n.taken, n.dead, n._fenced, n.events, n.left = {}, set(), set(), [], {}
    n.departing = set()
    n.claimed = {1: (0, 1)}  # a notice moves the node's two ranks
    n.claimed_takes = {1: (Take(0, "node-c", 0),)}  # rank 0 to a spare standing on node-c
    n.node, n.epoch = "node-b", 0
    n.engine_policy = SimpleNamespace(spare_node=lambda r: {1: "node-d"}.get(r))  # rank 1's
    n.start_worker_monitor = lambda: None
    out = n.node_change(SimpleNamespace(committed=SimpleNamespace(epoch=1, members=()), at=0.0))
    assert out["left"] == {0: "node-c", 1: "node-d"} and n.taken == {}
    assert {r: (v["epoch"], v["to"]) for r, v in n.left.items()} == {
        0: (1, "node-c"),
        1: (1, "node-d"),
    }


def test_the_entry_point_takes_the_engine_s_arguments_and_the_node_s_place():
    a = parse(
        [
            "--engine", "rig", "--node", "node-b", "--policy", "p.json", "--discovery-dir", "/d",
            "--model", "/m", "--tensor-parallel-size", "2", "-pp", "2", "--max-model-len", "4096",
            "--nnodes", "2", "--node-rank", "1", "--master-addr", "10.0.0.1",
        ]
    )  # fmt: skip
    assert (a.node, a.node_rank, a.nnodes, a.tp, a.pp, a.master_port) == (
        "node-b", 1, 2, 2, 2, 29501
    )  # fmt: skip
    # vLLM's own engine arguments pass through to vLLM's parser untouched, the head's alike
    assert a.engine_argv == [
        "--model", "/m", "--tensor-parallel-size", "2", "-pp", "2", "--max-model-len", "4096"
    ]  # fmt: skip


def test_a_set_whose_group_is_this_process_alone_gets_a_stand_in():
    from everserve.comm.collective import LocalStandIn

    layout = ParallelLayout(1, 2)  # TP1 x PP2: every tensor group is one member
    sets = [(0, 1), (0, "R1"), ("R0", 1)]
    shadows = {}
    worker.stand_ins(shadows, sets, 0, lambda s: layout.tensor_members(s, 0))
    assert set(shadows) == {(0, 1), (0, "R1")} and all(
        isinstance(g, LocalStandIn) for g in shadows.values()
    )
    pipeline = {(0, 1): "formed"}
    worker.stand_ins(pipeline, sets, 0, lambda s: layout.pipeline_members(s, 0))
    assert pipeline == {(0, 1): "formed"}  # pipeline groups of two: the rendezvous formed them


@pytest.mark.parametrize("cls", ["EverserveExecutor", "NodeExecutor"])
def test_only_the_head_registers_as_the_engine_s_head_and_says_goodbye(cls, monkeypatch):
    pytest.importorskip("vllm")
    from everserve_vllm import executor, node

    sent = []

    class Client:
        def __init__(self, *args, **kwargs):
            sent.append(("register", kwargs["head"], bool(kwargs["launcher_pid"])))
            self.changes = self.abandoned = None
            self.replay_asks = queue.Queue()  # the head's replay server reads it

        def goodbye(self, *cause):
            sent.append(("goodbye",))

        def close(self):
            sent.append(("close",))

    monkeypatch.setattr(executor, "DaemonClient", Client)
    monkeypatch.setattr(executor, "software_identity", lambda config: None)
    monkeypatch.setattr(executor.threading, "Thread", lambda *a, **k: SimpleNamespace(start=int))
    ex = bare(getattr(node, cls))
    ex.slot_choice, ex.failure, ex.dead, ex.vllm_config = None, None, set(), None
    ex._connect("addr", "engine", "node-b")
    ex._leave()
    head = cls == "EverserveExecutor"
    assert sent == [("register", head, head), ("goodbye",) if head else ("close",)]


def test_a_node_that_heard_only_a_commit_before_protocol_9_learns_the_ranks_it_lost_from_it(region):
    pytest.importorskip("vllm")
    from everserve_vllm.node import NodeExecutor

    n = bare(NodeExecutor)
    n.region, n.layout, n.slot_rows = region, ParallelLayout(1, 3), [0]
    n.local_world_size = 1
    n.parallel_config = SimpleNamespace(node_rank_within_dp=1)  # node-b: rank 1
    n.workers = []
    n.taken, n.dead, n._fenced, n.events, n.left = {}, set(), set(), [], {}
    n.departing = set()
    n.claimed, n.claimed_takes = {}, {}  # the claimed change never reached this node
    n.node, n.epoch = "node-b", 0
    n.engine_policy = SimpleNamespace(spare_node=lambda r: {1: "node-d"}.get(r))
    n.start_worker_monitor = lambda: None
    committed = SimpleNamespace(epoch=1, lost=(1,), members=())  # a commit before protocol 9
    out = n.node_change(SimpleNamespace(committed=committed, at=0.0))
    assert out["took"] == [1] and (n.left[1]["epoch"], n.left[1]["to"]) == (1, "node-d")


def _node_b(region):
    """Node-b of a TP1 x PP3 engine, rank 1 launched here, no claim heard."""
    from everserve_vllm.node import NodeExecutor

    n = bare(NodeExecutor)
    n.region, n.layout, n.slot_rows = region, ParallelLayout(1, 3), [0]
    n.local_world_size = 1
    n.parallel_config = SimpleNamespace(node_rank_within_dp=1)
    n.workers = [SimpleNamespace(rank=1)]
    n.taken, n.dead, n._fenced, n.events, n.left, n.released = {}, set(), set(), [], {}, {}
    n.departing = set()
    n.claimed, n.claimed_takes = {}, {}
    n.node, n.epoch, n.engine_policy = "node-b", 0, None
    n.start_worker_monitor = lambda: None
    return n


def test_the_commit_s_members_say_a_rank_launched_here_has_left_whatever_claims_were_missed(
    region,
):
    pytest.importorskip("vllm")
    from everserve.contract.signals import RankPlacement

    n = _node_b(region)
    # node-b heard no claim, and the commit names no rank lost under REPLACE
    members = (RankPlacement(0, "node-a", 0), RankPlacement(1, "node-d", 0))
    members += (RankPlacement(2, "node-c", 0),)
    committed = SimpleNamespace(epoch=1, lost=(), members=members)
    out = n.node_change(SimpleNamespace(committed=committed, at=0.0))
    assert out["took"] == [1] and out["left"] == {1: "node-d"} and n.taken == {}
    first = dict(n.left[1])
    # a later commit, read by a node that relinked, changes nothing it already knows
    later = SimpleNamespace(epoch=2, lost=(), members=members)
    assert n.node_change(SimpleNamespace(committed=later, at=0.0))["left"] == {}
    assert n.left[1] == first


def test_a_worker_whose_position_left_and_never_released_is_ended_after_the_hang_bound(
    region, monkeypatch
):
    pytest.importorskip("vllm")
    from everserve_vllm import settings

    n = _node_b(region)
    ended = []
    n._end = ended.append
    bound = settings.current().hang_s
    n.left = {1: {"epoch": 2, "to": "node-e", "at": 100.0}}
    n._end_overdue(now=100.0 + bound - 1)
    assert n.released == {}  # inside the bound: its boundary may still drain and release it
    n._end_overdue(now=100.0 + bound + 1)
    for _ in range(100):
        if ended:
            break
        threading.Event().wait(0.01)
    assert n.released[1]["overdue"] is True and ended == [1]
    n._end_overdue(now=100.0 + bound + 2)  # once only
    assert ended == [1]


def test_a_rank_its_claim_named_a_fault_is_ended_at_once_and_a_departing_one_after_its_grace(
    region,
):
    pytest.importorskip("vllm")
    from everserve.contract.signals import RankPlacement

    n = _node_b(region)
    n.workers = [SimpleNamespace(rank=1)]
    members = (RankPlacement(0, "node-a", 0), RankPlacement(1, "node-d", 0))
    committed = SimpleNamespace(epoch=1, lost=(), members=members)
    n.claimed = {1: (1,)}  # the claim heard, rank 1 not departing: a fault, though alive
    n.node_change(SimpleNamespace(committed=committed, at=0.0))
    assert n.left[1]["grace"] is False
    m = _node_b(region)
    m.claimed, m.departing = {1: (1,)}, {1}  # the claim named it departing
    m.node_change(SimpleNamespace(committed=committed, at=0.0))
    assert m.left[1]["grace"] is True
    ended = []
    n._end = m._end = ended.append
    n._end_overdue(now=n.left[1]["at"])
    m._end_overdue(now=m.left[1]["at"])
    for _ in range(100):
        if ended:
            break
        threading.Event().wait(0.01)
    assert ended == [1] and 1 in n.released and m.released == {}


def test_a_leaving_node_s_notice_is_never_under_the_policy_s_minimum():
    """Its ranks' notice goes at the policy's minimum when the lead it was started with is under
    it: a shorter one is read as none, and its ranks are lost when its processes end."""
    pytest.importorskip("vllm")
    from everserve.contract.daemon import HearsEngine
    from everserve.policy import NOTICE_TRANSIT_S, Loading, Policy
    from everserve_vllm.node import NodeExecutor

    class Client:
        policy = Policy("v1", (), 1, Loading.PRELOADED, 1, 8000)
        notice_lead_s = HearsEngine.notice_lead_s

        def __init__(self):
            self.notices = []

        def notice(self, rank, deadline_ms):
            self.notices.append((rank, deadline_ms))

    n = bare(NodeExecutor)
    n.client = Client()
    n.workers = [SimpleNamespace(rank=2), SimpleNamespace(rank=3)]
    n.left, n.released = {}, {}
    assert n.leave(5.0, now=1000.0) == [2, 3]
    lead = 8.0 + NOTICE_TRANSIT_S
    assert n.client.notices == [(2, int((1000.0 + lead) * 1000)), (3, int((1000.0 + lead) * 1000))]


def test_a_node_whose_worker_died_learns_from_the_placement_that_its_rank_left():
    """The change replacing a rank lost here goes to its survivors and spares, not to this node:
    a crashed node that waits for it holds nothing and never restarts to stand as a spare. The
    placement every connection hears says where the rank is."""
    pytest.importorskip("vllm")
    from everserve.contract.signals import EnginePlacement, RankPlacement
    from everserve_vllm.node import NodeExecutor

    class Proc:
        def is_alive(self):
            return False  # SIGKILLed

    class Client:
        placement = EnginePlacement("proof", 0, (RankPlacement(1, "node-b", 0, ""),))

    n = bare(NodeExecutor)
    n.client, n.node, n.epoch = Client(), "node-b", 0
    n.workers = [SimpleNamespace(rank=1, proc=Proc())]
    n.dead, n.left, n.taken, n.events = {1}, {}, {}, []
    n._left_by_placement()
    assert n.left == {} and not n.holds_nothing()  # the placement at its own epoch: nothing moved
    Client.placement = EnginePlacement(
        "proof", 1, (RankPlacement(0, "node-a", 0, ""), RankPlacement(1, "node-c", 0, ""))
    )
    n._left_by_placement()
    assert n.left[1]["to"] == "node-c" and n.left[1]["grace"] is False
    assert n.holds_nothing()  # so serve() returns, the node ends, and its pod asks again
    n.dead = set()
    n.left = {}
    n._left_by_placement()
    assert n.left == {}  # a live worker's rank moves by a commit this node hears, not here
