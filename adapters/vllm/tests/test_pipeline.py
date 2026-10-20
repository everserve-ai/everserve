# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""vLLM's pipeline traffic over the fabric family: the tensor dict and the sampled-token
broadcast go over the family's current group once this process is bootstrapped, over vLLM's own
path before, and a spare before its join sends nothing."""

from types import SimpleNamespace

import pytest
import torch

from everserve.comm.collective import LocalStandIn
from everserve.comm.layout import ParallelLayout
from everserve.comm.runtime import CHANGE_ABORT
from everserve_vllm import pipeline, worker


class FakeComm:
    def __init__(self):
        self.wire = []
        self.broadcasts = []
        self.device = "cpu"

    def send(self, tensor, dst):
        self.wire.append((dst, tensor.clone()))

    def recv(self, tensor, src):
        _, sent = self.wire.pop(0)
        tensor.copy_(sent)

    def broadcast(self, tensor, src):
        self.broadcasts.append((src, tensor.clone()))


def fake_group_class():
    """vLLM's class as the install sees it: the methods the pipeline uses, by name; fresh per test,
    since the install patches the class."""

    class FakeGroupCoordinator:
        def __init__(self, unique_name, world_size, rank_in_group):
            self.unique_name, self.world_size = unique_name, world_size
            self.rank_in_group = rank_in_group
            self.calls = []

        def send_tensor_dict(self, tensor_dict, dst=None, **kw):
            self.calls.append(("send", dst))

        def recv_tensor_dict(self, src=None, **kw):
            self.calls.append(("recv", src))
            return {"original": True}

        def isend_tensor_dict(self, tensor_dict, dst=None, **kw):
            self.calls.append(("isend", dst))
            return ["original-handle"]

        def irecv_tensor_dict(self, src=None, **kw):
            self.calls.append(("irecv", src))
            return {"original": True}, [], []

        def make_sibling_device_group(self, group_desc=None):
            self.calls.append(("sibling", group_desc))
            return "torch-sibling"

    return FakeGroupCoordinator


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(pipeline, "_ORIGINALS", {})
    original_broadcast = torch.distributed.broadcast
    cls = fake_group_class()
    pipeline.install(cls)
    yield cls
    torch.distributed.broadcast = original_broadcast


def _bootstrapped(monkeypatch, current):
    fabric = SimpleNamespace(current=current)
    collective = SimpleNamespace(fabric=fabric, layout=ParallelLayout(2, 2))
    monkeypatch.setattr(worker, "STATE", SimpleNamespace(collective=collective))


def _v2_runner(monkeypatch, tensors, pad=0, decode_query_len=1):
    """A model-runner-v2 shape: request state, graph manager, data-parallel attributes, and
    vLLM's two padding functions faked to pad by ``pad`` when the batch is a uniform decode."""
    import numpy as np

    states = SimpleNamespace(
        req_id_to_index={"a": 0, "b": 1},
        prefill_len=SimpleNamespace(np=np.array([4, 4], dtype=np.int32)),
    )

    def uniform_count(num_reqs, num_tokens, max_query_len, eligible):
        return 1 if eligible and num_tokens == num_reqs else None

    def dispatch(manager, num_reqs, num_tokens, uniform, dp_size, dp_rank, **kw):
        assert (
            manager == "graphs"
            and dp_size == 1
            and dp_rank == 0
            and kw["uniform_decode"] == (uniform == decode_query_len)
        )
        return SimpleNamespace(num_tokens=num_tokens + (pad if uniform else 0)), None

    monkeypatch.setattr(pipeline, "_v2_dispatch", lambda: (uniform_count, dispatch))
    return SimpleNamespace(
        intermediate_tensors=SimpleNamespace(tensors=tensors),
        req_states=states,
        cudagraph_manager="graphs",
        dp_size=1,
        dp_rank=0,
        parallel_config=None,
        decode_query_len=decode_query_len,
    )


def test_before_bootstrap_vllm_s_own_path_runs(installed, monkeypatch):
    monkeypatch.setattr(worker, "STATE", None)
    g = installed("pp:0", 2, 0)
    assert g.isend_tensor_dict({"h": torch.zeros(1)}) == ["original-handle"]
    assert g.irecv_tensor_dict()[0] == {"original": True}
    assert g.calls == [("isend", None), ("irecv", None)]


def test_once_bootstrapped_the_tensors_cross_the_family_s_group(installed, monkeypatch):
    comm = FakeComm()
    _bootstrapped(monkeypatch, SimpleNamespace(group=comm))
    sender, receiver = installed("pp:0", 2, 0), installed("pp:0", 2, 1)
    handles = sender.isend_tensor_dict({"hidden_states": torch.arange(6.0).reshape(2, 3)})
    assert [type(h).__name__ for h in handles] == ["Done"] and comm.wire[0][0] == 1  # to stage 1
    got, handles, post = receiver.irecv_tensor_dict()
    assert torch.equal(got["hidden_states"], torch.arange(6.0).reshape(2, 3)) and post == []
    assert sender.calls == [] and receiver.calls == []  # vLLM's path untouched
    tp = installed("tp:0", 2, 0)  # a TP group keeps vLLM's methods
    tp.send_tensor_dict({}, dst=1)
    assert tp.calls == [("send", 1)]


def test_the_sampled_token_broadcast_goes_over_the_family_from_the_last_stage(
    installed, monkeypatch
):
    comm = FakeComm()
    _bootstrapped(monkeypatch, SimpleNamespace(group=comm))
    g = installed("pp:0", 2, 0)
    group = g.make_sibling_device_group("pp_broadcast")
    assert isinstance(group, pipeline.FamilyBroadcastGroup) and group.original is None
    t = torch.tensor([7, 8])
    torch.distributed.broadcast(t, src=3, group=group)  # engine rank 3 is stage 1 under TP2 x PP2
    assert [(src, b.tolist()) for src, b in comm.broadcasts] == [(1, [7, 8])]


def test_a_spare_before_its_join_has_no_group_to_broadcast_on(installed, monkeypatch):
    _bootstrapped(monkeypatch, LocalStandIn())
    group = pipeline.FamilyBroadcastGroup(original=None)
    with pytest.raises(RuntimeError, match="before this process joined"):
        torch.distributed.broadcast(torch.zeros(1), src=0, group=group)


def test_a_refused_family_group_refuses_the_exchange_instead_of_touching_its_comm(
    installed, monkeypatch
):
    comm = FakeComm()
    _bootstrapped(monkeypatch, SimpleNamespace(group=comm, refused=True))
    g = installed("pp:0", 2, 0)
    with pytest.raises(pipeline.MemberSetChanged, match="member set changed"):
        g.isend_tensor_dict({"h": torch.zeros(1)})
    # vLLM's sampled-token broadcast (sample_tokens' PP receive): the path a node fault refuses
    with pytest.raises(pipeline.MemberSetChanged) as refused:
        torch.distributed.broadcast(
            torch.zeros(1), src=3, group=pipeline.FamilyBroadcastGroup(None)
        )
    # the executor reads the refusal from its text alone (vLLM's worker RPC carries no type)
    assert CHANGE_ABORT in str(refused.value)
    assert comm.wire == [] and comm.broadcasts == []


def test_a_refusal_during_the_exchange_s_wait_keeps_the_enqueue_off_the_comm(
    installed, monkeypatch
):
    """The check the exchange reads after its wait is bound to the group the call found."""
    comm = FakeComm()
    group = SimpleNamespace(group=comm, refused=False)
    _bootstrapped(monkeypatch, group)
    found = pipeline._family()
    assert found is not None
    _, _, check = found
    check()  # nothing refused yet
    group.refused = True  # a loss refused it while the call waited
    with pytest.raises(pipeline.MemberSetChanged, match="member set changed"):
        check()
    assert comm.wire == [] and comm.broadcasts == []


def test_a_step_whose_layout_the_stage_can_name_crosses_as_one_operation(installed, monkeypatch):
    """Both stages pack: the sender from its tensors, the receiver from the runner and the step."""
    from types import SimpleNamespace

    comm = FakeComm()
    _bootstrapped(monkeypatch, SimpleNamespace(group=comm, refused=False))
    pre = {"hidden_states": torch.zeros(16, 4), "residual": torch.zeros(16, 4)}

    monkeypatch.setattr(pipeline, "RUNNER", _v2_runner(monkeypatch, pre, pad=1))
    monkeypatch.setattr(
        pipeline,
        "_SCHEDULER_OUTPUT",
        SimpleNamespace(
            total_num_scheduled_tokens=2,
            num_scheduled_tokens={"a": 1, "b": 1},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"], num_computed_tokens=[9, 9]),
        ),
    )
    g = installed("pp:0", 2, 0)
    before = dict(pipeline.EXCHANGE.stats)
    h = torch.arange(12, dtype=torch.float32).view(3, 4)
    g.isend_tensor_dict({"hidden_states": h, "residual": -h})
    assert pipeline.EXCHANGE.stats["sends"] - before["sends"] == 1
    r = installed("pp:0", 2, 1)
    got, _, _ = r.irecv_tensor_dict()
    assert pipeline.EXCHANGE.stats["recvs"] - before["recvs"] == 1
    assert torch.equal(got["hidden_states"], h) and torch.equal(got["residual"], -h)
    # outside a step (no scheduler output: a warm-up) the header path carries it, on both sides
    monkeypatch.setattr(pipeline, "_SCHEDULER_OUTPUT", None)
    g.isend_tensor_dict({"hidden_states": h, "residual": -h})
    assert pipeline.EXCHANGE.stats["sends"] - before["sends"] == 1 + 4
    got2, _, _ = r.irecv_tensor_dict()
    assert torch.equal(got2["residual"], -h)
    # inside a step the first stage packs without a layout of its own (it has no intermediate
    # tensors); a receiving stage that cannot name the layout refuses rather than hang
    monkeypatch.setattr(
        pipeline,
        "_SCHEDULER_OUTPUT",
        SimpleNamespace(total_num_scheduled_tokens=2, num_scheduled_tokens={"a": 1, "b": 1}),
    )
    monkeypatch.setattr(pipeline, "RUNNER", None)
    g.isend_tensor_dict({"hidden_states": h, "residual": -h})
    assert pipeline.EXCHANGE.stats["sends"] - before["sends"] == 1 + 4 + 1
    with pytest.raises(RuntimeError, match="cannot name"):
        r.irecv_tensor_dict()


def test_a_step_whose_tensors_cannot_go_as_one_buffer_raises_rather_than_fall_back(
    installed, monkeypatch
):
    """Inside a step the receiver expects one packed operation: a sender with mixed tensors
    raises before sending anything, since a header send against it would hang both stages."""
    from types import SimpleNamespace

    comm = FakeComm()
    _bootstrapped(monkeypatch, SimpleNamespace(group=comm, refused=False))
    monkeypatch.setattr(
        pipeline,
        "_SCHEDULER_OUTPUT",
        SimpleNamespace(total_num_scheduled_tokens=2, num_scheduled_tokens={"a": 1, "b": 1}),
    )
    g = installed("pp:0", 2, 0)
    before = dict(pipeline.EXCHANGE.stats)
    mixed = {"hidden_states": torch.zeros(3, 4), "residual": torch.zeros(3, 4, dtype=torch.int8)}
    with pytest.raises(RuntimeError, match="not one dtype and one shape"):
        g.isend_tensor_dict(mixed)
    assert pipeline.EXCHANGE.stats["sends"] == before["sends"] and not comm.wire


def test_the_exchange_depth_is_the_stage_count_and_nothing_configures_the_route():
    from everserve.comm.layout import ParallelLayout
    from everserve_vllm import worker

    fields = worker.Bootstrap.__dataclass_fields__
    assert "pipeline_packed" not in fields and "exchange_in_flight" not in fields
    assert not hasattr(pipeline, "PACKED") and not hasattr(pipeline, "set_packed")
    cfg = worker.Bootstrap("e", "n", (), "", 4, (), ParallelLayout(2, 2), {}, 0)
    assert cfg.exchange_depth == 2


def test_the_padded_count_follows_v2_s_rule_from_the_scheduler_output(monkeypatch):
    """A uniform decode pads to the graph's bucket; a step with a prefill does not, and a
    prefilling request's one-token chunk past its first token still counts as a decode."""
    from types import SimpleNamespace

    runner = _v2_runner(monkeypatch, {}, pad=6)
    decode = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"a": 1, "b": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"], num_computed_tokens=[9, 9]),
    )
    assert pipeline.padded_tokens(runner, decode) == 8
    prefill = SimpleNamespace(
        total_num_scheduled_tokens=5,
        num_scheduled_tokens={"c": 4, "b": 1},
        scheduled_new_reqs=[
            SimpleNamespace(req_id="c", prefill_token_ids=[1, 2, 3, 4], num_computed_tokens=0)
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["b"], num_computed_tokens=[9]),
    )
    assert pipeline.padded_tokens(runner, prefill) == 5  # not a uniform decode: no padding
    chunk = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"a": 1, "b": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"], num_computed_tokens=[3, 9]),
    )
    assert pipeline.padded_tokens(runner, chunk) == 8  # a's last prompt token runs as a decode
    with pytest.raises(AttributeError):  # the old runner has no request state
        pipeline.padded_tokens(SimpleNamespace(intermediate_tensors=None), decode)


def test_a_stage_alone_receives_its_own_step_s_tensors_and_sends_nothing(installed, monkeypatch):
    """A spare's kernel warm-up past the first stage: inside ``local_stage`` a receive is the
    step's intermediate tensors made here at the runner's layout, a send and the sampled-token
    broadcast are dropped, and outside the block vLLM's own path runs as before."""
    from types import SimpleNamespace

    pre = {"hidden_states": torch.zeros(16, 4), "residual": torch.zeros(16, 4)}
    runner = _v2_runner(monkeypatch, pre, pad=1)
    runner.device = "cpu"
    so = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"a": 1, "b": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"], num_computed_tokens=[9, 9]),
    )
    r = installed("pp:0", 2, 1)
    seen = []

    def execute_model(scheduler_output):
        got = r.recv_tensor_dict()
        seen.append({k: tuple(v.shape) for k, v in got.items()})
        assert r.send_tensor_dict({"x": torch.ones(1)}) is None
        assert r.isend_tensor_dict({"x": torch.ones(1)}) and r.calls == []
        t = torch.ones(2)
        assert torch.distributed.broadcast(t, 1, group=pipeline.FamilyBroadcastGroup(None)) is None
        assert torch.equal(t, torch.ones(2))
        return "stepped"

    with pipeline.local_stage(runner):
        assert pipeline.stepping(execute_model)(so) == "stepped"
        with pytest.raises(RuntimeError, match="running alone"):
            r.irecv_tensor_dict()  # outside a step the stage cannot name its layout
    assert seen == [{"hidden_states": (3, 4), "residual": (3, 4)}]  # 2 tokens padded to 3
    assert pipeline.RUNNER is None and pipeline._LOCAL is None
    assert r.recv_tensor_dict() == {"original": True} and r.calls == [("recv", None)]


def test_a_stage_alone_refuses_a_step_it_cannot_lay_out(installed, monkeypatch):
    runner = _v2_runner(monkeypatch, {}, pad=0)
    r = installed("pp:0", 2, 1)
    with pipeline.local_stage(runner), pytest.raises(RuntimeError, match="running alone"):
        pipeline.stepping(lambda so: r.recv_tensor_dict())(
            SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"a": 1})
        )
