# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The connector's lifecycle mapping (docs/CONTRACT.md item 3) against a fake committed-KV."""

import pytest
import torch

from everserve.comm.collective import Collective
from everserve.comm.handle import LocalStandIn
from everserve.comm.runtime import Runtime
from everserve.outcome import Outcome
from everserve.state.committed import CommittedKV
from everserve_vllm.lifecycle import Lifecycle, StepRecord


class FakeKV:
    def __init__(self):
        self.log = []

    def save(self, layer_name, kv_layer, slots):
        self.log.append(("save", layer_name, slots.tolist()))

    def commit(self, epoch, step, committed_tokens):
        self.log.append(("commit", epoch, step, dict(committed_tokens)))

    def fence(self, step):
        self.log.append(("fence", step))

    def committed(self, request_id):
        return 0

    def stats(self):
        return {}

    def timeline(self):
        return []

    def evicted(self, request_id, kept_tokens):
        self.log.append(("evicted", request_id, kept_tokens))

    def finished(self, request_id):
        self.log.append(("finished", request_id))

    def clear(self):
        self.log.append(("clear",))

    def restore(self, dead_rank, moves):
        self.log.append(("restore", dead_rank, list(moves)))


def runtime():
    c = Collective(0, LocalStandIn(), {})
    c.members = (0, 1, 2, 3)
    return Runtime(0, 4, torch.device("cpu"), c)


def test_the_scheduler_side_records_committed_counts_and_preemptions():
    rec = Lifecycle.record([("a", 2048, 1), ("b", 0, 2048)], preempted={"z", "y"})
    assert rec == StepRecord(committed={"a": 2049, "b": 2048}, preempted=("y", "z"))
    kv = FakeKV()
    Lifecycle(kv).finished("a")
    assert kv.log == [("finished", "a")]


def test_a_step_maps_onto_evicted_clear_save_and_commit_through_the_runtime():
    kv, rt = FakeKV(), runtime()
    assert isinstance(kv, CommittedKV)
    lc = Lifecycle()
    lc.attach(kv, rt)
    assert rt.attempt_callbacks == [kv.clear] and rt.commit_callbacks == [lc.commit]
    rec = StepRecord(committed={"a": 10}, preempted=("p",))
    lc.preemptions(rec)
    lc.begin(rec)
    lc.save("model.layers.0.attn", torch.zeros(1), torch.tensor([3, 4]))
    assert lc.saved("model.layers.0.attn") and not lc.saved("model.layers.1.attn")
    assert lc.end()
    assert kv.log == [
        ("evicted", "p", 0),
        ("clear",),
        ("save", "model.layers.0.attn", [3, 4]),
        ("commit", 0, 0, {"a": 10}),
    ]
    assert lc.step == 1 and lc.report()["steps"] == 1


def test_a_step_interrupted_by_a_change_is_not_committed():
    kv, rt = FakeKV(), runtime()
    rt.collective._shadows[(0, 1, "R2", 3)] = LocalStandIn()
    lc = Lifecycle()
    lc.attach(kv, rt)
    lc.begin(StepRecord(committed={"a": 11}))
    lc.save("model.layers.0.attn", torch.zeros(1), torch.tensor([5]))
    rt.on_fault(2, outcome=Outcome.REPLACE)  # the daemon's change lands mid-step
    assert not lc.end()
    assert kv.log[-1] == ("clear",) and not any(e[0] == "commit" for e in kv.log)
    assert lc.stats.skipped == 1
    # after the boundary the next step commits at the new epoch
    rt.at_boundary()
    lc.begin(StepRecord(committed={"a": 11}))
    assert lc.end() and kv.log[-1] == ("commit", 1, 0, {"a": 11})


def test_a_commit_is_followed_by_the_fence_of_the_step_before_it():
    """Decisions 0009 and 0019: commit N starts, then fence N-2 waits, so an agreed step is
    durable through the step two back and the store's round trip never sits on the step thread."""
    kv, rt = FakeKV(), Runtime(0, 1, "cpu", LocalStandIn())
    lc = Lifecycle()
    lc.attach(kv, rt)
    for n in (1, 2, 3, 4):
        lc.begin(StepRecord(committed={"a": n}))
        assert lc.end()
    commits_and_fences = [e for e in kv.log if e[0] in ("commit", "fence")]
    assert commits_and_fences == [
        ("commit", 0, 0, {"a": 1}),
        ("commit", 0, 1, {"a": 2}),
        ("commit", 0, 2, {"a": 3}),
        ("fence", 0),
        ("commit", 0, 3, {"a": 4}),
        ("fence", 1),
    ]


def test_the_timeline_summary_splits_a_step_into_its_spans():
    from everserve_vllm.lifecycle import timeline_summary

    t = 100.0
    records = [
        {
            "step": 0,
            "commit": t,
            "dequeue": t + 0.001,
            "ready": t + 0.010,
            "sent": t + 0.012,
            "fence_start": t + 0.011,
            "fence_end": t + 0.012,
        },
        {
            "step": 1,
            "commit": t + 0.02,
            "dequeue": t + 0.021,
            "ready": t + 0.030,
            "sent": t + 0.0315,
        },
    ]
    s = timeline_summary(records, tail=1)
    assert s["steps"] == 2
    assert s["spans"]["event_lag"]["n"] == 2 and s["spans"]["event_lag"]["p50_ms"] == pytest.approx(
        10
    )
    assert s["spans"]["sender"]["max_ms"] == pytest.approx(2)
    assert s["spans"]["fence_wait"]["n"] == 1 and s["spans"]["fence_wait"][
        "p50_ms"
    ] == pytest.approx(1)
    assert "fence_after_ack" in s["spans"]
    assert [r["step"] for r in s["last_ms_from_commit"]] == [1]
    assert s["last_ms_from_commit"][0]["sent"] == pytest.approx(11.5)
    # the head keeps the prefill's commit, with when each step's commit came
    assert [r["step"] for r in s["first_ms_from_commit"]] == [0, 1]
    assert [r["at_ms"] for r in s["first_ms_from_commit"]] == pytest.approx([0, 20])
