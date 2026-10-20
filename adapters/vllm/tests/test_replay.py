# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare's start-up replay (decision 0044): the core's record, the calls' check, and their run
on a spare's worker in the core's order, refused by name where they cannot run.

``test_replay_vllm.py`` holds the record to vLLM's engine core at the pin and carries a replay
over vLLM's own queue.
"""

import dataclasses
import os
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from everserve_vllm import replay


def configs(world=2, size=1024):
    return [SimpleNamespace(kv_cache_tensors=[SimpleNamespace(size=size)] * 2)] * world


def started(*, fit=False, world=2, size=1024):
    """A core's record as ``EngineCore._initialize_kv_caches`` sends it, the auto-fit included."""
    r = replay.Recorder()
    r.record("get_kv_cache_spec")  # a question, not a start-up call
    r.record("set_kv_cache_layout", ("HND",))
    r.record("determine_available_memory")
    if fit:
        r.record("update_max_model_len", (4096,))
    r.record("initialize_from_config", (configs(world, size),))
    r.close()
    r.record("execute_model", ("batch",))  # after the warm-up: never a start-up call
    return r.complete()


class Wrapper:
    """vLLM's worker wrapper as far as a replay runs it: each call by name, recorded."""

    def __init__(self, global_rank=1):
        self.global_rank = global_rank
        self.ran = []

    def __getattr__(self, method):
        return lambda *args, **kwargs: self.ran.append((method, args))


def test_the_core_records_the_start_up_broadcasts_in_order_and_nothing_after_its_warm_up():
    calls = started(fit=True)
    assert [c[0] for c in calls.calls] == list(replay.METHODS)
    assert calls.calls[0] == ("set_kv_cache_layout", ("HND",), {})


def test_a_record_is_read_only_once_complete():
    r = replay.Recorder()
    with pytest.raises(RuntimeError, match="not all sent"):
        r.complete()
    stop = threading.Event()
    stop.set()
    assert r.wait(stop) is None  # a core that stops first serves nothing
    r.close()
    assert r.wait(threading.Event()) == replay.Calls(())


def test_a_spare_runs_the_core_s_calls_on_its_worker_in_the_core_s_order():
    calls = started(fit=True)
    w = Wrapper(global_rank=1)
    parts = {}
    replay.apply(w, 1, calls, parts, free=lambda: 1 << 30)
    assert [m for m, _ in w.ran] == list(replay.METHODS)
    # the whole list, as the core's workers got it: the wrapper takes its own rank's
    assert w.ran[-1][1] == (calls.calls[-1][1][0],)
    assert set(parts) == {f"{m}_s" for m in replay.METHODS}


def test_a_spare_whose_gpu_cannot_hold_the_rank_s_cache_is_refused_by_name_before_allocating():
    w = Wrapper(global_rank=1)
    with pytest.raises(replay.ReplayRefused) as e:
        replay.apply(w, 1, started(size=2000), free=lambda: 1999)
    assert str(e.value) == "rank 1's KV configuration needs 2000 bytes; this GPU has 1999 free"
    assert "initialize_from_config" not in [m for m, _ in w.ran]


def test_a_replay_for_another_rank_s_worker_or_past_the_core_s_world_is_refused():
    with pytest.raises(replay.ReplayRefused, match="on a worker built as rank 0"):
        replay.apply(Wrapper(global_rank=0), 1, started(), free=lambda: 1 << 30)
    with pytest.raises(replay.ReplayRefused, match="name 2 ranks, not 2"):
        replay.apply(Wrapper(global_rank=2), 2, started(world=2), free=lambda: 1 << 30)


def test_a_rank_s_device_bytes_are_its_one_backing_allocation_and_none_held_in_host_memory():
    """Every tensor of a configuration views one backing allocation (vLLM's
    ``allocate_kv_cache``): the groups overlay each other in it, so its bytes are counted once."""
    t = SimpleNamespace
    shared = t(kv_cache_tensors=[t(size=4096), t(size=4096), t(size=4096)])
    assert replay.kv_bytes(shared) == 4096
    host = t(kv_cache_tensors=[t(size=4096), t(size=1 << 40, host_resident=True)])
    assert replay.kv_bytes(host) == 4096
    assert replay.kv_bytes(t(kv_cache_tensors=[])) == 0  # an attention-free model


@pytest.mark.parametrize(
    ("calls", "named"),
    [
        ("calls", "a sequence of calls, not str"),
        ([("initialize_from_config", [1])], "call 0 is not a method, arguments and keywords"),
        ([("execute_model", (), {})], "call 0 is 'execute_model', not a start-up call"),
        (
            [("determine_available_memory", (), {}), ("set_kv_cache_layout", ("HND",), {})],
            "call 1 is 'set_kv_cache_layout' after 'determine_available_memory'",
        ),
        ([("set_kv_cache_layout", ("HND",), {})], "ends with the KV cache configuration"),
        ([], "ends with the KV cache configuration"),
    ],
)
def test_calls_that_are_not_the_core_s_start_up_in_its_order_are_refused_by_name(calls, named):
    with pytest.raises(replay.ReplayRefused, match=named):
        replay.check(calls)


def test_a_spare_asks_again_past_each_bound_and_reads_the_offer_that_comes(monkeypatch):
    asks, fetched = [], []

    def asking(bound):
        asks.append(bound)
        if len(asks) < 3:
            raise TimeoutError(f"no offer in {bound} s")  # no core yet, or the ask dropped
        return SimpleNamespace(rank=1, handle="h")

    def fetch(offer, timeout_s):
        fetched.append(offer.handle)
        return replay.Calls(())

    monkeypatch.setattr(replay, "fetch", fetch)
    assert replay.ask(asking, 1, timeout_s=60) == replay.Calls(())
    assert len(asks) == 3 and fetched == ["h"]


def test_a_spare_that_no_core_answers_is_told_why_within_its_bound(monkeypatch):
    def asking(bound):
        raise TimeoutError("no replay offer for rank 1 of k")

    monkeypatch.setattr(replay, "ASK_S", 0.05)
    with pytest.raises(TimeoutError, match="rank 1's replay not read in 0.2 s: no replay offer"):
        replay.ask(asking, 1, timeout_s=0.2)


@dataclasses.dataclass
class Spec:
    """A configuration as vLLM's are: dataclasses holding lists, a set, a dtype-like object."""

    layers: frozenset
    block_size: int
    dtype: object = "torch.bfloat16"


DIGEST_IN_A_PROCESS = """
import dataclasses, json, sys
sys.path.insert(0, {tests!r})
from test_replay import Spec
from everserve_vllm import replay
spec = Spec(frozenset(f"model.layers.{{i}}.attn" for i in range(40)), 16)
calls = replay.Calls((("initialize_from_config", ([spec, spec],), {{}}),))
print(replay.digest(calls))
"""


def test_equal_calls_digest_alike_in_any_process_whatever_its_hash_seed():
    """A set's order follows the process's hash seed, so a pickle's bytes are no digest: a spare
    and its core, or two cores, would differ on equal calls and a spare refuse for nothing."""
    import subprocess
    import sys
    from pathlib import Path

    code = DIGEST_IN_A_PROCESS.format(tests=str(Path(__file__).parent))
    digests = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert len(digests) == 1


def test_the_same_calls_are_the_same_record_and_another_configuration_is_not():
    calls = started()
    assert replay.digest(started()) == replay.digest(calls)
    assert replay.digest(started(size=2048)) != replay.digest(calls)
    assert replay.digest(started(fit=True)) != replay.digest(calls)


def test_a_spare_built_from_another_core_s_calls_refuses_its_join_by_name():
    from everserve_vllm import join as joins
    from everserve_vllm import spare_main

    built = started()

    def offer(startup):
        return joins.decode(joins.encode(joins.Offer(1, 3, "t", None, startup=startup)))

    assert spare_main.stale_for(offer(replay.digest(built)), built) == ""
    stale = spare_main.stale_for(offer(replay.digest(started(size=8))), built)
    assert stale.startswith("rank 1's spare was built from other start-up calls")
    # an offer that lost its digest is refused: every core that offers made its calls
    assert "the core's none" in spare_main.stale_for(offer(""), built)


class Asks:
    """The head's link as the server reads it."""

    def __init__(self):
        self.replay_asks = queue.Queue()


def serving(monkeypatch, cap=64):
    """A server whose queues are stood in for: each answer holds until its end is set."""
    answered = []

    def answer(self, ask, calls):
        end = threading.Event()
        answered.append((ask.asker.node, ask.rank, end))
        end.wait(10)
        with self._lock:
            self._open -= 1

    monkeypatch.setattr(replay.Server, "_answer", answer)
    r = replay.Recorder()
    r.close()
    asks, stop = Asks(), threading.Event()
    return replay.Server(asks, r, None, stop, cap=cap), asks, stop, answered


def ask(node, rank):
    from everserve.contract.signals import ProcessRef, ReplayAsk

    return ReplayAsk("k", rank, ProcessRef(node, 1, 2))


def until(ok, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not ok():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_a_core_holds_at_most_its_cap_of_queues_and_refuses_past_it(monkeypatch, caplog):
    server, asks, stop, answered = serving(monkeypatch, cap=2)
    try:
        for node in ("gpu-1", "gpu-2", "gpu-3"):
            asks.replay_asks.put(ask(node, 0))
        until(lambda: len(answered) == 2)
        until(lambda: "refused: 2 queues open, the cap" in caplog.text)
        assert server.held() == 2 and [a[0] for a in answered] == ["gpu-1", "gpu-2"]
        answered[0][2].set()  # one closes: the next ask is served
        until(lambda: server.held() == 1)
        asks.replay_asks.put(ask("gpu-3", 0))
        until(lambda: len(answered) == 3)
    finally:
        stop.set()
        for *_, end in answered:
            end.set()
