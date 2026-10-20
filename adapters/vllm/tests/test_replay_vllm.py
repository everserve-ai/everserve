# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare's start-up replay held to vLLM at the pin, and carried over vLLM's own queue. Needs
vLLM.

The replay is what vLLM's engine core sends its workers before it warms them
(``EngineCore._initialize_kv_caches``). A call vLLM adds there, or moves, fails here rather than
reaching a spare as a worker built otherwise than the engine's (a field the core sets after
building, which a spare's own derivation misses). The record is made from the executor's
broadcasts by name, so each start-up method of
vLLM's executor must broadcast under its own name; a worker takes its rank's configuration from
the list; and a configuration's device bytes are one backing allocation.
"""

import dataclasses
import inspect
import queue
import re
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from everserve_vllm import replay  # noqa: E402 - after the skip: needs vLLM

QUESTIONS = {"get_kv_cache_specs", "get_supported_kv_cache_layouts"}
"""What the engine core asks its executor at its start: answers read, nothing set on a worker."""


def test_the_engine_core_s_start_up_calls_are_the_replay_s_in_its_order():
    from vllm.v1.engine.core import EngineCore

    source = inspect.getsource(EngineCore._initialize_kv_caches)
    sent = re.findall(
        r"self\.(?:model_executor\.(\w+)\(|collective_rpc\(\s*\"(\w+)\")", source
    )  # fmt: skip
    calls = [a or b for a, b in sent]
    assert "compile_or_warm_up_model" in calls, calls
    before_warm = calls[: calls.index("compile_or_warm_up_model")]
    assert [c for c in before_warm if c not in QUESTIONS] == list(replay.METHODS), (
        f"the engine core's start-up calls are now {before_warm}: a spare's replay "
        "(everserve_vllm.replay.METHODS) must carry each one that sets a worker's state"
    )


def test_each_start_up_method_of_vllm_s_executor_broadcasts_under_its_own_name():
    """The executor records a start-up call by the broadcast's name (``collective_rpc``)."""
    from vllm.v1.executor.abstract import Executor

    for method in ("set_kv_cache_layout", "determine_available_memory", "initialize_from_config"):
        source = inspect.getsource(getattr(Executor, method))
        assert re.search(rf"collective_rpc\(\s*\"{method}\"", source), method


def test_a_worker_takes_its_own_rank_s_configuration_from_the_core_s_list():
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    source = inspect.getsource(WorkerWrapperBase.initialize_from_config)
    assert "kv_cache_configs[self.global_rank]" in source
    for method in replay.METHODS:
        assert callable(getattr(Worker, method, None)), method


def test_a_configuration_s_tensors_share_one_backing_allocation():
    """What ``replay.kv_bytes`` counts once, as vLLM allocates it."""
    from vllm.v1.kv_cache_interface import KVCacheTensor
    from vllm.v1.worker import utils

    assert "must share one backing allocation" in inspect.getsource(utils.allocate_kv_cache)
    assert {"size", "host_resident"} <= {f.name for f in dataclasses.fields(KVCacheTensor)}


class Client:
    """The head's link as the replay's server uses it: asks in, offers out."""

    def __init__(self):
        self.replay_asks = queue.Queue()
        self.offers = queue.Queue()

    def offer_replay(self, ask, handle):
        self.offers.put(SimpleNamespace(rank=ask.rank, handle=handle))


def configs():
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

    tensor = KVCacheTensor(size=1 << 20, layers=["model.layers.0.attn"], layer_stride=1 << 19,
                           block_stride=1 << 10)  # fmt: skip
    return [
        KVCacheConfig(
            num_blocks=1024, kv_cache_tensors=[tensor], kv_cache_groups=[], kv_cache_layout="HND"
        )  # fmt: skip
    ] * 2


def recorded(closed=True):
    recorder = replay.Recorder()
    recorder.record("set_kv_cache_layout", ("HND",))
    recorder.record("determine_available_memory")
    recorder.record("initialize_from_config", (configs(),))
    if closed:
        recorder.close()
    return recorder


def asked(node="gpu-2", rank=1):
    from everserve.contract.signals import ProcessRef, ReplayAsk

    return ReplayAsk("k", rank, ProcessRef(node, 7, 31))


def until(ok, timeout_s):
    deadline = time.monotonic() + timeout_s
    while not ok():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_a_spare_reads_the_core_s_calls_from_a_queue_made_for_it_alone():
    """The core's configuration crosses vLLM's queue whole, a field the core set after building
    included, and the spare reads it as the core recorded it; once read, the queue is closed."""
    recorder = recorded(closed=False)
    client, stop = Client(), threading.Event()
    server = replay.Server(client, recorder, "127.0.0.1", stop, reader_s=30, hold_s=60)
    client.replay_asks.put(asked())
    time.sleep(0.5)
    assert client.offers.empty()  # held until the core has sent every start-up call
    recorder.close()
    offer = client.offers.get(timeout=10)
    got = replay.fetch(offer, timeout_s=30)
    # the reader closed once it read: its queue is closed on that, not at the 60 s bound
    assert until(lambda: server.held() == 0, 10.0)
    stop.set()
    server.thread.join(5)
    assert replay.digest(got) == replay.digest(recorder.complete())
    assert [c[0] for c in got.calls] == [
        "set_kv_cache_layout",
        "determine_available_memory",
        "initialize_from_config",
    ]
    assert got.calls[-1][1][0] == configs()  # by value: the dataclass compares its fields
    assert got.calls[-1][1][0][1].kv_cache_layout == "HND"


def test_an_offer_s_handle_is_declared_json_and_another_build_s_is_refused_by_name():
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    handle = MessageQueue(1, 0, connect_ip="127.0.0.1").export_handle()
    assert replay.decode_handle(replay.encode_handle(handle)) == handle
    other = replay.encode_handle(handle).replace(f'"version": {replay.VERSION}', '"version": 99')
    with pytest.raises(replay.ReplayRefused, match="version 99"):
        replay.decode_handle(other)


READ_AND_EXIT = """
import os, sys
from everserve_vllm import replay
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
mq = MessageQueue.create_from_handle(replay.decode_handle(sys.argv[1]), 0)
mq.wait_until_ready()
mq.dequeue(timeout=30)
os._exit(0)  # gone without closing anything, as a process killed after its read
"""


def test_a_queue_is_closed_when_its_reader_s_process_ends_after_reading():
    """vLLM's queue reports a remote reader's leaving to its writer (an unsubscription on its
    XPUB socket, XPUB_VERBOSE set), even from a process that ends without closing."""
    import subprocess
    import sys

    client, stop = Client(), threading.Event()
    server = replay.Server(client, recorded(), "127.0.0.1", stop, reader_s=30, hold_s=60)
    try:
        client.replay_asks.put(asked())
        offer = client.offers.get(timeout=10)
        reader = subprocess.run(
            [sys.executable, "-c", READ_AND_EXIT, offer.handle], timeout=60, check=False
        )
        assert reader.returncode == 0
        assert until(lambda: server.held() == 0, 10.0)
    finally:
        stop.set()


def test_the_cap_refuses_an_ask_past_it_and_serves_once_a_queue_closes():
    client, stop = Client(), threading.Event()
    server = replay.Server(client, recorded(), "127.0.0.1", stop, reader_s=30, hold_s=60, cap=2)
    try:
        for node in ("gpu-1", "gpu-2", "gpu-3"):  # no reader: each queue is held
            client.replay_asks.put(asked(node))
        first = [client.offers.get(timeout=10) for _ in range(2)]
        time.sleep(1.0)
        assert client.offers.empty() and server.held() == 2  # gpu-3 is past the cap
        replay.fetch(first[0], timeout_s=30)  # one is read and closes
        assert until(lambda: server.held() == 1, 10.0)
        client.replay_asks.put(asked("gpu-3"))  # asked again: served now
        again = client.offers.get(timeout=10)
        assert again.handle not in {o.handle for o in first}
        got = replay.fetch(again, timeout_s=30)
        assert [c[0] for c in got.calls][0] == "set_kv_cache_layout"
    finally:
        stop.set()
