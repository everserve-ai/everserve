# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare's start-up replay: what the engine core sent its workers at its start (0044).

Sent again to a spare that stands for one of their ranks. vLLM's engine core configures its
workers' KV caches with four broadcasts, in this order (``EngineCore._initialize_kv_caches``):
the layout it resolved, the memory profiling, the model length auto-fit settled on (only when it
changed one) and every rank's KV cache configuration.
A worker built outside the engine's world, a spare, receives none of them. Before this module the
spare derived each from its own worker, which agrees with the core only where every pipeline
stage holds the same kind of cache, and misses a field the core sets after building (the
layout). Now the core records the broadcasts as it sends them (:class:`Recorder`), and a spare
asks for them through its daemon and the coordinator
(``ReplayAsk``): the core makes a queue for that spare alone, vLLM's own ``MessageQueue`` with
one remote reader, names it in its ``ReplayOffer`` and writes the calls on it
(:class:`Server`). The spare reads them (:func:`fetch`) and runs them in the core's order on its
worker, as vLLM's worker process runs a broadcast (:func:`apply`), before it warms.

The calls are vLLM's objects as the core sent them, pickled by vLLM's queue as every broadcast
is: the trust model is vLLM's multi-node one (the README's requirements), a trusted network
between an engine's nodes. Only the queue's handle crosses the control plane, as declared JSON.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

METHODS = (
    "set_kv_cache_layout",
    "determine_available_memory",
    "update_max_model_len",
    "initialize_from_config",
)
"""The broadcasts a replay carries, in the order vLLM's engine core sends them; the order a
replay is checked against."""

TAG = "everserve-vllm-replay"
VERSION = 1

READER_S = 30.0
"""How long the core waits for an offer's reader to subscribe: the spare that asked reads the
queue as soon as the offer reaches it."""
HOLD_S = 60.0
"""The bound on a queue after the calls are written on it. It is closed when its reader leaves,
which vLLM's queue reports as the reader's unsubscription: the reader closes once it has read."""
CAP = 64
"""The queues a core holds open at once; an ask past them is refused, and asked again."""
ASK_S = 30.0
"""One ask's wait for its offer; past it the spare asks again."""
FETCH_S = 1800.0
"""A spare's whole wait for its rank's calls, every ask included: a core starting beside it
answers once its workers' KV caches are configured, tens of minutes in for a large model
compiled cold."""


class ReplayRefused(RuntimeError):  # noqa: N818 - the spare's word for a replay it cannot run
    """The calls cannot be run on this spare, and why.

    A malformed replay, another rank's worker, or a KV cache that does not fit.
    """


Call = tuple[str, tuple, dict]
"""One broadcast: vLLM's method name, its arguments and keyword arguments."""


@dataclass(frozen=True)
class Calls:
    """What one core sent its workers at its start, in the order it sent them."""

    calls: tuple[Call, ...]


class Recorder:
    """The start-up broadcasts of this core, recorded as it sends them; complete once closed.

    The executor records every broadcast of :data:`METHODS` it sends, and closes the record when
    it warms its workers: the calls are then fixed for the core's life.
    """

    def __init__(self) -> None:
        """Nothing recorded yet."""
        self._calls: list[Call] = []
        self._done = threading.Event()
        self._lock = threading.Lock()

    def record(self, method: str, args: Sequence[Any] = (), kwargs: dict | None = None) -> None:
        """Record one broadcast of ``method``.

        Any other method, or one after the close, is not a start-up call and is passed over.
        """
        if method not in METHODS or self._done.is_set():
            return
        with self._lock:
            self._calls.append((method, tuple(args), dict(kwargs or {})))

    def close(self) -> None:
        """The core warms its workers: its start-up calls are all sent."""
        with self._lock:
            self._done.set()

    def complete(self) -> Calls:
        """The record, closed.

        Raises:
            RuntimeError: before the core warms its workers: the calls are not all sent.
        """
        if not self._done.is_set():
            raise RuntimeError("the core's start-up calls are not all sent: it has not warmed")
        with self._lock:
            return Calls(tuple(self._calls))

    def wait(self, stop: threading.Event) -> Calls | None:
        """The record once closed; None if ``stop`` is set first."""
        while not self._done.wait(0.5):
            if stop.is_set():
                return None
        return self.complete()


def canonical(value: Any) -> Any:
    """``value`` as JSON's data, the same for equal values in any process.

    Dataclasses by field, mappings by key, sets sorted: a pickle's bytes follow a set's order,
    which follows the process's hash seed, so they are no digest's input. Any other object is its
    text (a dtype, an enum's member).
    """
    import enum

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "": type(value).__name__,
            **{f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, dict):
        return [[canonical(k), canonical(v)] for k, v in sorted(value.items(), key=repr)]
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    return str(value)


def digest(calls: Calls) -> str:
    """The calls' digest: equal calls give the same one in every process, whatever its core."""
    import hashlib

    text = json.dumps(canonical(calls.calls), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def encode_handle(handle: Any) -> str:
    """The text of ``ReplayOffer.handle``: the queue's handle as declared JSON, tagged."""
    from everserve_vllm import join

    return json.dumps({"tag": TAG, "version": VERSION, "queue": join.handle_out(handle)})


def decode_handle(text: str) -> Any:
    """The queue handle of vLLM in :func:`encode_handle`'s text.

    Raises:
        ReplayRefused: for another tag, version or shape, named.
    """
    from everserve_vllm import join

    try:
        body = json.loads(text)
    except json.JSONDecodeError as e:
        raise ReplayRefused(f"a replay offer's handle is not JSON: {e}") from None
    if not isinstance(body, dict) or set(body) != {"tag", "version", "queue"}:
        raise ReplayRefused(f"a replay offer's handle is not one: {text[:80]!r}")
    if body["tag"] != TAG or body["version"] != VERSION:
        raise ReplayRefused(
            f"a replay offer's handle is {body['tag']!r} version {body['version']!r}; this build "
            f"reads {TAG!r} version {VERSION}"
        )
    try:
        return join.handle_in("replay", "queue", body["queue"])
    except join.JoinFormatError as e:
        raise ReplayRefused(str(e)) from None


def check(calls: Sequence[Any]) -> tuple[Call, ...]:
    """``calls`` as read from a queue, checked to be start-up calls in the core's order.

    Raises:
        ReplayRefused: naming the first call that is not one, or is out of order.
    """
    out: list[Call] = []
    last = -1
    if not isinstance(calls, (tuple, list)):
        raise ReplayRefused(f"a replay is a sequence of calls, not {type(calls).__name__}")
    for i, call in enumerate(calls):
        if not (
            isinstance(call, (tuple, list))
            and len(call) == 3
            and isinstance(call[0], str)
            and isinstance(call[1], (tuple, list))
            and isinstance(call[2], dict)
        ):
            raise ReplayRefused(f"a replay's call {i} is not a method, arguments and keywords")
        method = call[0]
        if method not in METHODS:
            raise ReplayRefused(f"a replay's call {i} is {method!r}, not a start-up call")
        at = METHODS.index(method)
        if at <= last:
            raise ReplayRefused(
                f"a replay's call {i} is {method!r} after {METHODS[last]!r}: not the core's order"
            )
        last = at
        out.append((method, tuple(call[1]), dict(call[2])))
    if not out or out[-1][0] != "initialize_from_config":
        raise ReplayRefused("a replay ends with the KV cache configuration, and this one does not")
    return tuple(out)


def kv_bytes(config: Any) -> int:
    """The device bytes a rank's KV cache configuration allocates.

    Every tensor of the configuration places its layers in one backing allocation of its
    ``size`` (vLLM's ``allocate_kv_cache``, which asserts the sizes are one): the cache groups
    overlay each other in it. A tensor held in host memory takes none of the device's.
    """
    sizes = [int(t.size) for t in config.kv_cache_tensors if not getattr(t, "host_resident", False)]
    return max(sizes, default=0)


def free_bytes() -> int:
    """This process's device's free memory, as CUDA reports it."""
    import torch

    return int(torch.cuda.mem_get_info()[0])


def apply(
    wrapper: Any,
    rank: int,
    calls: Calls,
    parts: dict[str, Any] | None = None,
    free: Callable[[], int] | None = None,
) -> None:
    """Run ``calls`` on ``wrapper``, vLLM's worker wrapper built as ``rank``, in their order.

    As vLLM's worker process runs a broadcast: the method of the wrapper by name, which takes
    its own rank's configuration from the list. Before the configuration is allocated, its
    bytes are checked against the device's free memory (``free``, else CUDA's), so a spare whose
    GPU cannot hold the rank's cache is refused by name rather than failing inside vLLM's
    allocation. Each call's seconds land in ``parts``.

    Raises:
        ReplayRefused: for a wrapper built as another rank, a configuration list without the
            rank, or a cache that does not fit.
    """
    built_as = getattr(wrapper, "global_rank", rank)
    if built_as != rank:
        raise ReplayRefused(f"a replay for rank {rank} on a worker built as rank {built_as}")
    for method, args, kwargs in check(calls.calls):
        if method == "initialize_from_config":
            configs = args[0]
            if rank >= len(configs):
                raise ReplayRefused(
                    f"the core's KV cache configurations name {len(configs)} ranks, not {rank}"
                )
            needed, have = kv_bytes(configs[rank]), (free or free_bytes)()
            if needed > have:
                raise ReplayRefused(
                    f"rank {rank}'s KV configuration needs {needed} bytes; this GPU has {have} free"
                )
        t = time.time()
        getattr(wrapper, method)(*args, **kwargs)
        if parts is not None:
            parts[f"{method}_s"] = round(time.time() - t, 3)


def fetch(offer: Any, timeout_s: float = READER_S) -> Calls:
    """Read the calls an offer's queue carries; the queue is this spare's alone.

    Raises:
        ReplayRefused: for a handle or calls this build cannot read.
        TimeoutError: if the queue is not ready, or carries nothing, within ``timeout_s``.
    """
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    handle = decode_handle(offer.handle)
    deadline = time.monotonic() + timeout_s
    mq = MessageQueue.create_from_handle(handle, 0)
    try:
        # the writer's handshake, polled so a writer gone leaves no wait behind
        if not _readable(mq, deadline):
            raise TimeoutError(f"rank {offer.rank}'s replay queue not ready in {timeout_s} s")
        mq.wait_until_ready()
        got = mq.dequeue(timeout=max(0.0, deadline - time.monotonic()))
    finally:
        _close(mq)  # read: the writer hears this reader leave and closes its end
    if not (isinstance(got, (tuple, list)) and len(got) == 2 and isinstance(got[1], str)):
        raise ReplayRefused("a replay is the calls and their digest, and this one is not")
    out = Calls(check(got[0]))
    if digest(out) != got[1]:
        raise ReplayRefused(f"rank {offer.rank}'s replay does not match the digest it came with")
    return out


def ask(asking: Callable[[float], Any], rank: int, timeout_s: float = FETCH_S) -> Calls:
    """Ask for ``rank``'s calls and read them, asking again past each bound, within ``timeout_s``.

    ``asking(bound)`` sends one ask and returns its offer (a link's ``replay``, or
    ``daemon.ask_replay`` on a connection of its own), raising ``TimeoutError`` past ``bound``.
    An offer whose queue is gone, its core ended or its hold over, is asked for again: the
    current core answers the next ask.

    Raises:
        TimeoutError: no core answered with the calls within ``timeout_s``; the last reason named.
        ReplayRefused: for calls this build cannot read.
    """
    deadline = time.monotonic() + timeout_s
    last = "no ask made"
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"rank {rank}'s replay not read in {timeout_s} s: {last}")
        try:
            offer = asking(min(ASK_S, left))
            return fetch(offer, min(READER_S, max(1.0, deadline - time.monotonic())))
        except TimeoutError as e:
            last = str(e)
            log.info("rank %d's replay: %s; asking again", rank, e)


class Server:
    """The core's half: each spare's ask answered on a queue of its own.

    Once the record is complete: an ask can come as soon as the core holds the head, before its
    workers are configured. At most ``cap`` queues are open at once; an ask past them is refused
    on the core, named in its log, and the spare's bounded wait asks again. A queue is closed once
    its reader leaves, or ``hold_s`` after the write.
    """

    def __init__(
        self,
        client: Any,
        recorder: Recorder,
        host_ip: str | None,
        stop: threading.Event,
        reader_s: float = READER_S,
        hold_s: float = HOLD_S,
        cap: int = CAP,
    ) -> None:
        """Serve ``client``'s asks from ``recorder`` until ``stop``, on a thread of its own."""
        self._client = client
        self._recorder = recorder
        self._host_ip = host_ip
        self._stop = stop
        self._reader_s = reader_s
        self._hold_s = hold_s
        self._cap = cap
        self._lock = threading.Lock()
        self._open = 0
        self.thread = threading.Thread(target=self._serve, daemon=True, name="EverserveReplay")
        self.thread.start()

    def held(self) -> int:
        """How many queues are open now."""
        with self._lock:
            return self._open

    def _serve(self) -> None:
        import queue

        calls = None
        while not self._stop.is_set():
            try:
                ask = self._client.replay_asks.get(timeout=0.5)
            except queue.Empty:
                continue
            if calls is None:
                calls = self._recorder.wait(self._stop)
                if calls is None:
                    return
            with self._lock:
                if self._open >= self._cap:
                    log.warning(
                        "rank %d's replay for %s refused: %d queues open, the cap",
                        ask.rank,
                        ask.asker,
                        self._cap,
                    )
                    continue
                self._open += 1
            threading.Thread(
                target=self._answer,
                args=(ask, calls),
                daemon=True,
                name=f"EverserveReplay-{ask.rank}",
            ).start()

    def _answer(self, ask: Any, calls: Calls) -> None:
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        mq = None
        try:
            mq = MessageQueue(1, 0, connect_ip=self._host_ip)
            self._client.offer_replay(ask, encode_handle(mq.export_handle()))
            # the reader's subscription, then vLLM's handshake, which then waits on nothing
            if not _readable(mq, time.monotonic() + self._reader_s, self._stop):
                log.warning(
                    "rank %d's replay: its reader %s never subscribed in %.0f s; closed",
                    ask.rank,
                    ask.asker,
                    self._reader_s,
                )
                return
            mq.wait_until_ready()
            mq.enqueue((calls.calls, digest(calls)))
            log.info(
                "rank %d's replay: %d calls written for %s", ask.rank, len(calls.calls), ask.asker
            )
            # the reader's unsubscription when it closes or ends: it has read, the queue is done
            _readable(mq, time.monotonic() + self._hold_s, self._stop)
        except Exception:  # noqa: BLE001 - one spare's replay; the core serves on
            log.exception("rank %d's replay for %s failed", ask.rank, ask.asker)
        finally:
            if mq is not None:
                _close(mq)
            with self._lock:
                self._open -= 1


def _readable(mq: Any, deadline: float, *ends: threading.Event) -> bool:
    """Whether the queue's remote socket has a message before ``deadline`` or any of ``ends``.

    Polled, so a wait is never left blocked inside vLLM's ``recv`` past its bound.
    """
    import zmq

    poller = zmq.Poller()
    poller.register(mq.remote_socket, zmq.POLLIN)
    while time.monotonic() < deadline and not any(e.is_set() for e in ends):
        left = min(0.2, max(0.0, deadline - time.monotonic()))
        if poller.poll(int(left * 1000)):
            return True
    return False


def _close(mq: Any) -> None:
    """Close the queue now, nothing kept to be sent (:func:`everserve_vllm.queues.close_queue`)."""
    from everserve_vllm.queues import close_queue

    close_queue(mq)
