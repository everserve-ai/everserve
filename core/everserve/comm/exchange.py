# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Named tensors between two members of a group, over a communicator with send and receive.

What a pipeline-parallel engine passes between its stages: a few named tensors (hidden states,
a residual) whose shapes the receiver does not know in advance. The metadata travels first, as
one fixed-size device tensor (the count, each tensor's dtype and shape) and the keys as bytes,
then the tensors in order; the receiver allocates each from the metadata and receives into it.
Nothing but the communicator's ``send`` and ``recv`` is used, so the exchange runs on whichever
group the fabric family holds now (:class:`~everserve.comm.collective.GroupFamily`): a change
swaps the communicator and the protocol stays. An engine maps its own intermediate-tensor
object onto :func:`send_named` and :func:`recv_named` at both ends.

An :class:`Exchange` keeps a process's transfers: one is in flight per communicator unless
its owner set a deeper gate, and a call waits for the communicator's previous transfer to
complete on the device before it enqueues its own. A peer that dies leaves the
communicator's transfers unfinished, and a loss then aborts the communicator beside the step
(decision 0025); a thread that has enqueued past what the device has completed is inside NCCL's
host-side queue when that abort runs, and the abort frees the queue under it (a survivor's
``ncclSend`` into a freed communicator takes the process down). Waiting on the previous transfer's
event
keeps the step thread in a device wait, which the abort releases cleanly, and the refusal is
read at the next call.
"""

from __future__ import annotations

import collections
import contextlib
import time
import weakref
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Protocol

import torch

from everserve.comm.guard import guard_of

MAX_TENSORS = 8
MAX_DIMS = 6
HEADER_WORDS = 2 + MAX_TENSORS * (2 + MAX_DIMS)
"""The count and the keys' byte length, then per tensor its dtype code, rank and dims."""

DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)


class TensorComm(Protocol):
    """A communicator over a group: point-to-point by rank within the group."""

    def send(self, tensor: torch.Tensor, dst: int) -> None: ...  # noqa: D102 - the protocol's

    def recv(self, tensor: torch.Tensor, src: int) -> None: ...  # noqa: D102 - the protocol's


class ExchangeError(ValueError):
    """Tensors the protocol cannot carry: too many, too deep, a dtype it does not name."""


_POLL_SPINS = 200
"""Polls of a pending event before each poll yields the core for a moment."""


def _wait(event: Any, check: Callable[[], None] | None) -> None:
    """Wait for ``event`` on the host: a device wait without a check, else a poll against it.

    The poll asks the event (``query``) and the check in turn, so a refusal ends the wait as
    soon as it is read, while a completed transfer ends it on the next poll; after
    :data:`_POLL_SPINS` polls each poll yields the core briefly. An event without ``query``
    (a stand-in) is waited on as it is.

    Raises:
        whatever ``check`` raises: the group was refused while the transfer was pending.
    """
    query = getattr(event, "query", None)
    if check is None or query is None:
        event.synchronize()
        return
    spins = 0
    while not query():
        check()
        spins += 1
        if spins > _POLL_SPINS:
            time.sleep(50e-6)


class Exchange:
    """A process's named-tensor transfers, their depth in flight, and the events behind them.

    One instance per process, owned by the adapter that drives the pipeline; the adapter sets
    ``in_flight`` to the engine's pipeline stage count at bootstrap, and nothing here reads the
    environment. One keeps the host out of NCCL's queue entirely; a small depth lets the next
    step's send and this step's broadcast overlap while the host still waits only in a device
    wait, within a few transfers of the device.

    ``events`` makes the event recorded behind each transfer: the device's by default, where
    the device is CUDA, and nothing elsewhere (a CPU device runs the body as it is); a test
    passes its own and the transfers are gated on a CPU as they are on a device.
    """

    def __init__(self, in_flight: int = 1, events: Callable[[], Any] | None = None) -> None:
        """An exchange of ``in_flight`` transfers per communicator, events made by ``events``."""
        self.in_flight = max(1, int(in_flight))
        self.events = events
        self.stats: dict[str, Any] = {
            "sends": 0,
            "recvs": 0,
            "transfers": 0,
            "first_transfer_s": None,
        }
        """Operation counts, and the seconds a communicator's first transfer took to complete,
        from its enqueue to the first wait on its event: a fresh communicator (a new ring after
        a swap) connects its transports lazily at that transfer, and formed groups are warmed at
        formation against it (``everserve_vllm.groups.warm_group``); this figure shows the term
        gone."""
        self._pending: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
        """Each communicator's transfers in flight: the events recorded on the stream behind
        them."""
        self._first: weakref.WeakKeyDictionary[Any, float] = weakref.WeakKeyDictionary()
        """Each communicator's first enqueue time, until its first transfer has been waited on."""

    def _event_for(self, device: Any) -> Callable[[], Any] | None:
        """How to make an event for a transfer on ``device``, or None where none can be."""
        if self.events is not None:
            return self.events
        return torch.cuda.Event if torch.device(device).type == "cuda" else None

    @contextlib.contextmanager
    def one_in_flight(
        self, comm: Any, device: Any, check: Callable[[], None] | None = None
    ) -> Iterator[None]:
        """Enqueue one transfer on ``comm``, after its previous transfer completed on the device.

        Waits (on the host, in a device wait) for the event behind the communicator's previous
        transfer, calls ``check``, runs the body, then records the event behind this one. On a
        device without events (CPU, tests without their own) the body runs as it is.

        ``check`` is read during and after the wait: the wait ends when the previous transfer
        completes, or when the group is refused because its peer is gone and the transfer never
        would. With a check, the wait polls the event against it rather than blocking on the
        device: a blocking wait on a dead peer's transfer ends only when the loss's abort frees
        the communicator, a quarter second or more after the refusal, and a survivor reaches its
        boundary that late. A check that raises when the group was refused keeps the enqueue
        off it and ends the step at the refusal; the abort frees the communicator behind it.

        The check and the body run under the communicator's shared guard
        (:mod:`everserve.comm.guard`): a release frees the communicator only once every holder
        has left, so a call that read "not refused" under the guard has it alive until it
        leaves. The wait stays outside the guard: an abort is what ends a wait on a dead peer's
        transfer, and the abort's exclusive side must never wait for it.
        """
        make = self._event_for(device)
        if make is None:
            yield
            return
        pending = self._pending.get(comm)
        if pending is None:
            pending = self._pending[comm] = collections.deque()
            self._first[comm] = time.perf_counter()
        while len(pending) >= self.in_flight:
            _wait(pending[0], check)
            pending.popleft()
            first = self._first.pop(comm, None)
            if first is not None:
                self.stats["first_transfer_s"] = round(time.perf_counter() - first, 4)
        with guard_of(comm).shared():
            if check is not None:
                check()
            yield
            event = make()
            event.record()
            pending.append(event)

    def send_named(
        self,
        comm: TensorComm,
        dst: int,
        tensors: Mapping[str, torch.Tensor],
        device: Any,
        check: Callable[[], None] | None = None,
    ) -> None:
        """Send ``tensors`` to ``dst``: the header, the keys, then each tensor contiguous.

        ``device`` is the communicator's device, where the header and the keys are staged;
        ``check`` is :meth:`one_in_flight`'s.

        Raises:
            ExchangeError: see :func:`encode_header`.
        """
        words, keys = encode_header(tensors)
        with self.one_in_flight(comm, device, check):
            self.stats["transfers"] += 1
            comm.send(torch.tensor(words, dtype=torch.int64, device=device), dst)
            self.stats["sends"] += 1
            if keys:
                comm.send(torch.frombuffer(bytearray(keys), dtype=torch.uint8).to(device), dst)
                self.stats["sends"] += 1
            for t in tensors.values():
                if t.numel():
                    comm.send(t.contiguous(), dst)
                    self.stats["sends"] += 1

    def recv_named(
        self, comm: TensorComm, src: int, device: Any, check: Callable[[], None] | None = None
    ) -> dict[str, torch.Tensor]:
        """Receive what :meth:`send_named` sent from ``src``: the tensors by key, on ``device``.

        ``check`` is :meth:`one_in_flight`'s.
        """
        header = torch.empty(HEADER_WORDS, dtype=torch.int64, device=device)
        with self.one_in_flight(comm, device, check):
            self.stats["transfers"] += 1
            comm.recv(header, src)
            self.stats["recvs"] += 1
            words = header.tolist()
            keys = b""
            if words[1]:
                raw = torch.empty(words[1], dtype=torch.uint8, device=device)
                comm.recv(raw, src)
                self.stats["recvs"] += 1
                keys = bytes(raw.tolist())
            out: dict[str, torch.Tensor] = {}
            for key, dtype, shape in decode_header(words, keys):
                t = torch.empty(shape, dtype=dtype, device=device)
                if t.numel():
                    comm.recv(t, src)
                    self.stats["recvs"] += 1
                out[key] = t
        return out

    def send_packed(
        self,
        comm: TensorComm,
        dst: int,
        tensors: Mapping[str, torch.Tensor],
        device: Any,
        check: Callable[[], None] | None = None,
    ) -> None:
        """Send ``tensors`` to ``dst`` as one operation: stacked into one contiguous buffer.

        The receiver names the layout itself (:meth:`recv_packed`), so nothing describes the
        tensors on the wire: a pipeline stage knows the step's token count and the model's
        intermediate tensors before it receives. The stack is a device copy of the step's few
        megabytes; the operation is one send where the header path issues two plus one per
        tensor, each a round trip across hosts.

        Raises:
            ExchangeError: when the tensors cannot go as one buffer (:func:`packable`).
        """
        if not packable(tensors):
            raise ExchangeError("tensors of one dtype and shape pack; these do not")
        with self.one_in_flight(comm, device, check):
            self.stats["transfers"] += 1
            comm.send(torch.stack(list(tensors.values())).flatten(), dst)  # one flat buffer
            self.stats["sends"] += 1

    def recv_packed(
        self,
        comm: TensorComm,
        src: int,
        layout: Layout,
        device: Any,
        check: Callable[[], None] | None = None,
        into: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Receive what :meth:`send_packed` sent from ``src``, by ``layout``: one operation.

        ``into`` is a buffer of at least the packed size to receive into (its first elements
        are used, as a view), else one is allocated. No host synchronisation: the views are
        ordered on the stream behind the receive.

        Raises:
            ExchangeError: for a layout that does not pack.
        """
        if not layout or any(d != layout[0][1] or s != layout[0][2] for _, d, s in layout):
            raise ExchangeError("a packed layout names one dtype and one shape")
        _, dtype, shape = layout[0]
        n = len(layout) * int(torch.Size(shape).numel())
        buf = into.flatten()[:n] if into is not None else torch.empty(n, dtype=dtype, device=device)
        with self.one_in_flight(comm, device, check):
            self.stats["transfers"] += 1
            comm.recv(buf, src)
            self.stats["recvs"] += 1
        packed = buf.view(len(layout), *shape)
        return {key: packed[i] for i, (key, _, _) in enumerate(layout)}


def encode_header(tensors: Mapping[str, torch.Tensor]) -> tuple[list[int], bytes]:
    """The header words and the keys' bytes for ``tensors``.

    Raises:
        ExchangeError: for more than :data:`MAX_TENSORS` tensors, a tensor of more than
            :data:`MAX_DIMS` dimensions, a dtype outside :data:`DTYPES`, or a value that is not a
            tensor.
    """
    if len(tensors) > MAX_TENSORS:
        raise ExchangeError(f"{len(tensors)} tensors; the exchange carries up to {MAX_TENSORS}")
    keys = "\0".join(tensors).encode()
    words = [len(tensors), len(keys)]
    for key, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise ExchangeError(f"{key!r} is {type(t).__name__}, not a tensor")
        if t.dtype not in DTYPES:
            raise ExchangeError(f"{key!r} has dtype {t.dtype}, which the exchange does not name")
        if t.dim() > MAX_DIMS:
            raise ExchangeError(
                f"{key!r} has {t.dim()} dimensions; the exchange carries up to {MAX_DIMS}"
            )
        words += [DTYPES.index(t.dtype), t.dim(), *t.shape, *([0] * (MAX_DIMS - t.dim()))]
    words += [0] * (HEADER_WORDS - len(words))
    return words, keys


def decode_header(words: list[int], keys: bytes) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    """Each tensor's key, dtype and shape from the header words and the keys' bytes."""
    n = words[0]
    names = keys.decode().split("\0") if n else []
    out = []
    for i in range(n):
        at = 2 + i * (2 + MAX_DIMS)
        dtype, dims = DTYPES[words[at]], words[at + 1]
        out.append((names[i], dtype, tuple(words[at + 2 : at + 2 + dims])))
    return out


# ------------------------------------------------- one operation per transfer, the layout known
Layout = list[tuple[str, torch.dtype, tuple[int, ...]]]
"""What the receiver expects: each tensor's key, dtype and shape, in the sender's order."""


def packable(tensors: Mapping[str, torch.Tensor]) -> bool:
    """Whether ``tensors`` go as one buffer: one dtype, every shape the same, none empty."""
    ts = list(tensors.values())
    return bool(ts) and all(
        t.dtype == ts[0].dtype and t.shape == ts[0].shape and t.numel() for t in ts
    )
