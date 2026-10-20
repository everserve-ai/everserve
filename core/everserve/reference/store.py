# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A step-committed KV store that lives outside every worker process.

One server process per engine. Each worker connects and, once per step, sends the K/V rows it
wrote this step for every unit it owns. A unit's store is a dense ``[layers, 2, slots, head_dim]``
tensor indexed by the engine's slot id (``block_id * block_size + offset``), plus a mask of written
slots. A survivor that absorbs a unit reads every written slot back and writes it into its own
paged cache at the same slot ids, which is valid because the block table is shared by every rank
of one engine core.

Puts are idempotent: a re-run step overwrites the same slots with the same values, so a step that
is recomputed after a fault leaves the store unchanged.

The transport is the standard library's authenticated connection over a loopback socket. It is
the harness transport; the mirror process in ``store/`` replaces it at frontier KV rates.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from typing import Any

import torch

log = logging.getLogger(__name__)

Address = tuple[str, int]
Unit = Hashable

DEFAULT_AUTH = b"everserve-kv-store"
_BF16_WIRE = torch.int16


def _to_wire(t: torch.Tensor) -> tuple[str, tuple[int, ...], bytes]:
    """Serialise a CPU tensor as (dtype name, shape, raw bytes). bfloat16 travels as int16."""
    t = t.contiguous()
    dtype = str(t.dtype).removeprefix("torch.")
    raw = t.view(_BF16_WIRE) if t.dtype == torch.bfloat16 else t
    return dtype, tuple(t.shape), raw.numpy().tobytes()


def _from_wire(packed: tuple[str, tuple[int, ...], bytes]) -> torch.Tensor:
    dtype, shape, raw = packed
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else getattr(torch, dtype)
    if not raw:  # a unit with nothing written yet; frombuffer rejects an empty buffer
        return torch.empty(shape, dtype=torch_dtype)
    if dtype == "bfloat16":
        return (
            torch.frombuffer(bytearray(raw), dtype=_BF16_WIRE).view(torch.bfloat16).reshape(shape)
        )
    return torch.frombuffer(bytearray(raw), dtype=getattr(torch, dtype)).reshape(shape)


class _State:
    def __init__(self) -> None:
        self.units: dict[Unit, torch.Tensor] = {}
        self.mask: dict[Unit, torch.Tensor] = {}
        self.lock = threading.Lock()
        self.puts = 0
        self.bytes = 0


def _serve_connection(conn: Connection, st: _State) -> None:
    try:
        while True:
            msg = conn.recv()
            try:
                if _handle(conn, st, msg):
                    return
            except Exception as e:  # noqa: BLE001 - every failure must reach the client as a reply
                conn.send(("error", f"{type(e).__name__}: {e}"))
    except (EOFError, ConnectionResetError, BrokenPipeError):
        return


def _handle(conn: Connection, st: _State, msg: tuple[Any, ...]) -> bool:
    """Serve one request. Returns True when the connection should close."""
    op = msg[0]
    if op == "init":
        _, units, layers, slots, head_dim = msg
        with st.lock:
            for u in units:
                if u not in st.units:
                    st.units[u] = torch.zeros(layers, 2, slots, head_dim, dtype=torch.bfloat16)
                    st.mask[u] = torch.zeros(slots, dtype=torch.bool)
        conn.send(("ok",))
    elif op == "put":
        with st.lock:
            for u, (slots_w, data_w) in msg[1].items():
                s, x = _from_wire(slots_w).long(), _from_wire(data_w)
                st.units[u][:, :, s] = x
                st.mask[u][s] = True
                st.bytes += len(data_w[2])
            st.puts += 1
        conn.send(("ok",))
    elif op == "get":
        with st.lock:
            s = st.mask[msg[1]].nonzero().squeeze(-1)
            x = st.units[msg[1]][:, :, s].contiguous()
        conn.send(("data", _to_wire(s.to(torch.int64)), _to_wire(x)))
    elif op == "stats":
        with st.lock:
            conn.send(("stats", {"puts": st.puts, "bytes": st.bytes, "units": sorted(st.units)}))
    elif op == "close":
        conn.send(("ok",))
        return True
    else:
        conn.send(("error", f"unknown op {op!r}"))
    return False


def serve(address: Address, auth: bytes = DEFAULT_AUTH, ready: Connection | None = None) -> None:
    """Run the store in the calling process until it is killed. One thread per connection.

    Args:
        address: where to listen; port 0 takes a free port.
        auth: the store's authentication key.
        ready: given, the address actually bound is sent on it once the store listens.
    """
    st = _State()
    with Listener(address, authkey=auth, backlog=64) as listener:
        if ready is not None:
            ready.send(listener.address)
            ready.close()
        while True:
            conn = listener.accept()
            threading.Thread(target=_serve_connection, args=(conn, st), daemon=True).start()


@dataclass
class StoreProcess:
    """A running store server and how to reach it."""

    process: mp.process.BaseProcess
    address: Address
    auth: bytes = DEFAULT_AUTH

    def client(self) -> KVStoreClient:
        """Open a new connection to this store."""
        return KVStoreClient(self.address, self.auth)

    def stop(self, timeout_s: float = 5.0) -> None:
        """Kill the server process and wait for it."""
        if self.process.is_alive():
            self.process.kill()
        self.process.join(timeout_s)


def start_server(
    address: Address, auth: bytes = DEFAULT_AUTH, ready_timeout_s: float = 10.0
) -> StoreProcess:
    """Start the store in its own (spawned) process and wait until it listens.

    Args:
        address: where it listens. Port 0 takes a free port, named by the returned process's
            ``address``: a port chosen beforehand and released can be taken by another process
            before the store binds it.
        auth: the store's authentication key.
        ready_timeout_s: how long the store may take to listen.

    Raises:
        RuntimeError: if the store does not listen within ``ready_timeout_s``.
    """
    ctx = mp.get_context("spawn")
    bound, ready = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=serve, args=(address, auth, ready), daemon=True, name="everserve-kv-store"
    )
    p.start()
    ready.close()
    try:
        if bound.poll(ready_timeout_s):
            return StoreProcess(p, bound.recv(), auth)
    except EOFError:
        pass  # the process ended before it listened
    finally:
        bound.close()
    p.kill()
    raise RuntimeError(f"kv store did not start at {address} within {ready_timeout_s}s")


class KVStoreClient:
    """One worker's connection to the store. Thread-safe; one request in flight at a time."""

    def __init__(
        self, address: Address, auth: bytes = DEFAULT_AUTH, timeout_s: float = 60.0
    ) -> None:
        """Connect to the store at ``address``.

        Args:
            address: host and port of a running store.
            auth: the store's authentication key.
            timeout_s: how long one request may wait for its reply before it fails. A store
                that stops answering must surface as an error, never as a hang.
        """
        self._conn = Client(address, authkey=auth)
        self._lock = threading.Lock()
        self._timeout_s = timeout_s

    def _rpc(self, *msg: Any) -> Any:
        with self._lock:
            self._conn.send(msg)
            if not self._conn.poll(self._timeout_s):
                raise TimeoutError(f"kv store did not answer {msg[0]!r} within {self._timeout_s}s")
            reply = self._conn.recv()
        if reply[0] == "error":
            raise RuntimeError(reply[1])
        return reply

    def init(self, units: Iterable[Unit], layers: int, slots: int, head_dim: int) -> None:
        """Allocate storage for ``units`` if the store does not hold them yet."""
        self._rpc("init", list(units), layers, slots, head_dim)

    def put(self, per_unit: Mapping[Unit, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Write this step's rows.

        Args:
            per_unit: unit -> ``(slots, data)`` with ``slots`` an int64 CPU tensor of shape
                ``[n]`` and ``data`` a bfloat16 CPU tensor of shape ``[layers, 2, n, head_dim]``.
        """
        self._rpc(
            "put",
            {u: (_to_wire(s.to(torch.int64)), _to_wire(x)) for u, (s, x) in per_unit.items()},
        )

    def get(self, unit: Unit) -> tuple[torch.Tensor, torch.Tensor]:
        """Every written slot of ``unit``: ``(slots [n], data [layers, 2, n, head_dim])``."""
        _, s, x = self._rpc("get", unit)
        return _from_wire(s), _from_wire(x)

    def stats(self) -> dict[str, Any]:
        """Put count, bytes received and the units held."""
        return self._rpc("stats")[1]

    def close(self) -> None:
        """Close this connection. The store keeps its data."""
        try:
            self._rpc("close")
        finally:
            self._conn.close()
