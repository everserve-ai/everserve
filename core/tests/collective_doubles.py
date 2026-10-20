# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Test doubles for the collectives a fault reaches, with production's failure semantics on CPU.

A CPU process has no slot kernel and no NCCL, and gloo, which stands in for both, cannot be
aborted: a member blocked on a dead peer is released only by gloo's own timeout, which production
never relies on and which gloo's macOS transport does not always enforce (a survivor sending to
the killed rank stayed in a 3 s-timeout all-reduce for minutes). These doubles carry the semantics
production has instead:

- :class:`SlotDouble` implements :class:`~everserve.comm.slot.SlotCollective` as the slot kernel
  does: each process's own :meth:`~SlotDouble.abort` releases only that process, an aborted call
  returns without raising with ``out`` undefined (filled with NaN here, so a caller that used it
  could not pass a bit-identical check), and members sum in the order :meth:`set_members` gave.
- :class:`AbortableGroup` is a :class:`~everserve.comm.group.ShrinkableGroup` with an NCCL group's:
  its :meth:`~AbortableGroup.shrink` aborts the old group, as the NCCL path does, and a call blocked
  on an aborted group raises. A refused group that was swapped out rather than shrunk is released
  by the harness, which plays the adapter (decision 0025).

Both reduce over a :class:`Mesh` of Unix sockets between the test's processes. gloo still forms
the groups (formation runs among live members only); it never carries a reduction.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from everserve.comm.group import ShrinkableGroup, ShrinkMethod
from everserve.comm.slotkernel import MAXN

_FRAME = struct.Struct(">II")
_PEER = struct.Struct(">I")


class Mesh:
    """Tagged messages between the processes of one test, one Unix socket per process.

    A reader thread per incoming connection files every message under its tag and sender; a
    caller gathers a tag once every peer it waits on has sent it, or gives up when its own
    release says so. Messages of a round nobody gathers again are dropped by :meth:`forget`.
    """

    def __init__(self, directory: str | Path, me: int, connect_timeout_s: float = 30.0) -> None:
        """Listen at ``directory``/mesh-``me``.sock; peers are connected at their first send."""
        self.me = me
        self._dir = Path(directory)
        self._connect_timeout_s = connect_timeout_s
        self._cond = threading.Condition()
        self._mail: dict[str, dict[int, bytes]] = {}
        self._out: dict[int, socket.socket | None] = {}
        self._send_lock = threading.Lock()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(self._path(me)))
        self._listener.listen(16)
        threading.Thread(target=self._accept, name="mesh-accept", daemon=True).start()

    def _path(self, peer: int) -> Path:
        return self._dir / f"mesh-{peer}.sock"

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._read, args=(conn,), name="mesh-read", daemon=True).start()

    def _read(self, conn: socket.socket) -> None:
        f = conn.makefile("rb")
        head = f.read(_PEER.size)
        if len(head) < _PEER.size:
            return
        (peer,) = _PEER.unpack(head)
        while True:
            frame = f.read(_FRAME.size)
            if len(frame) < _FRAME.size:
                return  # the peer is gone
            tag_len, payload_len = _FRAME.unpack(frame)
            tag = f.read(tag_len).decode()
            payload = f.read(payload_len)
            if len(payload) < payload_len:
                return
            with self._cond:
                self._mail.setdefault(tag, {})[peer] = payload
                self._cond.notify_all()

    def _connect(self, peer: int) -> socket.socket | None:
        """A connection to ``peer``, made on first use; None once it could not be made."""
        if peer in self._out:
            return self._out[peer]
        deadline = time.monotonic() + self._connect_timeout_s
        s: socket.socket | None = None
        while s is None:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(str(self._path(peer)))
                s.sendall(_PEER.pack(self.me))
            except OSError:
                if s is not None:
                    s.close()
                s = None
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
        self._out[peer] = s
        return s

    def send(self, peers: Iterable[int], tag: str, payload: bytes) -> None:
        """Send ``payload`` under ``tag`` to every one of ``peers``. A dead peer is skipped: who
        waits on it is released by an abort, never by this call."""
        t = tag.encode()
        frame = _FRAME.pack(len(t), len(payload)) + t + payload
        with self._send_lock:
            for peer in peers:
                s = self._connect(peer)
                if s is None:
                    continue
                try:
                    s.sendall(frame)
                except OSError:
                    s.close()
                    self._out[peer] = None

    def gather(
        self, tag: str, peers: Sequence[int], released: Callable[[], bool]
    ) -> dict[int, bytes] | None:
        """Every one of ``peers``'s payload under ``tag``; None once ``released()`` holds first."""
        with self._cond:
            while True:
                box = self._mail.get(tag, {})
                if all(p in box for p in peers):
                    return self._mail.pop(tag)
                if released():
                    return None
                self._cond.wait()

    def wake(self) -> None:
        """Make every gatherer check its release again."""
        with self._cond:
            self._cond.notify_all()

    def forget(self, keep: Callable[[str], bool]) -> None:
        """Drop the messages of every tag ``keep`` rejects."""
        with self._cond:
            for tag in [t for t in self._mail if not keep(t)]:
                del self._mail[tag]


def _to_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes()


def _sum_into(
    order: Sequence[int],
    me: int,
    mine: torch.Tensor,
    theirs: dict[int, bytes],
    out: torch.Tensor,
) -> None:
    """``out = sum of the members' tensors``, added in ``order`` on every member, so every member's
    result is the same bits (decision 0012)."""
    acc = torch.zeros(mine.shape, dtype=torch.promote_types(mine.dtype, torch.float32))
    for m in order:
        if m == me:
            acc += mine
        else:
            raw = torch.frombuffer(bytearray(theirs[m]), dtype=torch.uint8)
            acc += raw.view(mine.dtype).view(mine.shape)
    out.copy_(acc.to(out.dtype))


class SlotDouble:
    """The slot kernel's semantics on CPU (implements ``SlotCollective``, variant ``cpu-double``).

    The kernel's sequence counters and its zeroing at :meth:`resume` fence a released member's
    words from the next membership's; here every round is tagged with the epoch
    :meth:`set_members` gave and its count within it, and :meth:`resume` drops what is left of
    older epochs. Abort words belong to a process; a dead process's is cleared with it.
    """

    def __init__(self, mesh: Mesh, row: int, members: Iterable[int], epoch: int = 0) -> None:
        """This process's handle, as ``row``, among ``members`` at ``epoch``."""
        self._mesh = mesh
        self.row = row
        self._members = [int(m) for m in members]
        self._epoch = epoch
        self._seq = 0
        self._aborted = False

    @property
    def variant(self) -> str:
        """The double's name."""
        return "cpu-double"

    def can_take(self, t: torch.Tensor) -> bool:
        """The kernel's predicate, on the CPU: contiguous bf16, an even count of at most MAXN."""
        return (
            t.dtype == torch.bfloat16
            and t.is_contiguous()
            and t.device.type == "cpu"
            and t.numel() <= MAXN
            and t.numel() % 2 == 0
        )

    def all_reduce_into(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """Sum ``inp`` over the members into ``out``; an aborted call leaves ``out`` undefined."""
        if self._aborted:
            out.fill_(float("nan"))
            return
        self._seq += 1
        tag = f"slot/{self._epoch}/{self._seq}"
        peers = [m for m in self._members if m != self.row]
        self._mesh.send(peers, tag, _to_bytes(inp))
        got = self._mesh.gather(tag, peers, lambda: self._aborted)
        if got is None:
            out.fill_(float("nan"))
            return
        _sum_into(self._members, self.row, inp, got, out)

    def set_members(self, members: Iterable[int], epoch: int) -> None:
        """Take up ``members``, rows in rank position, at ``epoch``; called at a boundary.

        Raises:
            ValueError: for an order with a row twice.
        """
        members = [int(m) for m in members]
        if len(set(members)) != len(members):
            raise ValueError(f"a row twice in {members}")
        self._members = members
        self._epoch = epoch

    def abort(self) -> None:
        """Release this process from a wait on a dead peer. Safe at any time."""
        self._aborted = True
        self._mesh.wake()

    def resume(self) -> None:
        """Zero the round count and drop older epochs' rounds, then clear the abort word last."""
        self._seq = 0
        epoch = self._epoch
        self._mesh.forget(lambda t: not t.startswith("slot/") or int(t.split("/")[1]) >= epoch)
        self._aborted = False


class AbortableGroup(ShrinkableGroup):
    """A group with an NCCL group's failure semantics: aborted when it is shrunk or released, and
    a call blocked on it then raises. gloo forms it; the :class:`Mesh` carries its reductions."""

    def __init__(self, base: ShrinkableGroup, mesh: Mesh, name: str) -> None:
        """``base``'s membership, reducing over ``mesh``; ``name`` is the same on every member."""
        super().__init__(
            base.group,
            base.ranks,
            base.backend,
            base.store,
            base.method,
            base.first_timeout,
            base.later_timeout,
        )
        self._mesh = mesh
        self._name = name
        self._seq = 0
        self._aborted = False

    def all_reduce(self, t: torch.Tensor) -> None:
        """Sum ``t`` in place across the group, in rank order.

        Raises:
            RuntimeError: when the group is aborted before or while the call waits.
        """
        if self._aborted:
            raise RuntimeError(f"group {self._name} was aborted")
        self._seq += 1
        tag = f"group/{self._name}/{self._seq}"
        peers = [r for r in self.ranks if r != self._mesh.me]
        self._mesh.send(peers, tag, _to_bytes(t))
        got = self._mesh.gather(tag, peers, lambda: self._aborted)
        if got is None:
            raise RuntimeError(f"group {self._name} was aborted")
        _sum_into(self.ranks, self._mesh.me, t.clone(), got, t)

    def abort(self) -> None:
        """Abort the group, as ``ncclCommAbort`` does: a call waiting on it raises."""
        self._aborted = True
        self._mesh.wake()
        prefix = f"group/{self._name}/"
        self._mesh.forget(lambda t: not t.startswith(prefix))

    def shrink(self, lost: Sequence[int], me: int, epoch: int) -> ShrinkableGroup:
        """Abort this group, as the NCCL path does, then form the survivors' (with gloo)."""
        self.abort()
        new = super().shrink(lost, me, epoch)
        return AbortableGroup(new, self._mesh, f"{new.method.value}-{epoch}")


@dataclass
class FakeGroup(ShrinkableGroup):
    """A group whose all-reduce sums as if every member contributed one copy; it can be told to
    fail its next calls, and a shrink leaves the lost members out."""

    group: object = None
    ranks: list = field(default_factory=list)
    backend: str = "fake"
    fail_times: int = 0
    calls: int = 0

    def all_reduce(self, t):
        self.calls += 1
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("peer went away")
        t.mul_(len(self.ranks))

    def shrink(self, lost, me, epoch):
        return FakeGroup(
            ranks=[r for r in self.ranks if r not in lost], method=ShrinkMethod.RECREATE
        )
