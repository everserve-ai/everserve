# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The slot collective on CUDA: the region every member shares, and one member's handle on it.

The kernel is ``slot_all_reduce`` in ``core/csrc/slot.cu`` (decisions 0005, 0006). This module
holds what it needs around it:

* :class:`Region`: a small shared file every process of one engine maps. It has a header naming
  the variant, one control row per member (abort word, member order, published flag), one CUDA IPC
  handle per member, and, for the host-memory variant, every member's words. The executor creates
  it before the workers start and writes control words into it; workers map it into their device.
* :class:`SlotKernel`: one process's :class:`~everserve.comm.slot.SlotCollective`. It owns its
  word buffer (a row of the region, or device memory shared over CUDA IPC), its sequence counters,
  its member order in device memory, and the device table of every member's buffer; the kernel
  reads the last two on every replay without leaving the device.
* :func:`choose_variant`: peer-to-peer when every GPU of the engine reaches every other, else host
  memory. It asks NVML, so the engine process never starts CUDA before vLLM starts its workers.

A change has two halves. :meth:`SlotKernel.abort` is safe at any time and releases a kernel
spinning on a dead peer. Everything else happens at a step boundary with every member quiescent:
each process zeroes its counters and its own words, clears its own abort word, and writes its own
row's order into the region and its device memory, from the change its group applies
(:meth:`SlotKernel.resume`, :meth:`SlotKernel.set_members`); the executor clears the abort words
of the rows whose process is gone. A word left from the old epoch could otherwise match a new
call's sequence number. The executor writes the initial order when it creates the region
(:meth:`Region.set_members`), before any member maps it.

The order is the members' tensor-parallel rank positions, with a spare in the position of the rank
it replaced, and the kernel sums in it (decision 0012). Floating-point addition is not associative:
summed in region-row order, a spare's partial moves to a different place in the sum, the result
rounds differently, and the tokens after a REPLACE drift from the tokens without one.

This module is not the engine's integration: the vLLM adapter decides which tensors go through
the slot collective and when a change is applied.
"""

from __future__ import annotations

import ctypes
import json
import logging
import mmap
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

MAXW = 8
"""Rows in a region: up to seven ranks and the spare."""
MAXN = 64 * 1024
"""Largest tensor the kernel takes, in bf16 elements (32 tokens x 2048 hidden)."""
HOST, P2P = 0, 1
WORDS_PER_THREAD = {"host": (16, 16), "p2p": (1, 1)}
"""Eight-byte words each thread moves, as (small calls, large calls); a call is small up to
``SMALL_ELEMENTS``. Measured with ``bench/harness/slot_bench.py`` on 4 A100 SXM with NVLink, the
member order in device memory, us per call at 2,048 / 16,384 / 32,768 elements: the
peer-to-peer variant 3.99 / 4.21 / 5.36 with 1 word per thread (5.52 / 5.20 / 6.42 with 4,
8.56 / 10.04 / 10.10 with 16); stock vLLM's custom all-reduce 5.32 / 5.81 / 6.54; NCCL 13.06 /
16.78 / 17.08. One word per thread is fastest at every decode size: the most blocks, each pushing
its words over NVLink at once. The host-memory variant is not retuned (18.73 us at 16,384)."""
SMALL_ELEMENTS = 4096
VARIANTS = {"host": HOST, "p2p": P2P}

_MAGIC = 0x45565253_4C4F5432  # "EVRSLOT2": a member order, where "EVRSLOT1" had a mask
_HEADER_WORDS = 16
_CTRL_WORDS = 16
_HANDLE_BYTES = 64
_PUBLISHED, _ABORT, _COUNT, _ORDER = 0, 1, 2, 3  # the order is rows [_ORDER, _ORDER + MAXW)
# decision 0021: a mask of the rows the kernel waits too long for (the kernel's), the wait's bound
# in ns (the executor's), the row's process id (its own), the ranks its watchdog found lost (a
# mask, with the epoch above it), and the row's steps: the count of step RPCs it has answered,
# above a bit set while it runs one (its own)
_LATE, _BOUND, _PID, _LOST, _STEP = 11, 12, 13, 14, 15
_IN_FLIGHT = 4  # header word: the executor's step RPCs not yet answered (decision 0021)
_STOP = 5  # header word: the watches of this region stop; at the node's shutdown
# header words [_RELEASED, _RELEASED + MAXW): a row's process left its position on a change and
# was released (0033 step 3); one word per row, each written only by that row's process
_RELEASED = 6
_ENDS_RELEASED = _RELEASED + MAXW  # header word: this node's executor ends released processes
assert _ENDS_RELEASED < _HEADER_WORDS
_HEADER_B = _HEADER_WORDS * 8
_CTRL_B = MAXW * _CTRL_WORDS * 8
_HANDLES_B = MAXW * _HANDLE_BYTES
_WORDS_OFF = 4096  # page aligned
_ROW_WORDS = 2 * (MAXN // 2)  # host rows: two parities of MAXN/2 eight-byte words
_ROW_B = _ROW_WORDS * 8
_PUSH_B = MAXW * _ROW_B  # a peer-to-peer buffer: two parities x eight senders x MAXN/2 words
_SIZE = _WORDS_OFF + MAXW * _ROW_B


def check_layout(ext: Any) -> None:
    """The extension's slot region layout is this module's, or no region is mapped.

    The rows, the control words and the abort, late and bound words are declared once in
    ``csrc/kernels.h`` and exported by the build; a copy here that drifted would read and write
    other words than the kernels do, silently.

    Raises:
        RuntimeError: a build whose layout differs, naming each differing word.
    """
    ours = {
        "MAX_ROWS": MAXW,
        "CTRL_WORDS": _CTRL_WORDS,
        "ABORT": _ABORT,
        "LATE": _LATE,
        "BOUND": _BOUND,
    }
    differs = {k: (getattr(ext, k, None), v) for k, v in ours.items() if getattr(ext, k, None) != v}
    if differs:
        raise RuntimeError(
            "the slot kernels were built with another region layout than this module's "
            f"(extension, module): {differs}"
        )


@dataclass(frozen=True)
class VariantChoice:
    """The slot collective's variant and how it was decided, for the run report."""

    variant: str
    how: str
    """``override`` (the operator's), ``nvml`` (no peer reads), ``cache`` or ``probe``."""
    detail: str
    seconds: float = 0.0


def choose_variant(devices: Sequence[int | str], confirm: bool = True) -> str:
    """The variant :func:`decide_variant` chooses for ``devices``, without a cache."""
    return decide_variant(devices, confirm).variant


def decide_variant(
    devices: Sequence[int | str], confirm: bool = True, cache: Path | None = None
) -> VariantChoice:
    """``p2p`` when the peer-to-peer slot collective works on ``devices``, else ``host``.

    Every pair must have peer reads by NVML and, with ``confirm``, the collective must then work
    when run (:mod:`.p2pprobe`).

    NVML is the cheap first answer: without peer reads there is nothing to try. It is not the last:
    NVML can report peer reads between PCIe GPUs whose stores a container never delivers,
    where NCCL's own peer-to-peer hangs and the peer-to-peer slot collective would too (PCIe
    A40 and L40S pods). So the variant is chosen by running it: a spawned process per GPU
    reduces a known tensor once, and anything short of the exact sum on every GPU within the
    deadline gives ``"host"``, logged with what failed.

    ``devices`` are the entries of ``CUDA_VISIBLE_DEVICES`` plus the spare's: a GPU UUID
    (``GPU-...``) names the device exactly; an integer is taken as NVML's index, which is PCI bus
    order, and matches CUDA's index only under ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (CUDA's default
    order sorts by speed first), so integer entries without that setting are logged as a warning.
    ``"host"`` otherwise, including when NVML is not installed. This process starts no CUDA
    context, so it can run before vLLM starts its workers.

    A probe costs seconds (a CUDA context per GPU). With ``cache``, a JSON file, its verdict is
    kept under the driver version and the GPUs' UUIDs and reused on a later start with the same
    GPUs and driver; a box that changed either is probed again.
    """
    peer_reads, key = _nvml_peer_reads(devices)
    if not peer_reads:
        return VariantChoice("host", "nvml", "NVML reports no peer reads between every pair")
    if not confirm:
        return VariantChoice("p2p", "nvml", "NVML reports peer reads (not run)")
    known = _cached(cache, key)
    if known is not None:
        return VariantChoice(known["variant"], "cache", known["detail"])
    from everserve.comm.p2pprobe import probe

    result = probe(devices)
    if result.ok:
        log.info(
            "peer-to-peer slot collective confirmed on %s in %.1fs", list(devices), result.seconds
        )
        choice = VariantChoice("p2p", "probe", result.detail, result.seconds)
    else:
        log.warning(
            "NVML reports peer reads between %s, but the peer-to-peer slot collective does not "
            "work on them (%s, %.1fs): the host-memory slot collective",
            list(devices),
            result.detail,
            result.seconds,
        )
        choice = VariantChoice("host", "probe", result.detail, result.seconds)
    _remember(cache, key, choice)
    return choice


def _cached(cache: Path | None, key: str | None) -> dict | None:
    if cache is None or key is None:
        return None
    try:
        return json.loads(cache.read_text()).get(key)
    except (OSError, ValueError):
        return None


def _remember(cache: Path | None, key: str | None, choice: VariantChoice) -> None:
    if cache is None or key is None:
        return
    try:
        known = json.loads(cache.read_text()) if cache.exists() else {}
        known[key] = {"variant": choice.variant, "detail": choice.detail}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1))
    except (OSError, ValueError) as e:
        log.warning("could not keep the slot variant's verdict in %s: %s", cache, e)


def _nvml_peer_reads(devices: Sequence[int | str]) -> tuple[bool, str | None]:
    """Whether NVML reports peer reads between every pair of ``devices``, and the devices' key.

    The key is the driver version and the GPUs' UUIDs, what a cached verdict is valid for; None
    when NVML cannot give them.
    """
    try:
        import pynvml
    except ImportError:
        log.info("NVML not installed: the host-memory slot collective")
        return False, None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        log.warning("NVML unavailable (%s): the host-memory slot collective", e)
        return False, None
    try:
        handles = []
        for d in devices:
            text = str(d).strip()
            if text.startswith(("GPU-", "MIG-")):
                handles.append(pynvml.nvmlDeviceGetHandleByUUID(text))
            else:
                if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                    log.warning(
                        "device %s is an index: taken as NVML's PCI order, which CUDA matches "
                        "only under CUDA_DEVICE_ORDER=PCI_BUS_ID; name GPUs by UUID to be exact",
                        text,
                    )
                handles.append(pynvml.nvmlDeviceGetHandleByIndex(int(text)))
        read = 0  # NVML_P2P_CAPS_INDEX_READ; nvidia-ml-py spells the constant as the tuple (0,)
        for i, a in enumerate(handles):
            for j, b in enumerate(handles):
                if (
                    i != j
                    and pynvml.nvmlDeviceGetP2PStatus(a, b, read) != pynvml.NVML_P2P_STATUS_OK
                ):
                    log.info("GPUs %s and %s have no peer reads", devices[i], devices[j])
                    return False, None
        return True, _device_key(pynvml, handles)
    except (pynvml.NVMLError, ValueError) as e:
        log.warning("NVML could not report peer-to-peer (%s): the host-memory slot collective", e)
        return False, None
    finally:
        pynvml.nvmlShutdown()


def _device_key(pynvml: object, handles: list) -> str | None:
    """The driver version and the GPUs' UUIDs, in the devices' order; None if NVML has neither."""
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        uuids = [pynvml.nvmlDeviceGetUUID(h) for h in handles]
    except (AttributeError, pynvml.NVMLError):
        return None
    text = lambda v: v.decode() if isinstance(v, bytes) else str(v)  # noqa: E731
    return text(driver) + "|" + ",".join(text(u) for u in uuids)


def region_path(engine: str) -> str:
    """Where this node's region for ``engine`` is: one per engine and node, by name.

    The first process of the engine on the node creates it (:meth:`Region.create`): the executor
    on a node that serves the engine, a spare on a node that only holds spares. Every process of
    the engine on the node maps it, told nothing but the engine's name. Under
    ``EVERSERVE_SHM_DIR`` (the Kubernetes pod contract: a directory of the host's tmpfs mounted
    into every pod of the node, so pods share it), else ``/dev/shm``.
    """
    import os

    from everserve.podcontract import SHM_DIR_ENV

    return f"{os.environ.get(SHM_DIR_ENV) or '/dev/shm'}/everserve_slot_{engine}"


HEADER_WAIT_S = 10.0
"""How long a process opening a region another is creating waits for the header."""


class Region:
    """The shared slot region, mapped in this process.

    The control path (the executor) uses the NumPy views; a worker also maps the file into its
    device through :meth:`map_device`. NumPy rather than torch on the host side: a torch CPU op in
    the engine process before vLLM forks its workers starts OpenMP, and the workers then crash in
    ``gomp_team_start``.
    """

    def __init__(self, path: str, create: bool = False, variant: str = "host") -> None:
        """Map the region at ``path``; with ``create``, size and zero it and record ``variant``.

        Raises:
            ValueError: for an unknown variant, or a file that is not a slot region.
        """
        self.path = path
        if create:
            # every process of the engine on the node maps it, whichever pod it runs in
            from everserve.podcontract import make_shared

            make_shared(path, _SIZE)
        fd = os.open(path, os.O_RDWR)
        try:
            self._mm = mmap.mmap(fd, _SIZE)
        finally:
            os.close(fd)
        self.header = np.frombuffer(self._mm, np.int64, _HEADER_WORDS)
        self.ctrl = np.frombuffer(self._mm, np.int64, MAXW * _CTRL_WORDS, _HEADER_B).reshape(
            MAXW, _CTRL_WORDS
        )
        self.handles = np.frombuffer(self._mm, np.uint8, _HANDLES_B, _HEADER_B + _CTRL_B).reshape(
            MAXW, _HANDLE_BYTES
        )
        self.words = np.frombuffer(self._mm, np.int64, MAXW * _ROW_WORDS, _WORDS_OFF).reshape(
            MAXW, _ROW_WORDS
        )
        if create:
            if variant not in VARIANTS:
                raise ValueError(f"unknown slot variant {variant!r}")
            self.header[:] = 0
            self.ctrl[:] = 0
            self.handles[:] = 0
            self.words[:] = 0
            self.header[1], self.header[2], self.header[3] = VARIANTS[variant], MAXW, MAXN
            self.header[0] = _MAGIC
        elif int(self.header[0]) != _MAGIC:
            raise ValueError(f"{path} is not a slot region")
        self._buf = (ctypes.c_char * _SIZE).from_buffer(self._mm)
        self.addr = ctypes.addressof(self._buf)
        self.device_addr = 0

    @classmethod
    def create(cls, path: str, variant: str, replace: bool) -> Region:
        """Create the region at ``path`` atomically, or open the one that is there.

        The file is made complete under a temporary name, then given its name in one step, so
        no process ever maps a half-made region. With ``replace`` (the executor's: it owns the
        engine's run on this node) a region left by an earlier run is replaced; a process that
        mapped the old one keeps the old file. Without it (a spare's: two spares may start at
        once on a node that only holds spares) the first creator wins and this process opens
        the existing region, waiting up to :data:`HEADER_WAIT_S` for its header.

        Raises:
            ValueError: for an unknown variant, or a file at ``path`` that is not a region.
            TimeoutError: when the existing file gets no header in time.
        """
        tmp = f"{path}.{os.getpid()}.tmp"
        made = cls(tmp, create=True, variant=variant)
        made.close()
        try:
            if replace:
                os.replace(tmp, path)
            else:
                os.link(tmp, path)
        except FileExistsError:
            deadline = time.monotonic() + HEADER_WAIT_S
            while True:
                try:
                    return cls(path)
                except ValueError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{path} exists but has no region header") from None
                    time.sleep(0.05)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        return cls(path)

    @property
    def variant(self) -> str:
        """``"p2p"`` or ``"host"``, as the region's creator chose it."""
        return "p2p" if int(self.header[1]) == P2P else "host"

    # ------------------------------------------------------------------ the control path
    def set_members(self, rows: Iterable[int], order: Sequence[int]) -> None:
        """Every row in ``rows`` sums over the rows in ``order``, in that order, from its next call.

        ``order`` lists the members' rows by tensor-parallel rank position: a spare stands in the
        position of the rank it replaced.

        Raises:
            ValueError: for a row out of range or listed twice.
        """
        order = [int(m) for m in order]
        if len(set(order)) != len(order) or not all(0 <= m < MAXW for m in order):
            raise ValueError(f"a slot member order needs distinct rows below {MAXW}: {order}")
        for r in rows:
            self.ctrl[r, _ORDER : _ORDER + MAXW] = 0
            self.ctrl[r, _ORDER : _ORDER + len(order)] = order
            self.ctrl[r, _COUNT] = len(order)

    def order(self, row: int) -> list[int]:
        """The rows ``row`` sums over, in the order it sums them."""
        n = int(self.ctrl[row, _COUNT])
        return [int(m) for m in self.ctrl[row, _ORDER : _ORDER + n]]

    def set_abort(self, rows: Iterable[int], value: int = 1) -> None:
        """Set or clear the abort word of every row in ``rows``."""
        for r in rows:
            self.ctrl[r, _ABORT] = value

    # ------------------------------------------------------------------ detection (decision 0021)
    def set_bound(self, rows: Iterable[int], bound_s: float) -> None:
        """A kernel of every row in ``rows`` names a member it waits for longer than ``bound_s``."""
        for r in rows:
            self.ctrl[r, _BOUND] = int(bound_s * 1e9)

    def bound(self, row: int) -> float:
        """``row``'s bound in seconds; 0 when the kernel names nobody."""
        return int(self.ctrl[row, _BOUND]) / 1e9

    def late(self, row: int) -> set[int]:
        """The rows ``row``'s kernel has waited for past its bound and still waits for."""
        v = int(self.ctrl[row, _LATE])
        return {r for r in range(MAXW) if v >> r & 1}

    def set_in_flight(self, n: int) -> None:
        """The executor's count of step RPCs sent and not yet answered."""
        self.header[_IN_FLIGHT] = n

    def begin_step(self, row: int) -> None:
        """``row``'s process has received a step RPC and runs it."""
        self.ctrl[row, _STEP] = int(self.ctrl[row, _STEP]) | 1

    def end_step(self, row: int) -> None:
        """``row``'s process has answered the step RPC it ran."""
        self.ctrl[row, _STEP] = ((int(self.ctrl[row, _STEP]) >> 1) + 1) << 1

    def steps(self, row: int) -> int:
        """The step RPCs ``row``'s process has answered."""
        return int(self.ctrl[row, _STEP]) >> 1

    def in_flight(self) -> bool:
        """Whether a step is in flight on this node.

        Sent by the engine's executor and not yet answered, where the engine is on this node; or
        received by a member of this region and not yet answered, on any node. A member that dies
        inside a step leaves its word set, so the step it cut short stays in flight here until the
        change clears the word (:meth:`clear_watch`).
        """
        return int(self.header[_IN_FLIGHT]) > 0 or bool((self.ctrl[:, _STEP] & 1).any())

    def set_stop(self) -> None:
        """Every watch of this region stops at its next look; at the node's shutdown."""
        self.header[_STOP] = 1

    def stopped(self) -> bool:
        """Whether the watches have been told to stop."""
        return int(self.header[_STOP]) != 0

    def set_ends_released(self) -> None:
        """This node's executor ends the processes that mark themselves released.

        Set by the node role's executor, which reads the marks; a process releases itself only
        where it is set, since on a node whose executor does not read them it would hold its GPU
        with its link already closed.
        """
        self.header[_ENDS_RELEASED] = 1

    def ends_released(self) -> bool:
        """Whether this node's executor ends released processes."""
        return int(self.header[_ENDS_RELEASED]) != 0

    def mark_released(self, row: int) -> None:
        """``row``'s process left its position on a change, drained, and may be ended.

        Written by the row's own process only, a word per row, so two processes of one node
        leaving together never write one word.
        """
        self.header[_RELEASED + row] = 1

    def released(self) -> set[int]:
        """The rows whose processes have marked themselves released."""
        return {r for r in range(MAXW) if int(self.header[_RELEASED + r])}

    def set_pid(self, row: int, pid: int) -> None:
        """Record the process that holds ``row``."""
        self.ctrl[row, _PID] = pid

    def pid(self, row: int) -> int:
        """The process that holds ``row``, or 0 if none has joined."""
        return int(self.ctrl[row, _PID])

    def mark_lost(self, row: int, positions: Iterable[int], epoch: int) -> None:
        """``row``'s watchdog found the members at ``positions`` lost at ``epoch``.

        Positions in the member order: the ranks within this region's stage. The executor reads
        them and names the engine ranks.
        """
        v = int(self.ctrl[row, _LOST])
        mask = (v & 0xFF) if v >> 8 == epoch else 0
        for r in positions:
            mask |= 1 << r
        self.ctrl[row, _LOST] = (epoch << 8) | mask

    def lost_ranks(self, epoch: int) -> set[int]:
        """Every position some row's watchdog found lost at ``epoch``.

        A mark from an earlier epoch is about a member that has been replaced since: a watchdog
        slower than the change must not get the spare in that position killed.
        """
        out: set[int] = set()
        for v in self.ctrl[:, _LOST]:
            v = int(v)
            if v & 0xFF and v >> 8 == epoch:
                out |= {r for r in range(MAXW) if v >> r & 1}
        return out

    def clear_watch(self, rows: Iterable[int]) -> None:
        """Forget the late, lost and step words of ``rows``; at a change's boundary.

        The step word of a member that died inside a step is cleared here, and with it the step
        it cut short stops being in flight.
        """
        for r in rows:
            self.ctrl[r, _LATE] = 0
            self.ctrl[r, _LOST] = 0
            self.ctrl[r, _STEP] = 0

    # ------------------------------------------------------------------ a member's side
    def map_device(self, ext: object) -> int:
        """Map the whole file into the current device; returns its device address."""
        if not self.device_addr:
            self.device_addr = ext.host_register_mapped(self.addr, _SIZE)
            self._ext = ext
        return self.device_addr

    def ctrl_device(self) -> int:
        """The device address of the control rows."""
        return self.device_addr + _HEADER_B

    def row_device(self, row: int) -> int:
        """The device address of ``row``'s words (host-memory variant)."""
        return self.device_addr + _WORDS_OFF + row * _ROW_B

    def publish(self, row: int, handle: bytes) -> None:
        """Make ``row``'s IPC handle visible to the other members (peer-to-peer variant)."""
        self.handles[row, : len(handle)] = np.frombuffer(handle, np.uint8)
        self.ctrl[row, _PUBLISHED] = 1  # after the handle: x86 keeps the stores in order

    def handle(self, row: int, timeout_s: float) -> bytes:
        """``row``'s IPC handle, waiting up to ``timeout_s`` for it to be published.

        Raises:
            TimeoutError: if the row's process has not published by then.
        """
        deadline = time.monotonic() + timeout_s
        while not int(self.ctrl[row, _PUBLISHED]):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"slot row {row} did not publish its buffer in {timeout_s}s")
            time.sleep(0.01)
        return bytes(self.handles[row])

    def close(self) -> None:
        """Drop this process's mapping (the file stays until it is removed)."""
        if self.device_addr:
            self._ext.host_unregister(self.addr)
            self.device_addr = 0
        self.header = self.ctrl = self.handles = self.words = None
        self._buf = None
        self._mm.close()


class SlotKernel:
    """One member's handle on the slot collective (implements ``SlotCollective``)."""

    def __init__(
        self, region: Region, row: int, device: int, open_timeout_s: float = 300.0
    ) -> None:
        """Join ``region`` as ``row`` on CUDA device ``device``.

        For the peer-to-peer variant this allocates the row's buffer in device memory and
        publishes its IPC handle; peers' buffers are mapped when they first become members.
        """
        from everserve import kernels

        self.region = region
        self.row = row
        self.device = torch.device("cuda", device)
        self._ext = kernels.load()
        check_layout(self._ext)
        self._timeout = open_timeout_s
        with torch.cuda.device(self.device):
            region.map_device(self._ext)
            self._words = WORDS_PER_THREAD[region.variant]
            blocks = self._ext.slot_blocks(MAXN, 1)  # counters for any words-per-thread setting
            # and after them the count of blocks that named a late member (decision 0021)
            self.seq = torch.zeros(blocks + 1, dtype=torch.int64, device=self.device)
            self._table = torch.zeros(MAXW, dtype=torch.int64, device=self.device)
            # [count, rows...]: the region's order for this row, copied at a boundary
            self._order = torch.zeros(1 + MAXW, dtype=torch.int32, device=self.device)
        self._members = region.order(row)
        region.set_pid(row, os.getpid())
        self._addrs = [0] * MAXW
        self._opened: dict[int, int] = {}
        self._own = 0
        if region.variant == "p2p":
            with torch.cuda.device(self.device):
                self._own = self._ext.device_alloc(_PUSH_B)
            region.publish(row, self._ext.ipc_handle(self._own))
            self._addrs[row] = self._own
        else:
            self._addrs = [region.row_device(r) for r in range(MAXW)]
        self._write_table()
        self._write_order()

    @property
    def variant(self) -> str:
        """``"p2p"`` or ``"host"``."""
        return self.region.variant

    def _write_table(self) -> None:
        # the table may have been made inside the engine's forward, under inference mode
        with torch.inference_mode():
            self._table.copy_(torch.tensor(self._addrs, dtype=torch.int64))

    def _write_order(self) -> None:
        rows = [len(self._members), *self._members]
        rows += [0] * (1 + MAXW - len(rows))
        with torch.inference_mode():
            self._order.copy_(torch.tensor(rows, dtype=torch.int32))

    def _open(self, rows: Iterable[int]) -> None:
        """Map every member's buffer this process does not have yet (peer-to-peer variant).

        A mapping outlives its exporter: CUDA IPC counts the importers' references, so a dead
        rank's buffer stays valid memory until every survivor closes it. The kernel therefore
        keeps pushing words into a dead member's buffer harmlessly until the order drops it at the
        boundary, which every REPLACE on the peer-to-peer variant exercises.
        """
        if self.variant != "p2p":
            return
        changed = False
        for r in rows:
            if r == self.row or r in self._opened:
                continue
            with torch.cuda.device(self.device):
                self._opened[r] = self._ext.ipc_open(self.region.handle(r, self._timeout))
            self._addrs[r] = self._opened[r]
            changed = True
        if changed:
            self._write_table()

    # ------------------------------------------------------------------ SlotCollective
    def can_take(self, t: torch.Tensor) -> bool:
        """Decode sizes: contiguous bf16 on this device, an even count of at most ``MAXN``."""
        return (
            t.dtype == torch.bfloat16
            and t.is_contiguous()
            and t.device == self.device
            and t.numel() <= MAXN
            and t.numel() % 2 == 0
        )

    def all_reduce_into(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """Sum ``inp`` over the members into ``out``; capturable in a CUDA graph.

        The first call maps the current members' buffers; it must not be the one captured.
        """
        if self.variant == "p2p" and len(self._opened) + 1 < len(self._members):
            self._open(self._members)
        self._ext.slot_all_reduce(
            inp,
            out,
            self._table,
            self.region.ctrl_device(),
            self._order,
            self.seq,
            self.row,
            MAXN,
            self._words[0] if inp.numel() <= SMALL_ELEMENTS else self._words[1],
            self.variant == "p2p",
        )

    def set_members(self, members: Iterable[int], epoch: int) -> None:
        """Take up this process's order for its row; called at a boundary.

        ``members`` are region rows in rank position, as this process computed them from the
        change its group applies: every member derives the same order from the same change, so
        every member sums in the same order and the ranks' outputs are equal. The order is
        written into this row of the region (the record the executor and the watch read) and
        into this process's device memory, where the kernel reads it, so this runs with the
        member quiescent. Nothing is waited for: the change is the member's own.

        Raises:
            ValueError: for an order with a row twice or out of range.
        """
        del epoch  # the epoch fences the runtime; the kernel needs only the order
        members = [int(m) for m in members]
        self.region.set_members([self.row], members)
        self._open(members)
        self._members = members
        self._write_order()

    def abort(self) -> None:
        """Release this process from a kernel spinning on a dead peer. Safe at any time."""
        self.region.set_abort([self.row], 1)

    def resume(self) -> None:
        """Zero this process's counters and its own words once its stream is idle.

        Its own abort word is cleared last: nothing of this process waits in the kernel now, and
        the abort was for the wait the change cut short. The executor clears the abort words of
        the rows whose process is gone, after every member has done this.
        """
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            with torch.inference_mode():  # the counters may have been made under inference mode
                self.seq.zero_()
            own_bytes = _PUSH_B if self.variant == "p2p" else _ROW_B
            self._ext.memset_async(self._addrs[self.row], own_bytes)
            torch.cuda.synchronize(self.device)
        self.region.set_abort([self.row], 0)

    def close(self) -> None:
        """Unmap peers' buffers and free this process's own."""
        for p in self._opened.values():
            self._ext.ipc_close(p)
        self._opened.clear()
        if self._own:
            self._ext.device_free(self._own)
            self._own = 0
