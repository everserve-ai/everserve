# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The whole model resident in pinned host memory, and one rank's shard loaded from it.

A spare node that holds every shard in host memory and loads the one it takes (decisions 0028
and 0030) reads the model's safetensors files once, at start, into page-locked host buffers
(:class:`HostCopy`), and keeps them for as long as the process stands. When a change gives the
spare a position, vLLM builds that position's model as any rank builds it, sharded by the
stand-in groups for the rank, and loads its weights through the loader this module registers
(``load_format`` :data:`LOAD_FORMAT`), which yields every tensor of the copy as a view into the
pinned buffers; vLLM's own weight loaders take each rank's slice of each tensor and skip the
layers of other pipeline stages, as they do from disk. The copy to the device runs at the
host-to-device rate of pinned memory, which is the term the mode adds to a change: the record
of each load (:func:`last_load`) carries the bytes the loaders took and the time.

Nothing here reads vLLM at import: the loader is registered by :func:`install`, in a process
that has vLLM, and the copy is plain safetensors parsing over ``torch`` buffers.
"""

from __future__ import annotations

import json
import logging
import struct
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

LOAD_FORMAT = "everserve-resident"
"""The ``load_config.load_format`` that loads a model from the process's :class:`HostCopy`."""

HOST_REGISTER_PORTABLE = 1
"""``cudaHostRegisterPortable``: the buffers count as page-locked in every CUDA context of the
process, as torch's own registration passes (``torch.cuda._pin_memory_utils``)."""

READ_CHUNK = 64 << 20
"""Bytes per read into the pinned buffer; a read of the whole file at once pins nothing sooner."""

DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
}
"""safetensors dtype names to torch's; a name this build's torch lacks maps to None."""


@dataclass(frozen=True)
class Entry:
    """One tensor of the copy: which file's buffer, which bytes of it, its dtype and shape."""

    file: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    start: int
    end: int


def parse_header(raw: bytes, file: int) -> dict[str, Entry]:
    """The entries of one safetensors header (its JSON, without the length prefix).

    Raises:
        ValueError: a dtype this build cannot hold, or a tensor whose bytes do not match its
            dtype and shape.
    """
    header = json.loads(raw)
    entries: dict[str, Entry] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"tensor {name!r}: dtype {info['dtype']!r} is none this build holds")
        shape = tuple(int(x) for x in info["shape"])
        start, end = (int(x) for x in info["data_offsets"])
        n = 1
        for x in shape:
            n *= x
        if end - start != n * _itemsize(dtype):
            raise ValueError(f"tensor {name!r}: {end - start} bytes for shape {shape} of {dtype}")
        entries[name] = Entry(file, dtype, shape, start, end)
    return entries


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class HostCopy:
    """Every safetensors file of a model directory, read once into host buffers.

    ``pin`` page-locks the buffers (the device copy then runs by DMA at the bus's rate); a
    process without a device holds them unpinned, which is the same copy for a test.

    Each buffer is allocated at its file's exact size and page-locked in place
    (``cudaHostRegister``), not taken from torch's pinned allocator: that allocator rounds every
    block up to a power of two, so it holds 124.6 GB for a 72.7 GB model (fifteen files of about
    4.9 GB, each in an 8 GiB block) and takes 53 to 178 s to pin it. The copy holds
    :attr:`nbytes` and no more; the lock is released before the memory
    can be freed (:meth:`release`, or when the copy is collected).
    """

    def __init__(self, path: Path, buffers: list[torch.Tensor], entries: dict[str, Entry]) -> None:
        """The copy of ``path``: one buffer per file, ``entries`` naming each tensor's bytes."""
        self.path = path
        self.buffers = buffers
        self.entries = entries
        self.stage_s = 0.0
        # the locked buffers' addresses; the finalizer holds this list, not the copy, so a copy
        # collected without release() is unlocked before its buffers can be freed
        self._locked: list[int] = []
        weakref.finalize(self, _unregister, self._locked)

    @property
    def pinned(self) -> bool:
        """Whether the buffers are page-locked."""
        return bool(self._locked)

    def _pin(self) -> None:
        """Page-lock every buffer in place, each at its own size.

        Raises:
            RuntimeError: the driver refused a buffer; those locked before it are released.
        """
        cudart = torch.cuda.cudart()
        for buf in self.buffers:
            if not buf.numel():
                continue
            try:
                torch.cuda.check_error(
                    cudart.cudaHostRegister(buf.data_ptr(), buf.numel(), HOST_REGISTER_PORTABLE)
                )
            except RuntimeError:
                self.release()
                raise
            self._locked.append(buf.data_ptr())

    def release(self) -> None:
        """Release the page lock on every buffer; the copy stays readable, unpinned."""
        _unregister(self._locked)

    @classmethod
    def hold(cls, model_dir: str | Path, pin: bool = True) -> HostCopy:
        """Read every ``*.safetensors`` under ``model_dir`` into host memory.

        Raises:
            FileNotFoundError: no safetensors file under ``model_dir``.
            ValueError: a file whose header this build cannot hold (:func:`parse_header`).
        """
        path = Path(model_dir)
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors file under {path}")
        t0 = time.time()
        buffers: list[torch.Tensor] = []
        entries: dict[str, Entry] = {}
        for i, file in enumerate(files):
            with file.open("rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                entries.update(parse_header(f.read(header_len), i))
                size = file.stat().st_size - 8 - header_len
                buf = torch.empty(size, dtype=torch.uint8)
                view = memoryview(buf.numpy())
                done = 0
                while done < size:
                    got = f.readinto(view[done : min(size, done + READ_CHUNK)])
                    if not got:
                        raise ValueError(f"{file}: {size} bytes of tensors declared, {done} read")
                    done += got
            buffers.append(buf)
        copy = cls(path, buffers, entries)
        if pin:
            copy._pin()
        copy.stage_s = round(time.time() - t0, 3)
        log.info(
            "model %s held in %s host memory: %d tensors, %d bytes in %d files, %.1f s",
            path,
            "pinned" if pin else "unpinned",
            len(entries),
            copy.nbytes,
            len(files),
            copy.stage_s,
        )
        return copy

    @property
    def nbytes(self) -> int:
        """The bytes of tensors held."""
        return sum(int(b.numel()) for b in self.buffers)

    def tensor(self, name: str) -> torch.Tensor:
        """``name``'s tensor, a view into the copy's buffer (no bytes moved)."""
        e = self.entries[name]
        return self.buffers[e.file][e.start : e.end].view(e.dtype).reshape(e.shape)

    def tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Every tensor of the copy, by name, as vLLM's weight loaders take them."""
        for name in self.entries:
            yield name, self.tensor(name)


def _unregister(ptrs: list[int]) -> None:
    """Release the page lock on each of ``ptrs``, registered by :meth:`HostCopy._pin`."""
    if not ptrs:
        return
    cudart = torch.cuda.cudart()
    for ptr in ptrs:
        err = cudart.cudaHostUnregister(ptr)
        if err != cudart.cudaError.success:
            log.warning("host buffer at %#x not unregistered: %s", ptr, err)
    ptrs.clear()


_COPY: HostCopy | None = None
_LAST: dict[str, Any] = {}
_INSTALLED = False


def hold(model_dir: str | Path, pin: bool = True) -> HostCopy:
    """Hold ``model_dir`` in this process's host memory; the loader loads from it from here on."""
    global _COPY
    _COPY = HostCopy.hold(model_dir, pin=pin)
    return _COPY


def held() -> HostCopy | None:
    """The copy this process holds, if any."""
    return _COPY


def last_load() -> dict[str, Any]:
    """The last load from the copy: ``bytes`` the loaders took, ``tensors``, ``s``."""
    return dict(_LAST)


def load_into(model: Any) -> set[str] | None:
    """Load ``model``'s weights from the held copy; what ``model.load_weights`` reports loaded.

    Raises:
        RuntimeError: no copy held.
    """
    copy = _COPY
    if copy is None:
        raise RuntimeError("no model held in host memory: resident.hold(model_dir) first")
    taken = {"bytes": 0, "tensors": 0}

    def counted() -> Iterator[tuple[str, torch.Tensor]]:
        for name, t in copy.tensors():
            taken["bytes"] += t.numel() * t.element_size()
            taken["tensors"] += 1
            yield name, t

    t0 = time.time()
    loaded = model.load_weights(counted())
    _LAST.clear()
    _LAST.update(taken, s=round(time.time() - t0, 3))
    return loaded


def install() -> None:
    """Register the loader with vLLM under :data:`LOAD_FORMAT`; once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    @register_model_loader(LOAD_FORMAT)
    class ResidentLoader(BaseModelLoader):
        """vLLM's loader for a model this process holds in host memory."""

        def download_model(self, model_config: Any) -> None:
            """Nothing to fetch: the copy is in this process."""

        def load_weights(self, model: Any, model_config: Any) -> None:
            loaded = load_into(model)
            # vLLM's own rule (DefaultModelLoader): every parameter accounted for only for a model
            # loaded as stored; a quantized one makes parameters (scales) the checkpoint lacks
            if loaded is not None and getattr(model_config, "quantization", None) is None:
                missing = {n for n, _ in model.named_parameters()} - loaded
                if missing:
                    raise ValueError(
                        f"{len(missing)} parameters not in the host copy: {sorted(missing)[:5]}"
                    )

    _INSTALLED = True
