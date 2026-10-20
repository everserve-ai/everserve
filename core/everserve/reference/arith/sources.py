# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Where a rank reads its weights: the host copy every rank can reach.

A rank loads only the rows or columns of each tensor its units need, and loads more after a loss.
The checkpoint directory, read through the OS page cache, is that host copy in production;
:class:`DeterministicSource` plays the same role in tests with no files.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import torch

Rows = slice | torch.Tensor | None


class WeightSource(Protocol):
    """Reads tensors by their Hugging Face names, optionally a row or column slice."""

    def get(self, name: str, rows: Rows = None, cols: Rows = None) -> torch.Tensor:
        """The tensor ``name`` in bfloat16, restricted to ``rows`` and ``cols`` when given."""
        ...


class DeterministicSource:
    """Pseudo-random weights addressed by name, identical in every process. For tests."""

    def __init__(self, shapes: Callable[[str], tuple[int, ...]]) -> None:
        """``shapes(name)`` gives each tensor's shape."""
        self._shapes = shapes

    def get(self, name: str, rows: Rows = None, cols: Rows = None) -> torch.Tensor:
        """A tensor seeded by its name: near one for norms, scaled Gaussian otherwise."""
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:12], 16)
        g = torch.Generator().manual_seed(seed)
        shape = self._shapes(name)
        if "norm" in name:
            t = 1.0 + 0.1 * torch.randn(shape, generator=g)
        else:
            t = torch.randn(shape, generator=g) / math.sqrt(shape[-1])
        t = t.to(torch.bfloat16)
        if rows is not None:
            t = t[rows]
        if cols is not None:
            t = t[:, cols]
        return t.contiguous()


class SafetensorsSource:
    """A checkpoint directory of safetensors files, read lazily and sliced on read."""

    def __init__(self, path: str | Path) -> None:
        """Index the directory: a sharded ``model.safetensors.index.json`` or one file."""
        from safetensors import safe_open

        self._path = Path(path)
        self._open = safe_open
        index = self._path / "model.safetensors.index.json"
        if index.exists():
            self._where: dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self._path / "model.safetensors", framework="pt") as f:
                self._where = dict.fromkeys(f.keys(), "model.safetensors")
        self._files: dict[str, object] = {}
        self._lock = threading.Lock()

    def _handle(self, file: str):  # noqa: ANN202 - safetensors has no public handle type
        with self._lock:
            if file not in self._files:
                self._files[file] = self._open(self._path / file, framework="pt")
            return self._files[file]

    def get(self, name: str, rows: Rows = None, cols: Rows = None) -> torch.Tensor:
        """The tensor ``name`` in bfloat16, sliced before it leaves the file when possible."""
        h = self._handle(self._where[name])
        if rows is None and cols is None:
            return h.get_tensor(name).to(torch.bfloat16)
        s = h.get_slice(name)
        if cols is None:
            return s[rows].to(torch.bfloat16).contiguous()
        return s[rows if rows is not None else slice(None), cols].to(torch.bfloat16).contiguous()


def fetch(
    source: WeightSource, names: Iterable[str], device: torch.device, threads: int = 16
) -> dict[str, torch.Tensor]:
    """Read many whole tensors in parallel and move them to ``device``."""
    names = list(names)
    with ThreadPoolExecutor(threads) as ex:
        return dict(zip(names, ex.map(lambda n: source.get(n).to(device), names), strict=True))
