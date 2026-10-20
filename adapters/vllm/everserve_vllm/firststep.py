# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What a worker's first steps after a membership change spend their time on.

An instrument, not a fix. On two pods the first fault run of an arm took about 0.55 s longer in its
first step after resuming than every later run, and nothing recorded why. A worker arms
:class:`FirstSteps` at the boundary where it applies a change; for the next ``steps`` steps it
records:

* when each step began and ended, from the connector's step hooks, against the boundary;
* the GPU time of each step, from CUDA events on the current stream (read when the report is
  taken, so the probe adds no synchronisation to the step);
* what changed in torch's compile counters (``torch._dynamo.utils.counters``) and in the CUDA
  caching allocator (segments allocated, retries) over those steps: a compile or a new memory
  segment shows here;
* where the step thread was, sampled every ``interval_s`` from another thread: each sample is
  put in a class (compile, graph capture, a wait on the device, a wait on a queue or lock, or
  other) by the frames on its stack, and the most frequent innermost frames are kept.

The probe runs only for the steps after a change and stops itself after them or after
``limit_s``. :func:`report` is what the worker's ``stats`` returns.
"""

from __future__ import annotations

import collections
import sys
import threading
import time
from typing import Any

import torch

CLASSES = (
    # compiling, not running compiled code: dynamo's eval frame is on every compiled forward's stack
    ("compile", ("torch/_dynamo/convert_frame.py", "torch/_inductor/", "triton/compiler")),
    ("graph_capture", ("torch/cuda/graphs.py", "capture_begin", "capture_end")),
    ("device_wait", ("synchronize", "torch/cuda/streams.py", "Event.wait", ".item", "tolist")),
    ("queue_wait", ("shm_broadcast.py", "threading.py", "queue.py", "selectors.py")),
)
"""Sample classes, first match wins, by a substring of any frame's ``file:function``."""

_PROBE: FirstSteps | None = None
_DONE: list[FirstSteps] = []


def _where(frame: Any) -> list[str]:
    out = []
    while frame is not None:
        code = frame.f_code
        out.append(f"{code.co_filename}:{code.co_name}")
        frame = frame.f_back
    return out


def _short(where: str) -> str:
    path, _, fn = where.rpartition(":")
    parts = path.split("/")
    return "/".join(parts[-2:]) + ":" + fn


class FirstSteps:
    """The probe over the first ``steps`` steps after one change."""

    def __init__(
        self, steps: int = 2, interval_s: float = 0.005, top: int = 12, limit_s: float = 30.0
    ) -> None:
        """Arm now, sampling the calling thread: the one that applies changes and runs steps."""
        self.steps, self.interval_s, self.top, self.limit_s = steps, interval_s, top, limit_s
        self.t0 = time.time()
        self.thread = threading.get_ident()
        self.records: list[dict[str, Any]] = []
        self.events: list[tuple[Any, Any]] = []  # CUDA events per step; None without CUDA
        self.classes: collections.Counter[str] = collections.Counter()
        self.leaves: collections.Counter[str] = collections.Counter()
        self.samples = 0
        self._open: dict[str, Any] | None = None
        self.counters0, self.memory0 = _counters(), _memory()
        self.counters1: dict[str, int] = {}
        self.memory1: dict[str, int] = {}
        self._stop = threading.Event()
        self._sampler = threading.Thread(
            target=self._sample, name="everserve-firststep", daemon=True
        )
        self._sampler.start()

    def _sample(self) -> None:
        deadline = self.t0 + self.limit_s
        while not self._stop.wait(self.interval_s) and time.time() < deadline:
            frame = sys._current_frames().get(self.thread)
            if frame is None:
                continue
            where = _where(frame)
            self.samples += 1
            self.leaves[_short(where[0])] += 1
            joined = "\n".join(where)
            for name, marks in CLASSES:
                if any(m in joined for m in marks):
                    self.classes[name] += 1
                    break
            else:
                self.classes["other"] += 1

    def begin(self) -> None:
        """A step starts (the connector's ``handle_preemptions``)."""
        start = _event()
        self._open = {"begin_s": time.time() - self.t0, "start": start}

    def end(self) -> bool:
        """A step's forward is queued (the connector's ``wait_for_save``); True when done."""
        if self._open is None:
            return False
        stop = _event()
        self.events.append((self._open.pop("start"), stop))
        self.records.append({**self._open, "end_s": time.time() - self.t0})
        self._open = None
        if len(self.records) < self.steps:
            return False
        self.counters1, self.memory1 = _counters(), _memory()
        self.stop()
        return True

    def stop(self) -> None:
        """Stop sampling; the step thread does not wait for the sampler or the device."""
        self._stop.set()

    def finish(self) -> dict[str, Any]:
        """Summarise, reading the steps' GPU times (this waits for their events)."""
        self.stop()
        self._sampler.join(1.0)
        for rec, (a, b) in zip(self.records, self.events, strict=True):
            rec["gpu_ms"] = None
            if a is not None and b is not None:
                try:
                    b.synchronize()
                    rec["gpu_ms"] = round(a.elapsed_time(b), 3)
                except RuntimeError:  # an event that never ran: the step was abandoned
                    pass
            rec["begin_s"], rec["end_s"] = round(rec["begin_s"], 4), round(rec["end_s"], 4)
        counters, memory = self.counters1 or _counters(), self.memory1 or _memory()
        return {
            "armed_at": self.t0,
            "steps": self.records,
            "compile_counters": {
                k: v - self.counters0.get(k, 0)
                for k, v in counters.items()
                if v != self.counters0.get(k, 0)
            },
            "allocator": {k: memory[k] - self.memory0.get(k, 0) for k in memory},
            "samples": self.samples,
            "interval_s": self.interval_s,
            "classes_s": {k: round(n * self.interval_s, 3) for k, n in self.classes.items()},
            "top_frames": self.leaves.most_common(self.top),
        }


def _event() -> Any:
    if not torch.cuda.is_available():
        return None
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _counters() -> dict[str, int]:
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return {}
    return {f"{group}.{k}": int(v) for group, c in counters.items() for k, v in c.items()}


def _memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    s = torch.cuda.memory_stats()
    keys = ("segment.all.allocated", "num_alloc_retries", "num_device_alloc")
    return {k: int(s.get(k, 0)) for k in keys}


def arm(steps: int = 2) -> None:
    """At a change's boundary, on the step thread: probe the next ``steps`` steps."""
    global _PROBE
    if _PROBE is not None:
        _PROBE.stop()
        _DONE.append(_PROBE)
    _PROBE = FirstSteps(steps)


def step_begin() -> None:
    """Connector hook: a step starts. Nothing unless armed."""
    if _PROBE is not None:
        _PROBE.begin()


def step_end() -> None:
    """Connector hook: a step's forward is queued. Nothing unless armed."""
    global _PROBE
    if _PROBE is not None and _PROBE.end():
        _DONE.append(_PROBE)
        _PROBE = None


def report() -> list[dict[str, Any]]:
    """Every probe of this process, oldest first, an unfinished one last.

    Called off the step path (the worker's ``stats``): it waits for the probed steps' events.
    """
    return [p.finish() for p in [*_DONE, *([_PROBE] if _PROBE is not None else [])]]
