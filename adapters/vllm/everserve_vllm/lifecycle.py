# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The connector's request lifecycle, kept free of vLLM so the mapping is testable anywhere.

``docs/CONTRACT.md`` item 3 maps vLLM's ``KVConnectorBase_V1`` onto
:class:`~everserve.state.committed.CommittedKV`; :mod:`everserve_vllm.connector` is the thin vLLM
class and this module is what it delegates to:

* scheduler side, per step: :meth:`Lifecycle.record` turns the scheduler's view (tokens computed
  and scheduled per request, requests preempted) into a :class:`StepRecord` that travels to the
  workers as the connector metadata; :meth:`Lifecycle.finished` forgets a request once;
* worker side, per step: :meth:`Lifecycle.preemptions` lowers committed counts before slots are
  overwritten; :meth:`Lifecycle.begin` runs the runtime's attempt callbacks (drop anything
  uncommitted); :meth:`Lifecycle.save` keeps a layer's rows; :meth:`Lifecycle.end` runs the
  runtime's commit callbacks, whose commit is :meth:`Lifecycle.commit`.

A step that a membership change interrupted is not committed: :meth:`Lifecycle.end` commits only
when the runtime saw no fault and no epoch change since :meth:`Lifecycle.begin`, so rows written
by an aborted step never reach the mirror. The scheduler replays that step's tokens.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from everserve.comm.runtime import Runtime
from everserve.state.committed import CommittedKV, RequestId


@dataclass(frozen=True)
class StepRecord:
    """What the scheduler tells the workers about one step.

    Attributes:
        committed: per request, the tokens whose KV is complete once this step's rows land.
        preempted: requests whose blocks were freed since the last step.
    """

    committed: Mapping[RequestId, int] = field(default_factory=dict)
    preempted: tuple[RequestId, ...] = ()


@dataclass
class CommitStats:
    """Mirror commit time per step on this worker, for the run's report."""

    steps: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    skipped: int = 0
    recent: list[float] = field(default_factory=list)

    def add(self, s: float) -> None:
        """Count one commit of ``s`` seconds; the last 256 are kept for percentiles."""
        self.steps += 1
        self.total_s += s
        self.max_s = max(self.max_s, s)
        self.recent.append(s)
        del self.recent[:-256]


class Lifecycle:
    """One side of the connector's lifecycle over a :class:`CommittedKV`."""

    def __init__(self, kv: CommittedKV | None = None, runtime: Runtime | None = None) -> None:
        """Start detached; a worker attaches its transport and runtime once they exist."""
        self.kv = kv
        self.runtime = runtime
        self.current = StepRecord()
        self.step = 0
        self.ended = 0
        """Steps ended, committed or not, mirror or none: the slot watch's sign of progress."""
        self.stats = CommitStats()
        self._epoch = 0
        self._saved: set[str] = set()

    def attach(self, kv: CommittedKV, runtime: Runtime) -> None:
        """Commit to ``kv`` under ``runtime``'s step guard from the next step.

        The runtime's attempt callbacks get ``kv.clear`` and its commit callbacks get
        :meth:`commit`, so the runtime owns when rows are dropped and when they are made durable.
        """
        self.kv = kv
        self.runtime = runtime
        runtime.attempt_callbacks.append(kv.clear)
        runtime.commit_callbacks.append(self.commit)

    def continue_from(self, step: int) -> None:
        """Number this process's next commit ``step``: a spare continuing a rank it took.

        The store holds the rank's steps in sequence: a taken rank's first commit is the
        restored watermark plus one, at the change's epoch, never a step the rank already
        passed (decision 0030): commits numbered from 0 are refused by every replica.
        """
        self.step = step

    # ------------------------------------------------------------------ scheduler side
    @staticmethod
    def record(
        scheduled: Iterable[tuple[RequestId, int, int]], preempted: Iterable[RequestId] = ()
    ) -> StepRecord:
        """The step record from ``(request, tokens computed before, tokens scheduled)``."""
        return StepRecord(
            committed={rid: done + n for rid, done, n in scheduled},
            preempted=tuple(sorted(preempted)),
        )

    def finished(self, request_id: RequestId) -> None:
        """``request_finished``: forget the request at the mirror."""
        if self.kv is not None:
            self.kv.finished(request_id)

    # ------------------------------------------------------------------ worker side
    def preemptions(self, rec: StepRecord) -> None:
        """``handle_preemptions``: a preempted request keeps no blocks, so nothing is committed."""
        if self.kv is None:
            return
        for rid in rec.preempted:
            self.kv.evicted(rid, 0)

    def begin(self, rec: StepRecord) -> None:
        """Start of a step: remember its record and epoch, drop anything uncommitted."""
        self.current = rec
        self._saved = set()
        if self.runtime is None:
            return
        self._epoch = self.runtime.epoch
        for cb in self.runtime.attempt_callbacks:
            cb()

    def save(self, layer_name: str, kv_layer: torch.Tensor, slots: torch.Tensor) -> None:
        """Keep the rows a layer wrote this step."""
        if self.kv is None:
            return
        self.kv.save(layer_name, kv_layer, slots)
        self._saved.add(layer_name)

    def saved(self, layer_name: str) -> bool:
        """Whether ``layer_name``'s rows were kept this step (FULL graphs skip the layer hook)."""
        return layer_name in self._saved

    def end(self) -> bool:
        """End of a step: commit through the runtime's callbacks unless a change interrupted it.

        Returns whether the step was committed.
        """
        self.ended += 1
        if self.kv is None or self.runtime is None:
            return False
        rt = self.runtime
        if rt.fault is not None or rt.epoch != self._epoch:
            self.kv.clear()
            self.stats.skipped += 1
            return False
        for cb in rt.commit_callbacks:
            cb()
        return True

    def commit(self) -> None:
        """The runtime's commit callback: this step's rows and counts to the mirror.

        The commit of this step starts without waiting; the fence then waits for the step two
        back (decisions 0009 and 0019). The agreement that follows therefore accepts a step only
        when every rank's rows through step N-2 are durable, the acknowledgement of N-2 has long
        arrived when N commits, and the replay after a fault is at most three one-token steps.
        """
        if self.kv is None or self.runtime is None:
            return
        t0 = time.perf_counter()
        self.kv.commit(self.runtime.epoch, self.step, self.current.committed)
        if self.step > FENCE_BEHIND - 1:
            self.kv.fence(self.step - FENCE_BEHIND)
        self.stats.add(time.perf_counter() - t0)
        self.step += 1

    def report(self) -> dict[str, Any]:
        """Commit-time figures for the run log."""
        r = sorted(self.stats.recent)
        return {
            "steps": self.stats.steps,
            "skipped": self.stats.skipped,
            "mean_ms": 1e3 * self.stats.total_s / max(self.stats.steps, 1),
            "p50_ms": 1e3 * r[len(r) // 2] if r else None,
            "p99_ms": 1e3 * r[min(len(r) - 1, int(len(r) * 0.99))] if r else None,
            "max_ms": 1e3 * self.stats.max_s,
            "transport": dict(self.kv.stats()) if self.kv is not None else {},
            "timeline": timeline_summary(self.kv.timeline()) if self.kv is not None else {},
        }


FENCE_BEHIND = 2
"""The fence at the commit of step N waits for step N - FENCE_BEHIND (decision 0019)."""

SPANS = {
    "event_lag": ("commit", "ready"),  # commit queued -> the device copies had landed
    "sender": ("ready", "sent"),  # the sender's own work and the store's round trip
    "ack_lag": ("commit", "sent"),  # commit queued -> acknowledged
    "fence_wait": ("fence_start", "fence_end"),  # what the step thread paid
    "fence_after_ack": ("sent", "fence_end"),  # negative when the ack came before the fence
}
"""The spans a run report splits a step's commit into, each as (from stamp, to stamp)."""


def timeline_summary(
    records: Sequence[Mapping[str, float]], tail: int = 64, head: int = 16
) -> dict[str, Any]:
    """Percentiles of every span over ``records``, and the raw stamps of the first and last steps.

    The first ``head`` and the last ``tail``. A span is skipped for a record missing either stamp
    (a step never fenced, a commit still in flight). Milliseconds throughout, relative to each
    record's own ``commit`` stamp. The head holds the prefill's commit and the steps behind it,
    which the percentiles cannot place.
    """
    spans: dict[str, dict[str, float]] = {}
    for name, (a, b) in SPANS.items():
        xs = sorted(1e3 * (r[b] - r[a]) for r in records if a in r and b in r)
        if xs:
            spans[name] = {
                "p50_ms": xs[len(xs) // 2],
                "p90_ms": xs[min(len(xs) - 1, int(len(xs) * 0.9))],
                "max_ms": xs[-1],
                "n": len(xs),
            }

    def rel(rs: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
        return [
            {k: (v if k == "step" else 1e3 * (v - r["commit"])) for k, v in r.items()}
            for r in rs
            if "commit" in r
        ]

    first = rel(records[:head])
    # each head step's commit relative to the first's: when the step thread got to it
    t0 = next((r["commit"] for r in records[:head] if "commit" in r), None)
    for f, r in zip(first, (r for r in records[:head] if "commit" in r), strict=True):
        f["at_ms"] = 1e3 * (r["commit"] - t0)
    return {
        "steps": len(records),
        "spans": spans,
        "first_ms_from_commit": first,
        "last_ms_from_commit": rel(records[-tail:]),
    }
