# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Detection at the collective (decision 0021): a watchdog over the slot kernel's late word.

The slot kernel knows which member's words have not arrived. Past a bound it writes that member's
row into its control row's late word and keeps waiting (``core/csrc/slot.cu``). :class:`SlotWatch`
is a thread on each worker that reads the word and decides what the lateness means:

* the late member's process is **exiting or gone**: it is dead. Reported at once, without waiting
  for the driver to tear the process down; the executor's process sentinel fires only after that
  teardown, 6.0 s on a 72B shape (``bench/runs/2026-10-06-commit-path-72b/``).
* the late member's process is **alive**: it is slow, not dead. A rank's step thread can stall
  for hundreds of milliseconds while the others wait in the kernel (the first commit after a
  prefill holds one rank 350-760 ms on that shape), so lateness alone is not a death. It is
  reported only if the member is still late after ``hang_s``: a hung worker, which no process
  sentinel ever reports.

A report goes two ways: :meth:`Region.mark_lost` for the executor, which reads it while waiting on
the step, and ``report`` (the worker's daemon link) for the coordinator. The coordinator folds the
second sighting of one death: every survivor's kernel sees it, and so, later, does the executor's
sentinel.

Linux only: liveness is read from ``/proc``. That is correct because every member of a slot region
is on this host, by construction: the slot collective is scoped to one node's domain. A member on
another node would have a process id that means nothing here, and would read as dead. A killed
process has ``PF_EXITING`` set from the moment it starts to exit, and ``SIGKILL`` pending before
that; either is a death.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from everserve.comm.slotkernel import Region

log = logging.getLogger(__name__)

PF_EXITING = 0x4
"""``include/linux/sched.h``: set in a task's flags as it begins to exit."""
SIGKILL_BIT = 1 << (9 - 1)
STALL_SCAN_S = 0.01
"""How often a stalled worker reads its members' processes."""
SETTLE_STEPS = 2
"""Steps a worker ends after a change before the hang clock runs again: a step's end is its
forward queued, so the second end means the first, the replacement's included, has run."""


def process_state(pid: int, proc: Path = Path("/proc")) -> str:
    """``"dead"`` (exiting, killed, a zombie or gone), ``"alive"``, or ``"unknown"``."""
    if pid <= 0:
        return "unknown"
    try:
        stat = (proc / str(pid) / "stat").read_text()
        status = (proc / str(pid) / "status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return "dead"
    except OSError:
        return "unknown"
    # the command name is in parentheses and may contain anything; the fields follow the last ')'
    fields = stat[stat.rfind(")") + 2 :].split()
    state, flags = fields[0], int(fields[6])
    if state in ("Z", "X", "x") or flags & PF_EXITING:
        return "dead"
    for line in status.splitlines():
        if line.startswith(("SigPnd:", "ShdPnd:")) and int(line.split()[1], 16) & SIGKILL_BIT:
            return "dead"
    return "alive"


class SlotWatch:
    """A thread on one worker watching its slot row's late word."""

    def __init__(
        self,
        region: Region,
        row: int,
        report: Callable[[list[int], str, int], None],
        epoch: Callable[[], int],
        progress: Callable[[], int] | None = None,
        hang_s: float = 10.0,
        poll_s: float = 0.002,
        proc: Path = Path("/proc"),
        first_rank: int = 0,
    ) -> None:
        """Watch ``row`` of ``region``.

        Args:
            region: this node's slot region: one stage of the engine.
            row: this process's row.
            report: called as ``report(ranks, detail, epoch)`` for ranks found lost.
            epoch: this process's membership epoch now.
            progress: steps this process has ended; after a change the hang clock waits until
                it has moved by :data:`SETTLE_STEPS`, the replacement's first step done. Without
                it, until a wait at the new epoch completes after being late.
            hang_s: how long a live member may stay late before it is reported hung.
            poll_s: how often the late word is read.
            proc: the proc file system (a test points it elsewhere).
            first_rank: the engine rank in position 0 of this region's stage. The region names
                members by position; a report names ranks, from this one.
        """
        self.region = region
        self.row = row
        self.first_rank = first_rank
        self._report = report
        self._epoch = epoch
        self._progress = progress
        self.hang_s = hang_s
        self.poll_s = poll_s
        self._proc = proc
        self.reported: list[dict] = []
        """Every report made, with when, why and the epoch; for the run's record."""
        self.slow: list[dict] = []
        self._done: dict[int, set[int]] = {}  # rows reported, by the epoch they were seen in
        """Every late member that turned out alive and arrived: its rank and how late it was."""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="EverserveSlotWatch")

    def start(self) -> SlotWatch:
        """Start the thread; returns self."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the thread."""
        self._stop.set()

    def _ranks_of(self, rows: set[int]) -> dict[int, int]:
        """Each late row's position in this row's member order: its rank within the stage."""
        order = self.region.order(self.row)
        return {r: order.index(r) for r in rows if r in order}

    def check(self, now: float, seen: dict) -> None:
        """One look at the late word; ``seen`` carries the current episode between looks.

        An episode runs from the word going nonzero to its clearing. Every late member whose
        process is dead is reported, a node's members together in one report. A live member is
        reported hung once it has been late past ``hang_s``, except while the members settle after
        a change: from the change until a wait completes at the new epoch, the replacement may
        still be restoring, which takes seconds at long contexts.
        """
        epoch_now = self._epoch()
        if epoch_now != seen.get("last_epoch", epoch_now):
            # a change was applied on this worker: the replacement may still be restoring
            seen["settling"] = True
            seen["settle_from"] = self._progress() if self._progress else None
        seen["last_epoch"] = epoch_now
        if (
            seen.get("settling")
            and seen.get("settle_from") is not None
            and self._progress() - seen["settle_from"] >= SETTLE_STEPS
        ):
            seen["settling"] = False
        self._check_stall(now, seen, epoch_now)
        late = self.region.late(self.row)
        ep = seen.get("episode")
        if not late:
            if ep is not None:
                for row, since in ep["since"].items():
                    if row not in ep["reported"]:
                        self.slow.append(
                            {
                                "rank": ep["ranks"].get(row),
                                "late_s": round(now - since, 4),
                                "at": now,
                            }
                        )
                if ep["epoch"] == epoch_now:
                    seen["settling"] = False  # a wait completed at this epoch: settled
            seen["episode"] = None
            return
        if ep is None:
            # the epoch the lateness began in: a change can be applied on this worker while its
            # kernel still waits, and a report from the new epoch would name the replacement
            ep = seen["episode"] = {"epoch": epoch_now, "since": {}, "ranks": {}, "reported": set()}
        for row in late - ep["since"].keys():
            ep["since"][row] = now
        ep["ranks"].update(self._ranks_of(late - ep["ranks"].keys()))
        bound = self.region.bound(self.row)
        dead, hung = [], []
        for row in sorted(late - ep["reported"] - self._done.get(ep["epoch"], set())):
            if row not in ep["ranks"]:
                continue
            pid = self.region.pid(row)
            if process_state(pid, self._proc) == "dead":
                dead.append(row)
            elif not seen.get("settling") and now - ep["since"][row] > self.hang_s:
                hung.append(row)
        for rows, how in ((dead, "exiting"), (hung, "alive and hung")):
            if rows:
                ep["reported"].update(rows)
                self._lost(
                    rows,
                    ep["ranks"],
                    ep["epoch"],
                    f"no words from ranks {{ranks}} past {bound * 1e3:.0f} ms;"
                    f" their processes {{pids}} are {how}",
                    now,
                )

    def _check_stall(self, now: float, seen: dict, epoch_now: int) -> None:
        """A step in flight that does not end within the bound: scan every member's process.

        The slot kernel names a late member only while a slot all-reduce waits on it. A death
        late in a step leaves no survivor waiting in one: the dead rank's queued GPU work finishes
        the step's collectives, the survivors end the step and wait for the next, and the executor
        waits for an answer the dead rank never sends (5x A6000, the survivors' stacks at a kill
        0.9 into a step). A step can also wait in a collective the slot does not own. So while
        the executor has a step in flight (:meth:`Region.in_flight`) and this worker has not ended
        one within the bound, every member's process is read, at most every
        :data:`STALL_SCAN_S`; one that is exiting or killed is reported, the set together. A member
        alive but stuck in a collective falls to that collective's own bound, as before.

        In flight is not this worker's to say: in the case above every survivor has ended its
        step, and only the executor knows the step is unanswered. On the engine's node the
        executor's count says it; on any node a member's own step word does (a member running a
        step RPC, or one that died inside one and never answered), so a node without the engine
        has the same gate. With nothing in flight (an idle engine, or one shutting down while its
        workers exit) nothing is scanned.
        """
        if self._progress is None:
            return
        if not self.region.in_flight():
            # no step in flight: an idle engine, or one shutting down, whose workers exit
            seen["progress_at"] = None
            return
        progress = self._progress()
        if seen.get("progress_at") is None or progress != seen.get("progress"):
            # the epoch the step began in, as for the slot's episodes
            seen.update(progress=progress, progress_at=now, progress_epoch=epoch_now)
            return
        bound = self.region.bound(self.row)
        if bound <= 0 or now - seen["progress_at"] <= bound:
            return
        if now - seen.get("scanned_at", 0.0) < STALL_SCAN_S:
            return
        seen["scanned_at"] = now
        epoch = seen["progress_epoch"]
        order = self.region.order(self.row)
        rows = [r for r in order if r != self.row and r not in self._done.get(epoch, set())]
        dead = [r for r in rows if process_state(self.region.pid(r), self._proc) == "dead"]
        if dead:
            self._lost(
                dead,
                self._ranks_of(set(dead)),
                epoch,
                f"no step ended for {(now - seen['progress_at']) * 1e3:.0f} ms;"
                " ranks {ranks}' processes {pids} are exiting",
                now,
            )

    def _lost(self, rows: list[int], ranks_of: dict, epoch: int, why: str, now: float) -> None:
        """Report ``rows`` lost at ``epoch``: to the executor through the region, and ``report``.

        The region's mark names positions in the stage; the report names engine ranks.
        """
        rows = sorted(set(rows) - self._done.get(epoch, set()))
        if not rows:
            return
        self._done.setdefault(epoch, set()).update(rows)
        positions = [ranks_of[r] for r in rows]
        ranks = [self.first_rank + p for p in positions]
        pids = [self.region.pid(r) for r in rows]
        why = f"slot row {self.row}: " + why.format(ranks=ranks, pids=pids)
        self.region.mark_lost(self.row, positions, epoch)
        self.reported.append({"ranks": ranks, "detail": why, "epoch": epoch, "at": now})
        log.warning("%s; reported lost at epoch %d", why, epoch)
        self._report(ranks, why, epoch)

    def _run(self) -> None:
        seen: dict = {}
        while not self._stop.wait(self.poll_s):
            if self.region.stopped():
                return  # the node's executor is shutting the region down: its exits are not losses
            try:
                self.check(time.time(), seen)
            except Exception:  # noqa: BLE001 - the watchdog must outlive one bad read
                log.exception("slot watch on row %d", self.row)
