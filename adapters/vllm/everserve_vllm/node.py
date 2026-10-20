# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The executor's node role: the engine's processes on a node without the engine (decision 0022).

vLLM's headless node runs a stock executor with no engine: it launches the node's workers, and
when one dies it shuts the rest down. In the node role the executor is ours, as on the head, so
a death on any node is an event there:

* it creates the node's slot region before the workers start, named for the engine so the
  spares the node's daemon launches map it too (decision 0024);
* it joins the node's daemon as the engine's client and reports a worker's loss at the first
  sign of it: its process exiting, killed, a zombie or gone in ``/proc`` (read every
  :data:`POLL_S`, as the slot watch reads a late member; the sentinel, which fires only once
  the driver has torn the process down, a second or two later, records the exit code), reads
  the region for the ranks a worker's slot watch found lost (decision 0021) and fences a hung
  one, and clears the words of the rows a change left without a process;
* a step is in flight on this node by its workers' own step words (:meth:`Region.in_flight`),
  so the stall rule of decision 0021 holds here without the engine's count;
* a position a change moved to another node has *left* it, one a process on this node took is
  *taken*; a worker that left on a notice releases itself after its drain and this node ends its
  process (:mod:`everserve_vllm.vacate`), and once the node holds nothing :meth:`serve` returns
  and the node's process exits (decision 0033, step 3).

What it does not do: send a step, join a spare, or complete a change; the head's executor does,
over the ring and through the daemons. ``python -m everserve_vllm.node_main`` runs it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from everserve.contract.daemon import Heard
from everserve.discovery import local_daemon
from everserve.policy import Engine
from everserve.policy import load as load_policy
from everserve_vllm import settings, vacate, worker
from everserve_vllm.executor import EverserveExecutor
from everserve_vllm.losses import exiting_ranks

log = logging.getLogger(__name__)

POLL_S = 0.01
"""How often the node reads its workers' ``/proc`` state and its region's lost marks: the slot
watch's stall scan reads processes as often."""


class NodeExecutor(EverserveExecutor):
    """The multiprocess executor on a node without the engine; see the module."""

    HEAD = False
    """The engine core runs on another node: this one registers as a client, and its stop or loss
    is its workers', never the engine's (decision 0032)."""

    def __init__(
        self, vllm_config: Any, proc: Path = Path("/proc"), engine_policy: Engine | None = None
    ) -> None:
        """Create the region and launch this node's workers; the monitor starts in :meth:`serve`.

        ``proc`` is the proc file system the workers are watched in (a test points it elsewhere).
        ``engine_policy`` is the engine's policy as the node was given it on its link; None, the
        policy document the settings name, if any.
        """
        # before vLLM's start, whose finalizer shuts this node down if the start fails
        self._stop = threading.Event()
        super().__init__(vllm_config, monitor_workers=False)
        if self.region is None:
            raise RuntimeError(
                "the node role needs its slot region: start it through everserve_vllm.node_main, "
                "which sets EVERSERVE_GRAPH_SAFE"
            )
        # this node ends the processes that leave on a change (everserve_vllm.vacate)
        self.region.set_ends_released()
        self._proc = proc
        s = settings.current()
        self.node = s.node
        if engine_policy is None and s.policy:
            engine_policy = load_policy(s.policy).engine(s.engine)
        self.engine_policy = engine_policy
        self.left: dict[int, dict[str, Any]] = {}  # positions a change moved to another node

    def serve(
        self, stop: threading.Event | None = None, on_turn: Callable[[], None] | None = None
    ) -> None:
        """Join the daemon, watch the workers and the region until stopped or nothing is held.

        Stopped by ``stop``; nothing held by :meth:`holds_nothing`. Runs on the calling thread.
        ``on_turn`` is called as the loop starts and on each of its turns: the process's
        liveness (:class:`everserve_vllm.health.Health`), which a stuck loop stops.
        The workers are bootstrapped by the head's RPC over the ring, as on the head, so nothing
        is sent to them here.

        Raises:
            RuntimeError: when the node's daemon has not published its record.
        """
        self._stop = stop or self._stop
        s = settings.current()
        engine, node, directory = s.engine, s.node, s.discovery_dir
        record = local_daemon(node, directory)
        if record is None:
            raise RuntimeError(f"no daemon record for node {node} in {directory}")
        self._connect(record.address, engine, node)
        threading.Thread(target=self._follow_commits, daemon=True, name="EverserveCommits").start()
        self.start_worker_monitor()
        self.started = {
            "node": node,
            "ranks": [h.rank for h in self.workers],
            "slot": self.slot_choice,
            "detect": self.detect,
        }
        log.info("everserve node role started: %s", self.started)
        if on_turn is not None:
            on_turn()
        while not self._stop.wait(POLL_S):
            if on_turn is not None:
                on_turn()
            self._watch_processes()
            self._slot_lost()
            self._end_released()
            self._left_by_placement()
            self._end_overdue()
            if self.holds_nothing():
                log.info("this node holds nothing of %s: released %s", engine, self.released)
                return

    def _watch_processes(self) -> None:
        """Report every local worker whose process is exiting, at the first sign of it."""
        pids = {h.rank: h.proc.pid for h in self.workers}
        known = self.dead | set(self.taken) | set(self.released)
        for rank in exiting_ranks(pids, known, self._proc):
            self._lost(rank, "proc", time.time())
            if self.client is not None:
                self.client.report_lost([rank], "process exiting", self.epoch)
            log.warning("worker %d is exiting; reported to the daemon", rank)

    def _left_by_placement(self) -> None:
        """A dead worker's rank that a later placement puts on another node has left this node.

        The change that replaces a rank lost here goes to its survivors and spares, not to the node
        whose worker died: that node hears the engine's placement, which every connection gets at
        every change (protocol 11), and never the commit :meth:`node_change` reads. Without this, a
        node whose every worker died would go on serving nothing, its pod never started again to
        stand as a spare, and the engine's next loss of the rank would find no spare for it.
        """
        placed = self.client.placement if self.client is not None else None
        if placed is None or placed.epoch <= self.epoch:
            return
        where = {p.rank: p.node for p in placed.members}
        for h in self.workers:
            r = h.rank
            to = where.get(r, self.node)
            if r in self.dead and r not in self.left and to != self.node:
                # dead already: nothing to drain, so no grace before it counts as ended
                self.left[r] = {"epoch": placed.epoch, "to": to, "at": time.time(), "grace": False}
                self.events.append({"kind": "left", "rank": r, "to": to, "epoch": placed.epoch})
                log.info(
                    "worker %d's rank is placed on %s at epoch %d: it has left", r, to, placed.epoch
                )

    def _end_released(self) -> None:
        """End the process of every local worker that marked itself released since the last poll.

        It drained and closed its link first (:func:`everserve_vllm.vacate.release_position`).
        The monitor is re-armed without it before its death pipe is closed, so its exit is not a
        loss; the end runs on its own thread, so this node keeps watching the others meanwhile.
        """
        assert self.region is not None
        first = self._first_local()
        for row in sorted(self.region.released()):
            rank = first + row
            if rank in self.released or row >= len(self.workers):
                continue
            self.released[rank] = {"epoch": self.epoch, "at": time.time()}
            self.start_worker_monitor()
            threading.Thread(
                target=self._end, args=(rank,), daemon=True, name=f"EverserveRelease{rank}"
            ).start()

    def _end_overdue(self, now: float | None = None) -> None:
        """End every worker whose position left this node past its grace, unreleased.

        A leaving worker releases itself at the head's boundary, after its drain; one that has
        not by then is stuck, as a live member late past ``hang_s`` is hung, and its position is
        held elsewhere already. It is ended as a released one is, recorded ``overdue``: the
        grace period a Kubernetes pod's termination has before its kill.
        """
        now = time.time() if now is None else now
        for rank, left in sorted(self.left.items()):
            bound = settings.current().hang_s if left.get("grace", True) else 0.0
            if rank in self.released or now - left.get("at", now) < bound:
                continue
            if rank - self._first_local() >= len(self.workers):
                continue
            log.warning(
                "worker %d not released %.1f s after it left; ended", rank, now - left["at"]
            )
            self.released[rank] = {"epoch": left["epoch"], "at": now, "overdue": True}
            self.start_worker_monitor()
            threading.Thread(
                target=self._end, args=(rank,), daemon=True, name=f"EverserveRelease{rank}"
            ).start()

    def _end(self, rank: int) -> None:
        handle = self.workers[rank - self._first_local()]
        writer, handle.death_writer = handle.death_writer, None
        ended = vacate.end_process(handle.proc, writer)
        self.released[rank].update(how=ended.how, exitcode=ended.exitcode, end_s=ended.end_s)
        self.events.append({"kind": "released", "rank": rank, **self.released[rank]})
        log.info("released worker %d ended: %s", rank, self.released[rank])

    def holds_nothing(self) -> bool:
        """Whether every worker launched here has left and exited, and nothing is taken here."""
        launched = {h.rank: h.proc for h in self.workers}
        return vacate.holds_nothing(launched, self.left, self.taken)

    def leave(self, lead_s: float, now: float | None = None) -> list[int]:
        """Give notice for every rank this node still serves: each leaves ``lead_s`` from now.

        What a node told to stop does (its pod is being deleted: an eviction, a drain, a spot
        reclaim) while it still serves: the coordinator moves its ranks to their spares at the
        deadline, as an operator's drain does (decision 0033), and the node holds nothing after
        and returns from :meth:`serve`. The ranks it noticed, in order. The lead is never under
        the minimum of the policy this node heard (:meth:`~everserve.contract.daemon.HearsEngine.
        notice_lead_s`).
        """
        assert self.client is not None
        lead_s = self.client.notice_lead_s(lead_s)
        deadline_ms = int(((time.time() if now is None else now) + lead_s) * 1000)
        ranks = sorted(
            h.rank for h in self.workers if h.rank not in self.left and h.rank not in self.released
        )
        for rank in ranks:
            self.client.notice(rank, deadline_ms)
        log.info("leaving: ranks %s noticed to leave in %.1f s", ranks, lead_s)
        return ranks

    def _follow_commits(self) -> None:
        """Every change the daemon commits, applied to this node's bookkeeping."""
        assert self.client is not None
        while not self._stop.is_set():
            try:
                heard = self.client.committed.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.node_change(heard)
            except Exception:  # noqa: BLE001 - the next commit must still be followed
                log.exception("applying the change at epoch %d", heard.committed.epoch)

    def node_change(self, heard: Heard) -> dict[str, Any]:
        """This node's half of a committed change; the head completes it over the ring.

        A position of this node's stage that the change took is taken by a spare on this node
        (decision 0022), whose row takes the position in the sum (decision 0012); every member
        writes its own row's order at the boundary, from the change its group applies, so this
        node owns the region's lifetime and writes no order. The rows of the ranks that left have
        no process to clear their words, so they are cleared here; the survivors clear their own
        at the boundary. The ranks the change took are not dead from here: a later loss in those
        positions is a new one. The worker monitor is re-armed without the positions that left,
        whose old processes are gone or going.

        A position whose taker runs on another node has *left* (:func:`vacate.taker_node`): it is
        neither dead nor taken here, and its worker, if it left alive, releases itself.

        The commit's ``members`` (protocol 9) are the authority on where each rank runs: a rank
        launched here and placed on another node has left, whatever claims this node heard or
        missed, so a node that hears only commits converges too (a node that hears no claim for
        its own ranks, under a commit naming no rank lost, must still act on the commit). The
        claim adds the positions taken by a spare on this node.

        A daemon that relinks is sent no commit it missed, only the next: until then a launched
        rank no longer placed keeps running, which is safe, since every message it sends carries
        an epoch the membership has moved past, and the next commit settles it here.

        A position that left is given its grace (:meth:`_end_overdue`) by what this node heard: a
        rank its claim named departing, or a rank it heard no claim for, drains at the head's
        boundary after the commit and has ``hang_s`` to; a rank its claim named a fault has no
        drain to wait for (alive, it was faulted falsely) and none.
        """
        epoch = heard.committed.epoch
        placed = {p.rank: p for p in heard.committed.members}
        # a rank already left is not moved again: a commit read again changes nothing
        moved = {
            h.rank: placed[h.rank].node
            for h in self.workers
            if h.rank in placed and placed[h.rank].node != self.node and h.rank not in self.left
        }
        claimed = self.claimed.get(epoch)
        faulted = set(claimed or ()) - self.departing  # a heard claim's ranks, not departing
        took = tuple(dict.fromkeys((*(claimed or ()), *sorted(moved))))
        if not placed:
            took = took or tuple(heard.committed.lost)  # a commit before protocol 9
        takes = self.claimed_takes.get(epoch, ())
        if self.region is not None:
            left = worker.take_rows(self.layout, self._first_local(), self.slot_rows, took)
            self.region.clear_watch(left)
            self.region.set_abort(left, 0)
        self.epoch = epoch
        left = {}
        for r in took:
            self.dead.discard(r)
            self.departing.discard(r)
            self._fenced.discard(r)
            if not self._is_local(r) or r in self.left:
                continue  # another node's, or left at an earlier change
            to = moved.get(r) or vacate.taker_node(r, takes, self.engine_policy, self.node)
            if to == self.node:
                self.taken[r] = {"epoch": epoch}
            else:
                left[r] = self.left[r] = {
                    "epoch": epoch,
                    "to": to,
                    "at": time.time(),
                    "grace": r not in faulted,
                }
        self.start_worker_monitor()
        record = {
            "kind": "changed",
            "epoch": epoch,
            "took": list(took),
            "left": {r: v["to"] for r, v in left.items()},
            "at": time.time(),
        }
        self.events.append(record)
        log.info("change at epoch %d applied on this node: took %s", epoch, list(took))
        return record

    def sweep_dead_reader(self, rank: int) -> None:
        """Nothing: a node without the engine has no broadcast ring, so no reader to stand in for.

        The head sweeps the dead rank's slot in its ring (its writer would block on it once it
        wraps); here the rank read the ring over the network, and the head's sweep covers it.
        """

    def _stop_watches(self) -> None:
        """Stop this node's slot watches through the region: there is no ring to ask over."""
        if self.region is not None:
            self.region.set_stop()

    def shutdown(self) -> None:
        """Stop following, then the executor's shutdown: the watches, the client, the workers."""
        self._stop.set()
        super().shutdown()
