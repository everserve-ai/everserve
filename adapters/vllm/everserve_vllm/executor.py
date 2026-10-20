# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The executor: a worker death is an event, not a shutdown.

One subclass of vLLM's multiprocess executor: ``--distributed-executor-backend mp`` in a process
its launcher made an engine's (:mod:`everserve_vllm.backend`), or the class itself in a config a
driver builds. It

* creates the slot region before the workers start (:mod:`everserve.comm.slotkernel`), with the
  variant the topology allows;
* after the engine is warm, starts the spare, bootstraps every worker and the spare
  (:mod:`everserve_vllm.worker`), and connects to the node daemon as a client;
* on a worker's death reports ``RankLost`` to the daemon and keeps the engine core alive: the
  dead reader of the broadcast ring is swept, RPCs stop waiting for it, and the step in flight
  raises :class:`RankLostError` to the engine's caller. A death is seen first at the collective
  (decision 0021): a survivor's slot kernel names the member it waits for, the survivor's
  watchdog finds that member's process exiting, reports it, and marks it in the region, where
  this executor reads it while it waits on the step. The process sentinel is the second sighting,
  after the driver has torn the process down; the coordinator folds it. The executor does
  **not** choose the outcome; the daemon does, and sends the ``Change`` to the workers and the
  spare;
* on the daemon's ``MembershipCommitted`` runs the engine-core half of the change,
  :meth:`EverserveExecutor.complete_change`: a new ring, the survivors and the spare applying the
  change at the step boundary, the region reset, the spare in the dead rank's place, and the
  scheduler replaying the tail its committed rows do not hold.

``complete_change`` needs the engine core (its scheduler and batch queue); the harness driver
runs the engine in-process and calls it. A production engine core would call it from its step
loop;
that hook is listed in ``PRIVATE-INTERNALS.md`` with the rest. A notice ends the same way, with
no death: the daemon commits the replacement at the deadline, the executor swaps at the next
step boundary, and the leaving worker is never seen to fail.

Configuration is by environment, set by the launcher (:data:`ENV`).
"""

from __future__ import annotations

import logging
import multiprocessing.connection
import os
import pickle
import queue
import signal
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

import cloudpickle
from vllm.v1.executor.multiproc_executor import FutureWrapper, MultiprocExecutor, WorkerProc

from everserve import policy as everserve_policy
from everserve.comm.layout import ParallelLayout
from everserve.contract.daemon import DaemonClient, Heard, Refused, UndecidedConnection
from everserve.contract.head import ChangeAbandoned, launcher_pid, stop_cause, wait_commit
from everserve.contract.signals import Abandoned, Member, Role, SlotChoice
from everserve.discovery import local_daemon
from everserve.policy import Policy
from everserve_vllm import replay, settings, worker
from everserve_vllm.identity import software_identity
from everserve_vllm.losses import RankLostError, failed_step, follow_changes
from everserve_vllm.queues import close_queue
from everserve_vllm.resume import OneTokenReplay, resume_running
from everserve_vllm.spans import claim_to_commit
from everserve_vllm.spare import Spare
from everserve_vllm.worker import STEP_METHODS

log = logging.getLogger(__name__)


def chunk_bytes() -> int:
    """The chunk size of every ring this executor makes, as vLLM's executor sizes its own.

    From vLLM's setting (``VLLM_MQ_MAX_CHUNK_BYTES_MB``), never read back off the ring: a ring
    whose readers are all on other nodes has no shared-memory buffer to read it from (``buffer``
    is None, and a core reading it fails completing the change).
    """
    from vllm import envs

    return int(envs.VLLM_MQ_MAX_CHUNK_BYTES_MB) * 1024 * 1024


ENV = settings.ENV
"""Launcher settings, read through :mod:`everserve_vllm.settings`. Without ``EVERSERVE_ENGINE``
the executor behaves as stock, plus the death handling; without ``EVERSERVE_SPARE_GPU`` no
spare is started. The slot kernel names a member it waits for longer than
``EVERSERVE_DETECT_STEPS`` step periods (default 3), measured over the first decode steps, and
never sooner than ``EVERSERVE_DETECT_FLOOR_MS`` (default 30). A live member late past
``EVERSERVE_HANG_S`` (default 10) is reported hung."""

STEP_SAMPLE = (8, 40)
"""The step intervals the detection bound is measured over: after the first 8 (the prefill and
the first decodes), up to 40."""


def _aggregation(
    aggregators: list[Any], unique_reply_rank: int | None
) -> tuple[int | None, Callable[[Any], Any]]:
    """Which rank's reply a call returns, and how the replies are reduced to it.

    As vLLM's executor does: with output aggregators every rank answers and the aggregators
    fold them; without, the one rank asked for answers, or every rank's reply is a list.
    """
    if not aggregators:
        return unique_reply_rank, lambda outputs: outputs

    def aggregate(outputs: Any) -> Any:
        rank = unique_reply_rank or 0
        result = outputs[rank]
        for a in aggregators:
            result = a.aggregate(outputs, output_rank=rank)
        return result

    return None, aggregate


class EverserveExecutor(MultiprocExecutor):
    """The multiprocess executor with worker death as an event and the spare's handover."""

    HEAD = True
    """This executor runs beside the engine core and registers as the engine's head (0032)."""

    # ------------------------------------------------------------------ start
    def _init_executor(self) -> None:
        """Create the slot region before the workers start; each maps it at its first call."""
        if self.HEAD:
            # the engine core beside this executor completes changes in its step, or none at all
            from everserve_vllm import serving

            serving.require_installed()
        self.dead: set[int] = set()
        self.departing: set[int] = set()
        """Ranks a claimed change takes that leave on notice (protocol 9): alive, not dead. The
        engine core stops stepping for their change as for a loss, and the boundary awaits
        their drain."""
        self.events: list[dict[str, Any]] = []
        self.startup = replay.Recorder()
        """What this core sends its workers at its start, for a spare's replay (decision 0044)."""
        self.spare: Spare | None = None
        self.client: DaemonClient | None = None
        # the failure that ends this engine, said in the head's goodbye (decision 0032), and a
        # change the coordinator abandoned after its delivery, which is one
        self.failure: str | None = None
        self.abandoned: Abandoned | None = None
        self.region = None
        self.slot_rows: list[int] = []
        self.slot_choice: dict[str, Any] | None = None
        self._monitor_gen = 0
        self._sweeping: set[int] = set()
        self.epoch = 0
        self.claimed: dict[int, tuple[int, ...]] = {}  # the ranks each claimed change took
        self.claimed_takes: dict[int, tuple[Any, ...]] = {}  # and which spare took which, by node
        self.carrier = False  # the daemons own the spares and the join crosses them (0024)
        self.taken: dict[int, dict[str, Any]] = {}  # positions taken through the carrier
        self.released: dict[int, dict[str, Any]] = {}  # left on a change, its process ended here
        self.lost_at: dict[int, int] = {}  # the epoch each rank was first seen lost at
        self._fenced: set[int] = set()  # ranks whose process this executor killed
        self._steps: list[float] = []
        self._steps_out = 0  # step RPCs sent and not yet answered
        self.detect: dict[str, Any] = {
            "steps": settings.current().detect_steps,
            "floor_s": settings.current().detect_floor_ms / 1e3,
            "step_s": None,
        }
        self.detect["bound_s"] = self.detect["floor_s"]
        self.layout = ParallelLayout(
            self.parallel_config.tensor_parallel_size, self.parallel_config.pipeline_parallel_size
        )
        launched = settings.current()
        self.asking: UndecidedConnection | None = None
        self.given: Policy | None = None
        if self.HEAD and launched.engine and not launched.policy:
            self._ask()
        if launched.graph_safe:
            from dataclasses import asdict
            from pathlib import Path

            from everserve.comm.slotkernel import (
                VARIANTS,
                Region,
                VariantChoice,
                decide_variant,
                region_path,
            )

            engine = launched.engine
            # named for the engine, so the spares its daemon launches find it (decision 0024);
            # without a policy, for this process alone
            path = region_path(engine) if engine else f"/dev/shm/everserve_slot_{os.getpid()}"
            # the variant follows the topology of every GPU the engine and its spares use; NVML,
            # not CUDA, so this process starts no CUDA context before vLLM starts its workers
            gpus = list(launched.visible_devices)
            # an explicit variant is the operator's override; otherwise the hardware decides, by
            # NVML and then by running the peer-to-peer collective once (choose_variant)
            forced = launched.slot_variant
            if forced in VARIANTS:
                choice = VariantChoice(forced, "override", f"{ENV['slot_variant']}={forced}")
            else:
                # a verdict is kept per driver and GPU set, so a known box is not probed again
                choice = decide_variant(
                    gpus + self._spare_devices(),
                    cache=Path.home() / ".cache" / "everserve" / "slot-variant.json",
                )
            variant = choice.variant
            self.slot_choice = asdict(choice)  # in the start record, so the run report says it
            self.region = Region.create(path, variant, replace=True)
            log.info("slot collective: %s variant over GPUs %s", variant, gpus)
            # this node's region holds one stage: a row per column, and one for the spare; the
            # kernel sums in column order, which is tensor-parallel rank order (decision 0012)
            tp = self.layout.tensor_parallel
            self.slot_rows = list(range(tp))
            self.region.set_members(self.slot_rows, self.slot_rows)
            self.region.set_members([tp], [tp])  # the spare, alone until it joins
            # the floor until the step period is measured (decision 0021)
            self.region.set_bound(range(tp + 1), self.detect["bound_s"])
            # this process's settings, and the environment vLLM starts its workers with
            settings.export(region=path, layout=(tp, self.layout.pipeline_parallel))
        super()._init_executor()

    def _ask(self) -> None:
        """Ask this node's daemon for node 0's ranks before anything loads (decision 0041).

        A head whose policy is given on its link, not read from a document (the pod contract):
        its node's ranks are a stage like any node's, decided once by the control plane, and the
        connection that asked stays open as their liveness. The engine's policy comes on it before
        the answer, and its placement once every rank of the world is placed.

        The ask carries what this process steps with (its identity's step config): a head that
        resolved its configuration otherwise than its engine is refused there, naming each field.

        Raises:
            RuntimeError: when the node's daemon has no record, or node 0's ranks are another
                process's: a head serves them or nothing; or the coordinator refused it, which it
                says on its link before it raises.
        """
        launched = settings.current()
        record = local_daemon(launched.node, launched.discovery_dir or None)
        if record is None:
            raise RuntimeError(f"no daemon record for node {launched.node}")
        world = self.layout.tensor_parallel * self.layout.pipeline_parallel
        stage = tuple(range(world // max(1, self.parallel_config.nnodes)))
        self.asking = UndecidedConnection(
            record.address,
            launched.engine,
            stage,
            Member(launched.node, 0),
            identity=software_identity(self.vllm_config),
        )
        # the answer first: a refused process is closed on without the policy, which an accepted
        # one heard before its answer (read the other way, a refusal is a closed link)
        try:
            told = self.asking.assigned()
        except Refused as refused:
            self.asking.refused(refused)
            raise RuntimeError(
                f"engine {launched.engine} refused this head: {refused.refusal}"
            ) from refused
        self.given = self.asking.engine_policy()
        if told.role is not Role.WORKER:
            self.asking.close()
            raise RuntimeError(
                f"engine {launched.engine}'s node 0 ranks {list(stage)} are another process's"
            )
        log.info("head: node 0's ranks %s are this process's at epoch %d", list(stage), told.epoch)

    def _policy(self) -> Policy | None:
        """The engine's policy: given on the head's link, else the document the settings name."""
        if self.given is not None:
            return self.given
        launched = settings.current()
        return everserve_policy.load(launched.policy) if launched.policy else None

    def _spare_devices(self) -> list[str]:
        """The devices of this node's spares.

        The policy's on this node where the daemons launch them (decision 0024), else this
        executor's own.
        """
        launched = settings.current()
        policy = self._policy() if launched.engine else None
        if policy is not None:
            spec = policy.engine(launched.engine)
            if spec.spare_launch is not None:
                return [str(s.device) for s in spec.spares if s.node == launched.node]
        return [launched.spare_gpu] if launched.spare_gpu is not None else []

    def compile_or_warm_up_model(self) -> Any:
        """After the engine is warm: start the spare, bootstrap every process, join the daemon.

        Before the warm-up captures its graphs, every worker's NCCL handles are guarded against a
        capture (:func:`~everserve_vllm.worker.guard_engine_groups`).
        """
        # every start-up call is sent: a spare's replay is complete from here
        self.startup.close()
        if settings.current().engine:
            self.collective_rpc(worker.guard_engine_groups)
        out = super().compile_or_warm_up_model()
        if settings.current().engine:
            self._start_everserve()
        return out

    def _start_everserve(self) -> None:
        t0 = time.time()
        launched = settings.current()
        engine, node, directory = launched.engine, launched.node, launched.discovery_dir
        pol = self._policy()
        if pol is None:
            raise RuntimeError(f"engine {engine} has no policy: none given, no document named")
        spec = pol.engine(engine)
        gpu = launched.spare_gpu
        spare_nodes: dict[int, str] = {}
        if not spec.members:
            # placed by what registers (decision 0041): the engine's spares are the framework's
            # group's pods, standing by their own registrations, and where everything runs is the
            # control plane's placement, which comes once every rank of the world is placed
            if self.asking is None:
                raise RuntimeError(f"engine {engine} pins no members and was given no placement")
            self.carrier = True
            cfg = worker.bootstrap_from_placement(
                pol,
                engine,
                node,
                self.asking.engine_placement(),
                event_dir=launched.event_dir,
                hang_s=launched.hang_s,
            )
        else:
            if spec.spare_launch is not None:
                # the daemons launch and own the engine's spares where the policy places them
                # (decision 0024); this executor launches none
                self.carrier = True
                spare_nodes = worker.spare_nodes_of(spec)
            else:
                # one spare of this executor's, built as the rank it stands in for, on this node
                self.carrier = False
                rank = launched.spare_rank if launched.spare_rank is not None else 0
                spare_nodes = {rank: node} if gpu else {}
            cfg = worker.bootstrap_from_policy(
                pol,
                engine,
                node,
                directory,
                spare_nodes,
                event_dir=launched.event_dir,
                hang_s=launched.hang_s,
            )
        sets, spare_ranks = cfg.sets, sorted(spare_nodes)
        if gpu is not None and not self.carrier:
            self.spare = Spare.launch(
                self, spare_ranks[0], int(gpu), row=self.layout.tensor_parallel
            )
        t1 = time.time()
        fut = self.collective_rpc(worker.bootstrap, args=(cfg,), non_block=True)
        if self.spare is not None:
            self.spare.start(cfg)
            self.spare.info["start"] = self.spare.started()
        workers = fut.result()
        record = local_daemon(node, directory)
        if record is None:
            raise RuntimeError(f"no daemon record for node {node} in {directory}")
        self._connect(record.address, engine, node)
        self.started = {
            "spare": self.spare.info if self.spare is not None else None,
            "workers": workers,
            "sets": [list(s) for s in sets],
            "spare_s": round(t1 - t0, 2),
            "bootstrap_s": round(time.time() - t1, 2),
            "slot": self.slot_choice,
            "detect": self.detect,
        }
        log.info("everserve started: %s", self.started)

    def _connect(self, address: str, engine: str, node: str) -> None:
        """Join this node's daemon, as the engine's head when :attr:`HEAD`, and follow its changes.

        The head's connection is the engine's life on the control plane (decision 0032): the
        coordinator holds the engine up while it stands and down once it ends, and a stop reaches
        the process that launched the engine (:func:`~everserve.contract.head.launcher_pid`).
        """
        slot = SlotChoice(**self.slot_choice) if self.slot_choice else None
        self.client = DaemonClient(
            address,
            engine,
            Member(node, 0),
            slot=slot,
            head=self.HEAD,
            launcher_pid=launcher_pid() if self.HEAD else 0,
            identity=software_identity(self.vllm_config),
        )
        self.engine_name = engine
        self._follow_stop = threading.Event()
        if self.HEAD:
            # a spare's ask for what this core sent its rank at its start (decision 0044)
            self.replays = replay.Server(
                self.client, self.startup, settings.current().host_ip or None, self._follow_stop
            )
        threading.Thread(
            target=self._follow_abandoned, daemon=True, name="EverserveAbandonFollower"
        ).start()
        threading.Thread(
            target=follow_changes,
            args=(
                self.client.changes,
                self.dead.__contains__,
                self._coordinator_lost,
                self._follow_stop,
            ),
            kwargs={"on_change": self._claimed_change},
            daemon=True,
            name="EverserveChangeFollower",
        ).start()

    # ------------------------------------------------------------------ death is an event
    def start_worker_monitor(self, inline: bool = False) -> None:
        """Watch the current workers; a death is reported, not a shutdown. Re-armed on a change."""
        if not hasattr(self, "dead"):
            self.dead = set()
        self._monitor_gen = getattr(self, "_monitor_gen", 0) + 1
        gen = self._monitor_gen
        # a position taken through the carrier has no local process to watch; its old handle is
        # a dead process whose sentinel would fire at once
        # nor has one whose process left and is being ended (everserve_vllm.vacate)
        handles = [
            h for h in self.workers if h.rank not in self.taken and h.rank not in self.released
        ]
        self_ref = weakref.ref(self)

        def monitor() -> None:
            procs = {h.proc.sentinel: h for h in handles}
            while procs:  # every death is seen, not only the first
                died = multiprocessing.connection.wait(list(procs))
                s = self_ref()
                if s is None or getattr(s, "shutting_down", False) or s._monitor_gen != gen:
                    return
                for d in died:
                    h = procs.pop(d)
                    h.proc.join(timeout=1.0)  # the exit code is set once the process is reaped
                    s._on_death(h.rank, h.proc.exitcode, gen)
                del s

        threading.Thread(target=monitor, daemon=True, name="EverserveWorkerMonitor").start()

    def _on_death(self, rank: int, exitcode: int | None, gen: int | None = None) -> None:
        """The process sentinel: ``rank``'s worker has exited and been torn down.

        The last signal of a death, after the collective's (decision 0021) and the coordinator's
        (decision 0023), and sometimes after the change itself: a teardown can take seconds, and
        the monitor waits up to a second for the exit code. A sentinel from a monitor generation
        before the current one is about a worker already replaced: its exit is recorded and
        reported at the epoch the rank was lost, which the coordinator folds, and the rank is not
        marked lost again.
        """
        t = time.time()
        replaced = gen is not None and gen != self._monitor_gen
        if rank in self.dead or replaced:
            self.events.append({"kind": "exited", "rank": rank, "exitcode": exitcode, "at": t})
        else:
            self._lost(rank, "sentinel", t)
            self.events[-1]["exitcode"] = exitcode
        if self.client is not None:
            epoch = self.lost_at.get(rank, self.epoch)
            self.client.report_lost([rank], f"exit code {exitcode}", epoch)
        log.warning("worker %d exited (code %s); reported to the daemon", rank, exitcode)

    def _lost(self, rank: int, source: str, t: float) -> None:
        self.dead.add(rank)
        self.lost_at[rank] = self.epoch  # the epoch a later report of this loss is folded on
        if self._is_local(rank):
            self.sweep_dead_reader(rank)  # a remote rank reads the ring by subscription, no slot
        self.events.append({"kind": "rank_lost", "rank": rank, "source": source, "at": t})

    def _first_local(self) -> int:
        """The rank of this node's first worker; ``self.workers[rank - first]`` is its handle."""
        return self.local_world_size * self.parallel_config.node_rank_within_dp

    def _is_local(self, rank: int) -> bool:
        """Whether ``rank``'s worker is one of this node's."""
        first = self._first_local()
        return first <= rank < first + self.local_world_size

    def _follow_abandoned(self) -> None:
        """A change abandoned after its delivery fails the engine: no step runs on from it."""
        assert self.client is not None
        while not self._follow_stop.is_set():
            try:
                gone = self.client.abandoned.get(timeout=0.5)
            except queue.Empty:
                continue
            self.events.append(
                {"kind": "abandoned", "epoch": gone.epoch, "detail": gone.detail, "at": time.time()}
            )
            log.error("the change at epoch %d was abandoned: %s", gone.epoch, gone.detail)
            self._fail(str(ChangeAbandoned(gone)))
            self.abandoned = gone

    def _fail(self, failure: str) -> None:
        """Keep the first failure that ends this engine, for the head's goodbye."""
        if self.failure is None:
            self.failure = failure

    def _claimed_change(self, heard: Any) -> None:
        """Keep the ranks each claimed change took, by epoch; the commit does not repeat them."""
        self.claimed[heard.change.epoch] = tuple(heard.change.lost)
        self.departing.update(heard.change.departing)
        self.claimed_takes[heard.change.epoch] = tuple(heard.change.take)

    def _coordinator_lost(self, rank: int, heard: Any) -> None:
        """The coordinator claimed a change naming ``rank`` (decision 0023): lost from now.

        A rank of this node is seen here first and reported by this executor; the coordinator's
        change for it names a rank already dead and changes nothing. A rank of another node is
        seen only this way: marked dead, so the step in flight raises as it would for a local
        loss. What follows, the join of its spare, is its node's daemon's.
        """
        self._lost(rank, "coordinator", heard.at)

    def _slot_lost(self) -> None:
        """Ranks a survivor's slot watch found lost (decision 0021): dead, or fenced if hung.

        The watching worker has reported the rank to the daemon already. A hung rank's process is
        alive, so it is killed here: it must not run again beside the spare that replaces it.
        """
        if self.region is None:
            return
        # the region names positions in this node's stage; the rank is the stage's first plus it
        first = self.layout.stage(self._first_local()) * self.layout.tensor_parallel
        for rank in {first + p for p in self.region.lost_ranks(self.epoch)} - self._fenced:
            if not self._is_local(rank):
                continue
            if rank not in self.dead:
                self._lost(rank, "slot", time.time())
            # killed whichever signal marked it first: a hung rank must not run on beside its
            # replacement, and the coordinator's word often arrives before this wait reads the mark
            self._fenced.add(rank)
            try:
                os.kill(self.workers[rank - self._first_local()].proc.pid, signal.SIGKILL)
            except (ProcessLookupError, IndexError):
                pass
            log.warning("worker %d lost at the slot collective; its process is killed", rank)

    def _failed(self, step: bool, rank: int, error: Any) -> Exception:
        """The error a worker's failed RPC raises (:func:`~everserve_vllm.losses.failed_step`).

        A worker refuses its groups as soon as its daemon link hears a change, so a step can fail
        on the refusal before the coordinator's word reaches this executor; such a step waits up
        to ``EVERSERVE_HANG_S`` for that loss here and raises as any loss does. vLLM carries the
        failure as the exception's text alone (``PRIVATE-INTERNALS.md`` A8a).
        """

        def lost() -> bool:
            self._slot_lost()
            return bool(self.dead or self.departing)

        err, event = failed_step(
            step,
            rank,
            str(error),
            lost,
            lambda: self.dead | self.departing,
            settings.current().hang_s,
        )
        if event is not None:
            self.events.append(event)
        if not isinstance(err, RankLostError):
            self._fail(str(err))
        return err

    def _time_step(self) -> None:
        """Measure the step period over the first decode steps; then set the detection bound."""
        if self.detect["step_s"] is not None or self.region is None:
            return
        self._steps.append(time.monotonic())
        lo, hi = STEP_SAMPLE
        if len(self._steps) <= hi:
            return
        gaps = sorted(b - a for a, b in zip(self._steps[lo:-1], self._steps[lo + 1 :], strict=True))
        step = gaps[len(gaps) // 2]
        bound = max(self.detect["floor_s"], self.detect["steps"] * step)
        self.detect.update(step_s=round(step, 5), bound_s=round(bound, 5))
        self.region.set_bound(range(self.layout.tensor_parallel + 1), bound)
        log.info(
            "detection bound %.1f ms (%.1f steps of %.2f ms)",
            bound * 1e3,
            self.detect["steps"],
            step * 1e3,
        )

    def sweep_dead_reader(self, rank: int) -> None:
        """Act as ``rank``'s reader of the broadcast ring until the ring is replaced.

        Every written chunk is marked read, so the writer is not blocked once the ring wraps.
        """
        mq = self.rpc_broadcast_mq
        if rank in self._sweeping:
            return
        self._sweeping.add(rank)

        def sweep() -> None:
            buf, i = mq.buffer, rank + 1
            while not getattr(self, "shutting_down", False) and self.rpc_broadcast_mq is mq:
                for idx in range(buf.max_chunks):
                    with buf.get_metadata(idx) as md:
                        if md[0] == 1 and md[i] == 0:
                            md[i] = 1
                time.sleep(0.0005)
            self._sweeping.discard(rank)

        threading.Thread(target=sweep, daemon=True, name=f"EverserveSweep{rank}").start()

    def collective_rpc(  # type: ignore[override]
        self,
        method: Any,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: Any = None,
        ec_output_aggregator: Any = None,
    ) -> Any:
        """Stock ``collective_rpc``, except that a dead worker raises :class:`RankLostError`.

        The stock wait never ends: a survivor blocked in a collective with the dead rank does not
        answer until the daemon's change releases it.
        """
        self._slot_lost()
        if self.dead:
            raise RankLostError(f"worker(s) {sorted(self.dead)} lost")
        if self.departing:
            # a change takes ranks leaving on notice: no step starts until its boundary; the
            # step in flight ended as any does, since nothing was refused (protocol 9)
            raise RankLostError(f"worker(s) {sorted(self.departing)} leaving on notice")
        if self.abandoned is not None:
            raise ChangeAbandoned(self.abandoned)
        if self.is_failed:
            raise RuntimeError("Executor failed.")
        if isinstance(method, str):
            self.startup.record(method, args, kwargs)  # a start-up call, until the warm-up
        if method == "execute_model":
            self._time_step()
        step = method in STEP_METHODS
        if step:
            self._step_sent(1)
        deadline = None if timeout is None else time.monotonic() + timeout
        output_rank, aggregate = _aggregation(
            [a for a in (kv_output_aggregator, ec_output_aggregator) if a], unique_reply_rank
        )
        send = (
            method
            if isinstance(method, str)
            else cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        )
        self.rpc_broadcast_mq.enqueue((send, args, kwargs or {}, output_rank))
        ranks = [output_rank] if output_rank is not None else list(range(self.world_size))
        # with a slot region the wait also reads the watchdogs' lost words, so it wakes often
        poll = 0.02 if self.region is not None else 0.5

        def get_response() -> Any:
            try:
                out = [self._answer(r, method, deadline, poll, step) for r in ranks]
                return out[0] if output_rank is not None else out
            finally:
                if step:
                    self._step_sent(-1)

        future = FutureWrapper(self.futures_queue, get_response=get_response, aggregate=aggregate)
        return future if non_block else future.result()

    def _answer(
        self, rank: int, method: Any, deadline: float | None, poll: float, step: bool
    ) -> Any:
        """Rank ``rank``'s reply to ``method``, or what ended the wait for it.

        Raises:
            RankLostError: a worker died while this waited (the change completes it).
            ChangeAbandoned: the coordinator abandoned the change in flight.
            TimeoutError: ``deadline`` passed; the executor is failed, as stock does.
        """
        while True:
            self._slot_lost()
            if self.dead:
                raise RankLostError(f"worker(s) {sorted(self.dead)} lost")
            if self.abandoned is not None:
                raise ChangeAbandoned(self.abandoned)
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                self._fail(f"RPC call to {method} timed out.")
                raise TimeoutError(f"RPC call to {method} timed out.")
            try:
                status, result = self.response_mqs[rank].dequeue(
                    timeout=poll if left is None else min(poll, left)
                )
            except TimeoutError:
                continue
            if status != WorkerProc.ResponseStatus.SUCCESS:
                raise self._failed(step, rank, result)
            return result

    def _step_sent(self, n: int) -> None:
        """Count a step RPC sent (1) or answered (-1); the region tells the workers' watch."""
        self._steps_out = max(0, self._steps_out + n)
        if self.region is not None:
            self.region.set_in_flight(self._steps_out)

    # ------------------------------------------------------------------ the engine's half
    def serve_steps(self, engine_core: Any, step: Any) -> Any:
        """The engine core's step with this executor's changes completed around it.

        Installed by the plugin (:func:`everserve_vllm.serving.install`) over a serving engine
        core's ``step_fn``, so stock ``vllm serve`` completes changes as the harness driver's
        loop does.
        """
        from everserve_vllm import serving

        return serving.with_changes(self, engine_core, step)

    def pending_change(self) -> Heard | None:
        """A committed membership that has not been completed here yet, without waiting."""
        if self.client is None:
            return None
        try:
            return self.client.committed.get_nowait()
        except queue.Empty:
            return None

    def complete_change(
        self, engine_core: Any, heard: Heard | None = None, timeout_s: float = 120.0
    ) -> dict[str, Any]:
        """The engine-core half of a REPLACE, once the daemon has committed it.

        Called between steps on the engine core's thread. ``heard`` is the commit if the caller
        already took it from :meth:`pending_change`; otherwise this waits for it. The change is
        joined over the carrier where the daemons own the spares (decision 0024), else by this
        executor's own spare; the two share every step but the join itself.
        """
        assert self.client is not None
        t = {"t0": time.time()}
        if heard is None:
            heard = wait_commit(self.client.committed, lambda: self.abandoned, timeout_s)
        t["committed"] = heard.at
        epoch = heard.committed.epoch
        self.epoch = epoch
        took = self.claimed.get(epoch)
        if self.carrier:
            if not took:
                raise RuntimeError(f"no claim recorded for epoch {epoch}")
            return self._complete_over_carrier(engine_core, heard, sorted(took), t, timeout_s)
        assert self.spare is not None
        if took is not None and set(took) != {self.spare.rank}:
            # The local join is this executor's spare's, for the one rank it stands for: the
            # single-node shape. Under decision 0022 a claim takes a whole group, alive ranks
            # included, and the spares that take it are on another node, launched and owned by
            # that node's daemon, joined over the carrier. A change the local shape cannot join
            # is refused with the reason rather than joined wrongly.
            raise RuntimeError(
                f"the change at epoch {epoch} took ranks {sorted(took)}, but this executor's "
                f"spare stands for rank {self.spare.rank}"
            )
        return self._complete_locally(engine_core, heard, t, timeout_s)

    def _complete_over_carrier(
        self, engine_core: Any, heard: Heard, taken: list[int], t: dict, timeout_s: float
    ) -> dict[str, Any]:
        """The change joined through the daemons (decision 0024).

        The same boundary as the local join, for any number of positions, local or remote: a new
        ring whose every reader on another process subscribes over the network, the survivors
        applying the change over the old ring, one join offered per position through the daemon
        and the coordinator, every answer installed at once, then the replay.
        """
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        from everserve_vllm import join as joins

        assert self.client is not None
        epoch = heard.committed.epoch
        faulted = [r for r in taken if r in self.dead]
        survivors = [r for r in range(self.world_size) if r not in taken]
        tag = f"change-{epoch}-{time.time()}"
        self._cut_step(engine_core)
        # the spares read the ring over the network wherever they run: one path for every
        # position, local or not; the local survivors keep their shared-memory slots
        ring = self._new_ring([r for r in survivors if self._is_local(r)])
        if self.region is not None:
            worker.take_rows(self.layout, self._first_local(), self.slot_rows, taken)
        self._send_boundary(ring, tag, tuple(taken))
        offered = ring.export_handle()
        # the calls this core's workers were built with, which a spare's replay must be
        startup = replay.digest(self.startup.complete())
        for r in taken:
            offer = joins.Offer(r, epoch, tag, offered, faulted=r in faulted, startup=startup)
            self.client.join(r, epoch, joins.encode(offer))
        t["join_sent"] = time.time()  # the offers are on the control plane (0024's carrier)
        # a remote leaving rank answers its own node; a dead one answers nobody
        drained = {
            r: self._tagged_reply(r, tag, timeout_s)
            for r in taken
            if r not in faulted and self._is_local(r)
        }
        answers = self._join_answers(taken, tag, timeout_s)
        t["join_answered"] = time.time()  # every spare's answer is back through the daemons
        ring.wait_until_ready()
        t["ring_subscribed"] = time.time()  # the new broadcast queue has every reader
        responses = {r: MessageQueue.create_from_handle(a.response, 0) for r, a in answers.items()}
        for mq in responses.values():
            mq.wait_until_ready()
        t["ring_ready"] = time.time()  # and every spare's response queue has this reader
        boundaries = self._boundaries(survivors, tag, timeout_s)
        t["boundary"] = time.time()
        self._reset_region()
        for r, a in answers.items():
            self.taken[r] = {"epoch": epoch, "answer": a}
        self._install_ring(ring, responses)
        # every spare restored its rank's rows through the same commits; resume from the least
        safe: dict[str, int] = {}
        for a in answers.values():
            for rid, n in ((a.restore or {}).get("counts") or {}).items():
                safe[rid] = min(safe.get(rid, n), n)
        resumed = self._resume(engine_core, safe)
        return {
            "ranks": taken,
            "faulted": faulted,
            "spare_epochs": {
                r: (a.applied or {}).get("moved", {}).get("epoch") for r, a in answers.items()
            },
            "spares": {
                r: {"restore": a.restore, "join_s": a.join_s, "attachment": a.attachment}
                for r, a in answers.items()
            },
            "drained": drained,
            "t_boundary": t["boundary"],
            # the join's legs on the carrier path, from the commit: the offers out, every
            # answer back, the new ring subscribed (the broadcast queue's readers, then the
            # spares' response queues: 1 ms on one join and 255 ms on another), then the
            # survivors' boundary
            "join_legs_s": {
                "offers_out": round(t["join_sent"] - t["committed"], 4),
                "answers_back": round(t["join_answered"] - t["join_sent"], 4),
                "ring_subscribed": round(t["ring_subscribed"] - t["join_answered"], 4),
                "ring_ready": round(t["ring_ready"] - t["ring_subscribed"], 4),
                "survivors_boundary": round(t["boundary"] - t["ring_ready"], 4),
            },
            # how long after the offers each local survivor entered its boundary (one clock:
            # the executor's node), the leg the new ring's subscription waits on first
            "survivor_lag_s": {
                r: round(b["t_entered"] - t["join_sent"], 4)
                for r, b in boundaries.items()
                if isinstance(b, dict) and "t_entered" in b and self._is_local(r)
            },
            **self._replaced(heard, t, taken, faulted, boundaries, safe, resumed),
        }

    def _join_answers(self, taken: list[int], tag: str, timeout_s: float) -> dict[int, Any]:
        """Every taken position's answer to the offers named ``tag``, back through the daemons.

        Raises:
            RuntimeError: a position's answer did not come in ``timeout_s``, or said FAILED.
        """
        from everserve_vllm import join as joins

        assert self.client is not None
        answers: dict[int, joins.Answer] = {}
        deadline = time.monotonic() + timeout_s
        while len(answers) < len(taken):
            left = deadline - time.monotonic()
            if left <= 0:
                raise RuntimeError(f"no join answer from {sorted(set(taken) - set(answers))}")
            sig = self.client.joins.get(timeout=left)
            ans = joins.decode(sig.payload)
            if not isinstance(ans, joins.Answer) or ans.tag != tag:
                continue  # an answer to another boundary; this one's are still to come
            if ans.status != "READY":
                raise RuntimeError(f"rank {ans.rank}'s spare failed to join: {ans.error}")
            answers[ans.rank] = ans
        return answers

    def _complete_locally(
        self, engine_core: Any, heard: Heard, t: dict, timeout_s: float
    ) -> dict[str, Any]:
        """The change joined by this executor's own spare: the single-node shape.

        One position, the ring handed to the spare over its pipe, its response queue in the
        leaving worker's place.
        """
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
        from vllm.v1.executor.multiproc_executor import WorkerProcHandle

        assert self.spare is not None
        epoch = heard.committed.epoch
        rank = self.spare.rank
        faulted = rank in self.dead
        survivors = [r for r in range(self.world_size) if r != rank]
        tag = f"change-{epoch}-{time.time()}"
        self._cut_step(engine_core)
        ring = self._new_ring(list(range(self.local_world_size)))
        if self.region is not None:
            # the spare's row takes the replaced rank's position in the sum, so the sum after the
            # change is the one before it (decision 0012); every member writes its own row's order
            # at the boundary, this is the executor's record of it
            self.slot_rows[self.layout.column(rank)] = self.layout.tensor_parallel
        self._send_boundary(ring, tag, (rank,))
        drained = None
        if not faulted:
            # a notice: the leaving rank is alive and drains its last commits at the boundary; the
            # spare's catch-up restore at the join must find them in the store
            drained = self._tagged_reply(rank, tag, timeout_s)
        ready = self.spare.join(ring.export_handle(), ring.wait_until_ready)
        resp_mq = MessageQueue.create_from_handle(ready["handle"], 0)
        resp_mq.wait_until_ready()
        boundaries = self._boundaries(survivors, tag, timeout_s)
        t["boundary"] = time.time()
        self._reset_region()
        leaving = self.workers[rank]
        self.workers[rank] = WorkerProcHandle(
            proc=self.spare.proc,
            rank=rank,
            worker_response_mq=resp_mq,
            peer_worker_response_mqs=[],
            death_writer=self.spare.death_writer,
        )
        self._install_ring(ring, {rank: resp_mq})
        # the replaced rank's own counts: its restored rows are complete exactly through them
        safe = dict((ready.get("restore") or {}).get("counts", {}))
        resumed = self._resume(engine_core, safe)
        return {
            "rank": rank,
            "faulted": faulted,
            "leaving_pid": leaving.proc.pid,
            "spare_epoch": (ready.get("applied") or {}).get("moved", {}).get("epoch"),
            "spare": {k: v for k, v in ready.items() if k != "handle"},
            "leaving_drained": drained,
            **self._replaced(
                heard, t, [rank], [rank] if faulted else [], boundaries, safe, resumed
            ),
        }

    # ------------------------------------------------------------ the steps both joins share
    def _cut_step(self, engine_core: Any) -> None:
        """Drop the step the change cut: its RPCs are never answered, so none is in flight."""
        if engine_core.batch_queue is not None:
            engine_core.batch_queue.clear()
        self.futures_queue.clear()
        self._steps_out = 0
        if self.region is not None:
            self.region.set_in_flight(0)

    def _new_ring(self, local: list[int]) -> Any:
        """A broadcast ring for the world after the change.

        ``local`` ranks read it over shared memory; every other reader, a spare wherever it
        runs, subscribes over the network at this node's address.
        """
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        return MessageQueue(
            self.world_size,
            len(local),
            local_reader_ranks=local,
            max_chunk_bytes=chunk_bytes(),
            connect_ip=settings.current().host_ip or None,
        )

    def _send_boundary(self, ring: Any, tag: str, leaving: tuple[int, ...]) -> None:
        """Ask every worker to the boundary: apply the change and attach to ``ring``.

        A ``leaving`` one drains instead (:func:`worker.boundary_except`). The reply carries
        ``tag``, which tells it from the cut step's output.
        """
        fn = cloudpickle.dumps(worker.boundary_except, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((fn, (ring.export_handle(), tag, leaving), {}, None))

    def _tagged_reply(self, rank: int, tag: str, timeout_s: float) -> dict[str, Any]:
        """``rank``'s reply to the boundary named ``tag``, past the cut step's stale outputs."""
        while True:
            _status, out = self.response_mqs[rank].dequeue(timeout=timeout_s)
            if isinstance(out, dict) and out.get("tag") == tag:
                return out

    def _boundaries(
        self, survivors: list[int], tag: str, timeout_s: float
    ) -> dict[int, dict[str, Any]]:
        """Each survivor's record of the boundary named ``tag``, in rank order."""
        return {r: self._tagged_reply(r, tag, timeout_s) for r in survivors}

    def _reset_region(self) -> None:
        """Every member zeroed its own words and counters at the boundary; the watch too."""
        if self.region is not None:
            rows = range(self.layout.tensor_parallel + 1)
            self.region.clear_watch(rows)
            self.region.set_abort(list(rows), 0)

    def _install_ring(self, ring: Any, responses: dict[int, Any]) -> None:
        """Put ``ring`` and each taken position's response queue in the old ones' place.

        The world is whole again and the monitor watches the new members. The replaced ring and
        the taken positions' old queues are closed now, never left to the collector, whose close
        waits on anything a gone reader never took.
        """
        close_queue(self.rpc_broadcast_mq)
        self.rpc_broadcast_mq = ring
        for r, mq in responses.items():
            close_queue(self.response_mqs[r])
            self.response_mqs[r] = mq
            self._fenced.discard(r)  # the position is the spare's now; a later loss fences again
        self.dead.clear()
        self.departing.clear()
        self.start_worker_monitor()

    def _resume(self, engine_core: Any, safe: dict[str, int]) -> list[Any]:
        """Re-add the running requests from their committed counts.

        Every worker, the spares included, re-adds them from the same state; a request past one
        speculative token replays one.
        """
        resumed = resume_running(engine_core.scheduler, safe)
        self.collective_rpc(worker.resume_requests, args=(resumed,))
        if any(
            r.num_tokens_with_spec - r.num_computed_tokens > 1
            for r in engine_core.scheduler.running
        ):
            OneTokenReplay(engine_core.scheduler, [d.req_id for d in resumed])
        return resumed

    def _replaced(
        self,
        heard: Heard,
        t: dict,
        taken: list[int],
        faulted: list[int],
        boundaries: dict[int, dict[str, Any]],
        safe: dict[str, int],
        resumed: list[Any],
    ) -> dict[str, Any]:
        """The record every completed change carries, and the event of each position taken."""
        t["resumed"] = time.time()
        for r in taken:
            self.events.append(
                {"kind": "replaced", "rank": r, "at": t["resumed"], "faulted": r in faulted}
            )
        return {
            "epoch_committed": heard.committed.epoch,
            "epochs_applied": {
                r: (b.get("moved") or {}).get("epoch") for r, b in boundaries.items()
            },
            "boundaries": boundaries,
            "safe_tokens": safe,
            "slot_order": list(self.slot_rows) if self.region is not None else None,
            "n_replayed": len(resumed),
            "t_committed": t["committed"],
            "t_resumed": t["resumed"],
            "wait_commit_s": round(t["committed"] - t["t0"], 4),
            "boundary_s": round(t["boundary"] - max(t["t0"], t["committed"]), 4),
            "resume_s": round(t["resumed"] - t["boundary"], 4),
            "claim_to_commit": claim_to_commit(heard.committed, heard.at),
        }

    def _stop_watches(self) -> None:
        """Stop every worker's slot watch over the ring; a node without the ring uses the region."""
        if self.dead:
            return  # a dead rank never answers; the workers are stopped by the kill
        try:
            self.collective_rpc(worker.stop_watch, timeout=5.0)
        except Exception as e:  # noqa: BLE001 - shutting down regardless
            log.warning("could not stop the workers' slot watch: %s", e)

    def worker_stats(self) -> Any:
        """Every worker's commit times and runtime events."""
        return self.collective_rpc(worker.stats)

    def _leave(self) -> None:
        """End this process's link to its daemon.

        The head's goodbye says why the engine stops; without it the engine reads as lost (0032).
        Any other executor's link just closes: its leaving is its workers', not the engine's.
        """
        assert self.client is not None
        if self.HEAD:
            self.client.goodbye(*stop_cause(getattr(self, "failure", None)))
        else:
            self.client.close()

    def shutdown(self) -> None:
        """Stop the slot watch, an unjoined spare and the daemon link; remove the region.

        The watch stops first: a worker that exits during shutdown must not be reported lost by
        the others. Nothing is in flight from here, whatever a death cut short, so no stall scan
        runs even where the stop cannot be sent (a rank already dead). A spare that never joined
        is no member of any worker's order, so no watch reads it.
        """
        self._steps_out = 0
        if getattr(self, "region", None) is not None:
            self.region.set_in_flight(0)
        if getattr(self, "_follow_stop", None) is not None:
            self._follow_stop.set()
        if getattr(self, "client", None) is not None:
            self._stop_watches()
        if getattr(self, "spare", None) is not None and self.spare.proc not in [
            h.proc for h in self.workers
        ]:
            self.spare.shutdown()
        if getattr(self, "client", None) is not None:
            self._leave()
        if getattr(self, "asking", None) is not None:
            self.asking.close()  # node 0's ranks are released with the head
        super().shutdown()
        region = getattr(self, "region", None)
        if region is not None:
            try:
                os.unlink(region.path)
            except OSError:
                pass
