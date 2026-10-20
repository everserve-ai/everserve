# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The spare: a replacement process, pre-loaded on the spare GPU, that takes a rank's place.

Started by the executor with the engine (:class:`Spare`). It builds the engine's worker for rank
``R`` against stand-in parallel groups (it is in no group yet), so layers shard and load exactly
as rank ``R``; allocates rank ``R``'s KV from the engine's KV config; and warms up and captures
its CUDA graphs. Its collective starts on a local stand-in at its own slot row, so the captured
graphs run the slot kernel alone; they stay valid when the row's mask later names its peers.
At :meth:`Spare.start` it forms every replacement set beside the serving workers and registers
with the node daemon as the spare for rank ``R``, which is what makes REPLACE available.

On ``Change(REPLACE)`` for its rank, before acknowledging, it restores the dead rank's committed
rows from the mirror with the identity move table (:func:`moves`) and keeps the per-request
committed counts read with them, and the watermark: the rank's last committed step. The engine's
half then runs (:mod:`everserve_vllm.executor`): at the boundary the spare restores again, only
what the rank committed after the watermark (:meth:`SpareRuntime.catch_up`), applies the change,
attaches to the executor's new ring, and answers as rank ``R``; the scheduler replays from the
counts the tail the rows do not hold. After a fault the second restore finds little or nothing.
On a notice the leaving rank kept serving between the change and the swap, and has drained its
last commits to the store before the spare joins, so the tail is about one position.

Private vLLM lines are listed in ``PRIVATE-INTERNALS.md`` (stand-in groups, the worker process
subclass, the warm-up sequence).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from everserve.contract.signals import Role
from everserve.outcome import Outcome
from everserve.state.kvcopy import HeadSlot
from everserve_vllm import replay
from everserve_vllm.worker import AdapterRuntime, Bootstrap

log = logging.getLogger(__name__)


def moves(rank: int, heads: int) -> list[tuple[tuple[int, int], HeadSlot, HeadSlot]]:
    """The spare's move table: every unit of ``rank``, from its head slot to the same head slot."""
    return [((rank, h), h, h) for h in range(heads)]


STAGED_WAIT_S = 30.0
"""How long a restore waits for the rank's rows to be staged onto this node (decision 0030)."""

STAGED_POLL_S = 0.005


def wait_staged(
    kv: Any, rank: int, timeout_s: float = STAGED_WAIT_S, poll_s: float = STAGED_POLL_S
) -> dict[str, Any]:
    """Wait until this node's store holds ``rank``'s rows, caught up; the state seen.

    Held, and their source's carrier has said it is caught up through the step the store
    holds, or ``timeout_s``. A shared spare's node holds no rows of the engine until the
    coordinator stages the lost rank's rows onto it before the change goes out (decision 0030);
    a per-position spare's node holds them by continuous replication and reads the same two
    flags. The restore reads the rows only once both hold, so what it copies is what the rank
    committed through that step.

    Raises:
        TimeoutError: the rows did not arrive; names the state seen, which the change's
            acknowledgement carries as the failure.
    """
    t0 = time.time()
    polls = 0
    while True:
        held, epoch, step, caught_up = kv.state(rank)
        polls += 1
        if held and caught_up:
            return {
                "epoch": epoch,
                "step": step,
                "s": round(time.time() - t0, 4),
                "polls": polls,
            }
        if time.time() - t0 >= timeout_s:
            raise TimeoutError(
                f"rank {rank}'s rows did not arrive at this node's store in {timeout_s:g} s: "
                f"held={held} epoch={epoch} step={step} caught_up={caught_up} "
                f"after {polls} polls"
            )
        time.sleep(poll_s)


class SpareRuntime(AdapterRuntime):
    """The spare's runtime: a REPLACE of its rank restores the rows before it is acknowledged.

    The restore waits for the rank's rows to be held and caught up at this node's store first
    (:func:`wait_staged`); a wait past ``staged_wait_s`` fails the change with the state seen.
    """

    restore_info: dict[str, Any] | None = None
    staged_wait_s: float = STAGED_WAIT_S
    took_epoch: int | None = None
    """The epoch of the change this spare took its position at, once it has applied it."""

    def on_change(
        self,
        lost: Sequence[int],
        sent_at: float | None = None,
        outcome: Outcome = Outcome.SHRINK,
        epoch: int | None = None,
        departing: Sequence[int] = (),
    ) -> None:
        """Stage the change; restore this rank's rows if it is among the lost under REPLACE.

        A change delivered again at the epoch this spare already took its position at (a
        coordinator that restarted before the commit re-delivers the change it logged) is
        acknowledged as before and restores nothing again: the rows are restored once.
        """
        if epoch is not None and self.took_epoch == epoch:
            return  # re-delivered: staged and restored already; the link acknowledges again
        super().on_change(lost, sent_at, outcome, epoch, departing)
        if outcome is not Outcome.REPLACE or self.orig_rank not in lost:
            return
        self.took_epoch = self.identity()[1]
        dead = self.orig_rank
        from everserve_vllm import worker

        assert worker.STATE is not None
        kv = worker.STATE.kv
        heads = worker.head_slots(dict(kv.layers))
        staged = wait_staged(kv, dead, self.staged_wait_s)
        t0 = time.time()
        watermark = kv.restore(dead, moves(dead, heads))
        info: dict[str, Any] = {
            "restore_s": round(time.time() - t0, 4),
            "timing": dict(getattr(kv, "restore_timing", {})),
            "staged": staged,
            "counts": dict(kv.restored_counts),
            "rows": kv.restored_rows,
            "watermark": watermark,
            "dead": dead,
            "heads": heads,
        }
        self.restore_info = info

    def on_abandoned(self, epoch: int, detail: str) -> None:
        """A spare that took its position at the abandoned change stops; its daemon relaunches it.

        The membership it took a position in will never commit (decision 0032), and a spare
        holding one is no spare. It stops as on any stop (``SIGTERM``, the spare's own close,
        ``spare_main.until_stopped``), and its daemon starts a standing one in its place. A
        spare still standing is not in the change and stays.
        """
        super().on_abandoned(epoch, detail)
        if self.took_epoch is not None and self.took_epoch == epoch:
            log.error("this spare took its position at abandoned epoch %d: stopping", epoch)
            os.kill(os.getpid(), signal.SIGTERM)

    def catch_up(self) -> None:
        """At the swap: restore what the replaced rank committed after the first restore.

        Called once the leaving rank's last commits are in the store (on a notice it drains them at
        the boundary; a dead rank commits nothing more). Only slots written after the watermark are
        copied, and the counts become the rank's final committed counts, so the replay after the
        swap starts at most a position or two behind.
        """
        info = self.restore_info
        if info is None:
            return
        from everserve_vllm import worker

        assert worker.STATE is not None
        kv = worker.STATE.kv
        t0 = time.time()
        before = dict(info["counts"])
        watermark = kv.restore(
            info["dead"], moves(info["dead"], info["heads"]), since_step=info["watermark"]
        )
        info["catch_up"] = {
            "since_step": info["watermark"],
            "watermark": watermark,
            "rows": kv.restored_rows,
            "timing": dict(getattr(kv, "restore_timing", {})),
            "counts_before": before,
            "s": round(time.time() - t0, 4),
            "t_end": time.time(),  # after the restore's device synchronize
        }
        info["counts"] = dict(kv.restored_counts)
        info["watermark"] = watermark


# ----------------------------------------------------------------------- inside the spare process
def _install_stubs(
    vllm_config: Any,
    rank: int,
    distributed_init_method: Any = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Stand-in for ``init_worker_distributed_environment`` in a process in no group yet.

    Groups that report rank ``rank`` of the world, so layers shard and load as that rank
    (:func:`point_groups`), and the serving tactics and the stage-alone warm-ups installed once.
    """
    point_groups(vllm_config, rank, local_rank)
    _take_serving_tactics()
    _warm_pipeline_stage()


def point_groups(vllm_config: Any, rank: int, local_rank: int = -1) -> None:
    """The parallel groups as local stand-ins that report rank ``rank`` of the world.

    Collectives through them are local stand-ins and their outputs are discarded. A spare
    resident in host memory points them at the position it takes before it builds that
    position's model (:func:`attach` with a shard), so the stage's layers and the rank's slices
    load from the copy as they would from disk.
    """
    import weakref

    import torch
    from vllm.distributed import parallel_state as ps

    world = vllm_config.parallel_config.world_size

    class _Size:
        def __init__(self, n: int) -> None:
            self.n = n

        def size(self) -> int:
            return self.n

        def rank(self) -> int:
            return 0

    def stub(name: str, ranks: Any, me: int) -> Any:
        g = ps.GroupCoordinator.__new__(ps.GroupCoordinator)
        g.unique_name, g.ranks, g.world_size = f"stub-{name}", list(ranks), len(ranks)
        g.rank, g.rank_in_group, g.local_rank = rank, me, local_rank
        g.device_communicator, g.mq_broadcaster, g.use_device_communicator = None, None, False
        g.cpu_group, g.device_group = _Size(len(ranks)), _Size(len(ranks))
        g.device = torch.device(f"cuda:{local_rank}")
        g.broadcast_object = lambda obj=None, src=0: obj  # the connector syncs its engine id
        g.barrier = lambda: None  # a process alone has no one to wait for
        return g

    # the layout: the tensor-parallel group is this rank's row of the stage, the pipeline group
    # its column. A TP stub spanning the whole world would, under TP1 x PP2, shard attention for
    # two tensor-parallel ranks: half the KV heads, half the weights, a restore of half the head
    # slots, and every session diverging at the first token after the join.
    from everserve_vllm import settings

    tp_size, pp_size = settings.current().layout or (world, 1)
    column, stage = rank % tp_size, rank // tp_size
    row = [stage * tp_size + c for c in range(tp_size)]
    ps._WORLD = stub("world", range(world), rank)
    tp = stub("tp", row, column)
    tp.unique_name = "tp:0"  # compiled code finds the TP group by this name at call time
    tp.use_custom_op_call = True
    tp._all_reduce_out_place = lambda x: x.clone()
    tp._all_gather_out_place = lambda x, dim: torch.cat([x] * tp_size, dim)
    tp._reduce_scatter_out_place = lambda x, dim: x.chunk(tp_size, dim)[0].contiguous()
    ps._TP = ps._ETP = tp
    ps._groups[tp.unique_name] = weakref.ref(tp)
    ps._EP = stub("ep", row, column)  # expert parallelism spans the tensor-parallel ranks
    # the pipeline group carries this spare's stage: the column's ranks in stage order, so the
    # model shards the stage's layers and the runner knows whether it is first or last; its
    # traffic goes over the fabric family (everserve_vllm.pipeline), not through torch
    pp = stub("pp", [s * tp_size + column for s in range(pp_size)], stage)
    pp.unique_name = "pp:0"  # the pipeline routing finds PP groups by this prefix
    pp.group_ranks = [pp.ranks]
    if hasattr(ps, "_PP"):
        ps._PP = pp
    for name in ("_PCP", "_DCP", "_DP", "_EPLB", "_ENGRAM_DP"):
        if hasattr(ps, name):
            setattr(ps, name, stub(name, [rank], 0))
    if hasattr(ps, "_NODE_COUNT"):
        ps._NODE_COUNT = 1


def _warm_pipeline_stage() -> None:
    """A spare of a pipeline-parallel engine runs vLLM's kernel warm-ups on its stage alone.

    Both of them (``kernel_warmup`` and the V2 runner's ``warmup_kernels``, bound in the worker
    module) drive the worker's ``execute_model`` with synthetic steps that a spare has no peer
    for; :func:`everserve_vllm.warmup.install` wraps them to run the stage alone (A38), over the
    log-prob passes the plugin put on (A41), each timed as a part of the warm-up. Skipping the
    warm-ups costs the first step after a resume ~0.6 s on a node whose Triton cache has never
    compiled them (eight kernels of the V2 runner's input batch, block table, buffer writes and
    sampler).
    """
    try:
        from vllm.v1.worker import gpu_worker
    except ImportError:
        return
    from everserve_vllm import warmup

    warmup.install(gpu_worker, stage_alone=True, timed=_timed_part)


def _take_serving_tactics() -> None:
    """Make the spare's kernel warm-up load the serving ranks' FlashInfer tactics, not tune.

    On Hopper, vLLM's warm-up autotunes FlashInfer across the world group, averaging each
    tactic's timing over the ranks so all of them pick the same one, and the leader saves the
    choice. The spare is in no group: tuning alone, it could pick another tactic than the rank
    it replaces, and different kernels can round differently. It loads the saved choice instead.
    """
    try:
        from vllm.model_executor.warmup import kernel_warmup as kw
    except ImportError:
        return
    if not hasattr(kw, "flashinfer_autotune"):
        return

    def from_cache(runner: Any) -> None:
        from flashinfer.autotuner import AutoTuner

        path = kw.resolve_flashinfer_autotune_file(runner)
        if path.exists():
            AutoTuner.get().load_configs(str(path))
        log.info(
            "spare: FlashInfer tactics from %s (%s)",
            path,
            "loaded" if path.exists() else "none saved",
        )

    kw.flashinfer_autotune = from_cache


class _ShadowDeviceCommunicator:
    """The spare's TP device communicator once it joins: the replacement set's NCCL handle."""

    def __init__(self, comm: Any, world: int) -> None:
        self.pynccl_comm, self.world = comm, world

    def all_gather(self, x: Any, dim: int = -1) -> Any:
        import torch

        if dim < 0:
            dim += x.dim()
        out = torch.empty((self.world, *x.shape), dtype=x.dtype, device=x.device)
        self.pynccl_comm.all_gather(out, x.contiguous())
        return out.movedim(0, dim).reshape(
            x.shape[:dim] + (self.world * x.shape[dim],) + x.shape[dim + 1 :]
        )


def _point_stub_at(group: Any) -> None:
    """In the spare: its stub TP group talks to its peers through the replacement set.

    A tensor-parallel group of one has no peers and no communicator (a ``LocalStandIn``): the
    stub's collectives stay the local copies they were before the join.
    """
    from vllm.distributed import parallel_state as ps

    from everserve.comm.handle import LocalStandIn

    if isinstance(group, LocalStandIn):
        return
    tp = ps._TP
    dc = _ShadowDeviceCommunicator(group.group, tp.world_size)
    tp.device_communicator = dc
    tp._all_gather_out_place = lambda x, dim: dc.all_gather(x, dim)


WARM_PARTS: dict[str, float] = {}
"""The parts of the last warm-up (:func:`_warm`), seconds each: ``compile_sizes_s`` (vLLM's dummy
runs at the compile sizes), ``kernel_warmup_s`` and ``warmup_kernels_s`` (its kernel warm-ups),
``capture_s`` (the CUDA graph capture), each with the dummy runs it makes itself. Split, since
the whole warm-up reads 3.4 to 3.6 s a take on H100, 5.1 to 5.3 s at 32k context, and the parts
say where it goes."""

_IN_PART = threading.local()


@contextlib.contextmanager
def _timed_part(name: str) -> Iterator[None]:
    """Add the block's seconds to ``WARM_PARTS[name]``.

    A block inside another timed part is that part's: the graph capture's own dummy runs are the
    capture's, a kernel warm-up's are the warm-up's, and no second counts twice.
    """
    if getattr(_IN_PART, "on", False):
        yield
        return
    _IN_PART.on = True
    t = time.time()
    try:
        yield
    finally:
        _IN_PART.on = False
        WARM_PARTS[name] = round(WARM_PARTS.get(name, 0.0) + time.time() - t, 3)


def _timing(name: str, fn: Any) -> Any:
    """``fn`` with its seconds added to the warm-up's part ``name``."""

    def timed(*args: Any, **kwargs: Any) -> Any:
        with _timed_part(name):
            return fn(*args, **kwargs)

    return timed


@contextlib.contextmanager
def _null_block_zeroed_after(model_runner: Any) -> Iterator[None]:
    """Zero the KV cache's null block (block 0) once the body ends, as a serving runner keeps it."""
    try:
        yield
    finally:
        zeroer = getattr(model_runner, "kv_block_zeroer", None)
        if zeroer is not None:
            zeroer.zero_block_ids([0])


def _warm(worker: Any) -> tuple[float, float]:
    """Size the workspace with one max-size dummy run, then warm the kernels and capture the graphs.

    vLLM's elastic-EP warm-and-capture sequence. The spare warms right after its KV cache is
    allocated, before any request or restore, so there is no request state to keep out of it:
    that is asserted, not assumed. The warm-up's dummy runs may write the null block, which a
    serving runner keeps zeroed, so block 0 is zeroed after (what vLLM main's
    ``preserve_serving_state`` does, not in the 0.31.0 release this adapter pins). Returns the
    sizing run's seconds and the warm-up's (vLLM's ``compile_or_warm_up_model``: the kernel
    warm-ups, on this stage alone past the first, and the graph capture).
    """
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

    t = time.time()
    w = worker.worker if hasattr(worker, "worker") else worker
    mr = w.model_runner
    live = list(getattr(getattr(mr, "req_states", None), "req_id_to_index", {}) or ())
    if live:
        raise RuntimeError(f"a spare warms before it serves; it holds requests {live}")
    unlock_workspace()
    with set_current_vllm_config(w.vllm_config), _null_block_zeroed_after(mr):
        mr._dummy_run(
            mr.max_num_tokens,
            is_profile=True,
            skip_eplb=True,
            randomize_inputs=getattr(w, "randomize_dummy_inputs", False),
        )
        sized = time.time()
        WARM_PARTS.clear()
        # timed on this runner alone, for the warm-up only: the instance attributes shadow the
        # class's methods and are removed after
        mr._dummy_run = _timing("compile_sizes_s", mr._dummy_run)
        mr.capture_model = _timing("capture_s", mr.capture_model)
        from everserve_vllm.worker import guard_engine_groups

        guard_engine_groups()  # before the capture, as the executor does for its workers
        try:
            w.compile_or_warm_up_model()
        finally:
            del mr._dummy_run, mr.capture_model
    lock_workspace()
    return sized - t, time.time() - sized


@dataclass(frozen=True)
class Attachment:
    """A spare warm for one position of one engine: what it registers for, and only that.

    A spare's warm-up is a property of its attachment (engine, rank), not of the process: the
    KV allocated at the engine's configuration, the workspace sized, the kernels of the rank's
    pipeline stage compiled and its graphs captured, before the registration for that engine
    goes out. Today's preloaded spare attaches once at start; a spare resident in host memory
    (decision 0028) attaches when it takes a shard, per engine, and registers for each engine
    it stands ready for after warming for it.
    """

    engine: str
    rank: int
    stage: int
    sizing_s: float
    warm_s: float
    shard_copy_s: float = 0.0
    """The build of the position's model from the host copy, resident mode (else zero): the
    modules' construction and ``weights_s``."""
    weights_s: float = 0.0
    """Of ``shard_copy_s``, the weights' load from the host copy (the resident loader's own)."""
    kv_config_s: float = 0.0
    """The core's start-up calls replayed before the allocation: the layout, the memory
    profiling, the model length (:func:`everserve_vllm.replay.apply`)."""
    kv_alloc_s: float = 0.0
    """The KV cache's allocation at the core's configuration for the rank."""
    kv_config_parts: dict[str, Any] = field(default_factory=dict)
    """Each replayed call's seconds, by method (``<method>_s``)."""
    warm_parts: dict[str, float] = field(default_factory=dict)
    """``warm_s`` by part (:data:`WARM_PARTS`)."""


def attach(
    proc: Any, engine: str, rank: int, startup: replay.Calls, shard: Any = None
) -> Attachment:
    """Warm the loaded spare for rank ``rank`` of ``engine``; returns the attachment.

    The core's start-up calls replayed on its worker (``startup``, decision 0044: the KV cache
    layout, the memory profiling, the model length and the rank's KV cache configuration, as
    the core sent its workers), then the sizing run, the stage's kernel warm-ups and the graph
    capture. With ``shard`` (the :class:`~everserve_vllm.resident.HostCopy` the process holds)
    the spare is resident: it built no model at start, so the attachment first points the
    stand-in groups at ``rank`` and builds the position's model from the copy, the shard copied
    to the device (``shard_copy_s``).

    Raises:
        replay.ReplayRefused: for calls this worker cannot run, a KV cache that does not fit
            named.
    """
    shard_copy_s = weights_s = 0.0
    if shard is not None:
        w = proc.worker.worker
        from everserve_vllm import settings

        settings.update(spare_rank=rank)
        point_groups(w.vllm_config, rank, w.local_rank)
        proc.rank = w.rank = rank
        t = time.time()
        proc.worker.load_model()
        shard_copy_s = time.time() - t
        from everserve_vllm import resident

        weights_s = float(resident.last_load().get("s") or 0.0)
        log.info(
            "rank %d's shard built from the host copy in %.2f s (weights %.2f s)",
            rank,
            shard_copy_s,
            weights_s,
        )
    t = time.time()
    kv_config_parts: dict[str, Any] = {}
    replay.apply(proc.worker, rank, startup, kv_config_parts)
    kv_alloc_s = float(kv_config_parts.get("initialize_from_config_s", 0.0))
    kv_config_s = max(0.0, time.time() - t - kv_alloc_s)
    WARM_PARTS.clear()  # this attachment's parts, never an earlier warm-up's
    sizing_s, warm_s = _warm(proc.worker)
    try:
        from vllm.distributed import parallel_state as ps

        stage = int(ps.get_pp_group().rank_in_group)
    except AssertionError:  # vLLM's word for a pipeline group never initialised: one stage
        stage = 0
    return Attachment(
        engine,
        rank,
        stage,
        round(sizing_s, 3),
        round(warm_s, 3),
        round(shard_copy_s, 3),
        round(weights_s, 3),
        round(kv_config_s, 3),
        round(kv_alloc_s, 3),
        kv_config_parts,
        dict(WARM_PARTS),
    )


def register(
    attachment: Attachment, proc: Any, cfg: Any, slot_choice: Any = None
) -> dict[str, Any]:
    """Register the spare with its daemon for the attachment's position.

    ``worker.start`` as a spare for ``attachment.rank``, the compile cache pair read off the
    loaded model.

    Raises:
        RuntimeError: ``cfg`` bootstraps another engine than the one the spare is attached to.
    """
    from everserve_vllm import worker

    if cfg.engine != attachment.engine:
        raise RuntimeError(
            f"a spare attached to {attachment.engine!r} as rank {attachment.rank} does not "
            f"register for {cfg.engine!r}"
        )
    w = proc.worker.worker
    return worker.start(
        attachment.rank,
        w.device,
        cfg,
        role=Role.SPARE,
        runtime_cls=SpareRuntime,
        slot_choice=slot_choice,
        compile_cache=worker.compiled_with(
            w.vllm_config,
            getattr(getattr(w, "model_runner", None), "model", None),
            attachment.rank,
        ),
        identity=worker.held_identity(w.vllm_config),
    )


def load(
    vllm_config: Any, rank: int, lock: Any, local_rank: int | None = None, resident: bool = False
) -> Any:
    """Build the spare's worker as rank ``rank``: the model loaded, in no group yet.

    ``CUDA_VISIBLE_DEVICES`` maps ``local_rank`` (``rank`` by default) to the spare's GPU.
    Returns the ``WorkerProc`` with no queues; :func:`attach` allocates its KV and warms it.

    ``resident``: the process holds the model in host memory (:mod:`everserve_vllm.resident`)
    and builds no model here; ``rank`` is a placeholder until :func:`attach` names the position,
    whose model is then built from the copy through the resident loader.
    """
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.worker import gpu_worker as gw

    real_init = gw.init_worker_distributed_environment
    gw.init_worker_distributed_environment = _install_stubs
    real_load = gw.Worker.load_model
    if resident:
        from everserve_vllm import resident as res

        res.install()
        vllm_config.load_config.load_format = res.LOAD_FORMAT
        gw.Worker.load_model = lambda self, **kw: None  # built at the attachment, for its rank

    class SpareProc(WorkerProc):
        def _init_message_queues(self, input_shm_handle: Any, cfg: Any) -> None:
            self.rpc_broadcast_mq = None
            self.worker_response_mq = None
            self.peer_response_handles = []

    try:
        return SpareProc(
            vllm_config,
            local_rank=rank if local_rank is None else local_rank,
            rank=rank,
            distributed_init_method=None,
            input_shm_handle=None,
            shared_worker_lock=lock,
            is_driver_worker=False,
        )
    finally:
        gw.init_worker_distributed_environment = real_init
        gw.Worker.load_model = real_load


def continue_rank_steps(rt: Any) -> int | None:
    """Continue the taken rank's commit steps from the restored watermark; the next step.

    The spare's commits carry the change's epoch (the runtime's after the change applies) and
    continue the rank's step sequence, so its first is the watermark plus one. None when nothing
    was restored (no restore ran: the spare's own steps stand).
    """
    from everserve_vllm import connector

    info = getattr(rt, "restore_info", None)
    conn = connector.WORKER
    if not info or conn is None:
        return None
    nxt = int(info["watermark"]) + 1
    conn.lifecycle.continue_from(nxt)
    return nxt


def take_position(
    proc: Any, rank: int, ring_handle: Any, connect_ip: str | None = None, catch_up: bool = True
) -> dict:
    """Take the replaced rank's position at the boundary: the half of the join the spare does.

    The catch-up restore runs to completion, its device synchronize included, before this
    process applies the change and answers: the executor sends no step until it has the answer,
    so no survivor enters the first collective of the step back while the spare is still
    writing rows. The replaced rank's last commits are in the store by now. Then the spare
    attaches to the new ring and makes the queue the executor reads its output from, over the
    network when ``connect_ip`` names this node's address, else in shared memory.

    ``catch_up`` false skips the catch-up restore: the rank died, so its rows were complete at the
    take's restore (``Offer.faulted``).
    """
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    from everserve_vllm import worker

    t = time.time()
    assert worker.STATE is not None
    rt = worker.STATE.runtime
    if catch_up and hasattr(rt, "catch_up"):
        rt.catch_up()
    elif getattr(rt, "restore_info", None) is not None:
        rt.restore_info["catch_up"] = {"skipped": "the rank died: its rows were complete"}
    continue_rank_steps(rt)
    applied = worker.apply_change()
    worker.mark_steps(proc.worker)  # the position's step word, from the first step it serves
    proc.rpc_broadcast_mq = MessageQueue.create_from_handle(ring_handle, rank)
    proc.worker_response_mq = (
        MessageQueue(1, 0, connect_ip=connect_ip) if connect_ip else MessageQueue(1, 1)
    )
    return {
        "handle": proc.worker_response_mq.export_handle(),
        "join_s": round(time.time() - t, 4),
        "applied": applied,
        "restore": getattr(rt, "restore_info", None),
    }


def serve(proc: Any) -> None:
    """Serve the position: wait for the executor on both queues, then the worker's loop."""
    t = time.time()
    proc.rpc_broadcast_mq.wait_until_ready()
    t1 = time.time()
    proc.worker_response_mq.wait_until_ready()
    log.info(
        "subscribed to the ring in %.3f s (the broadcast queue %.3f, the response queue %.3f)",
        time.time() - t,
        t1 - t,
        time.time() - t1,
    )
    proc.worker_busy_loop()


def _main(
    vllm_config: Any, rank: int, ctl: Any, lock: Any, startup: replay.Calls, env: dict[str, str]
) -> None:
    """Entry point of the executor's own spare process (the single-node shape).

    ``CUDA_VISIBLE_DEVICES`` maps index ``rank`` to the spare GPU, so the worker builds on the
    spare as rank ``rank``. The daemons' spares enter at :mod:`everserve_vllm.spare_main`.
    """
    from everserve_vllm import settings

    os.environ.update(env)
    settings.reload()  # this process's entry: its environment is complete only now
    t0 = time.time()
    try:
        proc = load(vllm_config, rank, lock)
        # the executor's spare is attached to the executor's engine, named as the executor
        # reads it; its bootstrap at "start" must name the same
        attachment = attach(proc, settings.current().engine, rank, startup)
    except Exception:
        import traceback

        ctl.send({"status": "failed", "error": traceback.format_exc()})
        raise
    ctl.send(
        {
            "status": "loaded",
            "load_s": round(time.time() - t0, 2),
            "warm_s": attachment.sizing_s + attachment.warm_s,
        }
    )
    from everserve_vllm import worker

    while True:
        msg = ctl.recv()
        if msg[0] == "start":
            try:
                out = register(attachment, proc, msg[1])
                assert worker.STATE is not None
                worker.STATE.collective.on_apply = _point_stub_at
                ctl.send({"status": "started", **out})
            except Exception:
                import traceback

                ctl.send({"status": "failed", "error": traceback.format_exc()})
                raise
        elif msg[0] == "join":
            ctl.send({"status": "READY", **take_position(proc, rank, msg[1])})
            serve(proc)
            return


# ----------------------------------------------------------------------- engine side
class Spare:
    """The executor's handle on the spare process."""

    def __init__(
        self,
        proc: Any,
        ctl: Any,
        death_writer: Any,
        rank: int,
        info: dict[str, Any],
        lock: Any = None,
    ) -> None:
        """Wrap a started, loaded spare; ``lock`` is the worker lock it shares, kept alive here."""
        self.proc, self.ctl, self.death_writer, self.rank, self.info = (
            proc,
            ctl,
            death_writer,
            rank,
            info,
        )
        self.lock = lock

    @classmethod
    def launch(cls, executor: Any, rank: int, gpu: int, row: int) -> Spare:
        """Spawn the spare for ``rank`` on physical GPU ``gpu`` at slot row ``row``.

        Returns once it has loaded, allocated KV and captured its graphs.

        Raises:
            RuntimeError: if the spare failed to load.
        """
        from vllm.v1.executor.multiproc_executor import get_mp_context

        t0 = time.time()
        from everserve_vllm import settings

        serve = list(settings.current().visible_devices or ("0", "1", "2", "3"))
        vis = list(serve)
        vis[rank] = str(gpu)
        ctx = get_mp_context()
        ctl, child = ctx.Pipe()
        _reader, death_writer = ctx.Pipe(duplex=False)
        # held for the spare's life: Process.start drops its arguments, and a lock collected
        # here unlinks its semaphore before a slow-starting child can open it (FileNotFoundError
        # in SemLock._rebuild, the engine waiting on the spare)
        lock = ctx.Lock()
        env = {
            settings.ENV_SPARE: "1",
            settings.ENV_SPARE_RANK: str(rank),
            settings.ENV_ROW: str(row),
        }
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(vis)
        try:
            p = ctx.Process(
                target=_main,
                args=(
                    executor.vllm_config,
                    rank,
                    child,
                    lock,
                    executor.startup.complete(),  # the core's own start-up calls
                    env,
                ),
                name=f"EverserveSpare-{rank}",
                daemon=True,
            )
            p.start()
        finally:
            if prev is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev
        msg = ctl.recv()
        if msg.get("status") != "loaded":
            raise RuntimeError("the spare failed to load:\n" + msg.get("error", ""))
        info = {
            "gpu": gpu,
            "rank": rank,
            "visible": ",".join(vis),
            **msg,
            "ready_s": round(time.time() - t0, 2),
        }
        return cls(p, ctl, death_writer, rank, info, lock)

    def start(self, cfg: Bootstrap) -> None:
        """Ask the spare to form its sets and register; pair with :meth:`started`."""
        self.ctl.send(("start", cfg))

    def started(self) -> dict[str, Any]:
        """Wait for the spare's start; raises if it failed."""
        msg = self.ctl.recv()
        if msg.get("status") != "started":
            raise RuntimeError("the spare failed to start:\n" + msg.get("error", ""))
        return msg

    def join(self, mq_handle: Any, ring_ready: Callable[[], None]) -> dict[str, Any]:
        """Hand the spare the new ring; it answers with its response queue's handle.

        ``ring_ready`` is the ring writer's wait for its readers, run between the offer and the
        answer: the spare subscribes to the ring while it takes the position, so the executor
        waiting for the answer first would wait on a reader that waits on it.

        Raises:
            RuntimeError: the spare answered anything but READY.
        """
        self.ctl.send(("join", mq_handle))
        ring_ready()
        msg = self.ctl.recv()
        if msg.get("status") != "READY":
            raise RuntimeError(f"the spare failed to join: {msg}")
        return msg

    def shutdown(self) -> None:
        """Stop the spare if it never joined."""
        if self.proc.is_alive():
            self.proc.kill()
