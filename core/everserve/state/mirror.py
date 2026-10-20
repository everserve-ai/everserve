# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The mirror transport: this rank's committed rows to the node's store, a lost rank's rows back.

A worker holds one connection to the node's store over a Unix socket and speaks
``proto/everserve/proto/mirror.proto`` in frames. The socket is the control plane; rows never cross
it (decision 0011). The worker stages a step's rows in a shared-memory file under its rank directory
and commits by naming them; the store copies them into its tables, which are shared-memory files a
restoring rank maps directly.

:class:`MirrorCommittedKV` is the production :class:`~everserve.state.committed.CommittedKV`. Its
commit is asynchronous with a fence two steps behind (decisions 0009 and 0019) and the step thread
never reads device memory (decision 0010): each layer's rows leave the device in
:meth:`MirrorCommittedKV.save`, one gather for every held head slot on the compute stream and one
copy on a side stream into a pooled staging file that is pinned for the GPU;
:meth:`MirrorCommittedKV.commit` records one event behind the last copy and hands the step to a
sender thread, which waits for the event and sends one descriptor of the step's rows with the
per-request counts. At most ``depth`` steps are in flight; a further commit waits for the oldest and
counts the wait. Evictions and finishes queue behind the commits already queued, so a lowered count
can never be overtaken by rows the scheduler has since freed. A restore maps the lost rank's tables
and copies the written rows into this rank's cache, one layer at a time.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from everserve.contract.signals import PROTOCOL_OLDEST, PROTOCOL_VERSION
from everserve.proto import mirror_pb2 as pb
from everserve.state.kvcopy import HeadSlot, PagedKV, layer_index
from everserve.wire import read_frame, write_frame

log = logging.getLogger(__name__)

Unit = Hashable
RequestId = str

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
    torch.int8: "int8",
    torch.uint8: "uint8",
}
_DTYPE_FROM_NAME = {v: k for k, v in _DTYPE_NAMES.items()}
_HEAD_STRIDE = 1_000
"""Wire layer id = layer * stride + head slot: one row table per (layer, head slot)."""
TIMELINE_STEPS = 512
"""Commits whose per-step stamps the transport keeps for :meth:`MirrorCommittedKV.timeline`."""
TIMELINE_HEAD = 16
"""The first commits, kept as well however long the run: the prefill's commit is among them."""


class MirrorError(Exception):
    """The mirror refused a request; the message is the service's reason."""


class MirrorClient:
    """One connection to the mirror service. Thread-safe; one request in flight at a time."""

    def __init__(
        self, path: str, connect_timeout_s: float = 30.0, reply_timeout_s: float = 60.0
    ) -> None:
        """Connect to the mirror's Unix socket at ``path``, retrying while it starts."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self._sock.connect(path)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        self._sock.settimeout(reply_timeout_s)
        self._lock = threading.Lock()

    def _ask(self, req: pb.MirrorRequest) -> pb.MirrorReply:
        """Send one request and return its reply; raise the service's reason on a refusal.

        A reply later than ``reply_timeout_s`` raises ``TimeoutError``: a sender that appears to
        wait longer than that is not waiting here. At debug level each request is logged before
        it is written and after its reply is read, so a stall can be matched against the store's
        own log of what it received.
        """
        kind = req.WhichOneof("kind")
        with self._lock:
            log.debug("mirror request %s", kind)
            write_frame(self._sock, req.SerializeToString())
            reply = pb.MirrorReply.FromString(read_frame(self._sock))
            log.debug("mirror reply to %s%s", kind, f": {reply.error}" if reply.error else "")
        if reply.error:
            raise MirrorError(reply.error)
        return reply

    def open(self, engine: str, rank: int) -> tuple[str, str]:
        """Open ``rank`` of ``engine``: its directory and this connection's staging prefix.

        The directory holds the staging files and the tables; every staging file this connection
        creates is named from the prefix, and the store removes them when the connection ends.
        """
        reply = self._ask(
            pb.MirrorRequest(
                open=pb.Open(
                    engine=engine,
                    rank=rank,
                    protocol=PROTOCOL_VERSION,
                    protocol_oldest=PROTOCOL_OLDEST,
                )
            )
        )
        return reply.opened.rank_dir, reply.opened.stage_prefix

    def prepare(
        self,
        engine: str,
        rank: int,
        tables: Sequence[int],
        dtype: str,
        shape: Sequence[int],
        slots: int,
    ) -> None:
        """Have the store create and touch ``tables`` for ``rank`` now, before any commit.

        ``slots`` is how many slot ids the engine writes (its KV blocks times its block size): a
        store holding fewer refuses here, naming both, rather than at the first step that writes
        past it ("slot 4096 outside 0..4096" ends the engine mid-serve).
        """
        self._ask(
            pb.MirrorRequest(
                prepare=pb.Prepare(
                    engine=engine,
                    rank=rank,
                    tables=list(tables),
                    dtype=dtype,
                    shape=list(shape),
                    slots=slots,
                )
            )
        )

    def commit(
        self,
        engine: str,
        rank: int,
        epoch: int,
        step: int,
        staged: pb.StagedStep | None,
        committed: Mapping[RequestId, int],
    ) -> None:
        """Commit this step and the committed token count per request.

        ``staged`` is one descriptor of the step's rows, or None for a step that wrote none.
        """
        self._ask(
            pb.MirrorRequest(
                commit=pb.Commit(
                    engine=engine,
                    rank=rank,
                    epoch=epoch,
                    step=step,
                    staged=staged,
                    request_ids=list(committed),
                    committed_tokens=list(committed.values()),
                )
            )
        )

    def restore_snapshot(
        self, engine: str, rank: int, since_step: int = 0
    ) -> tuple[list[pb.TableRef], dict[RequestId, int], int]:
        """``rank``'s tables, its own committed counts, and its last committed step.

        The tables name the slots written after ``since_step`` (0: every written slot); the step
        is the watermark a later call continues from. The rows are complete through exactly those
        counts, whatever other ranks committed since.
        """
        reply = self._ask(
            pb.MirrorRequest(
                restore=pb.RestoreRequest(engine=engine, rank=rank, since_step=since_step)
            )
        ).restore
        counts = dict(zip(reply.request_ids, reply.committed_tokens, strict=True))
        return list(reply.tables), counts, reply.step

    def committed(self, engine: str, request_ids: Sequence[RequestId]) -> dict[RequestId, int]:
        """The committed token count per request; zero for one never seen."""
        reply = self._ask(
            pb.MirrorRequest(
                committed=pb.CommittedQuery(engine=engine, request_ids=list(request_ids))
            )
        ).committed
        return dict(zip(reply.request_ids, reply.committed_tokens, strict=True))

    def state(self, engine: str, rank: int) -> tuple[bool, int, int, bool]:
        """Where this store stands for ``rank``: held, epoch, step, caught up.

        ``held`` is whether any row of the rank is written; ``caught_up`` whether the source's
        carrier has said it has nothing more queued through that step. A spare whose rows arrive
        by a staged replication (decision 0030) restores once both hold.
        """
        reply = self._ask(
            pb.MirrorRequest(
                replica_state=pb.ReplicaStateQuery(
                    engine=engine,
                    rank=rank,
                    protocol=PROTOCOL_VERSION,
                    protocol_oldest=PROTOCOL_OLDEST,
                )
            )
        )
        s = reply.replica_state
        return s.held, s.epoch, s.step, s.caught_up

    def rows(
        self, engine: str, rank: int, table: int, slots: Sequence[int]
    ) -> tuple[int, int, list[tuple[int, int, bytes]]]:
        """What the store holds for ``slots`` of ``table`` of ``rank``, read under its lock.

        The rank's (epoch, step) and each slot's (slot, written_step, bytes). A diagnostic for
        comparing copies of a row across stores; the serving path never calls it.
        """
        reply = self._ask(
            pb.MirrorRequest(
                rows=pb.RowsQuery(engine=engine, rank=rank, table=table, slots=list(slots))
            )
        )
        r = reply.rows
        return r.epoch, r.step, [(x.slot, x.written_step, bytes(x.data)) for x in r.rows]

    def evicted(self, engine: str, request_id: RequestId, kept_tokens: int) -> None:
        """The scheduler evicted ``request_id``'s blocks beyond ``kept_tokens``."""
        self._ask(
            pb.MirrorRequest(
                evicted=pb.Evicted(engine=engine, request_id=request_id, kept_tokens=kept_tokens)
            )
        )

    def finished(self, engine: str, request_ids: Sequence[RequestId]) -> None:
        """Forget finished requests."""
        self._ask(
            pb.MirrorRequest(finished=pb.Finished(engine=engine, request_ids=list(request_ids)))
        )

    def close(self) -> None:
        """Close the connection."""
        self._sock.close()


@dataclass
class CommitStats:
    """What the step thread paid, for the run's report; written by the step thread alone."""

    steps: int = 0
    step_thread_s: float = 0.0
    backpressure_waits: int = 0
    fence_waits_s: float = 0.0
    max_in_flight: int = 0


@dataclass
class SendStats:
    """What the sender thread did, for the run's report; written by the sender thread alone."""

    bytes_sent: int = 0
    messages: int = 0
    """Commit requests sent to the store: one per step (decision 0019)."""


@dataclass
class _Stage:
    """A staging file in the rank directory, mapped as ``[layers, heads, capacity, 2, d]``.

    Pinned for the GPU on a CUDA host, so the side stream's copy into it is a true DMA. The store
    maps the same file read-only and copies the rows a commit names into its tables.
    """

    name: str
    path: Path
    tensor: torch.Tensor
    registered: bool = False
    device_ptr: int = 0
    """The file's address in the device's address space when it is mapped for the gather kernel."""
    unregister: Callable[[int], None] | None = None

    def release(self) -> None:
        """Unpin and delete the file."""
        if self.registered:
            if self.unregister is not None:
                self.unregister(self.tensor.data_ptr())
            else:
                torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr())
            self.registered = False
        self.tensor = torch.empty(0)
        self.path.unlink(missing_ok=True)


class _GatherKernel:
    """The step's gather as one launch of the ``gather_rows`` kernel (``core/csrc/gather.cu``).

    Built at registration when the kernels are usable and every layer's cache shares one shape,
    dtype and stride: it holds the layers' base pointers on the device and the layout as five
    strides. It writes every layer's rows of a step straight into the staging file through the
    file's device mapping, replacing a gather, a transpose and a copy per layer.
    """

    def __init__(self, ext: object, layers: Sequence[torch.Tensor], paged: PagedKV) -> None:
        """Describe ``layers`` for the kernel loaded as ``ext``."""
        first = layers[0]
        self._ext = ext
        self._ptrs = torch.tensor(
            [t.data_ptr() for t in layers], dtype=torch.long, device=first.device
        )
        self._strides = list(paged.layout_strides(first))
        self._block_size = paged.block_size
        self._d = paged.head_dim
        self._elem = first.element_size()

    @classmethod
    def for_layers(cls, layers: Sequence[torch.Tensor], paged: PagedKV) -> _GatherKernel | None:
        """The kernel for these caches, or None when the torch path must be used."""
        from everserve import kernels

        first = layers[0]
        if not first.is_cuda or not kernels.available():
            return None
        same = all(
            t.shape == first.shape and t.stride() == first.stride() and t.dtype == first.dtype
            for t in layers
        )
        if not same or any(t.data_ptr() % 16 for t in layers):
            log.info("caches differ in shape, stride, dtype or alignment: torch gather path")
            return None
        return cls(kernels.load(), layers, paged)

    def register(self, stage: _Stage) -> None:
        """Pin ``stage``'s file mapped into the device address space."""
        t = stage.tensor
        stage.device_ptr = self._ext.host_register_mapped(
            t.data_ptr(), t.numel() * t.element_size()
        )
        stage.unregister = self._ext.host_unregister
        stage.registered = True

    def gather(self, slots: torch.Tensor, heads: torch.Tensor, stage: _Stage) -> None:
        """Write the step's rows of every layer into ``stage`` on the current stream."""
        self._ext.gather_rows(
            self._ptrs,
            slots,
            heads,
            stage.device_ptr,
            stage.tensor.shape[2],
            self._d,
            self._block_size,
            self._strides,
            self._elem,
        )


@dataclass
class _StepGather:
    """One step's rows on their way to the host: issued layer by layer as the forward runs.

    ``stage`` is a pooled staging file; the step uses its first ``n`` rows. ``slots_host`` is the
    step's slot ids, copied off the device on the same stream, so the sender thread, never the
    step thread, learns them.
    """

    n: int
    heads: tuple[HeadSlot, ...]
    slots_device: torch.Tensor
    slots_host: torch.Tensor
    stage: _Stage
    filled: set[int] = field(default_factory=set)


@dataclass
class _Job:
    """One unit of work for the sender thread; ``done`` is set when it has been sent or failed."""

    done: threading.Event = field(default_factory=threading.Event, init=False)


@dataclass
class _CommitJob(_Job):
    epoch: int
    step: int
    gather: _StepGather | None
    ready: object | None  # a CUDA event the last copy signals, or None
    counts: dict[RequestId, int]
    error: BaseException | None = None
    times: dict[str, float] = field(default_factory=dict)
    """``perf_counter`` stamps of the job's life, see :meth:`CommittedKV.timeline`."""


@dataclass
class _EvictJob(_Job):
    request_id: RequestId
    kept_tokens: int


@dataclass
class _FinishJob(_Job):
    request_id: RequestId


@dataclass
class _DrainJob(_Job):
    """A marker: everything queued before it has reached the store when it is done."""


class MirrorCommittedKV:
    """The production committed-KV transport: this rank's rows to the mirror, a lost rank's back.

    Rows are keyed by head slot at the store: a unit's rows live in the layer table under the
    head slot that held it on the committing rank, and ``restore`` writes them into the slot this
    rank assigned to the unit. The per-unit mapping is the ``units`` callable, as for
    :class:`~everserve.reference.kvcopy.KVCopy`.
    """

    def __init__(
        self,
        client: MirrorClient,
        engine: str,
        rank: int,
        units: Callable[[], Sequence[tuple[HeadSlot, Unit]]],
        depth: int = 3,
    ) -> None:
        """Commit as ``rank`` of ``engine`` over ``client``; ``units`` gives the held head slots.

        Opens the rank at the store, which names the directory for staging files and tables.

        Args:
            client: the mirror connection; the sender thread owns its commit calls.
            engine: the engine's name at the mirror.
            rank: this rank's id.
            units: returns ``[(head slot, unit), ...]`` held right now.
            depth: steps that may be in flight before a commit waits for the oldest; three
                holds the fence's two steps of slack plus the step being committed (decision 0019).
        """
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self._client = client
        self._engine = engine
        self._rank = rank
        self._units = units
        self._depth = depth
        rank_dir, self._stage_prefix = client.open(engine, rank)
        self._rank_dir = Path(rank_dir)
        self._stage_seq = 0
        self.layers: list[tuple[str, torch.Tensor]] = []
        self.paged: PagedKV | None = None
        self._layer_of: dict[str, int] = {}
        self._step: _StepGather | None = None
        self._heads: tuple[tuple[HeadSlot, ...], torch.Tensor] | None = None
        self._committed: dict[RequestId, int] = {}
        self.restored_counts: dict[RequestId, int] = {}
        """The restored rank's own committed counts, read with its rows by the last restore."""
        self.restored_rows = 0
        """Rows the last restore copied, over every table."""
        self.restore_timing: dict[str, float] = {}
        """The last restore's store reply and copy, seconds each (:meth:`restore`)."""
        self.stats_record = CommitStats()
        self.sent_record = SendStats()
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._in_flight: dict[int, _CommitJob] = {}
        self._in_flight_lock = threading.Lock()
        self._history: deque[_CommitJob] = deque(maxlen=TIMELINE_STEPS)
        self._head: list[_CommitJob] = []
        self._pool: list[_Stage] = []  # staging files, at most depth + 1 kept
        self._pool_lock = threading.Lock()
        self._failed: BaseException | None = None
        self._side: torch.cuda.Stream | None = None
        self._kernel: _GatherKernel | None = None
        self._sender = threading.Thread(
            target=self._send_loop, daemon=True, name="everserve-mirror-send"
        )
        self._sender.start()

    def state(self, rank: int) -> tuple[bool, int, int, bool]:
        """Where this node's store stands for ``rank`` of the engine: held, epoch, step, caught up.

        What a restore of ``rank``'s rows waits on when they are staged onto this node for it
        (decision 0030); :meth:`MirrorClient.state`.
        """
        return self._client.state(self._engine, rank)

    # ---------------------------------------------------------------- registration
    def register(self, kv_caches: Mapping[str, torch.Tensor]) -> None:
        """Take the rank's per-layer cache tensors, in layer order, and prepare their tables.

        The store creates and touches one table per (layer, held head slot) now, so the first
        commit, the prefill's, costs what every later one does (decision 0020).
        """
        ordered = sorted((layer_index(n), n, t) for n, t in kv_caches.items())
        self.layers = [(n, t) for _, n, t in ordered]
        self._layer_of = {n: i for i, (n, _) in enumerate(self.layers)}
        first = self.layers[0][1]
        self.paged = PagedKV.of(first)
        if first.is_cuda:
            self._side = torch.cuda.Stream(device=first.device)
        heads = sorted(h for h, _unit in self._units())
        tables = [layer * _HEAD_STRIDE + h for layer in range(len(self.layers)) for h in heads]
        if tables:
            self._client.prepare(
                self._engine,
                self._rank,
                tables,
                _DTYPE_NAMES[first.dtype],
                [2, self.paged.head_dim],
                self.paged.num_slots,
            )
            self._kernel = _GatherKernel.for_layers([t for _, t in self.layers], self.paged)

    def save(self, layer_name: str, kv_layer: torch.Tensor, slots: torch.Tensor) -> None:
        """Issue this layer's rows at ``slots`` towards the host for every head slot held.

        Nothing here reads device memory on the calling thread. With the gather kernel, the layer
        is only marked: :meth:`commit` moves every marked layer in one launch on the side stream,
        behind the end of the forward. Without it, the gather runs on the compute stream behind
        the layer's own writes and the copy to the staging file runs on the side stream. Either
        way the forward never waits for the copy and the step thread never waits for the forward.
        ``slots`` may contain negative padding ids; they gather row 0 and the store skips them.
        """
        if self.paged is None:
            return
        layer = self._layer_of[layer_name]
        g = self._step
        if g is None:
            g = self._step = self._begin_gather(slots)
        elif g.n != slots.shape[0]:
            raise ValueError(
                f"one slot mapping per step: layer {layer_name} has {slots.shape[0]} slots, "
                f"the step has {g.n}"
            )
        if self._kernel is not None:
            g.filled.add(layer)
            return
        heads_index = self._head_index(g.heads, kv_layer.device)
        rows = self.paged.rows_of_heads(kv_layer, g.slots_device, heads_index)  # [h, n, 2, d]
        target = g.stage.tensor[layer, :, : g.n]
        if self._side is not None:
            self._side.wait_stream(torch.cuda.current_stream(kv_layer.device))
            with torch.cuda.stream(self._side):
                target.copy_(rows, non_blocking=True)
                rows.record_stream(self._side)  # freed only after the copy has read it
        else:
            target.copy_(rows)
        g.filled.add(layer)

    def _begin_gather(self, slots: torch.Tensor) -> _StepGather:
        """Start a step's gather: its head slots, its slot ids on both sides, a staging file."""
        heads = tuple(sorted(h for h, _unit in self._units()))
        n = slots.shape[0]
        dtype = self.layers[0][1].dtype
        stage = self._staging(n, len(heads), dtype)
        slots_long = slots.long()
        slots_host = torch.empty(n, dtype=torch.long, pin_memory=self._side is not None)
        if self._side is not None:
            self._side.wait_stream(torch.cuda.current_stream(slots.device))
            with torch.cuda.stream(self._side):
                slots_host.copy_(slots_long, non_blocking=True)
                slots_long.record_stream(self._side)
        else:
            slots_host.copy_(slots_long)
        return _StepGather(n, heads, slots_long.clamp(min=0), slots_host, stage)

    def _head_index(self, heads: tuple[HeadSlot, ...], device: torch.device) -> torch.Tensor:
        """The held head slots as an index tensor on ``device``, rebuilt only when they change."""
        if self._heads is None or self._heads[0] != heads or self._heads[1].device != device:
            self._heads = (heads, torch.tensor(heads, dtype=torch.long, device=device))
        return self._heads[1]

    # ---------------------------------------------------------------- staging files
    def _staging(self, n: int, heads: int, dtype: torch.dtype) -> _Stage:
        """A staging file mapped as ``[layers, heads, capacity, 2, d]`` with capacity >= ``n``.

        Capacities are powers of two from 8, so decode steps of nearby batch sizes share files
        and a prefill gets its own. From the pool when one fits, else created in the rank
        directory and pinned for the GPU on a CUDA host.
        """
        assert self.paged is not None
        capacity = max(8, 1 << (n - 1).bit_length())
        shape = (len(self.layers), heads, capacity, 2, self.paged.head_dim)
        with self._pool_lock:
            for i, st in enumerate(self._pool):
                if tuple(st.tensor.shape) == shape and st.tensor.dtype == dtype:
                    return self._pool.pop(i)
        self._stage_seq += 1
        name = f"{self._stage_prefix}{self._stage_seq}.rows"
        path = self._rank_dir / name
        numel = 1
        for s in shape:
            numel *= s
        # the node's store maps it, in another pod: made for the node's group, not by the umask
        from everserve.podcontract import make_shared

        make_shared(str(path), numel * torch.empty((), dtype=dtype).element_size())
        tensor = torch.from_file(str(path), shared=True, size=numel, dtype=dtype).view(shape)
        stage = _Stage(name, path, tensor)
        if self._kernel is not None:
            try:
                self._kernel.register(stage)
            except RuntimeError as e:
                stage.release()
                raise MirrorError(
                    f"mapping {name} for the device failed: {e}. Staging files must be on shared "
                    f"memory (tmpfs, /dev/shm): a disk-backed file cannot be pinned"
                ) from e
        elif self._side is not None:
            rc = torch.cuda.cudart().cudaHostRegister(
                tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0
            )
            if rc != 0:
                stage.release()
                raise MirrorError(
                    f"cudaHostRegister of {name} failed with {rc}. Staging files must be on shared "
                    f"memory (tmpfs, /dev/shm): a disk-backed file cannot be pinned"
                )
            stage.registered = True
        return stage

    def _release(self, stage: _Stage) -> None:
        """Return a staging file to the pool, keeping at most ``depth + 1``; delete the rest."""
        with self._pool_lock:
            if len(self._pool) <= self._depth:
                self._pool.append(stage)
                return
        stage.release()

    # ---------------------------------------------------------------- the pipeline
    def commit(self, epoch: int, step: int, committed_tokens: Mapping[RequestId, int]) -> None:
        """Queue this step's rows and counts; return without touching the device.

        The copies were issued layer by layer in :meth:`save`; this records one event behind the
        last of them and hands the step to the sender thread, which waits for the event and sends
        the step's description with the counts on the last frame. With ``depth`` steps already
        in flight this waits for the oldest first.
        """
        self._check_failed()
        t0 = time.perf_counter()
        g, self._step = self._step, None
        job = _CommitJob(epoch, step, g, None, dict(committed_tokens))
        job.times["step"] = step
        job.times["commit"] = t0
        if g is not None and self._side is not None:
            if self._kernel is not None:
                self._gather_step(g)
            with torch.cuda.stream(self._side):
                job.ready = self._side.record_event()
        with self._in_flight_lock:
            oldest = min(self._in_flight) if len(self._in_flight) >= self._depth else None
        if oldest is not None:
            self.stats_record.backpressure_waits += 1
            self.fence(oldest)
        with self._in_flight_lock:
            self._in_flight[step] = job
            self.stats_record.max_in_flight = max(
                self.stats_record.max_in_flight, len(self._in_flight)
            )
            self._history.append(job)
            if len(self._head) < TIMELINE_HEAD:
                self._head.append(job)
        self._queue.put(job)
        self._committed.update(job.counts)
        self.stats_record.steps += 1
        self.stats_record.step_thread_s += time.perf_counter() - t0

    def _gather_step(self, g: _StepGather) -> None:
        """Move the step's marked layers to the staging file on the side stream, behind the forward.

        One kernel launch when every layer was marked (the normal step); the torch path per layer
        otherwise. Reads no device memory on the calling thread.

        The side stream reads the caches while the next forward may already run. That forward
        writes the next step's slots; a slot of this step is written again only after its block
        is freed, which happens only to a finished or preempted request, whose committed rows the
        store has already been told to forget or to lower.
        """
        assert self._side is not None and self._kernel is not None and self.paged is not None
        device = self.layers[0][1].device
        heads = self._head_index(g.heads, device)
        forward_done = torch.cuda.current_stream(device).record_event()
        self._side.wait_event(forward_done)
        with torch.cuda.stream(self._side):
            if len(g.filled) == len(self.layers):
                self._kernel.gather(g.slots_device, heads, g.stage)
            else:
                for layer in sorted(g.filled):
                    rows = self.paged.rows_of_heads(self.layers[layer][1], g.slots_device, heads)
                    g.stage.tensor[layer, :, : g.n].copy_(rows, non_blocking=True)
            g.slots_device.record_stream(self._side)
            heads.record_stream(self._side)

    def fence(self, step: int) -> None:
        """Wait until the store acknowledged the commit of ``step``; raise if it failed."""
        with self._in_flight_lock:
            job = self._in_flight.get(step)
        if job is None:
            return
        t0 = time.perf_counter()
        job.times["fence_start"] = t0
        job.done.wait()
        job.times["fence_end"] = time.perf_counter()
        self.stats_record.fence_waits_s += job.times["fence_end"] - t0
        with self._in_flight_lock:
            self._in_flight.pop(step, None)
        if job.error is not None:
            raise MirrorError(f"commit of step {step} failed: {job.error}") from job.error
        self._check_failed()

    def _check_failed(self) -> None:
        """Raise if the sender thread failed: the transport is poisoned from the first error on.

        A step that fails to commit must not be agreed, so the error reaches the step thread
        through the next ``commit`` or ``fence`` and the step aborts like any other failed commit.
        """
        if self._failed is not None:
            raise MirrorError(f"mirror transport failed: {self._failed}") from self._failed

    def _send_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                if isinstance(job, _CommitJob):
                    self._send_commit(job)
                elif isinstance(job, _EvictJob):
                    self._client.evicted(self._engine, job.request_id, job.kept_tokens)
                elif isinstance(job, _FinishJob):
                    self._client.finished(self._engine, [job.request_id])
            except Exception as e:  # noqa: BLE001 - kept, surfaced on the step thread
                if isinstance(job, _CommitJob):
                    job.error = e
                if self._failed is None:
                    self._failed = e
                log.error("mirror send failed for rank %d of %s: %s", self._rank, self._engine, e)
            finally:
                job.done.set()

    def _send_commit(self, job: _CommitJob) -> None:
        job.times["dequeue"] = time.perf_counter()
        g = job.gather
        if g is None or not g.filled:
            if g is not None:
                self._release(g.stage)
            job.times["ready"] = time.perf_counter()
            self._client.commit(self._engine, self._rank, job.epoch, job.step, None, job.counts)
            self.sent_record.messages += 1
            job.times["sent"] = time.perf_counter()
            return
        if job.ready is not None:
            job.ready.synchronize()  # the sender thread waits; the step thread never does
        job.times["ready"] = time.perf_counter()
        try:
            self._client.commit(
                self._engine, self._rank, job.epoch, job.step, self._describe(g), job.counts
            )
            self.sent_record.messages += 1
            job.times["sent"] = time.perf_counter()
        finally:
            self._release(g.stage)

    def _describe(self, g: _StepGather) -> pb.StagedStep:
        """The step's rows as one descriptor of the staging file's layout (decision 0019)."""
        t = g.stage.tensor
        head_dim = t.shape[-1]
        capacity = t.shape[2]
        row_bytes = 2 * head_dim * t.element_size()
        slots = g.slots_host.tolist()  # pinned memory, complete once the event has fired
        layers = sorted(g.filled)
        self.sent_record.bytes_sent += (
            sum(1 for s in slots if s >= 0) * row_bytes * len(layers) * len(g.heads)
        )
        return pb.StagedStep(
            stage=g.stage.name,
            base_offset=0,
            layer_stride=len(g.heads) * capacity * row_bytes,
            head_stride=capacity * row_bytes,
            layers=layers,
            heads=list(g.heads),
            table_stride=_HEAD_STRIDE,  # one table per (layer, head slot)
            slots=slots,
            dtype=_DTYPE_NAMES[t.dtype],
            shape=[2, head_dim],
        )

    # ---------------------------------------------------------------- the lifecycle
    def committed(self, request_id: RequestId) -> int:
        """Locally tracked committed count; the mirror holds the authoritative copy."""
        return self._committed.get(request_id, 0)

    def stats(self) -> Mapping[str, float]:
        """The pipeline's figures so far: what the step thread paid and how often it waited."""
        return {**asdict(self.stats_record), **asdict(self.sent_record)}

    @property
    def gather_path(self) -> str:
        """``"kernel"`` when the step's gather is one launch of ``gather_rows``, else ``"torch"``.

        For the run report: a kernel that failed to build falls back to the torch path with one
        log line, and a run must show which path it measured.
        """
        return "kernel" if self._kernel is not None else "torch"

    def timeline(self) -> Sequence[Mapping[str, float]]:
        """The commits' stamps, oldest first; see the protocol.

        The first :data:`TIMELINE_HEAD` commits and the last :data:`TIMELINE_STEPS`.
        """
        with self._in_flight_lock:
            kept = list(self._history)
            jobs = [j for j in self._head if all(j is not k for k in kept)] + kept
        return [dict(j.times) for j in jobs]

    def evicted(self, request_id: RequestId, kept_tokens: int) -> None:
        """Lower the count locally now and at the mirror behind the commits already queued."""
        if kept_tokens < 0:
            raise ValueError(f"kept_tokens must be non-negative, got {kept_tokens}")
        if request_id in self._committed:
            self._committed[request_id] = min(self._committed[request_id], kept_tokens)
        self._queue.put(_EvictJob(request_id, kept_tokens))

    def finished(self, request_id: RequestId) -> None:
        """Forget the request locally now and at the mirror behind the commits already queued."""
        self._committed.pop(request_id, None)
        self._queue.put(_FinishJob(request_id))

    def clear(self) -> None:
        """Drop anything saved but not committed; its staging file goes back to the pool.

        A copy into that file may still be in flight on the side stream. The next step's copies
        queue behind it on the same stream, so the reuse is ordered.
        """
        g, self._step = self._step, None
        if g is not None:
            self._release(g.stage)

    def drain(self) -> None:
        """Wait for every queued commit, eviction and finish to reach the store, or raise."""
        with self._in_flight_lock:
            steps = sorted(self._in_flight)
        for step in steps:
            self.fence(step)
        marker = _DrainJob()
        self._queue.put(marker)
        marker.done.wait()
        self._check_failed()

    def close(self) -> None:
        """Drain, stop the sender thread, delete this rank's staging files.

        A send failure is raised after the thread stops.
        """
        error: MirrorError | None = None
        try:
            self.drain()
        except MirrorError as e:
            error = e
        self._queue.put(None)
        self._sender.join(10)
        with self._pool_lock:
            pool, self._pool = self._pool, []
        for stage in pool:
            stage.release()
        if error is not None:
            raise error

    def restore(
        self,
        dead_rank: int,
        moves: Sequence[tuple[Unit, HeadSlot, HeadSlot]],
        since_step: int = 0,
    ) -> int:
        """Write ``dead_rank``'s rows into this rank's cache, head slot by head slot.

        The store keys rows by the head slot they held on the committing rank, so each move's
        second field selects the table and its third field says where the rows go here. Each
        table is mapped from shared memory and its written rows are gathered on the host into
        pinned memory, copied to the device and scattered into the cache: no bytes cross the
        socket.

        With ``since_step`` only the slots written after that step of ``dead_rank``'s own are
        copied: a spare restores everything when a notice arrives and, at the swap, only what the
        leaving rank committed since, so the tail it replays is one position. Returns the rank's
        last committed step, the watermark for the next call; ``restored_counts`` holds its counts
        and ``restored_rows`` the rows this call copied, over every table; ``restore_timing`` the
        store's reply (``snapshot_s``, from the call to the reply, its lock wait included) and the
        copy into the cache (``copy_s``), wall-clock ``at`` the call. Split, because a catch-up
        that copies no row can still take 3 to 9 s, and the split says where.
        """
        if self.paged is None:
            raise RuntimeError("restore before register")
        self.restored_rows = 0
        head_map = {theirs: mine for _unit, theirs, mine in moves}
        device = self.layers[0][1].device
        at = time.time()
        tables, self.restored_counts, step = self._client.restore_snapshot(
            self._engine, dead_rank, since_step
        )
        replied = time.time()
        for table in tables:
            layer, head = divmod(table.layer, _HEAD_STRIDE)
            if head not in head_map or not table.written:
                continue
            dtype = _DTYPE_FROM_NAME[table.dtype]
            d = int(table.shape[1])
            row_elems = 2 * d
            num_slots = os.path.getsize(table.path) // (
                row_elems * torch.empty(0, dtype=dtype).element_size()
            )
            src = torch.from_file(table.path, shared=True, size=num_slots * row_elems, dtype=dtype)
            src = src.view(num_slots, 2, d)
            idx = torch.cat(
                [torch.arange(w.start, w.start + w.count, dtype=torch.long) for w in table.written]
            )
            rows = torch.empty((idx.numel(), 2, d), dtype=dtype, pin_memory=device.type == "cuda")
            torch.index_select(src, 0, idx, out=rows)
            self.restored_rows += idx.numel()
            on_device = rows.to(device, non_blocking=True)
            self.paged.write_rows(
                self.layers[layer][1], idx.to(device), head_map[head], on_device.transpose(0, 1)
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.restore_timing = {
            "at": at,
            "snapshot_s": round(replied - at, 4),
            "copy_s": round(time.time() - replied, 4),
        }
        return step
