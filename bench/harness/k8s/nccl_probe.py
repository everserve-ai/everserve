# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What NCCL's nonblocking initialisation does on the engine's own image (decision 0035, amended).

A cover group is joined beside serving. A blocking ``ncclCommInitRank`` whose peer never comes
hangs the joining thread for good, and there is no handle to abort it with. A nonblocking one
(``ncclCommInitRankConfig`` with ``blocking = 0``) returns a handle at once, its state read by
``ncclCommGetAsyncError``, and ``ncclCommAbort`` ends it. Whether that holds on the image the
engine runs, and what such a communicator then does under vLLM's calls, decides the join. This
probe answers it on GPUs, one process per pod, run by ``dynamo_proof.sh --nccl-probe``:

    python3 - missing --deadline-s 10            # a group of two whose second member never comes
    python3 - mint                               # an id whose maker then exits
    python3 - missing --as 1 --id HEX --deadline-s 10   # a member whose server is gone
    python3 - join --rank 0 [--how vllm|blocking|nonblocking] [--deadline-s 120]
    python3 - join --rank 1 --id HEX [--how ...] [--deadline-s 120]

``join`` at rank 0 prints its id first, then joins.

Each mode prints one JSON line, its answer, last.

The configuration is passed as the prefix NCCL has kept since 2.14 (size, magic, version,
blocking) and declares version 2.14.0. NCCL copies the bytes the caller's ``size`` names over its
own defaults and sets every field newer than the declared version to its default. No later
field's layout is assumed.
"""

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import threading
import time
import types

NCCL_SUCCESS = 0
NCCL_IN_PROGRESS = 7
CONFIG_MAGIC = 0xCAFEBEEF
CONFIG_VERSION = 21400  # NCCL_VERSION(2, 14, 0): the version that added `blocking`


class ConfigPrefix(ctypes.Structure):
    """``ncclConfig_t`` up to and including ``blocking``: its layout since NCCL 2.14."""

    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("blocking", ctypes.c_int),
    ]


def config(blocking: bool) -> ConfigPrefix:
    """A configuration naming only ``blocking``, every other field NCCL's default."""
    return ConfigPrefix(ctypes.sizeof(ConfigPrefix), CONFIG_MAGIC, CONFIG_VERSION, int(blocking))


class Nccl:
    """The NCCL vLLM loads, with the three calls the nonblocking join needs bound on its handle."""

    def __init__(self):
        """Load NCCL as vLLM does (``VLLM_NCCL_SO_PATH`` or its search) and bind the calls."""
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            NCCLLibrary,
            ncclComm_t,
            ncclUniqueId,
        )

        self.vllm = NCCLLibrary()
        self.configs: list[ConfigPrefix] = []
        lib = self.vllm.lib
        self.comm_t, self.unique_id_t = ncclComm_t, ncclUniqueId
        self.init_config = lib.ncclCommInitRankConfig
        self.init_config.restype = ctypes.c_int
        self.init_config.argtypes = [
            ctypes.POINTER(ncclComm_t),
            ctypes.c_int,
            ncclUniqueId,
            ctypes.c_int,
            ctypes.POINTER(ConfigPrefix),
        ]
        self.async_error = lib.ncclCommGetAsyncError
        self.async_error.restype = ctypes.c_int
        self.async_error.argtypes = [ncclComm_t, ctypes.POINTER(ctypes.c_int)]
        self.abort = lib.ncclCommAbort
        self.abort.restype = ctypes.c_int
        self.abort.argtypes = [ncclComm_t]
        self.split = lib.ncclCommSplit
        self.split.restype = ctypes.c_int
        self.split.argtypes = [
            ncclComm_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ncclComm_t),
            ctypes.POINTER(ConfigPrefix),
        ]
        self.all_reduce = lib.ncclAllReduce
        self.all_reduce.restype = ctypes.c_int
        self.all_reduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ncclComm_t,
            ctypes.c_void_p,
        ]

    def version(self) -> int:
        """NCCL's version code, e.g. 22907."""
        return self.vllm.ncclGetRawVersion()

    def mint(self) -> bytes:
        """A new group id, its maker this process.

        The id is held while its bytes are copied. A temporary's memory is freed as soon as
        ``addressof`` returns, and the allocator's own pointer then overwrites the id's magic:
        every join is then refused ("didn't receive a valid magic", probe 2).
        """
        uid = self.vllm.ncclGetUniqueId()
        return ctypes.string_at(ctypes.addressof(uid), ctypes.sizeof(uid))

    def start(self, world: int, uid: bytes, rank: int, blocking: bool = False):
        """A join's handle through ``ncclCommInitRankConfig``, and what the call returned.

        The configuration is kept for this object's life: a nonblocking initialisation runs on
        NCCL's own thread after the call returns, and nothing here frees what it may still read.
        """
        comm = self.comm_t()
        self.configs.append(config(blocking))
        rc = self.init_config(
            ctypes.byref(comm),
            world,
            self.vllm.unique_id_from_bytes(uid),
            rank,
            ctypes.byref(self.configs[-1]),
        )
        return comm, rc

    def split_blocking(self, parent, rank: int):
        """A blocking child of ``parent`` with every member, and what the call returned.

        One color, keyed by rank. Its configuration is kept as ``start``'s is.
        """
        child = self.comm_t()
        self.configs.append(config(blocking=True))
        rc = self.split(parent, 0, rank, ctypes.byref(child), ctypes.byref(self.configs[-1]))
        return child, rc

    def state(self, comm) -> int:
        """The comm's state (``ncclCommGetAsyncError``), or the call's own error."""
        out = ctypes.c_int(-1)
        rc = self.async_error(comm, ctypes.byref(out))
        return out.value if rc == NCCL_SUCCESS else rc

    def wait(self, comm, deadline_s: float, poll_s: float = 0.01):
        """The comm's state when it left ``ncclInProgress``, or at the deadline, and when."""
        t0 = time.monotonic()
        polls = 0
        while True:
            s = self.state(comm)
            polls += 1
            if s != NCCL_IN_PROGRESS or time.monotonic() - t0 >= deadline_s:
                return s, round(time.monotonic() - t0, 3), polls
            time.sleep(poll_s)

    def aborted(self, comm):
        """``ncclCommAbort``'s return code and the seconds it took."""
        t0 = time.monotonic()
        rc = self.abort(comm)
        return rc, round(time.monotonic() - t0, 3)


def device_still_serves() -> bool:
    """A kernel runs and a copy returns on this process's device after the abort."""
    import torch

    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    return float((x * 2).sum().item()) == 1023 * 1024.0


def missing(args) -> dict:
    """Q a1 and a2: a group of two that never forms, waited on to the deadline and aborted."""
    import torch

    torch.cuda.set_device(0)
    nccl = Nccl()
    uid = bytes.fromhex(args.id) if args.id else nccl.mint()
    comm, rc = nccl.start(2, uid, args.as_rank)
    out = {"mode": "missing", "as": args.as_rank, "nccl": nccl.version(), "init_rc": rc}
    if rc not in (NCCL_SUCCESS, NCCL_IN_PROGRESS):
        return out | {"answer": "the call failed", "state": rc}
    state, waited_s, polls = nccl.wait(comm, args.deadline_s)
    out |= {"state": state, "waited_s": waited_s, "polls": polls}
    out["abort_rc"], out["abort_s"] = nccl.aborted(comm)
    out["device_ok"] = device_still_serves()
    return out


def mint(_args) -> dict:
    """An id whose maker exits as soon as it is printed (Q a2's server)."""
    return {"mode": "mint", "id": Nccl().mint().hex()}


def join(args) -> dict:
    """Q b: a group of two formed one of three ways, then its all-reduces eager and graphed.

    ``vllm``: vLLM's blocking ``ncclCommInitRank`` (``PyNcclCommunicator.from_unique_id_bytes``),
    the control. ``blocking``: ``ncclCommInitRankConfig`` with ``blocking = 1``, which isolates
    the configuration's path. ``nonblocking``: with ``blocking = 0``, the question. A blocking
    join that never forms is ended by the caller's ``timeout``.
    """
    import torch

    nccl = Nccl()
    torch.cuda.set_device(0)
    if args.how == "rounds":
        if args.rank == 0:
            ids = [nccl.mint(), nccl.mint()]
            print(f"id {','.join(i.hex() for i in ids)}", flush=True)
        else:
            ids = [bytes.fromhex(h) for h in args.id.split(",")]
        return rounds(nccl, torch, args.rank, ids, args.rounds)
    if args.rank == 0:
        uid = nccl.mint()
        print(f"id {uid.hex()}", flush=True)
    else:
        uid = bytes.fromhex(args.id)
    out = {"mode": "join", "how": args.how, "rank": args.rank, "nccl": nccl.version()}
    t0 = time.monotonic()
    if args.how == "vllm":
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        comm = PyNcclCommunicator.from_unique_id_bytes(uid, args.rank, 2, 0).comm
        state, polls = NCCL_SUCCESS, 0
    else:
        comm, rc = nccl.start(2, uid, args.rank, blocking=args.how == "blocking")  # nonblocking*
        out["init_rc"] = rc
        state, _, polls = nccl.wait(comm, args.deadline_s)
    out |= {"state": state, "formed_s": round(time.monotonic() - t0, 3), "polls": polls}
    if state != NCCL_SUCCESS:
        out["abort_rc"], out["abort_s"] = nccl.aborted(comm)
        return out
    parent = None
    if args.how.startswith("split"):
        if args.how == "split-missing" and args.rank == 1:
            out["left"] = "formed the parent and left without splitting"
            return out
        parent = comm
        comm, out["split"] = split(nccl, parent, args.rank, args.split_deadline_s)
        if out["split"]["state"] != NCCL_SUCCESS:
            if comm.value:
                out["child_abort_rc"], out["child_abort_s"] = nccl.aborted(comm)
            out["abort_rc"], out["abort_s"] = nccl.aborted(parent)
            out["device_ok"] = device_still_serves()
            return out
    poll = args.how == "nonblocking-polled"
    out["eager"] = collectives(nccl, comm, torch, graphed=False, poll=poll)
    out["graphed"] = collectives(nccl, comm, torch, graphed=True, poll=poll)
    if args.how in ("vllm", "blocking", "nonblocking-polled", "split"):
        out["cost"] = cost(nccl, comm, torch, poll=poll)
    out["abort_rc"], out["abort_s"] = nccl.aborted(comm)
    if parent is not None:
        out["parent_abort_rc"], out["parent_abort_s"] = nccl.aborted(parent)
    return out


def split(nccl, parent, rank: int, deadline_s: float):
    """Q (C): a blocking child split from a nonblocking parent, waited on to a deadline.

    The split is a collective over the parent's members. On a nonblocking parent NCCL 2.29
    returns at once, ``ncclSuccess`` included, before the child exists, and fills in the child's
    handle when the split completes (probe 3). So the split is done only when the child has a
    handle and the child's own state reads ``ncclSuccess``. Until the handle comes, the parent's
    state is read, and an error there ends the wait.
    """
    t0 = time.monotonic()
    child, rc = nccl.split_blocking(parent, rank)
    polls = 0
    state = rc
    while time.monotonic() - t0 < deadline_s and state in (NCCL_SUCCESS, NCCL_IN_PROGRESS):
        if child.value:
            state = nccl.state(child)
            if state == NCCL_SUCCESS:
                break
        else:
            state = nccl.state(parent)
            if state == NCCL_SUCCESS:
                state = NCCL_IN_PROGRESS  # the parent is fine and the child is still to come
        time.sleep(0.001)
        polls += 1
    return child, {
        "rc": rc,
        "state": state,
        "s": round(time.monotonic() - t0, 3),
        "polls": polls,
        "child": bool(child.value),
    }


# hidden 896 (Qwen2.5-0.5B) in bf16 at batch sizes vLLM captures, and a 64 MiB all-reduce
SIZES = [896 * b for b in (1, 8, 64, 512)] + [32 * 1024 * 1024]
NCCL_BF16, NCCL_SUM = 9, 0


# decode's all-reduces: hidden 896 at batch 1 and 8, bf16
DECODE = [896, 896 * 8]


ARMS = ("V", "P", "C", "W")
"""vLLM's blocking communicator, the nonblocking parent used polled as (A), its blocking child
split as (C), and V called through the capture guard's per-call work (W)."""
REPLAYED = ("V", "P", "C")
"""The arms timed in graph replays too. W is eager only: inside a capture the guard refuses."""


class Guarded:
    """The capture guard's per-call work around a communicator.

    A delegated attribute and a check that no stream is capturing: what the wrapper the (C)
    design puts on vLLM's handles does on every call.
    """

    def __init__(self, comm, torch) -> None:
        """Guard ``comm``, reading the capture state through ``torch``."""
        self.wrapped = types.SimpleNamespace(comm=comm)
        self.torch = torch

    def __getattr__(self, name: str):
        """Every attribute is the wrapped communicator's."""
        return getattr(self.wrapped, name)

    def checked(self):
        """The communicator, refused while a stream is capturing."""
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("a collective inside a CUDA graph's capture")
        return self.comm


BATCHES = 7
EAGER_CALLS = 1000
GRAPH_CALLS = 50
REPLAYS = 20


def arm_order(round_index: int, arms=ARMS) -> list[str]:
    """The arms' order in a round, rotated so that each leads in turn.

    Both ranks compute the same order, so their calls match.
    """
    k = round_index % len(arms)
    return list(arms[k:]) + list(arms[:k])


def summary(values: list[float]) -> dict:
    """A list of per-round values: their median, the interquartile range, and the extremes."""
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": round(statistics.median(values), 2),
        "iqr": round(q3 - q1, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def gpu() -> dict:
    """The GPU, its driver and its SM clock, from ``nvidia-smi``. The probe locks no clocks."""
    q = "name,driver_version,clocks.sm,clocks.max.sm"
    try:
        line = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return dict(zip(q.split(","), (f.strip() for f in line.split(",")), strict=False)) | {
        "clocks_locked": False
    }


def rounds(nccl, torch, rank: int, ids: list[bytes], n_rounds: int) -> dict:
    """Probe 5: the three communicators in one process pair, timed in interleaved rounds.

    Each round runs every arm in ``arm_order``. For every size, an arm runs one discarded batch,
    then ``BATCHES`` eager batches of ``EAGER_CALLS`` calls, each synchronised at its end. Then it
    runs one discarded replay, then ``BATCHES`` batches of ``REPLAYS`` replays of a graph of
    ``GRAPH_CALLS`` calls. The round's value is the median of its batches, in microseconds per
    call. The graphs are captured once, before the rounds.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    torch.cuda.set_device(0)
    out: dict = {"mode": "rounds", "rank": rank, "nccl": nccl.version(), "gpu": gpu()}
    vllm_comm = PyNcclCommunicator.from_unique_id_bytes(ids[0], rank, 2, 0)
    parent, rc = nccl.start(2, ids[1], rank, blocking=False)
    state, _, _ = nccl.wait(parent, 60.0)
    if state != NCCL_SUCCESS:
        return out | {"error": f"the parent did not form: init {rc}, state {state}"}
    child, read = split(nccl, parent, rank, 20.0)
    out["split"] = read
    if read["state"] != NCCL_SUCCESS or not child.value:
        return out | {"error": "the split did not complete"}
    comms = {"V": (vllm_comm.comm, False), "P": (parent, True), "C": (child, False)}
    guarded = Guarded(vllm_comm.comm, torch)
    stream = torch.cuda.Stream()
    bufs = {n: torch.zeros(n, device="cuda", dtype=torch.bfloat16) for n in DECODE}
    torch.cuda.synchronize()

    def call(arm: str, buf) -> int:
        if arm == "W":
            comm, poll = guarded.checked(), False
        else:
            comm, poll = comms[arm]
        rc = nccl.all_reduce(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_void_p(buf.data_ptr()),
            buf.numel(),
            NCCL_BF16,
            NCCL_SUM,
            comm,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if poll and rc == NCCL_IN_PROGRESS:
            rc, _, _ = nccl.wait(comm, 30.0, poll_s=0.0)
        return rc

    bad: list[tuple[str, int, int]] = []

    def capture(arm: str, n: int):
        """The arm's graph at size ``n``, after its warm calls.

        Held only by ``graphs``. A communicator's abort waits for every graph captured on it to
        be freed: probe 5's first run hung in the child's abort on a capture variable left alive.
        """
        for _ in range(20):
            call(arm, bufs[n])
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(GRAPH_CALLS):
                rc = call(arm, bufs[n])
                if rc != NCCL_SUCCESS:
                    bad.append((arm, n, rc))
        return graph

    graphs = {(arm, n): capture(arm, n) for arm in REPLAYED for n in DECODE}
    torch.cuda.synchronize()

    def eager_batch(arm: str, n: int) -> float:
        t0 = time.perf_counter()
        for _ in range(EAGER_CALLS):
            rc = call(arm, bufs[n])
            if rc != NCCL_SUCCESS:
                bad.append((arm, n, rc))
        stream.synchronize()
        return (time.perf_counter() - t0) / EAGER_CALLS * 1e6

    def replay_batch(arm: str, n: int) -> float:
        t0 = time.perf_counter()
        for _ in range(REPLAYS):
            graphs[arm, n].replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / (REPLAYS * GRAPH_CALLS) * 1e6

    values = {arm: {n: {"eager": [], "replay": []} for n in DECODE} for arm in ARMS}
    for r in range(n_rounds):
        for arm in arm_order(r):
            for n in DECODE:
                eager_batch(arm, n)  # discarded: the first calls after a switch of communicator
                values[arm][n]["eager"].append(
                    statistics.median(eager_batch(arm, n) for _ in range(BATCHES))
                )
                if arm not in REPLAYED:
                    continue
                replay_batch(arm, n)  # discarded, as the eager one
                values[arm][n]["replay"].append(
                    statistics.median(replay_batch(arm, n) for _ in range(BATCHES))
                )
        # each round's figures as it ends: a later failure cannot lose the rounds measured
        line = {
            arm: {
                str(n): {m: round(vs[-1], 2) for m, vs in by.items() if vs}
                for n, by in sizes.items()
            }
            for arm, sizes in values.items()
        }
        print(json.dumps({"round": r, "order": arm_order(r), "us": line}), flush=True)
    out["arms"] = {
        arm: {
            str(n): {
                mode: {"rounds": [round(v, 2) for v in vs], **summary(vs)}
                for mode, vs in by.items()
                if vs  # W has no replays
            }
            for n, by in sizes.items()
        }
        for arm, sizes in values.items()
    }
    out["bad_calls"] = bad[:20]
    out["all_success"] = not bad
    out["gpu_after"] = gpu()
    graphs.clear()  # every graph captured on a communicator goes before its abort (vLLM's destroy)
    torch.cuda.synchronize()
    out["aborts"] = {name: bounded_abort(nccl, c) for name, c in (("C", child), ("P", parent))}
    return out


def bounded_abort(nccl, comm, timeout_s: float = 5.0) -> dict:
    """``ncclCommAbort`` on a thread, waited on for ``timeout_s`` at most.

    As vLLM's ``PyNcclCommunicator.destroy`` does: an abort may wait on graphs captured on the
    communicator, and the answer must not wait on the abort.
    """
    got: dict = {}

    def abort() -> None:
        rc, took = nccl.aborted(comm)
        got.update(rc=rc, s=took)

    t = threading.Thread(target=abort, daemon=True)
    t.start()
    t.join(timeout_s)
    return got if got else {"returned": False, "waited_s": timeout_s}


def cost(nccl, comm, torch, poll: bool, eager_calls: int = 200, graph_calls: int = 50) -> dict:
    """Microseconds per all-reduce at decode sizes, issued eagerly and replayed from a graph.

    A poll sits on the host's path only where the host issues the call: eagerly, not in a replay.
    """
    stream = torch.cuda.Stream()
    out: dict = {}
    for n in DECODE:
        buf = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()

        def call(buf=buf):
            rc = nccl.all_reduce(
                ctypes.c_void_p(buf.data_ptr()),
                ctypes.c_void_p(buf.data_ptr()),
                buf.numel(),
                NCCL_BF16,
                NCCL_SUM,
                comm,
                ctypes.c_void_p(stream.cuda_stream),
            )
            if poll and rc == NCCL_IN_PROGRESS:
                rc, _, _ = nccl.wait(comm, 30.0, poll_s=0.0)
            return rc

        for _ in range(20):  # warm
            call()
        stream.synchronize()
        t0 = time.perf_counter()
        rcs = [call() for _ in range(eager_calls)]
        stream.synchronize()
        eager_us = (time.perf_counter() - t0) / eager_calls * 1e6
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            rcs += [call() for _ in range(graph_calls)]
        graph.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        replay_us = (time.perf_counter() - t0) / (20 * graph_calls) * 1e6
        out[str(n)] = {
            "eager_us": round(eager_us, 1),
            "replay_us": round(replay_us, 1),
            "all_success": all(rc == NCCL_SUCCESS for rc in rcs),
        }
    return out


def collectives(nccl, comm, torch, graphed: bool, poll: bool = False) -> dict:
    """Every all-reduce's return code, and whether the sums are right.

    Eager on a side stream, or captured once in a CUDA graph and replayed three times.
    """
    stream = torch.cuda.Stream()
    bufs = [torch.ones(n, device="cuda", dtype=torch.bfloat16) for n in SIZES]
    rcs: list[int] = []
    polled: list[int] = []
    torch.cuda.synchronize()

    def issue():
        for b in bufs:
            rc = nccl.all_reduce(
                ctypes.c_void_p(b.data_ptr()),
                ctypes.c_void_p(b.data_ptr()),
                b.numel(),
                NCCL_BF16,
                NCCL_SUM,
                comm,
                ctypes.c_void_p(stream.cuda_stream),
            )
            if poll and rc == NCCL_IN_PROGRESS:
                # NCCL's contract for a nonblocking communicator: a call answered in progress
                # is complete once the communicator's state leaves it, polled before the next
                rc, _, _ = nccl.wait(comm, 30.0, poll_s=0.0005)
                polled.append(rc)
            rcs.append(rc)

    out: dict = {}
    try:
        if graphed:
            issue()  # vLLM warms a size eagerly before it captures it
            stream.synchronize()
            for b in bufs:
                b.fill_(1)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                issue()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            # the capture runs nothing; each of the three replays doubles every element
            want = 2.0**3
        else:
            with torch.cuda.stream(stream):
                issue()
            stream.synchronize()
            want = 2.0
        out["sums_right"] = all(
            float(b[0].item()) == want and float(b[-1].item()) == want for b in bufs
        )
    except RuntimeError as e:  # CUDA's and capture's errors: the answer is what failed
        out["error"] = f"{type(e).__name__}: {e}"
    out["rcs"] = rcs
    out["polled"] = len(polled)
    out["all_success"] = bool(rcs) and all(rc == NCCL_SUCCESS for rc in rcs)
    return out


def main(argv=None) -> int:
    """Run one mode and print its answer as one JSON line."""
    p = argparse.ArgumentParser(prog="nccl_probe")
    sub = p.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("missing")
    m.add_argument("--as", dest="as_rank", type=int, default=0)
    m.add_argument("--id")
    m.add_argument("--deadline-s", type=float, default=10.0)
    sub.add_parser("mint")
    j = sub.add_parser("join")
    j.add_argument("--rank", type=int, required=True)
    j.add_argument(
        "--how",
        choices=(
            "vllm",
            "blocking",
            "nonblocking",
            "nonblocking-polled",
            "split",
            "split-missing",
            "rounds",
        ),
        default="nonblocking",
    )
    j.add_argument("--id")
    j.add_argument("--deadline-s", type=float, default=120.0)
    j.add_argument("--split-deadline-s", type=float, default=20.0)
    j.add_argument("--rounds", type=int, default=12)
    args = p.parse_args(argv)
    print(
        json.dumps({"missing": missing, "mint": mint, "join": join}[args.mode](args)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
