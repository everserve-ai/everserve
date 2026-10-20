# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Microseconds per decode-size all-reduce: the slot collective, NCCL and stock's custom one.

    python bench/harness/slot_bench.py --ranks 4 --elements 2048 4096 8192 16384 32768

One process per GPU. Each times ``--iters`` calls captured in one CUDA graph and replayed, the way
a FULL decode graph runs them, for: the slot collective's peer-to-peer variant (at its configured
words per thread and at every ``--words`` override), its host-memory variant, NCCL (vLLM's
pynccl), and vLLM's custom all-reduce (``CustomAllreduce``, the one-shot NVLink reduction stock
runs at these sizes; decision 0012 makes it the target). Prints one JSON line per backend and
size, measured on rank 0.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp


def _time_graph(fn, iters: int) -> float:
    """Microseconds per call of ``fn`` captured ``iters`` times in one graph."""
    fn()  # outside the graph: first-call setup
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    return _replay(g, iters)


def _replay(g: torch.cuda.CUDAGraph, iters: int) -> float:
    g.replay()  # warm
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    return 1e3 * start.elapsed_time(end) / iters


def _time_custom(ca, x: torch.Tensor, iters: int) -> float:  # noqa: ANN001
    """The custom all-reduce as vLLM runs it in a graph: inputs registered when capture ends."""
    ca.custom_all_reduce(x)  # eager: through its staging buffer
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(g):
            for _ in range(iters):
                ca.custom_all_reduce(x)
    return _replay(g, iters)


def _rank(rank: int, a: argparse.Namespace, paths: dict, out_q) -> None:  # noqa: ANN001
    import torch.distributed as dist

    from everserve.comm.slotkernel import Region, SlotKernel

    torch.cuda.set_device(rank)
    kernels = {}
    for variant, path in paths.items():
        with torch.inference_mode():
            kernels[variant] = SlotKernel(Region(path), rank, rank)  # one mapping per process
        kernels[variant].set_members(range(a.ranks), 0)
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host="127.0.0.1", port=a.port, rank=rank, world_size=a.ranks)
    comm = PyNcclCommunicator(pg, device=torch.device("cuda", rank))
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{a.port + 1}", rank=rank, world_size=a.ranks
    )
    ca = CustomAllreduce(dist.group.WORLD, torch.device("cuda", rank))
    results = []
    for n in a.elements:
        x = torch.randn(n, device="cuda").to(torch.bfloat16)
        y = torch.empty_like(x)
        row: dict = {"elements": n}
        for variant, k in kernels.items():
            row[variant] = _time_graph(lambda k=k, x=x, y=y: k.all_reduce_into(x, y), a.iters)
            if variant == "p2p":
                configured = k._words
                for words in a.words:  # words-per-thread overrides: a measurement setting
                    k._words = (words, words)
                    row[f"p2p_words{words}"] = _time_graph(
                        lambda k=k, x=x, y=y: k.all_reduce_into(x, y), a.iters
                    )
                k._words = configured
        row["nccl"] = _time_graph(lambda x=x, y=y: comm.all_reduce(x, out_tensor=y), a.iters)
        row["custom_ar"] = (
            _time_custom(ca, x, a.iters) if not ca.disabled and ca.should_custom_ar(x) else None
        )
        results.append(row)
    if rank == 0:
        out_q.put(results)


def main() -> None:
    """Run the comparison and print it."""
    from everserve.comm.slotkernel import Region

    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--elements", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--port", type=int, default=29533)
    ap.add_argument("--words", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    a = ap.parse_args()
    root = Path(tempfile.mkdtemp(dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    paths = {}
    for variant in ("p2p", "host"):
        paths[variant] = str(root / variant)
        region = Region(paths[variant], create=True, variant=variant)
        region.set_members(range(a.ranks), range(a.ranks))  # the executor's half
        region.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_rank, args=(r, a, paths, q)) for r in range(a.ranks)]
    for p in procs:
        p.start()
    res = q.get(timeout=1200)
    for p in procs:
        p.join(60)
    for row in res:
        n = row.pop("elements")
        for backend, us in row.items():
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "ranks": a.ranks,
                        "elements": n,
                        "us_per_call": round(us, 2) if us is not None else None,
                    }
                )
            )


if __name__ == "__main__":
    main()
