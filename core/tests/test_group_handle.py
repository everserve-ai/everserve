# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A captured graph survives a group re-form only if it reaches the group through a handle at
call time and the collective writes in place into a captured buffer.

CPU gloo ranks stand in for TP ranks. A "captured graph" is a fixed list of operations over
buffers fixed at capture time, replayed without re-tracing. Three ways to put the all-reduce in
it:

* baked: the process group object is bound at capture (what a CUDA graph does to a ncclComm);
* indirect: the group is read from a GroupHandle at call time, reducing in place;
* outofplace: read at call time, but a fresh tensor is returned while the next node reads the
  address seen at capture.

Rank 2 is SIGKILLed; survivors and a replacement (whose graph was captured against a stand-in
before it joined) re-form at world size 4. Only ``indirect`` replays correctly afterwards.
"""

import multiprocessing as mp
import os
import signal
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from ports import free_port

from everserve.comm.handle import GroupHandle, LocalStandIn
from everserve.reference.world import join_world, leave_world

pytestmark = pytest.mark.slow

W, H, R = 4, 16, 2
TIMEOUT = timedelta(seconds=30)


def weight(rank):
    return torch.randn(H, H, generator=torch.Generator().manual_seed(10 + rank)) / H


def x_for(step):
    return torch.randn(H, generator=torch.Generator().manual_seed(1000 + step))


def expected(step):
    return 2 * sum(x_for(step) @ weight(r) for r in range(W))


def all_reduce_new(handle, inp):
    out = inp.clone()
    if not isinstance(handle.group, LocalStandIn):
        dist.all_reduce(out, group=handle.group)
    return out


def capture(mode, rank, handle):
    x, a, b, c = torch.zeros(H), torch.empty(H), torch.empty(H), torch.empty(H)
    w = weight(rank)
    ops = [lambda: a.copy_(x @ w)]
    if mode == "baked":
        pg = handle.group
        ops.append(lambda: (b.copy_(a), dist.all_reduce(b, group=pg)))
        ops.append(lambda: c.copy_(b * 2))
    elif mode == "indirect":
        ops.append(lambda: handle.all_reduce_into(a, b))
        ops.append(lambda: c.copy_(b * 2))
    else:
        seen = all_reduce_new(handle, a)
        ops.append(lambda: all_reduce_new(handle, a))
        ops.append(lambda: c.copy_(seen * 2))

    def replay(xv):
        x.copy_(xv)
        for op in ops:
            op()
        return c.clone()

    return replay


def worker(rank, init, conn, replacement=False):
    handle = GroupHandle()
    if replacement:
        handle.swap(LocalStandIn())
        graphs = {"indirect": capture("indirect", rank, handle)}
        conn.send("captured")
        init = conn.recv()
        join_world(init, rank, W, "gloo", timeout=TIMEOUT)
        handle.swap(dist.group.WORLD)
    else:
        join_world(init, rank, W, "gloo", timeout=TIMEOUT)
        handle.swap(dist.group.WORLD)
        graphs = {m: capture(m, rank, handle) for m in ("baked", "indirect", "outofplace")}
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg[0] == "replay":
            _, mode, step = msg
            try:
                conn.send(("ok", graphs[mode](x_for(step))))
            except Exception as e:  # noqa: BLE001 - the error kind is the test's observation
                conn.send(("error", type(e).__name__))
        elif msg[0] == "reform":
            leave_world()
            join_world(msg[1], rank, W, "gloo", timeout=TIMEOUT)
            handle.swap(dist.group.WORLD)
            conn.send("reformed")
        elif msg[0] == "exit":
            return


def test_empty_handle_is_an_error():
    with pytest.raises(RuntimeError):
        GroupHandle().all_reduce_into(torch.zeros(1), torch.zeros(1))


def test_stand_in_copies_locally():
    h = GroupHandle(LocalStandIn())
    out = torch.empty(3)
    h.all_reduce_into(torch.tensor([1.0, 2.0, 3.0]), out)
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_indirect_graph_survives_reform_baked_and_outofplace_do_not():
    ctx = mp.get_context("spawn")
    pipes, procs = {}, {}
    init = f"tcp://127.0.0.1:{free_port()}"
    for r in range(W):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(r, init, b), daemon=True)
        p.start()
        pipes[r], procs[r] = a, p
    for r in range(W):
        assert pipes[r].recv() == "ready"
    ra, rb = ctx.Pipe()
    rp = ctx.Process(target=worker, args=(R, None, rb, True), daemon=True)
    rp.start()
    assert ra.recv() == "captured"

    def replay(mode, step):
        for r in pipes:
            pipes[r].send(("replay", mode, step))
        return {r: pipes[r].recv() for r in sorted(pipes)}

    for step in (1, 2):
        for mode in ("baked", "indirect"):
            out = replay(mode, step)
            assert all(
                o[0] == "ok" and torch.allclose(o[1], expected(step), atol=1e-5)
                for o in out.values()
            ), mode
        out = replay("outofplace", step)
        assert not all(torch.allclose(o[1], expected(step), atol=1e-5) for o in out.values())

    os.kill(procs[R].pid, signal.SIGKILL)
    procs[R].join()
    del pipes[R], procs[R]
    init = f"tcp://127.0.0.1:{free_port()}"
    for r in pipes:
        pipes[r].send(("reform", init))
    ra.send(init)
    assert all(pipes[r].recv() == "reformed" for r in sorted(pipes))
    assert ra.recv() == "ready"
    pipes[R], procs[R] = ra, rp

    for step in (3, 4, 5):
        out = replay("indirect", step)
        assert all(
            o[0] == "ok" and torch.allclose(o[1], expected(step), atol=1e-5) for o in out.values()
        )
    for r in pipes:
        if r != R:
            pipes[r].send(("replay", "baked", 6))
    errs = [pipes[r].recv() for r in sorted(pipes) if r != R]
    assert all(e[0] == "error" for e in errs), errs
    for r in pipes:
        pipes[r].send(("exit",))
    for p in procs.values():
        p.join(10)
