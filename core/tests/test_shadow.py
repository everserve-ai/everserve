# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Shadow groups formed beside the live group; a fault is a pointer swap.

CPU gloo processes stand in for the TP ranks and the spare. While healthy, every member set a
single loss can need is formed on its own store beside the live default group. A "captured graph"
reaches its collective through one GroupHandle. After rank 2 is SIGKILLed, survivors and the
replacement swap the handle to the replacement set's group, with no new rendezvous, and the graph
replays right. A survivor set is also exercised on its own.
"""

import multiprocessing as mp
import os
import signal
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from ports import free_ports

from everserve.comm.handle import GroupHandle
from everserve.comm.shadow import form_groups, member_sets, replacement, set_key

pytestmark = pytest.mark.slow

W, H, R = 4, 8, replacement(2)
SETS = member_sets(W, replacements=[(2,)])
TIMEOUT = timedelta(seconds=30)


def weight(me):
    seed = 7 + (2 if me == R else me)
    return torch.randn(H, H, generator=torch.Generator().manual_seed(seed)) / H


def x_for(step):
    return torch.randn(H, generator=torch.Generator().manual_seed(500 + step))


def expected(step):
    return 2 * sum(x_for(step) @ weight(r) for r in range(W))


def capture(me, handle):
    x, a, b = torch.zeros(H), torch.empty(H), torch.empty(H)
    w = weight(me)
    ops = [lambda: a.copy_(x @ w), lambda: handle.all_reduce_into(a, a), lambda: b.copy_(a * 2)]

    def replay(xv):
        x.copy_(xv)
        for op in ops:
            op()
        return b.clone()

    return replay


def worker(me, live_port, shadow_ports, conn):
    handle = GroupHandle()
    if me != R:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{live_port}",
            rank=me,
            world_size=W,
            timeout=TIMEOUT,
        )
        handle.swap(dist.group.WORLD)
    shadows = form_groups(me, SETS, "127.0.0.1", shadow_ports, timeout=TIMEOUT)
    graph = capture(me, handle)
    conn.send(("ready", sorted(set_key(s) for s in shadows)))
    while True:
        msg = conn.recv()
        if msg[0] == "step":
            conn.send(("ok", graph(x_for(msg[1]))))
        elif msg[0] == "swap":
            handle.swap(shadows[msg[1]])
            conn.send("swapped")
        elif msg[0] == "sum":
            t = torch.ones(1) * (me + 1)
            dist.all_reduce(t, group=shadows[msg[1]])
            conn.send(float(t))
        elif msg[0] == "exit":
            return


def test_member_sets_and_keys():
    assert SETS == [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2), (0, 1, R, 3)]
    assert set_key(SETS[-1]) == "0-1-R2-3"
    assert member_sets(3) == [(1, 2), (0, 2), (0, 1)]


def test_shadow_groups_beside_live_and_swap_without_rendezvous():
    ctx = mp.get_context("spawn")
    live_port, *shadow_ports = free_ports(1 + len(SETS))
    pipes, procs = {}, {}
    for me in (0, 1, 2, 3, R):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(me, live_port, shadow_ports, b), daemon=True)
        p.start()
        pipes[me], procs[me] = a, p
    formed = {me: pipes[me].recv() for me in pipes}
    assert all(f[0] == "ready" for f in formed.values())
    assert len(formed[0][1]) == 4 and formed[R][1] == ["0-1-R2-3"]

    def step(k, who):
        for me in who:
            pipes[me].send(("step", k))
        return {me: pipes[me].recv() for me in who}

    for k in (1, 2):
        out = step(k, [0, 1, 2, 3])
        assert all(torch.allclose(o[1], expected(k), atol=1e-5) for o in out.values())
    for me in (0, 1, 3):
        pipes[me].send(("sum", (0, 1, 3)))
    assert {pipes[me].recv() for me in (0, 1, 3)} == {1.0 + 2.0 + 4.0}

    os.kill(procs[2].pid, signal.SIGKILL)
    procs[2].join()
    for me in (0, 1, 3, R):
        pipes[me].send(("swap", (0, 1, R, 3)))
    assert all(pipes[me].recv() == "swapped" for me in (0, 1, 3, R))
    for k in (3, 4, 5):
        out = step(k, [0, 1, 3, R])
        assert all(torch.allclose(o[1], expected(k), atol=1e-5) for o in out.values()), k
    for me in (0, 1, 3, R):
        pipes[me].send(("exit",))
    for me in (0, 1, 3, R):
        procs[me].join(10)
