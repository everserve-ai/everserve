# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The Collective on gloo: a current group plus pre-formed shadows; a change is a pointer swap; a
captured graph that reaches the collective through it keeps replaying correctly."""

import multiprocessing as mp
import os
import signal
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from ports import free_ports

from everserve.comm.collective import Collective, CollectiveMoved, ranks_of
from everserve.comm.group import ShrinkableGroup
from everserve.comm.handle import LocalStandIn
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


def worker(me, live_port, shadow_ports, conn):
    if me != R:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{live_port}",
            rank=me,
            world_size=W,
            timeout=TIMEOUT,
        )
        live = ShrinkableGroup(dist.group.WORLD, list(range(W)), "gloo")
    else:
        live = LocalStandIn()
    shadows = form_groups(me, SETS, "127.0.0.1", shadow_ports, timeout=TIMEOUT)
    col = Collective(me, live, shadows, backend="gloo")
    x, a, b = torch.zeros(H), torch.empty(H), torch.empty(H)
    w = weight(me)
    ops = [lambda: a.copy_(x @ w), lambda: col.all_reduce_into(a, a), lambda: b.copy_(a * 2)]

    def replay(xv):
        x.copy_(xv)
        for op in ops:
            op()
        return b.clone()

    conn.send(("ready", sorted(set_key(s) for s in col.shadow_sets)))
    while True:
        msg = conn.recv()
        if msg[0] == "step":
            conn.send(("ok", replay(x_for(msg[1]))))
        elif msg[0] == "change":
            col.change(msg[1], msg[2])
            conn.send(("epoch", col.epoch, col.members))
        elif msg[0] == "exit":
            return


def test_rejects_a_shadow_set_without_me():
    with pytest.raises(ValueError):
        Collective(0, LocalStandIn(), {(1, 2): LocalStandIn()})


def test_a_spare_for_several_positions_holds_each_s_sets_and_keeps_the_one_it_takes():
    """Decision 0030: a spare standing for any of ranks 0 and 1 pre-forms both replacement sets,
    in both scopes; taking rank 1 makes it R1 and sets aside R0's sets."""
    from everserve.comm.collective import GroupFamily
    from everserve.comm.layout import Scope

    r0, r1 = (replacement(0), 1), (0, replacement(1))
    fabric = GroupFamily(Scope.FABRIC, LocalStandIn(), {r0: LocalStandIn(), r1: LocalStandIn()})
    col = Collective(
        replacement(0),
        LocalStandIn(),
        {r0: LocalStandIn()},
        fabric=fabric,
        stands_for=(replacement(1),),
    )
    col.preform({r1: LocalStandIn()})
    with pytest.raises(ValueError, match="does not contain"):
        col.preform({(0, 1): LocalStandIn()})
    assert col.has_shadow(r0) and col.has_shadow(r1)
    with pytest.raises(ValueError, match="not one of the ids"):
        col.take_identity(replacement(2))
    dropped = col.take_identity(replacement(1))
    assert [(scope, s) for scope, s, _ in dropped] == [("domain", r0), ("fabric", r0)]
    assert col.me == replacement(1) and col.stands_for == (replacement(1),)
    assert col.shadow_sets == [r1] and fabric.shadow_sets == [r1] and not col.has_shadow(r0)
    col.change(r1, 1)
    assert col.members == r1 and col.epoch == 1


def test_change_without_a_shadow_is_a_key_error_and_change_to_group_works():
    col = Collective(0, LocalStandIn(), {})
    with pytest.raises(KeyError):
        col.change((0, 2), 1)
    col.change_to_group((0, 2), LocalStandIn(), 1)
    assert col.members == (0, 2) and col.epoch == 1
    out = torch.empty(2)
    col.all_reduce_into(torch.tensor([1.0, 2.0]), out)
    assert out.tolist() == [1.0, 2.0]
    with pytest.raises(RuntimeError):
        col.shrink(2, 2)


def test_a_call_pinned_to_an_old_epoch_is_refused_after_a_move():
    col = Collective(0, LocalStandIn(), {})
    t = torch.ones(2)
    col.all_reduce(t, epoch=0)  # matches: fine
    col.change_to_group((0, 2), LocalStandIn(), 1)
    with pytest.raises(CollectiveMoved):
        col.all_reduce(t, epoch=0)
    col.all_reduce(t, epoch=1)
    col.all_reduce(t)  # unpinned callers are not checked


def test_ranks_of_gives_the_spare_the_replaced_ranks_id():
    assert ranks_of((0, 1, R, 3)) == [0, 1, 2, 3]
    assert ranks_of((0, 1, 3)) == [0, 1, 3]


def test_pointer_swap_to_the_replacement_set_keeps_captured_graphs_right():
    ctx = mp.get_context("spawn")
    live_port, *shadow_ports = free_ports(1 + len(SETS))
    pipes, procs = {}, {}
    for me in (0, 1, 2, 3, R):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(me, live_port, shadow_ports, b), daemon=True)
        p.start()
        pipes[me], procs[me] = a, p
    ready = {me: pipes[me].recv() for me in pipes}
    assert len(ready[0][1]) == 4 and ready[R][1] == ["0-1-R2-3"]

    def step(k, who):
        for me in who:
            pipes[me].send(("step", k))
        return {me: pipes[me].recv() for me in who}

    for k in (1, 2):
        out = step(k, [0, 1, 2, 3])
        assert all(torch.allclose(o[1], expected(k), atol=1e-5) for o in out.values())

    os.kill(procs[2].pid, signal.SIGKILL)
    procs[2].join()
    for me in (0, 1, 3, R):
        pipes[me].send(("change", (0, 1, R, 3), 1))
    replies = {me: pipes[me].recv() for me in (0, 1, 3, R)}
    assert all(r == ("epoch", 1, (0, 1, R, 3)) for r in replies.values())
    for k in (3, 4, 5):
        out = step(k, [0, 1, 3, R])
        assert all(torch.allclose(o[1], expected(k), atol=1e-5) for o in out.values()), k
    for me in (0, 1, 3, R):
        pipes[me].send(("exit",))
    for me in (0, 1, 3, R):
        procs[me].join(10)
