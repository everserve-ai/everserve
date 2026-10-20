# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Re-form a four-rank group at constant world size with a pre-loaded replacement.

CPU gloo workers stand in for the tensor-parallel ranks. Each rank owns a weight shard and a
block-paged KV shard; a decode step writes this step's KV rows, all-reduces the ranks' partial
outputs, and appends the token. The controller plays the executor: it drives steps over pipes,
SIGKILLs rank 2 between steps, and re-forms the group with the replacement as rank 2, either
replaying every token (spare) or after staging the doomed rank's KV during a notice window
(notice). Tokens and every rank's KV must equal a no-fault run.
"""

import multiprocessing as mp
import os
import signal
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from ports import free_port

from everserve.reference.staging import blocks_to_stage
from everserve.reference.world import join_world, leave_world

pytestmark = pytest.mark.slow

W, H, BS, NB, PROMPT, STEPS = 4, 8, 4, 64, 6, 24
KILL, NOTICE, R = 12, 6, 2
TIMEOUT = timedelta(seconds=30)


def weights(rank):
    g = torch.Generator().manual_seed(100 + rank)
    return torch.randn(H, H, generator=g) / H**0.5


class Rank:
    def __init__(self, rank):
        self.rank, self.w = rank, weights(rank)
        self.kv = torch.zeros(NB, BS, H)

    def write_kv(self, x, pos, table):
        self.kv[table[pos // BS], pos % BS] = x @ self.w

    def partial(self, x, n, table):
        rows = torch.stack([self.kv[table[p // BS], p % BS] for p in range(n)])
        att = torch.softmax(rows @ x, 0) @ rows
        return att + x @ self.w


def embed(tok):
    g = torch.Generator().manual_seed(tok)
    return torch.randn(H, generator=g)


def worker(rank, init_method, conn, preload_only=False):
    st = Rank(rank)
    if preload_only:
        conn.send(("loaded", rank))
        _, init_method, staged = conn.recv()
        if staged is not None:
            for b, rows in staged.items():
                st.kv[b] = rows
    join_world(init_method, rank, W, "gloo", timeout=TIMEOUT)
    conn.send(("ready", rank))
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "step":
            _, toks, start, table = msg
            for p, t in enumerate(toks, start):
                st.write_kv(embed(t), p, table)
            n = start + len(toks)
            y = st.partial(embed(toks[-1]), n, table)
            try:
                dist.all_reduce(y)
            except Exception as e:  # noqa: BLE001 - a peer died mid-collective
                conn.send(("failed", rank, str(e)[:80]))
                continue
            conn.send(("out", rank, int(torch.argmax(y).item()) % 97 + 3))
        elif kind == "stage":
            conn.send(("staged", rank, {b: st.kv[b].clone() for b in msg[1]}))
        elif kind == "reform":
            done = leave_world()
            join_world(msg[1], rank, W, "gloo", timeout=TIMEOUT)
            conn.send(("reformed", rank, done))
        elif kind == "dump":
            conn.send(("kv", rank, st.kv.clone()))
        elif kind == "exit":
            return


class Controller:
    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.pipes, self.procs = {}, {}
        init = f"tcp://127.0.0.1:{free_port()}"
        for r in range(W):
            a, p = self._spawn(r, init)
            self.pipes[r], self.procs[r] = a, p
        for r in range(W):
            assert self.pipes[r].recv()[0] == "ready"
        self.table = list(range(NB))[::-1]
        self.tokens = [5 + i for i in range(PROMPT)]
        self.replacement = None

    def _spawn(self, r, init, preload=False):
        a, b = self.ctx.Pipe()
        p = self.ctx.Process(target=worker, args=(r, init, b, preload), daemon=True)
        p.start()
        return a, p

    def spawn_replacement(self):
        a, p = self._spawn(R, None, preload=True)
        assert a.recv() == ("loaded", R)
        self.replacement = (a, p)

    def step(self, toks, start):
        for r in self.pipes:
            self.pipes[r].send(("step", toks, start, self.table))
        outs = [self.pipes[r].recv() for r in sorted(self.pipes)]
        assert all(o[0] == "out" for o in outs), outs
        assert len({o[2] for o in outs}) == 1
        return outs[0][2]

    def kill(self, r):
        os.kill(self.procs[r].pid, signal.SIGKILL)
        self.procs[r].join()
        del self.pipes[r], self.procs[r]

    def reform(self, staged=None):
        init = f"tcp://127.0.0.1:{free_port()}"
        for r in self.pipes:
            self.pipes[r].send(("reform", init))
        a, p = self.replacement
        a.send(("join", init, staged))
        outs = [self.pipes[r].recv() for r in sorted(self.pipes)]
        assert all(o[0] == "reformed" for o in outs), outs
        assert all("destroy" in o[2] for o in outs), outs
        assert a.recv() == ("ready", R)
        self.pipes[R], self.procs[R] = a, p

    def dump(self):
        for r in self.pipes:
            self.pipes[r].send(("dump",))
        return {o[1]: o[2] for o in (self.pipes[r].recv() for r in sorted(self.pipes))}

    def close(self):
        for r in self.pipes:
            self.pipes[r].send(("exit",))
        for p in self.procs.values():
            p.join(10)


def run(mode):
    """mode: none | spare | notice. Returns tokens and the final KV of every rank."""
    c = Controller()
    if mode != "none":
        c.spawn_replacement()
    c.tokens.append(c.step(c.tokens, 0))
    staged, staged_kv = {}, {}
    for s in range(1, STEPS):
        if mode == "notice" and s >= NOTICE:
            ids = blocks_to_stage({"0": c.table}, {"0": len(c.tokens) - 1}, staged, BS)
            c.pipes[R].send(("stage", ids))
            staged_kv.update(c.pipes[R].recv()[2])
        if mode != "none" and s == KILL:
            c.kill(R)
            c.reform(staged_kv if mode == "notice" else None)
            if mode == "spare":
                for r in c.pipes:
                    c.pipes[r].send(("step", c.tokens[:-1], 0, c.table))
                assert all(c.pipes[r].recv()[0] == "out" for r in sorted(c.pipes))
        c.tokens.append(c.step([c.tokens[-1]], len(c.tokens) - 1))
    kv = c.dump()
    c.close()
    return c.tokens, kv


def test_spare_and_notice_reform_match_no_fault():
    ref_tok, ref_kv = run("none")
    for mode in ("spare", "notice"):
        tok, kv = run(mode)
        assert tok == ref_tok, mode
        for r in range(W):
            assert torch.equal(kv[r], ref_kv[r]), (mode, r)


def test_without_staging_or_replay_the_lost_kv_is_wrong():
    """Negative control: with neither staging nor replay the replacement's KV is wrong."""
    _, ref_kv = run("none")
    c = Controller()
    c.spawn_replacement()
    c.tokens.append(c.step(c.tokens, 0))
    for s in range(1, KILL + 3):
        if s == KILL:
            c.kill(R)
            c.reform(None)
        c.tokens.append(c.step([c.tokens[-1]], len(c.tokens) - 1))
    kv = c.dump()
    c.close()
    assert not torch.equal(kv[R], ref_kv[R])
