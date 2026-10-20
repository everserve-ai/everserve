# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The slot collective on CUDA, both variants, against ``everserve.reference.arith.ops.member_sum``.

Three ranks join one region and reduce at several sizes; every result must equal the reference
reduction of the members' inputs, in rank order, bit for bit. Then a REPLACE: rank 1 stops, the
others spin on it until the abort word releases them, the parent (in the executor's place) writes
the new order with the spare's row in rank 1's position, every member applies it at a boundary,
and the survivors replay a CUDA graph captured before the change while the spare, holding rank 1's
input, joins the call. The result must equal the reference over the new order and the result
before the change, bit for bit, with no recapture. While they spin, each survivor's kernel names
the dead rank's row in its late word's mask once the wait passes the bound, and clears it on release
(decision 0021). On the host-memory variant the dead rank first delivers part of that call's words,
block 0's share, as a push cut short by a kill would: only the survivors' other block still waits,
and it must name the dead rank. Before that replay the parent overwrites the
region's order with the row order: the kernel reads each member's device copy, taken at the
boundary, so the overwrite must not change the sum.

The graph's inputs are chosen so the order of the sum shows: rank 0 holds large values, rank 2
their negations, rank 1 small ones. In rank order the small values are absorbed by the large ones
before they cancel; in region-row order (0, 2, then the spare's row) they survive. A kernel that
sums in row order fails this test. Skipped without CUDA; the peer-to-peer variant also needs four
devices joined by NVLink.
"""

import multiprocessing as mp
import tempfile
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

RANKS = [0, 1, 2]  # rank i on region row i
DEAD = 1
SPARE_ROW = 3
AFTER = [0, SPARE_ROW, 2]  # rank positions after the REPLACE: the spare's row in rank 1's place
SIZES = [16384, 3000, 2]  # a decode step, a size that leaves a block partly used, the smallest
N = SIZES[0]
TIMEOUT_S = 120


def _input(rank, n):
    g = torch.Generator().manual_seed(1000 * rank + n)
    return torch.randn(n, generator=g).to(torch.bfloat16)


def _graph_input(rank):
    """Rank 0 large, rank 2 its negation, rank 1 small: the bf16 sum depends on the order."""
    g = torch.Generator().manual_seed(7)
    big = torch.randn(N, generator=g) * 2.0**20
    small = torch.randn(N, generator=g) * 2.0**-6
    return {0: big, 1: small, 2: -big}[rank].to(torch.bfloat16)


def _bits(t):
    """The tensor's raw bits as a NumPy array: pickled by value, unlike a torch CPU tensor, whose
    storage is shared through the sending process and gone once that process exits."""
    return t.cpu().view(torch.int16).numpy()


def _rank(path, rank, device, everyone, applied, ordered, cleared, results):
    from everserve.comm.slotkernel import Region, SlotKernel

    torch.cuda.set_device(device)
    with torch.inference_mode():  # as the engine makes it: inside its first forward
        k = SlotKernel(Region(path), rank, device)
    k.set_members(RANKS, 0)
    for n in SIZES:
        x = _input(rank, n).cuda(device)
        out = torch.empty_like(x)
        k.all_reduce_into(x, out)
        torch.cuda.synchronize()
        results.put((rank, f"size{n}", _bits(out)))
    x = _graph_input(rank).cuda(device)
    graph_out = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        k.all_reduce_into(x, graph_out)
    graph.replay()
    torch.cuda.synchronize()
    results.put((rank, "graph", _bits(graph_out)))
    everyone.wait(TIMEOUT_S)  # every rank has replayed once, and the spare is up
    if rank == DEAD:
        if k.variant == "host":
            _push_part_way(k)
        results.put((rank, "gone", None))
        return
    # the dead rank never joins this call: the survivors spin until the parent's abort word
    t0 = time.time()
    graph.replay()
    torch.cuda.synchronize()
    results.put((rank, "released", time.time() - t0))
    assert ordered.wait(TIMEOUT_S)  # the executor's order is in the region
    k.resume()
    k.set_members(AFTER, 1)
    applied.wait(TIMEOUT_S)  # every member of the new order zeroed and checked
    results.put((rank, "applied", None))
    assert cleared.wait(TIMEOUT_S)  # the parent has cleared the abort words, as the executor does
    graph.replay()  # the graph captured before the change, now over the new order
    torch.cuda.synchronize()
    results.put((rank, "after", _bits(graph_out)))
    k.close()


def _push_part_way(k):
    """The dead rank's next call, cut short: its words for block 0 delivered, none for block 1.

    A killed member's push can stop part way. The survivors' block 0 then has every word and
    finishes; only block 1 still waits, so the late member must be named by block 1, not by a
    single timing block (decision 0021). Host-memory variant: a member's words are its own row
    of the region, so this process writes them as its kernel would.
    """
    from everserve.comm.slotkernel import MAXN

    seq = int(k.seq[0].item()) + 1  # block 0's next call; the survivors are on it
    per_block = 256 * 16  # threads x words per thread at this size, host variant
    at = (seq & 1) * (MAXN // 2)
    k.region.words[DEAD, at : at + per_block] = seq << 32


def _spare(path, device, everyone, applied, ordered, cleared, results):
    from everserve.comm.slotkernel import Region, SlotKernel

    torch.cuda.set_device(device)
    with torch.inference_mode():
        k = SlotKernel(Region(path), SPARE_ROW, device)
    try:  # before the executor writes the new order, this process's order is refused
        k.set_members(AFTER, 1)
        results.put((SPARE_ROW, "refused", False))
    except ValueError:
        results.put((SPARE_ROW, "refused", True))
    everyone.wait(TIMEOUT_S)
    assert ordered.wait(TIMEOUT_S)
    k.resume()
    k.set_members(AFTER, 1)
    applied.wait(TIMEOUT_S)
    results.put((SPARE_ROW, "applied", None))
    assert cleared.wait(TIMEOUT_S)
    x = _graph_input(DEAD).cuda(device)  # the replaced rank's input: the spare holds its state
    out = torch.empty_like(x)
    k.all_reduce_into(x, out)
    torch.cuda.synchronize()
    results.put((SPARE_ROW, "after", _bits(out)))
    k.close()


def _devices(variant, nvlink):
    rows = RANKS + [SPARE_ROW]
    n = torch.cuda.device_count()
    if variant == "host":
        return [r % n for r in rows]
    if n < len(rows):
        pytest.skip(f"the peer-to-peer variant needs {len(rows)} devices")
    if not nvlink(len(rows)):
        pytest.skip("the peer-to-peer variant needs NVLink: CUDA's peer access is not enough")
    return rows


@pytest.mark.parametrize("variant", ["host", "p2p"])
def test_both_variants_sum_in_rank_order_and_a_spare_in_place_keeps_the_sum(variant, nvlink):
    from everserve import kernels
    from everserve.comm.slotkernel import Region
    from everserve.reference.arith.ops import member_sum

    if not kernels.available():
        pytest.skip("the Everserve kernels did not build here")
    devices = _devices(variant, nvlink)
    shm = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    path = str(Path(tempfile.mkdtemp(dir=shm)) / "slot")
    region = Region(path, create=True, variant=variant)
    region.set_members(RANKS, RANKS)
    region.set_members([SPARE_ROW], [SPARE_ROW])  # the spare, alone until it joins
    ctx = mp.get_context("spawn")
    live = [r for r in RANKS if r != DEAD]
    everyone = ctx.Barrier(len(RANKS) + 1)
    applied = ctx.Barrier(len(AFTER))
    ordered, cleared = ctx.Event(), ctx.Event()
    results = ctx.Queue()
    sync = (everyone, applied, ordered, cleared, results)
    procs = [ctx.Process(target=_rank, args=(path, r, devices[r], *sync)) for r in RANKS]
    procs.append(ctx.Process(target=_spare, args=(path, devices[SPARE_ROW], *sync)))
    for p in procs:
        p.start()
    got = {}

    def until(what, count):
        while sum(1 for (_, w) in got if w == what) < count:
            row, w, value = results.get(timeout=TIMEOUT_S)
            got[(row, w)] = value

    try:
        until("graph", len(RANKS))
        region.set_bound(live, 0.05)  # read at the kernels' abort checks, while they spin
        time.sleep(1.0)  # the survivors are now spinning on the dead rank
        late = {r: region.late(r) for r in live}
        region.set_abort(live, 1)
        until("released", len(live))
        released = {r: region.late(r) for r in live}
        region.set_members(AFTER, AFTER)  # the executor's half: before any member's boundary
        ordered.set()
        until("applied", len(AFTER))
        # the members took their device copies at the boundary; the region is now only a record
        region.set_members(AFTER, sorted(AFTER))
        region.set_abort(RANKS + [SPARE_ROW], 0)
        cleared.set()
        until("after", len(AFTER))
    finally:
        for p in procs:
            p.join(TIMEOUT_S)
            if p.is_alive():
                p.kill()

    for n in SIZES:
        want = member_sum([_input(r, n) for r in RANKS])
        for r in RANKS:
            assert torch.equal(torch.from_numpy(got[(r, f"size{n}")]), want.view(torch.int16)), (
                r,
                n,
            )
    assert late == {r: {DEAD} for r in live}, late
    assert released == {r: set() for r in live}, released
    in_rank_order = member_sum([_graph_input(r) for r in RANKS]).view(torch.int16)
    in_row_order = member_sum([_graph_input(r) for r in (0, 2, DEAD)]).view(torch.int16)
    assert not torch.equal(in_rank_order, in_row_order), "the inputs must make the order show"
    for r in RANKS:
        assert torch.equal(torch.from_numpy(got[(r, "graph")]), in_rank_order)
    assert got[(SPARE_ROW, "refused")], "a process's order that is not the region's is refused"
    for r in live:
        assert got[(r, "released")] < TIMEOUT_S  # the abort word released the spin
    for r in AFTER:  # the same sum as before the change, on every member, the spare's included
        assert torch.equal(torch.from_numpy(got[(r, "after")]), in_rank_order), r
    region.close()
