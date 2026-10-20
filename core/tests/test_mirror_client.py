# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The mirror client and transport against the real store binary: rows through shared memory,
the control plane over the socket (decision 0011). Skipped when the binary is not built."""

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch

from everserve.state import mirror
from everserve.state.committed import CommittedKV
from everserve.state.mirror import MirrorClient, MirrorCommittedKV, MirrorError

NB, BS, H, D, L = 6, 4, 2, 8, 2


def store_binary():
    """The real store binary if built, from EVERSERVE_STORE_BIN or the cargo target directory."""
    env = os.environ.get("EVERSERVE_STORE_BIN")
    if env and Path(env).exists():
        return env
    root = Path(__file__).resolve().parents[2]
    for profile in ("debug", "release"):
        candidate = root / "target" / profile / "everserve-store"
        if candidate.exists():
            return str(candidate)
    return shutil.which("everserve-store")


@pytest.fixture
def mirror_path():
    """A running store on a fresh socket and data directory."""
    binary = store_binary()
    if binary is None:
        pytest.skip("everserve-store binary not built")
    # on a CUDA host the staging files are pinned, which only works on shared memory (tmpfs)
    shm = Path("/dev/shm")
    root = Path(tempfile.mkdtemp(dir=shm if torch.cuda.is_available() and shm.is_dir() else None))
    path = str(root / "m.sock")
    proc = subprocess.Popen(
        [
            binary,
            "--socket",
            path,
            "--data-dir",
            str(root / "data"),
            "--slots",
            str(NB * BS),
            "--budget-bytes",
            str(1 << 24),
        ]
    )
    try:
        for _ in range(200):
            if Path(path).exists():
                break
            time.sleep(0.02)
        yield path
    finally:
        proc.terminate()
        proc.wait(10)


def make(layout="kv_first"):
    return {
        f"model.layers.{i}.self_attn.attn": torch.zeros(2, NB, BS, H, D, dtype=torch.bfloat16)
        for i in range(L)
    }


def test_mirror_transport_satisfies_the_protocol_and_round_trips(mirror_path):
    client = MirrorClient(mirror_path)
    dead = MirrorCommittedKV(client, "e", rank=2, units=lambda: [(0, 7), (1, 3)])
    dead.register(make())
    assert isinstance(dead, CommittedKV)
    slots = torch.tensor([1, 5, 9, 22])
    written = {}
    for i, (name, t) in enumerate(dead.layers):
        for head in (0, 1):
            x = torch.randn(2, 4, D).to(torch.bfloat16)
            dead.paged.write_rows(t, slots, head, x)
            written[(i, head)] = x
        dead.save(name, t, torch.tensor([1, 5, 9, 22, -1]))  # one padding slot
    dead.commit(epoch=1, step=10, committed_tokens={"a": 40})
    assert dead.committed("a") == 40  # local count moves at once; the store's after the fence
    dead.fence(10)
    assert client.committed("e", ["a", "zz"]) == {"a": 40, "zz": 0}

    # a spare takes rank 2: identity head map; every row comes back bit for bit
    spare = MirrorCommittedKV(
        MirrorClient(mirror_path), "e", rank=2, units=lambda: [(0, 7), (1, 3)]
    )
    spare.register(make())
    spare.restore(2, [(7, 0, 0), (3, 1, 1)])
    assert spare.restored_counts == {"a": 40}  # rank 2's own counts, read with its rows
    for (i, head), x in written.items():
        assert torch.equal(spare.paged.rows(spare.layers[i][1], slots, head), x)

    # a survivor absorbs unit 7 (dead head 0) into its head slot 1 only
    survivor = MirrorCommittedKV(
        MirrorClient(mirror_path), "e", rank=0, units=lambda: [(0, 0), (1, 7)]
    )
    survivor.register(make())
    survivor.restore(2, [(7, 0, 1)])
    for i in range(L):
        assert torch.equal(survivor.paged.rows(survivor.layers[i][1], slots, 1), written[(i, 0)])
        assert torch.equal(
            survivor.paged.rows(survivor.layers[i][1], slots, 0),
            torch.zeros(2, 4, D, dtype=torch.bfloat16),
        )

    dead.evicted("a", 16)
    assert dead.committed("a") == 16
    dead.drain()
    assert client.committed("e", ["a"]) == {"a": 16}
    dead.finished("a")
    dead.drain()
    assert client.committed("e", ["a"]) == {"a": 0}
    dead.close()
    with pytest.raises(MirrorError, match="stale"):
        client.commit("e", 2, epoch=0, step=11, staged=None, committed={})
    client.close()


def test_a_step_is_one_descriptor_whatever_the_layer_count(mirror_path):
    """Decision 0019: the commit names the staging file's layout once; the store derives every
    (layer, head slot) table from it, so the message does not grow with the layers."""
    client = MirrorClient(mirror_path)
    sent = []
    real = client.commit

    def recording(engine, rank, epoch, step, staged, committed):
        sent.append((step, staged, dict(committed)))
        real(engine, rank, epoch, step, staged, committed)

    client.commit = recording
    dead = MirrorCommittedKV(client, "e", rank=1, units=lambda: [(0, 0), (1, 1)])
    dead.register(make())
    slots = torch.tensor([0, 3, 4, 7, 11, 20, 23])
    written = {}
    for i, (name, t) in enumerate(dead.layers):
        for head in (0, 1):
            x = torch.randn(2, len(slots), D).to(torch.bfloat16)
            dead.paged.write_rows(t, slots, head, x)
            written[(i, head)] = x
        dead.save(name, t, slots)
    dead.commit(epoch=1, step=5, committed_tokens={"r": 7})
    dead.fence(5)
    assert len(sent) == 1, "one message for the step"
    (step, staged, counts) = sent[0]
    assert step == 5 and counts == {"r": 7}
    assert list(staged.layers) == [0, 1] and list(staged.heads) == [0, 1]
    assert list(staged.slots) == slots.tolist() and staged.table_stride == mirror._HEAD_STRIDE
    assert staged.layer_stride == 2 * staged.head_stride
    spare = MirrorCommittedKV(MirrorClient(mirror_path), "e", rank=1, units=lambda: [])
    spare.register(make())
    spare.restore(1, [(0, 0, 0), (1, 1, 1)])
    for (i, head), x in written.items():
        assert torch.equal(spare.paged.rows(spare.layers[i][1], slots, head), x)
    dead.close()


def test_staging_files_live_in_the_rank_directory_and_are_pooled_and_deleted(mirror_path):
    kv = MirrorCommittedKV(MirrorClient(mirror_path), "e", rank=4, units=lambda: [(0, 0)], depth=1)
    kv.register(make())
    rank_dir = kv._rank_dir
    assert rank_dir.is_dir() and rank_dir.name == "rank4" and rank_dir.parent.name == "e"
    for step in range(4):  # same batch size: one staging file is reused through the pool
        for name, t in kv.layers:
            kv.save(name, t, torch.tensor([1, 2]))
        kv.commit(epoch=1, step=step, committed_tokens={})
        kv.fence(step)
    stages = sorted(p.name for p in rank_dir.glob("stage-*.rows"))
    assert len(stages) == 1, stages
    for name, t in kv.layers:  # a bigger step gets a bigger file
        kv.save(name, t, torch.arange(20))
    kv.commit(epoch=1, step=4, committed_tokens={})
    kv.fence(4)
    assert len(list(rank_dir.glob("stage-*.rows"))) == 2
    assert sorted(p.name for p in rank_dir.glob("table*.rows")) == ["table0.rows", "table1000.rows"]
    kv.close()
    assert list(rank_dir.glob("stage-*.rows")) == [], "close deletes the staging files"
    assert len(list(rank_dir.glob("table*.rows"))) == 2, "the tables outlive the worker"


def test_a_connection_that_ends_without_close_leaves_no_staging_files(mirror_path):
    # A worker killed mid-step never runs close; the store removes the files named from the
    # prefix its connection was given.
    kv = MirrorCommittedKV(MirrorClient(mirror_path), "e", rank=5, units=lambda: [(0, 0)], depth=1)
    kv.register(make())
    rank_dir = kv._rank_dir
    for name, t in kv.layers:
        kv.save(name, t, torch.tensor([1, 2]))
    kv.commit(epoch=1, step=0, committed_tokens={})
    kv.fence(0)
    stages = list(rank_dir.glob("stage-*.rows"))
    assert stages and all(p.name.startswith(kv._stage_prefix) for p in stages), stages
    kv._client.close()
    deadline = time.monotonic() + 5
    while list(rank_dir.glob("stage-*.rows")):
        assert time.monotonic() < deadline, "the ended connection's staging files stayed"
        time.sleep(0.02)
    assert len(list(rank_dir.glob("table*.rows"))) == 2, "the tables outlive the worker"


class SlowClient:
    """A mirror client whose commits take ``delay`` seconds and record the thread that sent them."""

    def __init__(self, delay):
        self.delay = delay
        self.sent = []
        self.fail_step = None
        self.hold = threading.Event()  # a commit of ``fail_step`` waits for this first
        self.hold.set()

    def open(self, engine, rank):
        d = Path(tempfile.mkdtemp()) / engine / f"rank{rank}"
        d.mkdir(parents=True)
        return str(d), "stage-1-"

    def prepare(self, engine, rank, tables, dtype, shape, slots):
        self.sent.append(("prepare", list(tables), dtype, list(shape), slots))

    def commit(self, engine, rank, epoch, step, staged, committed):
        time.sleep(self.delay)
        if step == self.fail_step:
            self.hold.wait()
            raise MirrorError("store refused")
        self.sent.append(("commit", step, threading.current_thread().name, dict(committed)))

    def evicted(self, engine, request_id, kept_tokens):
        self.sent.append(("evicted", request_id, kept_tokens))

    def finished(self, engine, request_ids):
        self.sent.append(("finished", list(request_ids)))


def committed_step(kv, step, tokens):
    slots = torch.tensor([1, 2])
    for name, t in kv.layers:
        kv.save(name, t, slots)
    kv.commit(epoch=1, step=step, committed_tokens={"a": tokens})


def test_a_commit_returns_before_the_store_answers_and_the_fence_waits_for_it():
    client = SlowClient(delay=0.2)
    kv = MirrorCommittedKV(client, "e", rank=0, units=lambda: [(0, 0)], depth=2)
    kv.register(make())
    t0 = time.perf_counter()
    committed_step(kv, 0, 4)
    assert time.perf_counter() - t0 < 0.1  # the step thread did not pay the round trip
    assert [e for e in client.sent if e[0] == "commit"] == []
    kv.fence(0)
    commits = [e for e in client.sent if e[0] == "commit"]
    assert [e[1] for e in commits] == [0]
    assert commits[0][2] == "everserve-mirror-send" and commits[0][3] == {"a": 4}
    assert kv.gather_path == "torch"  # no CUDA here; a run report must say which path it measured
    (rec,) = kv.timeline()
    assert rec["step"] == 0
    assert rec["commit"] <= rec["dequeue"] <= rec["ready"] <= rec["sent"] <= rec["fence_end"]
    assert rec["fence_start"] <= rec["fence_end"]
    assert kv.stats_record.fence_waits_s > 0.1 and kv.stats_record.backpressure_waits == 0
    assert kv.stats()["steps"] == 1 and kv.stats()["bytes_sent"] > 0
    # tables prepared at register, with the slot ids the engine writes, for the store to refuse
    # if it holds fewer
    assert client.sent[0] == ("prepare", [0, 1000], "bfloat16", [2, D], NB * BS)
    kv.close()


def test_the_timeline_keeps_the_first_steps_of_a_run_longer_than_its_window(monkeypatch):
    from everserve.state import mirror

    monkeypatch.setattr(mirror, "TIMELINE_STEPS", 4)
    monkeypatch.setattr(mirror, "TIMELINE_HEAD", 2)
    client = SlowClient(delay=0.0)
    kv = MirrorCommittedKV(client, "e", rank=0, units=lambda: [(0, 0)], depth=2)
    kv.register(make())
    for step in range(10):
        committed_step(kv, step, 4 + step)
    kv.fence(9)
    # the prefill's commit is among the first; a long run must not lose it to the window
    assert [r["step"] for r in kv.timeline()] == [0, 1, 6, 7, 8, 9]
    kv.close()


def test_the_pipeline_holds_depth_steps_and_a_further_commit_waits_for_the_oldest():
    client = SlowClient(delay=0.15)
    kv = MirrorCommittedKV(client, "e", rank=0, units=lambda: [(0, 0)], depth=2)
    kv.register(make())
    committed_step(kv, 0, 1)
    committed_step(kv, 1, 2)
    t0 = time.perf_counter()
    committed_step(kv, 2, 3)  # two in flight: this one waits for step 0 first
    assert time.perf_counter() - t0 > 0.1
    assert kv.stats_record.backpressure_waits == 1 and kv.stats_record.max_in_flight == 2
    kv.drain()
    assert [e[1] for e in client.sent if e[0] == "commit"] == [0, 1, 2]
    kv.close()


def test_evictions_and_finishes_reach_the_store_behind_the_commits_queued_before_them():
    client = SlowClient(delay=0.05)
    kv = MirrorCommittedKV(client, "e", rank=0, units=lambda: [(0, 0)], depth=4)
    kv.register(make())
    committed_step(kv, 0, 8)
    kv.evicted("a", 2)  # queued behind step 0's rows, so the lowered count is never overtaken
    committed_step(kv, 1, 6)
    kv.finished("a")
    assert kv.committed("a") == 0
    kv.drain()
    assert [e[:2] for e in client.sent if e[0] != "prepare"] == [
        ("commit", 0),
        ("evicted", "a"),
        ("commit", 1),
        ("finished", ["a"]),
    ]
    kv.close()


def test_a_failed_send_surfaces_at_the_fence_and_poisons_the_transport():
    client = SlowClient(delay=0.0)
    client.fail_step = 1
    client.hold.clear()  # step 1 fails only once the test has fenced step 0
    kv = MirrorCommittedKV(client, "e", rank=0, units=lambda: [(0, 0)], depth=2)
    kv.register(make())
    committed_step(kv, 0, 1)
    committed_step(kv, 1, 2)
    kv.fence(0)
    client.hold.set()
    with pytest.raises(MirrorError, match="step 1 failed"):
        kv.fence(1)
    with pytest.raises(MirrorError, match="transport failed"):
        committed_step(kv, 2, 3)
    with pytest.raises(MirrorError):
        kv.close()


def test_save_and_commit_never_read_device_memory_on_the_step_thread(monkeypatch):
    """A slot tensor copied to the host on the step thread drains the GPU: one forward pass per
    step paid inside ``commit()``. Every host-side read of a tensor is
    forbidden on the step thread while rows are saved and committed; the sender thread may read.
    Covers the method forms of a read (``cpu``, ``tolist``, ``item``, ``numpy``, ``bool``), which
    is every form the transport used; a boolean-mask index would also read and is kept out by
    review."""
    step_thread = threading.current_thread()

    def forbid(name):
        real = getattr(torch.Tensor, name)

        def guarded(self, *args, **kwargs):
            if threading.current_thread() is step_thread:
                raise AssertionError(f"Tensor.{name} on the step thread")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, guarded)

    client = SlowClient(delay=0.0)
    kv = MirrorCommittedKV(client, "e", rank=0, units=lambda: [(0, 0), (1, 1)])
    kv.register(make())
    for name in ("cpu", "tolist", "item", "numpy", "__bool__"):
        forbid(name)
    slots = torch.tensor([3, 9, -1])  # one padding id, dropped by the sender
    for name, t in kv.layers:
        kv.save(name, t, slots)
    kv.commit(epoch=1, step=0, committed_tokens={"a": 2})
    kv.fence(0)
    assert client.sent[-1][1] == 0
    kv.close()


def test_the_sender_drops_padding_and_frames_every_held_head_of_every_saved_layer(mirror_path):
    client = MirrorClient(mirror_path)
    kv = MirrorCommittedKV(client, "e", rank=3, units=lambda: [(0, 5), (1, 6)])
    kv.register(make())
    slots = torch.tensor([2, 7, 13])
    written = {}
    for i, (name, t) in enumerate(kv.layers):
        for head in (0, 1):
            x = torch.randn(2, 3, D).to(torch.bfloat16)
            kv.paged.write_rows(t, slots, head, x)
            written[(i, head)] = x
        kv.save(name, t, torch.tensor([2, -1, 7, 13, -1]))
    kv.commit(epoch=1, step=0, committed_tokens={"q": 3})
    kv.fence(0)
    spare = MirrorCommittedKV(MirrorClient(mirror_path), "e", rank=3, units=lambda: [])
    spare.register(make())
    spare.restore(3, [(5, 0, 0), (6, 1, 1)])
    for (i, head), x in written.items():
        assert torch.equal(spare.paged.rows(spare.layers[i][1], slots, head), x)
    assert spare.restored_counts == {"q": 3}
    kv.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_on_the_device_the_gather_kernel_carries_the_step_and_restores_bit_for_bit(mirror_path):
    from everserve import kernels

    if not kernels.available():
        pytest.skip("the Everserve kernels did not build here")

    def cuda_caches():
        return {n: t.cuda() for n, t in make().items()}

    client = MirrorClient(mirror_path)
    dead = MirrorCommittedKV(client, "e", rank=2, units=lambda: [(0, 7), (1, 3)])
    dead.register(cuda_caches())
    assert dead._kernel is not None  # the kernel path, not the per-layer torch path
    slots = torch.tensor([1, 5, 9, 22], device="cuda")
    written = {}
    for i, (name, t) in enumerate(dead.layers):
        for head in (0, 1):
            x = torch.randn(2, 4, D, device="cuda").to(torch.bfloat16)
            dead.paged.write_rows(t, slots, head, x)
            written[(i, head)] = x
        dead.save(name, t, torch.tensor([1, 5, 9, 22, -1], device="cuda"))
    dead.commit(epoch=1, step=10, committed_tokens={"a": 40})
    dead.fence(10)

    spare = MirrorCommittedKV(
        MirrorClient(mirror_path), "e", rank=2, units=lambda: [(0, 7), (1, 3)]
    )
    spare.register(cuda_caches())
    spare.restore(2, [(7, 0, 0), (3, 1, 1)])
    for (i, head), x in written.items():
        got = spare.paged.rows(spare.layers[i][1], slots, head)
        assert torch.equal(got.view(torch.int16), x.view(torch.int16))


def test_a_restore_since_a_watermark_copies_only_the_rows_committed_after_it(mirror_path):
    client = MirrorClient(mirror_path)
    dead = MirrorCommittedKV(client, "e", rank=2, units=lambda: [(0, 0)])
    dead.register(make())
    first = torch.tensor([1, 2, 3])
    x1 = {}
    for i, (name, t) in enumerate(dead.layers):
        x1[i] = torch.randn(2, 3, D).to(torch.bfloat16)
        dead.paged.write_rows(t, first, 0, x1[i])
        dead.save(name, t, first)
    dead.commit(epoch=1, step=10, committed_tokens={"a": 3})
    dead.fence(10)

    # a spare restores everything when the notice arrives
    spare = MirrorCommittedKV(MirrorClient(mirror_path), "e", rank=2, units=lambda: [(0, 0)])
    spare.register(make())
    watermark = spare.restore(2, [(0, 0, 0)])
    assert watermark == 10 and spare.restored_counts == {"a": 3}
    assert spare.restored_rows == 3 * len(spare.layers)  # three slots in every layer's table

    # the leaving rank commits one more step; the spare's copy of the first rows is then stale
    # in the cache only if someone rewrote them, so the test rewrites slot 2 as well as slot 4
    later = torch.tensor([2, 4])
    x2 = {}
    for i, (name, t) in enumerate(dead.layers):
        x2[i] = torch.randn(2, 2, D).to(torch.bfloat16)
        dead.paged.write_rows(t, later, 0, x2[i])
        dead.save(name, t, later)
    dead.commit(epoch=1, step=11, committed_tokens={"a": 4})
    dead.fence(11)
    for _, t in spare.layers:  # scribble over the spare's slot 1 to prove the incremental
        # restore leaves it alone (slot 1 was not written after step 10)
        dead.paged.write_rows(t, torch.tensor([1]), 0, torch.zeros(2, 1, D, dtype=torch.bfloat16))

    assert spare.restore(2, [(0, 0, 0)], since_step=watermark) == 11
    assert spare.restored_counts == {"a": 4}
    assert spare.restored_rows == 2 * len(spare.layers), "only slots 2 and 4, in every layer"
    for i, (_, t) in enumerate(spare.layers):
        assert torch.equal(spare.paged.rows(t, later, 0), x2[i]), "slots 2 and 4 came over"
        assert torch.equal(
            spare.paged.rows(t, torch.tensor([1]), 0), torch.zeros(2, 1, D, dtype=torch.bfloat16)
        ), "slot 1 was not touched"
        assert torch.equal(spare.paged.rows(t, torch.tensor([3]), 0), x1[i][:, 2:3])
    dead.close()
    spare.close()


def test_register_prepares_every_table_before_the_first_commit(mirror_path):
    kv = MirrorCommittedKV(MirrorClient(mirror_path), "e", rank=6, units=lambda: [(0, 0), (1, 1)])
    kv.register(make())
    tables = sorted(p.name for p in kv._rank_dir.glob("table*.rows"))
    assert tables == ["table0.rows", "table1.rows", "table1000.rows", "table1001.rows"]
    assert all(p.stat().st_size == NB * BS * 2 * D * 2 for p in kv._rank_dir.glob("table*.rows"))
    kv.close()


def test_the_notice_catch_up_sequence_answers_every_request(mirror_path):
    """The notice path's store traffic in the harness driver's order, each rank on its own
    connection and
    thread: every request must be answered (a lost reply would stall a rank's step thread).

    Ranks 0-3 commit steps 1-5 at epoch 0; the spare restores rank 2 in full; the leaving rank 2
    commits steps 6-7 and drains; the spare restores rank 2 since the watermark; then ranks 0, 1,
    3 and the spare (as rank 2) commit steps 8-10 at epoch 1.
    """
    deadline_s = 20.0
    ranks = {
        r: MirrorCommittedKV(MirrorClient(mirror_path), "e", rank=r, units=lambda: [(0, 0)])
        for r in range(4)
    }
    for kv in ranks.values():
        kv.register(make())

    def commit_steps(kv, epoch, steps, base):
        for step in steps:
            slots = torch.tensor([base + step])
            for name, t in kv.layers:
                kv.paged.write_rows(t, slots, 0, torch.randn(2, 1, D).to(torch.bfloat16))
                kv.save(name, t, slots)
            kv.commit(epoch=epoch, step=step, committed_tokens={"a": base + step + 1})
        kv.drain()

    def run_all(jobs):
        threads = [threading.Thread(target=fn, args=args, daemon=True) for fn, args in jobs]
        for th in threads:
            th.start()
        for th in threads:
            th.join(deadline_s)
        stuck = [i for i, th in enumerate(threads) if th.is_alive()]
        assert not stuck, f"jobs {stuck} did not finish: a request went unanswered"

    run_all([(commit_steps, (ranks[r], 0, range(1, 6), 0)) for r in range(4)])
    spare = MirrorCommittedKV(MirrorClient(mirror_path), "e", rank=2, units=lambda: [(0, 0)])
    spare.register(make())
    got = {}
    run_all([(lambda: got.setdefault("w", spare.restore(2, [(0, 0, 0)])), ())])
    assert got["w"] == 5 and spare.restored_counts == {"a": 6}
    run_all([(commit_steps, (ranks[2], 0, range(6, 8), 0))])  # the leaving rank, then its drain
    run_all(
        [(lambda: got.setdefault("w2", spare.restore(2, [(0, 0, 0)], since_step=got["w"])), ())]
    )
    assert got["w2"] == 7 and spare.restored_counts == {"a": 8}
    assert spare.restored_rows == 2 * len(spare.layers)  # steps 6 and 7 only
    run_all(
        [(commit_steps, (ranks[r], 1, range(8, 11), 0)) for r in (0, 1, 3)]
        + [(commit_steps, (spare, 1, range(8, 11), 0))]
    )
    for r in (0, 1, 3):
        assert ranks[r]._client.committed("e", ["a"]) == {"a": 11}


def test_state_reads_held_epoch_step_and_caught_up(mirror_path):
    """A fresh store holds nothing for a rank, and the read needs no replication link (0030)."""
    client = MirrorClient(mirror_path)
    try:
        assert client.state("e", 0) == (False, 0, 0, False)
        # the committed KV asks for any rank of its engine: what a spare's restore waits on
        kv = MirrorCommittedKV(client, "e", rank=1, units=lambda: [])
        assert kv.state(0) == (False, 0, 0, False) and kv.state(1) == (False, 0, 0, False)
    finally:
        client.close()
