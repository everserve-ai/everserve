# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Drill instrumentation for prediction 5: are the spare's restored rows the dead rank's rows?

Not product code, and deliberately outside ``adapters/``: it reaches into the worker's KV caches
through the adapter's connector purely to measure. Before the kill the driver asks the rank that
will die for a digest of its committed rows (:func:`snapshot`); right after the swap, before the
next step can touch the cache, it asks the spare, now serving that rank, for the same digest at
the same slots (:func:`digests_at`). Equal digests per (layer, head slot) mean equal rows.

The functions run in the workers as cloudpickled RPCs; the driver registers this module to be
pickled by value, since the workers cannot import ``bench/``.
"""

from __future__ import annotations

import time
from typing import Any

import torch


def row_checksums(rows: torch.Tensor) -> torch.Tensor:
    """Two int64 checksums of each slot's raw bits in ``[2, n, d]`` rows, shape ``[n, 2]``."""
    x = rows.contiguous().view(torch.int16).to(torch.int64)
    w = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=torch.int64)
    a = (x * w).sum(dim=(0, 2))
    b = ((x * x) * w.flip(0)).sum(dim=(0, 2)) + x[0].sum(dim=-1) * 7919
    return torch.stack([a, b], dim=1)


def digest(rows: torch.Tensor) -> list[int]:
    """A position-weighted digest of :func:`row_checksums` over every slot: two int64s."""
    cs = row_checksums(rows)
    w = torch.arange(1, cs.shape[0] + 1, device=cs.device, dtype=torch.int64)[:, None]
    return (cs * w).sum(dim=0).cpu().tolist()


def kv_digests(kv_caches: dict[str, torch.Tensor], slots: list[int]) -> dict[str, list[int]]:
    """A digest of every head slot's rows at ``slots`` in every layer, keyed ``layer:head``."""
    from everserve.state.kvcopy import PagedKV, layer_index
    from everserve_vllm.worker import head_slots

    if not slots:
        return {}
    t0 = next(iter(kv_caches.values()))
    paged = PagedKV.of(t0)
    idx = torch.tensor(slots, dtype=torch.long, device=t0.device)
    return {
        f"{layer_index(name)}:{h}": digest(paged.rows(t, idx, h))
        for name, t in kv_caches.items()
        for h in range(head_slots(kv_caches))
    }


def live_kv_caches() -> dict[str, torch.Tensor]:
    """This process's live KV tensors by layer name: the ones the restore writes and commits read.

    ``worker.STATE.kv.layers`` where the adapter runtime is up (a serving worker, a spare after
    its join); the connector's registration before that. On a spare the connector's set is not
    the live one (it reads all zeros on an exact run).
    """
    from everserve_vllm import connector, worker

    state = worker.STATE
    if state is not None and getattr(state, "kv", None) is not None:
        layers = dict(state.kv.layers)
        if layers:
            return layers
    return connector.WORKER.kv_caches


def digests_at(worker: Any, rank: int, slots: list[int]) -> dict[str, Any] | None:
    """RPC target: on ``rank`` only, digests of its KV rows at ``slots``."""
    if worker.worker.rank != rank:
        return None

    t0 = time.time()
    sums = kv_digests(live_kv_caches(), slots)
    torch.cuda.synchronize()
    return {"rank": rank, "sums": sums, "s": round(time.time() - t0, 4)}


def digests_all(worker: Any, slots: list[int]) -> dict[str, Any]:
    """RPC target on every rank: digests of its own KV rows at ``slots``.

    Decision 0013's check that committed rows are byte-identical before and after a change.
    """
    sums = kv_digests(live_kv_caches(), slots)
    torch.cuda.synchronize()
    return {"rank": worker.worker.rank, "sums": sums}


def committed_slots(core: Any, mirror: Any, engine: str) -> list[int]:
    """Slots of tokens every rank has committed, from the scheduler's block tables.

    The engine-wide count is the latest commit by any rank; ranks are lock-stepped by the step's
    collectives, so one decode token below it every rank has committed too.
    """
    sched = core.scheduler
    bs = core.vllm_config.cache_config.block_size
    ids = [r.request_id for r in sched.running]
    counts = mirror.committed(engine, ids)
    slots = []
    for rid in ids:
        blocks = sched.kv_cache_manager.get_block_ids(rid)[0]
        for t in range(max(0, counts.get(rid, 0) - 1)):
            slots.append(blocks[t // bs] * bs + t % bs)
    return slots


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Tables whose digests differ between the dead rank's snapshot and the spare."""
    bad = sorted(k for k, v in before["sums"].items() if after["sums"].get(k) != v)
    return {"tables": len(before["sums"]), "mismatched": bad}


# ------------------------------------------------- the localisation of one differing row
def slot_checksums(kv_caches: dict[str, torch.Tensor], slots: list[int]) -> dict[str, list]:
    """Per-slot checksums of every head slot's rows at ``slots``, keyed ``layer:head``.

    Each value is a list of ``[a, b]`` per slot, in the order of ``slots``: two slots with the
    same pair hold the same bits with overwhelming likelihood, so a differing pair names the
    slot a restore got wrong (:func:`compare_slots`).
    """
    from everserve.state.kvcopy import PagedKV, layer_index
    from everserve_vllm.worker import head_slots

    if not slots:
        return {}
    t0 = next(iter(kv_caches.values()))
    paged = PagedKV.of(t0)
    idx = torch.tensor(slots, dtype=torch.long, device=t0.device)
    return {
        f"{layer_index(name)}:{h}": row_checksums(paged.rows(t, idx, h)).cpu().tolist()
        for name, t in kv_caches.items()
        for h in range(head_slots(kv_caches))
    }


def checksums_at(worker: Any, rank: int, slots: list[int]) -> dict[str, Any] | None:
    """RPC target: on ``rank`` only, per-slot checksums of its KV rows at ``slots``."""
    if worker.worker.rank != rank:
        return None

    sums = slot_checksums(live_kv_caches(), slots)
    torch.cuda.synchronize()
    return {"rank": rank, "slots": list(slots), "sums": sums}


def save_rows(worker: Any, rank: int, slots: list[int], path: str) -> dict[str, Any] | None:
    """RPC target: on ``rank`` only, its raw KV rows at ``slots`` for every table, to ``path``.

    The kill-time rows of the rank that dies, kept on its node's disk: the process goes, the
    file stays, and a row the spare restored differently can be diffed against it byte for
    byte (``torch.load`` gives ``{"slots", "rows": {layer:head -> [2, n, d] on the CPU}}``).
    """
    if worker.worker.rank != rank:
        return None
    from everserve.state.kvcopy import PagedKV, layer_index
    from everserve_vllm.worker import head_slots

    kv = live_kv_caches()
    t0 = next(iter(kv.values()))
    paged = PagedKV.of(t0)
    idx = torch.tensor(slots, dtype=torch.long, device=t0.device)
    rows = {
        f"{layer_index(name)}:{h}": paged.rows(t, idx, h).cpu()
        for name, t in kv.items()
        for h in range(head_slots(kv))
    }
    torch.save({"slots": list(slots), "rows": rows}, path)
    return {"rank": rank, "path": path, "tables": len(rows), "slots": len(slots)}


def rows_at(worker: Any, rank: int, items: list[tuple[str, int]]) -> dict[str, Any] | None:
    """RPC target: on ``rank`` only, the raw bytes (hex) of the rows at ``[(layer:head, slot)]``."""
    if worker.worker.rank != rank:
        return None
    from everserve.state.kvcopy import PagedKV, layer_index

    kv = live_kv_caches()
    by_key = {f"{layer_index(name)}": t for name, t in kv.items()}
    out = {}
    for key, slot in items:
        layer, head = key.split(":")
        t = by_key[layer]
        paged = PagedKV.of(t)
        idx = torch.tensor([slot], dtype=torch.long, device=t.device)
        r = paged.rows(t, idx, int(head)).contiguous().cpu()
        out[f"{key}@{slot}"] = r.view(torch.uint8).numpy().tobytes().hex()
    return {"rank": rank, "rows": out}


def request_slots(core: Any, counts: dict[str, int]) -> dict[str, list[int]]:
    """Each running request's slots below its count in ``counts``, in position order."""
    sched = core.scheduler
    km = sched.kv_cache_manager
    block_size = (
        km.block_size if hasattr(km, "block_size") else core.vllm_config.cache_config.block_size
    )
    out = {}
    for req in sched.running:
        n = counts.get(req.request_id)
        if n is None:
            continue
        blocks = km.get_block_ids(req.request_id)[0]
        out[req.request_id] = [
            b * block_size + o
            for p in range(n)
            for b, o in [(blocks[p // block_size], p % block_size)]
        ]
    return out


def compare_slots(
    before: dict[str, Any], after: dict[str, Any], by_request: dict[str, list[int]]
) -> dict[str, list]:
    """Per request, the ``[table, slot, position]`` whose checksums differ, below its count."""
    pos = {s: i for i, s in enumerate(before["slots"])}
    out: dict[str, list] = {}
    for rid, slots in by_request.items():
        bad = []
        for p, s in enumerate(slots):
            i = pos.get(s)
            if i is None:
                continue
            for key, sums in before["sums"].items():
                if after["sums"].get(key, [None] * (i + 1))[i] != sums[i]:
                    bad.append([key, s, p])
        if bad:
            out[rid] = bad
    return out


def kv_identity(worker: Any, rank: int, key: str, slot: int) -> dict[str, Any] | None:
    """RPC target: on ``rank`` only, one row read through every tensor set a process holds.

    Which set is the live cache the model runs on is the question this answers on a spare
    (where every checksum can differ and the rows read as zeros on an exact run):
    the runtime's KV object, the connector's registration, the runner's cache list and the
    attention layer's own attribute, each with its device pointer and the row's first bytes.
    """
    if worker.worker.rank != rank:
        return None
    from everserve.state.kvcopy import PagedKV, layer_index
    from everserve_vllm import connector
    from everserve_vllm import worker as wk

    layer, head = key.split(":")
    out: dict[str, Any] = {}

    def read(name: str, kv: Any) -> None:
        try:
            if isinstance(kv, dict):
                t = next(v for k, v in kv.items() if f"{layer_index(k)}" == layer)
            else:
                t = kv
            paged = PagedKV.of(t)
            idx = torch.tensor([slot], dtype=torch.long, device=t.device)
            r = paged.rows(t, idx, int(head)).contiguous().cpu()
            out[name] = {
                "ptr": t.data_ptr(),
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "row": r.view(torch.uint8).numpy().tobytes()[:16].hex(),
            }
        except Exception as e:  # noqa: BLE001 - a candidate that cannot be read is reported as such
            out[name] = {"error": f"{type(e).__name__}: {e}"[:160]}

    state = wk.STATE
    if state is not None and getattr(state, "kv", None) is not None:
        read("state_kv_layers", dict(state.kv.layers))
    if connector.WORKER is not None:
        read("connector_kv_caches", connector.WORKER.kv_caches)
    runner = getattr(worker.worker, "model_runner", None)
    caches = getattr(runner, "kv_caches", None)
    if isinstance(caches, list) and caches:
        try:
            read("runner_kv_caches", caches[int(layer) - min(int(layer), len(caches) - 1)])
            out["runner_kv_caches_len"] = len(caches)
        except Exception as e:  # noqa: BLE001
            out["runner_kv_caches"] = {"error": str(e)[:160]}
    model = getattr(runner, "model", None)
    if model is not None:
        for name, mod in model.named_modules():
            if name.endswith(f"layers.{layer}.self_attn.attn") and hasattr(mod, "kv_cache"):
                kc = mod.kv_cache
                kc = kc[0] if isinstance(kc, (list, tuple)) else kc
                read("attn_module_kv_cache", kc)
                break
    return out
