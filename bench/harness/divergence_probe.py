# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic instrumentation for the REPLACE divergence (bench/runs/2026-10-06-replace-divergence).

Drill-only, kept out of the product tree on purpose: it patches vLLM's V2 model runner in the
workers, at the pin, to record what each step ran, and reads K and V rows by position. Nothing
here changes what a step computes.

* :func:`install_batch_log` (RPC): every step's batch as the worker's runner built it, the request
  order, each request's scheduled tokens and computed prefill count, whether any request was
  prefilling and whether the batch was eligible for the decode graph, and the CUDA-graph mode the
  runner chose (``FULL`` is the decode path; ``PIECEWISE`` or ``NONE`` the mixed/prefill path).
* :func:`batch_log` (RPC): those records.
* :func:`kv_rows_by_position` (RPC): per request and position, one digest of this rank's K and V
  rows over every layer and head slot.
* :func:`log_scheduler` (engine side): every ``SchedulerOutput`` the engine core produced.
"""

from __future__ import annotations

import time
from typing import Any

import torch

_BATCHES: list[dict[str, Any]] = []


def _list(x: Any) -> list[Any] | None:
    if x is None:
        return None
    try:
        return [v.item() if hasattr(v, "item") else v for v in list(x)]
    except TypeError:
        return None


def install_batch_log(worker: Any) -> dict[str, Any]:
    """RPC target on every worker: wrap the runner's batch building and input preparation."""
    runner = worker.worker.model_runner
    if getattr(runner, "_es_diag", False):
        return {"rank": worker.worker.rank, "installed": "already"}
    gather, prepare = runner.gather_batch_req_state, runner.prepare_inputs

    def gather_logged(*args: Any, **kwargs: Any) -> Any:
        out = gather(*args, **kwargs)
        st = out[0] if isinstance(out, tuple) else out
        if st is not None:
            _BATCHES.append(
                {
                    "t": time.time(),
                    "req_ids": list(st.req_ids),
                    "num_scheduled_tokens": _list(st.num_scheduled_tokens),
                    "num_computed_prefill_tokens": _list(st.num_computed_prefill_tokens_np),
                    "prefill_len": _list(st.prefill_len_np),
                    "is_prefilling": _list(st.is_prefilling_np),
                    "has_prefill": bool(st.has_prefill),
                    "decode_graph_eligible": bool(st.decode_graph_eligible),
                }
            )
        return out

    def prepare_logged(*args: Any, **kwargs: Any) -> Any:
        desc = kwargs.get("batch_desc", args[2] if len(args) > 2 else None)
        if _BATCHES and "cg_mode" not in _BATCHES[-1]:
            _BATCHES[-1]["cg_mode"] = str(getattr(desc, "cg_mode", None))
            _BATCHES[-1]["padded_tokens"] = getattr(desc, "num_tokens", None)
        return prepare(*args, **kwargs)

    runner.gather_batch_req_state = gather_logged
    runner.prepare_inputs = prepare_logged
    runner._es_diag = True
    return {"rank": worker.worker.rank, "installed": True}


def batch_log(worker: Any) -> dict[str, Any]:
    """RPC target: this worker's batch records."""
    return {"rank": worker.worker.rank, "batches": list(_BATCHES)}


def kv_rows_by_position(worker: Any, slots: dict[str, list[int]]) -> dict[str, Any]:
    """RPC target: per request, per slot, one digest of this rank's K and V rows.

    The digest sums, over every layer and head slot, the two row checksums of
    ``restore_check.row_checksums`` weighted by the layer and head, so a change in any bit of any
    layer's K or V row at that position changes it.
    """
    from restore_check import row_checksums

    from everserve.state.kvcopy import PagedKV, layer_index
    from everserve_vllm import connector
    from everserve_vllm.worker import head_slots

    kv = connector.WORKER.kv_caches
    t0 = next(iter(kv.values()))
    paged = PagedKV.of(t0)
    heads = head_slots(kv)
    out: dict[str, list[list[int]]] = {}
    for rid, ss in slots.items():
        idx = torch.tensor(ss, dtype=torch.long, device=t0.device)
        acc = torch.zeros(len(ss), 2, dtype=torch.int64, device=t0.device)
        for name, t in kv.items():
            li = layer_index(name)
            for h in range(heads):
                acc += row_checksums(paged.rows(t, idx, h)) * (li * 1009 + h + 1)
        out[rid] = acc.cpu().tolist()
    torch.cuda.synchronize()
    return {"rank": worker.worker.rank, "rows": out}


def log_scheduler(core: Any, records: list[dict[str, Any]]) -> None:
    """Engine side: append every scheduler output's batch, in the scheduler's own order."""
    sched = core.scheduler
    schedule = sched.schedule

    def logged(*args: Any, **kwargs: Any) -> Any:
        so = schedule(*args, **kwargs)
        cached = getattr(so, "scheduled_cached_reqs", None)
        records.append(
            {
                "t": time.time(),
                "new": [
                    (r.req_id, getattr(r, "num_computed_tokens", None))
                    for r in getattr(so, "scheduled_new_reqs", [])
                ],
                "cached": list(getattr(cached, "req_ids", []) or []),
                "cached_num_computed": list(getattr(cached, "num_computed_tokens", []) or []),
                "resumed": [str(x) for x in (getattr(cached, "resumed_req_ids", None) or [])],
                "num_scheduled_tokens": dict(getattr(so, "num_scheduled_tokens", {}) or {}),
                "total": getattr(so, "total_num_scheduled_tokens", None),
                # a preempted request loses its blocks: the per-position digests of a run with
                # any are not comparable to the control's
                "preempted": [str(x) for x in (getattr(so, "preempted_req_ids", None) or [])],
            }
        )
        return so

    sched.schedule = logged
