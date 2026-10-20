# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The KV connector: each step's committed rows to the mirror, per ``docs/CONTRACT.md`` item 3.

Loaded with ``--kv-transfer-config '{"kv_connector": "EverserveConnector",
"kv_connector_module_path": "everserve_vllm.connector", "kv_role": "kv_both"}'``. The lifecycle
itself is :class:`everserve_vllm.lifecycle.Lifecycle`; this class maps vLLM's calls onto it:

| vLLM | here |
|---|---|
| ``build_connector_meta`` (scheduler) | :meth:`Lifecycle.record` from the scheduler output |
| ``request_finished`` (scheduler) | :meth:`Lifecycle.finished`; returns ``(False, None)`` |
| ``get_num_new_matched_tokens`` | ``(0, False)``: after a restore the prefix cache reuses blocks |
| ``handle_preemptions`` (worker) | :meth:`Lifecycle.preemptions` (``evicted``) and ``begin`` |
| ``save_kv_layer`` (worker) | :meth:`Lifecycle.save` with the layer's slot ids |
| ``wait_for_save`` (worker) | the layers FULL graphs skipped, then :meth:`Lifecycle.end` |

Under FULL CUDA graphs the attention layer's ``save_kv_layer`` call is inside the captured graph
and does not run on replay. The model runner hands the step's attention metadata to
``start_load_kv`` (before ``wait_for_save``), so the slot ids are kept there and
``wait_for_save`` saves every layer the per-layer hook did not.

The worker side does nothing until :func:`everserve_vllm.worker.bootstrap` attaches the mirror and
the runtime; until then a step is not committed.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from everserve_vllm import firststep
from everserve_vllm.lifecycle import Lifecycle, StepRecord

log = logging.getLogger(__name__)

WORKER: EverserveConnector | None = None
"""The worker-side connector of this process, for the bootstrap to attach to."""
SCHEDULER: EverserveConnector | None = None
"""The scheduler-side connector of the engine process, for the executor's replay."""


class EverserveMeta(KVConnectorMetadata):
    """The connector metadata: one :class:`StepRecord`."""

    def __init__(self, record: StepRecord) -> None:
        """Carry ``record`` to the workers."""
        self.record = record


def _slot_mapping(attn_metadata: Any) -> tuple[torch.Tensor, int] | None:
    """The step's slot ids from attention metadata (one object, or a dict by layer)."""
    if attn_metadata is None:
        return None
    meta = (
        next(iter(attn_metadata.values()), None)
        if isinstance(attn_metadata, dict)
        else attn_metadata
    )
    slots = getattr(meta, "slot_mapping", None)
    if slots is None:
        return None
    return slots, int(getattr(meta, "num_actual_tokens", slots.shape[0]))


class EverserveConnector(KVConnectorBase_V1):
    """Committed KV to the mirror; no external loads, no asynchronous saves."""

    def __init__(
        self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None
    ) -> None:
        """Create either side; each registers itself for the adapter code in its process."""
        global WORKER, SCHEDULER
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.lifecycle = Lifecycle()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self._slots: tuple[torch.Tensor, int] | None = None
        if role == KVConnectorRole.WORKER:
            WORKER = self
        else:
            SCHEDULER = self

    # ------------------------------------------------------------------ scheduler side
    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """No external hits: KV is either in place or recomputed."""
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        """Nothing to load."""
        return None

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        """The step record: committed count per scheduled request, preempted requests."""
        n = scheduler_output.num_scheduled_tokens
        scheduled = [
            (r.req_id, r.num_computed_tokens, n.get(r.req_id, 0))
            for r in scheduler_output.scheduled_new_reqs
        ]
        cached = scheduler_output.scheduled_cached_reqs
        scheduled += [
            (rid, done, n.get(rid, 0))
            for rid, done in zip(cached.req_ids, cached.num_computed_tokens, strict=True)
        ]
        preempted = scheduler_output.preempted_req_ids or ()
        return EverserveMeta(Lifecycle.record(scheduled, preempted))

    def request_finished(self, request: Any, block_ids: Any) -> tuple[bool, dict[str, Any] | None]:
        """Forget the request; we never hold its blocks."""
        self.lifecycle.finished(request.request_id)
        return False, None

    # ------------------------------------------------------------------ worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Keep the paged caches; the bootstrap registers them with the mirror transport."""
        self.kv_caches = dict(kv_caches)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Start of a step: lower counts for preempted requests, then begin the step."""
        rec = getattr(kv_connector_metadata, "record", StepRecord())
        firststep.step_begin()
        self.lifecycle.preemptions(rec)
        self.lifecycle.begin(rec)
        self._slots = None

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Nothing to load; keep the step's slot ids for the layers FULL graphs hide."""
        found = _slot_mapping(kwargs.get("attn_metadata")) or _slot_mapping(
            getattr(forward_context, "attn_metadata", None)
        )
        if found is not None:
            self._slots = found

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Nothing to load."""
        return None

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any
    ) -> None:
        """Keep this layer's rows (eager and PIECEWISE steps; never during graph capture)."""
        if torch.cuda.is_current_stream_capturing():
            return
        found = _slot_mapping(attn_metadata)
        if found is None:
            return
        self._slots = found
        slots, n = found
        self.lifecycle.save(layer_name, kv_layer, slots[:n])

    def wait_for_save(self) -> None:
        """End of a step: save the layers the per-layer hook missed, then commit."""
        if self.lifecycle.kv is not None and self._slots is not None:
            slots, n = self._slots
            for name, t in self.kv_caches.items():
                if not self.lifecycle.saved(name):
                    self.lifecycle.save(name, t, slots[:n])
        self.lifecycle.end()
        self._slots = None
        firststep.step_end()
