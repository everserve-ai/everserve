# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Drill-only: the adapter's connector with the mirror switched off.

Arm 2 of the asynchronous-commit run (``bench/runs/2026-10-05-async-commit-4gpu``) needs the
adapter fully resident (slot kernel, shadow sets,
spare, daemon link) but no mirror traffic, to attribute the steady overhead between the kernel and
the commit. The product has no "mirror off" mode and should not grow one for a measurement, so this
subclass lives here: the worker bootstrap still finds a connector and registers its caches, but no
step is saved, committed or fenced. vLLM loads it by module path, so the drill puts this directory
on PYTHONPATH for that arm only.
"""

from __future__ import annotations

from typing import Any

from everserve_vllm.connector import EverserveConnector


class NullConnector(EverserveConnector):
    """Registers the caches like the real connector; never saves, commits or evicts."""

    def handle_preemptions(self, kv_connector_metadata: Any) -> None:
        """Nothing to lower: nothing is committed."""

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any
    ) -> None:
        """No rows kept."""

    def wait_for_save(self) -> None:
        """No commit."""

    def request_finished(self, request: Any, block_ids: Any) -> tuple[bool, dict[str, Any] | None]:
        """Nothing to forget."""
        return False, None
