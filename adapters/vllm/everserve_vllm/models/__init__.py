# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Stock architectures ported to the model-owned communicator.

Each port is a stock vLLM model file with its collectives routed through ``everserve.comm``
and its expert and KV layout taken from ``everserve.reference.plan``. The first port (phase 1,
step 2) is a DeepSeek-V3 / K3-class MoE with MLA.
"""

from __future__ import annotations


def register_models() -> None:
    """Register every ported architecture with vLLM's model registry.

    No ports exist yet; this is the hook the ports attach to.
    """
