# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What happens when a rank leaves. See ``docs/RUNTIME.md`` section 2 and decision 0004.

Kept free of every other dependency so the runtime, the policy and the adapters share one enum.
"""

from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    """The three outcomes of a membership change."""

    REPLACE = "replace"
    """A spare takes the dead rank's id at constant world size."""
    SHRINK = "shrink"
    """Survivors absorb the lost rank's units; needs a placement-independent model."""
    REBUILD = "rebuild"
    """Rebuild the executor at a smaller parallel size and replay requests."""
