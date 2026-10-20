# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The harness's scripts import their siblings by name, as they run from their own directory."""

import sys
from pathlib import Path

TWONODE = Path(__file__).resolve().parents[1] / "twonode"
sys.path.insert(0, str(TWONODE))
