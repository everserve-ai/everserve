# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Everserve core: the engine-independent runtime.

A rank can leave, a spare can join, and sessions do not notice. See the package README for the
module map, and ``docs/ARCHITECTURE.md`` for where this runtime sits among the processes.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("everserve")
except importlib.metadata.PackageNotFoundError:  # imported from a checkout, not installed
    __version__ = "0+unknown"
