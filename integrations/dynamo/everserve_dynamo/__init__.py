# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Everserve's integration with NVIDIA Dynamo (decision 0041): everything Dynamo-specific."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("everserve-dynamo")
except importlib.metadata.PackageNotFoundError:  # imported from a checkout, not installed
    __version__ = "0+unknown"
