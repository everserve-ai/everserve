# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the core tests."""

import pytest


def _nvlink(n: int) -> bool:
    """Whether NVML reports an active NVLink on each of the first ``n`` devices.

    CUDA's peer access is not this: PCIe pods (A6000) report it where a container's stores never
    arrive, and a peer-to-peer test there hangs to its timeout.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - no NVML: no claim
        return False
    try:
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                if pynvml.nvmlDeviceGetNvLinkState(h, 0) != pynvml.NVML_FEATURE_ENABLED:
                    return False
            except pynvml.NVMLError:
                return False
        return True
    finally:
        pynvml.nvmlShutdown()


@pytest.fixture
def nvlink():
    """``nvlink(n)``: whether the first ``n`` devices are joined by NVLink."""
    return _nvlink
