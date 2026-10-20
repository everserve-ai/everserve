# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The peer-to-peer probe on real GPUs: a verdict within its deadline, and a confirmation where
the GPUs are joined by NVLink.

CUDA's peer access is not enough to expect a confirmation: on PCIe pods (A6000) it is reported
where a container's stores never arrive, which is what the probe exists to find.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)


def test_the_probe_answers_in_time_and_confirms_peer_to_peer_over_nvlink(nvlink):
    from everserve import kernels
    from everserve.comm.p2pprobe import probe

    if not kernels.available():
        pytest.skip("the Everserve kernels did not build here")
    n = min(torch.cuda.device_count(), 4)
    result = probe(list(range(n)), timeout_s=30)
    assert result.seconds < 30 + 15, result  # stragglers are killed after the deadline
    if nvlink(n):
        assert result.ok, result.detail
    elif not result.ok:
        assert result.detail, result
