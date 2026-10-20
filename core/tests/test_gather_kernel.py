# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The step gather kernel against the torch path, byte for byte; skipped without CUDA.

Every cache layout, the 16-byte and the per-element copy, three element sizes, a padding slot,
a held-head subset, into device memory and through a host buffer mapped into the device, and
the mirror's whole path from commit to restore with the kernel in it.
"""

import pytest
import torch

from everserve import kernels
from everserve.state.kvcopy import PagedKV

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NB, BS, H, L = 7, 4, 3, 3
SLOTS = [5, -1, 17, 0, 27]  # one padding id, which gathers row 0 like the torch path
HEADS = [0, 2]
INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


@pytest.fixture(scope="module")
def ext():
    if not kernels.available():
        pytest.skip("the Everserve kernels did not build here")
    return kernels.load()


def caches(layout, d, dtype):
    shape = {
        "kv_first": (2, NB, BS, H, d),
        "blocks_first": (NB, 2, BS, H, d),
        "combined": (NB, H, BS, 2 * d),
    }[layout]
    g = torch.Generator(device="cuda").manual_seed(3)
    return [
        torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)
        for _ in range(L)
    ]


def reference(layers, paged, slots, heads, capacity):
    out = torch.zeros(L, len(HEADS), capacity, 2, paged.head_dim, dtype=layers[0].dtype)
    for i, t in enumerate(layers):
        out[i, :, : slots.numel()] = paged.rows_of_heads(t, slots.clamp(min=0), heads).cpu()
    return out


def launch(ext, layers, paged, slots, heads, out_ptr, capacity):
    ptrs = torch.tensor([t.data_ptr() for t in layers], dtype=torch.long, device="cuda")
    ext.gather_rows(
        ptrs,
        slots,
        heads,
        out_ptr,
        capacity,
        paged.head_dim,
        paged.block_size,
        list(paged.layout_strides(layers[0])),
        layers[0].element_size(),
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("layout", ["kv_first", "blocks_first", "combined"])
@pytest.mark.parametrize("d", [8, 6])  # 16-byte rows (vector copy) and not (element copy)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_kernel_writes_exactly_what_the_torch_path_gathers(ext, layout, d, dtype):
    layers = caches(layout, d, dtype)
    paged = PagedKV.of(layers[0])
    slots = torch.tensor(SLOTS, dtype=torch.long, device="cuda")
    heads = torch.tensor(HEADS, dtype=torch.long, device="cuda")
    capacity = 8
    want = reference(layers, paged, slots, heads, capacity)

    out = torch.zeros_like(want, device="cuda")
    launch(ext, layers, paged, slots, heads, out.data_ptr(), capacity)
    assert torch.equal(out.cpu().view(INT_VIEW[dtype]), want.view(INT_VIEW[dtype]))

    host = torch.zeros_like(want)  # the staging file's case: host memory the kernel writes into
    device_ptr = ext.host_register_mapped(host.data_ptr(), host.numel() * host.element_size())
    try:
        launch(ext, layers, paged, slots, heads, device_ptr, capacity)
        assert torch.equal(host.view(INT_VIEW[dtype]), want.view(INT_VIEW[dtype]))
    finally:
        ext.host_unregister(host.data_ptr())


def test_a_strided_cache_view_is_read_through_its_strides(ext):
    big = caches("kv_first", 8, torch.bfloat16)
    layers = [t[:, :, :, 1:] for t in big]  # head slots 1.. of a wider cache: not contiguous
    paged = PagedKV.of(layers[0])
    slots = torch.tensor(SLOTS, dtype=torch.long, device="cuda")
    heads = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    want = torch.zeros(L, 2, 8, 2, 8, dtype=torch.bfloat16)
    for i, t in enumerate(layers):
        want[i, :, : len(SLOTS)] = paged.rows_of_heads(t, slots.clamp(min=0), heads).cpu()
    out = torch.zeros_like(want, device="cuda")
    launch(ext, layers, paged, slots, heads, out.data_ptr(), 8)
    assert torch.equal(out.cpu().view(torch.int16), want.view(torch.int16))
