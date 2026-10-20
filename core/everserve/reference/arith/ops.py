# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Elementary operations shared by the reference cores: RMS norm and rotary embeddings.

These follow the Hugging Face conventions exactly (``rotate_half`` RoPE, the llama3 frequency
schedule), so a reference core's output can be checked against a stock implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalisation in fp32, scaled by ``weight``, returned in ``x``'s dtype."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def inverse_frequencies(
    head_dim: int, theta: float, rope_scaling: Mapping[str, object] | None = None
) -> torch.Tensor:
    """The rotary inverse frequencies for ``head_dim``, with the llama3 schedule when configured.

    Args:
        head_dim: size of one attention head.
        theta: the RoPE base.
        rope_scaling: the model's ``rope_scaling`` mapping, or None. Only ``rope_type == "llama3"``
            changes the schedule; other types are the default schedule.
    """
    f = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    rs = rope_scaling or {}
    if rs.get("rope_type", rs.get("type")) == "llama3":
        factor = float(rs["factor"])  # type: ignore[arg-type]
        low = float(rs["low_freq_factor"])  # type: ignore[arg-type]
        high = float(rs["high_freq_factor"])  # type: ignore[arg-type]
        old_len = float(rs["original_max_position_embeddings"])  # type: ignore[arg-type]
        low_wavelength, high_wavelength = old_len / low, old_len / high
        wavelength = 2 * math.pi / f
        scaled = torch.where(wavelength > low_wavelength, f / factor, f)
        smooth = (old_len / wavelength - low) / (high - low)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        mid = ~(wavelength < high_wavelength) & ~(wavelength > low_wavelength)
        f = torch.where(mid, smoothed, scaled)
    return f


def rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``[tokens, heads, head_dim]`` by ``positions``, rotate-half form."""
    freqs = torch.outer(positions.float(), inv_freq.to(x.device))
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
    half = x.shape[-1] // 2
    xf = x.float()
    rotated = torch.cat([-xf[..., half:], xf[..., :half]], -1)
    return (xf * cos + rotated * sin).to(x.dtype)


def member_sum(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """The reference reduction of the slot collective: the members summed in order, in fp32.

    Starts from zero and adds each member's tensor in the order given (the collective's row
    order), then rounds once to the inputs' dtype. With one producer per element, as the decode
    slots are laid out, the sum is exact whatever the order.
    """
    acc = torch.zeros(inputs[0].shape, dtype=torch.float32, device=inputs[0].device)
    for x in inputs:
        acc += x.float()
    return acc.to(inputs[0].dtype)
