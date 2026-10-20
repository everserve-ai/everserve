# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare loads its position's compiled kernels, never its own (decision 0043, the worker half).

A spare that compiles its stage for itself runs other kernels than its rank's serving process
(autotune results chosen by timing on its own node) and is refused on its compile cache's digest.
Its node's store fetches the rank's set, exactly the files the serving
process's manifest lists, from a node that holds it, into the node's compile cache directory
(:data:`everserve.podcontract.COMPILE_CACHE_DIR`, the pods' ``VLLM_CACHE_ROOT``), with the
manifest last. Before the spare loads, its process checks what landed: the rank's digest,
recomputed over the files there (:func:`everserve.contract.compile_cache.rank_digest`), must be
the pair the control plane named. On a mismatch it loads nothing and is refused; it never
compiles in place of what was staged, which is the refusal this decision exists to avoid.
"""

from __future__ import annotations

from pathlib import Path

from everserve.contract.compile_cache import cache_root, manifest_path, rank_digest


class StagingRefused(RuntimeError):  # noqa: N818 - the spare's refusal, not an error's name
    """What was staged for a rank is not the pair the control plane named: load nothing."""


def artifact_dir(key: str, root: Path | None = None) -> Path:
    """Where the engine's artifact ``key`` lives under the node's compile cache."""
    return (root or cache_root()) / "torch_aot_compile" / key


def verify(key: str, rank: int, digest: str, dp: int = 0, root: Path | None = None) -> Path:
    """Check rank ``rank``'s staged set against the pair ``(key, digest)``; its artifact if so.

    Raises:
        StagingRefused: the manifest is missing (the set is not in place), or the files there
            digest to another pair than the one named.
    """
    art = artifact_dir(key, root)
    if not manifest_path(art, rank, dp).is_file():
        raise StagingRefused(f"rank {rank}'s compiled set for {key} is not in place under {art}")
    try:
        here = rank_digest(art, rank, dp)
    except FileNotFoundError as e:
        raise StagingRefused(f"rank {rank}'s compiled set for {key}: {e}") from e
    if here.digest != digest:
        raise StagingRefused(
            f"rank {rank}'s compiled set for {key} digests to {here.digest} here, "
            f"{digest} as its position holds it"
        )
    return art
