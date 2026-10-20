# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare checks its staged compiled set against its position's pair before it loads (0043)."""

import pytest

from everserve.contract.compile_cache import rank_manifest, write_manifest
from everserve_vllm import staging


def _staged(root, key, ranks=(1,)):
    art = root / "torch_aot_compile" / key
    for r in ranks:
        kernel = f"c{str(r) * 51}"
        (art / f"rank_{r}_0").mkdir(parents=True)
        (art / f"rank_{r}_0" / "model").write_bytes(b"graph " + kernel.encode())
        d = art / "inductor_cache" / kernel[1:3]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{kernel}.py").write_text(f"kernel {r}\n")
        (d / f"{kernel}.best_config").write_text("BLOCK=64\n")
    return art


def test_a_set_staged_as_its_position_holds_it_is_loaded_from(tmp_path):
    art = _staged(tmp_path, "e" * 64)
    m = rank_manifest(art, 1)
    write_manifest(art, m)  # the manifest lands last: the set is in place
    assert staging.verify("e" * 64, 1, m.digest, root=tmp_path) == art


def test_a_set_not_yet_in_place_is_not_loaded_from(tmp_path):
    art = _staged(tmp_path, "e" * 64)  # files there, the manifest not yet
    with pytest.raises(staging.StagingRefused, match="not in place"):
        staging.verify("e" * 64, 1, rank_manifest(art, 1).digest, root=tmp_path)


def test_a_set_that_digests_otherwise_than_its_position_is_refused(tmp_path):
    art = _staged(tmp_path, "e" * 64)
    m = rank_manifest(art, 1)
    write_manifest(art, m)
    best = next(art.rglob("*.best_config"))
    best.write_text("BLOCK=128\n")  # another autotune result than the rank's
    with pytest.raises(staging.StagingRefused, match="as its position holds it"):
        staging.verify("e" * 64, 1, m.digest, root=tmp_path)
