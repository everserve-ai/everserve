# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""An engine's compile cache digest: the key's files and the artifacts they name."""

from pathlib import Path

import pytest

from everserve.contract.compile_cache import digest, keys, referenced_artifacts

ARTIFACT = "a" * 64
OTHER = "b" * 64


def _cache(root: Path, best_config: bytes = b"BLOCK=64\n") -> Path:
    """A cache as vLLM lays it out: a key with a rank subtree naming an AOT artifact, and the
    artifact with an Inductor best_config inside; a second artifact nothing names."""
    key = root / "760834052d" / "rank_1_0" / "backbone"
    key.mkdir(parents=True)
    (key / "computation_graph.py").write_text("graph\n")
    (key / "cache_key_factors.json").write_text('{"code_hash": "c"}')
    (key / "aot_ref.json").write_text(f'{{"artifact": "{ARTIFACT}"}}')
    art = root / "torch_aot_compile" / ARTIFACT / "inductor_cache" / "j6"
    art.mkdir(parents=True)
    (art / "kernel.py").write_text("triton\n")
    (art / "tune.best_config").write_bytes(best_config)
    stray = root / "torch_aot_compile" / OTHER
    stray.mkdir(parents=True)
    (stray / "x.py").write_text("unrelated\n")
    return root


def test_the_digest_covers_the_key_and_the_artifact_it_names_and_nothing_else(tmp_path):
    root = _cache(tmp_path)
    d = digest(root, "760834052d")
    assert d.key == "760834052d" and d.files == 5  # 3 under the key, 2 under the named artifact
    assert [
        p.name for p in referenced_artifacts(root / "760834052d", root / "torch_aot_compile")
    ] == [ARTIFACT]
    (root / "torch_aot_compile" / OTHER / "y.py").write_text("still unrelated\n")
    assert digest(root, "760834052d").digest == d.digest


def test_two_caches_with_other_autotune_results_differ_and_equal_ones_match(tmp_path):
    a = digest(_cache(tmp_path / "node-b"), "760834052d")
    b = digest(_cache(tmp_path / "node-c", best_config=b"BLOCK=128\n"), "760834052d")
    c = digest(_cache(tmp_path / "node-c-staged"), "760834052d")
    assert a.digest != b.digest  # the same key, a tile chosen by timing elsewhere
    assert a.digest == c.digest  # the staged copy


def test_the_keys_are_the_cache_s_directories_but_the_artifacts(tmp_path):
    root = _cache(tmp_path)
    assert keys(root) == ["760834052d"]
    assert keys(tmp_path / "missing") == []
    with pytest.raises(FileNotFoundError):
        digest(root, "nope")


def test_without_a_named_artifact_every_artifact_counts_and_a_named_one_alone(tmp_path):
    """Nothing on disk links vLLM's key to its artifact; a member names the one it loaded."""
    root = _cache(tmp_path)
    (root / "760834052d" / "rank_1_0" / "backbone" / "aot_ref.json").unlink()
    everything = digest(root, "760834052d")
    assert everything.files == 5  # 2 under the key, 2 + 1 under the two artifacts
    named = digest(root, "760834052d", artifact=root / "torch_aot_compile" / ARTIFACT)
    assert named.files == 4 and named.digest != everything.digest
    (root / "torch_aot_compile" / OTHER / "x.py").write_text("changed\n")
    assert (
        digest(root, "760834052d", artifact=root / "torch_aot_compile" / ARTIFACT).digest
        == named.digest
    )


def test_an_artifact_alone_is_a_pair_keyed_by_its_hash(tmp_path):
    """vLLM's AOT path names only the artifact; its hash is the key, its files the digest."""
    from everserve.contract.compile_cache import artifact_digest

    root = _cache(tmp_path)
    art = root / "torch_aot_compile" / ARTIFACT
    a = artifact_digest(art)
    assert a.key == ARTIFACT and a.files == 2
    other = _cache(tmp_path / "node-c", best_config=b"BLOCK=128\n") / "torch_aot_compile" / ARTIFACT
    assert artifact_digest(other).digest != a.digest  # another autotune result, another pair
    with pytest.raises(FileNotFoundError):
        artifact_digest(tmp_path / "missing")


def _artifact(root: Path, ranks: tuple[int, ...], tile: dict[int, bytes]) -> Path:
    """An AOT artifact as vLLM lays it out: one subtree per rank naming its kernels, one
    shared inductor_cache holding every rank's kernels and their autotune results."""
    from everserve.contract.compile_cache import rank_digest  # noqa: F401 - the layout under test

    art = root / "torch_aot_compile" / ("f" * 64)
    for r in ranks:
        kernel = f"c{str(r) * 51}"
        (art / f"rank_{r}_0").mkdir(parents=True)
        (art / f"rank_{r}_0" / "model").write_bytes(b"graph " + kernel.encode() + b" end")
        d = art / "inductor_cache" / kernel[1:3]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{kernel}.py").write_text(f"kernel {r}\n")
        (d / f"{'e' * 64}.best_config").write_bytes(tile.get(r, b"X=4\n"))
    return art


def test_a_rank_s_digest_covers_its_subtree_and_the_kernels_it_names_only(tmp_path):
    from everserve.contract.compile_cache import rank_digest

    both = _artifact(tmp_path / "spare-node", (0, 1), {})  # staged with both ranks' shares
    only1 = _artifact(tmp_path / "node-b", (1,), {})  # the serving node, its own rank
    assert rank_digest(both, 1).digest == rank_digest(only1, 1).digest
    assert rank_digest(both, 0).digest != rank_digest(both, 1).digest
    assert rank_digest(both, 1).key == rank_digest(both, 0).key == "f" * 64
    tuned = _artifact(tmp_path / "node-c-own", (1,), {1: b"X=64\n"})  # another autotune result
    assert rank_digest(tuned, 1).digest != rank_digest(only1, 1).digest
    with pytest.raises(FileNotFoundError):
        rank_digest(only1, 0)


def test_a_rank_s_manifest_lists_exactly_what_its_digest_covers_and_lands_whole(tmp_path):
    import json
    import stat

    from everserve.contract.compile_cache import (
        manifest_path,
        rank_digest,
        rank_manifest,
        write_manifest,
    )

    art = _artifact(tmp_path / "node-b", (0, 1), {1: b"X=64\n"})
    m = rank_manifest(art, 1)
    assert m.pair == rank_digest(art, 1)  # one pass: the set moved is the set the digest names
    assert all(not f.path.startswith("/") and ".." not in f.path for f in m.files)
    assert {f.path.split("/")[0] for f in m.files} <= {"rank_1_0", "inductor_cache"}
    written = write_manifest(art, m)
    assert written == manifest_path(art, 1) == art / "rank_1_0.manifest.json"
    doc = json.loads(written.read_text())
    assert (doc["key"], doc["rank"], doc["dp"], doc["digest"]) == ("f" * 64, 1, 0, m.digest)
    assert [f["path"] for f in doc["files"]] == [f.path for f in m.files]
    assert stat.S_IMODE(written.stat().st_mode) == 0o660  # the node group's, as the store reads
    assert not list(art.glob(".*.tmp"))  # renamed into place, nothing left beside it


def test_a_published_set_is_readable_by_the_node_s_group_though_vllm_wrote_it_for_itself(
    tmp_path,
):
    import stat

    from everserve.contract.compile_cache import rank_manifest, share

    art = _artifact(tmp_path / "node-b", (1,), {1: b"X=64\n"})
    m = rank_manifest(art, 1)
    for f in m.files:  # as vLLM leaves them: a temporary file renamed into place, 0600
        (art / f.path).chmod(0o600)
    (art / "rank_1_0").chmod(0o700)
    share(art, m)
    for f in m.files:
        assert (art / f.path).stat().st_mode & stat.S_IRGRP  # the store reads as the group
        parent = (art / f.path).parent
        while parent != art.parent:
            assert parent.stat().st_mode & stat.S_IXGRP and parent.stat().st_mode & stat.S_IRGRP
            parent = parent.parent
    assert not (art / m.files[0].path).stat().st_mode & stat.S_IROTH  # the group's, no one else's
