# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The compile cache pair a worker reports at registration (decision 0029)."""

from types import SimpleNamespace

from everserve.contract.compile_cache import digest
from everserve_vllm.worker import compiled_with


def _cache(root, best_config=b"BLOCK=64\n"):
    """vLLM's layout: the key's directory and the AOT artifact the process loaded."""
    key = root / "torch_compile_cache" / "760834052d" / "rank_1_0" / "backbone"
    key.mkdir(parents=True)
    (key / "computation_graph.py").write_text("graph\n")
    art = root / "torch_compile_cache" / "torch_aot_compile" / ("a" * 64) / "inductor_cache" / "jx"
    art.mkdir(parents=True)
    (art / "k.py").write_text("triton\n")
    (art / "k.best_config").write_bytes(best_config)
    return key.parent.parent, art.parent.parent


def _config(cache_dir, artifact):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cache_dir=str(cache_dir), local_cache_dir=str(artifact))
    )


def test_the_pair_is_the_artifact_s_hash_whichever_path_compiled(tmp_path):
    """A rank that compiled fresh has cache_dir set; the artifact it saved is still the key."""
    from everserve.contract.compile_cache import artifact_digest

    key_dir, artifact = _cache(tmp_path)
    pair = compiled_with(_config(key_dir, artifact))
    assert pair is not None and pair.key == "a" * 64
    assert pair.digest == artifact_digest(artifact).digest
    # without any artifact, the key's directory names the pair
    bare = compiled_with(_config(key_dir, key_dir / "rank_1_0"))
    assert bare is not None and bare.key == "760834052d"
    assert bare.digest == digest(key_dir.parent, "760834052d").digest
    # another node's autotune result under the same key: another pair
    other_key, other_artifact = _cache(tmp_path / "node-c", best_config=b"BLOCK=128\n")
    assert compiled_with(_config(other_key, other_artifact)).digest != pair.digest


def test_no_pair_when_the_engine_compiled_nothing_or_the_cache_is_gone(tmp_path):
    assert compiled_with(SimpleNamespace(compilation_config=SimpleNamespace(cache_dir=""))) is None
    assert compiled_with(SimpleNamespace()) is None
    assert compiled_with(_config(tmp_path / "missing" / "k", None)) is None


def test_on_the_aot_path_the_pair_is_the_artifact_the_process_loaded(tmp_path):
    """cache_dir stays empty on vLLM's AOT path; local_cache_dir names the artifact."""
    from everserve.contract.compile_cache import artifact_digest

    _key_dir, artifact = _cache(tmp_path)
    pair = compiled_with(_config("", artifact))
    assert pair is not None and pair.key == "a" * 64
    assert pair.digest == artifact_digest(artifact).digest


def test_the_pair_comes_from_the_compiled_model_when_the_config_names_nothing(tmp_path):
    """The spare registers from a configuration object the compile never wrote to; the model
    carries the artifact it loaded."""
    from everserve.contract.compile_cache import artifact_digest

    _key_dir, artifact = _cache(tmp_path)
    bare = SimpleNamespace(compilation_config=SimpleNamespace(cache_dir="", local_cache_dir=None))
    model = SimpleNamespace(_aot_cache_dir=str(artifact))
    pair = compiled_with(bare, model)
    assert pair is not None and pair.key == "a" * 64
    assert pair.digest == artifact_digest(artifact).digest
    assert compiled_with(bare, SimpleNamespace()) is None


def test_with_a_rank_the_pair_is_the_rank_s_share_of_the_artifact(tmp_path):
    """A spare node staged with every rank's share reports, for rank 1, rank 1's digest."""
    from everserve.contract.compile_cache import rank_digest

    art = tmp_path / "torch_aot_compile" / ("f" * 64)
    for r in (0, 1):
        kernel = f"c{str(r) * 51}"
        (art / f"rank_{r}_0").mkdir(parents=True)
        (art / f"rank_{r}_0" / "model").write_bytes(b"graph " + kernel.encode())
        d = art / "inductor_cache" / kernel[1:3]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{kernel}.py").write_text(f"kernel {r}\n")
    pair = compiled_with(_config("", art), rank=1)
    assert pair is not None and pair.key == "f" * 64
    assert pair.digest == rank_digest(art, 1).digest
    assert compiled_with(_config("", art), rank=0).digest != pair.digest


def test_a_rank_directory_vllm_names_reports_the_engine_s_hash_and_the_rank_s_share(tmp_path):
    """vLLM 0.31 names ``torch_aot_compile/<hash>/rank_<r>_0`` as the artifact; the key is the
    hash, the same on every stage, and the digest the rank's (never the key "rank_1_0")."""
    from everserve.contract.compile_cache import rank_digest

    art = tmp_path / "torch_aot_compile" / ("e" * 64)
    for r in (0, 1):
        kernel = f"c{str(r) * 51}"
        (art / f"rank_{r}_0").mkdir(parents=True)
        (art / f"rank_{r}_0" / "model").write_bytes(b"graph " + kernel.encode())
        d = art / "inductor_cache" / kernel[1:3]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{kernel}.py").write_text(f"kernel {r}\n")
    stage1 = compiled_with(_config("", art / "rank_1_0"), rank=1)
    stage0 = compiled_with(_config("", art / "rank_0_0"), rank=0)
    assert stage1 is not None and stage0 is not None
    assert stage1.key == stage0.key == "e" * 64  # the engine's one key
    assert stage1.digest == rank_digest(art, 1).digest != stage0.digest
    spare = SimpleNamespace(compilation_config=SimpleNamespace(cache_dir="", local_cache_dir=None))
    on_model = compiled_with(spare, SimpleNamespace(_aot_cache_dir=str(art / "rank_1_0")), rank=1)
    assert on_model == stage1


def test_the_serving_rank_publishes_its_manifest_and_a_spare_never_does(tmp_path):
    """The serving process's report writes the manifest its node's store serves (decision
    0043); a spare's report reads, and writes nothing beside what was staged for it."""
    from everserve.contract.compile_cache import manifest_path, rank_manifest

    art = tmp_path / "torch_aot_compile" / ("d" * 64)
    for r in (0, 1):
        (art / f"rank_{r}_0").mkdir(parents=True)
        (art / f"rank_{r}_0" / "model").write_bytes(f"graph {r}".encode())
    spare = compiled_with(_config("", art / "rank_1_0"), rank=1)
    assert not manifest_path(art, 1).exists()
    serving = compiled_with(_config("", art / "rank_1_0"), rank=1, publish=True)
    m = rank_manifest(art, 1)
    assert serving == spare and (serving.key, serving.digest) == (m.key, m.digest)
    assert manifest_path(art, 1).exists() and not manifest_path(art, 0).exists()


def test_the_artifact_is_found_on_the_backbone_under_the_model(tmp_path):
    """vLLM decorates the backbone, a submodule: the pair is read off the module tree."""
    from everserve.contract.compile_cache import artifact_digest

    _key_dir, artifact = _cache(tmp_path)
    backbone = SimpleNamespace(_aot_cache_dir=str(artifact))
    model = SimpleNamespace(modules=lambda: [backbone])
    bare = SimpleNamespace(compilation_config=SimpleNamespace(cache_dir="", local_cache_dir=None))
    pair = compiled_with(bare, model)
    assert pair is not None and pair.digest == artifact_digest(artifact).digest
