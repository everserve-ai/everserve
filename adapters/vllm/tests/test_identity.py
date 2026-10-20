# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What a process says it runs (decision 0034): each reading, and that none of them guesses."""

import hashlib
import shutil
import subprocess

import pytest
import torch

from everserve_vllm import identity


@pytest.mark.parametrize(
    ("version", "recorded", "commit"),
    [
        ("0.11.1.dev45+g6e517b15c.cu129", None, "6e517b15c"),
        (
            "0.11.1.dev45+g6e517b15c",
            "g6e517b15c1833cf72a7f557ee32524d98682e617",
            "6e517b15c1833cf72a7f557ee32524d98682e617",
        ),
        ("0.31.0", None, "v0.31.0"),  # a release names one tagged commit
        ("dev", None, ""),  # vLLM's own fallback says nothing
        ("0.31.0.dev3", None, ""),  # a dev build without its commit says nothing either
    ],
)
def test_the_engine_s_commit_is_read_from_its_version_record_never_guessed(
    version, recorded, commit
):
    assert identity.engine_commit(version, recorded) == commit


def test_an_archived_tree_is_its_stamped_commit_and_clean(tmp_path):
    assert identity.source_commit("a" * 40, tmp_path) == ("a" * 40, False)


def _git(tree, *args):
    subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="a checkout needs git")
def test_a_checkout_is_its_head_and_dirty_only_for_core_or_adapters(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "a.py").write_text("x = 1\n")
    source = tmp_path / "adapters" / "vllm" / "everserve_vllm" / "_source.py"
    source.parent.mkdir(parents=True)
    source.write_text("COMMIT = '$Format:%H$'\n")
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "b.txt").write_text("run\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "c")
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    stamp = "$Format:%H$"
    assert identity.source_commit(stamp, tmp_path, source_file=source) == (head, False)
    (tmp_path / "bench" / "b.txt").write_text("another run\n")  # a run record is not the code
    (tmp_path / "core" / "new.py").write_text("")  # nor is an untracked file
    assert identity.source_commit(stamp, tmp_path, source_file=source) == (head, False)
    (tmp_path / "core" / "a.py").write_text("x = 2\n")
    assert identity.source_commit(stamp, tmp_path, source_file=source) == (head, True)


def test_no_stamp_and_no_checkout_is_no_commit(tmp_path):
    assert identity.source_commit("$Format:%H$", tmp_path / "gone") == ("", False)


@pytest.mark.skipif(shutil.which("git") is None, reason="a checkout needs git")
def test_a_checkout_that_is_not_this_adapter_s_tree_is_no_commit(tmp_path):
    # the package installed into a virtual environment that lives inside some other repository:
    # three directories above the module is that repository's checkout, whose commit is not ours
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README").write_text("another project\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "theirs")
    venv_module = tmp_path / ".venv" / "lib" / "site-packages" / "everserve_vllm" / "_source.py"
    assert identity.source_commit("$Format:%H$", tmp_path, source_file=venv_module) == ("", False)
    # a nested checkout of this tree inside another repository is believed only at its own root
    nested = tmp_path / "vendor" / "everserve"
    source = nested / "adapters" / "vllm" / "everserve_vllm" / "_source.py"
    source.parent.mkdir(parents=True)
    source.write_text("")
    assert identity.source_commit("$Format:%H$", nested, source_file=source) == ("", False)


def _hf_snapshot(root, files):
    """A Hugging Face cache layout: large files as symlinks to blobs named by their sha256."""
    blobs = root / "blobs"
    snap = root / "snapshots" / "f50dbad2"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, data, lfs in files:
        # a large file's blob is named by its sha256, a small one's by git's sha1
        blob = blobs / (hashlib.sha256(data).hexdigest() if lfs else hashlib.sha1(data).hexdigest())
        blob.write_bytes(data)
        (snap / name).symlink_to(blob)
    return snap


FILES = [
    ("model-00001.safetensors", b"weights one", True),
    ("model-00002.safetensors", b"weights two", True),
    ("config.json", b'{"hidden": 8}', False),
]


def test_a_checkpoint_digests_the_same_in_the_cache_and_copied_out(tmp_path, monkeypatch):
    snap = _hf_snapshot(tmp_path / "hub", FILES)
    plain = tmp_path / "plain"
    plain.mkdir()
    for name, data, _ in FILES:
        (plain / name).write_bytes(data)
    read = []
    original = identity._file_sha256
    monkeypatch.setattr(identity, "_file_sha256", lambda p: read.append(p.name) or original(p))
    assert identity.checkpoint_digest(snap) == identity.checkpoint_digest(plain)
    assert read.count("config.json") == 2  # the small file is read in both
    assert read.count("model-00001.safetensors") == 1  # a large one only outside the cache


def _xet_snapshot(root, files):
    """The layout Hugging Face's Xet downloader leaves: a large file's blob, named by its sha256,
    links on into a store whose file is named by another 64-hex hash."""
    snap = _hf_snapshot(root / "models--x", files)
    store = root / "blobs" / "bb"
    store.mkdir(parents=True)
    for _, data, lfs in files:
        if lfs:
            blob = root / "models--x" / "blobs" / hashlib.sha256(data).hexdigest()
            other = store / hashlib.sha256(b"xet:" + data).hexdigest()
            other.write_bytes(data)
            blob.unlink()
            blob.symlink_to(other)
    return snap


def test_one_checkpoint_digests_alike_whichever_downloader_filled_the_cache(tmp_path):
    """One node holds plain blobs and another Xet's; were the digests to differ, a rank's spare
    would be refused as another checkpoint."""
    plain = _hf_snapshot(tmp_path / "plain", FILES)
    xet = _xet_snapshot(tmp_path / "xet", FILES)
    assert identity.checkpoint_digest(xet) == identity.checkpoint_digest(plain) != ""


def test_a_file_outside_the_snapshot_layout_is_hashed_by_its_bytes(tmp_path, monkeypatch):
    """No digest by another method that could differ between nodes holding the same bytes: a link
    to a 64-hex name outside a ``blobs`` directory, or to any other name, is read."""
    data = b"weights one"
    store = tmp_path / "store"
    store.mkdir()
    (store / hashlib.sha256(b"not the content").hexdigest()).write_bytes(data)
    (store / "model.bin").write_bytes(data)
    odd = tmp_path / "odd.safetensors"
    odd.symlink_to(store / hashlib.sha256(b"not the content").hexdigest())
    named = tmp_path / "named.safetensors"
    named.symlink_to(store / "model.bin")
    read = []
    original = identity._file_sha256
    monkeypatch.setattr(identity, "_file_sha256", lambda p: read.append(p.name) or original(p))
    true = hashlib.sha256(data).hexdigest()
    assert identity.content_sha256(odd) == identity.content_sha256(named) == true
    assert read == ["odd.safetensors", "named.safetensors"]


def test_a_checkpoint_s_digest_changes_with_any_file(tmp_path):
    a = _hf_snapshot(tmp_path / "a", FILES)
    b = _hf_snapshot(tmp_path / "b", [*FILES[:2], ("config.json", b'{"hidden": 9}', False)])
    assert identity.checkpoint_digest(a) != identity.checkpoint_digest(b)


def test_two_processes_that_fetched_other_files_of_one_checkpoint_digest_alike(tmp_path):
    vllm = _hf_snapshot(tmp_path / "a", FILES)  # what vLLM fetches: the weights it loads, config
    whole = _hf_snapshot(  # a framework's downloader: the repository, both weight formats
        tmp_path / "b",
        [
            *FILES,
            ("tokenizer.json", b"{}", False),
            ("README.md", b"# model", False),
            ("pytorch_model.bin", b"the same weights, pickled", True),
            ("model.safetensors.index.json", b"{}", False),
        ],
    )
    assert identity.checkpoint_digest(vllm) == identity.checkpoint_digest(whole) != ""
    names = [p.name for p in identity.checkpoint_files(whole)]
    assert names == ["config.json", "model-00001.safetensors", "model-00002.safetensors"]


def test_a_checkpoint_without_weights_says_nothing(tmp_path):
    configs_only = _hf_snapshot(tmp_path / "a", [("config.json", b"{}", False)])
    assert identity.checkpoint_digest(configs_only) == ""


def test_the_snapshot_is_found_in_the_cache_as_it_is_on_disk(tmp_path):
    root = tmp_path / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct"
    commit = "7ae557604adf67be50417f59c2c2f167def9a775"
    (root / "snapshots" / commit).mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(commit + "\n")
    cache = tmp_path / "hub"
    found = identity.hf_snapshot("Qwen/Qwen2.5-0.5B-Instruct", None, cache)
    assert found == root / "snapshots" / commit  # however few of its files are there
    assert identity.hf_snapshot("Qwen/Qwen2.5-0.5B-Instruct", commit, cache) == found
    assert identity.hf_snapshot("Qwen/Qwen2.5-0.5B-Instruct", "v2", cache) is None  # no such ref
    assert identity.hf_snapshot("Qwen/Qwen2.5-0.5B-Instruct", "0" * 40, cache) is None
    assert identity.hf_snapshot("Qwen/other", None, cache) is None


def test_the_snapshot_says_its_revision_and_a_plain_directory_none(tmp_path):
    assert identity.snapshot_revision(tmp_path / "snapshots" / "f50dbad2") == "f50dbad2"
    assert identity.snapshot_revision(tmp_path / "model") == ""


def test_the_kv_format_is_read_off_the_cache_tensors():
    nhd = torch.zeros(2, 4, 16, 8, 64, dtype=torch.bfloat16)  # K/V, blocks, block, heads, dim
    hnd = torch.zeros(2, 4, 8, 16, 64, dtype=torch.bfloat16).transpose(2, 3)  # same shape
    assert nhd.shape == hnd.shape
    a = identity.kv_format({"layers.0": nhd, "layers.1": nhd}, "FLASH_ATTN")
    assert (a.dtype, a.block_size, len(a.layout_hash)) == ("bfloat16", 16, 64)
    assert identity.kv_format({"layers.0": hnd, "layers.1": hnd}, "FLASH_ATTN") != a  # strides
    assert identity.kv_format({"layers.0": nhd}, "FLASH_ATTN") != a  # another stage's layers
    assert identity.kv_format({"layers.0": nhd, "layers.1": nhd}, "FLASHINFER") != a
    with pytest.raises(ValueError):
        identity.kv_format({}, "FLASH_ATTN")


def _resolved(max_num_seqs=256, sizes=(1, 2, 4), dtype=torch.bfloat16, spec=None):
    from enum import Enum
    from types import SimpleNamespace as NS

    class Mode(Enum):
        FULL_AND_PIECEWISE = 2

    return NS(
        scheduler_config=NS(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            async_scheduling=False,
        ),
        model_config=NS(
            max_model_len=4096, enforce_eager=False, dtype=dtype, quantization=None, max_logprobs=20
        ),
        compilation_config=NS(
            mode=3,
            cudagraph_mode=Mode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=list(sizes),
            splitting_ops=["everserve::all_reduce_into", "vllm::unified_attention"],
        ),
        cache_config=NS(block_size=16, cache_dtype="auto"),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=2),
        speculative_config=spec,
    )


def test_a_stage_that_resolved_its_step_otherwise_than_its_head_differs_by_name():
    head_digest, head = identity.step_config(_resolved())
    node_digest, node = identity.step_config(_resolved(max_num_seqs=128))
    assert head_digest != node_digest
    assert {k for k in head if head[k] != node[k]} == {"max_num_seqs"}  # the proof's mismatch
    assert head["max_num_seqs"] == "256" and node["max_num_seqs"] == "128"


def test_the_step_config_reads_the_same_on_every_process_that_resolved_the_same():
    a, fields = identity.step_config(_resolved(sizes=(4, 1, 2)))
    b, _ = identity.step_config(_resolved(sizes=[1, 2, 4]))
    assert a == b  # the capture sizes as a set, in any order
    assert fields["dtype"] == "bfloat16"  # torch's name without its module
    assert fields["cudagraph_mode"] == "FULL_AND_PIECEWISE"  # an enum by its name
    assert fields["speculative_method"] == fields["num_speculative_tokens"] == ""  # none
    assert fields["prefill_context_parallel"] == ""  # a field this vLLM lacks says nothing
    sections = {f"vllm.{s.removesuffix('_config')}" for s in identity.VLLM_SECTIONS}
    assert set(fields) == {name for name, _ in identity.STEP_FIELDS} | sections | {
        "vllm.additional",
        "vllm.config",
    }
    assert fields["vllm.scheduler"] == ""  # a section that cannot hash itself says nothing
    text = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    assert a == hashlib.sha256(text.encode()).hexdigest()


def test_a_section_vllm_hashes_otherwise_is_named_by_its_hash():
    from types import SimpleNamespace as NS

    def resolved(scheduler_hash):
        cfg = _resolved()
        cfg.scheduler_config.compute_hash = lambda: scheduler_hash
        cfg.cache_config.compute_hash = lambda: "c4c4"
        cfg.compute_hash = lambda: f"whole-{scheduler_hash}"
        return cfg

    head_digest, head = identity.step_config(resolved("aaaa"))  # Dynamo's head, say
    node_digest, node = identity.step_config(resolved("bbbb"))
    assert head_digest != node_digest
    assert {k for k in head if head[k] != node[k]} == {"vllm.scheduler", "vllm.config"}
    assert head["vllm.cache"] == node["vllm.cache"] == "c4c4"
    broken = _resolved()
    broken.lora_config = NS(compute_hash=lambda: 1 / 0)
    assert identity.step_config(broken)[1]["vllm.lora"] == "unhashable: ZeroDivisionError"


def test_an_additional_config_the_head_s_launcher_added_is_named_as_such():
    head, node = _resolved(), _resolved()
    head.additional_config = {"consolidator_endpoints": None}  # what Dynamo's head adds
    node.additional_config = {}
    _, h = identity.step_config(head)
    _, n = identity.step_config(node)
    assert {k for k in h if h[k] != n[k]} == {"vllm.additional"}  # the one difference, named
    same = _resolved()
    same.additional_config = {"consolidator_endpoints": None}
    assert identity.step_config(same) == identity.step_config(head)
