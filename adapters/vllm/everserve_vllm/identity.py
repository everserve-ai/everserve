# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What this process runs, as its registration says it (decision 0034).

The coordinator compares a destination's identity with the position's before anything moves, and
an empty compared field never matches, so every reading here says nothing rather than guess:

- the engine's commit from vLLM's own version record (:func:`engine_commit`), never its version
  string, which dev builds share;
- this tree's commit (:func:`source_commit`): stamped by ``git archive`` into ``_source.py``
  (``export-subst``), which is how a node gets its tree, or read from the checkout with whether
  ``core`` or ``adapters`` differ from it;
- the checkpoint's digest (:func:`checkpoint_digest`) over its weights and its config, every
  weight file and never a shard, found in the Hugging Face cache as it is on disk
  (:func:`hf_snapshot`): a large file's blob is named by its sha256, so only the config is read;
  a checkpoint outside the cache is read whole, once per process;
- the KV format from the cache tensors themselves (:func:`kv_format`), as the store reads them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from everserve.contract.signals import KvFormat, ParallelWidths, SoftwareIdentity, StepConfig
from everserve_vllm import _source, serving, settings

log = logging.getLogger(__name__)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_LOCAL_COMMIT = re.compile(r"\+g([0-9a-f]{7,40})")
_RELEASE = re.compile(r"\d+\.\d+\.\d+")
_CHUNK = 16 * 1024 * 1024
TREE = Path(__file__).resolve().parents[3]
"""The repository root of a checkout or a ``git archive``: ``adapters/vllm/everserve_vllm``'s
third parent."""


def engine_commit(version: str, commit_id: str | None = None) -> str:
    """VLLM's source commit: its recorded commit, else a dev build's ``+g<sha>``, else a release.

    A release's version names one tagged commit, so ``v<version>`` stands for it. Empty when the
    version says nothing (vLLM's own fallback is ``"dev"``).
    """
    if commit_id:
        return commit_id.removeprefix("g")
    m = _LOCAL_COMMIT.search(version)
    if m:
        return m.group(1)
    if _RELEASE.fullmatch(version):
        return f"v{version}"
    return ""


def source_commit(
    stamped: str = _source.COMMIT,
    tree: Path = TREE,
    git: str = "git",
    source_file: Path | str = _source.__file__,
) -> tuple[str, bool]:
    """This tree's commit and whether ``core`` or ``adapters`` differ from it.

    A ``git archive`` stamps the commit (an archive is that commit exactly, so never dirty); a
    checkout is asked, and only a checkout whose root is ``tree`` and whose adapter is this very
    module (``source_file``) is believed: a package installed into a virtual environment that
    happens to live inside some other repository's checkout would otherwise report that
    repository's commit as its own. ``("", False)`` when nothing says, which a destination is
    refused for.
    """
    if not stamped.startswith("$Format"):
        return stamped, False
    if (tree / "adapters" / "vllm" / "everserve_vllm" / "_source.py").resolve() != Path(
        source_file
    ).resolve():
        log.warning("no commit stamped and %s is not this adapter's checkout: no commit", tree)
        return "", False
    try:
        top = subprocess.run(
            [git, "-C", str(tree), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        if Path(top).resolve() != tree.resolve():
            log.warning("no commit stamped and %s is inside another checkout: no commit", tree)
            return "", False
        head = subprocess.run(
            [git, "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        changed = subprocess.run(
            [git, "-C", str(tree), "status", "--porcelain", "--untracked-files=no", "--"]
            + ["core", "adapters"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        log.warning("no commit stamped and no checkout at %s: the identity has no commit", tree)
        return "", False
    return head, bool(changed)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(path: Path) -> str:
    """A file's sha256: its blob's name when the Hugging Face cache named it so, else read.

    A snapshot's file links to ``../../blobs/<name>``, and a large file's blob is named by its
    sha256. One link only: the blob may itself link into a store named by another hash. Hugging
    Face's Xet downloader leaves ``blobs/<sha256>`` pointing at ``hub/blobs/xx/<Xet hash>``, also
    64 hex, which is not the file's sha256. Following every link would read that name as the
    digest, so one checkpoint digests two ways on two nodes whose caches different downloaders
    filled, and a rank's spare is refused as another checkpoint.
    """
    if path.is_symlink():
        target = Path(os.readlink(path))
        if target.parent.name == "blobs" and _HEX64.fullmatch(target.name):
            return target.name
    return _file_sha256(path)


WEIGHTS = (".safetensors", ".bin", ".pt", ".pth")
"""The suffixes of a checkpoint's weight files, in vLLM's order of preference."""
CONFIG = "config.json"


def checkpoint_files(snapshot: Path) -> list[Path]:
    """The files of ``snapshot`` that decide a process's numerics: its weights and its config.

    The weights of the first suffix in :data:`WEIGHTS` that the checkpoint has, as vLLM's loader
    prefers safetensors over a ``.bin`` beside it. Two processes fetch a checkpoint differently
    (vLLM fetches what it loads, a framework's downloader may fetch the whole repository with its
    tokenizer, README and both weight formats), so a digest over every file would tell two
    processes holding the same weights apart.
    """
    files = [
        p
        for p in snapshot.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(snapshot).parts
    ]
    for suffix in WEIGHTS:
        weights = [p for p in files if p.name.endswith(suffix)]
        if weights:
            break
    else:
        weights = []
    config = [p for p in files if p.relative_to(snapshot).as_posix() == CONFIG]
    return sorted([*weights, *config])


def checkpoint_digest(snapshot: Path) -> str:
    """sha256 over the sorted (relative name, content sha256) of :func:`checkpoint_files`.

    Empty when ``snapshot`` holds no weights: a checkpoint not yet fetched says nothing.
    """
    names = checkpoint_files(snapshot)
    if not any(p.name.endswith(WEIGHTS) for p in names):
        return ""
    h = hashlib.sha256()
    for p in names:
        h.update(f"{p.relative_to(snapshot).as_posix()}\t{content_sha256(p)}\n".encode())
    return h.hexdigest()


def snapshot_revision(snapshot: Path) -> str:
    """The Hugging Face snapshot's commit (``.../snapshots/<commit>``), else empty."""
    parts = snapshot.parts
    return parts[-1] if len(parts) >= 2 and parts[-2] == "snapshots" else ""


def layout_hash(backend: str, layout: str, kv_heads: int, head_size: int, layers: int) -> str:
    """What decides whether one position's KV rows are another's, as one sha256.

    Over the attention backend, the cache layout, the rank's KV heads, the head size and the
    stage's layer count.
    """
    text = f"{backend}\n{layout}\n{kv_heads}\n{head_size}\n{layers}\n"
    return hashlib.sha256(text.encode()).hexdigest()


def cache_layout(t: Any) -> str:
    """One layer's cache tensor as the store reads it, and its axes in memory order.

    The shape says which of the three layouts it is (:class:`~everserve.state.kvcopy.PagedKV`);
    the strides say the order its axes are stored in, which vLLM's NHD and HND layouts differ by
    with one shape.
    """
    from everserve.state.kvcopy import PagedKV

    order = sorted(range(t.dim()), key=lambda d: -t.stride(d))
    return f"{PagedKV.of(t).layout.value}:{','.join(map(str, order))}"


def kv_format(kv_caches: Mapping[str, Any], backend: str) -> KvFormat:
    """This rank's KV format, read off its cache tensors (one per layer of its stage).

    Raises:
        ValueError: for no cache tensor, or one of a layout the store does not read.
    """
    from everserve.state.kvcopy import CacheLayout, PagedKV

    if not kv_caches:
        raise ValueError("no KV cache tensors to read a format from")
    t = next(iter(kv_caches.values()))
    paged = PagedKV.of(t)
    heads = t.shape[1] if paged.layout is CacheLayout.COMBINED else t.shape[-2]
    return KvFormat(
        dtype=str(t.dtype).removeprefix("torch."),
        block_size=paged.block_size,
        layout_hash=layout_hash(backend, cache_layout(t), heads, paged.head_dim, len(kv_caches)),
    )


def attention_backend(vllm_config: Any) -> str:
    """The attention backend's name, from the layers vLLM registered for the forward pass.

    One name for the stage; several (a hybrid model's) are joined in sorted order.
    """
    layers = getattr(vllm_config.compilation_config, "static_forward_context", {}) or {}
    names: set[str] = set()
    for layer in layers.values():
        backend = getattr(layer, "attn_backend", None)
        if backend is not None:
            names.add(backend.get_name())
    return "+".join(sorted(names))


def widths(parallel: Any) -> ParallelWidths:
    """The engine's parallel widths from vLLM's parallel configuration."""
    tp = parallel.tensor_parallel_size
    dp = parallel.data_parallel_size
    return ParallelWidths(
        tensor=tp,
        pipeline=parallel.pipeline_parallel_size,
        data=dp,
        expert=tp * dp if getattr(parallel, "enable_expert_parallel", False) else 1,
        context=getattr(parallel, "decode_context_parallel_size", 1),
    )


STEP_FIELDS: tuple[tuple[str, str], ...] = (
    ("max_num_seqs", "scheduler_config.max_num_seqs"),
    ("max_num_batched_tokens", "scheduler_config.max_num_batched_tokens"),
    ("max_model_len", "model_config.max_model_len"),
    ("enable_chunked_prefill", "scheduler_config.enable_chunked_prefill"),
    ("async_scheduling", "scheduler_config.async_scheduling"),
    ("compile_mode", "compilation_config.mode"),
    ("cudagraph_mode", "compilation_config.cudagraph_mode"),
    ("cudagraph_capture_sizes", "compilation_config.cudagraph_capture_sizes"),
    ("splitting_ops", "compilation_config.splitting_ops"),
    ("enforce_eager", "model_config.enforce_eager"),
    ("block_size", "cache_config.block_size"),
    ("cache_dtype", "cache_config.cache_dtype"),
    ("dtype", "model_config.dtype"),
    ("quantization", "model_config.quantization"),
    ("max_logprobs", "model_config.max_logprobs"),
    ("tensor_parallel", "parallel_config.tensor_parallel_size"),
    ("pipeline_parallel", "parallel_config.pipeline_parallel_size"),
    ("decode_context_parallel", "parallel_config.decode_context_parallel_size"),
    ("prefill_context_parallel", "parallel_config.prefill_context_parallel_size"),
    ("speculative_method", "speculative_config.method"),
    ("num_speculative_tokens", "speculative_config.num_speculative_tokens"),
)
"""The resolved settings that shape what every stage of an engine steps with, by their names in
the identity and their paths in vLLM's configuration.

Each process resolves its own configuration from the engine arguments, and vLLM fills what the
arguments leave out with defaults that depend on how the process was launched (the usage
context) and on its hardware. A stage that resolved any of these differently from its head
runs collectives of other shapes than its head, and NCCL waits on a mismatched collective
rather than failing (``max_num_seqs`` 128 on a node against 256 on the head: a sampled-token
broadcast that never completes). Every process logs them and
their digest (:func:`step_config`) as it builds its identity; the step guard (protocol 11's
step config, with shared groups) carries them in the registration, so that a mismatch is
refused there, naming the setting, instead of hanging at the first step."""


def _setting(vllm_config: Any, path: str) -> str:
    """The value at ``path`` in vLLM's configuration as text; empty when the path is absent."""
    value: Any = vllm_config
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return ",".join(sorted(str(v) for v in value))
    if hasattr(value, "name") and not isinstance(value, (str, int, float, bool)):
        return str(value.name)  # an enum: its name, stable across vLLM's reprs
    return str(value).removeprefix("torch.")


VLLM_SECTIONS = (
    "model_config",
    "cache_config",
    "parallel_config",
    "scheduler_config",
    "device_config",
    "load_config",
    "offload_config",
    "attention_config",
    "engram_config",
    "lora_config",
    "speculative_config",
    "structured_outputs_config",
    "profiler_config",
    "observability_config",
    "compilation_config",
    "kernel_config",
    "kv_transfer_config",
    "ec_transfer_config",
    "aux_output_config",
)
"""The sections vLLM folds into its configuration's hash (``VllmConfig.compute_hash``), from
which it derives the compile cache's key: each section's own hash is a step field, so a process
whose key differs from its engine's is told which section made it differ."""


def _section_hash(vllm_config: Any, name: str) -> str:
    section = getattr(vllm_config, name, None)
    if section is None or not callable(getattr(section, "compute_hash", None)):
        return ""
    try:
        return str(section.compute_hash())
    except Exception as e:  # noqa: BLE001 - a section that cannot hash says so, not nothing
        return f"unhashable: {type(e).__name__}"


def _additional_hash(vllm_config: Any) -> str:
    """``additional_config`` as vLLM hashes it; empty where there is none.

    A dict's is sha256 over its JSON with sorted keys, as ``VllmConfig.compute_hash`` takes it;
    a config object's is its own hash.
    """
    extra = getattr(vllm_config, "additional_config", None)
    if not extra:
        return ""
    if isinstance(extra, dict):
        text = json.dumps(extra, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()
    return _section_hash(vllm_config, "additional_config")


def step_config(vllm_config: Any) -> tuple[str, dict[str, str]]:
    """The digest of :data:`STEP_FIELDS` and vLLM's section hashes as this process resolved them.

    The fields are :data:`STEP_FIELDS`' readable settings, ``vllm.<section>`` for each of
    :data:`VLLM_SECTIONS`, ``vllm.additional`` for the free-form ``additional_config`` vLLM also
    hashes (under Dynamo the head's differs from its nodes' in that alone), and ``vllm.config``
    for the whole configuration's hash. The digest is sha256 over the
    sorted ``name=value`` lines; the fields name what differs.
    """
    fields = {name: _setting(vllm_config, path) for name, path in STEP_FIELDS}
    for name in VLLM_SECTIONS:
        fields[f"vllm.{name.removesuffix('_config')}"] = _section_hash(vllm_config, name)
    fields["vllm.additional"] = _additional_hash(vllm_config)
    whole = getattr(vllm_config, "compute_hash", None)
    fields["vllm.config"] = str(whole()) if callable(whole) else ""
    text = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    return hashlib.sha256(text.encode()).hexdigest(), fields


_COMMIT = re.compile(r"[0-9a-f]{40}")


def hf_snapshot(repo: str, revision: str | None, cache: Path) -> Path | None:
    """The snapshot of ``repo`` at ``revision`` in the Hugging Face cache ``cache``, as on disk.

    Read from the cache's own layout (``models--<org>--<name>/refs/<revision>`` names the commit,
    ``snapshots/<commit>`` holds the files), never through the hub's client: a process fetches
    the files it loads, and the client refuses a snapshot it holds only part of
    (``IncompleteSnapshotError``, huggingface_hub 2). None when the cache has no such snapshot.
    """
    root = cache / f"models--{repo.replace('/', '--')}"
    rev = revision or "main"
    if _COMMIT.fullmatch(rev):
        commit = rev
    else:
        ref = root / "refs" / rev
        if not ref.is_file():
            return None
        commit = ref.read_text().strip()
    snap = root / "snapshots" / commit
    return snap if snap.is_dir() else None


def _snapshot(model: Any) -> Path | None:
    path = Path(model.model)
    if path.is_dir():
        return path.resolve()
    try:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
    except ImportError:
        log.warning("no huggingface_hub here: checkpoint %s has no digest", model.model)
        return None
    snap = hf_snapshot(model.model, model.revision, cache)
    if snap is None:
        log.warning(
            "checkpoint %s at %s is not in the cache at %s: the identity has no digest",
            model.model,
            model.revision or "main",
            cache,
        )
    return snap


def _nccl_version() -> str:
    try:
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

        return NCCLLibrary().ncclGetVersion()
    except Exception:  # noqa: BLE001 - no NCCL here: the identity says nothing
        return ""


def _driver_version() -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return str(pynvml.nvmlSystemGetDriverVersion())
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - reported only
        return ""


def software_identity(
    vllm_config: Any, kv_caches: Mapping[str, Any] | None = None
) -> SoftwareIdentity:
    """This process's identity; ``kv_caches`` for a process holding a position.

    A process holding none (the head, a spare for any position before it takes one) reports the
    KV dtype and block size it would allocate, without a layout hash (PRIVATE-INTERNALS.md A44).
    """
    import torch
    import vllm

    try:
        from vllm import _version

        recorded = getattr(_version, "__commit_id__", None)
    except ImportError:
        recorded = None
    commit, dirty = source_commit()
    model = vllm_config.model_config
    snapshot = _snapshot(model)
    cache = vllm_config.cache_config
    if kv_caches:
        kv = kv_format(kv_caches, attention_backend(vllm_config))
    else:
        # the element type the cache tensors will have, named as a tensor's is, so a spare that
        # holds no position yet compares with one that does
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        dtype = kv_cache_dtype_str_to_dtype(cache.cache_dtype, model)
        kv = KvFormat(dtype=str(dtype).removeprefix("torch."), block_size=cache.block_size or 0)
    # what this process steps with, in every process's log beside its registration: two
    # processes of one engine whose lines differ compile and step differently
    digest, fields = step_config(vllm_config)
    log.info("step config %s: %s", digest[:12], " ".join(f"{k}={v}" for k, v in fields.items()))
    return SoftwareIdentity(
        engine="vllm",
        engine_commit=engine_commit(vllm.__version__, recorded),
        adapter_commit=commit,
        adapter_dirty=dirty,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        driver_version=_driver_version(),
        nccl_version=_nccl_version(),
        model=model.model,
        model_revision=model.revision or (snapshot_revision(snapshot) if snapshot else ""),
        checkpoint_digest=checkpoint_digest(snapshot) if snapshot else "",
        quantization=model.quantization or "",
        model_dtype=str(model.dtype).removeprefix("torch."),
        kv_format=kv,
        widths=widths(vllm_config.parallel_config),
        step=StepConfig(digest, fields),
        guarantee=settings.current().guarantee,
        # decided once the plugin loaded, before the engine core sets its step (serving.py)
        serves_changes=serving.installed(),
    )
