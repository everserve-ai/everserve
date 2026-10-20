# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""An engine's compile cache as a (key, digest) pair: what a member actually loaded (decision 0029).

vLLM compiles the model's backbone once per engine and keeps the result under
``$VLLM_CACHE_ROOT/torch_compile_cache``: the key's directory (``<key>/rank_<r>_<x>/backbone``,
the key over the code, compiler, configuration and a fixed environment list) and the AOT
artifact it references under ``torch_aot_compile/<artifact>``, Inductor's autotuned
``best_config`` entries inside. Two nodes with the same key can hold different autotune
results, chosen by timing on each machine; a spare that compiled on its own node then runs the
same fused kernels with other tiles and rounds differently from the rank it replaces, from the
resumed position on: two launches of the serving engine match at every position, and a spare on
its own kernels does not.

Exactness needs every member of an engine, spares included, on the same kernels, so each member
reports the pair it loaded and the coordinator holds the engine's members to the first one's.
:func:`digest` is that pair's digest: the sorted (relative path, sha256) of every file under the
key's directory and under every artifact it references, hashed; equal digests, equal kernels.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_ARTIFACT = re.compile(rb"\b([0-9a-f]{64})\b")
_CHUNK = 1 << 20


@dataclass(frozen=True)
class CompileCache:
    """The cache a member loaded: vLLM's key and the digest of its files."""

    key: str
    digest: str
    files: int = 0
    bytes: int = 0


def cache_root() -> Path:
    """The compile cache directory on this host (``$VLLM_CACHE_ROOT/torch_compile_cache``)."""
    root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    return Path(root) / "torch_compile_cache"


def _file_hash(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _files(base: Path, label: str) -> list[tuple[str, Path]]:
    """Every file under ``base`` as (label/relative path, path), for a stable listing."""
    return sorted(
        (f"{label}/{p.relative_to(base).as_posix()}", p) for p in base.rglob("*") if p.is_file()
    )


def referenced_artifacts(key_dir: Path, artifacts_dir: Path) -> list[Path]:
    """The AOT artifacts under ``artifacts_dir`` that files under ``key_dir`` name by hash.

    vLLM records the artifact a compiled graph loads by its 64-hex hash in the key directory's
    metadata; an artifact nothing names is not part of this key.
    """
    if not artifacts_dir.is_dir():
        return []
    present = {p.name: p for p in artifacts_dir.iterdir() if p.is_dir()}
    named: set[str] = set()
    for p in key_dir.rglob("*"):
        if not p.is_file() or p.stat().st_size > 8 * _CHUNK:
            continue
        for m in _ARTIFACT.finditer(p.read_bytes()):
            name = m.group(1).decode()
            if name in present:
                named.add(name)
    return [present[n] for n in sorted(named)]


def digest(root: Path | str, key: str, artifact: Path | str | None = None) -> CompileCache:
    """The (key, digest) of the compile cache ``key`` under ``root``.

    Over every file under ``root/<key>`` and under the AOT artifact, as sorted (relative path,
    sha256) pairs hashed together. The artifact is ``artifact`` when the caller names it (a
    member reports the one it loaded: vLLM keeps its directory on
    ``compilation_config.local_cache_dir``, derived from the configuration and the model's
    code at load, not written beside the key); else the artifacts the key's files name by
    hash; else, since nothing on disk links a key to its artifact, every artifact under
    ``root/torch_aot_compile`` (a node's whole cache, which is what a deploy stages).

    Raises:
        FileNotFoundError: no such key under ``root``.
    """
    root = Path(root)
    key_dir = root / key
    if not key_dir.is_dir():
        raise FileNotFoundError(f"no compile cache {key!r} under {root}")
    listing = _files(key_dir, key)
    artifacts_dir = root / "torch_aot_compile"
    if artifact is not None:
        artifacts = [Path(artifact)] if Path(artifact).is_dir() else []
    else:
        artifacts = referenced_artifacts(key_dir, artifacts_dir) or (
            sorted(p for p in artifacts_dir.iterdir() if p.is_dir())
            if artifacts_dir.is_dir()
            else []
        )
    for artifact_dir in artifacts:
        listing += _files(artifact_dir, f"torch_aot_compile/{artifact_dir.name}")
    whole = hashlib.sha256()
    total = 0
    for rel, path in sorted(listing):
        h, n = _file_hash(path)
        total += n
        whole.update(rel.encode())
        whole.update(b"\0")
        whole.update(h.encode())
        whole.update(b"\n")
    return CompileCache(key=key, digest=whole.hexdigest(), files=len(listing), bytes=total)


def artifact_digest(artifact: Path | str) -> CompileCache:
    """The pair of an AOT artifact alone: its hash as the key, the digest over its files.

    On vLLM's AOT path the process loads ``torch_aot_compile/<hash>`` directly and keeps only
    that directory on its compilation config (``local_cache_dir``); the key's directory is
    never named. The artifact's hash is then the identity a process runs with, and its files
    (Inductor's autotune results included) are what the digest must cover.

    Raises:
        FileNotFoundError: no such directory.
    """
    artifact = Path(artifact)
    if not artifact.is_dir():
        raise FileNotFoundError(f"no compile artifact at {artifact}")
    whole = hashlib.sha256()
    total = 0
    listing = _files(artifact, artifact.name)
    for rel, path in listing:
        h, n = _file_hash(path)
        total += n
        whole.update(rel.encode())
        whole.update(b"\0")
        whole.update(h.encode())
        whole.update(b"\n")
    return CompileCache(
        key=artifact.name, digest=whole.hexdigest(), files=len(listing), bytes=total
    )


_TOKEN = re.compile(rb"\b(c[a-z0-9]{51}|[0-9a-f]{64})\b")


def rank_manifest(artifact: Path | str, rank: int, dp_rank: int = 0) -> RankManifest:
    """One rank's share of an AOT artifact, file by file: its subtree and the kernels it names.

    vLLM's artifact holds one subtree per rank (``rank_<r>_<dp>/model``, the serialized
    compiled graph) beside one ``inductor_cache`` every rank on the node writes into. The
    graph names the kernels it runs by their Inductor names (``c`` and 51 characters) and
    their autotune and AOT-autograd entries by 64-hex hashes, so a rank's digest is over its
    subtree, every kernel source (``.py``) under ``inductor_cache`` whose path carries a name
    the subtree names, and the ``best_config`` files beside those kernels (the autotune
    results, keyed by a hash the graph does not name). What a launch builds from those
    (Triton's binaries and runtime helpers, the serialized graphs) is left out: a serving rank's
    launch builds a ``cuda_utils`` shared object after the staging, and a digest over it would
    put the spare's off by that one file. A node holding several ranks'
    subtrees then reports, for rank ``r``, the digest rank ``r`` reports on a node holding only
    its own, and a spare's pair can equal its rank's; nothing of another rank's kernels enters
    it.

    Raises:
        FileNotFoundError: no such artifact, or no subtree for the rank in it.
    """
    artifact = Path(artifact)
    subtree = artifact / f"rank_{rank}_{dp_rank}"
    if not subtree.is_dir():
        raise FileNotFoundError(f"no subtree for rank {rank} in {artifact}")
    named: set[bytes] = set()
    for p in subtree.rglob("*"):
        if p.is_file():
            named.update(m.group(1) for m in _TOKEN.finditer(p.read_bytes()))
    listing = _files(subtree, subtree.name)
    kernels = artifact / "inductor_cache"
    if kernels.is_dir():
        chosen: list[Path] = []
        dirs: set[Path] = set()
        for p in kernels.rglob("*"):
            # the kernel sources and the autotune results decide the numerics; what a launch
            # builds from them (Triton's binaries, its runtime helpers, the serialized graphs
            # keyed by hash) is excluded, since a launch may rebuild one and change nothing
            if p.suffix not in (".py", ".best_config"):
                continue
            if p.is_file() and any(n in p.as_posix().encode() for n in named):
                chosen.append(p)
                dirs.add(p.parent)
        for d in dirs:
            chosen += [p for p in d.iterdir() if p.is_file() and p.suffix == ".best_config"]
        seen: set[Path] = set()
        for p in chosen:
            if p not in seen:
                seen.add(p)
                listing.append((f"inductor_cache/{p.relative_to(kernels).as_posix()}", p))
    entries = []
    for rel, path in sorted(set(listing)):
        h, n = _file_hash(path)
        entries.append(ManifestFile(rel, n, h))
    return RankManifest(artifact.name, rank, dp_rank, _digest_of(entries), tuple(entries))


def _digest_of(entries: Sequence[ManifestFile]) -> str:
    """The rank's digest: sha256 over each file's relative path and sha256, in path order."""
    whole = hashlib.sha256()
    for f in sorted(entries, key=lambda f: f.path):
        whole.update(f.path.encode())
        whole.update(b"\0")
        whole.update(f.sha256.encode())
        whole.update(b"\n")
    return whole.hexdigest()


@dataclass(frozen=True)
class ManifestFile:
    """One file of a rank's set: its path relative to the artifact, its size and its sha256."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RankManifest:
    """A rank's share of an AOT artifact, file by file (decision 0043).

    Written beside the artifact as ``rank_<r>_<dp>.manifest.json`` by the process that serves the
    rank, once it computed its pair, and read by the node's store to serve exactly these files to
    another node's store for the rank's spare. ``digest`` is :func:`rank_digest`'s: a function of
    ``files`` alone, so the set moved is the set the digest names.
    """

    key: str
    rank: int
    dp: int
    digest: str
    files: tuple[ManifestFile, ...]

    @property
    def pair(self) -> CompileCache:
        """The (key, digest) pair, as a member reports it."""
        return CompileCache(
            self.key, self.digest, files=len(self.files), bytes=sum(f.size for f in self.files)
        )

    def as_json(self) -> dict:
        """The manifest's document, in the format the store reads."""
        return {
            "key": self.key,
            "rank": self.rank,
            "dp": self.dp,
            "digest": self.digest,
            "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in self.files],
        }


def manifest_path(artifact: Path | str, rank: int, dp_rank: int = 0) -> Path:
    """Where rank ``rank``'s manifest lives: beside its subtree, under the artifact."""
    return Path(artifact) / f"rank_{rank}_{dp_rank}.manifest.json"


def share(artifact: Path | str, m: RankManifest) -> None:
    """Make every file ``m`` lists readable by the node's group, and the directories to them.

    vLLM writes its compiled artifact for its own process alone (a temporary file renamed into
    place, mode 0600), and the node's store, another user of the node's group, serves the set to
    a spare's node from those files: left at 0600, the store reads "Permission denied". The
    process that wrote them shares them as it publishes the manifest; it owns them, so it may.
    """
    art = Path(artifact)
    dirs: set[Path] = set()
    for f in m.files:
        path = art / f.path
        os.chmod(path, os.stat(path).st_mode | stat.S_IRGRP)
        parent = path.parent
        while parent != art and art in parent.parents:
            dirs.add(parent)
            parent = parent.parent
    for d in {art, *dirs}:
        os.chmod(d, os.stat(d).st_mode | stat.S_IRGRP | stat.S_IXGRP)


def write_manifest(artifact: Path | str, m: RankManifest) -> Path:
    """Write ``m`` beside its rank's subtree, whole or not at all, for the node's group.

    A temporary file is renamed into place, so a reader sees the previous manifest or this one,
    never part of one; it is the node group's to read (mode 0660, decision 0043).
    """
    path = manifest_path(artifact, m.rank, m.dp)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660)
    try:
        os.fchmod(fd, 0o660)
        os.write(fd, json.dumps(m.as_json(), indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    return path


def rank_digest(artifact: Path | str, rank: int, dp_rank: int = 0) -> CompileCache:
    """The pair of one rank's share of an AOT artifact (:func:`rank_manifest`'s).

    Raises:
        FileNotFoundError: no such artifact, or no subtree for the rank in it.
    """
    return rank_manifest(artifact, rank, dp_rank).pair


def keys(root: Path | str) -> list[str]:
    """The cache keys under ``root`` (every directory but the artifacts')."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "torch_aot_compile")
