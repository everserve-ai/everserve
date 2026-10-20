# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Every file written on this node since a mark, by directory, for the record.

A start's warm-up that a staged cache does not cover (13 s inside vLLM's ``warmup_kernels`` on
a node that never ran it, with the compile cache and Triton's cache staged) is named by what it
leaves on disk: a mark before the start,
this read after it. An empty read says the cost is held in the process (an autotune, a lazy
load) and no staging covers it.

Walks the root file system only (``/``'s device: tmpfs, ``/proc`` and ``/sys`` are other
devices), skipping the excluded prefixes, and counts each regular file modified after the
mark's own modification time. JSON on stdout:

    {"mark", "mark_at", "read_at", "start", "takes", "exit", "excluded",
     "files", "bytes", "roots": {root: {files, bytes}}, "dirs": [{dir, files, bytes}, ...],
     "named": [{path, bytes}, ...], "autotune": [{path, bytes}, ...]}

``roots`` groups by the first ``--depth`` components (``/home/everserve/.triton/cache``);
``dirs`` lists the directories holding the files, the largest ``--top`` by bytes; ``named`` the
files themselves, the largest ``--top`` by bytes (a directory total alone cannot name the one
file of 55,832 bytes a take wrote in a compile cache directory); ``autotune`` every file
whose path marks it an autotune result (:data:`AUTOTUNE`), however many: what the compile cache
prediction (decision 0029) says a trial never writes.

    python written_since.py MARK [--exclude PREFIX ...] [--start N] [--takes N] [--exit KIND]
                                 [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict


def walk(mark_at: float, excluded: list[str], root: str = "/") -> list[tuple[str, int]]:
    """``(path, size)`` of every regular file on ``root``'s device modified after ``mark_at``."""
    dev = os.stat(root).st_dev
    out: list[tuple[str, int]] = []
    for top, dirs, files in os.walk(root, onerror=lambda e: None):
        keep = []
        for d in dirs:
            p = os.path.join(top, d)
            if any(p == x or p.startswith(x.rstrip("/") + "/") for x in excluded):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if st.st_dev == dev and not os.path.islink(p):
                keep.append(d)
        dirs[:] = keep
        for f in files:
            p = os.path.join(top, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            regular = os.path.isfile(p) and not os.path.islink(p)
            if st.st_dev == dev and regular and st.st_mtime > mark_at:
                out.append((p, st.st_size))
    return out


AUTOTUNE = re.compile(r"autotun|\.best_config$")
"""An autotune result's path: FlashInfer's autotune cache (``flashinfer_autotune_cache``),
Triton's (``*.autotune.json``), inductor's chosen configs (``*.best_config``)."""


def summarise(found: list[tuple[str, int]], depth: int, top: int) -> dict:
    """Totals, the roots at ``depth`` components and the ``top`` directories by bytes."""
    roots: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    dirs: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p, size in found:
        parts = p.strip("/").split("/")
        root = "/" + "/".join(parts[: min(depth, len(parts) - 1)])
        for acc in (roots[root], dirs[os.path.dirname(p)]):
            acc[0] += 1
            acc[1] += size
    ranked = sorted(dirs.items(), key=lambda kv: -kv[1][1])[:top]
    largest = sorted(found, key=lambda f: -f[1])[:top]
    return {
        "files": len(found),
        "bytes": sum(s for _, s in found),
        "roots": {r: {"files": n, "bytes": b} for r, (n, b) in sorted(roots.items())},
        "dirs": [{"dir": d, "files": n, "bytes": b} for d, (n, b) in ranked],
        "named": [{"path": f, "bytes": b} for f, b in largest],
        "autotune": [{"path": f, "bytes": b} for f, b in sorted(found) if AUTOTUNE.search(f)],
    }


def main(argv: list[str]) -> int:
    """Print the read of the files written since the mark."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mark", help="the mark file, touched before the start")
    ap.add_argument("--exclude", nargs="*", default=[], help="path prefixes not walked")
    ap.add_argument("--start", type=int, help="the start's index on this node, for the record")
    ap.add_argument("--takes", type=int, help="takes on this node so far, this start's included")
    ap.add_argument("--exit", help="how its processes ended, once stopped: clean, killed, none")
    ap.add_argument("--root", default="/", help="where the walk starts (its device only)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv[1:])
    mark_at = os.stat(a.mark).st_mtime
    excluded = [os.path.abspath(x) for x in a.exclude]
    found = walk(mark_at, excluded, a.root)
    print(
        json.dumps(
            {
                "mark": a.mark,
                "mark_at": mark_at,
                "read_at": time.time(),
                "start": a.start,
                "takes": a.takes,
                "exit": a.exit,
                "excluded": excluded,
                **summarise(found, a.depth, a.top),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
