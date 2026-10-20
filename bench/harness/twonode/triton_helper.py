# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Build Triton's driver helper in every staged inductor Triton cache, in place on this node.

Staging a serving node's compile cache onto the spare's node (``nodes.sh cache_sync``) replaces
the spare's own, and Triton's compiled driver helper (``cuda_utils``, built against this node's
driver and Python) lives in the inductor Triton directories inside it. When the serving node's
copy lacks it, the take rebuilds it on its path (55,832 bytes written in the compile cache), and
a cold take that writes any byte stops the smoke. This builds it after the staging, off the
take's path: one process per directory, since
Triton reads ``TRITON_CACHE_DIR`` once, each initialising the driver as a take would. JSON on
stdout, per directory the files it gained; exit 1 if a directory's driver could not initialise.

    python triton_helper.py ROOT        # the node's torch_compile_cache
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

INIT = "from triton.runtime.driver import driver; driver.active.utils"
"""What a take's first kernel does to the driver: load its helper, building it if it is absent."""


def triton_dirs(root: Path) -> list[Path]:
    """Every inductor Triton cache directory (``.../inductor_cache/triton/<device>``) under root."""
    return sorted(
        p for p in root.glob("**/inductor_cache/triton/*") if p.is_dir() and p.name.isdigit()
    )


def files(d: Path) -> set[str]:
    """The files under ``d``, relative to it."""
    return {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()}


def build(d: Path, python: str = sys.executable) -> dict:
    """Initialise Triton's driver with its cache at ``d``; what ``d`` gained, and whether it ran."""
    before = files(d)
    env = {**os.environ, "TRITON_CACHE_DIR": str(d)}
    r = subprocess.run([python, "-c", INIT], env=env, capture_output=True, text=True, timeout=300)
    gained = sorted(files(d) - before)
    return {"dir": str(d), "ok": r.returncode == 0, "gained": gained, "error": r.stderr[-500:]}


def main(argv: list[str]) -> int:
    """Build in every directory under the root; the record on stdout."""
    root = Path(argv[1])
    out = [build(d) for d in triton_dirs(root)]
    print(json.dumps({"root": str(root), "dirs": out}))
    return 0 if all(d["ok"] for d in out) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
