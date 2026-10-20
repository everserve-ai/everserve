# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The (key, digest) of every compile cache on this node, for the record (decision 0029).

A thin call of :mod:`everserve.contract.compile_cache` (the one implementation, the one the
adapter and the spare report at registration): every key under
``$VLLM_CACHE_ROOT/torch_compile_cache`` with its digest, as JSON on stdout.

    python cache_digest.py [ROOT]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from everserve.contract.compile_cache import cache_root, digest, keys


def main(argv: list[str]) -> int:
    """Print ``{"root", "caches": [{key, digest, files, bytes}, ...]}``."""
    root = argv[1] if len(argv) > 1 else str(cache_root())
    print(json.dumps({"root": root, "caches": [asdict(digest(root, k)) for k in keys(root)]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
