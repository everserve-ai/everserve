# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The time to copy one position's weight shard from this node's host memory to a GPU.

The one term a host-resident spare adds over a warm one: the
spare node holds the whole model in host memory and loads the position's shard on join. This
measures that copy as bytes: the model's safetensors files, read once so the page cache holds
them, then one tensor-parallel share of their bytes (``1 / tp`` of each file) copied host to
device ``n`` times, the median and worst reported. It is the copy, not vLLM's loader: the
loader's own work on top of it (sharding, dtype, the module tree) is what the arm on the real
spare will add.

    shard_copy.py MODEL_DIR --tp 4 --n 5   -> one JSON line
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time

import torch


def warm_page_cache(paths: list[str]) -> int:
    """Read every file once so the copies below come from host memory; returns the bytes."""
    total = 0
    for p in paths:
        with open(p, "rb") as f:
            while chunk := f.read(1 << 24):
                total += len(chunk)
    return total


def copy_shard(paths: list[str], tp: int, device: torch.device) -> tuple[float, int]:
    """Copy ``1 / tp`` of each file's bytes host to device; returns (seconds, bytes)."""
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    moved = 0
    for p in paths:
        size = os.path.getsize(p)
        share = size // tp
        with open(p, "rb") as f:
            buf = f.read(share)
        host = torch.frombuffer(bytearray(buf), dtype=torch.uint8).pin_memory()
        dev = host.to(device, non_blocking=True)
        moved += dev.numel()
        del dev
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0, moved


def main(argv: list[str]) -> int:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("model_dir")
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args(argv[1:])
    paths = sorted(glob.glob(os.path.join(a.model_dir, "*.safetensors")))
    if not paths:
        print(json.dumps({"error": f"no safetensors under {a.model_dir}"}))
        return 1
    device = torch.device("cuda", 0)
    total = warm_page_cache(paths)
    took = []
    moved = 0
    for _ in range(a.n):
        s, moved = copy_shard(paths, a.tp, device)
        took.append(s)
    out = {
        "model_dir": a.model_dir,
        "files": len(paths),
        "model_bytes": total,
        "shard_bytes": moved,
        "tp": a.tp,
        "n": a.n,
        "copy_s": [round(t, 4) for t in took],
        "median_s": round(statistics.median(took), 4),
        "worst_s": round(max(took), 4),
        "gb_s_median": round(moved / statistics.median(took) / 1e9, 2),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
