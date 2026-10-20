# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Whether the peer-to-peer slot collective actually works on these GPUs, found by running it.

NVML reports what the driver and the topology allow; it cannot see whether a container's PCIe or
IOMMU settings let one GPU's stores reach another's memory. On PCIe pods (A40, L40S) it reports
peer reads that NCCL's own peer-to-peer cannot use, and an engine that trusted it hangs at its
first collective. :func:`probe` therefore runs the thing itself, small: one process per
GPU, as the engine's workers are, each joining a peer-to-peer slot region and reducing a known
tensor once. The result is accepted only if every process returns the exact sum before the
deadline; a hang, an error or a wrong sum means the peer-to-peer variant must not be used.

The probe's processes start their own CUDA contexts and exit; the calling process starts none,
so the executor can probe before vLLM starts its workers. Nothing here imports torch at module
level for the same reason.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ELEMENTS = 2048
"""Elements each member reduces: one decode-size call."""


@dataclass(frozen=True)
class ProbeResult:
    """What the probe found: whether every member got the exact sum, and if not, why."""

    ok: bool
    detail: str
    seconds: float


def _member(path: str, row: int, n: int, visible: str, timeout_s: float, out: mp.Queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible  # before torch starts CUDA in this process
    try:
        import torch

        from everserve.comm.slotkernel import Region, SlotKernel

        torch.cuda.set_device(row)
        with torch.inference_mode():
            k = SlotKernel(Region(path), row, row, open_timeout_s=timeout_s)
            k.set_members(range(n), 0)
            x = torch.full((ELEMENTS,), float(row + 1), dtype=torch.bfloat16, device="cuda")
            y = torch.empty_like(x)
            k.all_reduce_into(x, y)
            torch.cuda.synchronize()
            want = float(n * (n + 1) // 2)
            exact = bool(torch.all(y == want).item())
        out.put((row, "exact" if exact else f"wrong sum: got {y[0].item()} want {want}"))
        k.close()
    except Exception as e:  # noqa: BLE001 - reported to the parent, which decides
        out.put((row, f"{type(e).__name__}: {e}"))


def probe(devices: Sequence[int | str], timeout_s: float = 15.0) -> ProbeResult:
    """Run one peer-to-peer slot all-reduce over ``devices`` (``CUDA_VISIBLE_DEVICES`` entries).

    One spawned process per device, as the engine's workers; each sees all of ``devices`` and
    uses its own. ``ok`` only if every process reports the exact sum within ``timeout_s``: a
    working probe takes a couple of seconds, so a known-bad box pays at most the deadline.
    """
    from everserve.comm.slotkernel import Region

    t0 = time.time()
    n = len(devices)
    if n < 2:
        return ProbeResult(True, "one device: nothing to reach", 0.0)
    shm = Path("/dev/shm")
    root = Path(tempfile.mkdtemp(dir=shm if shm.is_dir() else None))
    path = str(root / "probe")
    region = Region(path, create=True, variant="p2p")
    region.set_members(range(n), range(n))
    visible = ",".join(str(d) for d in devices)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [
        ctx.Process(target=_member, args=(path, r, n, visible, timeout_s, out), daemon=True)
        for r in range(n)
    ]
    for p in procs:
        p.start()
    got: dict[int, str] = {}
    deadline = t0 + timeout_s
    try:
        while len(got) < n:
            left = deadline - time.time()
            if left <= 0:
                break
            try:
                row, what = out.get(timeout=left)
            except queue.Empty:
                break
            got[row] = what
    finally:
        for p in procs:
            p.join(max(0.0, min(5.0, deadline - time.time())))
            if p.is_alive():
                p.kill()
                p.join(5)
        region.close()
        try:
            os.unlink(path)
            root.rmdir()
        except OSError:
            pass
    seconds = round(time.time() - t0, 2)
    missing = [r for r in range(n) if r not in got]
    if missing:
        return ProbeResult(
            False,
            f"no answer from devices {[devices[r] for r in missing]} in {timeout_s}s",
            seconds,
        )
    bad = {devices[r]: what for r, what in got.items() if what != "exact"}
    if bad:
        return ProbeResult(False, f"failed on {bad}", seconds)
    return ProbeResult(True, f"exact sum on all {n} devices", seconds)
