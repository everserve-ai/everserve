# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""``python -m everserve_vllm.flags``: the vLLM flags the adapter requires of another's head.

A head started by another command line (NVIDIA Dynamo's ``python -m dynamo.vllm``, decision 0041)
takes this adapter's requirements as vLLM flags. They are computed in the pod, from the vLLM
installed there (the compile settings name vLLM's own attention split points), so a deployment
never carries a copy that drifts: the head's command expands
``$(python -m everserve_vllm.flags --head)``. The values are :func:`everserve_vllm.engine_args.
required`'s, the same every other process of the engine applies, printed as one line of
arguments with JSON in compact form, so the shell's word splitting keeps each whole.
"""

from __future__ import annotations

import argparse
import json
import sys

from everserve_vllm import engine_args


def flags(guarantee: str = "exact", head: bool = False) -> list[str]:
    """The vLLM flags for ``engine_args.required``; ``head`` adds the executor."""
    from everserve.policy import Guarantee

    req = engine_args.required(Guarantee[guarantee.upper()])
    out: list[str] = []
    if head:
        # `mp`, which vLLM's pipeline-parallel check takes, resolved to Everserve's executor in
        # the engine's process (everserve_vllm.backend); the executor's name is refused there
        out += ["--distributed-executor-backend", "mp"]
    out += ["--compilation-config", json.dumps(req["compilation_config"], separators=(",", ":"))]
    out += ["--kv-transfer-config", json.dumps(req["kv_transfer_config"], separators=(",", ":"))]
    if req["disable_custom_all_reduce"]:
        out.append("--disable-custom-all-reduce")
    return out


def main(argv: list[str] | None = None) -> None:
    """Print the flags on one line."""
    from everserve_vllm import settings

    ap = argparse.ArgumentParser(prog="everserve_vllm.flags")
    ap.add_argument(
        "--guarantee",
        default=settings.current().guarantee or None,
        choices=["exact", "continuous"],
        help="the engine's guarantee class (policy Engine.guarantee); EVERSERVE_GUARANTEE unset",
    )
    ap.add_argument("--head", action="store_true", help="the head's flags: with its executor")
    a = ap.parse_args(argv)
    if a.guarantee is None:
        # never a default: flags resolved for another class than the policy's are refused later
        ap.error("name the engine's guarantee: --guarantee, or EVERSERVE_GUARANTEE")
    sys.stdout.write(" ".join(flags(a.guarantee, a.head)) + "\n")


if __name__ == "__main__":
    main()
