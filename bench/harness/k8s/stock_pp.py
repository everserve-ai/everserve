# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Stock vLLM, pipeline parallel across two pods: the control for the Dynamo proof's engine.

The engine image's vLLM with nothing of Everserve's configured (no ``EVERSERVE_*`` in the
environment: the plugin then changes no executor, collective or pipeline), two pods of one GPU
each running ``vllm serve`` as node 0 and node 1 of a ``--pipeline-parallel-size 2`` engine over
the ``mp`` executor, and a headless Service they rendezvous on. Whether it serves tells an
Everserve hang from vLLM's or the cluster's own. Prints a v1 List for ``kubectl apply -f -``.

    stock_pp.py --image <registry>/everserve-engine:<tag> --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json

NAME = "stock-pp"
PORT = 29500


PICK = {"compile": "--compilation-config", "kv": "--kv-transfer-config"}


def command(args: list[str], ours: list[str]) -> dict:
    """``vllm serve`` with ``args``, plus the adapter's flags ``ours`` names, computed in the pod.

    The flags (``everserve_vllm.flags``) are the image's vLLM's, so the pod computes them;
    ``compile`` and ``kv`` take the flag and its value, ``allreduce`` the bare switch.
    """
    if not ours:
        return {"command": ["vllm", "serve"], "args": args}
    want = [PICK[o] for o in ours if o in PICK]
    bare = ["--disable-custom-all-reduce"] if "allreduce" in ours else []
    pick = (
        "from everserve_vllm.flags import flags; f = flags('continuous'); "
        f"want = {want!r}; "
        "print(' '.join(x for i, x in enumerate(f) if x in want or (i and f[i - 1] in want)))"
    )
    script = f'exec vllm serve "$@" $(python3 -c "{pick}") {" ".join(bare)}'
    return {"command": ["/bin/sh", "-c", script, "vllm"], "args": args}


def pod(i: int, a: argparse.Namespace) -> dict:
    """Node ``i`` of the engine: node 0 serves the API, node 1 runs headless."""
    args = [
        "--model", a.model,
        "--pipeline-parallel-size", "2",
        "--nnodes", "2",
        "--node-rank", str(i),
        "--master-addr", f"{NAME}.{a.namespace}.svc",
        "--master-port", str(PORT),
        "--distributed-executor-backend", "mp",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.8",
    ]  # fmt: skip
    args += a.extra
    if i:
        args.append("--headless")
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"{NAME}-{i}",
            "namespace": a.namespace,
            "labels": {"app": NAME, "node-rank": str(i)},
        },
        "spec": {
            "restartPolicy": "Never",
            "imagePullSecrets": [{"name": s} for s in a.pull_secrets],
            # one GPU pod per node, as the proof's engine has
            "affinity": {
                "podAntiAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "labelSelector": {"matchLabels": {"app": NAME}},
                            "topologyKey": "kubernetes.io/hostname",
                        }
                    ]
                }
            },
            "containers": [
                {
                    "name": "main",
                    "image": a.image,
                    **command(args, a.ours),
                    "env": [
                        {
                            "name": "VLLM_HOST_IP",
                            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
                        },
                        *({"name": k, "value": v} for k, v in (e.split("=", 1) for e in a.env)),
                    ],
                    "resources": {"limits": {"nvidia.com/gpu": "1"}},
                    "volumeMounts": [{"name": "shm", "mountPath": "/dev/shm"}],
                }
            ],
            "volumes": [{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}],
        },
    }


def main() -> None:
    """Print the List."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--namespace", default="stock")
    ap.add_argument("--pull-secrets", nargs="*", default=["nebius-cr"])
    ap.add_argument("--env", nargs="*", default=[], help="NAME=VALUE for both pods: a variant")
    ap.add_argument(
        "--ours",
        nargs="*",
        default=[],
        choices=["compile", "kv", "allreduce"],
        help="the adapter's flags to add: a variant",
    )
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="vLLM flags: a variant")
    a = ap.parse_args()
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": NAME, "namespace": a.namespace},
        "spec": {
            "clusterIP": "None",
            "publishNotReadyAddresses": True,
            "selector": {"app": NAME, "node-rank": "0"},
            "ports": [{"name": "rendezvous", "port": PORT}],
        },
    }
    items = [service, pod(0, a), pod(1, a)]
    print(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=1))


if __name__ == "__main__":
    main()
