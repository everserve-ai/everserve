# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A managed-spare policy for a drill shape, written for the nodes' paths.

The checked-in policies carry one model path and one shape each; a battery runs the same
shape with the run's model and a smoke's shape with a small one. This writes the policy from
the shape: the serving members in rank order across the serving nodes (``--members a:4 b:4``
is ranks 0-3 on node-a's devices 0-3 and 4-7 on node-b's), the spares on the spare node for
the last ranks, in order across the spare nodes (``--spares c:4``: node-c's devices 0-3 stand
in for ranks 4-7; ``--spares d:1 e:1``: node-d's for rank 1 and node-e's for rank 2 of three),
the launch the daemon runs for each spare, and the layout.

    policy.py --tp 4 --pp 2 --members a:4 b:4 --spares c:4 --model /workspace/models/M
        --max-model-len 4096 --blocks 4000 --max-num-seqs 8 --out policy.json
"""

import argparse
import json
import sys


def engine_argv(a: argparse.Namespace) -> list[str]:
    """The engine's arguments in vLLM's spelling.

    The head's (adapter_drill.py) and nodes.sh's engine_args alike: every process of the engine
    builds one configuration from them (everserve_vllm.engine_args).
    """
    return [
        "--model", a.model,
        "--tensor-parallel-size", str(a.tp),
        "--pipeline-parallel-size", str(a.pp),
        "--dtype", "bfloat16",
        "--max-model-len", str(a.max_model_len),
        "--max-num-seqs", str(a.max_num_seqs),
        "--max-num-batched-tokens", str(a.max_num_seqs * a.prompt_len),
        "--enable-prefix-caching",
        "--seed", "0",
        "--num-gpu-blocks-override", str(a.blocks),
        "--gpu-memory-utilization", str(a.gpu_util),
    ]  # fmt: skip


def policy(a: argparse.Namespace) -> dict:
    """The policy document for the shape in ``a``."""
    members = []
    for spec in a.members:
        node, count = spec.split(":")
        members += [{"node": f"node-{node}", "device": d} for d in range(int(count))]
    world = len(members)
    if world != a.tp * a.pp:
        raise SystemExit(f"{world} members for tp {a.tp} x pp {a.pp}")
    spares = []
    total = sum(int(spec.split(":")[1]) for spec in a.spares)
    for spec in a.spares:
        node, count = spec.split(":")
        n = int(count)
        if a.resident:
            # each stands for any position of the engine and takes one at a change (0030)
            spares += [{"node": f"node-{node}", "device": d} for d in range(n)]
        else:
            # the spares stand in for the last ranks, in the order the spare nodes are named
            first = world - total + len(spares)
            spares += [{"node": f"node-{node}", "device": d, "rank": first + d} for d in range(n)]
    policy_path = f"/workspace/everserve/{a.policy_path}"
    launch = [
        "/usr/bin/env",
        "PYTHONPATH=/workspace/everserve/bench/harness:/workspace/everserve/bench/harness/sitehooks",
        "/workspace/everserve/.venv/bin/python",
        "-m",
        "everserve_vllm.spare_main",
        "--engine",
        "{engine}",
        "--rank",
        "{rank}",
        "--node",
        "{node}",
        "--policy",
        policy_path,
        "--discovery-dir",
        "/workspace/disc",
        *engine_argv(a),
    ]
    if a.resident:
        # the spare holds the whole model in pinned host memory and builds the rank's shard from
        # it at the attachment (decisions 0028 and 0030); the shard's copy is in its record
        launch.append("--resident")
    engine: dict = {
        "name": a.name,
        "members": members,
        "tensorParallel": a.tp,
        "expertParallel": 1,
        "pipelineParallel": a.pp,
        "placementIndependentModel": False,
        "spares": spares,
        "releaseBoundMs": a.release_bound_ms,
    }
    if not a.resident:
        # the preloaded shape restores from the spare's own node, so that node's store takes the
        # rows; absent, they go to the rank's siblings (decision 0030), where a resident spare
        # reads them through the coordinator's one-shot replication at the take
        engine["replicaNodes"] = sorted({s["node"] for s in spares})
    engine["spareLaunch"] = {
        "command": launch,
        "env": {
            "CUDA_VISIBLE_DEVICES": "{device}",
            "EVERSERVE_GRAPH_SAFE": "1",
            "EVERSERVE_LAYOUT": f"{a.tp},{a.pp}",
            "EVERSERVE_EVENT_DIR": "/workspace/out/events_spare",
            "PYTHONUNBUFFERED": "1",
        },
    }
    return {
        "version": f"{a.name}-tp{a.tp}pp{a.pp}-managed",
        "outcomesInOrder": ["OUTCOME_REPLACE", "OUTCOME_REBUILD"],
        "spares": {
            "perEngine": len(spares),
            "loading": "LOADING_PRELOADED",
            "allowMismatched": False,
            "maxDegradePercent": 0,
        },
        # as many ranks as the spares can take at once, unless the run sets fewer: one node's
        # worth sequences the drains of several nodes (decision 0036)
        "maxConcurrentLosses": a.max_concurrent_losses or max(1, len(spares)),
        "noticeMinMs": 2000,
        "engines": [engine],
        # the coordinator plans the groups every process pre-forms (decision 0035)
        **({"cover": {"enabled": True}} if a.cover else {}),
    }


def main(argv: list[str]) -> int:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--name", default="rig")
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--pp", type=int, required=True)
    ap.add_argument("--members", nargs="+", required=True, help="node:count in rank order")
    ap.add_argument("--spares", nargs="*", default=[], help="node:count for the last ranks")
    ap.add_argument("--model", required=True, help="the model directory on the nodes")
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, required=True)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--release-bound-ms", type=int, default=10000)
    ap.add_argument(
        "--resident",
        action="store_true",
        help="the spares hold the whole model in host memory and load the shard at the take",
    )
    ap.add_argument(
        "--cover",
        action="store_true",
        help="the coordinator plans every pre-formed group (decision 0035), not the policy",
    )
    ap.add_argument(
        "--max-concurrent-losses",
        type=int,
        default=0,
        help="ranks that may change at once; 0 (the default) is as many as there are spares",
    )
    ap.add_argument("--policy-path", required=True, help="the policy's path under the repo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv[1:])
    with open(a.out, "w") as f:
        json.dump(policy(a), f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
