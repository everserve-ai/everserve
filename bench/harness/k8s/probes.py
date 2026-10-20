# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The Kubernetes test's probe pods, as a v1 List for ``kubectl apply -f -``.

Two StatefulSets under the pod contract (``everserve.podcontract``), as a rendered instance's pods
are: ``probe-head`` asks for node 0's ranks, as the head does, and ``probe-nodes`` stands for a
framework's group, each pod asking by its ordinal (``apps.kubernetes.io/pod-index``) as
``node_main --slot`` does. Every rank of the world is then asked for, so the coordinator places the
engine and pushes its placement.

    probes.py --image <registry>/everserve-probe:<tag> --engine k --world 4 --nnodes 2 --nodes 3
"""

from __future__ import annotations

import argparse
import json

from everserve import podcontract

PROBE = ["python", "/probe/stage_probe.py"]
POD_INDEX_ENV = "POD_INDEX"
UID = 1000


def statefulset(name: str, image: str, replicas: int, args: list[str], namespace: str) -> dict:
    """One StatefulSet of probes; every pod gets its ordinal as ``$(POD_INDEX)``."""
    env = [
        *podcontract.env(),
        # the class the test's policy states for its engine (none named: exact), as a renderer
        # states its instance's for every engine pod
        {"name": "EVERSERVE_GUARANTEE", "value": "exact"},
        {
            "name": POD_INDEX_ENV,
            "valueFrom": {
                "fieldRef": {"fieldPath": "metadata.labels['apps.kubernetes.io/pod-index']"}
            },
        },
    ]
    labels = {"app.kubernetes.io/name": name, "app.kubernetes.io/part-of": "everserve-e2e"}
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "replicas": replicas,
            "serviceName": name,
            "podManagementPolicy": "Parallel",  # every pod asks at once, as a gang starts
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "securityContext": {**podcontract.security_context(), "runAsUser": UID},
                    "terminationGracePeriodSeconds": 5,
                    "containers": [
                        {
                            "name": "probe",
                            "image": image,
                            "command": [*PROBE, *args],
                            "env": env,
                            "volumeMounts": podcontract.mounts(),
                            "resources": {"requests": {"cpu": "100m", "memory": "512Mi"}},
                        }
                    ],
                    "volumes": podcontract.volumes(),
                },
            },
        },
    }


def main() -> None:
    """Print the List."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--engine", default="k")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--nnodes", type=int, default=2, help="the nodes the engine spans")
    ap.add_argument("--nodes", type=int, default=3, help="probe-nodes' pods: the group's size")
    ap.add_argument("--namespace", default="everserve-e2e")
    a = ap.parse_args()
    common = ["--engine", a.engine, "--world", str(a.world), "--nnodes", str(a.nnodes)]
    items = [
        statefulset("probe-head", a.image, 1, [*common, "--node-rank", "0"], a.namespace),
        statefulset(
            "probe-nodes", a.image, a.nodes, [*common, "--slot", f"$({POD_INDEX_ENV})"], a.namespace
        ),
        # none until the test scales it: a pod that steps otherwise than its engine, which the
        # step guard refuses before it is told anything
        statefulset(
            "probe-odd",
            a.image,
            0,
            [*common, "--slot", f"$({POD_INDEX_ENV})", "--step", "other"],
            a.namespace,
        ),
    ]
    print(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=1))


if __name__ == "__main__":
    main()
