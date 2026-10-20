# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""RunPod from the laptop for a drill: deploy, describe, terminate, balance, and a watchdog.

    python bench/harness/runpod.py balance
    python bench/harness/runpod.py deploy --gpus 5 --gpu-type "NVIDIA A40" --log run.log
    python bench/harness/runpod.py pod POD_ID
    python bench/harness/runpod.py spend --start-balance 52.03 --log run.log --label arm
    python bench/harness/runpod.py watchdog POD_ID --at UNIX_SECONDS --log run.log
    python bench/harness/runpod.py terminate POD_ID --log run.log

The API key is read from the file ``RUNPOD_API_KEY_FILE`` names (``~/.runpod/api-key`` unless set)
and never printed. Every call that
spends or stops spending is appended to ``--log`` with a UTC timestamp and the price.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

API = "https://api.runpod.io/graphql"
KEY = Path(os.environ.get("RUNPOD_API_KEY_FILE", str(Path.home() / ".runpod" / "api-key")))
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def gql(query: str, variables: dict | None = None) -> dict:
    """One GraphQL call; returns the whole response (``data`` and any ``errors``)."""
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        API,
        body,
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY.read_text().strip()}",
            "User-Agent": "everserve-bench",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def log(path: str | None, line: str) -> None:
    """Append ``line`` with a UTC timestamp to ``path`` and print it."""
    stamped = f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} {line}"
    print(stamped, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(stamped + "\n")


def balance() -> float:
    """The account balance in dollars."""
    return float(gql("query { myself { clientBalance } }")["data"]["myself"]["clientBalance"])


def deploy(gpus: int, gpu_type: str, cloud: str, disk_gb: int, name: str) -> dict:
    """Ask for an on-demand pod; returns the response (``errors`` when there is no supply)."""
    q = """mutation($in: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $in) {
        id costPerHr machine { gpuDisplayName location } } }"""
    return gql(
        q,
        {
            "in": {
                "cloudType": cloud,
                "gpuCount": gpus,
                "gpuTypeId": gpu_type,
                "name": name,
                "imageName": IMAGE,
                "containerDiskInGb": disk_gb,
                "volumeInGb": 0,
                "minVcpuCount": 16,
                "minMemoryInGb": 120,
                "ports": "22/tcp",
                "startSsh": True,
                "supportPublicIp": True,
            }
        },
    )


REST = "https://rest.runpod.io/v1"


def rest(method: str, path: str, body: dict | None = None) -> dict:
    """One call to RunPod's REST API, which has what GraphQL lacks (global networking)."""
    req = urllib.request.Request(
        REST + path,
        json.dumps(body).encode() if body is not None else None,
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY.read_text().strip()}",
            "User-Agent": "everserve-bench",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return {"errors": [{"message": e.read().decode(errors="replace"), "status": e.code}]}


def deploy_networked(
    gpus: int, gpu_types: list[str], disk_gb: int, name: str, data_centers: list[str]
) -> dict:
    """A secure on-demand pod on RunPod's private network (``globalNetworking``).

    Pods there reach each other on every port, which a multi-node engine needs: its queues bind
    random ports. ``data_centers`` empty lets RunPod pick; a second pod names the first's, so
    both share one.
    """
    body = {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuCount": gpus,
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "availability",
        "name": name,
        "imageName": IMAGE,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "globalNetworking": True,
    }
    if data_centers:
        body["dataCenterIds"] = data_centers
        body["dataCenterPriority"] = "custom"
    return rest("POST", "/pods", body)


def cluster_create(gpu_type: str, pods: int, gpus_per_pod: int, data_center: str) -> dict:
    """An Instant Cluster: ``pods`` nodes of ``gpus_per_pod`` GPUs on one fast network.

    RunPod's unmanaged offering (``type: SLURM``): plain nodes reachable from each other on every
    port, with the fabric between them. Whether ``gpus_per_pod`` below the node's size is sold,
    and whether there is stock, only the attempt says; a refusal creates nothing.
    """
    q = """mutation($in: CreateClusterInput!) { createCluster(input: $in) {
        id name type gpuTypeId podCount gpuCountPerPod pods { id name } } }"""
    body = {
        "gpuTypeId": gpu_type,
        "podCount": pods,
        "gpuCountPerPod": gpus_per_pod,
        "type": "SLURM",
    }
    if data_center:
        body["dataCenterId"] = data_center
    return gql(q, {"in": body})


def clusters() -> list[dict]:
    """The account's clusters with their pods."""
    q = """query { myself { clusters {
        id name type gpuTypeId podCount gpuCountPerPod pods { id name } } } }"""
    return gql(q)["data"]["myself"]["clusters"]


def cluster_delete(cluster_id: str) -> dict:
    """Delete a cluster and every pod in it."""
    return gql(
        "mutation($in: DeleteClusterInput!) { deleteCluster(input: $in) }",
        {"in": {"id": cluster_id}},
    )


def pod_rest(pod_id: str) -> dict:
    """The pod as the REST API shows it: ``publicIp``, ``portMappings``, ``machine``."""
    return rest("GET", f"/pods/{pod_id}")


def pod(pod_id: str) -> dict:
    """The pod's status, price and public SSH endpoint."""
    q = """query($id: String!) { pod(input: {podId: $id}) { id desiredStatus costPerHr
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }"""
    return gql(q, {"id": pod_id})["data"]["pod"]


def terminate(pod_id: str) -> dict:
    """Terminate the pod (its disk goes with it)."""
    return gql("mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pod_id})


def main() -> None:
    """Command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "balance",
            "deploy",
            "pod",
            "terminate",
            "watchdog",
            "spend",
            "cluster",
            "clusters",
            "cluster-delete",
        ],
    )
    ap.add_argument("pod_id", nargs="?")
    ap.add_argument("--gpus", type=int, default=5)
    ap.add_argument("--gpu-type", default="NVIDIA A40")
    ap.add_argument("--disk", type=int, default=200)
    ap.add_argument(
        "--cloud", choices=["COMMUNITY", "SECURE"], help="only this cloud (default: both)"
    )
    ap.add_argument(
        "--networked",
        action="store_true",
        help="deploy over the REST API with global networking on (secure cloud only); "
        "--gpu-type may be a comma-separated list, tried by availability",
    )
    ap.add_argument(
        "--data-center", default="", help="with --networked or cluster: the data center"
    )
    ap.add_argument("--pods", type=int, default=2, help="cluster: nodes")
    ap.add_argument("--name", default="everserve-adapter-port")
    ap.add_argument("--at", type=float, default=0.0)
    ap.add_argument("--start-balance", type=float, default=0.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--log")
    a = ap.parse_args()
    if a.cmd == "balance":
        log(a.log, f"balance {balance():.2f}")
    elif a.cmd == "deploy" and a.networked:
        out = deploy_networked(
            a.gpus,
            [g.strip() for g in a.gpu_type.split(",") if g.strip()],
            a.disk,
            a.name,
            [a.data_center] if a.data_center else [],
        )
        log(a.log, f"deploy {a.gpus}x {a.gpu_type} networked: {json.dumps(out)[:400]}")
        if out.get("id"):
            print(json.dumps({"id": out["id"], "costPerHr": out.get("costPerHr")}))
            return
        raise SystemExit("no supply, or refused")
    elif a.cmd == "deploy":
        for cloud in (a.cloud,) if a.cloud else ("COMMUNITY", "SECURE"):
            out = gql_safe(lambda c=cloud: deploy(a.gpus, a.gpu_type, c, a.disk, a.name))
            log(a.log, f"deploy {a.gpus}x {a.gpu_type} {cloud}: {json.dumps(out)}")
            p = (out.get("data") or {}).get("podFindAndDeployOnDemand")
            if p:
                print(json.dumps(p))
                return
        raise SystemExit("no supply in either cloud")
    elif a.cmd == "cluster":
        out = gql_safe(lambda: cluster_create(a.gpu_type, a.pods, a.gpus, a.data_center))
        log(
            a.log,
            f"cluster {a.pods}x {a.gpus}x {a.gpu_type} {a.data_center}: {json.dumps(out)[:500]}",
        )
        c = (out.get("data") or {}).get("createCluster")
        if c:
            print(json.dumps(c))
            return
        raise SystemExit("no cluster: " + json.dumps(out.get("errors"))[:300])
    elif a.cmd == "clusters":
        print(json.dumps(clusters()))
    elif a.cmd == "cluster-delete":
        log(a.log, f"cluster-delete {a.pod_id}: {json.dumps(cluster_delete(a.pod_id))}")
    elif a.cmd == "pod":
        print(json.dumps(pod_rest(a.pod_id) if a.networked else pod(a.pod_id)))
    elif a.cmd == "terminate":
        log(a.log, f"terminate {a.pod_id}: {json.dumps(terminate(a.pod_id))}")
    elif a.cmd == "spend":
        p = pod(a.pod_id) if a.pod_id else None
        up = p["runtime"]["uptimeInSeconds"] if p and p["runtime"] else 0
        now = balance()
        rate = f", pod {a.pod_id} ${p['costPerHr']}/h up {up}s" if p else ""
        log(
            a.log,
            f"spend [{a.label}] balance {now:.2f}, "
            f"spent since start ${a.start_balance - now:.2f}{rate}",
        )
    elif a.cmd == "watchdog":
        while time.time() < a.at:
            time.sleep(min(30.0, max(1.0, a.at - time.time())))
        log(
            a.log,
            f"WATCHDOG at the cap time: terminate {a.pod_id}: {json.dumps(terminate(a.pod_id))}",
        )


def gql_safe(fn):  # noqa: ANN001, ANN201 - a thunk returning a GraphQL response
    """Call ``fn``, turning a transport error into an ``errors`` response."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - logged by the caller
        return {"errors": [{"message": f"{type(e).__name__}: {e}"}]}


if __name__ == "__main__":
    main()
