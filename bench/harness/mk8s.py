# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Rent and release a Nebius managed Kubernetes cluster for the integration's tests.

A cluster in the project of ``--parent-id`` (or ``EVERSERVE_NEBIUS_PROJECT``), on the project's
subnet, with a public endpoint, and one node group of a VM shape: CPU nodes for the Kubernetes test
without GPUs (the chart, the DaemonSet, a rendered instance's pods and their registrations), GPU
nodes for the Dynamo integration's proof. Every command prints JSON; ``--log`` appends a dated line
per action, as ``nebius.py`` does.

    mk8s.py up --name everserve-e2e --version 1.34 --platform cpu-d3 --preset 8vcpu-32gb --nodes 3
        --kubeconfig ~/.kube/everserve-e2e
    mk8s.py up ... --platform gpu-l40s-a --preset 1gpu-8vcpu-32gb --drivers cuda13.0 [--preemptible]
    mk8s.py down --name everserve-e2e | list | watchdog --name everserve-e2e --at <epoch seconds>

``down`` deletes the node groups and then the cluster, and is what ``watchdog`` runs at ``--at``:
the time cap a rental holds whatever its driver does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from nebius import Nebius, detach, log  # noqa: E402


def cluster(nb: Nebius, name: str) -> dict[str, Any] | None:
    """The cluster called ``name`` in the project, or None."""
    for c in nb.run("mk8s", "cluster", "list").get("items", []):
        if c["metadata"].get("name") == name:
            return c
    return None


def node_groups(nb: Nebius, cluster_id: str) -> list[dict[str, Any]]:
    """The cluster's node groups."""
    return nb.run("mk8s", "node-group", "list", "--parent-id", cluster_id, parent=False).get(
        "items", []
    )


def wait(nb: Nebius, kind: str, rid: str, timeout_s: float) -> dict[str, Any]:
    """Until the resource's state is RUNNING, or raise at ``timeout_s``."""
    deadline = time.time() + timeout_s
    while True:
        d = nb.run("mk8s", kind, "get", "--id", rid, parent=False)
        state = (d.get("status") or {}).get("state")
        if state == "RUNNING":
            return d
        if time.time() > deadline:
            raise RuntimeError(f"{kind} {rid} is {state} after {timeout_s:.0f} s")
        time.sleep(15)


def nodes_running(nb: Nebius, cid: str, a: argparse.Namespace) -> dict[str, Any]:
    """The cluster's node group once RUNNING, or raise naming what was asked and for how long.

    The cloud does not refuse nodes it has no capacity for: their node group waits, with no
    reason given (3 x gpu-l40s, 30 minutes, quota free). The
    run's bound, ``a.nodes_within_s``, ends the wait, and the error names the shape, the region and
    the node group's last state.
    """
    deadline = time.time() + a.nodes_within_s
    last: dict[str, Any] = {}
    while True:
        groups = node_groups(nb, cid)
        if groups:
            gid = groups[0]["metadata"]["id"]
            last = nb.run("mk8s", "node-group", "get", "--id", gid, parent=False)
            if (last.get("status") or {}).get("state") == "RUNNING":
                return last
        if time.time() > deadline:
            region = (nb.project().get("spec") or {}).get("region", "the project's region")
            raise RuntimeError(
                f"no capacity for {a.nodes} x {a.platform} {a.preset} in {region} after "
                f"{a.nodes_within_s:.0f} s: the node group's last status was "
                f"{json.dumps(last.get('status') or {}, sort_keys=True)[:600]}"
            )
        time.sleep(15)


NON_GPU_VCPU = "compute.instance.non-gpu.vcpu"


def cpu_room(nb: Nebius, a: argparse.Namespace) -> None:
    """Refuse a node group of CPU nodes the project's region has no vCPU quota for, before renting.

    The cloud does not refuse it: the node group's create waits for nodes it can never get, until
    the CLI's own timeout, with no reason given (15 minutes in a region that allows no CPU VMs).
    A GPU platform's nodes count against GPU quotas instead.
    """
    if a.platform.startswith("gpu-"):
        return
    vcpus = int(a.preset.split("vcpu", 1)[0])
    need = vcpus * a.nodes
    limit = nb.quota_limit(NON_GPU_VCPU)
    if limit is not None and limit < need:
        raise RuntimeError(
            f"{a.nodes} x {a.platform} {a.preset} need {need} vCPUs and this project's region "
            f"allows {limit} ({NON_GPU_VCPU}): name a project in a region that allows them"
        )


def up(nb: Nebius, a: argparse.Namespace) -> dict[str, Any]:
    """Create the cluster and its node group, wait for both, write the kubeconfig.

    A run creates its own cluster and never adopts one of the name: an existing cluster is the
    last run's in its take-down (which, between its node groups' deletion and its own, looks like
    a cluster just made) or another run's, and is waited out until it is gone.
    """
    cpu_room(nb, a)
    deadline = time.time() + 1800
    while cluster(nb, a.name) is not None:
        if time.time() > deadline:
            raise RuntimeError(f"a cluster {a.name} still exists after 30 min: another run's?")
        time.sleep(15)
    out = nb.run(
        "mk8s",
        "cluster",
        "create",
        "--name",
        a.name,
        "--control-plane-version",
        a.version,
        "--control-plane-subnet-id",
        nb.subnet(),
        "--control-plane-endpoints-public-endpoint=true",
        timeout_s=1800,
    )
    cid = out.get("metadata", {}).get("id") or out.get("resource_id", "")
    wait(nb, "cluster", cid, 1800)
    if not node_groups(nb, cid):
        args = [
            "mk8s",
            "node-group",
            "create",
            "--parent-id",
            cid,
            "--name",
            f"{a.name}-nodes",
            "--fixed-node-count",
            str(a.nodes),
            "--template-resources-platform",
            a.platform,
            "--template-resources-preset",
            a.preset,
            "--template-boot-disk-size-gibibytes",
            str(a.disk),
            "--template-boot-disk-type",
            "network_ssd",
        ]
        if a.drivers:
            args += ["--template-gpu-settings-drivers-preset", a.drivers]
        if a.preemptible:
            args += ["--template-preemptible"]
        # asked without waiting: the cloud's own wait for nodes it cannot place has no end and no
        # reason, and the run's bound says what it saw instead (nodes_running)
        # --async answers with the operation's id as text, not JSON; the answer is kept in the
        # run's log as the CLI gave it, the recording a test fakes it from
        asked = nb.run(*args, "--async", parent=False, timeout_s=120, text=True)
        log(a.log, f"node-group create --async answered: {asked!r}")
    ng = nodes_running(nb, cid, a)
    nb.run(
        "mk8s",
        "cluster",
        "get-credentials",
        "--id",
        cid,
        "--external",
        "--force",
        "--kubeconfig",
        str(Path(a.kubeconfig).expanduser()),
        parent=False,
        text=True,
    )
    return {"cluster": cid, "node_group": ng["metadata"]["id"], "kubeconfig": a.kubeconfig}


def down(nb: Nebius, name: str) -> dict[str, Any]:
    """Delete the cluster's node groups, then the cluster.

    A deletion already under way (an interrupted earlier ``down``) is waited out, not asked again.
    """
    c = cluster(nb, name)
    if c is None:
        return {"down": name, "found": False}
    if (c.get("status") or {}).get("state") == "DELETING":
        return {"down": name, "cluster": c["metadata"]["id"], "already": True}
    cid = c["metadata"]["id"]
    for ng in node_groups(nb, cid):
        if (ng.get("status") or {}).get("state") != "DELETING":
            nb.run(
                "mk8s",
                "node-group",
                "delete",
                "--id",
                ng["metadata"]["id"],
                parent=False,
                timeout_s=1800,
            )
    deadline = time.time() + 1800
    while node_groups(nb, cid):
        if time.time() > deadline:
            raise RuntimeError(f"{name}'s node groups are still being deleted after 30 min")
        time.sleep(15)
    nb.run("mk8s", "cluster", "delete", "--id", cid, parent=False, timeout_s=1800)
    return {"down": name, "cluster": cid}


def main() -> None:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("cmd", choices=["up", "down", "list", "watchdog"])
    ap.add_argument("--profile", default=os.environ.get("EVERSERVE_NEBIUS_PROFILE", "everserve"))
    ap.add_argument("--parent-id", default=os.environ.get("EVERSERVE_NEBIUS_PROJECT", ""))
    ap.add_argument("--name", default="everserve-e2e")
    ap.add_argument(
        "--version",
        default="1.33",
        help="Kubernetes: 1.32 or newer (TokenReview's node name for a pod-bound token)",
    )
    ap.add_argument("--platform", default="cpu-d3")
    ap.add_argument("--preset", default="8vcpu-32gb")
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--disk", type=int, default=200, help="each node's boot disk, GiB")
    ap.add_argument("--drivers", default="", help="a GPU node group's driver preset")
    ap.add_argument("--preemptible", action="store_true", help="spot nodes, which Nebius reclaims")
    ap.add_argument(
        "--nodes-within-s",
        type=float,
        default=900.0,
        help="how long the node group may take to run before the run says there is no capacity",
    )
    ap.add_argument("--kubeconfig", default="~/.kube/everserve-e2e")
    ap.add_argument("--at", type=float, default=0.0, help="watchdog: take it down at this time")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if not a.parent_id:
        ap.error("--parent-id or EVERSERVE_NEBIUS_PROJECT is required")
    nb = Nebius(a.profile, a.parent_id)
    if a.cmd == "up":
        out = up(nb, a)
        log(a.log, f"up {a.name} {a.platform}/{a.preset} x{a.nodes}: {json.dumps(out)}")
    elif a.cmd == "down":
        out = down(nb, a.name)
        log(a.log, f"down: {json.dumps(out)}")
    elif a.cmd == "list":
        out = [
            {
                "name": c["metadata"]["name"],
                "id": c["metadata"]["id"],
                "state": (c.get("status") or {}).get("state"),
            }
            for c in nb.run("mk8s", "cluster", "list").get("items", [])
        ]
    else:
        if detach():
            return  # the driver's side: the watchdog runs on in its own session
        while time.time() < a.at:
            time.sleep(min(30, max(1, a.at - time.time())))
        try:
            out = down(nb, a.name)
        except Exception as e:  # noqa: BLE001 - already gone is the usual case
            out = {"watchdog": a.name, "error": str(e)}
        log(a.log, f"watchdog {a.name}: {json.dumps(out)}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
