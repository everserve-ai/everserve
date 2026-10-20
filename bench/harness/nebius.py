# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Rent and release Nebius AI Cloud virtual machines for the drills; the ``nebius`` CLI underneath.

A profile of the CLI (``--profile``, ``everserve`` by default) authenticates as the harness's
service account; its parent is the region's project, where every VM, disk and GPU cluster of a
run lives. VMs of one project share its VPC subnet: private addresses that reach each other on
every port, which a multi-node engine needs (its queues bind random ports), plus a public address
for ssh. The boot disk comes from a public image family with the NVIDIA driver and CUDA toolkit
(``ubuntu24.04-cuda13.0``); the login user and key are the harness's, by cloud-init.

    nebius.py deploy --name everserve-a --platform gpu-l40s-a --preset 1gpu-8vcpu-32gb
    nebius.py get <id> | list | terminate <id> | watchdog <id> --at <epoch seconds>
    nebius.py budget <id> <id> ... --since <epoch seconds> --gpus-each 1 --rate <USD per GPU-hour>
        --cap <USD>
    nebius.py cluster --name everserve-rig --fabric fabric-3 | clusters | cluster-delete <id>

``deploy --gpu-cluster <id>`` puts a VM on that cluster's InfiniBand fabric (H100/H200 8-GPU
presets): the fabric the cluster rental runs on. Every command prints JSON; ``--log`` appends a
dated line per action to a run log, as ``runpod.py`` does.

``budget`` is the money cap beside the watchdog's time cap: it reckons the rental's spend from
``--since`` (taken before the first create, so the reckoning runs ahead of the bill) at
``--rate``, the per-GPU-hour price as the billing page shows it at booking (the CLI cannot read
it; there is no default), logs it every ``--report-min`` minutes, and releases every VM it names
once the spend, with ``--reserve-min`` minutes of the whole rental's burn for the releases
themselves, reaches ``--cap``. A watchdog time worked out from an assumed rate is no money cap:
three H100s at a rate the booking page showed and the assumption did not billed $65 against $60.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CLI = str(Path.home() / ".nebius" / "bin" / "nebius")
IMAGE_FAMILY = "ubuntu24.04-cuda13.0"


def public_images(project: str) -> str:
    """The region's project of public images: its id carries the region's code, as the project's.

    ``project-e00...`` (eu-north1) has ``project-e00public-images``; ``project-e04...``
    (eu-west2), ``project-e04public-images``.
    """
    code = project.removeprefix("project-")[:3]
    if len(code) != 3:
        raise ValueError(f"{project} is not a project id")
    return f"project-{code}public-images"


USER = "everserve"


def log(path: str | None, line: str) -> None:
    """Append a dated line to ``path`` (and stderr)."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} {line}", file=sys.stderr)
    if path and path != "/dev/null":
        with open(path, "a") as f:
            f.write(f"{stamp} {line}\n")


def spend_usd(since: float, now: float, gpus: int, rate: float) -> float:
    """What ``gpus`` GPUs cost from ``since`` to ``now`` at ``rate`` dollars per GPU-hour."""
    return max(0.0, now - since) / 3600 * gpus * rate


def at_cap(spend: float, gpus: int, rate: float, cap: float, reserve_min: float) -> bool:
    """Whether ``spend`` and ``reserve_min`` minutes more of the rental's burn reach ``cap``."""
    return spend + gpus * rate * reserve_min / 60 >= cap


TRANSIENT = ("code = Unavailable", "code = DeadlineExceeded")
"""The CLI's failures that say the API was not reached, or did not answer in time: the call may be
made again. Not made again, a DNS timeout to the API (``code = Unavailable``) fails a build VM's
release, and the VM stays up."""
NOT_FOUND = "code = NotFound"
REPEATABLE = frozenset({"get", "list", "get-credentials", "delete"})
"""The verbs a call is made again for after a transient failure: those whose second call does what
the first would have. A delete's second call finds nothing when the first deleted it unanswered,
which is the delete done. A create is made once: its answer lost, the run finds what it made by
name, as :meth:`Nebius.deploy`'s caller does."""
RUN_WAITS_S = (5, 15, 30, 60)
"""The waits between a repeatable call's attempts: about two minutes before its error stands."""


class Nebius:
    """The CLI under one profile and one parent project."""

    def __init__(
        self,
        profile: str,
        parent: str,
        waits_s: tuple[float, ...] = RUN_WAITS_S,
        sleep: Any = time.sleep,
    ) -> None:
        """``profile`` is the CLI profile; ``parent`` the project every resource is created in.

        ``waits_s`` are the waits between a repeatable call's attempts (:data:`REPEATABLE`).
        """
        self.profile = profile
        self.parent = parent
        self.waits_s = waits_s
        self.sleep = sleep

    def run(
        self, *args: str, parent: bool = True, timeout_s: float = 900, text: bool = False
    ) -> Any:
        """One CLI call with JSON output; raises with the CLI's message on failure.

        ``text``: a command that answers in text whatever the format (``mk8s cluster
        get-credentials``); its output is returned as it is. A repeatable call
        (:data:`REPEATABLE`) whose failure is transient (:data:`TRANSIENT`), or which timed out,
        is made again after each of :attr:`waits_s`; any other call is made once.
        """
        cmd = [CLI, "--profile", self.profile, *([] if text else ["--format", "json"]), *args]
        if parent:
            cmd += ["--parent-id", self.parent]
        verb = args[2] if len(args) > 2 else ""
        waits = self.waits_s if verb in REPEATABLE else ()
        for attempt, wait_s in enumerate((*waits, None)):
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                if wait_s is None:
                    raise
                error = f"no answer in {timeout_s} s"
            else:
                if p.returncode == 0:
                    if text:
                        return p.stdout
                    return json.loads(p.stdout) if p.stdout.strip() else {}
                error = p.stderr.strip().splitlines()[0] if p.stderr.strip() else p.stdout
                if verb == "delete" and attempt and NOT_FOUND in error:
                    return {}  # an earlier attempt deleted it, its answer lost
                if wait_s is None or not any(t in error for t in TRANSIENT):
                    raise RuntimeError(error)
            print(f"{' '.join(args[:3])} failed ({error}); again in {wait_s} s", file=sys.stderr)
            self.sleep(wait_s)
        raise AssertionError("unreachable")

    def project(self) -> dict[str, Any]:
        """The parent project's record: its region and its tenant."""
        return self.run("iam", "project", "get", self.parent, parent=False)

    def quota_limit(self, name: str) -> int | None:
        """The tenant's limit on ``name`` in the parent project's region; None where it states none.

        The limits are the tenant's, per region (``quotas quota-allowance list`` on the tenant):
        the project's own list states none.
        """
        p = self.project()
        region = p.get("spec", {}).get("region") or p.get("status", {}).get("region", "")
        tenant = p.get("metadata", {}).get("parent_id", "")
        items = self.run(
            "quotas", "quota-allowance", "list", "--parent-id", tenant, parent=False
        ).get("items", [])
        for q in items:
            if (
                q.get("metadata", {}).get("name") == name
                and q.get("spec", {}).get("region") == region
            ):
                limit = q["spec"].get("limit")
                return int(limit) if limit is not None else None
        return None

    def subnet(self) -> str:
        """The project's subnet: the first one, which the default project has exactly one of."""
        items = self.run("vpc", "subnet", "list").get("items", [])
        if not items:
            raise RuntimeError(f"no subnet in {self.parent}")
        return items[0]["metadata"]["id"]

    def deploy(
        self,
        name: str,
        platform: str,
        preset: str,
        disk_gb: int,
        ssh_key: Path,
        image_family: str,
        gpu_cluster: str | None,
    ) -> dict[str, Any]:
        """Create a VM and wait until it runs; returns its record with both addresses."""
        user_data = "\n".join(
            [
                "#cloud-config",
                "users:",
                f"  - name: {USER}",
                "    sudo: ALL=(ALL) NOPASSWD:ALL",
                "    shell: /bin/bash",
                "    ssh_authorized_keys:",
                f"      - {ssh_key.read_text().strip()}",
                "runcmd:",
                f"  - mkdir -p /workspace && chown {USER}:{USER} /workspace",
                "",
            ]
        )
        iface = [
            {"name": "eth0", "ip_address": {}, "public_ip_address": {}, "subnet_id": self.subnet()}
        ]
        args = [
            "compute",
            "instance",
            "create",
            "--name",
            name,
            "--resources-platform",
            platform,
            "--resources-preset",
            preset,
            "--boot-disk-managed-disk-name",
            f"{name}-boot",
            "--boot-disk-managed-disk-type",
            "network_ssd",
            "--boot-disk-managed-disk-size-gibibytes",
            str(disk_gb),
            "--boot-disk-managed-disk-source-image-family-image-family",
            image_family,
            "--boot-disk-managed-disk-source-image-family-parent-id",
            public_images(self.parent),
            "--boot-disk-attach-mode",
            "READ_WRITE",
            "--cloud-init-user-data",
            user_data,
            "--network-interfaces",
            json.dumps(iface),
        ]
        if gpu_cluster:
            args += ["--gpu-cluster-id", gpu_cluster]
        out = self.run(*args)
        instance_id = out.get("metadata", {}).get("id") or out.get("resource_id") or ""
        if not instance_id:
            raise RuntimeError(f"no instance id in the create response: {json.dumps(out)[:300]}")
        return self.wait_running(instance_id)

    def get(self, instance_id: str) -> dict[str, Any]:
        """The VM's state and addresses."""
        d = self.run("compute", "instance", "get", instance_id, parent=False)
        st = d.get("status", {}) or {}
        nics = st.get("network_interfaces") or []
        addr = lambda k: (nics[0].get(k) or {}).get("address", "").split("/")[0] if nics else ""  # noqa: E731
        return {
            "id": d["metadata"]["id"],
            "name": d["metadata"].get("name"),
            "state": st.get("state"),
            "public_ip": addr("public_ip_address"),
            "private_ip": addr("ip_address"),
            "platform": (d.get("spec", {}).get("resources") or {}).get("platform"),
            "preset": (d.get("spec", {}).get("resources") or {}).get("preset"),
        }

    def wait_running(self, instance_id: str, timeout_s: float = 600) -> dict[str, Any]:
        """Poll until the VM runs with both addresses, or raise after ``timeout_s``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            r = self.get(instance_id)
            if r["state"] == "RUNNING" and r["public_ip"] and r["private_ip"]:
                return r
            if r["state"] in ("ERROR", "DELETING"):
                raise RuntimeError(f"instance {instance_id} is {r['state']}")
            time.sleep(5)
        raise TimeoutError(f"instance {instance_id} not running within {timeout_s:.0f} s")

    def list(self) -> list[dict[str, Any]]:
        """Every VM of the project."""
        return [
            self.get(i["metadata"]["id"])
            for i in self.run("compute", "instance", "list").get("items", [])
        ]

    def disks(self) -> list[dict[str, Any]]:
        """Every disk of the project: ``{id, name, state}``."""
        return [
            {
                "id": d["metadata"]["id"],
                "name": d["metadata"].get("name"),
                "state": (d.get("status") or {}).get("state"),
            }
            for d in self.run("compute", "disk", "list").get("items", [])
        ]

    def terminate(self, instance_id: str) -> dict[str, Any]:
        """Delete the VM, then any boot disk made for it that outlived it.

        The disk is found by the name :meth:`deploy` gave it (``<vm name>-boot``); a managed
        boot disk goes with its VM in the cases seen, and this is for the ones that do not.
        Every call it makes is repeatable (:meth:`run`): a moment's loss of the API does not
        leave a paid VM up.
        """
        r = self.get(instance_id)
        self.run("compute", "instance", "delete", instance_id, parent=False)
        gone = [d["id"] for d in self.disks() if d["name"] == f"{r['name']}-boot"]
        for disk_id in gone:
            self.run("compute", "disk", "delete", disk_id, parent=False)
        return {"deleted": instance_id, "disks": gone}

    def cluster_create(self, name: str, fabric: str) -> dict[str, Any]:
        """A GPU cluster on one InfiniBand fabric; VMs join it at creation."""
        out = self.run(
            "compute", "gpu-cluster", "create", "--name", name, "--infiniband-fabric", fabric
        )
        return {"id": out.get("metadata", {}).get("id") or out.get("resource_id"), "fabric": fabric}

    def clusters(self) -> list[dict[str, Any]]:
        """Every GPU cluster of the project."""
        return [
            {
                "id": c["metadata"]["id"],
                "name": c["metadata"].get("name"),
                "fabric": c.get("spec", {}).get("infiniband_fabric"),
            }
            for c in self.run("compute", "gpu-cluster", "list").get("items", [])
        ]

    def cluster_delete(self, cluster_id: str) -> dict[str, Any]:
        """Delete a GPU cluster (its VMs must be gone)."""
        self.run("compute", "gpu-cluster", "delete", cluster_id, parent=False)
        return {"deleted": cluster_id}


def detach() -> bool:
    """Continue in a session of its own, out of the caller's process group; True in the caller.

    What a watchdog runs in: a driver stopped as a whole (its process group signalled, or killed
    outright with its traps unrun) takes everything in its group with it, and a watchdog there
    would go too, leaving the rental up past its cap. Detached, the
    watchdog belongs to no driver and releases the rental at ``--at`` whatever became of the run.
    The caller returns at once; the watchdog's own output goes to its ``--log`` only.
    """
    if os.fork() > 0:
        return True
    os.setsid()
    if os.fork() > 0:
        os._exit(0)  # the session's leader exits; its child can never take a terminal again
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    return False


def main() -> None:
    """The command line; see the module."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "cmd",
        choices=[
            "deploy",
            "get",
            "list",
            "terminate",
            "watchdog",
            "budget",
            "cluster",
            "clusters",
            "cluster-delete",
        ],
    )
    ap.add_argument("id", nargs="?")
    ap.add_argument("more", nargs="*", help="budget: the rental's other VMs")
    ap.add_argument("--profile", default=os.environ.get("EVERSERVE_NEBIUS_PROFILE", "everserve"))
    ap.add_argument(
        "--parent-id",
        default=os.environ.get("EVERSERVE_NEBIUS_PROJECT", ""),
        help="the region's project",
    )
    ap.add_argument("--name", default="everserve")
    ap.add_argument("--platform", default="gpu-l40s-a")
    ap.add_argument("--preset", default="1gpu-8vcpu-32gb")
    ap.add_argument("--disk", type=int, default=200, help="boot disk, GiB")
    ap.add_argument("--image-family", default=IMAGE_FAMILY)
    ap.add_argument(
        "--ssh-key",
        default=os.environ.get("EVERSERVE_SSH_KEY", str(Path.home() / ".ssh" / "everserve"))
        + ".pub",
        help="the public key installed on every VM (EVERSERVE_SSH_KEY names its private key)",
    )
    ap.add_argument("--gpu-cluster", default=None, help="deploy: join this GPU cluster's fabric")
    ap.add_argument("--fabric", default="fabric-3", help="cluster: the InfiniBand fabric")
    ap.add_argument("--at", type=float, default=0.0, help="watchdog: terminate at this epoch time")
    ap.add_argument(
        "--since", type=float, help="budget: the rental's start, before the first create"
    )
    ap.add_argument("--gpus-each", type=int, default=1, help="budget: GPUs per VM")
    ap.add_argument("--rate", type=float, help="budget: USD per GPU-hour, from the billing page")
    ap.add_argument("--cap", type=float, help="budget: USD; every VM is released at it")
    ap.add_argument("--reserve-min", type=float, default=10.0, help="budget: the releases' minutes")
    ap.add_argument("--report-min", type=float, default=10.0, help="budget: minutes between logs")
    ap.add_argument("--log")
    a = ap.parse_args()
    if not a.parent_id:
        ap.error("--parent-id or EVERSERVE_NEBIUS_PROJECT is required")
    nb = Nebius(a.profile, a.parent_id)
    if a.cmd == "deploy":
        out = nb.deploy(
            a.name, a.platform, a.preset, a.disk, Path(a.ssh_key), a.image_family, a.gpu_cluster
        )
        log(a.log, f"deploy {a.name} {a.platform}/{a.preset}: {json.dumps(out)}")
    elif a.cmd == "get":
        out = nb.get(a.id)
    elif a.cmd == "list":
        out = nb.list()
    elif a.cmd == "terminate":
        out = nb.terminate(a.id)
        log(a.log, f"terminate {a.id}: {json.dumps(out)}")
    elif a.cmd == "watchdog":
        # the time cap: whatever the run did, the VM is released at --at
        if detach():
            return
        while time.time() < a.at:
            time.sleep(min(30, max(1, a.at - time.time())))
        try:
            out = nb.terminate(a.id)
        except Exception as e:  # noqa: BLE001 - already gone is the usual case
            out = {"watchdog": a.id, "error": str(e)}
        log(a.log, f"watchdog {a.id}: {json.dumps(out)}")
    elif a.cmd == "budget":
        if a.since is None or not a.rate or not a.cap or not a.id:
            ap.error("budget needs the VMs, --since, --rate (the billing page's) and --cap")
        ids = [a.id, *a.more]
        gpus = a.gpus_each * len(ids)
        log(a.log, f"budget: {len(ids)} VMs, {gpus} GPUs at ${a.rate}/GPU-h, cap ${a.cap}")
        reported = 0.0
        while True:
            spend = spend_usd(a.since, time.time(), gpus, a.rate)
            if at_cap(spend, gpus, a.rate, a.cap, a.reserve_min):
                break
            if time.time() - reported >= a.report_min * 60:
                log(a.log, f"budget: ${spend:.2f} spent of ${a.cap}")
                reported = time.time()
            time.sleep(30)
        log(a.log, f"budget: ${spend:.2f} spent, the cap ${a.cap} reached; releasing {ids}")

        def release(i: str) -> dict[str, Any]:
            try:
                return nb.terminate(i)
            except Exception as e:  # noqa: BLE001 - already gone is the usual case
                return {"budget": i, "error": str(e)}

        with ThreadPoolExecutor(len(ids)) as pool:
            out = list(pool.map(release, ids))
        log(a.log, f"budget: released: {json.dumps(out)}")
    elif a.cmd == "cluster":
        out = nb.cluster_create(a.name, a.fabric)
        log(a.log, f"cluster {a.name} on {a.fabric}: {json.dumps(out)}")
    elif a.cmd == "clusters":
        out = nb.clusters()
    else:
        out = nb.cluster_delete(a.id)
        log(a.log, f"cluster-delete {a.id}: {json.dumps(out)}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
