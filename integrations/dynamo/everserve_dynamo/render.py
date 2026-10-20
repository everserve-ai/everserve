# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""One protected instance under NVIDIA Dynamo, rendered as the objects that deploy it.

An instance is a DynamoGraphDeployment (Dynamo's operator, ``nvidia.com/v1beta1``) of three
components, a headless Service for its rendezvous, and its engine's entry for Everserve's policy
(decision 0041):

- ``Frontend``: Dynamo's frontend, unchanged.
- ``head``: one pod running Dynamo's vLLM worker as node 0 of the engine, with Everserve's
  executor. It alone registers the instance with Dynamo, and it is placed by the instance's head
  placement, on-demand capacity for a spot engine (decision 0041). Its node's loss restarts the
  engine, and Dynamo's frontend migrates its streams (decision 0045).
- ``nodes``: the engine's other nodes and its spare nodes, one pod each, every one running the
  node role. Each registers with its node's daemon undecided and is told to serve a rank or stand
  as a spare, so Grove's recreation of a lost pod becomes the instance's next spare and never a
  second holder of a rank.

The instance is two worker components rather than Dynamo's ``multinode``, whose leader is a GPU
worker launched with vLLM's own executor backend; a spare node in a ``multinode`` group would also
join the instance's collectives. The engine's entry is shape only (decision 0036): its widths and
head placement, and no members, which the control plane places from the nodes that register.

    python -m everserve_dynamo instance.json --out DIR
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from everserve import podcontract
from everserve_dynamo import vllm

API_VERSION = "nvidia.com/v1beta1"
"""The DynamoGraphDeployment version rendered: the operator's storage version."""
GRAPH_LABEL = "nvidia.com/dynamo-graph-deployment-name"
COMPONENT_LABEL = "nvidia.com/dynamo-component"
COMPONENT_TYPE_LABEL = "nvidia.com/dynamo-component-type"
"""The labels Dynamo's operator puts on a component's pods, which the head's Service selects."""
HEAD = "head"
NODES = "nodes"
RENDEZVOUS = "rendezvous"
"""The head's rendezvous Service is ``<instance>-rendezvous``."""
RENDEZVOUS_PORT = 29500
POD_INDEX_ENV = "GROVE_PCLQ_POD_INDEX"
"""The pod's index in its clique, which Grove sets in every pod it creates: the node's slot."""
_VERSION = re.compile(r"^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$")
_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


@dataclass(frozen=True)
class Placement:
    """Where a component's pods may run: the scheduler's selector and tolerations."""

    node_selector: Mapping[str, str] = field(default_factory=dict)
    tolerations: tuple[Mapping, ...] = ()


@dataclass(frozen=True)
class Instance:
    """One instance to deploy."""

    name: str
    namespace: str
    image: str
    """An image with Dynamo's vLLM runtime and Everserve's adapter installed."""
    engine_args: tuple[str, ...]
    """The engine's arguments in vLLM's spelling, the same for every process."""
    gpus_per_node: int
    spare_nodes: int = 0
    """Nodes beyond the engine's own, each standing as a spare for any of its ranks."""
    guarantee: str = "exact"
    head_on_demand: bool = True
    """The head is placed on capacity that is not preempted (decision 0041): its node's loss
    restarts the engine (decision 0045), so a spot engine keeps it off spot."""
    head: Placement = Placement()
    nodes: Placement = Placement()
    frontend_image: str | None = None
    """Dynamo's frontend's image; the instance's image when unset."""
    node_resources: Mapping[str, str] = field(default_factory=dict)
    """Limits beyond the GPUs every GPU pod takes, such as a fabric's devices."""
    image_pull_secrets: tuple[str, ...] = ()
    """Secrets every pod pulls its image with, from a private registry."""
    metrics_namespaces: tuple[str, ...] = ()
    """Namespaces whose pods may reach the engine's pods' metrics and health ports (a Prometheus);
    nothing else outside the instance reaches them (:func:`engine_policy`)."""
    migration_limit: int = 3
    """How many times Dynamo's frontend re-sends a broken stream to another instance, with the
    tokens it has generated (Dynamo 1.5.1's ``--migration-limit``, off at 0 by default there).
    A lost head breaks every stream it serves, and only migration continues them, on another
    instance (decision 0045, ``bench/runs/2026-10-15-dynamo-migration-read``). 0
    turns migration off."""
    global_dynamo_namespace: bool = False
    """The instance's components in Dynamo's global namespace rather than the graph's own
    (``globalDynamoNamespace``): every instance so placed serves its model behind every one of
    their frontends, and a stream a lost head breaks migrates to another instance's engine
    (:attr:`migration_limit`). Off, a frontend sees its own graph's engine alone."""
    dynamo_runtime_version: str = ""
    """The Dynamo runtime the image carries (MAJOR.MINOR.PATCH): Dynamo's operator reads it from an
    image's tag, which an Everserve image's is not, and refuses a graph without it then."""


@dataclass(frozen=True)
class Rendered:
    """What deploys an instance: Kubernetes objects, and the engine's policy entry."""

    objects: tuple[dict, ...]
    engine: dict
    values: dict
    """What Everserve's chart must be installed with for this instance: its node stores' slots."""


_KEYS = {
    "name": "name",
    "namespace": "namespace",
    "image": "image",
    "engineArgs": "engine_args",
    "gpusPerNode": "gpus_per_node",
    "spareNodes": "spare_nodes",
    "guarantee": "guarantee",
    "headOnDemand": "head_on_demand",
    "head": "head",
    "nodes": "nodes",
    "frontendImage": "frontend_image",
    "nodeResources": "node_resources",
    "imagePullSecrets": "image_pull_secrets",
    "dynamoRuntimeVersion": "dynamo_runtime_version",
    "migrationLimit": "migration_limit",
    "globalDynamoNamespace": "global_dynamo_namespace",
    "metricsNamespaces": "metrics_namespaces",
}


def _placement(where: str, d: Mapping) -> Placement:
    unknown = set(d) - {"nodeSelector", "tolerations"}
    if unknown:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}")
    return Placement(dict(d.get("nodeSelector", {})), tuple(d.get("tolerations", ())))


def load(path: str | Path) -> Instance:
    """Read an instance (JSON, the fields of :class:`Instance` in camelCase).

    Raises:
        ValueError: for an unknown field.
    """
    d = json.loads(Path(path).read_text())
    unknown = set(d) - set(_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
    kw = {_KEYS[k]: v for k, v in d.items()}
    kw["engine_args"] = tuple(kw["engine_args"])
    kw["image_pull_secrets"] = tuple(kw.get("image_pull_secrets", ()))
    kw["metrics_namespaces"] = tuple(kw.get("metrics_namespaces", ()))
    for p in ("head", "nodes"):
        if p in kw:
            kw[p] = _placement(f"{path}: {p}", kw[p])
    return Instance(**kw)


def _tag(image: str) -> str:
    """The tag of ``image`` (``registry/name:tag``), or empty when it has none."""
    last = image.rsplit("/", 1)[-1]
    return last.rsplit(":", 1)[1] if ":" in last else ""


def render(inst: Instance) -> Rendered:
    """The objects and the policy entry for ``inst``.

    Raises:
        ValueError: for a name Kubernetes refuses, a guarantee class not known, an engine
            whose ranks do not fill its nodes, an engine that does not state its KV cache
            (:func:`everserve_dynamo.vllm.store_slots`), or an image whose Dynamo runtime version
            Dynamo's operator could not read.
    """
    for what, v in (("name", inst.name), ("namespace", inst.namespace)):
        if not _NAME.match(v) or len(v) > 63 - len(f"-{RENDEZVOUS}"):
            raise ValueError(f"{what} {v!r} is not a DNS label short enough for its Service")
    if inst.dynamo_runtime_version and not _VERSION.match(inst.dynamo_runtime_version):
        raise ValueError(
            f"dynamo runtime version {inst.dynamo_runtime_version!r}: MAJOR.MINOR.PATCH"
        )
    if not inst.dynamo_runtime_version and not _VERSION.match(_tag(inst.image)):
        # Dynamo's operator reads the runtime's version from the image's tag and refuses the graph
        # when the tag is none, as an Everserve image's never is: refused here, by name
        raise ValueError(
            f"image {inst.image!r} carries no Dynamo version in its tag: state the Dynamo runtime "
            "it carries as dynamoRuntimeVersion (MAJOR.MINOR.PATCH, deploy/images/engine/"
            "DYNAMO_PIN for Everserve's engine image)"
        )
    slots = vllm.store_slots(inst.engine_args)
    if inst.guarantee not in ("exact", "continuous"):
        raise ValueError(f"guarantee {inst.guarantee!r}: exact or continuous")
    if not 0 <= inst.migration_limit < 2**32:
        raise ValueError(f"migrationLimit {inst.migration_limit}: 0 (off) to 2^32 - 1")
    tp, pp = vllm.widths(inst.engine_args)
    world = tp * pp
    if inst.gpus_per_node < 1 or world % inst.gpus_per_node:
        raise ValueError(
            f"{world} ranks (tp {tp} x pp {pp}) do not fill nodes of {inst.gpus_per_node}"
        )
    nnodes = world // inst.gpus_per_node
    if inst.spare_nodes < 0:
        raise ValueError("spare nodes cannot be negative")
    if nnodes == 1 and inst.spare_nodes:
        # a spare node stands for another node's ranks; one node's are the head's, whose loss
        # restarts the engine (decision 0045), never a replacement
        raise ValueError("an engine on one node has no node a spare node could stand for")
    # its own name: Dynamo's operator keeps a Service of its own named for each component
    # (`<graph>-<component>`, the component's system port), which would take ours over
    service = f"{inst.name}-{RENDEZVOUS}"
    rendezvous = f"{service}.{inst.namespace}.svc"
    labels = {GRAPH_LABEL: inst.name, COMPONENT_LABEL: HEAD}
    gpus = {"nvidia.com/gpu": str(inst.gpus_per_node), **inst.node_resources}

    def container(command: dict, extra: Mapping | None = None) -> dict:
        return {
            "name": "main",
            "image": inst.image,
            **command,
            "env": [*vllm.env(inst.name, inst.guarantee), *podcontract.env()],
            "volumeMounts": podcontract.mounts(),
            "resources": {"limits": dict(gpus)},
            **(extra or {}),
        }

    pulls = (
        {"imagePullSecrets": [{"name": n} for n in inst.image_pull_secrets]}
        if inst.image_pull_secrets
        else {}
    )

    def pod(c: dict, where: Placement) -> dict:
        spec: dict = {
            "securityContext": {**podcontract.security_context(), "runAsUser": vllm.IMAGE_UID},
            "containers": [c],
            "volumes": podcontract.volumes(),
        }
        if where.node_selector:
            spec["nodeSelector"] = dict(where.node_selector)
        if where.tolerations:
            spec["tolerations"] = [dict(t) for t in where.tolerations]
        spec.update(pulls)
        return {"spec": spec}

    # one configuration for every process: the head's launcher's additions on the nodes too
    args = vllm.engine_args(inst.engine_args)
    head = vllm.head(args, nnodes, rendezvous, RENDEZVOUS_PORT)
    components = [
        {
            "name": "Frontend",
            "type": "frontend",
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "image": inst.frontend_image or inst.image,
                            "env": [
                                {"name": "DYN_MIGRATION_LIMIT", "value": str(inst.migration_limit)}
                            ],
                        }
                    ],
                    **pulls,
                }
            },
        },
    ]
    components.append(
        {
            "name": HEAD,
            "type": "worker",
            "replicas": 1,
            "podTemplate": pod(container(head), inst.head),
        }
    )
    replicas = nnodes - 1 + inst.spare_nodes
    if replicas:
        node = vllm.node(
            inst.name,
            podcontract.NODE_ENV,
            POD_INDEX_ENV,
            args,
            nnodes,
            rendezvous,
            RENDEZVOUS_PORT,
        )
        components.append(
            {
                "name": NODES,
                "type": "worker",
                "replicas": replicas,
                "podTemplate": pod(
                    container(node, vllm.probes()),
                    inst.nodes,
                ),
            }
        )
        # a node pod being deleted hands its ranks to their spares before it stops: its grace
        # outlasts that (vllm.GRACE_S), so Kubernetes never kills it mid-handover
        components[-1]["podTemplate"]["spec"]["terminationGracePeriodSeconds"] = vllm.GRACE_S
    if inst.dynamo_runtime_version:
        for c in components:
            c["runtimeVersionOverride"] = inst.dynamo_runtime_version
    if inst.global_dynamo_namespace:
        for c in components:
            c["globalDynamoNamespace"] = True
    graph = {
        "apiVersion": API_VERSION,
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": inst.name, "namespace": inst.namespace},
        "spec": {"backendFramework": "vllm", "components": components},
    }
    # the nodes reach the head's rendezvous by name before it is ready: it is ready only once
    # they have joined
    rendezvous_service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": service, "namespace": inst.namespace, "labels": dict(labels)},
        "spec": {
            "clusterIP": "None",
            "publishNotReadyAddresses": True,
            "selector": labels,
            "ports": [{"name": "rendezvous", "port": RENDEZVOUS_PORT}],
        },
    }
    engine = {
        "name": inst.name,
        "tensorParallel": tp,
        "pipelineParallel": pp,
        "dataParallel": 1,
        "expertParallel": 1,
        "placementIndependentModel": False,
        "headOnDemand": inst.head_on_demand,
        "guarantee": f"GUARANTEE_{inst.guarantee.upper()}",
    }
    return Rendered(
        objects=(
            graph,
            rendezvous_service,
            engine_policy(inst),
            *((frontends_policy(inst),) if inst.global_dynamo_namespace else ()),
        ),
        engine=engine,
        values={"node": {"store": {"slots": slots}}},
    )


SYSTEM_PORT = 9090
"""Dynamo's system port on a worker pod: its health and metrics."""


def engine_policy(inst: Instance) -> dict:
    """The NetworkPolicy that keeps the engine's pods to the instance's own traffic.

    The head and the engine's nodes talk over transports with no authentication of their own:
    vLLM's rendezvous store, its workers' output queues, NCCL's sockets. Only the instance's pods
    (its frontend, head and nodes) reach them; the namespaces ``metricsNamespaces`` names reach
    the metrics and health ports alone. The frontend is left out: clients reach it as before.
    """
    from everserve_dynamo import vllm as backend

    own = {"podSelector": {"matchLabels": {GRAPH_LABEL: inst.name}}}
    rules: list[dict] = [{"from": [own]}]
    if inst.metrics_namespaces:
        rules.append(
            {
                "from": [
                    {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": n}}}
                    for n in inst.metrics_namespaces
                ],
                "ports": [
                    {"port": SYSTEM_PORT, "protocol": "TCP"},
                    {"port": backend.HEALTH_PORT, "protocol": "TCP"},
                ],
            }
        )
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": f"{inst.name}-engine", "namespace": inst.namespace},
        "spec": {
            "podSelector": {
                "matchLabels": {GRAPH_LABEL: inst.name},
                "matchExpressions": [
                    {"key": COMPONENT_LABEL, "operator": "In", "values": [HEAD, NODES]}
                ],
            },
            "policyTypes": ["Ingress"],
            "ingress": rules,
        },
    }


def frontends_policy(inst: Instance) -> dict:
    """The NetworkPolicy that lets the namespace's frontends reach the instance's head.

    In Dynamo's global namespace (:attr:`Instance.global_dynamo_namespace`), another instance's
    frontend sends requests to this one's head, a stream a lost head broke among them: its own
    graph's label does not admit it (:func:`engine_policy`). Frontends of the instance's namespace
    reach its head alone; the engine's nodes stay its own pods'. A frontend is Dynamo's, by
    ``nvidia.com/dynamo-component-type``.
    """
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": f"{inst.name}-frontends", "namespace": inst.namespace},
        "spec": {
            "podSelector": {"matchLabels": {GRAPH_LABEL: inst.name, COMPONENT_LABEL: HEAD}},
            "policyTypes": ["Ingress"],
            "ingress": [
                {"from": [{"podSelector": {"matchLabels": {COMPONENT_TYPE_LABEL: "frontend"}}}]}
            ],
        },
    }


def write(r: Rendered, out: Path, name: str) -> tuple[Path, Path, Path]:
    """Write the instance's three files into ``out``.

    ``<name>.k8s.json``, a v1 List for ``kubectl apply -f``; ``<name>.engine.json``, the engine's
    policy entry; ``<name>.values.json``, values for Everserve's chart (``helm -f``).
    """
    out.mkdir(parents=True, exist_ok=True)
    objects = out / f"{name}.k8s.json"
    engine = out / f"{name}.engine.json"
    values = out / f"{name}.values.json"
    objects.write_text(
        json.dumps({"apiVersion": "v1", "kind": "List", "items": list(r.objects)}, indent=2) + "\n"
    )
    engine.write_text(json.dumps(r.engine, indent=2) + "\n")
    values.write_text(json.dumps(r.values, indent=2) + "\n")
    return objects, engine, values


def main(argv: Sequence[str] | None = None) -> None:
    """Render an instance file into a directory."""
    import argparse

    ap = argparse.ArgumentParser(prog="everserve_dynamo", description=__doc__.split("\n", 1)[0])
    ap.add_argument("instance", help="the instance, JSON")
    ap.add_argument("--out", required=True, help="the directory to write into")
    a = ap.parse_args(argv)
    inst = load(a.instance)
    for p in write(render(inst), Path(a.out), inst.name):
        print(p)
