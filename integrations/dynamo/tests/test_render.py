# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""An instance renders to a graph Dynamo's operator admits, a rendezvous its nodes reach, and an
engine entry the policy loader takes."""

import json
from pathlib import Path

import jsonschema
import pytest

from everserve import podcontract
from everserve_dynamo import render as r

SCHEMA = json.loads(
    (Path(__file__).parents[1] / "schema" / "dynamographdeployment-v1beta1.json").read_text()
)
KV = ["--num-gpu-blocks-override", "4096", "--block-size", "16"]
ARGS = [
    "--model",
    "/models/m",
    "--tensor-parallel-size",
    "8",
    "--pipeline-parallel-size",
    "2",
    *KV,
]


def instance(**kw):
    base = dict(
        name="llama",
        namespace="serving",
        image="registry/everserve-dynamo:1",
        dynamo_runtime_version="1.6.0",
        engine_args=tuple(ARGS),
        gpus_per_node=8,
        spare_nodes=1,
        head=r.Placement({"pool": "on-demand"}),
        nodes=r.Placement({"pool": "spot"}, ({"key": "spot", "operator": "Exists"},)),
    )
    return r.Instance(**{**base, **kw})


def component(graph, name):
    return next(c for c in graph["spec"]["components"] if c["name"] == name)


def test_the_graph_is_one_the_operator_admits():
    graph, *_ = r.render(instance()).objects
    jsonschema.validate(graph, SCHEMA)


def test_the_head_is_node_0_of_dynamo_s_worker_and_its_service_finds_it_before_it_is_ready():
    graph, service, _ = r.render(instance()).objects
    head = component(graph, r.HEAD)
    assert head["replicas"] == 1
    pod = head["podTemplate"]["spec"]
    c = pod["containers"][0]
    assert "python3 -m dynamo.vllm" in c["command"][2]
    # the head's flags are resolved for the guarantee its pod's environment states, once
    assert "$(python3 -m everserve_vllm.flags --head)" in c["command"][2]
    assert {"name": "EVERSERVE_GUARANTEE", "value": "exact"} in c["env"]
    a = c["args"]
    assert a[: len(ARGS)] == ARGS
    assert a[a.index("--nnodes") + 1] == "2" and a[a.index("--node-rank") + 1] == "0"
    assert a[a.index("--master-addr") + 1] == "llama-rendezvous.serving.svc"
    assert pod["nodeSelector"] == {"pool": "on-demand"}
    # never the operator's own Service for the component (`<graph>-<component>`)
    assert service["metadata"]["name"] == "llama-rendezvous"
    assert service["spec"]["clusterIP"] == "None"
    assert service["spec"]["publishNotReadyAddresses"] is True
    assert service["spec"]["selector"] == {r.GRAPH_LABEL: "llama", r.COMPONENT_LABEL: r.HEAD}


ADDITIONAL = ["--additional-config", '{"consolidator_endpoints":null}']
"""What Dynamo's head adds to vLLM's additional_config, which every process takes."""


def test_the_head_and_its_nodes_take_one_set_of_engine_arguments_with_what_dynamo_adds():
    graph, *_ = r.render(instance()).objects
    head = component(graph, r.HEAD)["podTemplate"]["spec"]["containers"][0]["args"]
    node = component(graph, r.NODES)["podTemplate"]["spec"]["containers"][0]["args"]
    # the head's additional_config, set by Dynamo, is otherwise the one difference refused
    assert head[: len(ARGS) + 2] == [*ARGS, *ADDITIONAL] == node[-len(ARGS) - 2 :]


def test_an_operator_s_additional_config_keeps_its_keys_and_takes_dynamo_s():
    from everserve_dynamo import vllm

    given = ["--model", "m", "--additional-config", '{"mine": 1}']
    assert vllm.engine_args(given) == [
        "--model",
        "m",
        "--additional-config",
        '{"consolidator_endpoints":null,"mine":1}',
    ]
    theirs = ['--additional-config={"consolidator_endpoints": "tcp://kvbm:5555"}']
    assert vllm.engine_args(theirs)[-1] == '{"consolidator_endpoints":"tcp://kvbm:5555"}'


def test_a_node_pod_being_deleted_has_the_grace_to_hand_its_ranks_over():
    from everserve_dynamo import vllm

    graph, *_ = r.render(instance()).objects
    spec = component(graph, r.NODES)["podTemplate"]["spec"]
    args = spec["containers"][0]["args"]
    within = float(args[args.index("--leave-within-s") + 1])
    # Kubernetes kills the pod at its grace; the handover's bound and the process's own shutdown
    # must fit inside it
    assert spec["terminationGracePeriodSeconds"] == vllm.GRACE_S >= within + 30


def test_the_image_refuses_a_dynamo_whose_head_adds_what_the_renderer_does_not():
    from everserve_dynamo import vllm

    # as dynamo/vllm/main.py writes them at the pin
    pinned = (
        'vllm_config.additional_config["consolidator_endpoints"] = consolidator_endpoints\n'
        'vllm_config.additional_config["benchmark"] = bench\n'
        'bench_cfg = vllm_config.additional_config.get("benchmark")\n'
    )
    assert vllm.unaccounted_head_additions(pinned) == set()
    later = pinned + "vllm_config.additional_config['router_hint'] = hint\n"
    assert vllm.unaccounted_head_additions(later) == {"router_hint"}


def test_the_nodes_are_the_engine_s_others_and_its_spares_each_deciding_at_registration():
    graph, *_ = r.render(instance(spare_nodes=2)).objects
    nodes = component(graph, r.NODES)
    assert nodes["replicas"] == 1 + 2
    pod = nodes["podTemplate"]["spec"]
    c = pod["containers"][0]
    assert c["command"] == ["python3", "-m", "everserve_vllm.node_main"]
    assert "--node-rank" not in c["args"]  # the control plane tells it which, or to stand
    assert c["args"][c["args"].index("--slot") + 1] == f"$({r.POD_INDEX_ENV})"
    a = c["args"]
    assert a[a.index("--node") + 1] == f"$({podcontract.NODE_ENV})"  # the Node, from the pod
    assert c["args"][-len(ARGS) - 2 :] == [*ARGS, *ADDITIONAL]
    assert c["livenessProbe"]["httpGet"]["path"] == "/live"
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert pod["tolerations"] == [{"key": "spot", "operator": "Exists"}]


def test_every_gpu_pod_meets_the_pod_contract():
    graph, *_ = r.render(instance()).objects
    for name in (r.HEAD, r.NODES):
        pod = component(graph, name)["podTemplate"]["spec"]
        c = pod["containers"][0]
        names = {e["name"] for e in c["env"]}
        assert {e["name"] for e in podcontract.env()} <= names
        assert {"EVERSERVE_ENGINE", "EVERSERVE_GRAPH_SAFE"} <= names
        assert pod["volumes"] == podcontract.volumes()
        assert c["volumeMounts"] == podcontract.mounts()
        sc = pod["securityContext"]  # never root, and in the node's group
        assert sc["runAsNonRoot"] and sc["runAsUser"] != 0
        assert sc["supplementalGroups"] == [podcontract.NODE_GID]


def test_a_single_node_engine_without_spares_has_no_node_component():
    one = ["--model", "m", "-tp", "8", *KV]
    graph, *_ = r.render(instance(engine_args=tuple(one), spare_nodes=0)).objects
    assert [c["name"] for c in graph["spec"]["components"]] == ["Frontend", r.HEAD]
    jsonschema.validate(graph, SCHEMA)


def test_an_engine_whose_ranks_do_not_fill_its_nodes_is_refused():
    with pytest.raises(ValueError, match="do not fill"):
        r.render(instance(gpus_per_node=6))
    with pytest.raises(ValueError, match="DNS label"):
        r.render(instance(name="Llama"))
    with pytest.raises(ValueError, match="guarantee"):
        r.render(instance(guarantee="best"))
    with pytest.raises(ValueError, match="one node"):
        r.render(instance(engine_args=("--model", "m", "-tp", "8", *KV), spare_nodes=1))


def test_the_engine_entry_is_shape_only_and_the_policy_loader_takes_it(tmp_path):
    from everserve.policy import Guarantee, load

    engine = r.render(instance()).engine
    assert "members" not in engine
    doc = {
        "version": "v1",
        "outcomesInOrder": ["OUTCOME_REPLACE"],
        "spares": {"perEngine": 1, "loading": "LOADING_PRELOADED"},
        "cover": {"enabled": True},  # its spares stand after bring-up: cover plans their sets
        "engines": [engine],
    }
    (tmp_path / "policy.json").write_text(json.dumps(doc))
    e = load(tmp_path / "policy.json").engine("llama")
    assert (e.tensor_parallel, e.pipeline_parallel, e.world_size) == (8, 2, 16)
    assert e.guarantee is Guarantee.EXACT


def test_the_node_role_takes_the_command_it_is_rendered_with(monkeypatch):
    from everserve_vllm import node_main

    graph, *_ = r.render(instance()).objects
    c = component(graph, r.NODES)["podTemplate"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in c["env"]}
    # the instance's guarantee is stated once, in the pod's environment, which node_main reads
    monkeypatch.setattr(node_main, "settings_guarantee", lambda: env["EVERSERVE_GUARANTEE"])
    pod = {f"$({podcontract.NODE_ENV})": "gpu-7", f"$({r.POD_INDEX_ENV})": "1"}
    a = node_main.parse([pod.get(x, x) for x in c["args"]])
    assert (a.engine, a.node, a.nnodes, a.tp, a.pp) == ("llama", "gpu-7", 2, 8, 2)
    assert (a.slot, a.node_rank) == (1, None)  # it asks the control plane what it is
    # it resolves what it steps with under the instance's guarantee, as the head's flags are
    assert a.guarantee == instance().guarantee and "--guarantee" not in c["args"]


def test_an_instance_file_with_an_unknown_field_is_refused(tmp_path):
    p = tmp_path / "i.json"
    p.write_text(json.dumps({"name": "x", "replicas": 2}))
    with pytest.raises(ValueError, match="replicas"):
        r.load(p)
    p.write_text(
        json.dumps(
            {
                "name": "llama",
                "namespace": "serving",
                "image": "i",
                "engineArgs": ARGS,
                "gpusPerNode": 8,
                "head": {"nodeSelector": {"pool": "on-demand"}},
            }
        )
    )
    assert r.load(p).head.node_selector == {"pool": "on-demand"}


def test_every_pod_pulls_from_a_private_registry_with_the_instance_s_secrets():
    graph, *_ = r.render(instance(image_pull_secrets=("nebius-cr",))).objects
    for c in graph["spec"]["components"]:
        assert c["podTemplate"]["spec"]["imagePullSecrets"] == [{"name": "nebius-cr"}]
    jsonschema.validate(graph, SCHEMA)
    graph, *_ = r.render(instance()).objects
    assert all(
        "imagePullSecrets" not in c["podTemplate"]["spec"] for c in graph["spec"]["components"]
    )


def test_the_image_s_dynamo_runtime_is_declared_on_every_component():
    """The operator refuses a graph whose image tag is no Dynamo version without it."""
    graph, *_ = r.render(instance(dynamo_runtime_version="1.6.0")).objects
    assert {c["runtimeVersionOverride"] for c in graph["spec"]["components"]} == {"1.6.0"}
    jsonschema.validate(graph, SCHEMA)
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        r.render(instance(dynamo_runtime_version="main"))


def test_only_the_instance_s_pods_reach_the_engine_s_unauthenticated_transports():
    *_, policy = r.render(instance(metrics_namespaces=("monitoring",))).objects
    assert policy["kind"] == "NetworkPolicy"
    sel = policy["spec"]["podSelector"]
    assert sel["matchLabels"] == {r.GRAPH_LABEL: "llama"}
    assert sel["matchExpressions"][0]["values"] == [r.HEAD, r.NODES]  # the frontend stays open
    own, metrics = policy["spec"]["ingress"]
    assert own == {"from": [{"podSelector": {"matchLabels": {r.GRAPH_LABEL: "llama"}}}]}
    assert {p["port"] for p in metrics["ports"]} == {r.SYSTEM_PORT, 8090}
    *_, closed = r.render(instance()).objects
    assert len(closed["spec"]["ingress"]) == 1  # no namespace named: the instance's own alone


def test_the_engine_states_its_kv_cache_and_the_chart_s_store_slots_come_from_it(tmp_path):
    """The node store holds a row per slot the engine can write: the instance states its KV cache
    and the renderer gives the chart the number, which nothing else knows before the engine runs
    (a chart default of 4,096 slots against an engine of about 6 million)."""
    rendered = r.render(instance())
    assert rendered.values == {"node": {"store": {"slots": 4096 * 16}}}
    *_, values = r.write(rendered, tmp_path, "llama")
    assert json.loads(values.read_text()) == rendered.values
    eq = [a for a in ARGS if a not in KV] + ["--num-gpu-blocks-override=8", "--block-size=32"]
    assert r.render(instance(engine_args=tuple(eq))).values["node"]["store"]["slots"] == 256
    for args in (
        [a for a in ARGS if a not in KV],
        [*[a for a in ARGS if a not in KV], "--block-size", "16"],
        [*[a for a in ARGS if a not in KV], "--num-gpu-blocks-override", "0", "--block-size", "16"],
    ):
        with pytest.raises(ValueError, match="num-gpu-blocks-override"):
            r.render(instance(engine_args=tuple(args)))


def test_an_image_whose_tag_is_no_dynamo_version_needs_the_runtime_stated():
    """Dynamo's operator reads the runtime's version from the image's tag and refuses the graph
    when the tag is none: the renderer refuses first, naming the field."""
    with pytest.raises(ValueError, match="dynamoRuntimeVersion"):
        r.render(instance(dynamo_runtime_version=""))
    graph, *_ = r.render(
        instance(image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0", dynamo_runtime_version="")
    ).objects
    assert all("runtimeVersionOverride" not in c for c in graph["spec"]["components"])


def test_the_frontend_migrates_a_broken_stream_three_times_unless_told_otherwise():
    """A lost head breaks every stream it serves; only Dynamo's migration continues them, on another
    instance (decision 0045). Dynamo 1.5.1's frontend has it off (0) unless given."""

    def limit(inst):
        graph, *_ = r.render(inst).objects
        (main,) = component(graph, "Frontend")["podTemplate"]["spec"]["containers"]
        return {e["name"]: e["value"] for e in main["env"]}["DYN_MIGRATION_LIMIT"]

    assert limit(instance()) == "3"
    assert limit(instance(migration_limit=0)) == "0"  # off, said so
    with pytest.raises(ValueError, match="migrationLimit"):
        r.render(instance(migration_limit=-1))
    graph, *_ = r.render(instance()).objects
    jsonschema.validate(graph, SCHEMA)  # the operator admits the frontend's environment


def test_instances_in_dynamo_s_global_namespace_serve_behind_each_other_s_frontends():
    """A stream a lost head breaks migrates to another instance's engine only where the frontend
    sees it: every component in Dynamo's global namespace, as the schema states it."""
    graph, *_ = r.render(instance(global_dynamo_namespace=True)).objects
    jsonschema.validate(graph, SCHEMA)
    assert all(c["globalDynamoNamespace"] is True for c in graph["spec"]["components"])
    alone, *rest = r.render(instance()).objects
    assert not any("globalDynamoNamespace" in c for c in alone["spec"]["components"])
    assert [o["metadata"]["name"] for o in rest if o["kind"] == "NetworkPolicy"] == ["llama-engine"]


def test_another_instance_s_frontend_reaches_the_head_alone_in_the_global_namespace():
    """A frontend pod carries nvidia.com/dynamo-component-type frontend; the engine's nodes stay
    its own pods'."""
    _, *rest = r.render(instance(global_dynamo_namespace=True)).objects
    (p,) = [o for o in rest if o["metadata"]["name"] == "llama-frontends"]
    assert p["spec"]["podSelector"]["matchLabels"] == {
        r.GRAPH_LABEL: "llama",
        r.COMPONENT_LABEL: r.HEAD,
    }
    (rule,) = p["spec"]["ingress"]
    assert rule == {
        "from": [{"podSelector": {"matchLabels": {"nvidia.com/dynamo-component-type": "frontend"}}}]
    }
