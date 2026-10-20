// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The Helm chart (decisions 0036, 0037), rendered with `helm` and applied to a real API server:
//! every object it renders is accepted, and its RBAC lets the coordinator take, renew and release
//! its Lease and never delete it, nor touch another Lease.

mod common;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

use everserve_controller::api::{Credential, Gate, KubernetesGate, Refusal, ask_of};
use everserve_controller::http::Method;
use everserve_controller::kubernetes::Access;
use everserve_controller::lease::{KubernetesLease, Lease};
use everserve_controller::proto::everserve::coord::v1::LeaseRecord;

use common::bounded;
use common::kube::ApiServer;

const NAMESPACE: &str = "everserve";
const LEASE: &str = "everserve-coordinator";

/// The chart rendered for `NAMESPACE` with the pilot policy, and the node values it requires.
fn rendered() -> String {
    let chart = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../deploy/kubernetes/everserve");
    let policy = std::fs::read_to_string(common::example("policy-pilot.json")).expect("policy");
    let dir = tempfile::tempdir().expect("tmp");
    let ca = dir.path().join("ca.crt");
    std::fs::write(
        &ca,
        "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n",
    )
    .expect("ca");
    let out = std::process::Command::new("helm")
        .args(["template", "everserve", "--include-crds"])
        .arg(&chart)
        .args(["--namespace", NAMESPACE, "--set-json"])
        .arg(format!("policy={}", policy.replace('\n', "")))
        .args([
            "--set",
            "node.store.slots=4096",
            "--set",
            "node.store.budgetBytes=1073741824",
            "--set",
            "imagePullSecrets[0].name=registry",
        ])
        .arg("--set-file")
        .arg(format!("tls.caCertificate={}", ca.display()))
        .output()
        .unwrap_or_else(|e| panic!("the chart tests need helm on PATH: {e}"));
    assert!(
        out.status.success(),
        "helm template: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8")
}

#[tokio::test]
async fn the_chart_applies_and_its_coordinator_may_hold_its_lease_but_never_delete_it() {
    bounded(Box::pin(async {
        let api = ApiServer::start().await;
        api.kubectl_ok(&["create", "namespace", NAMESPACE], None);
        // every object the chart renders is one the API server takes
        api.kubectl_ok(&["apply", "-f", "-"], Some(&rendered()));

        // the coordinator, under its own service account's token and nothing more
        let token = api.kubectl_ok(
            &["create", "token", "everserve-coordinator", "-n", NAMESPACE],
            None,
        );
        let dir = tempfile::tempdir().expect("tmp");
        let token_file = dir.path().join("token");
        std::fs::write(&token_file, token.trim()).expect("token");
        let as_coordinator = Access {
            token_file,
            ..api.access.clone()
        };
        let lease = KubernetesLease::new(
            &as_coordinator,
            NAMESPACE,
            LEASE,
            "coordinator-0",
            Duration::from_secs(10),
        )
        .expect("lease");
        let hold = lease
            .try_acquire()
            .await
            .expect("may create its Lease")
            .expect("free");
        hold.publish(LeaseRecord {
            address: "10.0.0.1:7000".into(),
            ..LeaseRecord::default()
        })
        .await
        .expect("may update its Lease");
        hold.release().await.expect("may release its Lease");

        let client = as_coordinator.client().expect("client");
        let leases = format!("/apis/coordination.k8s.io/v1/namespaces/{NAMESPACE}/leases");
        let delete = client
            .send(
                Method::DELETE,
                &as_coordinator.url(&format!("{leases}/{LEASE}")),
                &[],
                None,
            )
            .await
            .expect("delete");
        assert_eq!(delete.status, 403, "it may never delete its Lease");
        // another Lease in the namespace is not its to read or write
        api.kubectl_ok(&["create", "-n", NAMESPACE, "-f", "-"], Some(
            r#"{"apiVersion":"coordination.k8s.io/v1","kind":"Lease","metadata":{"name":"other"},"spec":{}}"#,
        ));
        let other = client
            .get(&as_coordinator.url(&format!("{leases}/other")), &[])
            .await
            .expect("get");
        assert_eq!(other.status, 403, "another Lease is not its to read");

        may_keep_its_spec_store(&as_coordinator).await;

        the_api_roles_grant_exactly_their_routes(&api, &as_coordinator).await;
    }))
    .await;
}

/// The operator API's gate, under the coordinator's own token: the chart's delegation lets it
/// review, and each of its roles is granted exactly its routes, every operation the API serves
/// and the watch of every list.
async fn the_api_roles_grant_exactly_their_routes(api: &ApiServer, as_coordinator: &Access) {
    for (who, role) in [
        ("alice", "everserve-viewer"),
        ("bob", "everserve-operator"),
        ("carol", "everserve-launcher"),
    ] {
        api.kubectl_ok(&["create", "serviceaccount", who, "-n", NAMESPACE], None);
        api.kubectl_ok(
            &[
                "create",
                "rolebinding",
                who,
                "-n",
                NAMESPACE,
                &format!("--clusterrole={role}"),
                &format!("--serviceaccount={NAMESPACE}:{who}"),
            ],
            None,
        );
    }
    let token = |who: &str| {
        Credential::Bearer(
            api.kubectl_ok(&["create", "token", who, "-n", NAMESPACE], None)
                .trim()
                .to_owned(),
        )
    };
    let (alice, bob, carol) = (token("alice"), token("bob"), token("carol"));
    let gate = KubernetesGate::new(as_coordinator.clone(), NAMESPACE).expect("gate");
    // every operation the API serves, and the watch of every list: what each role is granted
    let document = everserve_controller::api::routes::document();
    let mut asks = Vec::new();
    for (path, item) in &document.paths.paths {
        for (method, op) in [
            (Method::GET, &item.get),
            (Method::PUT, &item.put),
            (Method::POST, &item.post),
        ] {
            if op.is_none() {
                continue;
            }
            let name = path.contains('{').then(|| "x".to_owned());
            asks.extend(ask_of(&method, path, name.clone(), false));
            if method == Method::GET && name.is_none() && path != "/v1/topology" {
                asks.extend(ask_of(&method, path, None, true));
            }
        }
    }
    assert!(asks.len() >= 18, "{asks:?}");
    for ask in &asks {
        let launcher = ask.subresource == Some("launcher");
        for (who, credential, granted) in [
            ("a viewer", &alice, ask.reads() && !launcher),
            ("an operator", &bob, !launcher),
            ("a launcher", &carol, launcher),
        ] {
            let admitted = gate.admit(credential, ask).await;
            match (granted, admitted) {
                (true, Ok(_)) | (false, Err(Refusal::Forbidden(_))) => {}
                (granted, other) => {
                    panic!("{who} {ask}: granted {granted}, answered {other:?}")
                }
            }
        }
    }
}

/// The string constant `name` in the pod contract (`core/everserve/podcontract.py`), which the
/// engines' pods are rendered from.
fn pod_contract(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/everserve/podcontract.py");
    let text = std::fs::read_to_string(&path).expect("the pod contract");
    let prefix = format!("{name} = ");
    let line = text
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("{name} is not in the pod contract"));
    let value = line[prefix.len()..].trim();
    // a plain string, or an f-string of DAEMON_DIR
    let value = value.trim_start_matches('f').trim_matches('"');
    value.replace("{DAEMON_DIR}", &pod_contract_dir())
}

fn pod_contract_dir() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/everserve/podcontract.py");
    let text = std::fs::read_to_string(&path).expect("the pod contract");
    let line = text
        .lines()
        .find(|l| l.starts_with("DAEMON_DIR = "))
        .expect("DAEMON_DIR");
    line["DAEMON_DIR = ".len()..]
        .trim()
        .trim_matches('"')
        .to_owned()
}

#[tokio::test]
async fn the_node_daemons_meet_the_pod_contract_and_may_read_only_what_they_follow() {
    bounded(Box::pin(async {
        let api = ApiServer::start().await;
        api.kubectl_ok(&["create", "namespace", NAMESPACE], None);
        api.kubectl_ok(&["apply", "-f", "-"], Some(&rendered()));

        // the DaemonSet as the API server holds it
        let daemonset: serde_json::Value = serde_json::from_str(&api.kubectl_ok(
            &["get", "daemonset", "everserve-node", "-n", NAMESPACE, "-o", "json"],
            None,
        ))
        .expect("json");
        meets_the_pod_contract(&daemonset["spec"]["template"]["spec"]);
        // the store's tables count against its memory limit: derived from its budget, 1 GiB
        // here, and the chart's 512 MiB for itself, request equal to limit
        let store = daemonset["spec"]["template"]["spec"]["containers"]
            .as_array()
            .expect("containers")
            .iter()
            .find(|c| c["name"] == "store")
            .expect("the store");
        for side in ["requests", "limits"] {
            assert_eq!(store["resources"][side]["memory"], "1610612736", "{side}");
        }
        reached_only_by_whom_each_port_serves(&api);
        // a private registry's secret, on the node daemons' pods and the coordinator's; and each
        // runs as a numbered user that reads what it mounts
        let coordinator: serde_json::Value = serde_json::from_str(&api.kubectl_ok(
            &["get", "deployment", "everserve-coordinator", "-n", NAMESPACE, "-o", "json"],
            None,
        ))
        .expect("json");
        for pod in [
            &daemonset["spec"]["template"]["spec"],
            &coordinator["spec"]["template"]["spec"],
        ] {
            assert_eq!(pod["imagePullSecrets"][0]["name"], "registry");
            runs_as_a_numbered_user_that_reads_its_volumes(pod);
        }
        // a voluntary disruption takes one coordinator at a time, the one the budget selects
        let budget: serde_json::Value = serde_json::from_str(&api.kubectl_ok(
            &["get", "pdb", "everserve-coordinator", "-n", NAMESPACE, "-o", "json"],
            None,
        ))
        .expect("json");
        assert_eq!(budget["spec"]["maxUnavailable"], 1);
        assert_eq!(
            budget["spec"]["selector"]["matchLabels"],
            coordinator["spec"]["selector"]["matchLabels"],
            "the budget holds the coordinator's pods"
        );
        the_holder_is_selected_by_the_label_it_sets(&api, &coordinator);

        // a node daemon, under its own service account's token: it reads the coordinator's Lease
        // and a Node, and writes neither, and the coordinator's Secret is not its to read
        api.kubectl_ok(&["create", "-n", NAMESPACE, "-f", "-"], Some(&format!(
            r#"{{"apiVersion":"coordination.k8s.io/v1","kind":"Lease","metadata":{{"name":"{LEASE}"}},"spec":{{}}}}"#
        )));
        api.kubectl_ok(&["create", "-f", "-"], Some(
            r#"{"apiVersion":"v1","kind":"Node","metadata":{"name":"node-a"}}"#,
        ));
        api.kubectl_ok(
            &["create", "secret", "generic", "everserve-tls", "-n", NAMESPACE, "--from-literal=ca.key=x"],
            None,
        );
        let token = api.kubectl_ok(&["create", "token", "everserve-node", "-n", NAMESPACE], None);
        let dir = tempfile::tempdir().expect("tmp");
        let token_file = dir.path().join("token");
        std::fs::write(&token_file, token.trim()).expect("token");
        let as_node = Access {
            token_file,
            ..api.access.clone()
        };
        let client = as_node.client().expect("client");
        let status = |method: Method, path: String, body: Option<&str>| {
            let (client, url) = (client.clone(), as_node.url(&path));
            let body = body.map(|b| b.as_bytes().to_vec());
            async move {
                client
                    .send(method, &url, &[("Content-Type", "application/merge-patch+json")], body)
                    .await
                    .expect("request")
                    .status
            }
        };
        let lease = format!("/apis/coordination.k8s.io/v1/namespaces/{NAMESPACE}/leases/{LEASE}");
        assert_eq!(status(Method::GET, lease.clone(), None).await, 200, "it follows the Lease");
        assert_eq!(
            status(Method::PATCH, lease, Some("{}")).await,
            403,
            "and never writes it"
        );
        assert_eq!(status(Method::GET, "/api/v1/nodes/node-a".into(), None).await, 200);
        assert_eq!(
            status(Method::PATCH, "/api/v1/nodes/node-a".into(), Some("{}")).await,
            403
        );
        assert_eq!(
            status(
                Method::GET,
                format!("/api/v1/namespaces/{NAMESPACE}/secrets/everserve-tls"),
                None
            )
            .await,
            403,
            "the CA's key is the coordinator's alone"
        );
    }))
    .await;
}

/// The node `DaemonSet`'s pod `pod` meets the pod contract an engine's pod is rendered from: the
/// same host directories and socket, the same node group, and the issuer's token.
fn meets_the_pod_contract(pod: &serde_json::Value) {
    let host_paths: Vec<&str> = pod["volumes"]
        .as_array()
        .expect("volumes")
        .iter()
        .filter_map(|v| v["hostPath"]["path"].as_str())
        .collect();
    // an engine's pod mounts what the daemon's does, at the same paths
    assert_eq!(
        host_paths,
        [
            pod_contract("DAEMON_DIR"),
            pod_contract("SHM_DIR"),
            pod_contract("COMPILE_CACHE_DIR")
        ],
        "the host directories are the pod contract's"
    );
    // the store fetches and serves compile caches where the engines' vLLM keeps them (0043)
    let store = pod["containers"]
        .as_array()
        .expect("containers")
        .iter()
        .find(|c| c["name"] == "store")
        .expect("the store");
    let cache = store["env"]
        .as_array()
        .expect("env")
        .iter()
        .find(|e| e["name"] == "EVERSERVE_STORE_COMPILE_CACHE_DIR")
        .and_then(|e| e["value"].as_str())
        .expect("its compile cache directory");
    assert_eq!(cache, pod_contract("COMPILE_CACHE_DIR"));
    let daemon = pod["containers"]
        .as_array()
        .expect("containers")
        .iter()
        .find(|c| c["name"] == "daemon")
        .expect("the daemon");
    let socket = daemon["env"]
        .as_array()
        .expect("env")
        .iter()
        .find(|e| e["name"] == "EVERSERVE_SOCKET")
        .and_then(|e| e["value"].as_str())
        .expect("its socket");
    assert_eq!(socket, pod_contract("DAEMON_SOCKET"));
    // the node's group, which every engine pod joins, is the one its directories are given
    // to and the daemon and the store run in
    let group: u64 = pod_contract("NODE_GID").parse().expect("a number");
    assert_eq!(pod["securityContext"]["runAsGroup"], group);
    assert_eq!(pod["securityContext"]["supplementalGroups"][0], group);
    let dirs = pod["initContainers"]
        .as_array()
        .expect("init containers")
        .iter()
        .find(|c| c["name"] == "directories")
        .expect("the directories' init container");
    assert_eq!(dirs["args"][2], group.to_string());
    for path in &host_paths {
        assert!(
            dirs["args"]
                .as_array()
                .expect("args")
                .iter()
                .any(|a| a == path),
            "{path} is made the node group's: {}",
            dirs["args"]
        );
    }
    // every container of ours writes its logs as the release says (`logFormat`)
    for c in pod["initContainers"]
        .as_array()
        .expect("init containers")
        .iter()
        .chain(pod["containers"].as_array().expect("containers"))
        .filter(|c| c["name"] != "directories")
    {
        assert!(
            c["env"].as_array().is_some_and(|env| env
                .iter()
                .any(|e| e["name"] == "EVERSERVE_LOG_FORMAT" && e["value"] == "text")),
            "{} writes its logs as the release says",
            c["name"]
        );
    }
    // the token the issuer takes: the daemons' account, for its audience
    assert_eq!(pod["serviceAccountName"], "everserve-node");
    let audience = pod["volumes"]
        .as_array()
        .expect("volumes")
        .iter()
        .find_map(|v| v["projected"]["sources"][0]["serviceAccountToken"]["audience"].as_str())
        .expect("a projected token");
    assert_eq!(audience, everserve_controller::issuer::AUDIENCE);
}

/// Under `as_coordinator`'s token, the coordinator makes its spec store's objects and deletes
/// none (decision 0036).
async fn may_keep_its_spec_store(as_coordinator: &Access) {
    let client = as_coordinator.client().expect("client");
    let states = format!("/apis/everserve.ai/v1alpha1/namespaces/{NAMESPACE}/everservestates");
    let state = r#"{"apiVersion":"everserve.ai/v1alpha1","kind":"EverserveState","metadata":{"name":"everserve"},"spec":{"term":"1","state":{}}}"#;
    let made = client
        .send(
            Method::POST,
            &as_coordinator.url(&states),
            &[("Content-Type", "application/json")],
            Some(state.as_bytes().to_vec()),
        )
        .await
        .expect("create");
    assert_eq!(made.status, 201, "{}", made.body);
    let delete = client
        .send(
            Method::DELETE,
            &as_coordinator.url(&format!("{states}/everserve")),
            &[],
            None,
        )
        .await
        .expect("delete");
    assert_eq!(
        delete.status, 403,
        "it may never delete its store's objects"
    );
}

/// Every container of `pod` that must run as non-root names its user by number, which is all a
/// kubelet verifies `runAsNonRoot` against; and every Secret, projected or `ConfigMap` volume a
/// non-root pod mounts is readable by it: by anyone, or by the pod's group (`fsGroup`), which
/// owns the files the kubelet writes. Neither shows until a pod is scheduled.
fn runs_as_a_numbered_user_that_reads_its_volumes(pod: &serde_json::Value) {
    let pod_context = &pod["securityContext"];
    let containers = pod["containers"]
        .as_array()
        .into_iter()
        .chain(pod["initContainers"].as_array())
        .flatten();
    let mut non_root = false;
    for c in containers {
        let context = &c["securityContext"];
        let effective = |key: &str| {
            if context[key].is_null() {
                pod_context[key].clone()
            } else {
                context[key].clone()
            }
        };
        if effective("runAsNonRoot") == true {
            non_root = true;
            let user = effective("runAsUser").as_u64();
            assert!(
                user.is_some_and(|u| u > 0),
                "{} must run as non-root and names no numbered user",
                c["name"]
            );
        }
    }
    if !non_root {
        return;
    }
    let group_owns = !pod_context["fsGroup"].is_null();
    for v in pod["volumes"].as_array().into_iter().flatten() {
        let mode = ["secret", "projected", "configMap"]
            .iter()
            .find(|kind| !v[**kind].is_null())
            .map(|kind| v[*kind]["defaultMode"].as_u64().unwrap_or(0o644));
        if let Some(mode) = mode {
            assert!(
                mode & 0o004 != 0 || (group_owns && mode & 0o040 != 0),
                "volume {} (mode {mode:o}) is not readable by its non-root pod",
                v["name"]
            );
        }
    }
}

/// The chart's `NetworkPolicy` objects: the coordinator's link and the stores' replication port are
/// reached from the node daemons' pods alone, and every port each workload serves is named by
/// some rule, so none is left reachable by default or shut by mistake.
fn reached_only_by_whom_each_port_serves(api: &ApiServer) {
    let policy = |name: &str| -> serde_json::Value {
        serde_json::from_str(&api.kubectl_ok(
            &["get", "networkpolicy", name, "-n", NAMESPACE, "-o", "json"],
            None,
        ))
        .expect("json")
    };
    let from_node_pods_only = |rule: &serde_json::Value| {
        rule["from"].as_array().is_some_and(|from| {
            from.iter().all(|peer| {
                peer["podSelector"]["matchLabels"]["app.kubernetes.io/component"] == "node"
                    && peer["namespaceSelector"].is_null()
            })
        })
    };
    let ports_of = |rule: &serde_json::Value| -> Vec<u64> {
        rule["ports"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| p["port"].as_u64())
            .collect()
    };
    for (name, private, served) in [
        ("everserve-coordinator", 7000, vec![7000, 7443, 9107]),
        ("everserve-node", 7100, vec![7100, 9108, 9109]),
    ] {
        let p = policy(name);
        let rules = p["spec"]["ingress"].as_array().expect("ingress rules");
        for rule in rules {
            if ports_of(rule).contains(&private) {
                assert!(
                    from_node_pods_only(rule),
                    "{name}: port {private} is open to more than the node daemons: {rule}"
                );
            }
        }
        let named: BTreeSet<u64> = rules.iter().flat_map(ports_of).collect();
        for port in served {
            assert!(
                named.contains(&port),
                "{name}: port {port} is named by no rule"
            );
        }
    }
}

/// Every coordinator pod is ready while it stands, so a rollout completes and the budget counts
/// the standby; the Service selects the one pod the holder labelled, which it may label and
/// nothing else of pods.
fn the_holder_is_selected_by_the_label_it_sets(api: &ApiServer, coordinator: &serde_json::Value) {
    // every coordinator pod is ready while it stands, so a rollout completes and the budget
    // counts the standby; the Service selects the one pod the holder labelled
    let probe = &coordinator["spec"]["template"]["spec"]["containers"][0]["readinessProbe"];
    assert_eq!(probe["httpGet"]["path"], "/healthz");
    let service: serde_json::Value = serde_json::from_str(&api.kubectl_ok(
        &[
            "get",
            "service",
            "everserve-coordinator",
            "-n",
            NAMESPACE,
            "-o",
            "json",
        ],
        None,
    ))
    .expect("json");
    assert_eq!(
        service["spec"]["selector"]["everserve.ai/lease-holder"], "true",
        "the Service selects the holder by its label: {}",
        service["spec"]["selector"]
    );
    let env = coordinator["spec"]["template"]["spec"]["containers"][0]["env"]
        .as_array()
        .expect("env");
    assert!(
        env.iter().any(|e| e["name"] == "EVERSERVE_POD_NAME"
            && e["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.name"),
        "the coordinator knows its pod's name, to label it"
    );
    // and may label its pod, and nothing else of pods
    let role: serde_json::Value = serde_json::from_str(&api.kubectl_ok(
        &[
            "get",
            "role",
            "everserve-coordinator",
            "-n",
            NAMESPACE,
            "-o",
            "json",
        ],
        None,
    ))
    .expect("json");
    let pod_rules: Vec<&serde_json::Value> = role["spec"]["rules"]
        .as_array()
        .or_else(|| role["rules"].as_array())
        .expect("rules")
        .iter()
        .filter(|r| {
            r["resources"]
                .as_array()
                .is_some_and(|rs| rs.iter().any(|x| x == "pods"))
        })
        .collect();
    assert_eq!(pod_rules.len(), 1, "one rule on pods: {role}");
    let verbs: Vec<&str> = pod_rules[0]["verbs"]
        .as_array()
        .expect("verbs")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(verbs, ["get", "patch"]);
}

#[test]
fn the_chart_is_released_at_the_workspaces_version() {
    // a release tags the image and the chart with one version (.github/workflows/release.yml),
    // and refuses a tag they disagree on: they are kept equal here, at every commit, not found
    // apart at the tag
    let chart = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../deploy/kubernetes/everserve/Chart.yaml"),
    )
    .expect("Chart.yaml");
    let field = |name: &str| {
        chart
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{name}: ")))
            .map_or_else(
                || panic!("no {name} in Chart.yaml"),
                |v| v.trim_matches('"').to_owned(),
            )
    };
    let workspace = env!("CARGO_PKG_VERSION");
    assert_eq!(field("version"), workspace, "the chart's version");
    assert_eq!(field("appVersion"), workspace, "the chart's appVersion");
    // and the chart's default image is the one that release publishes, at that version
    let expected = format!("image: ghcr.io/everserve-ai/everserve:{workspace}");
    assert!(
        rendered().contains(&expected),
        "the rendered chart names {expected}"
    );
}
