// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator API knows who asks (decision 0039): no credential is refused, a viewer reads and
//! writes nothing, an operator's write is recorded as theirs, a node's certificate is no
//! operator's whatever the settings name, and on Kubernetes a bearer token's rights are the RBAC
//! the cluster grants.

mod common;

use std::sync::Arc;

use everserve_controller::api::{CertificateRoles, Gates, KubernetesGate};
use everserve_controller::config::CertsArgs;
use everserve_controller::coord::Coordinator;
use everserve_controller::http::{Bearer, Client, Method};
use everserve_controller::policy;
use everserve_telemetry::Telemetry;
use tokio_util::sync::CancellationToken;

use common::api::{client_as, serve, serve_behind};
use common::bounded;
use common::kube::ApiServer;

fn coordinator() -> Arc<Coordinator> {
    let policy = policy::load(&common::example("policy-pilot.json")).expect("policy");
    Coordinator::new(policy, &Telemetry::new()).0
}

#[tokio::test]
async fn a_viewer_reads_an_operator_writes_and_no_one_else_does_either() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let coord = coordinator();
    let api = serve(&coord, dir.path(), &cancel).await;
    let cordon = "/v1/nodes/node-a";
    let body = r#"{"spec":{"unschedulable":true,"reason":"firmware"}}"#;

    let (status, _) = api
        .call(&api.anonymous, Method::GET, "/v1/engines", "")
        .await;
    assert_eq!(status, 401, "no credential: no answer, reads included");
    let (status, _) = api.call(&api.anonymous, Method::PUT, cordon, body).await;
    assert_eq!(status, 401);

    let (status, _) = api.call(&api.viewer, Method::GET, "/v1/engines", "").await;
    assert_eq!(status, 200, "a viewer reads");
    let (status, _) = api.call(&api.viewer, Method::PUT, cordon, body).await;
    assert_eq!(status, 403, "a viewer writes nothing");
    let (status, _) = api
        .call(
            &api.viewer,
            Method::POST,
            "/v1/engines/k3-a/stacks",
            r#"{"rank":0}"#,
        )
        .await;
    assert_eq!(status, 403, "nor signals a process");

    let (status, body_) = api.call(&api.operator, Method::PUT, cordon, body).await;
    assert_eq!(status, 200, "{body_}");
    let spec = coord.node_spec("node-a").await.expect("written");
    assert_eq!(
        spec.written_by, "operator api: operator",
        "the write is recorded as its caller's, not a name the caller chose"
    );
    cancel.cancel();
}

#[tokio::test]
async fn a_nodes_certificate_is_no_operators_whatever_the_settings_name() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let coord = coordinator();
    // settings that name the node and the stores' shared name as operators: a node's certificate
    // is valid for both, and still for no operator's name
    let api = serve_behind(&coord, dir.path(), &cancel, |_| Gates {
        certificates: Some(CertificateRoles {
            operators: vec!["node-a".into(), "store".into()],
            viewers: vec!["node-a".into()],
        }),
        kubernetes: None,
    })
    .await;
    everserve_controller::certs::write(&CertsArgs {
        out: api.certs.clone(),
        nodes: vec!["node-a".into()],
        operators: Vec::new(),
        coordinator_name: "localhost".into(),
        store_name: "store".into(),
        days: 1,
        overwrite: false,
    })
    .expect("a node's certificate under the same CA");
    let node = client_as(&api.certs, "node-a");
    let (status, _) = api
        .call(
            &node,
            Method::PUT,
            "/v1/nodes/node-b",
            r#"{"spec":{"unschedulable":true}}"#,
        )
        .await;
    assert_eq!(status, 403, "a node may not cordon its neighbour");
    let (status, _) = api.call(&node, Method::GET, "/v1/engines", "").await;
    assert_eq!(status, 403, "nor read as a viewer");
    cancel.cancel();
}

#[tokio::test]
async fn on_kubernetes_a_tokens_rights_are_the_rbac_the_cluster_grants() {
    bounded(Box::pin(async {
        let k8s = ApiServer::start().await;
        k8s.kubectl_ok(&["create", "namespace", "everserve"], None);
        k8s.kubectl_ok(&["create", "serviceaccount", "ops", "-n", "everserve"], None);
        // ops may update and list nodes and update pools in everserve.ai, and nothing else: not
        // watch nodes, not write a pool's launcher
        k8s.kubectl_ok(
            &["apply", "-f", "-"],
            Some(
                r#"{"apiVersion":"rbac.authorization.k8s.io/v1","kind":"Role","metadata":{"name":"drainer","namespace":"everserve"},"rules":[{"apiGroups":["everserve.ai"],"resources":["nodes"],"verbs":["update","list"]},{"apiGroups":["everserve.ai"],"resources":["pools"],"verbs":["update"]}]}"#,
            ),
        );
        k8s.kubectl_ok(
            &[
                "create", "rolebinding", "drainer", "-n", "everserve", "--role=drainer",
                "--serviceaccount=everserve:ops",
            ],
            None,
        );
        let token = k8s.kubectl_ok(&["create", "token", "ops", "-n", "everserve"], None);

        let dir = tempfile::tempdir().expect("tmp");
        let cancel = CancellationToken::new();
        let coord = coordinator();
        // the coordinator reviews with its own credentials: here the test's administrator
        let access = k8s.access.clone();
        let api = serve_behind(&coord, dir.path(), &cancel, move |_| Gates {
            certificates: None,
            kubernetes: Some(KubernetesGate::new(access, "everserve").expect("gate")),
        })
        .await;
        let roots = everserve_wire::transport::root_store(&api.certs.join("ca.crt")).expect("ca");
        let as_token = |t: &str| Client::with(Some(roots.clone()), Some(Bearer::Token(t.into())));
        let ops = as_token(token.trim());

        let (status, body) = api
            .call(
                &ops,
                Method::PUT,
                "/v1/nodes/node-a",
                r#"{"spec":{"unschedulable":true}}"#,
            )
            .await;
        assert_eq!(status, 200, "granted update on nodes: {body}");
        let spec = coord.node_spec("node-a").await.expect("written");
        assert_eq!(spec.written_by, "operator api: system:serviceaccount:everserve:ops");
        let (status, _) = api.call(&ops, Method::GET, "/v1/engines", "").await;
        assert_eq!(status, 403, "not granted list on engines");
        let (status, _) = api.call(&ops, Method::GET, "/v1/nodes", "").await;
        assert_eq!(status, 200, "granted list on nodes");
        let (status, _) = api
            .call(&ops, Method::GET, "/v1/nodes?watch=true", "")
            .await;
        assert_eq!(status, 403, "a watch is its own verb, not granted");
        let (status, _) = api
            .call(
                &ops,
                Method::PUT,
                "/v1/pools/p/launcher",
                r#"{"command":["sh"]}"#,
            )
            .await;
        assert_eq!(status, 403, "update on pools grants no launcher");
        let (status, _) = api
            .call(&as_token("not-a-token"), Method::GET, "/v1/engines", "")
            .await;
        assert_eq!(status, 401, "a token the cluster does not know");
        cancel.cancel();
    }))
    .await;
}

#[tokio::test]
async fn a_caller_that_sends_slowly_or_not_at_all_is_cut_off() {
    // a caller holding connections open by sending nothing, or a body that never ends, would
    // hold the coordinator's connections and its routes' tasks for as long as it liked
    use tokio::io::AsyncWriteExt;
    bounded(Box::pin(async {
        let dir = tempfile::tempdir().expect("tmp");
        let cancel = CancellationToken::new();
        let coord = coordinator();
        let deadlines = everserve_controller::api::Deadlines {
            header_read: std::time::Duration::from_millis(300),
            request: std::time::Duration::from_millis(300),
        };
        let roles = CertificateRoles {
            operators: vec!["operator".into()],
            viewers: Vec::new(),
        };
        let api = common::api::serve_within(
            &coord,
            dir.path(),
            &cancel,
            |_| Gates {
                certificates: Some(roles),
                kubernetes: None,
            },
            deadlines,
        )
        .await;
        let bound = std::time::Duration::from_secs(5);

        // a connection that sends nothing
        let mut silent = common::api::operator_connection(&api).await;
        common::api::read_to_close(&mut silent, bound).await;

        // a connection kept alive after its answer, then idle
        let mut kept = common::api::operator_connection(&api).await;
        kept.write_all(b"GET /v1/engines HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .expect("send");
        let answered = common::api::read_to_close(&mut kept, bound).await;
        assert!(answered.starts_with("HTTP/1.1 200"), "{answered}");

        // a body that never ends: answered 408 once its headers are in
        let mut slow = common::api::operator_connection(&api).await;
        slow.write_all(
            b"PUT /v1/nodes/node-a HTTP/1.1\r\nHost: localhost\r\n\
              Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{",
        )
        .await
        .expect("send");
        slow.flush().await.expect("flush");
        let answered = common::api::read_to_close(&mut slow, bound).await;
        assert!(answered.starts_with("HTTP/1.1 408"), "{answered}");
        cancel.cancel();
    }))
    .await;
}
