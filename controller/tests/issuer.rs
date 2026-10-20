// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A node proves its name to be issued its certificate (decision 0037): against a real API server,
//! a node daemon's token bound to a pod on node-a is issued node-a's certificate, which chains to
//! the deployment's CA and is valid for node-a alone; a token from another service account, for
//! another audience, or bound to no pod, is issued nothing.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use everserve_controller::api::Gates;
use everserve_controller::certs::{CA_NAME, NodeKey, authority_from};
use everserve_controller::config::CertsArgs;
use everserve_controller::enroll::{CA_FILE, CERT_FILE, EnrollError, Enrollment, Held, KEY_FILE};
use everserve_controller::issuer::{AUDIENCE, IssueError, Issuer};
use everserve_wire::transport::{Address, Listener, Tls, connect};

use common::kube::ApiServer;
use tokio_util::sync::CancellationToken;

const NAMESPACE: &str = "default";

fn objects() -> String {
    let node = |n: &str| format!("apiVersion: v1\nkind: Node\nmetadata:\n  name: {n}\n---\n");
    let account = |a: &str| {
        format!(
            "apiVersion: v1\nkind: ServiceAccount\nmetadata:\n  name: {a}\n  namespace: {NAMESPACE}\n---\n"
        )
    };
    let pod = |p: &str, node: &str, account: &str| {
        format!(
            "apiVersion: v1\nkind: Pod\nmetadata:\n  name: {p}\n  namespace: {NAMESPACE}\nspec:\n  \
             nodeName: {node}\n  serviceAccountName: {account}\n  containers:\n  - name: c\n    \
             image: x\n---\n"
        )
    };
    [
        node("node-a"),
        node("node-b"),
        account("everserve-node"),
        account("other"),
        pod("daemon-a", "node-a", "everserve-node"),
        pod("daemon-b", "node-b", "everserve-node"),
        pod("other-a", "node-a", "other"),
    ]
    .concat()
}

/// A token for `account`, for `audience`, bound to `pod` when one is named.
fn token(api: &ApiServer, account: &str, audience: &str, pod: Option<&str>) -> String {
    let mut args = vec![
        "create",
        "token",
        account,
        "-n",
        NAMESPACE,
        "--audience",
        audience,
    ];
    if let Some(p) = pod {
        args.extend(["--bound-object-kind", "Pod", "--bound-object-name", p]);
    }
    api.kubectl_ok(&args, None).trim().to_owned()
}

/// The CA `certs` writes in `dir`, rebuilt to sign with, and its certificate.
fn deployment_ca(dir: &Path) -> (rcgen::CertifiedKey, String) {
    everserve_controller::certs::write(&CertsArgs {
        out: dir.to_owned(),
        nodes: Vec::new(),
        // certs writes a CA only beside a certificate it signs
        operators: vec!["ops".into()],
        coordinator_name: "coordinator".into(),
        store_name: "store".into(),
        days: 1,
        overwrite: false,
    })
    .expect("certs");
    let key = std::fs::read_to_string(dir.join("ca.key")).expect("ca.key");
    let ca = authority_from(&key, CA_NAME).expect("the CA");
    (
        ca,
        std::fs::read_to_string(dir.join("ca.crt")).expect("ca.crt"),
    )
}

/// Whether a server presenting `cert` (with `key`) is taken, by a client of the CA in `dir`, for
/// `name`.
async fn serves_as(dir: &Path, cert: &str, key: &rcgen::KeyPair, name: &str) -> bool {
    let (cert_path, key_path) = (dir.join("node.crt"), dir.join("node.key"));
    std::fs::write(&cert_path, cert).expect("cert");
    std::fs::write(&key_path, key.serialize_pem()).expect("key");
    let tls = Tls::load(&cert_path, &key_path, &dir.join("ca.crt")).expect("tls");
    serves(&tls, name).await
}

/// Whether a server presenting `tls`'s certificate is taken, by a client of its CA, for `name`.
async fn serves(tls: &Tls, name: &str) -> bool {
    let address = Address::parse("127.0.0.1:0").expect("address");
    let listener = Listener::bind(&address, Some(tls)).await.expect("bind");
    let bound = listener.local_address().expect("bound");
    let server = tokio::spawn(async move { listener.accept().await.map(|_| ()) });
    let client = connect(&bound, Some(tls), name).await;
    server.abort();
    client.is_ok()
}

#[tokio::test]
async fn a_daemons_token_is_issued_its_pods_nodes_certificate_and_nothing_else() {
    let api = ApiServer::start().await;
    api.kubectl_ok(&["apply", "-f", "-"], Some(&objects()));
    let dir = tempfile::tempdir().expect("tmp");
    let (ca, ca_pem) = deployment_ca(dir.path());
    let issuer = Issuer::new(
        ca,
        ca_pem,
        "store",
        Duration::from_hours(1),
        api.access.clone(),
        NAMESPACE,
        "everserve-node",
    )
    .expect("issuer");
    let key = rcgen::KeyPair::generate().expect("key");
    let node_key = || NodeKey::new(key.public_key_raw().to_vec()).expect("a P-256 point");

    // the token of the daemon on node-a: node-a's certificate, whatever the daemon might want
    let a = token(&api, "everserve-node", AUDIENCE, Some("daemon-a"));
    let for_a = issuer.issue(&a, node_key()).await.expect("issued");
    assert_eq!(for_a.node, "node-a");
    assert!(serves_as(dir.path(), &for_a.certificate, &key, "node-a").await);
    assert!(
        !serves_as(dir.path(), &for_a.certificate, &key, "node-b").await,
        "node-a's certificate is no other node's"
    );
    assert!(
        serves_as(dir.path(), &for_a.certificate, &key, "store").await,
        "and carries the name stores verify on a peer (0026)"
    );
    let b = token(&api, "everserve-node", AUDIENCE, Some("daemon-b"));
    assert_eq!(
        issuer.issue(&b, node_key()).await.expect("issued").node,
        "node-b"
    );

    // another service account's pod on node-a: nothing
    let other = token(&api, "other", AUDIENCE, Some("other-a"));
    assert!(matches!(
        issuer.issue(&other, node_key()).await,
        Err(IssueError::Forbidden(_))
    ));
    // the daemon's token for another audience: nothing
    let elsewhere = token(&api, "everserve-node", "somewhere-else", Some("daemon-a"));
    assert!(matches!(
        issuer.issue(&elsewhere, node_key()).await,
        Err(IssueError::Unauthenticated(_))
    ));
    // the daemon's account with no pod behind the token: nothing
    let unbound = token(&api, "everserve-node", AUDIENCE, None);
    assert!(matches!(
        issuer.issue(&unbound, node_key()).await,
        Err(IssueError::Forbidden(_))
    ));
}

#[tokio::test]
async fn a_node_is_issued_its_certificate_through_the_operator_api_and_renews_it_in_place() {
    let api = ApiServer::start().await;
    api.kubectl_ok(&["apply", "-f", "-"], Some(&objects()));
    let dir = tempfile::tempdir().expect("tmp");
    let (ca, ca_pem) = deployment_ca(dir.path());
    let issuer = Issuer::new(
        ca,
        ca_pem,
        "store",
        Duration::from_hours(1),
        api.access.clone(),
        NAMESPACE,
        "everserve-node",
    )
    .expect("issuer");
    let coordinator = Tls::load(
        &dir.path().join("coordinator.crt"),
        &dir.path().join("coordinator.key"),
        &dir.path().join("ca.crt"),
    )
    .expect("coordinator tls");
    let cancel = CancellationToken::new();
    // the issuer beside an operator API that admits no one: a node is no operator
    let port = everserve_controller::api::start(
        axum::Router::new().route("/v1/nodes", axum::routing::get(|| async { "[]" })),
        everserve_controller::issuer::router(Arc::new(issuer)),
        "127.0.0.1:0".parse().expect("addr"),
        &coordinator,
        Arc::new(Gates {
            certificates: None,
            kubernetes: None,
        }),
        everserve_controller::api::Deadlines::SERVED,
        cancel.clone(),
    )
    .await
    .expect("bound")
    .port();

    let token_file = dir.path().join("token");
    let out = dir.path().join("node");
    std::fs::create_dir_all(&out).expect("out");
    let enroll = |token: &str| {
        std::fs::write(&token_file, token).expect("token");
        Enrollment::new(
            &format!("127.0.0.1:{port}"),
            "coordinator",
            &dir.path().join("ca.crt"),
            &token_file,
            &out,
        )
        .expect("enrollment")
    };

    let daemon_a = enroll(&token(&api, "everserve-node", AUDIENCE, Some("daemon-a")));
    let first = daemon_a.once().await.expect("issued");
    assert_eq!(first.node, "node-a");
    assert_eq!(
        daemon_a.held(),
        Some(Held {
            node: "node-a".into(),
            not_after_ms: first.not_after_ms,
        }),
        "what was issued is recorded, so a renewing node-cert starts from it"
    );
    let node = Tls::load(
        &out.join(CERT_FILE),
        &out.join(KEY_FILE),
        &out.join(CA_FILE),
    )
    .expect("the files are TLS material");
    assert!(serves(&node, "node-a").await);
    assert!(!serves(&node, "node-b").await);

    // renewal: a new key and its certificate, replacing the files a process already reads
    let key_before = std::fs::read(out.join(KEY_FILE)).expect("key");
    daemon_a.once().await.expect("renewed");
    assert_ne!(std::fs::read(out.join(KEY_FILE)).expect("key"), key_before);
    assert!(
        node.reload().expect("reload"),
        "the renewal is read in place"
    );
    assert!(serves(&node, "node-a").await);

    // another service account's token: refused, and the files in place stay
    let other = enroll(&token(&api, "other", AUDIENCE, Some("other-a")));
    assert!(matches!(
        other.once().await,
        Err(EnrollError::Refused { status: 403, .. })
    ));
    assert!(!node.reload().expect("reload"), "nothing was written");
    cancel.cancel();
}

#[tokio::test]
async fn the_open_endpoint_bounds_what_a_caller_can_make_it_do() {
    // outside the operator API's gate: a body too large, a bearer that is no token, and a caller
    // past the reviews in flight are each answered at once, never sent to the API server
    let api = ApiServer::start().await;
    let dir = tempfile::tempdir().expect("tmp");
    let (ca, ca_pem) = deployment_ca(dir.path());
    let issuer = Issuer::new(
        ca,
        ca_pem,
        "store",
        Duration::from_hours(1),
        api.access.clone(),
        NAMESPACE,
        "everserve-node",
    )
    .expect("issuer")
    // none may be in flight: a well-formed request is told to come back
    .reviews_at_once(0);
    let coordinator = Tls::load(
        &dir.path().join("coordinator.crt"),
        &dir.path().join("coordinator.key"),
        &dir.path().join("ca.crt"),
    )
    .expect("coordinator tls");
    let cancel = CancellationToken::new();
    let port = everserve_controller::api::start(
        axum::Router::new().route("/v1/nodes", axum::routing::get(|| async { "[]" })),
        everserve_controller::issuer::router(Arc::new(issuer)),
        "127.0.0.1:0".parse().expect("addr"),
        &coordinator,
        Arc::new(Gates {
            certificates: None,
            kubernetes: None,
        }),
        everserve_controller::api::Deadlines::SERVED,
        cancel.clone(),
    )
    .await
    .expect("bound")
    .port();
    let roots = everserve_wire::transport::root_store(&dir.path().join("ca.crt")).expect("ca");
    let url = format!(
        "https://127.0.0.1:{port}{}",
        everserve_controller::issuer::PATH
    );
    let ask = |bearer: &str, body: String| {
        let client = everserve_controller::http::Client::with(
            Some(Arc::clone(&roots)),
            Some(everserve_controller::http::Bearer::Token(bearer.to_owned())),
        )
        .server_name("coordinator");
        let url = url.clone();
        async move {
            client
                .send(
                    everserve_controller::http::Method::POST,
                    &url,
                    &[("content-type", "application/json")],
                    Some(body.into_bytes()),
                )
                .await
                .expect("answered")
        }
    };
    // a well-formed key: an uncompressed P-256 point, as a node sends
    let point = rcgen::KeyPair::generate()
        .expect("key")
        .public_key_raw()
        .to_vec();
    let key = format!(
        r#"{{"publicKey":"{}"}}"#,
        everserve_controller::issuer::hex(&point)
    );
    let token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0.c2ln";
    let large = format!(r#"{{"publicKey":"{}"}}"#, "ab".repeat(4096));
    assert_eq!(ask(token, large).await.status, 413, "a body over the bound");
    assert_eq!(
        ask("not-a-token", key.clone()).await.status,
        401,
        "no token"
    );
    let busy = ask(token, key).await;
    assert_eq!(
        busy.status, 503,
        "past the reviews in flight: {}",
        busy.body
    );
    cancel.cancel();
}
