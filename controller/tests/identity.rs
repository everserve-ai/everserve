// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Decision 0026: a daemon is the node its certificate names. Over TLS, a hello for the
//! certificate's node links; a hello for another node is closed.

mod common;

use std::path::Path;
use std::sync::Arc;

use everserve_controller::certs;
use everserve_controller::config::CertsArgs;
use everserve_controller::coord::Coordinator;
use everserve_controller::policy;
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::proto::everserve::coord::v1::{
    CoordToDaemon, DaemonHello, DaemonToCoord, daemon_to_coord,
};
use everserve_wire::transport::{Address, Listener, connect};
use tokio_util::sync::CancellationToken;

/// Whether a daemon holding `cert` is linked when it says it is `node`.
async fn linked_as(dir: &Path, address: &Address, cert: &str, node: &str) -> bool {
    let mut stream = connect(address, Some(&common::tls(dir, cert)), "coordinator")
        .await
        .expect("connect");
    let hello = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::Hello(DaemonHello {
            node: node.into(),
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            heartbeat_ms: 0,
            ..Default::default()
        })),
    };
    frame::send(&mut stream, &hello).await.expect("send");
    // a refused hello is closed at once; a kept link stays open with nothing to say
    let next = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        frame::recv::<_, CoordToDaemon>(&mut stream),
    )
    .await;
    !matches!(next, Ok(Err(_)))
}

#[tokio::test]
async fn a_daemon_is_the_node_its_certificate_names() {
    let dir = tempfile::tempdir().expect("tmp");
    certs::write(&CertsArgs {
        out: dir.path().to_path_buf(),
        nodes: vec!["node-a".into(), "node-b".into()],
        operators: vec![],
        coordinator_name: "coordinator".into(),
        store_name: "store".into(),
        days: 30,
        overwrite: false,
    })
    .expect("certs");
    let policy_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-rig-4gpu.json");
    let telemetry = Telemetry::new();
    let (coord, _rx) = Coordinator::new(policy::load(&policy_path).expect("policy"), &telemetry);
    let listener = Listener::bind(
        &Address::parse("127.0.0.1:0").unwrap(),
        Some(&common::tls(dir.path(), "coordinator")),
    )
    .await
    .expect("bind");
    let address = listener.local_address().expect("address");
    let cancel = CancellationToken::new();
    tokio::spawn(Arc::clone(&coord).serve(listener, cancel.clone()));

    assert!(linked_as(dir.path(), &address, "node-a", "node-a").await);
    assert!(
        !linked_as(dir.path(), &address, "node-a", "node-b").await,
        "node-a's certificate claimed node-b and was kept"
    );
    assert!(linked_as(dir.path(), &address, "node-b", "node-b").await);
    cancel.cancel();
}
