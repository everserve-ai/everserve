// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The material `certs` writes is what the processes verify each other by: a daemon reaches the
//! coordinator by its name, a store reaches a peer store by the store name on the node's
//! certificate, a wrong name is refused, and a second run reuses the CA for a new node.

mod common;

use std::path::Path;

use everserve_controller::certs;
use everserve_controller::config::CertsArgs;
use everserve_wire::transport::{Address, Listener, Tls, connect};

fn args(out: &Path, nodes: &[&str]) -> CertsArgs {
    CertsArgs {
        out: out.to_path_buf(),
        nodes: nodes.iter().map(|n| (*n).to_owned()).collect(),
        operators: vec![],
        coordinator_name: "coordinator".into(),
        store_name: "store".into(),
        days: 30,
        overwrite: false,
    }
}

/// Whether `client` reaches a listener under `server` when it asks for `server_name`.
async fn reaches(server: Tls, client: Tls, server_name: &str) -> bool {
    let listener = Listener::bind(&Address::parse("127.0.0.1:0").unwrap(), Some(&server))
        .await
        .expect("bind");
    let address = listener.local_address().expect("address");
    let accept = tokio::spawn(async move { listener.accept().await.is_ok() });
    let connected = connect(&address, Some(&client), server_name).await.is_ok();
    let accepted = tokio::time::timeout(std::time::Duration::from_secs(2), accept)
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or(false);
    connected && accepted
}

#[tokio::test]
async fn the_written_material_lets_each_role_verify_the_other_by_name() {
    let dir = tempfile::tempdir().expect("tmp");
    let written = certs::write(&args(dir.path(), &["node-a", "node-b"])).expect("write");
    assert!(!written.ca_reused);
    let names: Vec<String> = written
        .files
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        [
            "ca.crt",
            "ca.key",
            "coordinator.crt",
            "coordinator.key",
            "node-a.crt",
            "node-a.key",
            "node-b.crt",
            "node-b.key"
        ]
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(dir.path().join("node-a.key"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "a key is the owner's only");
    }
    // a daemon on node-a reaches the coordinator by its name
    assert!(
        reaches(
            common::tls(dir.path(), "coordinator"),
            common::tls(dir.path(), "node-a"),
            "coordinator"
        )
        .await
    );
    // node-b's store reaches node-a's store by the store name on node-a's certificate
    assert!(
        reaches(
            common::tls(dir.path(), "node-a"),
            common::tls(dir.path(), "node-b"),
            "store"
        )
        .await
    );
    // and by the node's own name
    assert!(
        reaches(
            common::tls(dir.path(), "node-a"),
            common::tls(dir.path(), "node-b"),
            "node-a"
        )
        .await
    );
    // a wrong name is refused
    assert!(
        !reaches(
            common::tls(dir.path(), "node-a"),
            common::tls(dir.path(), "node-b"),
            "coordinator"
        )
        .await
    );
    assert!(
        !reaches(
            common::tls(dir.path(), "coordinator"),
            common::tls(dir.path(), "node-a"),
            "store"
        )
        .await
    );
}

#[tokio::test]
async fn a_second_run_reuses_the_ca_and_keeps_existing_files_unless_told_to_overwrite() {
    let dir = tempfile::tempdir().expect("tmp");
    certs::write(&args(dir.path(), &["node-a"])).expect("write");
    let ca_before = std::fs::read(dir.path().join("ca.crt")).unwrap();
    let coordinator_before = std::fs::read(dir.path().join("coordinator.crt")).unwrap();
    let written = certs::write(&args(dir.path(), &["node-c"])).expect("write again");
    assert!(written.ca_reused);
    assert_eq!(
        std::fs::read(dir.path().join("ca.crt")).unwrap(),
        ca_before,
        "the CA is kept"
    );
    assert_eq!(
        std::fs::read(dir.path().join("coordinator.crt")).unwrap(),
        coordinator_before,
        "an existing certificate is kept: a running coordinator holds it"
    );
    assert_eq!(written.kept.len(), 2, "{written:?}");
    assert!(
        written
            .files
            .iter()
            .all(|p| p.to_string_lossy().contains("node-c")),
        "{written:?}"
    );
    let mut again = args(dir.path(), &["node-a"]);
    again.overwrite = true;
    let written = certs::write(&again).expect("overwrite");
    assert!(written.kept.is_empty());
    assert_ne!(
        std::fs::read(dir.path().join("coordinator.crt")).unwrap(),
        coordinator_before,
        "--overwrite reissues under the same CA"
    );
    // node-a's old certificate and node-c's new one verify each other under the kept CA
    assert!(
        reaches(
            common::tls(dir.path(), "node-a"),
            common::tls(dir.path(), "node-c"),
            "store"
        )
        .await
    );
    assert!(
        reaches(
            common::tls(dir.path(), "coordinator"),
            common::tls(dir.path(), "node-a"),
            "coordinator"
        )
        .await
    );
    assert!(
        certs::write(&args(dir.path(), &[])).is_err(),
        "no node named"
    );
}
