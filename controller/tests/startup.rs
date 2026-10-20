// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A process that cannot bind a port it was given stops at startup with a usage failure, rather
//! than running without its metrics, its health or its operator API.

mod common;

use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use everserve_controller::config::CertsArgs;

use common::Spawned;

/// The binary with `args`, its output discarded.
fn controller(args: &[&str]) -> Spawned {
    Command::new(env!("CARGO_BIN_EXE_everserve-controller"))
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(Spawned)
        .expect("spawn the controller")
}

/// The exit code of `child`, which must exit within ten seconds.
fn exit_code(mut child: Spawned) -> Option<i32> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.0.try_wait().expect("wait") {
            return status.code();
        }
        assert!(
            Instant::now() < deadline,
            "still running with its port taken"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn policy() -> String {
    common::example("policy-two-node.json")
        .display()
        .to_string()
}

fn unix(dir: &Path, name: &str) -> String {
    format!("unix:{}", dir.join(name).display())
}

#[test]
fn a_taken_metrics_port_stops_the_daemon_and_the_coordinator() {
    let dir = tempfile::tempdir().expect("tmp");
    let taken = TcpListener::bind("127.0.0.1:0").expect("bind");
    let metrics = taken.local_addr().expect("addr").to_string();
    let socket = dir.path().join("daemon.sock").display().to_string();
    let daemon = controller(&["node", "--socket", &socket, "--metrics-addr", &metrics]);
    assert_eq!(exit_code(daemon), Some(2), "the daemon");
    let coordinator = controller(&[
        "coord",
        "--policy",
        &policy(),
        "--listen",
        &unix(dir.path(), "coord.sock"),
        "--metrics-addr",
        &metrics,
    ]);
    assert_eq!(exit_code(coordinator), Some(2), "the coordinator");
}

#[test]
fn a_taken_api_port_stops_the_coordinator() {
    let dir = tempfile::tempdir().expect("tmp");
    let certs = dir.path().join("certs");
    everserve_controller::certs::write(&CertsArgs {
        out: certs.clone(),
        nodes: Vec::new(),
        operators: vec!["operator".into()],
        coordinator_name: "localhost".into(),
        store_name: "store".into(),
        days: 1,
        overwrite: false,
    })
    .expect("certs");
    let taken = TcpListener::bind("127.0.0.1:0").expect("bind");
    let api = taken.local_addr().expect("addr").to_string();
    let file = |name: &str| certs.join(name).display().to_string();
    let coordinator = controller(&[
        "coord",
        "--policy",
        &policy(),
        "--listen",
        &unix(dir.path(), "coord.sock"),
        "--tls-cert",
        &file("coordinator.crt"),
        "--tls-key",
        &file("coordinator.key"),
        "--tls-ca",
        &file("ca.crt"),
        "--api-addr",
        &api,
        "--api-operators",
        "operator",
    ]);
    assert_eq!(exit_code(coordinator), Some(2));
}
