// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A store that cannot start as asked stops at startup with a usage failure: one that cannot bind
//! its metrics port, rather than running without its metrics and health; one whose budget its data
//! directory's file system has no room for, rather than dying of a SIGBUS at a commit.

use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn a_taken_metrics_port_stops_the_store() {
    let dir = tempfile::tempdir().expect("tmp");
    let taken = TcpListener::bind("127.0.0.1:0").expect("bind");
    let metrics = taken.local_addr().expect("addr").to_string();
    let mut store = Command::new(env!("CARGO_BIN_EXE_everserve-store"))
        .arg("--socket")
        .arg(dir.path().join("store.sock"))
        .arg("--data-dir")
        .arg(dir.path().join("data"))
        .args(["--slots", "16", "--budget-bytes", "1048576"])
        .args(["--metrics-addr", &metrics])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the store");
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = store.try_wait().expect("wait") {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = store.kill();
            let _ = store.wait();
            panic!("the store is still running with its metrics port taken");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(2));
}

/// The store on `data`, with `budget` bytes of tables, and its socket.
fn store_on(dir: &std::path::Path, budget: u64) -> (std::process::Child, std::path::PathBuf) {
    let socket = dir.join("store.sock");
    let store = Command::new(env!("CARGO_BIN_EXE_everserve-store"))
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir.join("data"))
        .args(["--slots", "16", "--budget-bytes", &budget.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the store");
    (store, socket)
}

/// The store's exit code, or None once it serves its socket; bounded.
fn exited_or_served(store: &mut std::process::Child, socket: &std::path::Path) -> Option<i32> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = store.try_wait().expect("wait") {
            return status.code();
        }
        if socket.exists() {
            let _ = store.kill();
            let _ = store.wait();
            return None;
        }
        assert!(
            Instant::now() < deadline,
            "the store neither served nor exited"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_budget_its_file_system_has_no_room_for_stops_the_store_and_an_earlier_runs_tables_count() {
    let dir = tempfile::tempdir().expect("tmp");
    std::fs::create_dir_all(dir.path().join("data")).expect("data");
    let free = everserve_wire::host::free_bytes(dir.path()).expect("free");
    let over = free + (1 << 30);

    let (mut store, socket) = store_on(dir.path(), over);
    assert_eq!(
        exited_or_served(&mut store, &socket),
        Some(2),
        "no room: refused"
    );

    // an earlier run's table, which this run makes again in place: the room is its
    let rank = dir.path().join("data").join("e").join("rank0");
    std::fs::create_dir_all(&rank).expect("rank");
    let table = std::fs::File::create(rank.join("layer0")).expect("table");
    table.set_len(2 << 30).expect("sparse");
    let (mut store, socket) = store_on(dir.path(), over);
    assert_eq!(
        exited_or_served(&mut store, &socket),
        None,
        "room with its tables: serves"
    );
}
