// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The daemon's spare supervisor (decision 0024): a launched spare runs with the placeholders
//! filled in, a second launch of the same spare is a no-op, a spare that exits is launched again
//! after a backoff, and stopping the supervisor stops the spares.

mod common;

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use everserve_controller::node::spares::{RESTART_MIN, Supervisor};
use everserve_controller::proto::everserve::coord::v1::LaunchSpare;
use tokio_util::sync::CancellationToken;

fn spec(rank: u32, command: &[&str]) -> LaunchSpare {
    LaunchSpare {
        engine: "k3-a".into(),
        rank: Some(rank),
        device: rank % 4,
        command: command.iter().map(|s| (*s).to_owned()).collect(),
        env: HashMap::from([("EVERSERVE_RANK".to_owned(), "{rank}".to_owned())]),
        pool: String::new(),
        engines: vec!["k3-a".into()],
    }
}

#[tokio::test]
async fn a_spare_runs_with_its_placeholders_filled_and_stops_with_the_supervisor() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let sup = Supervisor::new("node-c", Path::new("/run/x/node.sock"), cancel.clone());
    let out = dir.path().join("spare-{rank}.txt");
    // the spare writes what it was given, then stays up
    let script = format!(
        "echo \"$EVERSERVE_RANK {{rank}} {{device}} {{node}} {{socket}}\" > {} && exec sleep 60",
        out.display()
    );
    sup.launch(spec(5, &["sh", "-c", &script]))
        .await
        .expect("launch");
    sup.launch(spec(5, &["sh", "-c", "exit 7"]))
        .await
        .expect("the same spare again: a no-op");
    let written = dir.path().join("spare-5.txt");
    for _ in 0..300 {
        if written.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let text = std::fs::read_to_string(&written).expect("the spare ran");
    assert_eq!(text.trim(), "5 5 1 node-c /run/x/node.sock");
    let launched = sup.launched().await;
    assert_eq!(launched.len(), 1);
    assert!(launched[0].pid > 0 && common::alive(launched[0].pid));
    assert!(
        sup.launch(spec(6, &[])).await.is_err(),
        "an empty command is refused"
    );
    let pid = launched[0].pid;
    sup.stop_all().await;
    for _ in 0..300 {
        if !common::alive(pid) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(!common::alive(pid), "the spare stopped with the supervisor");
    assert!(sup.launched().await.is_empty());
}

#[tokio::test]
async fn a_spare_that_exits_is_launched_again_after_the_backoff() {
    let cancel = CancellationToken::new();
    let sup = Supervisor::new("node-c", Path::new("/run/x/node.sock"), cancel.clone());
    sup.launch(spec(4, &["sh", "-c", "exit 0"]))
        .await
        .expect("launch");
    tokio::time::sleep(RESTART_MIN + Duration::from_millis(400)).await;
    let launched = sup.launched().await;
    assert!(launched[0].restarts >= 1, "{launched:?}");
    cancel.cancel();
}

#[tokio::test]
async fn a_pools_spare_is_its_own_slot_beside_an_engines_on_one_device() {
    // decision 0031: a pool's spare names its pool, not an engine
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let sup = Supervisor::new("node-e", Path::new("/run/x/node.sock"), cancel.clone());
    let out = dir.path().join("pool.txt");
    let pool = |command: &[&str]| LaunchSpare {
        engine: String::new(),
        rank: None,
        device: 1,
        command: command.iter().map(|s| (*s).to_owned()).collect(),
        env: HashMap::new(),
        pool: "h100".into(),
        engines: vec!["k3-a".into(), "k3-b".into()],
    };
    let script = format!(
        "echo \"{{pool}} {{engines}} {{device}}\" > {} && exec sleep 60",
        out.display()
    );
    sup.launch(pool(&["sh", "-c", &script]))
        .await
        .expect("launch");
    sup.launch(pool(&["sh", "-c", "exit 7"]))
        .await
        .expect("the same pool's spare again: a no-op");
    // an engine's own spare on the same device is another slot
    sup.launch(LaunchSpare {
        device: 1,
        ..spec(1, &["sh", "-c", "exec sleep 60"])
    })
    .await
    .expect("launch");
    for _ in 0..300 {
        if out.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        std::fs::read_to_string(&out).expect("ran").trim(),
        "h100 k3-a,k3-b 1"
    );
    let launched = sup.launched().await;
    assert_eq!(launched.len(), 2, "{launched:?}");
    let p = launched
        .iter()
        .find(|l| l.pool == "h100")
        .expect("the pool's");
    assert!(p.engine.is_empty() && p.rank.is_none());
    sup.stop_all().await;
}
