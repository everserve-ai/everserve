// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The daemon link seen from the coordinator (decision 0023): a daemon lost without a goodbye
//! loses its ranks as one change, one that misses its heartbeats is lost and one that says
//! goodbye is not, one that relinks before its old link ends is not lost, and a notice its hello
//! carries is armed once.

mod common;

use std::sync::Arc;

use everserve_controller::node::Daemon;
use everserve_controller::proto::everserve::signals::v1::{
    Ack, Notice, Reason, Role, Signal, signal,
};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use tokio_util::sync::CancellationToken;

use common::cluster::{cluster, expect_change, fake_daemon, fake_daemon_with_notices, wait_view};

#[tokio::test]
async fn a_daemon_lost_without_a_goodbye_loses_its_ranks_as_one_change() {
    // decision 0023: node "far" holds ranks 2 and 3 through a daemon that drops its link; the
    // real daemon on node-a holds ranks 0 and 1 and the spares for 2 and 3
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let mut s2 = common::register(&sock_a, 2, 2, Role::Spare).await;
    let mut s3 = common::register(&sock_a, 3, 3, Role::Spare).await;
    let far = fake_daemon(&cl, "far", vec![2, 3], 0).await;
    common::wait_members(&cl.coord, 4).await;
    drop(far); // the link closes with no goodbye: the node is lost at once
    let c = expect_change(&mut w0, 0, 1, 1).await;
    assert_eq!(c.lost, vec![2, 3]);
    expect_change(&mut w1, 1, 1, 1).await;
    expect_change(&mut s2, 2, 1, 1).await;
    expect_change(&mut s3, 3, 1, 1).await;
    for _ in 0..300 {
        if cl.coord.engine("k3-a").await.is_some_and(|v| {
            (2..4).all(|r| v.members.get(&r).map(String::as_str) == Some("node-a"))
        }) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.epoch, 1);
    assert!(
        (2..4).all(|r| view.members.get(&r).map(String::as_str) == Some("node-a")),
        "{view:?}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_daemon_that_misses_its_heartbeats_is_lost_and_one_that_says_goodbye_is_not() {
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let mut s2 = common::register(&sock_a, 2, 2, Role::Spare).await;
    let mut s3 = common::register(&sock_a, 3, 3, Role::Spare).await;
    // a daemon that promised a heartbeat every 50 ms and sends none: lost after three misses
    let silent = fake_daemon(&cl, "far", vec![2, 3], 50).await;
    common::wait_members(&cl.coord, 4).await;
    let t0 = std::time::Instant::now();
    let c = expect_change(&mut w0, 0, 1, 1).await;
    assert_eq!(c.lost, vec![2, 3]);
    let took = t0.elapsed();
    assert!(
        took >= std::time::Duration::from_millis(120) && took < std::time::Duration::from_secs(2),
        "{took:?}"
    );
    expect_change(&mut w1, 1, 1, 1).await;
    expect_change(&mut s2, 2, 1, 1).await;
    expect_change(&mut s3, 3, 1, 1).await;
    drop(silent);

    // a real daemon stopped on purpose says goodbye: its ranks are departed, not lost
    let cancel_b = CancellationToken::new();
    let telemetry = Telemetry::new();
    let (db, sock_b) = {
        let (d, _rx) = Daemon::new("node-b", &telemetry);
        let sock = cl.dir.path().join("node-b.sock");
        tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel_b.clone()));
        tokio::spawn(Arc::clone(&d).link(
            cl.address.clone(),
            None,
            "coordinator".into(),
            cancel_b.clone(),
        ));
        for _ in 0..200 {
            if sock.exists() && d.linked().await {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        (d, sock)
    };
    assert!(db.linked().await);
    let _w_b = common::register(&sock_b, 3, 0, Role::Worker).await; // rank 3 now serves from node-b
    for _ in 0..300 {
        if cl
            .coord
            .engine("k3-a")
            .await
            .is_some_and(|v| v.members.get(&3).map(String::as_str) == Some("node-b"))
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let epoch_before = cl.coord.engine("k3-a").await.expect("view").epoch;
    cancel_b.cancel();
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(
        view.epoch, epoch_before,
        "a goodbye is not a loss: {view:?}"
    );
    assert!(view.departed.contains(&3), "{view:?}");
    assert!(!cl.coord.daemons().await.iter().any(|n| n == "node-b"));
    cl.cancel.cancel();
}

/// Answer every delivery `link` is sent with every rank it names acknowledged, until the link
/// ends: what a daemon whose processes all acknowledge does.
fn answer_deliveries(mut link: everserve_wire::transport::Stream) {
    use everserve_controller::proto::everserve::coord::v1::{
        CoordToDaemon, DaemonToCoord, Delivered, coord_to_daemon, daemon_to_coord,
    };
    tokio::spawn(async move {
        while let Ok(msg) = frame::recv::<_, CoordToDaemon>(&mut link).await {
            let Some(coord_to_daemon::Kind::Deliver(d)) = msg.kind else {
                continue;
            };
            let epoch = d.change.as_ref().map_or(0, |c| c.epoch);
            let acks = d
                .to_ranks
                .iter()
                .map(|&rank| Ack {
                    rank,
                    epoch,
                    error: String::new(),
                    received_at_us: 0,
                })
                .collect();
            let delivered = DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::Delivered(Delivered {
                    request_id: d.request_id,
                    engine: d.engine,
                    epoch,
                    acks,
                    error: String::new(),
                })),
            };
            if frame::send(&mut link, &delivered).await.is_err() {
                return;
            }
        }
    });
}

#[tokio::test]
async fn a_daemon_that_relinks_before_its_old_link_ends_is_not_lost() {
    // a daemon that notices a broken link relinks within a second; the coordinator's reader of
    // the old link times out later. The new link replaces the old: when the old one ends,
    // nothing of the node is taken as lost
    use everserve_controller::proto::everserve::coord::v1::{
        DaemonToCoord, WorkerEvent, daemon_to_coord, worker_event,
    };
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let mut s2 = common::register(&sock_a, 2, 2, Role::Spare).await;
    let _s3 = common::register(&sock_a, 3, 3, Role::Spare).await;
    let old = fake_daemon(&cl, "far", vec![2, 3], 0).await;
    common::wait_members(&cl.coord, 4).await;
    let mut new = fake_daemon(&cl, "far", vec![2, 3], 0).await;
    // something only the new link can have said, so its hello is known to be taken before the
    // old link closes
    let spare = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::WorkerEvent(WorkerEvent {
            engine: "k3-a".into(),
            kind: worker_event::Kind::SpareRegistered as i32,
            rank: 0,
            world_size: 4,
            device: 7,
            identity: Some(common::identity()),
            ..WorkerEvent::default()
        })),
    };
    frame::send(&mut new, &spare).await.expect("spare");
    wait_view(&cl.coord, "the new link's spare", |v| {
        v.spares.contains_key(&0)
    })
    .await;
    drop(old); // the old link closes with no goodbye, as a broken one does
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.epoch, 0, "no change was driven: {view:?}");
    assert!(
        (2..4).all(|r| view.members.get(&r).map(String::as_str) == Some("far")),
        "{view:?}"
    );
    let daemons = cl.coord.daemons().await;
    assert!(daemons.iter().any(|n| n == "far"), "{daemons:?}");
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(300), common::recv(&mut s2))
            .await
            .is_err(),
        "the spare was sent nothing"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_notice_a_hello_carries_is_armed_once_and_its_change_runs_at_the_deadline() {
    // a coordinator that starts or takes the lease learns a standing notice from the daemon's
    // hello and arms its deadline; the same hello again (a relink) forwards the notice again and
    // arms nothing twice
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let mut s2 = common::register(&sock_a, 2, 2, Role::Spare).await;
    let mut s3 = common::register(&sock_a, 3, 3, Role::Spare).await;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
    let notice = Notice {
        rank: 2,
        deadline_ms: now_ms + 600,
        reason: Reason::Notice as i32,
    };
    let far = fake_daemon_with_notices(&cl, "far", vec![2, 3], 0, vec![notice]).await;
    common::wait_members(&cl.coord, 4).await;
    answer_deliveries(far);
    let far_again = fake_daemon_with_notices(&cl, "far", vec![2, 3], 0, vec![notice]).await;
    answer_deliveries(far_again);
    // every process heard the notice once per hello; at the deadline, one change for rank 2,
    // delivered to the survivors and to the spare that takes it
    for (w, rank) in [(&mut w0, 0), (&mut w1, 1), (&mut s2, 2)] {
        let lost = common::expect_notices_then_change(w, rank, 2, 1, Reason::Notice).await;
        assert_eq!(lost, vec![2], "rank {rank}");
    }
    wait_view(&cl.coord, "the commit", |v| v.epoch == 1).await;
    // the spare for rank 3, not lost, heard the two notices and nothing else
    for _ in 0..2 {
        let sig: Signal = common::recv(&mut s3).await.expect("notice");
        assert!(
            matches!(sig.kind, Some(signal::Kind::Notice(ref n)) if n.rank == 2),
            "{sig:?}"
        );
    }
    // and no second change: the deadline was armed once
    for w in [&mut w0, &mut w1] {
        loop {
            match tokio::time::timeout(std::time::Duration::from_millis(700), common::recv(w)).await
            {
                Err(_) => break,
                Ok(Ok(Signal {
                    kind: Some(signal::Kind::MembershipCommitted(m)),
                })) => assert_eq!(m.epoch, 1),
                Ok(other) => panic!("a second change: {other:?}"),
            }
        }
    }
    cl.cancel.cancel();
}
