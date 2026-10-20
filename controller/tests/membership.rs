// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The coordinator over node daemons, all on one machine: a change delivered to every node that
//! holds a survivor and committed to every daemon of the engine; the spare promoted on its node;
//! clients hearing the commit; a stale-epoch delivery refused; a report held across a missing
//! coordinator and delivered when the link comes up; a coordinator restart that resumes at the
//! right epoch; a notice forwarded and driven at its deadline; the daemon's record published; a
//! spare on a slower class refused by a policy that allows no mismatch and admitted by one that
//! does; the spare node's daemon launching the policy's spares; a change that fails after its
//! delivery putting the engine down; and the commit reaching every rank it places.

mod common;

use std::path::Path;
use std::sync::Arc;

use everserve_controller::coord::{Cause, ChangeError, Coordinator};
use everserve_controller::discovery::{Discovery, FileDiscovery};
use everserve_controller::node::Daemon;
use everserve_controller::policy;
use everserve_controller::proto::everserve::coord::v1::Deliver;
use everserve_controller::proto::everserve::signals::v1::{
    Abandoned, Ack, Change, MembershipCommitted, Notice, Outcome, RankLost, Reason, Role, Signal,
    signal,
};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::Address;
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

use common::cluster::{
    ack, cluster, expect_change, fake_daemon, next_change, node_with_devices, wait_view, within,
};

/// The next signal on `w` that is an abandon (decision 0032).
async fn expect_abandoned(w: &mut UnixStream) -> Abandoned {
    let next = async {
        loop {
            let sig: Signal = common::recv(w).await.expect("abandoned");
            if let Some(signal::Kind::Abandoned(a)) = sig.kind {
                return a;
            }
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(5), next)
        .await
        .expect("no abandon reached the process")
}

async fn coordinator(
    policy_file: &str,
    dir: &Path,
) -> (Arc<Coordinator>, Telemetry, Address, CancellationToken) {
    let telemetry = Telemetry::new();
    let cancel = CancellationToken::new();
    let policy = policy::load(&common::example(policy_file)).expect("policy");
    let (coord, address) = common::coordinator(policy, dir, &telemetry, &cancel).await;
    (coord, telemetry, address, cancel)
}

#[tokio::test]
async fn a_change_crosses_nodes_and_the_spare_on_the_other_node_is_promoted() {
    let test_started_us = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros(),
    )
    .unwrap();
    let cl = cluster("policy-rig-4gpu.json").await;
    let (da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let (db, sock_b) =
        common::daemon(cl.dir.path(), "node-b", &cl.address, Vec::new(), &cl.cancel).await;
    // ranks 0 and 1 on node a; ranks 2 and 3 and the spare for rank 2 on node b
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let w2 = common::register(&sock_b, 2, 2, Role::Worker).await;
    let mut w3 = common::register(&sock_b, 3, 3, Role::Worker).await;
    let mut spare = common::register(&sock_b, 2, 2, Role::Spare).await;
    let mut client_a = common::register(&sock_a, 0, 0, Role::Client).await;
    common::wait_members(&cl.coord, 4).await;
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.members.get(&2).map(String::as_str), Some("node-b"));
    assert_eq!(view.spares.get(&2).map(String::as_str), Some("node-b"));

    // rank 0 on node a reports rank 2 lost
    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![2],
            reason: Reason::Fault as i32,
            detail: "process exited".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    drop(w2);
    // every survivor on both nodes and the spare receive the change at epoch 1
    let c = expect_change(&mut w0, 0, 1, 1).await;
    assert_eq!(
        (c.lost.clone(), c.outcome),
        (vec![2], Outcome::Replace as i32)
    );
    expect_change(&mut w1, 1, 1, 1).await;
    expect_change(&mut w3, 3, 1, 1).await;
    expect_change(&mut spare, 2, 1, 1).await;
    // the client on node a hears the change as it is claimed (decision 0023), then the commit;
    // both daemons and the coordinator are at epoch 1
    let sig: Signal = common::recv(&mut client_a).await.expect("change");
    assert!(
        matches!(sig.kind, Some(signal::Kind::Change(ref c)) if c.lost == vec![2] && c.epoch == 1),
        "{sig:?}"
    );
    let sig: Signal = common::recv(&mut client_a).await.expect("committed");
    let Some(signal::Kind::MembershipCommitted(m)) = sig.kind else {
        panic!("{sig:?}")
    };
    assert_eq!(
        (m.epoch, m.active_ranks.clone(), m.world_size),
        (1, vec![0, 1, 2, 3], 4)
    );
    // the change's moments on the coordinator's clock, in order and inside this test's own
    let now_us = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros(),
    )
    .unwrap();
    assert!(
        test_started_us <= m.reported_at_us
            && m.reported_at_us <= m.delivered_at_us
            && m.delivered_at_us <= m.acknowledged_at_us
            && m.acknowledged_at_us <= m.committed_at_us
            && m.committed_at_us <= now_us,
        "stamps out of order or off the clock: {m:?}"
    );
    assert_eq!(
        c.delivered_at_us, m.delivered_at_us,
        "the Change carries the delivery stamp"
    );
    assert_acks(&m, &[0, 1, 2, 3]);
    // the commit reaches both daemons after the acknowledgements; under a full parallel run it
    // can take a while
    for _ in 0..500 {
        if da.epoch("k3-a").await == 1
            && db.epoch("k3-a").await == 1
            && db.registered("k3-a").await == vec![2, 3]
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!((da.epoch("k3-a").await, db.epoch("k3-a").await), (1, 1));
    assert_eq!(
        db.registered("k3-a").await,
        vec![2, 3],
        "the spare now serves as rank 2"
    );
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.epoch, 1);
    assert_eq!(view.members.get(&2).map(String::as_str), Some("node-b"));
    assert!(view.spares.is_empty());
    let text = cl.coord_telemetry.render();
    assert!(
        text.contains("everserve_changes_total{cause=\"fault\",engine=\"k3-a\",outcome=\"replace\",result=\"committed\"} 1"),
        "{text}"
    );
    assert!(text.contains("everserve_daemons_connected 2"), "{text}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_node_new_to_the_engine_after_a_commit_is_brought_to_its_epoch_before_the_next_change() {
    // the worker-loss e2e's second loss: a spare standing on a node whose daemon heard none of
    // the engine's commits refused the next change as stale, and the engine went down. The
    // node's first process of the engine brings its daemon the membership as it stands; its
    // clients hear none of it, since they claimed no change
    let cl = cluster("policy-rig-4gpu.json").await;
    let (da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let (_db, sock_b) =
        common::daemon(cl.dir.path(), "node-b", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let w2 = common::register(&sock_b, 2, 2, Role::Worker).await;
    let mut w3 = common::register(&sock_b, 3, 3, Role::Worker).await;
    let mut spare = common::register(&sock_b, 2, 2, Role::Spare).await;
    common::wait_members(&cl.coord, 4).await;
    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![2],
            reason: Reason::Fault as i32,
            detail: "process exited".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    drop(w2);
    expect_change(&mut w0, 0, 1, 1).await;
    expect_change(&mut w1, 1, 1, 1).await;
    expect_change(&mut w3, 3, 1, 1).await;
    expect_change(&mut spare, 2, 1, 1).await;
    within("the commit at node-a", async {
        while da.epoch("k3-a").await != 1 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;
    // node-c links holding nothing of the engine; a client of it there, then a spare for rank 2
    let (dc, sock_c) =
        common::daemon(cl.dir.path(), "node-c", &cl.address, Vec::new(), &cl.cancel).await;
    assert_eq!(dc.epoch("k3-a").await, 0);
    let mut client_c = common::register(&sock_c, 0, 0, Role::Client).await;
    let _standing = common::register(&sock_c, 2, 0, Role::Spare).await;
    within("node-c at the engine's epoch", async {
        while dc.epoch("k3-a").await != 1 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;
    let heard = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        common::recv(&mut client_c),
    )
    .await;
    assert!(
        !matches!(
            heard,
            Ok(Ok(Signal {
                kind: Some(signal::Kind::MembershipCommitted(_))
            }))
        ),
        "a client heard a commit it claimed no change for: {heard:?}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_delivery_for_an_epoch_that_is_not_the_next_is_refused() {
    let telemetry = Telemetry::new();
    let (d, _rx) = Daemon::new("n", &telemetry);
    let delivered = d
        .deliver(Deliver {
            request_id: 1,
            engine: "k3-a".into(),
            change: Some(Change {
                delivered_at_us: 0,
                take: vec![],
                departing: vec![],
                lost: vec![2],
                outcome: Outcome::Replace as i32,
                epoch: 5,
            }),
            to_ranks: vec![],
            to_spares: vec![],
            take: vec![],
        })
        .await;
    assert!(
        delivered.error.contains("engine k3-a unknown"),
        "{delivered:?}"
    );
    let dir = tempfile::tempdir().expect("tmp");
    let sock = dir.path().join("n.sock");
    let cancel = CancellationToken::new();
    tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel.clone()));
    for _ in 0..100 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let _w = common::register(&sock, 0, 0, Role::Worker).await;
    for _ in 0..100 {
        if !d.registered("k3-a").await.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let delivered = d
        .deliver(Deliver {
            request_id: 2,
            engine: "k3-a".into(),
            change: Some(Change {
                delivered_at_us: 0,
                take: vec![],
                departing: vec![],
                lost: vec![2],
                outcome: Outcome::Replace as i32,
                epoch: 5,
            }),
            to_ranks: vec![0],
            to_spares: vec![],
            take: vec![],
        })
        .await;
    assert!(delivered.error.contains("stale epoch"), "{delivered:?}");
    let delivered = d
        .deliver(Deliver {
            request_id: 3,
            engine: "k3-a".into(),
            change: Some(Change {
                delivered_at_us: 0,
                take: vec![],
                departing: vec![],
                lost: vec![2],
                outcome: Outcome::Replace as i32,
                epoch: 1,
            }),
            to_ranks: vec![7],
            to_spares: vec![],
            take: vec![],
        })
        .await;
    assert!(
        delivered.error.contains("rank 7 is not on this node"),
        "{delivered:?}"
    );
    cancel.cancel();
}

#[tokio::test]
async fn a_report_is_held_without_a_coordinator_and_delivered_when_one_appears() {
    let dir = tempfile::tempdir().expect("tmp");
    let address = Address::Unix(dir.path().join("coord.sock"));
    let cancel = CancellationToken::new();
    // the daemon starts linking before any coordinator listens
    let telemetry = Telemetry::new();
    let (d, _rx) = Daemon::new("node-a", &telemetry);
    let sock = dir.path().join("node-a.sock");
    tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel.clone()));
    tokio::spawn(Arc::clone(&d).link(address.clone(), None, "coordinator".into(), cancel.clone()));
    for _ in 0..100 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let mut w0 = common::register(&sock, 0, 0, Role::Worker).await;
    let w1 = common::register(&sock, 1, 1, Role::Worker).await;
    let mut spare = common::register(&sock, 1, 1, Role::Spare).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(!d.linked().await);
    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![1],
            reason: Reason::Fault as i32,
            detail: "gone".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    // nothing happened yet: no coordinator; now one comes up
    let (coord, _t, _a, coord_cancel) = coordinator("policy-rig-4gpu.json", dir.path()).await;
    for _ in 0..300 {
        if d.linked().await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(d.linked().await, "the daemon never linked");
    // the held report drives the change: rank 1 is replaced by the spare
    let _ = w1;
    expect_change(&mut w0, 0, 1, 1).await;
    expect_change(&mut spare, 1, 1, 1).await;
    for _ in 0..100 {
        if coord.engine("k3-a").await.is_some_and(|v| v.epoch == 1) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(coord.engine("k3-a").await.expect("view").epoch, 1);
    coord_cancel.cancel();
    cancel.cancel();
}

#[tokio::test]
async fn a_restarted_coordinator_resumes_at_the_epoch_the_daemons_hold() {
    let cl = cluster("policy-rig-4gpu.json").await;
    let daemon_cancel = CancellationToken::new(); // outlives the first coordinator
    let (da, sock_a) = common::daemon(
        cl.dir.path(),
        "node-a",
        &cl.address,
        Vec::new(),
        &daemon_cancel,
    )
    .await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    // on device 5, so the rank's holder after the take is told from its holder before
    let mut spare = common::register(&sock_a, 1, 5, Role::Spare).await;
    common::wait_members(&cl.coord, 2).await;
    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![1],
            reason: Reason::Fault as i32,
            detail: "gone".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    expect_change(&mut w0, 0, 1, 1).await;
    expect_change(&mut spare, 1, 1, 1).await;
    for _ in 0..100 {
        if da.epoch("k3-a").await == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(da.epoch("k3-a").await, 1);
    // the coordinator dies; a new one takes the same address; the daemon reconnects and hellos
    cl.cancel.cancel();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let (coord2, _t, _a, cancel2) = coordinator("policy-rig-4gpu.json", cl.dir.path()).await;
    for _ in 0..300 {
        if coord2
            .engine("k3-a")
            .await
            .is_some_and(|v| v.epoch == 1 && v.members.len() == 2)
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let view = coord2.engine("k3-a").await.expect("view");
    assert_eq!(view.epoch, 1, "{view:?}");
    assert_eq!(view.members.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
    // the hello named the spare as rank 1's holder, on its own device, though the old worker's
    // connection is still open
    assert_eq!(view.devices.get(&1), Some(&5), "{view:?}");
    assert!(view.identities.contains_key(&1));
    // a change under the new coordinator is epoch 2, not 1
    let err = coord2.rank_lost("k3-a", 1, Cause::Fault).await.unwrap_err();
    assert!(
        matches!(err, ChangeError::Rebuild(1)),
        "no spare is ready now: {err}"
    );
    let _ = (w1, cancel2);
    daemon_cancel.cancel();
}

#[tokio::test]
async fn a_notice_reaches_every_worker_and_the_change_runs_at_the_deadline() {
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let mut spare = common::register(&sock_a, 1, 1, Role::Spare).await;
    let mut client = common::register(&sock_a, 0, 0, Role::Client).await;
    common::wait_members(&cl.coord, 2).await;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
    let notice = Signal {
        kind: Some(signal::Kind::Notice(Notice {
            rank: 1,
            deadline_ms: now_ms + 300,
            reason: Reason::Notice as i32,
        })),
    };
    frame::send(&mut client, &notice).await.expect("notice");
    for w in [&mut w0, &mut w1, &mut spare] {
        let sig: Signal = common::recv(w).await.expect("forwarded notice");
        assert!(
            matches!(sig.kind, Some(signal::Kind::Notice(ref n)) if n.rank == 1),
            "{sig:?}"
        );
    }
    // at the deadline the change runs with the rank still connected
    expect_change(&mut w0, 0, 1, 1).await;
    expect_change(&mut spare, 1, 1, 1).await;
    let sig: Signal = common::recv(&mut client).await.expect("change");
    assert!(
        matches!(sig.kind, Some(signal::Kind::Change(ref c)) if c.lost == vec![1]),
        "{sig:?}"
    );
    let sig: Signal = common::recv(&mut client).await.expect("committed");
    assert!(matches!(sig.kind, Some(signal::Kind::MembershipCommitted(ref m)) if m.epoch == 1));
    let text = cl.coord_telemetry.render();
    assert!(
        text.contains(
            "cause=\"notice\",engine=\"k3-a\",outcome=\"replace\",result=\"committed\"} 1"
        ),
        "{text}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_survivor_error_fails_the_change_and_rebuild_is_refused() {
    let cl = cluster("policy-pilot.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    common::wait_members(&cl.coord, 2).await;
    // no spare, no placement-independent model: the pilot policy falls to REBUILD, refused
    let err = cl
        .coord
        .rank_lost("k3-a", 1, Cause::Fault)
        .await
        .unwrap_err();
    assert!(matches!(err, ChangeError::Rebuild(1)), "{err}");
    // with a spare for rank 1 in the pool, a survivor that answers with an error fails the
    // change; the spare takes it and answers well
    let mut spare = common::register(&sock_a, 1, 2, Role::Spare).await;
    for _ in 0..200 {
        if cl
            .coord
            .engine("k3-a")
            .await
            .is_some_and(|v| v.spares.contains_key(&1))
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let drive = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 1, Cause::Fault).await }
    });
    let sig: Signal = common::recv(&mut spare).await.expect("change to the spare");
    assert!(matches!(sig.kind, Some(signal::Kind::Change(_))));
    frame::send(
        &mut spare,
        &Signal {
            kind: Some(signal::Kind::Ack(Ack {
                rank: 1,
                epoch: 1,
                error: String::new(),
                received_at_us: 0,
            })),
        },
    )
    .await
    .expect("spare ack");
    let sig: Signal = common::recv(&mut w0).await.expect("change");
    assert!(matches!(sig.kind, Some(signal::Kind::Change(_))));
    frame::send(
        &mut w0,
        &Signal {
            kind: Some(signal::Kind::Ack(Ack {
                rank: 0,
                epoch: 1,
                error: "collective refused".into(),
                received_at_us: 0,
            })),
        },
    )
    .await
    .expect("ack");
    let err = drive.await.expect("join").unwrap_err();
    assert!(
        matches!(err, ChangeError::Survivor { rank: 0, .. }),
        "{err}"
    );
    let text = cl.coord_telemetry.render();
    assert!(text.contains("result=\"survivor_error\"} 1"), "{text}");
    assert!(text.contains("result=\"refused_rebuild\"} 1"), "{text}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_survivor_that_leaves_after_acknowledging_fails_the_change() {
    // rank 0 acknowledges and its engine shuts down while the spare's take runs: committed, the
    // change would commit an engine with no process left
    let cl = cluster("policy-pilot.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    common::wait_members(&cl.coord, 2).await;
    let mut spare = common::register(&sock_a, 1, 2, Role::Spare).await;
    for _ in 0..200 {
        if cl
            .coord
            .engine("k3-a")
            .await
            .is_some_and(|v| v.spares.contains_key(&1))
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let drive = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 1, Cause::Fault).await }
    });
    expect_change(&mut w0, 0, 1, 1).await;
    drop(w0); // the survivor's process exits after its acknowledgement
    for _ in 0..500 {
        if cl
            .coord
            .engine("k3-a")
            .await
            .is_some_and(|v| !v.members.contains_key(&0))
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    expect_change(&mut spare, 1, 1, 1).await; // the spare's take ends after it
    let err = drive.await.expect("join").unwrap_err();
    assert!(
        matches!(err, ChangeError::Departed { ref ranks } if ranks == &[0]),
        "{err}"
    );
    let text = cl.coord_telemetry.render();
    assert!(text.contains("result=\"survivor_departed\"} 1"), "{text}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn the_daemon_publishes_its_record_and_withdraws_it_on_stop() {
    let dir = tempfile::tempdir().expect("tmp");
    let telemetry = Telemetry::new();
    let (d, _rx) = Daemon::new("rig", &telemetry);
    let sock = dir.path().join("d.sock");
    d.publish_to(
        dir.path().join("disc"),
        &sock,
        Some(Path::new("/tmp/m.sock")),
        Some("10.0.0.7"),
    )
    .await
    .expect("publish");
    let cancel = CancellationToken::new();
    let server = tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel.clone()));
    for _ in 0..100 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let _w = common::register(&sock, 0, 0, Role::Worker).await;
    for _ in 0..100 {
        if !d.registered("k3-a").await.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let fd = FileDiscovery::new(dir.path().join("disc")).expect("fd");
    let rec = fd.daemon("rig").expect("read").expect("published");
    assert_eq!(rec.engines, vec!["k3-a".to_owned()]);
    assert_eq!(rec.mirror_address, "/tmp/m.sock");
    assert_eq!(rec.host, "10.0.0.7");
    cancel.cancel();
    server.await.expect("join").expect("serve");
    assert!(
        fd.daemon("rig").expect("read").is_none(),
        "withdrawn on stop"
    );
}

#[tokio::test]
async fn a_node_lost_is_one_change_for_every_rank_on_it_with_every_spare_promoted() {
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let (db, sock_b) =
        common::daemon(cl.dir.path(), "node-b", &cl.address, Vec::new(), &cl.cancel).await;
    // ranks 0 and 1 on node a; ranks 2 and 3 on node b; spares for 2 and 3 on node a (the spare
    // node); node b is about to be lost as a whole
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let w2 = common::register(&sock_b, 2, 2, Role::Worker).await;
    let w3 = common::register(&sock_b, 3, 3, Role::Worker).await;
    let mut s2 = common::register(&sock_a, 2, 2, Role::Spare).await;
    let mut s3 = common::register(&sock_a, 3, 3, Role::Spare).await;
    common::wait_members(&cl.coord, 4).await;
    for _ in 0..100 {
        if cl
            .coord
            .engine("k3-a")
            .await
            .is_some_and(|v| v.spares.len() == 2)
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    drop((w2, w3));
    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![2, 3],
            reason: Reason::Fault as i32,
            detail: "node b unreachable".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    // one change, one epoch, to both survivors and both spares
    let c = expect_change(&mut w0, 0, 1, 1).await;
    assert_eq!(
        (c.lost.clone(), c.outcome),
        (vec![2, 3], Outcome::Replace as i32)
    );
    expect_change(&mut w1, 1, 1, 1).await;
    expect_change(&mut s2, 2, 1, 1).await;
    expect_change(&mut s3, 3, 1, 1).await;
    for _ in 0..100 {
        if cl
            .coord
            .engine("k3-a")
            .await
            .is_some_and(|v| v.epoch == 1 && v.spares.is_empty())
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.epoch, 1);
    assert_eq!(view.members.get(&2).map(String::as_str), Some("node-a"));
    assert_eq!(view.members.get(&3).map(String::as_str), Some("node-a"));
    assert!(view.spares.is_empty());
    let _ = db;
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_spare_on_a_slower_class_is_refused_unless_the_policy_allows_the_degrade() {
    // the two-node policy allows no mismatch
    let cl = cluster("policy-two-node.json").await;
    let (_da, sock_a) = node_with_devices(&cl, "node-a", 4, "NVIDIA H100 80GB HBM3", None).await;
    let (_dc, sock_c) = node_with_devices(&cl, "node-c", 4, "NVIDIA A100-SXM4-80GB", None).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    common::wait_members(&cl.coord, 1).await;
    let _spare = common::register(&sock_c, 0, 0, Role::Spare).await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert!(
        view.spares.is_empty(),
        "an A100 spare for an H100 rank, no mismatch allowed: {view:?}"
    );
    assert!(cl.coord.pool("k3-a").await.is_empty());
    cl.cancel.cancel();

    // the mismatched policy allows it within a 50% degrade; A100 ranks at half an H100
    let cl = cluster("policy-two-node-mismatched.json").await;
    let (_da, sock_a) = node_with_devices(&cl, "node-a", 4, "NVIDIA H100 80GB HBM3", None).await;
    let (_dc, sock_c) = node_with_devices(&cl, "node-c", 4, "NVIDIA A100-SXM4-80GB", None).await;
    let (_dd, sock_d) = node_with_devices(&cl, "node-d", 4, "NVIDIA L40S", None).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    common::wait_members(&cl.coord, 2).await;
    let _spare_c = common::register(&sock_c, 0, 0, Role::Spare).await;
    let _spare_d = common::register(&sock_d, 1, 1, Role::Spare).await;
    for _ in 0..200 {
        if !cl.coord.pool("k3-a").await.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let pool = cl.coord.pool("k3-a").await;
    assert_eq!(
        pool.len(),
        1,
        "the A100 is within the bound, the L40S is not: {pool:?}"
    );
    assert_eq!(pool[0].node, "node-c");
    assert!((pool[0].speed - 0.5).abs() < 1e-9, "{pool:?}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn the_coordinator_has_the_spare_nodes_daemon_launch_the_policys_spares() {
    // decision 0024: the managed policy places four spares on node-c with a launch command; the
    // daemon on node-c launches and keeps them, the daemon on node-a launches none
    let cl = cluster("policy-two-node-managed.json").await;
    let (da, _sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let (dc, _sock_c) =
        common::daemon(cl.dir.path(), "node-c", &cl.address, Vec::new(), &cl.cancel).await;
    for _ in 0..300 {
        if dc.launched().await.len() == 4 && dc.launched().await.iter().all(|l| l.pid > 0) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let launched = dc.launched().await;
    assert_eq!(launched.len(), 4, "{launched:?}");
    assert!(
        launched.iter().all(|l| l.pid > 0 && l.engine == "k3-a"),
        "{launched:?}"
    );
    assert_eq!(
        launched.iter().map(|l| l.rank).collect::<Vec<_>>(),
        vec![Some(4), Some(5), Some(6), Some(7)]
    );
    assert!(da.launched().await.is_empty());
    cl.cancel.cancel();
    // the spares die with the daemon
    let pids: Vec<u32> = launched.iter().map(|l| l.pid).collect();
    for _ in 0..600 {
        if pids.iter().all(|p| !common::alive(*p)) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        pids.iter().all(|p| !common::alive(*p)),
        "spares still running after the daemon stopped"
    );
}

#[tokio::test]
async fn a_change_that_fails_after_its_delivery_puts_the_engine_down_and_reaches_every_process() {
    // decision 0032: a survivor's error leaves the applied processes on a membership that will
    // not commit; the engine is down, every daemon takes the epoch as spent, and every process of
    // the engine hears it
    let cl = cluster("policy-pilot.json").await;
    let (da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let mut head = common::register_head(&sock_a, 4242).await;
    common::wait_members(&cl.coord, 2).await;
    let mut spare = common::register(&sock_a, 1, 2, Role::Spare).await;
    wait_view(&cl.coord, "the spare", |v| v.spares.contains_key(&1)).await;
    let drive = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 1, Cause::Fault).await }
    });
    expect_change(&mut spare, 1, 1, 1).await;
    let sig: Signal = common::recv(&mut w0).await.expect("change");
    assert!(matches!(sig.kind, Some(signal::Kind::Change(_))));
    frame::send(
        &mut w0,
        &Signal {
            kind: Some(signal::Kind::Ack(Ack {
                rank: 0,
                epoch: 1,
                error: "collective refused".into(),
                received_at_us: 0,
            })),
        },
    )
    .await
    .expect("ack");
    let err = drive.await.expect("join").unwrap_err();
    assert!(
        matches!(err, ChangeError::Survivor { rank: 0, .. }),
        "{err}"
    );
    let v = wait_view(&cl.coord, "the engine down", |v| v.down.is_some()).await;
    assert_eq!(v.down.expect("down").cause, "change_failed");
    for w in [&mut w0, &mut spare, &mut head] {
        let a = expect_abandoned(w).await;
        assert_eq!(a.epoch, 1);
        assert!(a.detail.contains("collective refused"), "{a:?}");
    }
    for _ in 0..200 {
        if da.epoch("k3-a").await == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        da.epoch("k3-a").await,
        1,
        "the daemon took the epoch as spent"
    );
    let err = cl
        .coord
        .rank_lost("k3-a", 1, Cause::Fault)
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            ChangeError::Down {
                cause: "change_failed",
                ..
            }
        ),
        "{err}"
    );
    cl.cancel.cancel();
}

/// The commit lists every acknowledgement it waited on, each received between the delivery and
/// the commit (decision 0032's instrumentation).
fn assert_acks(m: &MembershipCommitted, ranks: &[u32]) {
    let mut acked: Vec<u32> = m.acks.iter().map(|a| a.rank).collect();
    acked.sort_unstable();
    assert_eq!(acked, ranks, "every acknowledgement the commit waited on");
    assert!(
        m.acks.iter().all(|a| m.delivered_at_us <= a.received_at_us
            && a.received_at_us <= m.committed_at_us),
        "each received between the delivery and the commit: {m:?}"
    );
}

#[tokio::test]
async fn readiness_reports_the_committed_epoch_and_a_change_in_flight_beside_it() {
    // readiness read between a change's claim and its commit must not show the epoch moved, the
    // lost rank gone from the members and the spare not yet in
    let cl = cluster("policy-pilot.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    common::wait_members(&cl.coord, 2).await;
    let mut spare = common::register(&sock_a, 1, 2, Role::Spare).await;
    wait_view(&cl.coord, "the spare", |v| v.spares.contains_key(&1)).await;
    let drive = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 1, Cause::Fault).await }
    });
    expect_change(&mut w0, 0, 1, 1).await;
    // the spare has not answered: the change is in flight
    let e = cl.coord.readiness().await.engines.remove(0);
    assert_eq!(e.epoch, 0, "no epoch is committed yet: {e:?}");
    let c = e.change.expect("the change in flight");
    assert_eq!((c.epoch, c.lost), (1, vec![1]));
    expect_change(&mut spare, 1, 1, 1).await;
    drive.await.expect("join").expect("committed");
    let e = cl.coord.readiness().await.engines.remove(0);
    assert_eq!(e.epoch, 1, "{e:?}");
    assert!(e.change.is_none(), "{e:?}");
    assert!(
        e.members.iter().any(|m| m.rank == 1),
        "the spare serves rank 1: {e:?}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_node_the_change_empties_hears_its_commit() {
    // a change moves node-b's only rank to a spare on another node: node-b's daemon hears the
    // commit, or its node role never learns its position has left
    let cl = cluster("policy-two-node.json").await;
    let (da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let (_db, sock_b) =
        common::daemon(cl.dir.path(), "node-b", &cl.address, Vec::new(), &cl.cancel).await;
    let (_dc, sock_c) =
        common::daemon(cl.dir.path(), "node-c", &cl.address, Vec::new(), &cl.cancel).await;
    let mut on_a = Vec::new();
    let mut on_b = Vec::new();
    let mut on_c = Vec::new();
    for r in 0..4 {
        on_a.push(common::register(&sock_a, r, r, Role::Worker).await);
        on_b.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    common::wait_members(&cl.coord, 8).await;
    for r in 0..4 {
        on_c.push(common::register(&sock_c, r, r, Role::Spare).await);
    }
    wait_view(&cl.coord, "four spares", |v| v.spares.len() == 4).await;
    // rank 0's spare is outside its domain: node-a's whole group moves to node-c (0022)
    let drive = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 0, Cause::Fault).await }
    });
    for (r, w) in (4u32..)
        .zip(on_b.iter_mut())
        .chain((0u32..).zip(on_c.iter_mut()))
    {
        let change = next_change(w).await;
        assert_eq!(change.lost, vec![0, 1, 2, 3]);
        ack(w, r, 1).await;
    }
    drive.await.expect("join").expect("committed");
    // node-a holds nothing of the engine now, and its daemon heard the commit all the same
    for _ in 0..300 {
        if da.epoch("k3-a").await == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        da.epoch("k3-a").await,
        1,
        "node-a's daemon heard the commit"
    );
    drop(on_a);
    cl.cancel.cancel();
}

/// Whatever `w` is sent within `wait`: None if it is sent nothing.
async fn quiet(w: &mut UnixStream, wait: std::time::Duration) -> Option<Signal> {
    tokio::time::timeout(wait, frame::recv::<_, Signal>(w))
        .await
        .ok()
        .and_then(Result::ok)
}

#[tokio::test]
async fn a_departing_rank_hears_its_change_outside_the_barrier_and_the_commit_places_every_rank() {
    // protocol 9: a rank that leaves alive gets the Change too, marked departing, and nothing
    // waits on it; a faulted rank is dead and gets nothing; every commit carries the whole
    // membership, so a listener that hears only commits knows where each rank runs
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let (_db, sock_b) =
        common::daemon(cl.dir.path(), "node-b", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let mut w2 = common::register(&sock_b, 2, 2, Role::Worker).await;
    let mut w3 = common::register(&sock_b, 3, 3, Role::Worker).await;
    let mut s2 = common::register(&sock_a, 2, 4, Role::Spare).await;
    let mut s3 = common::register(&sock_b, 3, 5, Role::Spare).await;
    common::wait_members(&cl.coord, 4).await;
    wait_view(&cl.coord, "both spares", |v| v.spares.len() == 2).await;

    // rank 2 drains on notice
    let drained = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 2, Cause::Notice).await }
    });
    for (w, r) in [(&mut w0, 0), (&mut w1, 1), (&mut w3, 3), (&mut s2, 2)] {
        let c = within(&format!("rank {r}'s change"), next_change(w)).await;
        assert_eq!((c.epoch, c.departing.clone()), (1, vec![2]), "rank {r}");
        ack(w, r, 1).await;
    }
    let to_departing = within("the departing change", next_change(&mut w2)).await;
    assert_eq!(
        (
            to_departing.epoch,
            to_departing.lost.clone(),
            to_departing.departing.clone()
        ),
        (1, vec![2], vec![2]),
        "the departing worker hears its own change"
    );
    // it never acknowledges, and the change commits all the same
    let committed = within("rank 2's commit, unacknowledged by rank 2", drained)
        .await
        .expect("join")
        .expect("rank 2's change");
    let placed = |m: &MembershipCommitted| {
        m.members
            .iter()
            .map(|p| {
                (
                    p.rank,
                    p.node.clone(),
                    p.device.expect("a committed rank's device"),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        placed(&committed),
        vec![
            (0, "node-a".to_owned(), 0),
            (1, "node-a".to_owned(), 1),
            (2, "node-a".to_owned(), 4),
            (3, "node-b".to_owned(), 3),
        ],
        "the commit places every active rank, rank 2 on its spare's device"
    );

    // rank 3 faults: it is dead, so nothing is forwarded to it and no rank departs
    let faulted = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 3, Cause::Fault).await }
    });
    for (w, r) in [(&mut w0, 0), (&mut w1, 1), (&mut s2, 2), (&mut s3, 3)] {
        let c = within(&format!("rank {r}'s change"), next_change(w)).await;
        assert_eq!((c.epoch, c.departing.clone()), (2, vec![]), "rank {r}");
        ack(w, r, 2).await;
    }
    let committed = within("rank 3's commit", faulted)
        .await
        .expect("join")
        .expect("rank 3's change");
    assert_eq!(
        placed(&committed)[3],
        (3, "node-b".to_owned(), 5),
        "rank 3 on its spare's device"
    );
    let heard = quiet(&mut w3, std::time::Duration::from_millis(300)).await;
    assert!(
        !matches!(heard.and_then(|s| s.kind), Some(signal::Kind::Change(_))),
        "a faulted rank is sent no change"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_process_whose_pid_its_daemon_cannot_see_is_still_recorded() {
    // on Kubernetes a process in another pod is outside the daemon's pid namespace, and the
    // socket's peer credentials say 0: the process is recorded all the same, by its connection,
    // with no pid, so readiness names it and a stacks request reaches it (decision 0041)
    use everserve_controller::proto::everserve::coord::v1::{
        DaemonToCoord, WorkerEvent, daemon_to_coord, worker_event,
    };
    let cl = cluster("policy-pilot.json").await;
    let mut d = fake_daemon(&cl, "node-a", vec![0, 1, 2, 3], 0).await;
    let event = |kind: worker_event::Kind, rank: u32, device: u32| DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::WorkerEvent(WorkerEvent {
            engine: "k3-a".into(),
            kind: kind as i32,
            rank,
            world_size: 4,
            device,
            pid: 0,
            identity: Some(common::identity()),
            ..WorkerEvent::default()
        })),
    };
    frame::send(&mut d, &event(worker_event::Kind::WorkerRegistered, 0, 0))
        .await
        .expect("worker");
    frame::send(&mut d, &event(worker_event::Kind::SpareRegistered, 1, 5))
        .await
        .expect("spare");
    let v = wait_view(&cl.coord, "both processes", |v| {
        v.processes.members.contains_key(&0) && v.processes.spares.contains_key(&1)
    })
    .await;
    assert_eq!(v.processes.members[&0].pid, None, "no pid, not pid 0");
    assert_eq!(v.processes.spares[&1].node, "node-a");
    cl.cancel.cancel();
}
