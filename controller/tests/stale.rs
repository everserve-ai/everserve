// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What arrives around a change: a second report of a death already acted on is folded,
//! a departure from before a rank's change never removes its new holder, reports during a
//! change are folded into one change after it, and a moved rank's old worker closing after the
//! commit never unseats its new holder.

mod common;

use std::sync::Arc;

use everserve_controller::coord::{Cause, ChangeError};
use everserve_controller::proto::everserve::signals::v1::{
    Ack, RankLost, Reason, Role, Signal, signal,
};
use everserve_wire::frame;

use common::cluster::{ack, cluster, expect_change, fake_daemon, next_change, wait_view};

#[tokio::test]
async fn a_second_report_of_a_death_already_acted_on_is_folded() {
    let cl = cluster("policy-rig-4gpu.json").await;
    let (d, sock) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock, 0, 0, Role::Worker).await;
    let w1 = common::register(&sock, 1, 1, Role::Worker).await;
    let mut spare = common::register(&sock, 1, 1, Role::Spare).await;
    common::wait_members(&cl.coord, 2).await;
    // the slot kernel on a survivor sees rank 1 first, at epoch 0
    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![1],
            reason: Reason::Fault as i32,
            detail: "slot: no push from rank 1; its process is exiting".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    // another survivor's watch reports too, while the change is already on its way to w0: the
    // daemon must take the report as a report, not as w0's ack
    w0.readable().await.expect("the change is in flight");
    frame::send(&mut w0, &lost).await.expect("second report");
    expect_change(&mut w0, 0, 1, 1).await;
    expect_change(&mut spare, 1, 1, 1).await;
    // committed: the daemon moves to the epoch only when the change is (a failed delivery would
    // leave the coordinator's claim at 1 and the daemon at 0)
    for _ in 0..100 {
        if d.epoch("k3-a").await == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(d.epoch("k3-a").await, 1);
    let _ = w1;
    // the executor's sentinel reports the same death later, still from epoch 0: folded
    let again = cl.coord.ranks_reported("k3-a", &[1], 0, Cause::Fault).await;
    assert!(
        matches!(
            again,
            Err(ChangeError::Stale {
                rank: 1,
                seen: 0,
                changed: 1
            })
        ),
        "{again:?}"
    );
    assert_eq!(cl.coord.engine("k3-a").await.expect("view").epoch, 1);
    // a report from epoch 1 is about the spare now in rank 1's position: not folded
    let later = cl.coord.ranks_reported("k3-a", &[1], 1, Cause::Fault).await;
    assert!(
        !matches!(later, Err(ChangeError::Stale { .. })),
        "{later:?}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_departure_from_before_a_ranks_change_never_removes_its_new_holder() {
    // a worker dies and a spare on its own node takes its rank; the worker's departure, sent
    // before the commit, reaches the coordinator after it, and must not remove the spare that now
    // holds the rank
    use everserve_controller::proto::everserve::coord::v1::{
        CoordToDaemon, DaemonToCoord, Delivered, WorkerEvent, coord_to_daemon, daemon_to_coord,
        worker_event,
    };
    let cl = cluster("policy-pilot.json").await;
    let mut d = fake_daemon(&cl, "node-a", vec![0, 1, 2, 3], 0).await;
    let event = |kind: worker_event::Kind, rank: u32, epoch: u64| DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::WorkerEvent(WorkerEvent {
            engine: "k3-a".into(),
            kind: kind as i32,
            rank,
            world_size: 4,
            device: rank + 4,
            epoch,
            identity: Some(common::identity()),
            ..WorkerEvent::default()
        })),
    };
    frame::send(&mut d, &event(worker_event::Kind::SpareRegistered, 1, 0))
        .await
        .expect("spare");
    wait_view(&cl.coord, "the spare", |v| v.spares.contains_key(&1)).await;
    let drive = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 1, Cause::Fault).await }
    });
    // the coordinator delivers to node-a: ranks 0, 2 and 3 survive, the spare takes rank 1
    let request_id = loop {
        let msg: CoordToDaemon = frame::recv(&mut d).await.expect("frame");
        if let Some(coord_to_daemon::Kind::Deliver(dl)) = msg.kind {
            break dl.request_id;
        }
    };
    let ack = |rank| Ack {
        rank,
        epoch: 1,
        error: String::new(),
        received_at_us: 0,
    };
    let delivered = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::Delivered(Delivered {
            request_id,
            engine: "k3-a".into(),
            epoch: 1,
            acks: vec![ack(0), ack(1), ack(2), ack(3)],
            error: String::new(),
        })),
    };
    frame::send(&mut d, &delivered).await.expect("delivered");
    drive.await.expect("join").expect("committed");
    wait_view(&cl.coord, "rank 1 held again", |v| {
        v.members.contains_key(&1)
    })
    .await;

    // the old worker's departure, seen at epoch 0, arrives now
    frame::send(&mut d, &event(worker_event::Kind::WorkerDeparted, 1, 0))
        .await
        .expect("stale departure");
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let v = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(
        v.members.get(&1).map(String::as_str),
        Some("node-a"),
        "the spare that took rank 1 still holds it: {v:?}"
    );
    // a departure at the rank's own epoch is the new holder's, and does remove it
    frame::send(&mut d, &event(worker_event::Kind::WorkerDeparted, 1, 1))
        .await
        .expect("departure");
    wait_view(&cl.coord, "rank 1 departed", |v| {
        !v.members.contains_key(&1)
    })
    .await;
    cl.cancel.cancel();
}

#[tokio::test]
async fn reports_during_a_change_are_folded_into_one_change_after_it() {
    // decision 0036: one transition per engine at a time, and every report the controller finds
    // pending is one change; before the controller a report that raced a change claimed an
    // overlapping epoch and was refused by the daemons
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let _w2 = common::register(&sock_a, 2, 2, Role::Worker).await;
    let mut w3 = common::register(&sock_a, 3, 3, Role::Worker).await;
    let mut s2 = common::register(&sock_a, 2, 4, Role::Spare).await;
    let mut s3 = common::register(&sock_a, 3, 5, Role::Spare).await;
    common::wait_members(&cl.coord, 4).await;
    wait_view(&cl.coord, "both spares", |v| v.spares.len() == 2).await;

    // rank 2's change is delivered and held: nobody has acknowledged it yet
    let first = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 2, Cause::Fault).await }
    });
    for w in [&mut w0, &mut w1, &mut w3, &mut s2] {
        assert_eq!(next_change(w).await.epoch, 1);
    }
    // two reports of rank 3 arrive while it runs
    let reports: Vec<_> = (0..2)
        .map(|_| {
            let c = Arc::clone(&cl.coord);
            tokio::spawn(async move { c.rank_lost("k3-a", 3, Cause::Fault).await })
        })
        .collect();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    for (w, r) in [(&mut w0, 0), (&mut w1, 1), (&mut w3, 3), (&mut s2, 2)] {
        ack(w, r, 1).await;
    }
    let one = first.await.expect("join").expect("rank 2's change");
    assert_eq!((one.epoch, one.lost.clone()), (1, vec![]));

    // then one change for rank 3, which answers both reports
    for (w, r) in [(&mut w0, 0), (&mut w1, 1), (&mut s2, 2), (&mut s3, 3)] {
        assert_eq!(next_change(w).await.epoch, 2, "rank {r}");
        ack(w, r, 2).await;
    }
    for report in reports {
        let m = report.await.expect("join").expect("rank 3's change");
        assert_eq!(m.epoch, 2, "both reports are answered by the one change");
    }
    let v = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(v.committed_epoch, 2, "two changes, not three: {v:?}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_moved_ranks_old_worker_closing_after_the_commit_never_unseats_its_new_holder() {
    // the adapter releases a worker whose position moved once the change commits (0033, step 3):
    // its link closes after the commit, on its own node or another than the taker's. The daemon
    // dropped the rank from that connection at the commit, so the close reports nothing, and the
    // node's next hello claims nothing it does not hold.
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let (db, sock_b) =
        common::daemon(cl.dir.path(), "node-b", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let mut w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let w2 = common::register(&sock_b, 2, 2, Role::Worker).await;
    let mut w3 = common::register(&sock_b, 3, 3, Role::Worker).await;
    // rank 2's spare on node-a (a move across nodes), rank 3's on node-b (a move within one)
    let mut s2 = common::register(&sock_a, 2, 4, Role::Spare).await;
    let mut s3 = common::register(&sock_b, 3, 5, Role::Spare).await;
    common::wait_members(&cl.coord, 4).await;
    wait_view(&cl.coord, "both spares", |v| v.spares.len() == 2).await;

    let moved = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 2, Cause::Notice).await }
    });
    for (w, r) in [(&mut w0, 0), (&mut w1, 1), (&mut w3, 3), (&mut s2, 2)] {
        assert_eq!(next_change(w).await.epoch, 1);
        ack(w, r, 1).await;
    }
    moved.await.expect("join").expect("rank 2 moved to node-a");
    // committed, and the old worker not yet released: node-b already claims nothing of rank 2,
    // so a hello now, to a restarted coordinator or its standby, would not either
    for _ in 0..200 {
        if db.epoch("k3-a").await == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(db.epoch("k3-a").await, 1);
    assert!(
        !db.registered("k3-a").await.contains(&2),
        "node-b dropped rank 2 at the commit, before its old worker closed"
    );
    drop(w2); // node-b's old worker for rank 2 is released after the commit
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let v = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(
        v.members.get(&2).map(String::as_str),
        Some("node-a"),
        "{v:?}"
    );
    assert!(
        !db.registered("k3-a").await.contains(&2),
        "node-b claims nothing of rank 2"
    );

    let moved = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 3, Cause::Notice).await }
    });
    for (w, r) in [(&mut w0, 0), (&mut w1, 1), (&mut s2, 2), (&mut s3, 3)] {
        assert_eq!(next_change(w).await.epoch, 2);
        ack(w, r, 2).await;
    }
    moved
        .await
        .expect("join")
        .expect("rank 3 moved within node-b");
    drop(w3); // node-b's old worker for rank 3, on the taker's own node
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let v = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(
        v.members.get(&3).map(String::as_str),
        Some("node-b"),
        "{v:?}"
    );
    assert_eq!(
        db.registered("k3-a").await,
        vec![3],
        "node-b holds rank 3 through the spare that took it"
    );
    cl.cancel.cancel();
}
