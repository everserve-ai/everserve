// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Replication of a rank's rows (decision 0026): to its sibling when the policy names no replica
//! node, to the node of the spare placed for it, staged onto the spare's node before a REPLACE,
//! the old holder told it replicates nowhere, every member store told of the spare node, a spare
//! standing for any position taking the lost rank, and a restore from a replica already there.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use everserve_controller::node::Daemon;
use everserve_controller::policy;
use everserve_controller::proto::everserve::signals::v1::{
    Ack, RankLost, Reason, Register, Role, Signal, signal,
};
use everserve_wire::frame;
use everserve_wire::proto::everserve::mirror::v1::{
    Done, Expect, MirrorReply, MirrorRequest, Replicate, mirror_reply, mirror_request,
};
use tokio::net::UnixStream;

use common::cluster::{
    Cluster, cache, cluster, cluster_of, engine_readiness, expect_change, node_with_devices,
    register_with,
};

type Seen<T> = Arc<tokio::sync::Mutex<Vec<T>>>;

/// Whether `seen` holds an item `pred` accepts within two seconds.
async fn saw<T>(seen: &Seen<T>, pred: impl Fn(&T) -> bool) -> bool {
    for _ in 0..200 {
        if seen.lock().await.iter().any(&pred) {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    false
}

/// A stand-in for a node's store: answers every request with `Done` and records the
/// `Replicate` and `Expect` requests it saw.
fn fake_store(dir: &Path, node: &str) -> (PathBuf, Seen<Replicate>, Seen<Expect>) {
    let sock = dir.join(format!("{node}-store.sock"));
    let listener = tokio::net::UnixListener::bind(&sock).expect("bind store");
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let expects: Seen<Expect> = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let record = Arc::clone(&seen);
    let record_expects = Arc::clone(&expects);
    tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            let req: MirrorRequest = frame::recv(&mut s).await.expect("request");
            match req.kind {
                Some(mirror_request::Kind::Replicate(r)) => record.lock().await.push(r),
                Some(mirror_request::Kind::Expect(e)) => record_expects.lock().await.push(e),
                _ => {}
            }
            frame::send(
                &mut s,
                &MirrorReply {
                    kind: Some(mirror_reply::Kind::Done(Done {})),
                    error: String::new(),
                },
            )
            .await
            .expect("reply");
        }
    });
    (sock, seen, expects)
}

#[tokio::test]
async fn a_ranks_rows_replicate_to_its_sibling_when_the_policy_names_no_replica_node() {
    // decision 0030: the shared-spare shape. One stage per node, a spare node named for
    // eligibility only: rank 1's rows go to node-a's store, rank 0's to node-b's, and the spare
    // node's store hears of nothing until a spare there joins.
    let cl = cluster("policy-two-node-resident.json").await;
    let (store_a, seen_a, expect_a) = fake_store(cl.dir.path(), "node-a");
    let (store_b, seen_b, expect_b) = fake_store(cl.dir.path(), "node-b");
    let (store_c, seen_c, expect_c) = fake_store(cl.dir.path(), "node-c");
    let (_da, sock_a) = node_with_devices(
        &cl,
        "node-a",
        1,
        "NVIDIA L40S",
        Some((store_a, "node-a:7402")),
    )
    .await;
    let (_db, sock_b) = node_with_devices(
        &cl,
        "node-b",
        1,
        "NVIDIA L40S",
        Some((store_b, "node-b:7402")),
    )
    .await;
    let (_dc, _sock_c) = node_with_devices(
        &cl,
        "node-c",
        1,
        "NVIDIA L40S",
        Some((store_c, "node-c:7402")),
    )
    .await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_b, 1, 0, Role::Worker).await;
    common::wait_members(&cl.coord, 2).await;
    assert!(saw(&seen_a, |r| r.rank == 0 && r.targets == ["node-b:7402"]).await);
    assert!(saw(&seen_b, |r| r.rank == 1 && r.targets == ["node-a:7402"]).await);
    assert!(saw(&expect_a, |e| e.rank == 1 && e.from_node == "node-b").await);
    assert!(saw(&expect_b, |e| e.rank == 0 && e.from_node == "node-a").await);
    assert!(
        seen_c.lock().await.is_empty() && expect_c.lock().await.is_empty(),
        "the spare node's store takes no rows: {:?} {:?}",
        seen_c.lock().await,
        expect_c.lock().await
    );
    cl.cancel.cancel();
}

/// The two-node resident shape's three one-GPU nodes with their stores, in name order; the
/// daemons come back so they live as long as the test.
async fn three_nodes(cl: &Cluster, stores: [(PathBuf, &str); 3]) -> Vec<(Arc<Daemon>, PathBuf)> {
    let mut out = Vec::new();
    for (name, store) in ["node-a", "node-b", "node-c"].into_iter().zip(stores) {
        out.push(node_with_devices(cl, name, 1, "NVIDIA L40S", Some(store)).await);
    }
    out
}

/// Register a spare standing for any position of the engine on `device` (decision 0030).
async fn register_any(path: &Path, device: u32) -> UnixStream {
    common::register_as(
        path,
        Register {
            any_rank: true,
            ..common::registration(0, device, Role::Spare)
        },
    )
    .await
}

/// After node-c's spare took rank 1 (decision 0030): rank 0's rows go to node-c, its new sibling,
/// and node-c expects rank 0 unfenced, since rank 0 never moved.
async fn replicates_after_the_take(seen_a: &Seen<Replicate>, expect_c: &Seen<Expect>) {
    // siblings follow the live membership: rank 0's rows now go to the node that took rank 1
    assert!(
        saw(seen_a, |r| r.rank == 0 && r.targets == ["node-c:7402"]).await,
        "rank 0 does not replicate to its new sibling: {:?}",
        seen_a.lock().await
    );
    // rank 0 never moved: its new sibling expects it unfenced, so its steps at its own epoch
    // are taken (fenced at the engine's epoch, every one would be refused)
    assert!(
        saw(expect_c, |e| e.rank == 0 && e.from_node == "node-a").await,
        "{:?}",
        expect_c.lock().await
    );
    assert!(
        expect_c
            .lock()
            .await
            .iter()
            .filter(|e| e.rank == 0)
            .all(|e| e.epoch == 0),
        "rank 0 was fenced though it never moved: {:?}",
        expect_c.lock().await
    );
}

#[tokio::test]
async fn a_spare_standing_for_any_position_takes_the_lost_rank_the_change_names() {
    // decision 0030: the shared pool. A spare on node-c holds no shard and stands for any rank;
    // when rank 1 dies the change names it (Change.take), the spare gets the change, its rows
    // are staged onto node-c, and at the commit it serves as rank 1.
    let cl = cluster("policy-two-node-resident.json").await;
    let (store_a, seen_a, _expect_a) = fake_store(cl.dir.path(), "node-a");
    let (store_b, _seen_b, _expect_b) = fake_store(cl.dir.path(), "node-b");
    let (store_c, _seen_c, expect_c) = fake_store(cl.dir.path(), "node-c");
    let nodes = three_nodes(
        &cl,
        [
            (store_a, "node-a:7402"),
            (store_b, "node-b:7402"),
            (store_c, "node-c:7402"),
        ],
    )
    .await;
    let mut w0 = register_with(&nodes[0].1, 0, 0, Role::Worker, Some(cache("r0"))).await;
    let w1 = register_with(&nodes[1].1, 1, 0, Role::Worker, Some(cache("r1"))).await;
    let mut spare = register_any(&nodes[2].1, 0).await;
    common::wait_members(&cl.coord, 2).await;
    let e = engine_readiness(&cl).await;
    assert_eq!(
        e.any_spares
            .iter()
            .map(|s| (s.node.as_str(), s.device))
            .collect::<Vec<_>>(),
        [("node-c", 0)],
        "{e:?}"
    );
    assert!(e.spares.is_empty(), "{e:?}");
    let node_b = e.nodes.iter().find(|n| n.node == "node-b").expect("node-b");
    assert!(
        node_b.reclaimable && node_b.outcome == "replace",
        "a spare for any rank makes REPLACE available: {node_b:?}"
    );

    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![1],
            reason: Reason::Fault as i32,
            detail: "process exited".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    drop(w1);
    expect_change(&mut w0, 0, 1, 1).await;
    // the spare hears the change with its take named
    let sig: Signal = common::recv(&mut spare).await.expect("change");
    let Some(signal::Kind::Change(c)) = sig.kind else {
        panic!("not a change: {sig:?}");
    };
    assert_eq!(c.lost, [1]);
    assert_eq!(
        c.take
            .iter()
            .map(|t| (t.rank, t.node.as_str(), t.device))
            .collect::<Vec<_>>(),
        [(1, "node-c", 0)]
    );
    // the position's pair rides with the take, though its serving rank has gone (0029)
    assert_eq!(
        c.take[0].compile_cache.as_ref().map(|p| p.digest.as_str()),
        Some("r1"),
        "{c:?}"
    );
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
    .expect("ack");
    assert!(saw(&seen_a, |r| r.rank == 1 && r.targets == ["node-c:7402"]).await);
    assert!(saw(&expect_c, |e| e.rank == 1 && e.from_node == "node-a").await);
    for _ in 0..200 {
        if cl
            .coord
            .engine("k3-a")
            .await
            .is_some_and(|v| v.members.get(&1).is_some_and(|n| n == "node-c"))
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.members.get(&1).map(String::as_str), Some("node-c"));
    assert!(
        view.any_spares.is_empty() && view.taking.is_empty(),
        "{view:?}"
    );
    let e = engine_readiness(&cl).await;
    assert!(e.any_spares.is_empty(), "{e:?}");
    replicates_after_the_take(&seen_a, &expect_c).await;
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_replace_stages_the_lost_ranks_rows_onto_its_spares_node_before_the_change() {
    // decision 0030: the spare node holds no rows; the sibling that replicates rank 1 is pointed
    // at the spare's store before the change goes out, and both are released at the commit
    let cl = cluster("policy-two-node-resident.json").await;
    let (store_a, seen_a, expect_a) = fake_store(cl.dir.path(), "node-a");
    let (store_b, _seen_b, _expect_b) = fake_store(cl.dir.path(), "node-b");
    let (store_c, seen_c, expect_c) = fake_store(cl.dir.path(), "node-c");
    let (_da, sock_a) = node_with_devices(
        &cl,
        "node-a",
        1,
        "NVIDIA L40S",
        Some((store_a, "node-a:7402")),
    )
    .await;
    let (_db, sock_b) = node_with_devices(
        &cl,
        "node-b",
        1,
        "NVIDIA L40S",
        Some((store_b, "node-b:7402")),
    )
    .await;
    let (_dc, sock_c) = node_with_devices(
        &cl,
        "node-c",
        1,
        "NVIDIA L40S",
        Some((store_c, "node-c:7402")),
    )
    .await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let w1 = common::register(&sock_b, 1, 0, Role::Worker).await;
    let mut spare = common::register(&sock_c, 1, 0, Role::Spare).await;
    common::wait_members(&cl.coord, 2).await;
    // rank 1's rows sit on node-a (its sibling); node-c's store has none
    assert!(saw(&expect_a, |e| e.rank == 1 && e.from_node == "node-b").await);
    assert!(seen_c.lock().await.is_empty(), "{:?}", seen_c.lock().await);

    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![1],
            reason: Reason::Fault as i32,
            detail: "process exited".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    drop(w1);
    expect_change(&mut w0, 0, 1, 1).await;
    expect_change(&mut spare, 1, 1, 1).await;

    // before the change reached anyone: node-a's store was pointed at node-c for rank 1, and
    // node-c's store told to expect rank 1 from node-a at the new epoch
    assert!(
        saw(&seen_a, |r| r.rank == 1 && r.targets == ["node-c:7402"]).await,
        "node-a's store was not pointed at the spare's node: {:?}",
        seen_a.lock().await
    );
    assert!(
        saw(&expect_c, |e| e.rank == 1 && e.from_node == "node-a").await,
        "node-c's store was not told to expect rank 1 from node-a: {:?}",
        expect_c.lock().await
    );
    // the staging moves no fence: node-c's store must take rank 1's rows from epoch 0, the
    // epoch they were committed at (a fence at epoch 1 refuses the staged rows)
    assert!(
        expect_c
            .lock()
            .await
            .iter()
            .filter(|e| e.rank == 1 && e.from_node == "node-a")
            .all(|e| e.epoch == 0),
        "the staging fenced node-c's store past the staged rows: {:?}",
        expect_c.lock().await
    );
    // at the commit: the staging ends, and rank 1, now on node-c, replicates to its sibling
    assert!(
        saw(&seen_a, |r| r.rank == 1 && r.targets.is_empty()).await,
        "node-a's store was not released from carrying rank 1: {:?}",
        seen_a.lock().await
    );
    assert!(
        saw(&expect_c, |e| e.rank == 1
            && e.from_node.is_empty()
            && e.epoch == 1)
        .await,
        "node-c's store still expects rank 1 from node-a: {:?}",
        expect_c.lock().await
    );
    assert!(
        saw(&seen_c, |r| r.rank == 1 && r.targets == ["node-a:7402"]).await,
        "rank 1 on node-c does not replicate to its sibling: {:?}",
        seen_c.lock().await
    );
    assert!(saw(&expect_a, |e| e.rank == 1 && e.from_node == "node-c").await);
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert!(view.restoring.is_empty(), "{view:?}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_rank_replicates_to_the_node_of_the_spare_placed_for_it() {
    // the three-node policy: one stage per node, a spare placed with `spares` for rank 1 on
    // node-c (decision 0024), and node-c named as the replica node (decision 0030). With no
    // target named the spare restores no rows, and every session replays from zero
    let cl = cluster("policy-three-node.json").await;
    let (store_a, seen_a, _expect_a) = fake_store(cl.dir.path(), "node-a");
    let (store_b, seen_b, _expect_b) = fake_store(cl.dir.path(), "node-b");
    let (store_c, seen_c, expect_c) = fake_store(cl.dir.path(), "node-c");
    let (_da, sock_a) = node_with_devices(
        &cl,
        "node-a",
        1,
        "NVIDIA L40S",
        Some((store_a, "node-a:7402")),
    )
    .await;
    let (_db, sock_b) = node_with_devices(
        &cl,
        "node-b",
        1,
        "NVIDIA L40S",
        Some((store_b, "node-b:7402")),
    )
    .await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_b, 1, 0, Role::Worker).await;
    common::wait_members(&cl.coord, 2).await;
    let (_dc, _sock_c) = node_with_devices(
        &cl,
        "node-c",
        1,
        "NVIDIA L40S",
        Some((store_c, "node-c:7402")),
    )
    .await;
    for _ in 0..200 {
        if seen_b.lock().await.len() == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let b = seen_b.lock().await.clone();
    assert_eq!(b.len(), 1, "node-b's store heard rank 1's target: {b:?}");
    assert_eq!(
        (b[0].rank, b[0].targets.clone()),
        (1, vec!["node-c:7402".to_owned()])
    );
    assert_eq!(
        b[0].node, "node-b",
        "the source names itself to the replica (decision 0026)"
    );
    for _ in 0..200 {
        if !expect_c.lock().await.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let c = expect_c.lock().await.clone();
    assert!(
        c.iter().any(|e| e.rank == 1 && e.from_node == "node-b"),
        "node-c's store expects rank 1 from node-b: {c:?}"
    );
    // a named replica node takes every rank's rows (decision 0030), the spare slot's rank
    // notwithstanding: rank 0's store is told the same target
    assert!(
        saw(&seen_a, |r| r.rank == 0 && r.targets == ["node-c:7402"]).await,
        "node-a's store was not told rank 0's replica node: {:?}",
        seen_a.lock().await
    );
    assert!(seen_c.lock().await.is_empty());
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(
        view.replication.get(&1),
        Some(&vec!["node-c:7402".to_owned()]),
        "{view:?}"
    );
}

#[tokio::test]
async fn a_replaced_ranks_old_holder_is_told_it_replicates_the_rank_nowhere() {
    // after rank 1 moves from node-b to node-c, node-b's store would drain its backlog for
    // rank 1 into node-c's replica, refused as stale every 40 ms
    let cl = cluster("policy-three-node.json").await;
    let (store_a, _seen_a, _expect_a) = fake_store(cl.dir.path(), "node-a");
    let (store_b, seen_b, _expect_b) = fake_store(cl.dir.path(), "node-b");
    let (store_c, seen_c, expect_c) = fake_store(cl.dir.path(), "node-c");
    let (_da, sock_a) = node_with_devices(
        &cl,
        "node-a",
        1,
        "NVIDIA L40S",
        Some((store_a, "node-a:7402")),
    )
    .await;
    let (_db, sock_b) = node_with_devices(
        &cl,
        "node-b",
        1,
        "NVIDIA L40S",
        Some((store_b, "node-b:7402")),
    )
    .await;
    let (_dc, sock_c) = node_with_devices(
        &cl,
        "node-c",
        1,
        "NVIDIA L40S",
        Some((store_c, "node-c:7402")),
    )
    .await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let w1 = common::register(&sock_b, 1, 0, Role::Worker).await;
    let mut spare = common::register(&sock_c, 1, 0, Role::Spare).await;
    common::wait_members(&cl.coord, 2).await;
    assert!(saw(&seen_b, |r| r.rank == 1 && !r.targets.is_empty()).await);
    assert!(
        seen_b
            .lock()
            .await
            .iter()
            .any(|r| r.rank == 1 && r.targets == vec!["node-c:7402".to_owned()]),
        "node-b's store replicates rank 1 to node-c before the loss: {:?}",
        seen_b.lock().await
    );

    // rank 0 reports rank 1 lost; the spare on node-c takes the position
    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![1],
            reason: Reason::Fault as i32,
            detail: "process exited".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    drop(w1);
    expect_change(&mut w0, 0, 1, 1).await;
    expect_change(&mut spare, 1, 1, 1).await;
    assert!(saw(&seen_b, |r| r.rank == 1 && r.targets.is_empty()).await);
    let b = seen_b.lock().await.clone();
    assert!(
        b.last()
            .is_some_and(|r| r.rank == 1 && r.targets.is_empty()),
        "node-b's store was not told rank 1 replicates nowhere: {b:?}"
    );
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.members.get(&1).map(String::as_str), Some("node-c"));
    assert_eq!(
        view.replicating_from.get(&1).map(String::as_str),
        Some("node-c")
    );
    // node-c's store is fenced at the epoch of the move: the clear carries epoch 1
    assert!(saw(&expect_c, |e| e.rank == 1 && e.epoch == 1).await);
    let fences = expect_c.lock().await.clone();
    assert!(
        fences
            .iter()
            .any(|e| e.rank == 1 && e.from_node.is_empty() && e.epoch == 1),
        "node-c's store was not fenced at epoch 1 after the move: {fences:?}"
    );
    // node-c holds rank 1 now and no spare stands in for it: nothing to replicate to
    assert!(
        seen_c.lock().await.iter().all(|r| r.targets.is_empty()),
        "{:?}",
        seen_c.lock().await
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn every_member_store_is_told_to_replicate_to_the_spare_node() {
    let cl = cluster("policy-two-node.json").await;
    let (store_a, seen_a, _expect_a) = fake_store(cl.dir.path(), "node-a");
    let (store_b, seen_b, _expect_b) = fake_store(cl.dir.path(), "node-b");
    let (store_c, seen_c, _expect_c) = fake_store(cl.dir.path(), "node-c");
    let (_da, sock_a) = node_with_devices(
        &cl,
        "node-a",
        4,
        "NVIDIA H100 80GB HBM3",
        Some((store_a, "node-a:7001")),
    )
    .await;
    let (_db, sock_b) = node_with_devices(
        &cl,
        "node-b",
        4,
        "NVIDIA H100 80GB HBM3",
        Some((store_b, "node-b:7001")),
    )
    .await;
    // the members register before the spare node is up: their targets are empty for now
    let mut workers = Vec::new();
    for r in 0..4 {
        workers.push(common::register(&sock_a, r, r, Role::Worker).await);
        workers.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    common::wait_members(&cl.coord, 8).await;
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(
        view.classes.len(),
        8,
        "every rank's class is known: {view:?}"
    );
    assert!(view.replication.values().all(Vec::is_empty), "{view:?}");

    // the spare node links with its store's replica address: every member's store hears of it
    let (_dc, sock_c) = node_with_devices(
        &cl,
        "node-c",
        4,
        "NVIDIA H100 80GB HBM3",
        Some((store_c, "node-c:7001")),
    )
    .await;
    let want = vec!["node-c:7001".to_owned()];
    for _ in 0..200 {
        if seen_a.lock().await.len() == 4 && seen_b.lock().await.len() == 4 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let a = seen_a.lock().await.clone();
    let b = seen_b.lock().await.clone();
    assert_eq!(a.len(), 4, "node-a's store heard four targets: {a:?}");
    assert_eq!(b.len(), 4, "node-b's store heard four targets: {b:?}");
    let mut ranks: Vec<u32> = a.iter().chain(b.iter()).map(|r| r.rank).collect();
    ranks.sort_unstable();
    assert_eq!(ranks, (0..8).collect::<Vec<u32>>());
    assert!(
        a.iter()
            .chain(b.iter())
            .all(|r| r.engine == "k3-a" && r.targets == want)
    );
    assert!(
        seen_c.lock().await.is_empty(),
        "the spare node's own store replicates nowhere"
    );
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert!(view.replication.values().all(|t| *t == want), "{view:?}");

    // spares on the spare node join the pool at full speed, and the pool says so
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(common::register(&sock_c, r, r, Role::Spare).await);
    }
    for _ in 0..200 {
        if cl.coord.pool("k3-a").await.len() == 4 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let pool = cl.coord.pool("k3-a").await;
    assert_eq!(pool.len(), 4, "{pool:?}");
    assert!(
        pool.iter()
            .all(|s| s.node == "node-c" && (s.speed - 1.0).abs() < f64::EPSILON)
    );
    assert_eq!(cl.coord.topology().await.devices().count(), 12);
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_spare_whose_node_holds_a_replica_restores_there_with_nothing_staged() {
    // rank 1 replicates to node-c and node-d, and its spare is on node-c. The spare's own store
    // holds the rows, so nothing is staged: staging them from node-d would point node-c's store
    // at a new source mid-stream, and the spare would wait for it to catch up (measured: 1.48 s)
    let mut policy = policy::load(&common::example("policy-three-node.json")).expect("policy");
    policy.engines[0].replica_nodes = vec!["node-c".into(), "node-d".into()];
    let cl = cluster_of(policy).await;
    let mut nodes = Vec::new();
    let mut seen = Vec::new();
    let mut expected = Vec::new();
    for node in ["node-a", "node-b", "node-c", "node-d"] {
        let (store, replicate, expect) = fake_store(cl.dir.path(), node);
        let addr = format!("{node}:7402");
        let (_d, sock) =
            node_with_devices(&cl, node, 1, "NVIDIA L40S", Some((store, addr.as_str()))).await;
        nodes.push(sock);
        seen.push(replicate);
        expected.push(expect);
    }
    let mut w0 = common::register(&nodes[0], 0, 0, Role::Worker).await;
    let w1 = common::register(&nodes[1], 1, 0, Role::Worker).await;
    let mut spare = common::register(&nodes[2], 1, 0, Role::Spare).await;
    common::wait_members(&cl.coord, 2).await;
    // both replica nodes take rank 1's rows from node-b
    assert!(saw(&expected[2], |e| e.rank == 1 && e.from_node == "node-b").await);
    assert!(saw(&expected[3], |e| e.rank == 1 && e.from_node == "node-b").await);

    let lost = Signal {
        kind: Some(signal::Kind::RankLost(RankLost {
            ranks: vec![1],
            reason: Reason::Fault as i32,
            detail: "process exited".into(),
            epoch: 0,
        })),
    };
    frame::send(&mut w0, &lost).await.expect("report");
    drop(w1);
    expect_change(&mut w0, 0, 1, 1).await;
    expect_change(&mut spare, 1, 1, 1).await;
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert!(view.restoring.is_empty(), "nothing was staged: {view:?}");
    assert!(
        !expected[2]
            .lock()
            .await
            .iter()
            .any(|e| e.rank == 1 && e.from_node == "node-d"),
        "node-c's store was pointed at node-d mid-stream: {:?}",
        expected[2].lock().await
    );
    assert!(
        !seen[3]
            .lock()
            .await
            .iter()
            .any(|r| r.rank == 1 && r.targets.iter().any(|t| t == "node-c:7402")),
        "node-d's store was told to carry rank 1 to node-c: {:?}",
        seen[3].lock().await
    );
    cl.cancel.cancel();
}
