// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Domain widening: a loss whose spare is outside its group's domain moves the whole group, an
//! engine whose collectives span the fabric replaces one rank across domains, and a group that
//! cannot move whole is refused with the reason.

mod common;

use std::sync::Arc;

use everserve_controller::coord::{Cause, ChangeError, Coordinator};
use everserve_controller::policy;
use everserve_controller::proto::everserve::signals::v1::{Outcome, Role};
use everserve_telemetry::Telemetry;
use everserve_wire::transport::{Address, Listener};
use tokio_util::sync::CancellationToken;

use common::cluster::{Cluster, cluster, expect_change, two_node_engine};

#[tokio::test]
async fn a_loss_whose_spare_is_outside_its_groups_domain_moves_the_whole_group() {
    // decision 0022: TP4 groups 0-3 (node-a) and 4-7 (node-b); the spares for 4-7 are on node-c
    let cl = cluster("policy-two-node.json").await;
    let (mut on_a, _on_b, mut spares, _) = two_node_engine(&cl).await;
    let change = {
        let c = Arc::clone(&cl.coord);
        tokio::spawn(async move { c.rank_lost("k3-a", 5, Cause::Fault).await })
    };
    // every survivor outside the group and every spare of the group hears one change for the
    // whole group, the alive ranks 4, 6 and 7 included
    for (rank, w) in (0u32..).zip(on_a.iter_mut()) {
        let c = expect_change(w, rank, 1, 1).await;
        assert_eq!(c.lost, vec![4, 5, 6, 7]);
        assert_eq!(c.outcome, Outcome::Replace as i32);
    }
    for (rank, s) in (4u32..).zip(spares.iter_mut()) {
        let c = expect_change(s, rank, 1, 1).await;
        assert_eq!(c.lost, vec![4, 5, 6, 7]);
    }
    let committed = change.await.expect("join").expect("the group moves");
    assert_eq!(committed.epoch, 1);
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert!(
        (4..8).all(|r| view.members.get(&r).map(String::as_str) == Some("node-c")),
        "{view:?}"
    );
    assert!(view.spares.is_empty());
    cl.cancel.cancel();
}

#[tokio::test]
async fn an_engine_whose_collectives_span_the_fabric_replaces_one_rank_across_domains() {
    let dir = tempfile::tempdir().expect("tmp");
    let telemetry = Telemetry::new();
    let mut policy = policy::load(&common::example("policy-two-node.json")).expect("policy");
    policy.engines[0].fabric_collectives = true;
    let (coord, _rx) = Coordinator::new(policy, &telemetry);
    let address = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    let cancel = CancellationToken::new();
    tokio::spawn(Arc::clone(&coord).serve(listener, cancel.clone()));
    let cl = Cluster {
        coord,
        coord_telemetry: telemetry,
        cancel,
        address,
        dir,
    };
    let (mut on_a, mut on_b, mut spares, _) = two_node_engine(&cl).await;
    let change = {
        let c = Arc::clone(&cl.coord);
        tokio::spawn(async move { c.rank_lost("k3-a", 5, Cause::Fault).await })
    };
    for (rank, w) in (0u32..).zip(on_a.iter_mut()) {
        assert_eq!(expect_change(w, rank, 1, 1).await.lost, vec![5]);
    }
    for (i, w) in on_b.iter_mut().enumerate() {
        if i == 1 {
            continue; // rank 5 is the dead one
        }
        let rank = 4 + u32::try_from(i).unwrap();
        assert_eq!(expect_change(w, rank, 1, 1).await.lost, vec![5]);
    }
    assert_eq!(expect_change(&mut spares[1], 5, 1, 1).await.lost, vec![5]);
    change.await.expect("join").expect("one rank moves");
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.members.get(&5).map(String::as_str), Some("node-c"));
    assert_eq!(view.spares.len(), 3);
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_group_that_cannot_move_whole_is_refused_with_the_reason() {
    let cl = cluster("policy-two-node.json").await;
    let mut socks = Vec::new();
    for node in ["node-a", "node-b", "node-c"] {
        let (_d, sock) =
            common::daemon(cl.dir.path(), node, &cl.address, Vec::new(), &cl.cancel).await;
        socks.push(sock);
    }
    let mut on = Vec::new();
    for r in 0..4 {
        on.push(common::register(&socks[0], r, r, Role::Worker).await);
        on.push(common::register(&socks[1], 4 + r, r, Role::Worker).await);
    }
    common::wait_members(&cl.coord, 8).await;
    // only rank 5 has a spare: its group cannot move with it
    let _spare = common::register(&socks[2], 5, 1, Role::Spare).await;
    for _ in 0..300 {
        if cl.coord.pool("k3-a").await.len() == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let err = cl
        .coord
        .rank_lost("k3-a", 5, Cause::Fault)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ChangeError::Domain { rank: 5, ref reason } if reason.contains("rank 4")),
        "{err}"
    );
    assert_eq!(cl.coord.engine("k3-a").await.expect("view").epoch, 0);
    cl.cancel.cancel();
}
