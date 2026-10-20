// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A change its coordinator died in (decision 0038): the successor delivers it again and commits
//! it, a commit logged before any daemon heard it reaches them, a committed change found in the
//! log changes nothing, a loss reported meanwhile waits for it, and a change whose receiver left
//! is abandoned.

mod common;

use std::path::Path;
use std::sync::Arc;

use everserve_controller::coord::{Cause, ChangeError, Coordinator};
use everserve_controller::policy;
use everserve_controller::proto::everserve::signals::v1::{
    Change, MembershipCommitted, Outcome, Role, Signal, signal,
};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::Address;
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

use common::cluster::{Cluster, ack, expect_change, two_node_engine, wait_view, within};

/// A coordinator under the two-node policy keeping its state in `store`, serving at `dir`'s
/// socket until `cancel` fires.
async fn coordinator_on(
    store: &Arc<everserve_controller::spec::FileStore>,
    dir: &Path,
    cancel: &CancellationToken,
) -> (Arc<Coordinator>, Address) {
    let policy = policy::load(&common::example("policy-two-node.json")).expect("policy");
    let store: Arc<dyn everserve_controller::spec::SpecStore> = store.clone();
    let (coord, _rx) = Coordinator::with_store(policy, &Telemetry::new(), store, 0)
        .await
        .expect("coordinator");
    let address = common::serve(&coord, dir, cancel).await;
    (coord, address)
}

/// The change `store` logs for k3-a.
fn logged(
    store: &everserve_controller::spec::FileStore,
) -> Option<everserve_controller::proto::everserve::spec::v1::ChangeLog> {
    use everserve_controller::proto::everserve::spec::v1::resource_status;
    match store
        .read()
        .expect("read")
        .statuses
        .get(&everserve_controller::spec::engine_key("k3-a"))
        .and_then(|s| s.kind.clone())
    {
        Some(resource_status::Kind::Engine(e)) => e.change,
        None => None,
    }
}

/// Read the change rank `rank`'s process is sent, and acknowledge nothing.
async fn read_change(w: &mut UnixStream, rank: u32) -> Change {
    let sig: Signal = within("a change", common::recv(w)).await.expect("change");
    let Some(signal::Kind::Change(c)) = sig.kind else {
        panic!("rank {rank}: expected a change, got {sig:?}")
    };
    c
}

/// A coordinator that died mid-change: it logged and delivered the change moving node-b's group
/// to node-c's spares after rank 5's loss, and no process acknowledged it.
struct DiedMidChange {
    store: Arc<everserve_controller::spec::FileStore>,
    cl: Cluster,
    on_a: Vec<UnixStream>,
    spares: Vec<UnixStream>,
    /// A client of the engine on node-a, which hears every commit.
    client: UnixStream,
    /// The change as node-a's processes were sent it.
    sent: Change,
}

async fn died_mid_change() -> DiedMidChange {
    let dir = tempfile::tempdir().expect("tmp");
    let store = Arc::new(everserve_controller::spec::FileStore::new(
        dir.path().join("spec.pb"),
    ));
    let first_cancel = CancellationToken::new();
    let (first, address) = coordinator_on(&store, dir.path(), &first_cancel).await;
    let cl = Cluster {
        coord: first,
        coord_telemetry: Telemetry::new(),
        cancel: CancellationToken::new(), // the daemons', apart from the coordinator's
        address,
        dir,
    };
    let (mut on_a, _on_b, mut spares, (sock_a, _, _)) = two_node_engine(&cl).await;
    let client = common::register(&sock_a, 0, 0, Role::Client).await;
    let change = {
        let c = Arc::clone(&cl.coord);
        tokio::spawn(async move { c.rank_lost("k3-a", 5, Cause::Fault).await })
    };
    let mut sent = Vec::new();
    for (rank, w) in (0u32..).zip(on_a.iter_mut()) {
        sent.push(read_change(w, rank).await);
    }
    for (rank, sp) in (4u32..).zip(spares.iter_mut()) {
        read_change(sp, rank).await;
    }
    let log = logged(&store).expect("the change logged before its delivery");
    assert_eq!(
        log.change.as_ref(),
        Some(&sent[0]),
        "the change logged is the one sent"
    );
    assert!(log.committed.is_none(), "not committed");
    first_cancel.cancel();
    change.abort();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    DiedMidChange {
        store,
        cl,
        on_a,
        spares,
        client,
        sent: sent.swap_remove(0),
    }
}

#[tokio::test]
async fn a_change_its_coordinator_died_in_is_delivered_again_and_committed_by_its_successor() {
    // decision 0038: the coordinator logs a change before it delivers it; it dies with the change
    // delivered and unacknowledged; its successor finds the change in the store, delivers the
    // same change again, byte for byte, and commits it, rather than claiming another under its
    // epoch
    let DiedMidChange {
        store,
        cl,
        mut on_a,
        mut spares,
        sent,
        ..
    } = died_mid_change().await;
    let (second, _) = coordinator_on(&store, cl.dir.path(), &CancellationToken::new()).await;
    for (rank, w) in (0u32..).zip(on_a.iter_mut()) {
        let again = within("the change again", expect_change(w, rank, 1, 1)).await;
        assert_eq!(
            again, sent,
            "rank {rank} is sent the logged change, byte for byte"
        );
    }
    for (rank, sp) in (4u32..).zip(spares.iter_mut()) {
        within("the change again", expect_change(sp, rank, 1, 1)).await;
    }
    let view = wait_view(&second, "the successor's commit", |v| {
        v.committed_epoch == 1
    })
    .await;
    assert_eq!(view.epoch, 1, "no other change claimed under or beside it");
    for r in 4..8 {
        assert_eq!(
            view.members.get(&r).map(String::as_str),
            Some("node-c"),
            "{view:?}"
        );
    }
    // the view takes the commit as the store is written, and daemons hear it only once the store
    // holds it (decision 0038): the log is waited for, not read at the view's word
    within("its commit logged", async {
        while logged(&store).is_none_or(|l| l.committed.is_none()) {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;
    cl.cancel.cancel();
}

/// Read `client`'s signals until a commit.
async fn expect_commit(client: &mut UnixStream) -> MembershipCommitted {
    loop {
        let sig: Signal = common::recv(client).await.expect("frame");
        if let Some(signal::Kind::MembershipCommitted(m)) = sig.kind {
            return m;
        }
    }
}

/// Log in `store` the commit of the change it logs, as a coordinator does before any daemon
/// hears it: node-b's ranks moved to node-c's spares.
async fn log_commit(store: &everserve_controller::spec::FileStore) {
    use everserve_controller::proto::everserve::coord::v1::Committed;
    use everserve_controller::proto::everserve::signals::v1::RankPlacement;
    use everserve_controller::proto::everserve::spec::v1::{ResourceStatus, resource_status};
    use everserve_controller::spec::SpecStore as _;
    let key = everserve_controller::spec::engine_key("k3-a");
    let snapshot = store.read().expect("read");
    let Some(resource_status::Kind::Engine(mut status)) =
        snapshot.statuses.get(&key).and_then(|s| s.kind.clone())
    else {
        panic!("no status for k3-a")
    };
    let log = status.change.as_mut().expect("a logged change");
    let placed = |rank: u32, node: &str, device: u32| RankPlacement {
        rank,
        node: node.into(),
        device: Some(device),
        ..RankPlacement::default()
    };
    log.committed = Some(Committed {
        engine: "k3-a".into(),
        outcome: Outcome::Replace as i32,
        lost: vec![4, 5, 6, 7],
        membership: Some(MembershipCommitted {
            epoch: 1,
            world_size: 8,
            active_ranks: (0..8).collect(),
            members: (0..4)
                .map(|r| placed(r, "node-a", r))
                .chain((4..8).map(|r| placed(r, "node-c", r - 4)))
                .collect(),
            ..MembershipCommitted::default()
        }),
        take: Vec::new(),
    });
    store
        .put_status(
            key,
            ResourceStatus {
                kind: Some(resource_status::Kind::Engine(status)),
            },
            snapshot.term,
        )
        .await
        .expect("the commit logged");
}

#[tokio::test]
async fn a_commit_logged_before_any_daemon_heard_it_reaches_them_from_the_successor() {
    // decision 0038: the coordinator logs the commit, then dies before any daemon hears it; its
    // successor sends the logged commit to every daemon of the engine, and delivers nothing again
    let DiedMidChange {
        store,
        cl,
        mut on_a,
        mut client,
        ..
    } = died_mid_change().await;
    log_commit(&store).await;
    let (second, _) = coordinator_on(&store, cl.dir.path(), &CancellationToken::new()).await;
    let committed = within("the logged commit", expect_commit(&mut client)).await;
    assert_eq!(committed.epoch, 1);
    let view = wait_view(&second, "the committed view", |v| v.committed_epoch == 1).await;
    for r in 4..8 {
        assert_eq!(
            view.members.get(&r).map(String::as_str),
            Some("node-c"),
            "{view:?}"
        );
    }
    let next: Result<Result<Signal, _>, _> = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        frame::recv(&mut on_a[0]),
    )
    .await;
    assert!(
        !matches!(
            &next,
            Ok(Ok(Signal {
                kind: Some(signal::Kind::Change(_))
            }))
        ),
        "a committed change was delivered again: {next:?}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_successor_finding_a_committed_change_in_the_log_changes_nothing() {
    // decision 0038: a logged commit is sent again to every daemon of the engine, and a daemon
    // that applied it ignores it; no process is sent the change again
    let DiedMidChange {
        store,
        cl,
        mut on_a,
        mut spares,
        ..
    } = died_mid_change().await;
    let second_cancel = CancellationToken::new();
    let (second, _) = coordinator_on(&store, cl.dir.path(), &second_cancel).await;
    for (rank, w) in (0u32..).zip(on_a.iter_mut()) {
        within("the change again", expect_change(w, rank, 1, 1)).await;
    }
    for (rank, sp) in (4u32..).zip(spares.iter_mut()) {
        within("the change again", expect_change(sp, rank, 1, 1)).await;
    }
    wait_view(&second, "committed", |v| v.committed_epoch == 1).await;
    // that successor dies too, after the commit; a third finds the change committed in the log
    second_cancel.cancel();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let third_cancel = CancellationToken::new();
    let (third, _) = coordinator_on(&store, cl.dir.path(), &third_cancel).await;
    let view = wait_view(&third, "the committed view", |v| {
        v.committed_epoch == 1 && (4..8).all(|r| v.members.get(&r).is_some_and(|n| n == "node-c"))
    })
    .await;
    assert_eq!(view.epoch, 1);
    assert!(view.down.is_none());
    let next: Result<Result<Signal, _>, _> = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        frame::recv(&mut on_a[0]),
    )
    .await;
    assert!(
        !matches!(
            &next,
            Ok(Ok(Signal {
                kind: Some(signal::Kind::Change(_))
            }))
        ),
        "a committed change was delivered again: {next:?}"
    );
    third_cancel.cancel();
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_loss_reported_to_a_successor_waits_until_the_logged_change_is_finished() {
    // decision 0038: the successor claims no change of the engine before it finishes the logged
    // one, so a loss reported meanwhile is never claimed under the logged change's epoch; the
    // logged change cannot commit before its receivers acknowledge it again, so the loss is
    // still waiting until they do
    let DiedMidChange {
        store,
        cl,
        mut on_a,
        mut spares,
        sent,
        ..
    } = died_mid_change().await;
    let (second, _) = coordinator_on(&store, cl.dir.path(), &CancellationToken::new()).await;
    let loss = {
        let c = Arc::clone(&second);
        tokio::spawn(async move { c.rank_lost("k3-a", 1, Cause::Fault).await })
    };
    let first = within("the logged change", read_change(&mut on_a[0], 0)).await;
    assert_eq!(first, sent, "the logged change goes out before any other");
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(
        !loss.is_finished(),
        "the loss was decided while the logged change was unfinished"
    );
    ack(&mut on_a[0], 0, 1).await;
    for (rank, w) in (1u32..).zip(on_a.iter_mut().skip(1)) {
        within("the logged change", expect_change(w, rank, 1, 1)).await;
    }
    for (rank, sp) in (4u32..).zip(spares.iter_mut()) {
        within("the logged change", expect_change(sp, rank, 1, 1)).await;
    }
    // decided against the engine after the logged change: rank 1 has no spare left
    let answer = within("the loss answered", loss).await.expect("join");
    assert!(matches!(answer, Err(ChangeError::Rebuild(1))), "{answer:?}");
    assert_eq!(
        second
            .engine("k3-a")
            .await
            .map(|v| (v.epoch, v.committed_epoch)),
        Some((1, 1)),
        "nothing was claimed beside the logged change"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_change_whose_receiver_left_while_its_coordinator_was_down_is_abandoned() {
    // decision 0038: a logged change is sent again only to the receivers it names; rank 4's spare
    // left while no coordinator held the lease, so the change cannot be sent again, and a change
    // placed afresh under its epoch would be another change: it fails as one does after its
    // delivery (0032), and every process hears its epoch is spent
    let DiedMidChange {
        store,
        cl,
        mut on_a,
        mut spares,
        ..
    } = died_mid_change().await;
    drop(spares.remove(0));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let (second, _) = coordinator_on(&store, cl.dir.path(), &CancellationToken::new()).await;
    // not sent again first: the first word node-a's processes hear is that the epoch is spent
    let sig: Signal = within("the change abandoned", common::recv(&mut on_a[0]))
        .await
        .expect("frame");
    let Some(signal::Kind::Abandoned(abandoned)) = sig.kind else {
        panic!("the change was sent again to a receiver set it no longer has: {sig:?}")
    };
    assert_eq!(abandoned.epoch, 1);
    let view = wait_view(&second, "the engine down", |v| v.down.is_some()).await;
    assert_eq!(view.epoch, 1, "its epoch is spent, not claimed again");
    assert_eq!(view.committed_epoch, 0);
    cl.cancel.cancel();
}
