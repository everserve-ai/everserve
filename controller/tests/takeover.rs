// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A coordinator that takes over from another (decision 0036, amended 2026-10-15), with the
//! daemons' links held back by the test: a rank its daemon relinks within the allowance is no
//! change; one never relinked is lost at the allowance after the first hello; a coordinator no
//! daemon reaches says it is unreachable and changes nothing; an engine that never held every
//! rank, or went down, has no kept membership and loses nothing; and a head lost before the
//! ranks' daemons relinked waits for them, then reaches every rank.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use everserve_controller::coord::Coordinator;
use everserve_controller::node::Daemon;
use everserve_controller::policy;
use everserve_controller::proto::everserve::policy::v1::Policy;
use everserve_controller::proto::everserve::signals::v1::{Ack, Role, Signal, signal};
use everserve_controller::proto::everserve::spec::v1::resource_status;
use everserve_controller::readiness::EngineReadiness;
use everserve_controller::spec::{FileStore, SpecStore};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::{Address, Listener};
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

const H100: &str = "NVIDIA H100 80GB HBM3";

/// A daemon serving its socket, whose link to the coordinator the test starts and cuts.
struct Node {
    daemon: Arc<Daemon>,
    sock: PathBuf,
    link: CancellationToken,
}

impl Node {
    fn link(&mut self, address: &Address) {
        self.link = CancellationToken::new();
        tokio::spawn(Arc::clone(&self.daemon).link(
            address.clone(),
            None,
            "coordinator".into(),
            self.link.clone(),
        ));
    }

    fn cut(&self) {
        self.link.cancel();
    }
}

async fn node(dir: &Path, name: &str, devices: u32, cancel: &CancellationToken) -> Node {
    let daemon = common::new_daemon(name, common::devices(name, devices, H100)).await;
    let sock = dir.join(format!("{name}.sock"));
    tokio::spawn(Arc::clone(&daemon).serve(sock.clone(), cancel.clone()));
    tokio::time::sleep(Duration::from_millis(30)).await;
    Node {
        daemon,
        sock,
        link: CancellationToken::new(),
    }
}

/// A coordinator over the file store at `store`, serving at `coord.sock` in `dir`, with the
/// takeover's allowance and unreachable bound given: it, its telemetry and its stop.
async fn coordinator(
    dir: &Path,
    policy: &Policy,
    store: &Path,
    allowance: Duration,
    unreachable: Duration,
) -> (Arc<Coordinator>, Telemetry, CancellationToken) {
    let telemetry = Telemetry::new();
    let (coord, _rx) = Coordinator::with_store(
        policy.clone(),
        &telemetry,
        Arc::new(FileStore::new(store)) as Arc<dyn SpecStore>,
        0,
    )
    .await
    .expect("store");
    coord.set_takeover_allowance(allowance);
    coord.set_unreachable_after(unreachable);
    let cancel = CancellationToken::new();
    let listener = Listener::bind(&address(dir), None).await.expect("bind");
    tokio::spawn(Arc::clone(&coord).serve(listener, cancel.clone()));
    (coord, telemetry, cancel)
}

fn address(dir: &Path) -> Address {
    Address::Unix(dir.join("coord.sock"))
}

/// The members kept for the engine, as the file store holds them.
fn kept(store: &Path) -> usize {
    let snapshot = FileStore::new(store).read().expect("read");
    match snapshot
        .statuses
        .get(&everserve_controller::spec::engine_key(common::ENGINE))
        .and_then(|s| s.kind.clone())
    {
        Some(resource_status::Kind::Engine(e)) => e.members.len(),
        _ => 0,
    }
}

async fn until(what: &str, ok: impl Fn() -> bool) {
    for _ in 0..500 {
        if ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("never {what}");
}

async fn engine(coord: &Coordinator) -> Option<EngineReadiness> {
    coord.readiness().await.engines.into_iter().next()
}

async fn seen(coord: &Coordinator, what: &str, ok: impl Fn(&EngineReadiness) -> bool) {
    for _ in 0..1000 {
        if engine(coord).await.is_some_and(|e| ok(&e)) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("never {what}: {:?}", coord.readiness().await);
}

fn unheard_lost(t: &Telemetry) -> Option<String> {
    t.render()
        .lines()
        .find(|l| l.starts_with("everserve_unheard_ranks_lost_total{"))
        .map(str::to_owned)
}

/// The cell under a first coordinator, which keeps the engine's membership once all eight ranks
/// registered: the nodes, linked, the workers, and the first coordinator's stop.
struct Cell {
    dir: tempfile::TempDir,
    store: PathBuf,
    policy: Policy,
    cancel: CancellationToken,
    a: Node,
    b: Node,
    /// Held open, so the ranks stay registered while the cell lives.
    _workers: Vec<UnixStream>,
}

async fn served(policy: Policy) -> (Cell, Arc<Coordinator>, CancellationToken) {
    let dir = tempfile::tempdir().expect("tmp");
    let store = dir.path().join("spec.json");
    let cancel = CancellationToken::new();
    let (first, _t, first_cancel) = coordinator(
        dir.path(),
        &policy,
        &store,
        everserve_controller::coord::default_allowance(),
        everserve_controller::coord::UNREACHABLE_AFTER,
    )
    .await;
    let mut a = node(dir.path(), "node-a", 4, &cancel).await;
    let mut b = node(dir.path(), "node-b", 4, &cancel).await;
    a.link(&address(dir.path()));
    b.link(&address(dir.path()));
    let mut workers = Vec::new();
    for r in 0..4 {
        workers.push(common::register(&a.sock, r, r, Role::Worker).await);
    }
    for r in 0..4 {
        workers.push(common::register(&b.sock, 4 + r, r, Role::Worker).await);
    }
    until("the membership kept", || kept(&store) == 8).await;
    (
        Cell {
            dir,
            store,
            policy,
            cancel,
            a,
            b,
            _workers: workers,
        },
        first,
        first_cancel,
    )
}

fn two_node() -> Policy {
    policy::load(&common::example("policy-two-node.json")).expect("policy")
}

#[tokio::test]
async fn a_rank_whose_daemon_relinks_within_the_allowance_is_no_change() {
    let (cell, _first, first_cancel) = served(two_node()).await;
    let (mut a, mut b) = (cell.a, cell.b);
    a.cut();
    b.cut();
    first_cancel.cancel();
    let allowance = Duration::from_millis(1500);
    let (coord, telemetry, cancel) = coordinator(
        cell.dir.path(),
        &cell.policy,
        &cell.store,
        allowance,
        Duration::from_mins(1),
    )
    .await;
    // the daemons' path opens a second after the coordinator listens (a Service's endpoints
    // moving, say): the clock starts at the first hello, not at the listening
    tokio::time::sleep(Duration::from_secs(1)).await;
    a.link(&address(cell.dir.path()));
    // node-b is slow, within the allowance from node-a's hello though past it from the listening
    tokio::time::sleep(Duration::from_millis(700)).await;
    b.link(&address(cell.dir.path()));
    tokio::time::sleep(allowance + Duration::from_millis(500)).await;
    let e = engine(&coord).await.expect("engine");
    assert_eq!((e.epoch, e.members.len()), (0, 8), "{e:?}");
    assert!(e.change.is_none(), "{e:?}");
    assert_eq!(unheard_lost(&telemetry), None);
    cancel.cancel();
    cell.cancel.cancel();
}

#[tokio::test]
async fn a_rank_whose_daemon_never_relinks_is_lost_at_the_allowance_after_the_first_hello() {
    let (cell, _first, first_cancel) = served(two_node()).await;
    let (mut a, b) = (cell.a, cell.b);
    a.cut();
    b.cut();
    first_cancel.cancel();
    let allowance = Duration::from_millis(500);
    let (coord, telemetry, cancel) = coordinator(
        cell.dir.path(),
        &cell.policy,
        &cell.store,
        allowance,
        Duration::from_mins(1),
    )
    .await;
    a.link(&address(cell.dir.path()));
    // node-b never relinks: its ranks are lost once the allowance after node-a's hello passed
    until("node-b's ranks lost", || {
        unheard_lost(&telemetry).is_some_and(|l| l.ends_with(" 4"))
    })
    .await;
    let events = coord
        .engine_events(common::ENGINE)
        .await
        .unwrap_or_default();
    assert!(
        events.iter().any(|e| e.reason == "RanksUnheard"),
        "{events:?}"
    );
    cancel.cancel();
    cell.cancel.cancel();
}

#[tokio::test]
async fn a_coordinator_no_daemon_reaches_says_it_is_unreachable_and_changes_nothing() {
    let (cell, _first, first_cancel) = served(two_node()).await;
    cell.a.cut();
    cell.b.cut();
    first_cancel.cancel();
    let (coord, telemetry, cancel) = coordinator(
        cell.dir.path(),
        &cell.policy,
        &cell.store,
        Duration::from_millis(200),
        Duration::from_millis(300),
    )
    .await;
    // the sample's own line: the metric's help text begins with "1 while" too
    until("the coordinator unreachable", || {
        telemetry
            .render()
            .lines()
            .any(|l| l == "everserve_coordinator_unreachable 1")
    })
    .await;
    assert_eq!(unheard_lost(&telemetry), None);
    assert!(
        engine(&coord)
            .await
            .is_none_or(|e| e.epoch == 0 && e.change.is_none())
    );
    cancel.cancel();
    cell.cancel.cancel();
}

#[tokio::test]
async fn an_engine_still_registering_when_a_coordinator_starts_loses_nothing() {
    // nothing kept: the engine never held every rank, and its ranks still coming are not lost
    let dir = tempfile::tempdir().expect("tmp");
    let store = dir.path().join("spec.json");
    let cancel = CancellationToken::new();
    let (coord, telemetry, coord_cancel) = coordinator(
        dir.path(),
        &two_node(),
        &store,
        Duration::from_millis(300),
        Duration::from_mins(1),
    )
    .await;
    let mut a = node(dir.path(), "node-a", 4, &cancel).await;
    a.link(&address(dir.path()));
    let mut workers = Vec::new();
    for r in 0..4 {
        workers.push(common::register(&a.sock, r, r, Role::Worker).await);
    }
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert_eq!(kept(&store), 0);
    assert_eq!(unheard_lost(&telemetry), None);
    assert!(
        engine(&coord)
            .await
            .is_some_and(|e| e.epoch == 0 && e.change.is_none())
    );
    coord_cancel.cancel();
    cancel.cancel();
}

#[tokio::test]
async fn an_engine_that_went_down_keeps_no_membership_and_a_successor_loses_nothing_of_it() {
    // the kept membership is its incarnation's: once the engine is down, a new incarnation's
    // ranks still registering are not the old one's unheard
    let (cell, first, first_cancel) = served(two_node()).await;
    let head = common::register_head(&cell.a.sock, std::process::id()).await;
    seen(&first, "the head", |e| e.head.is_some()).await;
    drop(head);
    seen(&first, "the engine down", |e| e.down.is_some()).await;
    until("the membership cleared", || kept(&cell.store) == 0).await;
    let (mut a, b) = (cell.a, cell.b);
    a.cut();
    b.cut();
    first_cancel.cancel();
    let (_coord, telemetry, cancel) = coordinator(
        cell.dir.path(),
        &cell.policy,
        &cell.store,
        Duration::from_millis(300),
        Duration::from_mins(1),
    )
    .await;
    a.link(&address(cell.dir.path()));
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert_eq!(unheard_lost(&telemetry), None);
    cancel.cancel();
    cell.cancel.cancel();
}

/// Acknowledge, as `rank`, every change `w` is sent until it closes; keeps `w` open meanwhile.
async fn ack_any(mut w: UnixStream, rank: u32) {
    while let Ok(sig) = common::recv(&mut w).await {
        if let Some(signal::Kind::Change(c)) = sig.kind {
            let ack = Signal {
                kind: Some(signal::Kind::Ack(Ack {
                    rank,
                    epoch: c.epoch,
                    ..Ack::default()
                })),
            };
            if frame::send(&mut w, &ack).await.is_err() {
                return;
            }
        }
    }
}

/// Under a first coordinator, node-a's four ranks, three of node-b's (bring-up under way, rank 7
/// not yet), and spares for node-a's group on node-c; then a takeover with node-b not relinked,
/// and a loss of rank 0 the successor changes on what it knows. Its coordinator, the cell's
/// stop, node-b, and node-b's workers.
async fn changed_without_node_b(
    dir: &Path,
) -> (Arc<Coordinator>, CancellationToken, Node, Vec<UnixStream>) {
    let store = dir.join("spec.json");
    let cancel = CancellationToken::new();
    let policy = two_node();
    let forever = Duration::from_mins(1);
    let (_first, _t, first_cancel) = coordinator(dir, &policy, &store, forever, forever).await;
    let mut a = node(dir, "node-a", 4, &cancel).await;
    let mut b = node(dir, "node-b", 4, &cancel).await;
    let mut c = node(dir, "node-c", 4, &cancel).await;
    for n in [&mut a, &mut b, &mut c] {
        n.link(&address(dir));
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let mut on_a = Vec::new();
    for r in 0..4 {
        on_a.push(common::register(&a.sock, r, r, Role::Worker).await);
    }
    let mut on_b = Vec::new();
    for r in 0..3 {
        on_b.push(common::register(&b.sock, 4 + r, r, Role::Worker).await);
    }
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(common::register(&c.sock, r, r, Role::Spare).await);
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(kept(&store), 0, "bring-up: nothing kept");
    a.cut();
    b.cut();
    c.cut();
    first_cancel.cancel();
    let (coord, _t, coord_cancel) = coordinator(dir, &policy, &store, forever, forever).await;
    a.link(&address(dir));
    c.link(&address(dir));
    seen(&coord, "node-a's ranks and the spares", |e| {
        e.members.len() == 4 && !e.spares.is_empty()
    })
    .await;
    drop(on_a.remove(0));
    for (i, w) in on_a.into_iter().enumerate() {
        tokio::spawn(ack_any(w, u32::try_from(i + 1).unwrap_or(0)));
    }
    for (r, w) in spares.into_iter().enumerate() {
        tokio::spawn(ack_any(w, u32::try_from(r).unwrap_or(0)));
    }
    seen(&coord, "epoch 1 committed without node-b", |e| {
        e.epoch == 1 && e.change.is_none()
    })
    .await;
    // the cell's daemons stop with the coordinator's stop, at the test's end
    let stop = CancellationToken::new();
    let (c1, c2, s) = (cancel, coord_cancel, stop.clone());
    tokio::spawn(async move {
        s.cancelled().await;
        c1.cancel();
        c2.cancel();
    });
    (coord, stop, b, on_b)
}

#[tokio::test]
async fn ranks_a_node_registered_before_a_commit_it_never_heard_are_not_members_and_are_told_it() {
    // a takeover mid bring-up: the successor commits epoch 1 on the ranks it knows, and node-b,
    // whose ranks 4 to 6 registered at epoch 0, relinks after it. Taken as members, they served
    // on with epoch 0's groups beside epoch 1, told nothing
    let dir = tempfile::tempdir().expect("tmp");
    let (coord, stop, mut b, mut on_b) = changed_without_node_b(dir.path()).await;
    b.link(&address(dir.path()));
    let mut told = 0;
    for w in &mut on_b {
        let commit = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let sig = common::recv(w).await.expect("signal");
                if let Some(signal::Kind::MembershipCommitted(m)) = sig.kind {
                    return m;
                }
            }
        })
        .await
        .expect("node-b's workers told the commit they missed");
        assert_eq!(commit.epoch, 1);
        assert!(
            commit.members.iter().all(|m| m.node != "node-b"),
            "{commit:?}"
        );
        told += 1;
    }
    assert_eq!(told, 3);
    let e = engine(&coord).await.expect("engine");
    assert!(
        e.members.iter().all(|m| m.node == "node-c") && e.members.len() == 4,
        "node-b's stale ranks counted as members: {:?}",
        e.members
    );
    let events = coord
        .engine_events(common::ENGINE)
        .await
        .unwrap_or_default();
    assert!(
        events.iter().any(|ev| ev.reason == "StaleRanks"),
        "{events:?}"
    );
    stop.cancel();
}
