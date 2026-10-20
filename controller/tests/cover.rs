// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Cover plans through the daemon (decision 0035), against a coordinator played by the test: the
//! newest plan reaches every worker and spare of the engine on the node, including one that
//! registers after it, and an older generation never does; a process's reports go up as they
//! come; the groups a process held are released for it when its connection ends; and the hello
//! after a reconnection carries the generation the daemon saw and the groups still held.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use everserve_controller::lease::BoxFuture;
use everserve_controller::node::Daemon;
use everserve_controller::proto::everserve::coord::v1::{
    CoordToDaemon, CoverReport, DaemonHello, DaemonToCoord, coord_to_daemon, cover_report,
    daemon_to_coord,
};
use everserve_controller::proto::everserve::signals::v1::{
    CoverGroup, CoverPlan, CoverSet, GroupFormed, GroupId, Holder, Role, Scope, SetMember, Signal,
    SpareRef, set_member, signal,
};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::{Address, Listener, Stream};
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

fn rank(r: u32) -> SetMember {
    SetMember {
        kind: Some(set_member::Kind::Rank(r)),
    }
}

fn spare_for(r: u32) -> SetMember {
    SetMember {
        kind: Some(set_member::Kind::Spare(SpareRef {
            for_rank: r,
            node: "node-a".into(),
            device: 2,
        })),
    }
}

fn holder(node: &str, device: u32) -> Holder {
    Holder {
        node: node.into(),
        device,
    }
}

fn plan(generation: u64) -> CoordToDaemon {
    let set = vec![rank(0), spare_for(1)];
    CoordToDaemon {
        term: 0,
        kind: Some(coord_to_daemon::Kind::CoverPlan(CoverPlan {
            engine: "k3-a".into(),
            generation,
            epoch: 0,
            placement: [(0, holder("node-a", 0)), (1, holder("node-a", 1))].into(),
            groups: vec![CoverGroup {
                scope: Scope::Domain as i32,
                spans: set.clone(),
                server: Some(rank(0)),
            }],
            sets: vec![CoverSet {
                members: set,
                groups: vec![0],
                purpose: "current".into(),
            }],
        })),
    }
}

/// The next cover report the coordinator receives over `link`, within five seconds.
async fn next_report(link: &mut Stream) -> cover_report::Kind {
    tokio::time::timeout(Duration::from_secs(5), next_report_unbounded(link))
        .await
        .expect("a report within 5 s")
}

async fn next_report_unbounded(link: &mut Stream) -> cover_report::Kind {
    loop {
        let msg: DaemonToCoord = frame::recv(link).await.expect("frame");
        if let Some(daemon_to_coord::Kind::CoverReport(CoverReport { kind: Some(k) })) = msg.kind {
            return k;
        }
    }
}

/// Accept the daemon's link and read its hello.
async fn accept(listener: &Listener) -> (Stream, DaemonHello) {
    let mut link = listener.accept().await.expect("accept").stream;
    let msg: DaemonToCoord = frame::recv(&mut link).await.expect("hello");
    let Some(daemon_to_coord::Kind::Hello(hello)) = msg.kind else {
        panic!("not a hello: {msg:?}");
    };
    (link, hello)
}

#[tokio::test]
async fn the_newest_plan_reaches_every_process_and_held_groups_are_released_with_their_process() {
    let dir = tempfile::tempdir().expect("tmp");
    let address = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    let cancel = CancellationToken::new();
    let (d, _rx) = Daemon::new("node-a", &Telemetry::new());
    let sock = dir.path().join("node-a.sock");
    tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel.clone()));
    tokio::spawn(Arc::clone(&d).link(address, None, "coordinator".into(), cancel.clone()));
    let (mut link, hello) = accept(&listener).await;
    assert!(hello.engines.is_empty(), "{hello:?}");
    for _ in 0..200 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let mut worker = reached(&sock, 0, 0, Role::Worker).await;
    frame::send(&mut link, &plan(3)).await.expect("plan 3");
    assert_eq!(common::next_plan(&mut worker).await.generation, 3);
    // an older generation is never forwarded: the next plan the worker sees is 4
    frame::send(&mut link, &plan(2)).await.expect("plan 2");
    frame::send(&mut link, &plan(4)).await.expect("plan 4");
    assert_eq!(common::next_plan(&mut worker).await.generation, 4);
    // a spare that registers while plan 4 stands receives it at once
    let mut spare = reached(&sock, 1, 2, Role::Spare).await;
    assert_eq!(common::next_plan(&mut spare).await.generation, 4);

    // each process forms its group; the reports go up as they come
    let formed = |member: SetMember| Signal {
        kind: Some(signal::Kind::GroupFormed(GroupFormed {
            engine: "k3-a".into(),
            generation: 4,
            spans: vec![rank(0), spare_for(1)],
            scope: Scope::Domain as i32,
            seconds: 0.5,
            member: Some(member),
        })),
    };
    frame::send(&mut worker, &formed(rank(0)))
        .await
        .expect("formed");
    match next_report(&mut link).await {
        cover_report::Kind::Formed(f) => assert_eq!(f.member, Some(rank(0))),
        other => panic!("{other:?}"),
    }
    frame::send(&mut spare, &formed(spare_for(1)))
        .await
        .expect("formed");
    match next_report(&mut link).await {
        cover_report::Kind::Formed(f) => assert_eq!(f.member, Some(spare_for(1))),
        other => panic!("{other:?}"),
    }

    // the spare's process ends: its group is released for it, at the generation it formed under
    drop(spare);
    match next_report(&mut link).await {
        cover_report::Kind::Released(r) => {
            assert_eq!((r.member, r.generation), (Some(spare_for(1)), 4));
            assert_eq!(r.spans, vec![rank(0), spare_for(1)]);
        }
        other => panic!("{other:?}"),
    }

    // the link drops; the daemon's next hello carries the generation it saw and what is held
    drop(link);
    let (_link, hello) = accept(&listener).await;
    let engine = hello
        .engines
        .iter()
        .find(|e| e.engine == "k3-a")
        .expect("k3-a");
    assert_eq!(engine.plan_generation, 4);
    assert_eq!(engine.held.len(), 1, "only the worker's group: {engine:?}");
    assert_eq!(engine.held[0].member, Some(rank(0)));
    cancel.cancel();
}

// ---------------------------------------------------------------------------------------------
// The coordinator's half, with cover on: a two-node TP4 x PP2 engine, a spare for each of
// node-a's ranks on node-c.

use everserve_controller::coord::{Cause, ChangeError, Coordinator, DrainState};
use everserve_controller::policy;
use everserve_controller::proto::everserve::policy::v1::CoverPolicy;
use everserve_controller::proto::everserve::signals::v1::{
    Ack, GroupFailed, GroupReleased, Reason,
};
use everserve_controller::readiness::EngineReadiness;
use everserve_controller::status::Truth;

struct Cluster {
    coord: Arc<Coordinator>,
    dir: tempfile::TempDir,
    address: Address,
    cancel: CancellationToken,
}

fn cover_policy() -> everserve_controller::proto::everserve::policy::v1::Policy {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-two-node.json");
    let mut p = policy::load(&path).expect("policy");
    p.cover = Some(CoverPolicy {
        enabled: true,
        ..CoverPolicy::default()
    });
    p
}

async fn coordinator(dir: &Path, cancel: &CancellationToken) -> (Arc<Coordinator>, Address) {
    coordinator_with(dir, cancel, cover_policy()).await
}

async fn coordinator_with(
    dir: &Path,
    cancel: &CancellationToken,
    policy: everserve_controller::proto::everserve::policy::v1::Policy,
) -> (Arc<Coordinator>, Address) {
    common::coordinator(policy, dir, &Telemetry::new(), cancel).await
}

async fn node(cl: &Cluster, node: &str, cancel: &CancellationToken) -> std::path::PathBuf {
    common::daemon(cl.dir.path(), node, &cl.address, Vec::new(), cancel)
        .await
        .1
}

/// One fake process: its connection, its id in a set, and every plan it read.
struct Proc {
    s: UnixStream,
    me: SetMember,
    seen: Vec<CoverPlan>,
}

impl Proc {
    /// Read plans until one at `generation` or above; report every group of it this process
    /// spans as formed (or kept) at that generation.
    async fn form(&mut self, generation: u64) -> CoverPlan {
        let plan = loop {
            let p = common::next_plan(&mut self.s).await;
            self.seen.push(p.clone());
            if p.generation >= generation {
                break p;
            }
        };
        for g in plan.groups.iter().filter(|g| g.spans.contains(&self.me)) {
            let sig = Signal {
                kind: Some(signal::Kind::GroupFormed(GroupFormed {
                    engine: "k3-a".into(),
                    generation: plan.generation,
                    spans: g.spans.clone(),
                    scope: g.scope,
                    seconds: 1.0,
                    member: Some(self.me.clone()),
                })),
            };
            frame::send(&mut self.s, &sig).await.expect("formed");
        }
        plan
    }
}

/// Eight workers on node-a and node-b, and a spare for each of node-a's ranks on node-c.
async fn engine(cl: &Cluster) -> (Vec<Proc>, Vec<Proc>) {
    let a = node(cl, "node-a", &cl.cancel).await;
    let b = node(cl, "node-b", &cl.cancel).await;
    let c = node(cl, "node-c", &cl.cancel).await;
    let mut workers = Vec::new();
    for r in 0..4 {
        workers.push(Proc {
            s: reached(&a, r, r, Role::Worker).await,
            me: rank(r),
            seen: Vec::new(),
        });
    }
    for r in 4..8 {
        workers.push(Proc {
            s: reached(&b, r, r - 4, Role::Worker).await,
            me: rank(r),
            seen: Vec::new(),
        });
    }
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(Proc {
            // on devices 4 to 7: a holder's device tells node-c's processes from node-a's
            s: reached(&c, r, r + 4, Role::Spare).await,
            me: SetMember {
                kind: Some(set_member::Kind::Spare(SpareRef {
                    for_rank: r,
                    node: "node-c".into(),
                    device: r + 4,
                })),
            },
            seen: Vec::new(),
        });
    }
    (workers, spares)
}

/// Register a process of `role` for `rank` on `device`, saying where its peers reach it, as every
/// engine process does: here, one machine's loopback.
async fn reached(path: &std::path::Path, rank: u32, device: u32, role: Role) -> UnixStream {
    common::register_as(
        path,
        everserve_controller::proto::everserve::signals::v1::Register {
            host: "127.0.0.1".into(),
            ..common::registration(rank, device, role)
        },
    )
    .await
}

async fn readiness(coord: &Coordinator) -> EngineReadiness {
    engine_readiness(coord).await.expect("k3-a")
}

/// The engine's readiness, once the coordinator heard of it.
async fn engine_readiness(coord: &Coordinator) -> Option<EngineReadiness> {
    coord
        .readiness()
        .await
        .engines
        .into_iter()
        .find(|e| e.engine == "k3-a")
}

async fn wait_ready(
    coord: &Coordinator,
    what: &str,
    ok: impl Fn(&EngineReadiness) -> bool,
) -> EngineReadiness {
    // up to ten seconds: a daemon reconnects with backoff
    for _ in 0..1000 {
        if let Some(e) = engine_readiness(coord).await
            && ok(&e)
        {
            return e;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("never {what}: {:?}", engine_readiness(coord).await);
}

fn covered(e: &EngineReadiness) -> (Truth, String, String) {
    let c = e
        .conditions
        .iter()
        .find(|c| c.kind == "Covered")
        .expect("Covered");
    (c.status, c.reason.clone(), c.message.clone())
}

#[tokio::test]
async fn a_plan_reaches_every_process_and_a_set_is_covered_once_every_member_reported_it() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    // planned once every position is held and the four spares stand
    let e = wait_ready(&cl.coord, "the plan with the spares", |e| {
        e.cover.as_ref().is_some_and(|c| {
            c.sets
                .iter()
                .any(|s| s.purpose == "REPLACE of ranks [0, 1, 2, 3]")
        })
    })
    .await;
    let cover = e.cover.clone().expect("cover");
    let generation = cover.generation;
    let sets: Vec<&str> = cover.sets.iter().map(|s| s.purpose.as_str()).collect();
    assert_eq!(
        sets,
        [
            "current",
            "REPLACE of ranks [0, 1, 2, 3]",
            "REPLACE of ranks [0]",
            "REPLACE of ranks [1]",
            "REPLACE of ranks [2]",
            "REPLACE of ranks [3]",
        ]
    );
    // node-b has no spare: its loss and its positions' are uncovered, and say why
    assert_eq!(cover.uncovered.len(), 5, "{:?}", cover.uncovered);
    assert!(
        cover.sets.iter().all(|s| !s.covered),
        "nothing reported yet"
    );
    assert_eq!(covered(&e).0, Truth::False);

    // every worker and spare forms its part; every set is covered, and the Covered condition
    // names only node-b's losses
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        assert_eq!(p.form(generation).await.generation, generation);
    }
    let e = wait_ready(&cl.coord, "every set covered", |e| {
        e.cover
            .as_ref()
            .is_some_and(|c| c.sets.iter().all(|s| s.covered))
    })
    .await;
    let (status, reason, message) = covered(&e);
    assert_eq!(
        (status, reason.as_str()),
        (Truth::False, "LossUncovered"),
        "{message}"
    );
    // each set's covering is an event, timed when its last group completed
    let events = cl.coord.engine_events("k3-a").await.expect("events");
    let covered_sets: Vec<&str> = events
        .iter()
        .filter(|e| e.reason == "SetCovered")
        .map(|e| e.message.as_str())
        .collect();
    assert_eq!(covered_sets.len(), 6, "{covered_sets:?}");
    assert!(
        covered_sets
            .iter()
            .any(|m| m.contains("(REPLACE of ranks [0, 1, 2, 3])")),
        "{covered_sets:?}"
    );
    assert!(
        message.contains("[4, 5, 6, 7]") && !message.contains("[0, 1, 2, 3]"),
        "{message}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_new_generation_starts_coverage_again_and_an_older_report_counts_toward_nothing() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let e = wait_ready(&cl.coord, "the plan with the spares", |e| {
        e.cover.as_ref().is_some_and(|c| c.sets.len() == 6)
    })
    .await;
    let first = e.cover.expect("cover").generation;
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        p.form(first).await;
    }
    wait_ready(&cl.coord, "covered", |e| {
        e.cover
            .as_ref()
            .is_some_and(|c| c.sets.iter().all(|s| s.covered))
    })
    .await;

    // rank 3's spare leaves: the next plan names neither node-a's loss nor rank 3's
    let gone = spares.pop().expect("spare 3");
    drop(gone);
    let e = wait_ready(&cl.coord, "the next plan", |e| {
        e.cover.as_ref().is_some_and(|c| c.generation > first)
    })
    .await;
    let cover = e.cover.expect("cover");
    let second = cover.generation;
    let purposes: Vec<&str> = cover.sets.iter().map(|s| s.purpose.as_str()).collect();
    assert_eq!(
        purposes,
        [
            "current",
            "REPLACE of ranks [0]",
            "REPLACE of ranks [1]",
            "REPLACE of ranks [2]"
        ]
    );
    // coverage starts again: nothing counts until the processes report at the new generation
    assert!(cover.sets.iter().all(|s| !s.covered), "{cover:?}");

    // reports at the old generation count toward nothing: all four of stage 0 report the current
    // set's domain group, which would complete it if they counted
    for p in &mut workers[..4] {
        let stale = Signal {
            kind: Some(signal::Kind::GroupFormed(GroupFormed {
                engine: "k3-a".into(),
                generation: first,
                // stage 0's domain group, which the current set needs
                spans: (0..4).map(rank).collect(),
                scope: Scope::Domain as i32,
                seconds: 0.0,
                member: Some(p.me.clone()),
            })),
        };
        frame::send(&mut p.s, &stale).await.expect("stale");
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        readiness(&cl.coord)
            .await
            .cover
            .expect("cover")
            .sets
            .iter()
            .all(|s| s.formed == 0),
        "an old generation's report counts toward nothing"
    );

    // every process reports what it keeps at the new generation, and releases what it held of
    // the old plan's groups
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        p.form(second).await;
    }
    release_dropped(workers.iter_mut().chain(spares.iter_mut()), first, second).await;
    let e = wait_ready(&cl.coord, "covered again", |e| {
        e.cover
            .as_ref()
            .is_some_and(|c| c.sets.iter().all(|s| s.covered))
    })
    .await;
    assert_eq!(e.cover.expect("cover").generation, second);
    cl.cancel.cancel();
}

/// Every process releases the groups it spans that plan `old` named and plan `new` does not,
/// at `new`'s generation: a group the dropped sets shared with a set the new plan keeps stays
/// (decision 0035, amended). Rank 3's spare is gone, released by its daemon.
async fn release_dropped(procs: impl Iterator<Item = &mut Proc>, old: u64, new: u64) {
    for p in procs {
        let at = |generation: u64| {
            p.seen
                .iter()
                .find(|plan| plan.generation == generation)
                .cloned()
                .expect("a plan the process saw")
        };
        let (before, after) = (at(old), at(new));
        let dropped: Vec<CoverGroup> = before
            .groups
            .iter()
            .filter(|g| g.spans.contains(&p.me))
            .filter(|g| {
                !after
                    .groups
                    .iter()
                    .any(|n| n.scope == g.scope && n.spans == g.spans)
            })
            .cloned()
            .collect();
        for g in dropped {
            let sig = Signal {
                kind: Some(signal::Kind::GroupReleased(GroupReleased {
                    engine: "k3-a".into(),
                    generation: new,
                    spans: g.spans,
                    scope: g.scope,
                    member: Some(p.me.clone()),
                })),
            };
            frame::send(&mut p.s, &sig).await.expect("released");
        }
    }
}

#[tokio::test]
async fn a_restarted_coordinator_adopts_the_plan_its_daemons_hold_and_delivers_none() {
    let dir = tempfile::tempdir().expect("tmp");
    let coord_cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &coord_cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel: CancellationToken::new(), // the daemons', apart from the coordinator's
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let e = wait_ready(&cl.coord, "the plan with the spares", |e| {
        e.cover.as_ref().is_some_and(|c| c.sets.len() == 6)
    })
    .await;
    let before = e.cover.expect("cover").generation;
    let mut delivered = None;
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        delivered = Some(p.form(before).await);
    }
    let delivered = delivered.expect("a plan");

    // the coordinator dies; a new one on the same address hears the daemons' hellos, each with
    // the plan its daemon forwarded, and adopts it: same generation, same ports, every set
    // covered by what the processes hold (a restart that re-plans at each relink, with new ports,
    // refuses a drain due then)
    coord_cancel.cancel();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (coord2, _) = coordinator(cl.dir.path(), &CancellationToken::new()).await;
    let e = wait_ready(&coord2, "the adopted plan, covered", |e| {
        e.cover.as_ref().is_some_and(|c| {
            c.generation == before && c.sets.len() == 6 && c.sets.iter().all(|s| s.covered)
        })
    })
    .await;
    assert_eq!(e.cover.expect("cover").generation, before);
    let view = coord2.engine("k3-a").await.expect("view");
    assert_eq!(
        view.cover.plan.as_ref().map(|p| &p.groups),
        Some(&delivered.groups),
        "the adopted plan is the delivered one, ports and all"
    );
    // every position's device and identity came back with the hellos
    assert_eq!(view.devices.len(), 8, "{:?}", view.devices);
    assert_eq!(view.devices.get(&5), Some(&1));
    assert_eq!(view.identities.len(), 8);
    // and no process is asked to form anything again
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        let next = tokio::time::timeout(Duration::from_secs(1), common::next_plan(&mut p.s)).await;
        assert!(
            next.is_err(),
            "a restart delivered {:?}",
            next.map(|p| p.generation)
        );
    }
    cl.cancel.cancel();
}

/// A file store whose status writes take `delay`, and fail while `refuse` is set.
struct Slow {
    inner: everserve_controller::spec::FileStore,
    delay: Duration,
    refuse: std::sync::atomic::AtomicBool,
}

impl Slow {
    fn new(path: std::path::PathBuf, delay: Duration) -> Self {
        Self {
            inner: everserve_controller::spec::FileStore::new(path),
            delay,
            refuse: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl everserve_controller::spec::SpecStore for Slow {
    fn take(
        &self,
        term: u64,
    ) -> BoxFuture<
        '_,
        Result<
            everserve_controller::proto::everserve::spec::v1::SpecSnapshot,
            everserve_controller::spec::SpecError,
        >,
    > {
        self.inner.take(term)
    }

    fn put(
        &self,
        key: String,
        spec: everserve_controller::proto::everserve::spec::v1::ResourceSpec,
        read: u64,
        term: u64,
    ) -> BoxFuture<
        '_,
        Result<
            everserve_controller::proto::everserve::spec::v1::ResourceSpec,
            everserve_controller::spec::SpecError,
        >,
    > {
        self.inner.put(key, spec, read, term)
    }

    fn put_status(
        &self,
        key: String,
        status: everserve_controller::proto::everserve::spec::v1::ResourceStatus,
        term: u64,
    ) -> BoxFuture<'_, Result<(), everserve_controller::spec::SpecError>> {
        Box::pin(async move {
            tokio::time::sleep(self.delay).await;
            if self.refuse.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(everserve_controller::spec::SpecError::Io {
                    path: key.into(),
                    source: std::io::Error::other("refused by the test"),
                });
            }
            self.inner.put_status(key, status, term).await
        })
    }

    fn describe(&self) -> String {
        format!("{}, slowly", self.inner.describe())
    }
}

/// Every process forms the plan last delivered, from readiness `e` on, until no later one
/// follows it; that plan.
async fn form_settled(
    coord: &Coordinator,
    workers: &mut [Proc],
    spares: &mut [Proc],
    e: EngineReadiness,
) -> CoverPlan {
    let mut at = e.cover.expect("cover").generation;
    loop {
        for p in workers.iter_mut().chain(spares.iter_mut()) {
            p.form(at).await;
        }
        let plan = coord
            .engine("k3-a")
            .await
            .and_then(|v| v.cover.plan)
            .expect("a delivered plan");
        if plan.generation == at {
            return plan;
        }
        at = plan.generation;
    }
}

/// The highest plan generation `store` keeps for k3-a.
fn kept_generation(store: &Slow) -> u64 {
    use everserve_controller::proto::everserve::spec::v1::resource_status;
    match store
        .inner
        .read()
        .expect("read")
        .statuses
        .get(&everserve_controller::spec::engine_key("k3-a"))
        .and_then(|s| s.kind.clone())
    {
        Some(resource_status::Kind::Engine(e)) => e.plan_generation,
        None => 0,
    }
}

/// Keep `generation` as k3-a's highest in `store`, as a coordinator does before it delivers a
/// plan of it.
async fn keep_generation(store: &Slow, generation: u64) {
    use everserve_controller::proto::everserve::spec::v1::{
        EngineStatus, ResourceStatus, resource_status,
    };
    use everserve_controller::spec::SpecStore as _;
    store
        .inner
        .put_status(
            everserve_controller::spec::engine_key("k3-a"),
            ResourceStatus {
                kind: Some(resource_status::Kind::Engine(EngineStatus {
                    engine: "k3-a".into(),
                    plan_generation: generation,
                    ..EngineStatus::default()
                })),
            },
            0,
        )
        .await
        .expect("the generation kept, its plan undelivered");
}

#[tokio::test]
async fn a_restarted_coordinator_numbers_its_plans_above_any_the_store_kept() {
    let dir = tempfile::tempdir().expect("tmp");
    // a status write as slow as a store across the network's, so plans asked for meanwhile meet
    // the one being written
    let store = Arc::new(Slow::new(
        dir.path().join("spec.pb"),
        Duration::from_millis(200),
    ));
    let coord_cancel = CancellationToken::new();
    let at = dir.path().to_owned();
    let with_store = |cancel: CancellationToken| {
        let (store, dir) = (Arc::clone(&store), at.clone());
        async move {
            let (coord, _) = Coordinator::with_store(cover_policy(), &Telemetry::new(), store, 0)
                .await
                .expect("coordinator");
            let address = common::serve(&coord, &dir, &cancel).await;
            (coord, address)
        }
    };
    let (coord, address) = with_store(coord_cancel.clone()).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel: CancellationToken::new(),
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let e = wait_ready(&cl.coord, "the plan with the spares", |e| {
        e.cover.as_ref().is_some_and(|c| c.sets.len() == 6)
    })
    .await;
    let delivered = form_settled(&cl.coord, &mut workers, &mut spares, e).await;
    wait_ready(&cl.coord, "covered", |e| {
        e.cover.as_ref().is_some_and(|c| {
            c.generation == delivered.generation && c.sets.iter().all(|s| s.covered)
        })
    })
    .await;
    // one plan of an engine is made at a time: the four spares registering while the store
    // writes one are planned together after it, at one more write, not one write and one plan
    // for each, every one of which each process would form
    let seen: Vec<(u64, usize)> = workers[0]
        .seen
        .iter()
        .map(|p| (p.generation, p.groups.len()))
        .collect();
    assert!(
        seen.len() < 1 + spares.len(),
        "a plan for each spare: {seen:?}"
    );
    // decision 0038: the store holds the delivered plan's generation; the plan is the daemons'
    assert_eq!(kept_generation(&store), delivered.generation);

    // the coordinator numbered its next plan, kept the number and died before any daemon had
    // the plan, or after only a daemon lost since had it
    coord_cancel.cancel();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let lost = delivered.generation + 1;
    keep_generation(&store, lost).await;
    let (coord2, _) = with_store(CancellationToken::new()).await;
    // its successor adopts the plan the daemons hold, delivering nothing...
    wait_ready(&coord2, "the daemons' plan, covered", |e| {
        e.cover.as_ref().is_some_and(|c| {
            c.generation == delivered.generation && c.sets.iter().all(|s| s.covered)
        })
    })
    .await;
    // ...and numbers its next plan above the lost one: a spare leaves, and the plan without it
    // is not the lost plan's number, whatever that plan held
    drop(spares.pop().expect("a spare"));
    let next = tokio::time::timeout(
        Duration::from_secs(10),
        common::next_plan(&mut workers[0].s),
    )
    .await
    .expect("a plan without the spare");
    assert!(
        next.generation > lost,
        "plan {} reuses a number the store kept ({lost})",
        next.generation
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_plan_the_store_did_not_keep_is_made_again_and_delivered_once_it_does() {
    // decision 0038: no daemon has a plan the store does not, so a plan whose write fails is
    // not delivered, and the engine plans again on its own, with nothing else changing
    let dir = tempfile::tempdir().expect("tmp");
    let store = Arc::new(Slow::new(dir.path().join("spec.pb"), Duration::ZERO));
    store
        .refuse
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let (coord, _) = Coordinator::with_store(cover_policy(), &Telemetry::new(), store.clone(), 0)
        .await
        .expect("coordinator");
    let address = common::serve(&coord, dir.path(), &CancellationToken::new()).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel: CancellationToken::new(),
    };
    let (mut workers, _spares) = engine(&cl).await;
    let none = tokio::time::timeout(
        Duration::from_millis(500),
        common::next_plan(&mut workers[0].s),
    )
    .await;
    assert!(none.is_err(), "a plan the store refused was delivered");
    store
        .refuse
        .store(false, std::sync::atomic::Ordering::SeqCst);
    let plan = tokio::time::timeout(Duration::from_secs(5), common::next_plan(&mut workers[0].s))
        .await
        .expect("planned again once the store keeps it");
    assert_eq!(plan.engine, "k3-a");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_coordinator_whose_store_does_not_answer_in_time_stops_acting() {
    // decision 0038: a write that outruns the deadline may still land, so nothing the coordinator
    // decides may rest on it: it ends its hold of the lease, and a successor reads the store
    let dir = tempfile::tempdir().expect("tmp");
    let store = Arc::new(Slow::new(
        dir.path().join("spec.pb"),
        Duration::from_millis(500),
    ));
    let (coord, _) = Coordinator::with_store(cover_policy(), &Telemetry::new(), store, 1)
        .await
        .expect("coordinator");
    coord.set_store_write_deadline(Duration::from_millis(50));
    let hold = CancellationToken::new();
    assert!(
        coord.act_under(2, hold.clone()).is_err(),
        "a coordinator acts under the term it took its store under, no other"
    );
    coord.act_under(1, hold.clone()).expect("act");
    let address = common::serve(&coord, dir.path(), &CancellationToken::new()).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel: CancellationToken::new(),
    };
    // the engine's registrations make a plan, which the coordinator writes first
    let _procs = engine(&cl).await;
    tokio::time::timeout(Duration::from_secs(5), hold.cancelled())
        .await
        .expect("the hold ended when the store did not answer");
    assert!(
        cl.coord.deposed(),
        "a coordinator that stopped acting is deposed"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_drain_refused_for_a_set_still_forming_goes_once_the_set_is_covered() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let generation = planned(&cl).await;
    // node-a drains while its set is planned and not formed: refused at the deadline, blocked
    let deadline = everserve_controller::clock::now_ms() + 50;
    cl.coord
        .drain("node-a", deadline, "test", "drill")
        .await
        .expect("drain");
    for _ in 0..300 {
        if matches!(
            cl.coord.drain_state("node-a").await,
            Some(DrainState::Blocked { .. })
        ) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        matches!(
            cl.coord.drain_state("node-a").await,
            Some(DrainState::Blocked { .. })
        ),
        "{:?}",
        cl.coord.drain_state("node-a").await
    );
    // the processes form the plan: the set is covered, and the drain goes at once, without
    // waiting out its backoff (a second from the refusal)
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        p.form(generation).await;
    }
    let covered = std::time::Instant::now();
    for (r, w) in (4u32..).zip(workers.iter_mut().skip(4)) {
        assert_eq!(w.ack_change(r).await, 1, "rank {r}");
    }
    assert!(
        covered.elapsed() < Duration::from_millis(500),
        "the change came {:?} after the set was covered",
        covered.elapsed()
    );
    for (r, sp) in (0u32..).zip(spares.iter_mut()) {
        assert_eq!(sp.ack_change(r).await, 1, "rank {r}'s spare");
    }
    wait_ready(&cl.coord, "node-a's ranks on node-c", |e| {
        (0..4).all(|r| e.members.iter().any(|m| m.rank == r && m.node == "node-c"))
    })
    .await;
    cl.cancel.cancel();
}

impl Proc {
    /// Acknowledge the next change this process receives, at its epoch.
    async fn ack_change(&mut self, rank: u32) -> u64 {
        let epoch = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let sig: Signal = common::recv(&mut self.s).await.expect("signal");
                if let Some(signal::Kind::Change(c)) = sig.kind {
                    return c.epoch;
                }
            }
        })
        .await
        .expect("a change within 5 s");
        let ack = Signal {
            kind: Some(signal::Kind::Ack(Ack {
                rank,
                epoch,
                error: String::new(),
                received_at_us: 0,
            })),
        };
        frame::send(&mut self.s, &ack).await.expect("ack");
        epoch
    }
}

async fn planned(cl: &Cluster) -> u64 {
    wait_ready(&cl.coord, "the plan with the spares", |e| {
        e.cover.as_ref().is_some_and(|c| c.sets.len() == 6)
    })
    .await
    .cover
    .expect("cover")
    .generation
}

#[tokio::test]
async fn a_loss_goes_through_its_covered_set_and_is_refused_while_the_set_is_forming() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let generation = planned(&cl).await;

    // rank 0's spare is outside its domain, so its loss moves node-a's whole group (0022):
    // planned, not formed, it is refused at once, naming the set
    let err = cl
        .coord
        .rank_lost("k3-a", 0, Cause::Fault)
        .await
        .unwrap_err();
    match &err {
        ChangeError::Uncovered { ranks, why } => {
            assert_eq!(ranks, &[0, 1, 2, 3]);
            assert!(
                why.contains("R0@node-c:4-R1@node-c:5-R2@node-c:6-R3@node-c:7")
                    && why.contains("not yet formed"),
                "{why}"
            );
        }
        other => panic!("{other}"),
    }

    // formed: the loss goes through its covered set, onto the spares the set names
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        p.form(generation).await;
    }
    wait_ready(&cl.coord, "covered", |e| {
        e.cover
            .as_ref()
            .is_some_and(|c| c.sets.iter().all(|s| s.covered))
    })
    .await;
    let change = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 0, Cause::Fault).await }
    });
    // node-b's workers survive; node-c's spares take node-a's positions
    for (r, w) in (4u32..).zip(workers.iter_mut().skip(4)) {
        assert_eq!(w.ack_change(r).await, 1);
    }
    for (r, s) in (0u32..).zip(spares.iter_mut()) {
        assert_eq!(s.ack_change(r).await, 1);
    }
    change.await.expect("join").expect("committed");
    let e = wait_ready(&cl.coord, "rank 0 on node-c", |e| {
        e.members.iter().any(|m| m.rank == 0 && m.node == "node-c")
    })
    .await;
    // the next plan names node-c's processes as the holders of node-a's ranks, by device: a
    // process holding a group that names a rank tells its holder changed
    let plan = loop {
        let p = common::next_plan(&mut workers[4].s).await;
        if p.epoch == 1 {
            break p;
        }
    };
    for r in 0..4u32 {
        assert_eq!(
            plan.placement.get(&r),
            Some(&holder("node-c", r + 4)),
            "rank {r}"
        );
    }
    assert_eq!(plan.placement.get(&4), Some(&holder("node-b", 0)));
    let record = e.last_change.expect("record");
    assert!(
        record.cover
            == "through covered set R0@node-c:4-R1@node-c:5-R2@node-c:6-R3@node-c:7-4-5-6-7",
        "{}",
        record.cover
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn after_a_replace_spares_that_stand_again_are_covered_and_take_the_next_loss() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let generation = planned(&cl).await;
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        p.form(generation).await;
    }
    wait_ready(&cl.coord, "covered", |e| {
        e.cover
            .as_ref()
            .is_some_and(|c| c.sets.iter().all(|s| s.covered))
    })
    .await;
    // the first loss: node-c's spares take node-a's positions
    let change = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 0, Cause::Fault).await }
    });
    for (r, w) in (4u32..).zip(workers.iter_mut().skip(4)) {
        w.ack_change(r).await;
    }
    for (r, s) in (0u32..).zip(spares.iter_mut()) {
        s.ack_change(r).await;
    }
    change.await.expect("join").expect("committed");
    // node-a's processes leave, and node-a comes back as spares for the ranks it held: the
    // planner covers a loss of node-c's group with them (the same node, lost and back, is taken
    // twice)
    let mut workers = workers.split_off(4);
    let a = cl.dir.path().join("node-a.sock");
    let mut back = Vec::new();
    for r in 0..4 {
        back.push(Proc {
            s: reached(&a, r, r, Role::Spare).await,
            me: SetMember {
                kind: Some(set_member::Kind::Spare(SpareRef {
                    for_rank: r,
                    node: "node-a".into(),
                    device: r,
                })),
            },
            seen: Vec::new(),
        });
    }
    // planned once all four stand: the set that answers a loss of node-c's group with them
    let e = wait_ready(&cl.coord, "a set of node-a's spares planned", |e| {
        e.cover.as_ref().is_some_and(|c| {
            c.epoch == 1
                && c.sets
                    .iter()
                    .any(|s| s.set == "R0@node-a:0-R1@node-a:1-R2@node-a:2-R3@node-a:3-4-5-6-7")
        })
    })
    .await;
    let generation = e.cover.expect("cover").generation;
    // node-c's processes hold ranks 0 to 3 now: the plan names them by rank
    for (r, s) in (0u32..).zip(spares.iter_mut()) {
        s.me = rank(r);
    }
    for p in workers
        .iter_mut()
        .chain(spares.iter_mut())
        .chain(back.iter_mut())
    {
        p.form(generation).await;
    }
    wait_ready(&cl.coord, "covered again", |e| {
        e.cover
            .as_ref()
            .is_some_and(|c| c.generation == generation && c.sets.iter().all(|s| s.covered))
    })
    .await;
    // the second loss goes through the set node-a's spares formed
    let change = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 0, Cause::Fault).await }
    });
    for (r, w) in (4u32..).zip(workers.iter_mut()) {
        assert_eq!(w.ack_change(r).await, 2);
    }
    for (r, b) in (0u32..).zip(back.iter_mut()) {
        assert_eq!(b.ack_change(r).await, 2);
    }
    change.await.expect("join").expect("committed");
    let e = wait_ready(&cl.coord, "rank 0 back on node-a", |e| {
        e.members.iter().any(|m| m.rank == 0 && m.node == "node-a")
    })
    .await;
    let record = e.last_change.expect("record");
    assert_eq!(
        record.cover,
        "through covered set R0@node-a:0-R1@node-a:1-R2@node-a:2-R3@node-a:3-4-5-6-7"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_loss_on_the_heads_node_restarts_the_engine_and_is_named_beside_cover_not_counted() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let generation = planned(&cl).await;
    let condition = |e: &EngineReadiness| {
        e.conditions
            .iter()
            .find(|c| c.kind == "Covered")
            .cloned()
            .expect("Covered")
    };
    // no head yet: node-b's losses are ones no spare answers, counted as any other
    let e = wait_ready(&cl.coord, "node-b's loss counted", |e| {
        condition(e).message.contains("[4, 5, 6, 7]")
    })
    .await;
    assert!(!condition(&e).message.contains("the head's node"), "{e:?}");
    // the head registers on node-b: a loss of that node or of any rank on it restarts the engine
    // (decision 0038), so they are named, not counted; once node-a's sets form on node-c's
    // spares, every loss left is answered, and the engine is covered (the state a Dynamo engine
    // sits in, its head's rank always listed before)
    let _head = common::register_as(
        &cl.dir.path().join("node-b.sock"),
        everserve_controller::proto::everserve::signals::v1::Register {
            role: Role::Head as i32,
            launcher_pid: std::process::id(),
            ..common::registration(0, 0, Role::Head)
        },
    )
    .await;
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        p.form(generation).await;
    }
    let e = wait_ready(&cl.coord, "covered, the head's node named", |e| {
        condition(e).status == Truth::True
    })
    .await;
    let c = condition(&e);
    assert_eq!(c.reason, "EveryLossCovered", "{c:?}");
    assert!(
        c.message.contains("the head's node holds [4, 5, 6, 7]"),
        "{c:?}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_policy_that_forms_on_the_critical_path_proceeds_and_says_so() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let mut policy = cover_policy();
    policy.cover.as_mut().expect("cover").form_on_critical_path = true;
    let (coord, address) = coordinator_with(dir.path(), &cancel, policy).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    planned(&cl).await;
    // nothing formed: the change proceeds through the pool, and its record says it formed
    let change = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 0, Cause::Fault).await }
    });
    for (r, w) in (4u32..).zip(workers.iter_mut().skip(4)) {
        w.ack_change(r).await;
    }
    for (r, s) in (0u32..).zip(spares.iter_mut()) {
        s.ack_change(r).await;
    }
    change.await.expect("join").expect("committed");
    let e = wait_ready(&cl.coord, "the record", |e| e.last_change.is_some()).await;
    assert_eq!(e.last_change.expect("record").cover, "formed in the change");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_failed_group_is_retried_by_delivering_the_plan_again_after_its_backoff() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let telemetry = Telemetry::new();
    let (coord, address) =
        common::coordinator(cover_policy(), dir.path(), &telemetry, &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, _spares) = engine(&cl).await;
    let first = planned(&cl).await;
    let plan = loop {
        let p = common::next_plan(&mut workers[0].s).await;
        if p.generation >= first {
            break p;
        }
    };
    // a group of rank 0 the plan forms, not one the engine serves on
    let group = not_serving(&plan)
        .into_iter()
        .find(|g| g.spans.contains(&rank(0)))
        .expect("a group of rank 0");
    let failed = Signal {
        kind: Some(signal::Kind::GroupFailed(GroupFailed {
            engine: "k3-a".into(),
            generation: plan.generation,
            spans: group.spans.clone(),
            scope: group.scope,
            error: "store timeout".into(),
            member: Some(rank(0)),
        })),
    };
    frame::send(&mut workers[0].s, &failed)
        .await
        .expect("failed");
    let e = wait_ready(&cl.coord, "the failure", |e| {
        e.cover.as_ref().is_some_and(|c| !c.failed.is_empty())
    })
    .await;
    let f = &e.cover.expect("cover").failed[0];
    assert_eq!(
        (f.member.as_str(), f.error.as_str()),
        ("0", "store timeout")
    );
    // at once a plan without the failed group, so every member lets go of what it holds of it
    let without = common::next_plan(&mut workers[0].s).await;
    let left_out = std::time::Instant::now();
    assert_eq!(without.generation, plan.generation + 1);
    assert!(!without.groups.contains(&group), "{without:?}");
    // after a second's backoff the same groups come again, at a new generation, formed afresh
    let again = common::next_plan(&mut workers[0].s).await;
    assert!(
        left_out.elapsed() >= Duration::from_millis(800),
        "the retry came before its backoff: {:?}",
        left_out.elapsed()
    );
    assert_eq!(again.generation, plan.generation + 2);
    assert_eq!(again.groups, plan.groups);
    let events = cl.coord.engine_events("k3-a").await.expect("events");
    assert!(
        events.iter().any(|e| e.reason == "GroupFailed"),
        "{events:?}"
    );
    // the retry is counted once its plan went out
    let retried = "everserve_cover_retries_total{engine=\"k3-a\"} 1";
    assert!(
        telemetry.render().contains(retried),
        "{}",
        telemetry.render()
    );
    cl.cancel.cancel();
}

/// The groups of `plan` its current set does not need: the ones the plan forms.
fn not_serving(plan: &CoverPlan) -> Vec<CoverGroup> {
    let serving: Vec<usize> = plan
        .sets
        .iter()
        .filter(|s| s.purpose == "current")
        .flat_map(|s| s.groups.iter().map(|&i| i as usize))
        .collect();
    plan.groups
        .iter()
        .enumerate()
        .filter(|(i, _)| !serving.contains(i))
        .map(|(_, g)| g.clone())
        .collect()
}

/// The two-node policy with cover, its groups to be formed within `timeout_ms`.
fn timed_policy(timeout_ms: u32) -> everserve_controller::proto::everserve::policy::v1::Policy {
    let mut p = cover_policy();
    p.cover.as_mut().expect("cover").form_timeout_ms = timeout_ms;
    p
}

#[tokio::test]
async fn a_group_not_formed_in_time_is_failed_left_out_and_formed_afresh_at_the_retry() {
    // decision 0035, amended with (C): a member whose formation reported success without its
    // peers holds a dead communicator until something tells it the group never formed
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator_with(dir.path(), &cancel, timed_policy(300)).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, _spares) = engine(&cl).await;
    let first = planned(&cl).await;
    // rank 0 forms its groups; its peers say nothing
    let plan = workers[0].form(first).await;
    let groups_of_0: Vec<_> = not_serving(&plan)
        .into_iter()
        .filter(|g| g.spans.contains(&rank(0)))
        .collect();
    assert!(!groups_of_0.is_empty());
    // past the timeout and its transit, a plan without them
    let without = tokio::time::timeout(
        Duration::from_secs(10),
        common::next_plan(&mut workers[0].s),
    )
    .await
    .expect("a plan without the groups never formed");
    assert_eq!(without.generation, plan.generation + 1);
    assert!(
        groups_of_0.iter().all(|g| !without.groups.contains(g)),
        "{without:?}"
    );
    let events = cl.coord.engine_events("k3-a").await.expect("events");
    assert!(
        events
            .iter()
            .any(|e| e.reason == "GroupFailed" && e.message.contains("not formed by every member")),
        "{events:?}"
    );
    // one failure for each group the plan forms, none for the groups the engine serves on, which
    // are not the plan's to form
    let failures = events.iter().filter(|e| e.reason == "GroupFailed").count();
    assert_eq!(failures, not_serving(&plan).len(), "{events:?}");
    // and at the retry, after its backoff, they come again at a new generation
    let left_out = std::time::Instant::now();
    let again = tokio::time::timeout(
        Duration::from_secs(10),
        common::next_plan(&mut workers[0].s),
    )
    .await
    .expect("the retry");
    assert!(
        left_out.elapsed() >= Duration::from_millis(800),
        "the retry came before its backoff: {:?}",
        left_out.elapsed()
    );
    assert_eq!(again.generation, plan.generation + 2);
    assert_eq!(again.groups, plan.groups);
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_change_across_a_plans_deadline_fails_none_of_its_groups() {
    // during a change members only record plans; a deadline counted through it would fail groups
    // whose members are simply waiting for the commit
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let mut policy = timed_policy(300);
    policy.cover.as_mut().expect("cover").form_on_critical_path = true;
    let (coord, address) = coordinator_with(dir.path(), &cancel, policy).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    planned(&cl).await;
    let change = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 0, Cause::Fault).await }
    });
    // the plan's deadline passes while the change waits for its acknowledgements
    tokio::time::sleep(
        Duration::from_millis(300) + Duration::from_secs(2) + Duration::from_millis(500),
    )
    .await;
    for (r, w) in (4u32..).zip(workers.iter_mut().skip(4)) {
        w.ack_change(r).await;
    }
    for (r, s) in (0u32..).zip(spares.iter_mut()) {
        s.ack_change(r).await;
    }
    change.await.expect("join").expect("committed");
    let events = cl.coord.engine_events("k3-a").await.expect("events");
    assert!(
        !events.iter().any(|e| e.reason == "GroupFailed"),
        "a group failed across the change: {events:?}"
    );
    cl.cancel.cancel();
}

/// The leak `w`'s process reports, its `total`th.
async fn leak(w: &mut Proc, rank: u32, total: u32) {
    let sig = Signal {
        kind: Some(signal::Kind::ReleaseLeaked(
            everserve_controller::proto::everserve::signals::v1::ReleaseLeaked {
                engine: "k3-a".into(),
                rank,
                release: format!("abort of group {total}"),
                run_ms: 12_000,
                leaked_total: total,
            },
        )),
    };
    frame::send(&mut w.s, &sig).await.expect("leak");
}

/// Read `w`'s signals for up to `within` until a notice; none at the deadline.
async fn notice_within(
    w: &mut Proc,
    within: Duration,
) -> Option<everserve_controller::proto::everserve::signals::v1::Notice> {
    tokio::time::timeout(within, async {
        loop {
            let sig = common::recv(&mut w.s).await.expect("signal");
            if let Some(signal::Kind::Notice(n)) = sig.kind {
                return n;
            }
        }
    })
    .await
    .ok()
}

fn degraded(e: &EngineReadiness) -> (String, String) {
    let c = e
        .conditions
        .iter()
        .find(|c| c.kind == "Degraded")
        .expect("Degraded");
    (c.reason.clone(), c.message.clone())
}

/// Every process forms the plan at `generation`, and the engine is covered.
async fn form_all(cl: &Cluster, workers: &mut [Proc], spares: &mut [Proc], generation: u64) {
    for p in workers.iter_mut().chain(spares.iter_mut()) {
        p.form(generation).await;
    }
    wait_ready(&cl.coord, "every set covered", |e| {
        e.cover
            .as_ref()
            .is_some_and(|c| c.sets.iter().all(|s| s.covered))
    })
    .await;
}

#[tokio::test]
async fn a_process_that_leaves_a_release_behind_is_counted_and_retired_once_by_notice() {
    // decision 0025, amended: the release holds its communicator's memory until the process ends;
    // serving is unaffected, so a spare takes the rank by notice, with no gap
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let telemetry = Telemetry::new();
    let (coord, address) =
        common::coordinator(cover_policy(), dir.path(), &telemetry, &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let generation = planned(&cl).await;
    form_all(&cl, &mut workers, &mut spares, generation).await;
    leak(&mut workers[1], 1, 1).await;
    let notice = notice_within(&mut workers[0], Duration::from_secs(5))
        .await
        .expect("a notice retiring rank 1");
    assert_eq!(notice.rank, 1);
    assert_eq!(notice.reason, Reason::Retired as i32);
    let e = wait_ready(&cl.coord, "degraded", |e| degraded(e).0 == "ReleaseLeaked").await;
    assert!(
        degraded(&e).1.contains("retired by notice"),
        "{:?}",
        degraded(&e)
    );
    assert!(
        telemetry
            .render()
            .contains("everserve_releases_leaked_total{engine=\"k3-a\"} 1"),
        "{}",
        telemetry.render()
    );
    // a second leak before it leaves is counted, and sends no second notice
    leak(&mut workers[1], 1, 2).await;
    for _ in 0..200 {
        if telemetry
            .render()
            .contains("everserve_releases_leaked_total{engine=\"k3-a\"} 2")
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        notice_within(&mut workers[0], Duration::from_millis(500))
            .await
            .is_none()
    );
    let events = cl.coord.engine_events("k3-a").await.expect("events");
    // the leak reports' own events, not the Degraded condition's, which shares their reason
    let leaks = events
        .iter()
        .filter(|e| e.reason == "ReleaseLeaked" && e.message.contains("left release"))
        .count();
    let retired = events
        .iter()
        .filter(|e| e.reason == "ProcessRetired")
        .count();
    assert_eq!((leaks, retired), (2, 1), "{events:?}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_leak_with_no_spare_ready_waits_degraded_and_retires_once_the_rank_is_covered() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let generation = planned(&cl).await;
    // nothing formed yet: no spare can take rank 1 with no gap
    leak(&mut workers[1], 1, 1).await;
    let e = wait_ready(&cl.coord, "degraded", |e| degraded(e).0 == "ReleaseLeaked").await;
    assert!(
        degraded(&e).1.contains("waiting for a spare"),
        "{:?}",
        degraded(&e)
    );
    // no notice yet, read from the events so no process's plans are taken from it
    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = cl.coord.engine_events("k3-a").await.expect("events");
    assert!(
        !events.iter().any(|e| e.reason == "ProcessRetired"),
        "{events:?}"
    );
    // the plan formed: the rank is covered, and its process goes
    form_all(&cl, &mut workers, &mut spares, generation).await;
    let notice = notice_within(&mut workers[0], Duration::from_secs(5))
        .await
        .expect("a notice once covered");
    assert_eq!((notice.rank, notice.reason), (1, Reason::Retired as i32));
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_leak_reported_for_a_rank_the_node_does_not_hold_changes_nothing() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let telemetry = Telemetry::new();
    let (coord, address) =
        common::coordinator(cover_policy(), dir.path(), &telemetry, &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, _spares) = engine(&cl).await;
    planned(&cl).await;
    // node-a's process names rank 5, which node-b holds
    leak(&mut workers[1], 5, 1).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !telemetry
            .render()
            .contains("everserve_releases_leaked_total{")
    );
    assert_ne!(degraded(&readiness(&cl.coord).await).0, "ReleaseLeaked");
    cl.cancel.cancel();
}

#[tokio::test]
async fn after_a_take_the_hello_names_one_holder_per_rank_the_spare_that_took_it() {
    take_then_relink(1).await;
}

#[tokio::test]
async fn a_commit_sent_again_changes_nothing_the_first_applied() {
    // a restarted coordinator rolls a logged commit forward to every daemon below its epoch
    // (decision 0038), and one may have applied it already: applied twice, the spare that took
    // rank 1 would be dropped as the rank's old holder
    take_then_relink(2).await;
}

/// A daemon whose spare takes rank 1, sent the commit `times` times, relinks: its hello names
/// the spare as rank 1's one holder.
async fn take_then_relink(times: usize) {
    use everserve_controller::proto::everserve::coord::v1::Committed;
    use everserve_controller::proto::everserve::signals::v1::{MembershipCommitted, Outcome};
    let dir = tempfile::tempdir().expect("tmp");
    let address = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    let cancel = CancellationToken::new();
    let (d, _rx) = Daemon::new("node-a", &Telemetry::new());
    let sock = dir.path().join("node-a.sock");
    tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel.clone()));
    tokio::spawn(Arc::clone(&d).link(address, None, "coordinator".into(), cancel.clone()));
    let (mut link, _) = accept(&listener).await;
    for _ in 0..200 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let _w0 = reached(&sock, 0, 0, Role::Worker).await;
    // rank 1's old worker keeps its connection open past the change
    let _w1 = reached(&sock, 1, 1, Role::Worker).await;
    let _spare = reached(&sock, 1, 5, Role::Spare).await;
    for _ in 0..200 {
        if d.registered("k3-a").await.len() == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let committed = CoordToDaemon {
        term: 0,
        kind: Some(coord_to_daemon::Kind::Committed(Committed {
            engine: "k3-a".into(),
            membership: Some(MembershipCommitted {
                epoch: 1,
                active_ranks: vec![0, 1],
                world_size: 2,
                ..MembershipCommitted::default()
            }),
            outcome: Outcome::Replace as i32,
            lost: vec![1],
            take: vec![],
        })),
    };
    frame::send(&mut link, &committed).await.expect("committed");
    for _ in 0..200 {
        if d.epoch("k3-a").await == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(d.epoch("k3-a").await, 1, "the commit was applied");
    for _ in 1..times {
        frame::send(&mut link, &committed)
            .await
            .expect("committed again");
        // the daemon takes each message on a task of its own, nothing to wait on when it does
        // nothing; applied, the copy would drop the holder well within this
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    drop(link);
    let (_link, hello) = accept(&listener).await;
    let engine = hello
        .engines
        .iter()
        .find(|e| e.engine == "k3-a")
        .expect("k3-a");
    let rank1: Vec<u32> = engine
        .serving
        .iter()
        .filter(|ev| ev.rank == 1)
        .map(|ev| ev.device)
        .collect();
    assert_eq!(
        rank1,
        [5],
        "one holder of rank 1, the spare: {:?}",
        engine.serving
    );
    assert!(engine.standing.is_empty(), "the spare stands no longer");
    cancel.cancel();
}

/// The next group id `s`'s process is handed, past every other signal.
/// `p` says it holds `plan` and waits for `group`'s id.
async fn ready(p: &mut Proc, plan: &CoverPlan, group: &CoverGroup) {
    let sig = Signal {
        kind: Some(signal::Kind::GroupReady(
            everserve_controller::proto::everserve::signals::v1::GroupReady {
                engine: "k3-a".into(),
                generation: plan.generation,
                spans: group.spans.clone(),
                scope: group.scope,
                member: Some(p.me.clone()),
            },
        )),
    };
    frame::send(&mut p.s, &sig).await.expect("ready");
}

/// A group id `s` is handed within `within`; none at the deadline.
async fn group_id_within(s: &mut UnixStream, within: Duration) -> Option<GroupId> {
    tokio::time::timeout(within, async {
        loop {
            let sig: Signal = frame::recv(s).await.expect("frame");
            if let Some(signal::Kind::GroupId(id)) = sig.kind {
                return id;
            }
        }
    })
    .await
    .ok()
}

async fn next_group_id(s: &mut UnixStream) -> GroupId {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let sig: Signal = frame::recv(s).await.expect("frame");
            if let Some(signal::Kind::GroupId(id)) = sig.kind {
                return id;
            }
        }
    })
    .await
    .expect("a group id")
}

#[tokio::test]
async fn a_groups_id_reaches_its_other_members_from_its_server_alone() {
    // decision 0035, amended: a group meets on the control plane's links, not on a store any
    // pod can reach. Its server member makes its id; the coordinator takes it from the server's
    // node alone and relays it to the other members' daemons, which hand it to them
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = coordinator(dir.path(), &cancel).await;
    let cl = Cluster {
        coord,
        dir,
        address,
        cancel,
    };
    let (mut workers, mut spares) = engine(&cl).await;
    let e = wait_ready(&cl.coord, "the plan with the spares", |e| {
        e.cover.as_ref().is_some_and(|c| c.sets.len() == 6)
    })
    .await;
    let plan = spares[0].form(e.cover.expect("cover").generation).await;
    // node-a's loss: the column of rank 0 is node-c's spare for it and rank 4 on node-b, the
    // spare its server
    let column = plan
        .groups
        .iter()
        .find(|g| g.spans == vec![spares[0].me.clone(), rank(4)])
        .expect("the column of rank 0 in node-a's loss")
        .clone();
    assert_eq!(column.server.as_ref(), Some(&spares[0].me));
    let id_of = |bytes: u8| Signal {
        kind: Some(signal::Kind::GroupId(GroupId {
            engine: "k3-a".into(),
            generation: plan.generation,
            scope: column.scope,
            spans: column.spans.clone(),
            id: vec![bytes; 128],
        })),
    };
    // rank 4's process, on node-b, is no server of it: its id is refused
    frame::send(&mut workers[4].s, &id_of(1))
        .await
        .expect("forged");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert!(
        view.cover.ids.is_empty(),
        "an id from a node not the server's was kept"
    );
    // the spare, its server, makes it: held, since no member said it is ready to join
    frame::send(&mut spares[0].s, &id_of(7))
        .await
        .expect("made");
    assert!(
        group_id_within(&mut workers[4].s, Duration::from_millis(300))
            .await
            .is_none(),
        "an id relayed before its members were ready"
    );
    // rank 4 ready, the server not yet: still held
    ready(&mut workers[4], &plan, &column).await;
    assert!(
        group_id_within(&mut workers[4].s, Duration::from_millis(300))
            .await
            .is_none(),
        "an id relayed before its server was ready"
    );
    // every member ready: each is handed it, the server too, and that is its go
    ready(&mut spares[0], &plan, &column).await;
    let handed = next_group_id(&mut workers[4].s).await;
    assert_eq!(
        (handed.id.as_slice(), handed.spans.clone()),
        (&[7u8; 128][..], column.spans.clone())
    );
    let back = next_group_id(&mut spares[0].s).await;
    assert_eq!(back.id.as_slice(), &[7u8; 128][..]);
    assert_eq!(
        cl.coord.engine("k3-a").await.expect("view").cover.ids.len(),
        1
    );
    cl.cancel.cancel();
}
