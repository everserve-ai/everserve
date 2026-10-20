// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Which process serves a rank is decided once (decision 0041): a process registers undecided
//! for its rank, before it loads anything, and is told to serve it or to stand as its spare.
//! Exactly one is told to serve; the rank is held by that answer while its process loads, and by
//! its worker once it registers; the answer is released if its process leaves first or its node
//! is lost, and a coordinator's restart never gives it twice.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use everserve_controller::coord::Coordinator;
use everserve_controller::policy;
use everserve_controller::proto::everserve::coord::v1::{
    ConnectionRef, CoordToDaemon, DaemonHello, DaemonToCoord, EngineState, RoleAsked,
    coord_to_daemon, daemon_to_coord,
};
use everserve_controller::proto::everserve::signals::v1::{Role, RoleAssigned, Signal, signal};
use everserve_controller::spec::{FileStore, SpecStore};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::Address;
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

const ENGINE: &str = "k3-a";

/// `f`, which must finish within five seconds.
async fn within<T>(what: &str, f: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), f)
        .await
        .unwrap_or_else(|_| panic!("{what}: not within 5 s"))
}

/// A coordinator under the two-node policy keeping its state in `store`, serving at `dir`'s
/// socket until `cancel` fires.
async fn coordinator(
    store: &Arc<FileStore>,
    dir: &Path,
    cancel: &CancellationToken,
) -> (Arc<Coordinator>, Address) {
    coordinator_edited(store, dir, cancel, |_| {}).await
}

/// A coordinator under the two-node policy, its engine as `edit` makes it.
async fn coordinator_edited(
    store: &Arc<FileStore>,
    dir: &Path,
    cancel: &CancellationToken,
    edit: impl FnOnce(&mut everserve_controller::proto::everserve::policy::v1::Engine),
) -> (Arc<Coordinator>, Address) {
    let mut policy = policy::load(&common::example("policy-two-node.json")).expect("policy");
    edit(&mut policy.engines[0]);
    let store: Arc<dyn SpecStore> = store.clone();
    let (coord, _rx) = Coordinator::with_store(policy, &Telemetry::new(), store, 0)
        .await
        .expect("coordinator");
    let address = common::serve(&coord, dir, cancel).await;
    (coord, address)
}

/// A process registering undecided for `rank` with the daemon at `sock`.
async fn undecided(sock: &Path, rank: u32) -> UnixStream {
    common::register_as(sock, common::registration(rank, rank, Role::Undecided)).await
}

/// What `s`'s process is told.
async fn told(s: &mut UnixStream) -> RoleAssigned {
    within("told what it is", async {
        loop {
            let sig: Signal = common::recv(s).await.expect("frame");
            if let Some(signal::Kind::RoleAssigned(r)) = sig.kind {
                return r;
            }
        }
    })
    .await
}

/// Register, on `s`, as the `role` it was told for `rank`: loaded, and serving or standing.
async fn registers_as(s: &mut UnixStream, rank: u32, role: Role) {
    frame::send(
        s,
        &Signal {
            kind: Some(signal::Kind::Register(common::registration(
                rank, rank, role,
            ))),
        },
    )
    .await
    .expect("register");
}

/// Whether `s` was closed by the daemon.
async fn closed(s: &mut UnixStream) -> bool {
    let next: Result<Result<Signal, _>, _> =
        tokio::time::timeout(Duration::from_secs(2), frame::recv(s)).await;
    matches!(next, Ok(Err(_)))
}

/// Wait until `coord`'s view of k3-a satisfies `ok`.
async fn view_until(
    coord: &Coordinator,
    what: &str,
    ok: impl Fn(&everserve_controller::coord::EngineView) -> bool,
) {
    within(what, async {
        while !coord.engine(ENGINE).await.is_some_and(|v| ok(&v)) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
}

/// Wait until `coord`'s assignments for k3-a satisfy `ok`.
async fn assigned_until(
    coord: &Coordinator,
    what: &str,
    ok: impl Fn(&[everserve_controller::proto::everserve::spec::v1::Assignment]) -> bool,
) {
    within(what, async {
        while !ok(&coord.assignments(ENGINE)) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
}

struct Rig {
    dir: tempfile::TempDir,
    store: Arc<FileStore>,
    coord: Arc<Coordinator>,
    coord_cancel: CancellationToken,
    address: Address,
    /// The daemons'.
    cancel: CancellationToken,
}

async fn rig() -> Rig {
    rig_edited(|_| {}).await
}

/// A rig whose engine is as `edit` makes it.
async fn rig_edited(
    edit: impl FnOnce(&mut everserve_controller::proto::everserve::policy::v1::Engine),
) -> Rig {
    let dir = tempfile::tempdir().expect("tmp");
    let store = Arc::new(FileStore::new(dir.path().join("spec.pb")));
    let coord_cancel = CancellationToken::new();
    let (coord, address) = coordinator_edited(&store, dir.path(), &coord_cancel, edit).await;
    Rig {
        dir,
        store,
        coord,
        coord_cancel,
        address,
        cancel: CancellationToken::new(),
    }
}

impl Rig {
    async fn daemon(&self, node: &str) -> std::path::PathBuf {
        common::daemon(
            self.dir.path(),
            node,
            &self.address,
            Vec::new(),
            &self.cancel,
        )
        .await
        .1
    }
}

#[tokio::test]
async fn of_two_undecided_processes_for_a_rank_one_serves_and_the_other_stands() {
    let r = rig().await;
    let (a, c) = (r.daemon("node-a").await, r.daemon("node-c").await);
    let mut first = undecided(&a, 0).await;
    let mut second = undecided(&c, 0).await;
    let roles = [told(&mut first).await.role, told(&mut second).await.role];
    let serving = roles.iter().filter(|r| **r == Role::Worker as i32).count();
    assert_eq!(serving, 1, "exactly one serves rank 0: {roles:?}");
    let (mut worker, mut spare) = if roles[0] == Role::Worker as i32 {
        (first, second)
    } else {
        (second, first)
    };
    // the one told to serve loads for minutes; a third asker meanwhile stands
    let mut third = undecided(&a, 0).await;
    assert_eq!(told(&mut third).await.role, Role::Spare as i32);
    registers_as(&mut worker, 0, Role::Worker).await;
    registers_as(&mut spare, 0, Role::Spare).await;
    view_until(&r.coord, "the worker holds rank 0", |v| {
        v.members.contains_key(&0)
    })
    .await;
    assigned_until(&r.coord, "its assignment spent", <[_]>::is_empty).await;
    r.cancel.cancel();
}

#[tokio::test]
async fn a_rank_that_has_had_a_holder_is_never_served_by_a_registration() {
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let _w = common::register(&a, 0, 0, Role::Worker).await;
    common::wait_members(&r.coord, 1).await;
    let mut asker = undecided(&a, 0).await;
    assert_eq!(told(&mut asker).await.role, Role::Spare as i32);
    r.cancel.cancel();
}

#[tokio::test]
async fn a_process_that_leaves_before_it_loads_releases_its_rank() {
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let mut first = undecided(&a, 0).await;
    assert_eq!(told(&mut first).await.role, Role::Worker as i32);
    drop(first);
    assigned_until(&r.coord, "the assignment released", <[_]>::is_empty).await;
    let mut next = undecided(&a, 0).await;
    assert_eq!(told(&mut next).await.role, Role::Worker as i32);
    r.cancel.cancel();
}

#[tokio::test]
async fn a_process_on_a_lost_node_releases_its_rank() {
    let r = rig().await;
    let a = r.daemon("node-a").await;
    // a daemon on node "far" asks for rank 0's process, then its node is lost
    let (far, role) = ask_as(&r.address, "far").await;
    assert_eq!(role, Role::Worker as i32);
    drop(far); // no goodbye: the node is lost
    assigned_until(&r.coord, "the assignment released", <[_]>::is_empty).await;
    let mut next = undecided(&a, 0).await;
    assert_eq!(told(&mut next).await.role, Role::Worker as i32);
    r.cancel.cancel();
}

#[tokio::test]
async fn a_coordinators_restart_never_tells_two_processes_to_serve_one_rank() {
    let r = rig().await;
    let (a, c) = (r.daemon("node-a").await, r.daemon("node-c").await);
    let mut loading = undecided(&a, 0).await;
    assert_eq!(told(&mut loading).await.role, Role::Worker as i32);
    // the coordinator restarts while rank 0's process loads
    r.coord_cancel.cancel();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (second, _) = coordinator(&r.store, r.dir.path(), &CancellationToken::new()).await;
    assert!(
        second.assignments(ENGINE).iter().any(|a| a.rank == 0),
        "the successor read the assignment from the store"
    );
    let mut asker = undecided(&c, 0).await;
    assert_eq!(
        told(&mut asker).await.role,
        Role::Spare as i32,
        "the successor knows rank 0 is assigned"
    );
    registers_as(&mut loading, 0, Role::Worker).await;
    view_until(&second, "rank 0's worker", |v| v.members.contains_key(&0)).await;
    r.cancel.cancel();
}

#[tokio::test]
async fn an_assignment_whose_process_left_while_the_coordinator_was_down_is_released() {
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let mut loading = undecided(&a, 0).await;
    assert_eq!(told(&mut loading).await.role, Role::Worker as i32);
    r.coord_cancel.cancel();
    drop(loading);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (second, _) = coordinator(&r.store, r.dir.path(), &CancellationToken::new()).await;
    // node-a's hello no longer names the process: its assignment goes
    assigned_until(&second, "the assignment released", <[_]>::is_empty).await;
    let mut next = undecided(&a, 0).await;
    assert_eq!(told(&mut next).await.role, Role::Worker as i32);
    r.cancel.cancel();
}

/// A daemon on node `node`, played by the test: its hello names k3-a, and it asks what the
/// process on its connection 1 is; the answer's role.
async fn ask_as(address: &Address, node: &str) -> (everserve_wire::transport::Stream, i32) {
    let mut link = everserve_wire::transport::connect(address, None, "coordinator")
        .await
        .expect("connect");
    let hello = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::Hello(DaemonHello {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            node: node.into(),
            engines: vec![EngineState {
                engine: ENGINE.into(),
                ..EngineState::default()
            }],
            ..DaemonHello::default()
        })),
    };
    frame::send(&mut link, &hello).await.expect("hello");
    let ask = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::RoleAsked(RoleAsked {
            request_id: 1,
            engine: ENGINE.into(),
            rank: 0,
            connection: Some(ConnectionRef {
                instance: 7,
                serial: 1,
                pid: 0,
            }),
            identity: Some(common::identity()),
            stage: Vec::new(),
            host: String::new(),
        })),
    };
    frame::send(&mut link, &ask).await.expect("ask");
    let role = within("the answer", async {
        loop {
            let msg: CoordToDaemon = frame::recv(&mut link).await.expect("frame");
            if let Some(coord_to_daemon::Kind::RoleAnswer(a)) = msg.kind {
                return a.role;
            }
        }
    })
    .await;
    (link, role)
}

#[tokio::test]
async fn an_assignment_on_a_node_that_never_returns_after_a_restart_is_released_once_settled() {
    // a process on node "far" is told to serve rank 0; the coordinator restarts and "far" never
    // links again: until the daemons have had their time to return its assignment stands, after
    // it the node is taken as gone
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let (far, role) = ask_as(&r.address, "far").await;
    assert_eq!(role, Role::Worker as i32);
    r.coord_cancel.cancel();
    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(far);
    let (second, _) = coordinator(&r.store, r.dir.path(), &CancellationToken::new()).await;
    let mut early = undecided(&a, 0).await;
    assert_eq!(
        told(&mut early).await.role,
        Role::Spare as i32,
        "far may still return: its process holds rank 0"
    );
    drop(early);
    tokio::time::sleep(everserve_controller::coord::SETTLE).await;
    let mut late = undecided(&a, 0).await;
    assert_eq!(
        told(&mut late).await.role,
        Role::Worker as i32,
        "far never returned: its assignment is released"
    );
    assert!(
        second
            .assignments(ENGINE)
            .iter()
            .all(|a| a.node == "node-a")
    );
    r.cancel.cancel();
}

/// A process registering undecided for the stage of `ranks` with the daemon at `sock`.
async fn undecided_stage(sock: &Path, ranks: &[u32]) -> UnixStream {
    common::register_as(
        sock,
        everserve_controller::proto::everserve::signals::v1::Register {
            stage: ranks.to_vec(),
            ..common::registration(ranks[0], ranks[0], Role::Undecided)
        },
    )
    .await
}

#[tokio::test]
async fn a_stage_is_decided_as_one_and_each_rank_is_spent_by_its_own_worker() {
    // a pipeline stage's ranks move together (0022): one process decides for the stage, and its
    // workers register on their own connections from the same node
    let r = rig().await;
    let (a, c) = (r.daemon("node-a").await, r.daemon("node-c").await);
    let mut stage = undecided_stage(&a, &[0, 1]).await;
    let told_stage = told(&mut stage).await;
    assert_eq!(
        (told_stage.role, told_stage.stage.as_slice()),
        (Role::Worker as i32, [0, 1].as_slice())
    );
    // another process for the stage, or for one of its ranks, stands
    let mut other = undecided_stage(&c, &[0, 1]).await;
    assert_eq!(told(&mut other).await.role, Role::Spare as i32);
    let mut one = undecided(&c, 1).await;
    assert_eq!(told(&mut one).await.role, Role::Spare as i32);
    // a worker for rank 0 on another node takes nothing: rank 0 is node-a's process's
    let _stray = common::register(&c, 0, 0, Role::Worker).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        r.coord
            .engine(ENGINE)
            .await
            .is_none_or(|v| !v.members.contains_key(&0)),
        "a worker on another node was taken as rank 0's holder"
    );
    // node-a's workers register on their own connections: each spends its rank
    let _w0 = common::register(&a, 0, 0, Role::Worker).await;
    assigned_until(&r.coord, "rank 0 spent, rank 1 still held", |a| {
        a.len() == 1 && a[0].rank == 1
    })
    .await;
    // the stage's process leaves before rank 1's worker came: rank 1 alone is released
    drop(stage);
    assigned_until(&r.coord, "rank 1 released", <[_]>::is_empty).await;
    view_until(&r.coord, "rank 0 still its worker's", |v| {
        v.members.get(&0).map(String::as_str) == Some("node-a")
    })
    .await;
    let mut next = undecided(&a, 1).await;
    assert_eq!(told(&mut next).await.role, Role::Worker as i32);
    r.cancel.cancel();
}

#[tokio::test]
async fn a_process_is_given_its_engines_policy_on_its_link_before_it_decides() {
    // decision 0041: the coordinator holds the only copy of the policy, and the processes take
    // theirs from it through their daemon, never from a file of their own
    use everserve_controller::proto::everserve::policy::v1::Policy;
    use prost::Message as _;
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let mut asker = undecided(&a, 0).await;
    let first = within("the first signal", async {
        let sig: Signal = frame::recv(&mut asker).await.expect("frame");
        sig
    })
    .await;
    let Some(signal::Kind::EnginePolicy(p)) = first.kind else {
        panic!("the policy comes before the answer: {first:?}")
    };
    let policy = Policy::decode(p.policy.as_slice()).expect("a policy");
    assert_eq!(p.engine, ENGINE);
    assert_eq!(
        policy
            .engines
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>(),
        [ENGINE],
        "narrowed to the process's engine"
    );
    assert!(
        !policy.outcomes_in_order.is_empty(),
        "with the policy-wide fields"
    );
    assert_eq!(told(&mut asker).await.role, Role::Worker as i32);
    // a process registering as itself is given it too
    let mut worker = common::register(&a, 1, 1, Role::Worker).await;
    within("the worker's policy", async {
        loop {
            let sig: Signal = frame::recv(&mut worker).await.expect("frame");
            if matches!(sig.kind, Some(signal::Kind::EnginePolicy(_))) {
                return;
            }
        }
    })
    .await;
    r.cancel.cancel();
}

/// The next placement `s`'s process is handed.
async fn next_placement(
    s: &mut UnixStream,
) -> everserve_controller::proto::everserve::signals::v1::EnginePlacement {
    within("a placement", async {
        loop {
            let sig: Signal = frame::recv(s).await.expect("frame");
            if let Some(signal::Kind::EnginePlacement(p)) = sig.kind {
                return p;
            }
        }
    })
    .await
}

#[tokio::test]
async fn a_shape_only_engines_processes_learn_where_it_runs_before_anything_loads() {
    // decision 0041: no file names each rank's node and host; once every rank is placed by the
    // process told to serve it, every process of the engine is handed the placement it joins by
    let dir = tempfile::tempdir().expect("tmp");
    let store = Arc::new(FileStore::new(dir.path().join("spec.pb")));
    let mut p = policy::load(&common::example("policy-two-node.json")).expect("policy");
    p.engines[0].members.clear();
    p.engines[0].spares.clear();
    let store_dyn: Arc<dyn SpecStore> = store.clone();
    let (coord, _rx) = Coordinator::with_store(p, &Telemetry::new(), store_dyn, 0)
        .await
        .expect("coordinator");
    let cancel = CancellationToken::new();
    let address = common::serve(&coord, dir.path(), &cancel).await;
    let (_, a) = common::daemon(dir.path(), "node-a", &address, Vec::new(), &cancel).await;
    let (_, b) = common::daemon(dir.path(), "node-b", &address, Vec::new(), &cancel).await;
    let (_, c) = common::daemon(dir.path(), "node-c", &address, Vec::new(), &cancel).await;
    let stage = |sock: &Path, ranks: &[u32], host: &str| {
        let reg = everserve_controller::proto::everserve::signals::v1::Register {
            stage: ranks.to_vec(),
            host: host.into(),
            ..common::registration(ranks[0], ranks[0], Role::Undecided)
        };
        let sock = sock.to_owned();
        async move { common::register_as(&sock, reg).await }
    };
    let mut first = stage(&a, &[0, 1, 2, 3], "10.0.0.1").await;
    assert_eq!(told(&mut first).await.role, Role::Worker as i32);
    let mut second = stage(&b, &[4, 5, 6, 7], "10.0.0.2").await;
    assert_eq!(told(&mut second).await.role, Role::Worker as i32);
    // every rank placed by its stage's process: both, silent, are handed the placement
    for s in [&mut first, &mut second] {
        let placed = next_placement(s).await;
        let members: Vec<(u32, &str, Option<u32>, &str)> = placed
            .members
            .iter()
            .map(|m| (m.rank, m.node.as_str(), m.device, m.host.as_str()))
            .collect();
        assert_eq!(members.len(), 8, "{members:?}");
        assert_eq!(members[0], (0, "node-a", None, "10.0.0.1"));
        assert_eq!(members[7], (7, "node-b", None, "10.0.0.2"));
    }
    // a process told to stand, alone of its engine on its node, is handed the placement too: a
    // spare bootstraps from it
    let mut third = stage(&c, &[0, 1, 2, 3], "10.0.0.3").await;
    assert_eq!(told(&mut third).await.role, Role::Spare as i32);
    assert_eq!(next_placement(&mut third).await.members.len(), 8);
    // and so is one on a node whose daemon links only now, at once, not at the next change
    let (_, late) = common::daemon(dir.path(), "node-d", &address, Vec::new(), &cancel).await;
    let mut fourth = stage(&late, &[4, 5, 6, 7], "10.0.0.4").await;
    assert_eq!(told(&mut fourth).await.role, Role::Spare as i32);
    assert_eq!(next_placement(&mut fourth).await.members.len(), 8);
    // a spare standing for rank 4 changes where the engine runs: handed again, with the spare
    let _spare = common::register_as(
        &c,
        everserve_controller::proto::everserve::signals::v1::Register {
            host: "10.0.0.3".into(),
            ..common::registration(4, 0, Role::Spare)
        },
    )
    .await;
    let placed = within("the spare in the placement", async {
        loop {
            let p = next_placement(&mut first).await;
            if !p.spares.is_empty() {
                return p;
            }
        }
    })
    .await;
    let spare = &placed.spares[0];
    assert_eq!(
        (spare.for_rank, spare.node.as_str(), spare.host.as_str()),
        (Some(4), "node-c", "10.0.0.3")
    );
    cancel.cancel();
}

#[tokio::test]
async fn an_undecided_registration_needs_protocol_eleven() {
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let mut old = common::register_as(
        &a,
        everserve_controller::proto::everserve::signals::v1::Register {
            protocol: 10,
            protocol_oldest: 10,
            ..common::registration(0, 0, Role::Undecided)
        },
    )
    .await;
    assert!(
        closed(&mut old).await,
        "an undecided process of protocol 10 is refused"
    );
    r.cancel.cancel();
}

#[tokio::test]
async fn a_downed_engine_starts_anew_when_its_last_process_is_gone_and_one_asks() {
    // a head that fails and is restarted in place (as Grove restarts its pod) must not be told to
    // stand for ranks that once had a holder, or the engine never comes back: a downed engine
    // with no process of its old incarnation left starts anew
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let worker = common::register(&a, 0, 0, Role::Worker).await;
    let head = common::register_head(&a, std::process::id()).await;
    view_until(&r.coord, "the engine with its head", |v| {
        v.head.is_some() && v.members.contains_key(&0)
    })
    .await;
    drop(head);
    view_until(&r.coord, "the engine down", |v| v.down.is_some()).await;
    // a process of the old incarnation still serves: no restart, two incarnations never mix
    let mut early = undecided(&a, 0).await;
    assert_eq!(told(&mut early).await.role, Role::Spare as i32);
    drop(early);
    // its last process gone, the next asker restarts it, decided as at bring-up
    drop(worker);
    view_until(&r.coord, "the old worker gone", |v| v.members.is_empty()).await;
    // built anew with other scheduler limits: the new incarnation's step config is its own
    let rebuilt = everserve_controller::proto::everserve::signals::v1::Register {
        identity: Some(
            everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
                step: Some(common::step(&[
                    ("max_num_seqs", "128"),
                    ("block_size", "16"),
                ])),
                ..common::identity()
            },
        ),
        ..common::registration(0, 0, Role::Undecided)
    };
    let mut restarted = common::register_as(&a, rebuilt).await;
    let answer = told(&mut restarted).await;
    assert!(answer.refusal.is_empty(), "{answer:?}");
    assert_eq!(answer.role, Role::Worker as i32);
    let view = r.coord.engine(ENGINE).await.expect("view");
    assert!(view.down.is_none(), "up again");
    assert!(
        view.status
            .events()
            .iter()
            .any(|e| e.reason == "EngineRestarted"),
        "the restart is the engine's event"
    );
    r.cancel.cancel();
}

#[tokio::test]
async fn a_process_whose_step_config_differs_from_its_engines_is_refused_before_it_loads() {
    // a stage built with other scheduler limits than its head's joins the engine's collectives
    // in another shape and hangs in them: its ask names the difference, and the process is told
    // to stop rather than load
    let r = rig().await;
    let (a, b) = (r.daemon("node-a").await, r.daemon("node-b").await);
    let mut first = undecided(&a, 0).await;
    assert_eq!(told(&mut first).await.role, Role::Worker as i32);
    let other = everserve_controller::proto::everserve::signals::v1::Register {
        identity: Some(
            everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
                step: Some(common::step(&[
                    ("max_num_seqs", "128"),
                    ("block_size", "16"),
                ])),
                ..common::identity()
            },
        ),
        ..common::registration(4, 0, Role::Undecided)
    };
    let mut odd = common::register_as(&b, other).await;
    let refused = told(&mut odd).await;
    assert!(
        refused
            .refusal
            .contains(r#"max_num_seqs is "128" here, "256" on the engine"#),
        "{refused:?}"
    );
    assert!(closed(&mut odd).await, "its connection ends");
    let view = r.coord.engine(ENGINE).await.expect("view");
    assert!(
        view.status
            .events()
            .iter()
            .any(|e| e.reason == "StepRefused" && e.message.contains("max_num_seqs")),
        "the refusal is the engine's event"
    );
    assert!(
        !r.coord.assignments(ENGINE).iter().any(|a| a.rank == 4),
        "a refused process is assigned nothing"
    );
    // one built the same way is answered as before
    let mut same = undecided(&b, 4).await;
    let answer = told(&mut same).await;
    assert!(answer.refusal.is_empty(), "{answer:?}");
    assert_eq!(answer.role, Role::Worker as i32);
    r.cancel.cancel();
}

#[tokio::test]
async fn a_process_started_for_another_guarantee_than_its_policys_is_refused_before_it_loads() {
    // an exact policy run over a continuous engine: every process says the guarantee it was
    // started for, and one that is not its engine's in the policy is
    // refused at its ask, before it is the engine's first and sets its step config
    use everserve_controller::proto::everserve::signals::v1::{Register, SoftwareIdentity};
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let ask = |guarantee: &str| Register {
        identity: Some(SoftwareIdentity {
            guarantee: guarantee.into(),
            ..common::identity()
        }),
        ..common::registration(0, 0, Role::Undecided)
    };
    let mut continuous = common::register_as(&a, ask("continuous")).await;
    let refused = told(&mut continuous).await;
    assert!(
        refused
            .refusal
            .contains(r#"the guarantee is "continuous" here, "exact" in the policy"#),
        "{refused:?}"
    );
    assert!(closed(&mut continuous).await, "its connection ends");
    let mut none = common::register_as(&a, ask("")).await;
    assert!(
        told(&mut none)
            .await
            .refusal
            .contains("the guarantee is empty here")
    );
    let view = r.coord.engine(ENGINE).await.expect("view");
    assert!(
        view.status
            .events()
            .iter()
            .any(|e| e.reason == "GuaranteeRefused" && e.message.contains("continuous")),
        "the refusal is the engine's event"
    );
    // the engine's own guarantee is answered as before
    let mut exact = common::register_as(&a, ask("exact")).await;
    let answer = told(&mut exact).await;
    assert!(answer.refusal.is_empty(), "{answer:?}");
    assert_eq!(answer.role, Role::Worker as i32);
    r.cancel.cancel();
}

#[tokio::test]
async fn a_process_a_guard_refused_says_why_and_its_engine_shows_it() {
    // a process refused before it serves (its store too small, its step config another) says
    // so as it leaves, and the operator reads it in the engine's events, not only in a pod's log
    use everserve_controller::proto::everserve::signals::v1::{Goodbye, StopCause};
    let r = rig().await;
    let a = r.daemon("node-a").await;
    let mut w = common::register(&a, 0, 0, Role::Worker).await;
    common::wait_members(&r.coord, 1).await;
    let why = "engine k3-a rank 0 writes 6014144 slot ids, this store holds 4096";
    frame::send(
        &mut w,
        &Signal {
            kind: Some(signal::Kind::Goodbye(Goodbye {
                cause: StopCause::Refused as i32,
                detail: why.into(),
            })),
        },
    )
    .await
    .expect("goodbye");
    drop(w);
    view_until(&r.coord, "the refusal among the engine's events", |v| {
        v.status
            .events()
            .iter()
            .any(|e| e.reason == "ProcessRefused" && e.message.contains(why))
    })
    .await;
    r.cancel.cancel();
}

#[tokio::test]
async fn a_ranks_compile_cache_and_the_stores_holding_it_are_in_its_placement() {
    // decision 0043: a spare fetches its rank's compile cache from the stores the placement names,
    // the serving rank's first, then a spare's admitted on the same pair
    use everserve_controller::proto::everserve::signals::v1::{CompileCache, Register};
    let dir = tempfile::tempdir().expect("tmp");
    let p = policy::load(&common::example("policy-two-node.json")).expect("policy");
    let (coord, _rx) = Coordinator::new(p, &Telemetry::new());
    let cancel = CancellationToken::new();
    let address = common::serve(&coord, dir.path(), &cancel).await;
    let mut socks = std::collections::BTreeMap::new();
    for (node, store) in [
        ("node-a", "10.0.0.1:7100"),
        ("node-b", "10.0.0.2:7100"),
        ("node-c", "10.0.0.3:7100"),
    ] {
        let d = common::new_daemon(node, Vec::new()).await;
        d.set_mirror(
            dir.path().join(format!("{node}-m.sock")),
            Some(store.into()),
        )
        .await;
        socks.insert(node, common::start(&d, dir.path(), &address, &cancel).await);
    }
    let pair = CompileCache {
        key: "586b266d".into(),
        digest: "d0".into(),
    };
    let mut workers = Vec::new();
    for r in 0..4 {
        workers.push(
            common::register_as(
                &socks["node-a"],
                Register {
                    compile_cache: (r == 0).then(|| pair.clone()),
                    ..common::registration(r, r, Role::Worker)
                },
            )
            .await,
        );
        workers.push(common::register(&socks["node-b"], 4 + r, r, Role::Worker).await);
    }
    let rank0 = |p: &everserve_controller::proto::everserve::signals::v1::EnginePlacement| {
        p.members
            .iter()
            .find(|m| m.rank == 0)
            .cloned()
            .expect("rank 0")
    };
    let placed = within("rank 0's pair in the placement", async {
        loop {
            let p = next_placement(&mut workers[0]).await;
            if rank0(&p).compile_cache.is_some() {
                return p;
            }
        }
    })
    .await;
    assert_eq!(rank0(&placed).compile_cache, Some(pair.clone()));
    assert_eq!(rank0(&placed).cache_holders, ["10.0.0.1:7100"]);
    assert!(
        placed
            .members
            .iter()
            .filter(|m| m.rank != 0)
            .all(|m| m.compile_cache.is_none() && m.cache_holders.is_empty()),
        "a rank that reported no pair has no holders: {placed:?}"
    );

    // a spare for rank 0 on node-c, on the rank's pair: its store holds the set too
    let _spare = common::register_as(
        &socks["node-c"],
        Register {
            compile_cache: Some(pair.clone()),
            ..common::registration(0, 0, Role::Spare)
        },
    )
    .await;
    let placed = within("node-c's store among the holders", async {
        loop {
            let p = next_placement(&mut workers[0]).await;
            if rank0(&p).cache_holders.len() == 2 {
                return p;
            }
        }
    })
    .await;
    assert_eq!(
        rank0(&placed).cache_holders,
        ["10.0.0.1:7100", "10.0.0.3:7100"]
    );
    cancel.cancel();
}

/// A node's store keeping compile caches under `root`: its node socket and its peer socket.
async fn store_with_caches(
    dir: &Path,
    node: &str,
    root: &Path,
    cancel: &CancellationToken,
) -> (std::path::PathBuf, std::path::PathBuf) {
    use everserve_store::cache::CompileCaches;
    let service = everserve_store::service::Service::with_tls(
        everserve_store::storage::Storage::new(dir.join(format!("{node}-data")), 64, 1 << 20),
        &Telemetry::new(),
        None,
        "store",
        Some(CompileCaches::new(root, 1 << 30, None, "store")),
    );
    let peer = dir.join(format!("{node}-peer.sock"));
    let listener = everserve_wire::transport::Listener::bind(&Address::Unix(peer.clone()), None)
        .await
        .expect("bind");
    tokio::spawn(Arc::clone(&service).serve_peers(listener, cancel.clone()));
    let sock = dir.join(format!("{node}-m.sock"));
    tokio::spawn(service.serve(sock.clone(), cancel.clone()));
    tokio::time::sleep(Duration::from_millis(50)).await;
    (sock, peer)
}

#[tokio::test]
async fn a_spare_is_given_its_ranks_compile_cache_from_the_serving_ranks_node() {
    // decision 0043: told to stand, a spare asks for its rank's compile cache; its node's daemon
    // has its store fetch the set from the store the placement names, and answers once it is on
    // the node; a node with no store refuses
    use everserve_controller::proto::everserve::signals::v1::{
        CacheWanted, CompileCache, Register,
    };
    let dir = tempfile::tempdir().expect("tmp");
    let p = policy::load(&common::example("policy-two-node.json")).expect("policy");
    let (coord, _rx) = Coordinator::new(p, &Telemetry::new());
    let cancel = CancellationToken::new();
    let address = common::serve(&coord, dir.path(), &cancel).await;
    let root = |node: &str| dir.path().join(format!("{node}-cache"));
    let key_dir = |node: &str| root(node).join("torch_compile_cache/torch_aot_compile/586b266d");
    // rank 0's set on node-a, as its serving worker left it
    std::fs::create_dir_all(key_dir("node-a").join("rank_0_0")).expect("dirs");
    std::fs::write(key_dir("node-a").join("rank_0_0/model"), b"graph").expect("graph");
    let sha = ring_sha256(b"graph");
    std::fs::write(
        key_dir("node-a").join("rank_0_0.manifest.json"),
        format!(
            r#"{{"key":"586b266d","rank":0,"dp":0,"digest":"d0","files":[{{"path":"rank_0_0/model","size":5,"sha256":"{sha}"}}]}}"#
        ),
    )
    .expect("manifest");
    let mut socks = std::collections::BTreeMap::new();
    for node in ["node-a", "node-b", "node-c"] {
        let d = common::new_daemon(node, Vec::new()).await;
        if node != "node-b" {
            let (sock, peer) = store_with_caches(dir.path(), node, &root(node), &cancel).await;
            d.set_mirror(sock, Some(peer.display().to_string())).await;
        }
        socks.insert(node, common::start(&d, dir.path(), &address, &cancel).await);
    }
    let pair = CompileCache {
        key: "586b266d".into(),
        digest: "d0".into(),
    };
    let mut workers = Vec::new();
    for r in 0..4 {
        workers.push(
            common::register_as(
                &socks["node-a"],
                Register {
                    compile_cache: (r == 0).then(|| pair.clone()),
                    ..common::registration(r, r, Role::Worker)
                },
            )
            .await,
        );
        workers.push(common::register(&socks["node-b"], 4 + r, r, Role::Worker).await);
    }

    let ask = async |sock: &Path| {
        let mut spare = undecided(sock, 0).await;
        assert_eq!(told(&mut spare).await.role, Role::Spare as i32);
        frame::send(
            &mut spare,
            &Signal {
                kind: Some(signal::Kind::CacheWanted(CacheWanted {
                    engine: common::ENGINE.into(),
                    rank: 0,
                    dp: 0,
                })),
            },
        )
        .await
        .expect("ask");
        within("the cache's answer", async {
            loop {
                let sig: Signal = frame::recv(&mut spare).await.expect("frame");
                match sig.kind {
                    Some(signal::Kind::CacheReady(r)) => return Ok(r),
                    Some(signal::Kind::CacheRefused(r)) => return Err(r),
                    _ => {}
                }
            }
        })
        .await
    };
    let ready = ask(&socks["node-c"]).await.expect("ready");
    assert_eq!(
        (ready.key.as_str(), ready.digest.as_str()),
        ("586b266d", "d0")
    );
    assert_eq!(
        std::fs::read(key_dir("node-c").join("rank_0_0/model")).expect("on node-c"),
        b"graph"
    );
    assert!(key_dir("node-c").join("rank_0_0.manifest.json").exists());

    // node-b has no store to fetch into: refused, and the spare never compiles for itself
    let refused = ask(&socks["node-b"]).await.expect_err("refused");
    assert!(refused.reason.contains("no store"), "{refused:?}");
    cancel.cancel();
}

fn ring_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write;
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[tokio::test]
async fn a_head_that_registers_never_sets_the_engines_step() {
    let r = rig().await;
    let cpu = r.daemon("cpu-x").await;
    // a head whose step config differs from the ranks' that follow
    let head = everserve_controller::proto::everserve::signals::v1::Register {
        role: Role::Head as i32,
        launcher_pid: std::process::id(),
        identity: Some(
            everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
                step: Some(common::step(&[
                    ("max_num_seqs", "128"),
                    ("block_size", "16"),
                ])),
                ..common::identity()
            },
        ),
        ..common::registration(0, 0, Role::Head)
    };
    let _head = common::register_as(&cpu, head).await;
    view_until(&r.coord, "the head registered", |v| v.head.is_some()).await;
    let a = r.daemon("node-a").await;
    let mut rank = undecided(&a, 0).await;
    let answer = told(&mut rank).await;
    assert!(
        answer.refusal.is_empty() && answer.role == Role::Worker as i32,
        "a head's registration set the engine's step config: {answer:?}"
    );
    r.cancel.cancel();
}
