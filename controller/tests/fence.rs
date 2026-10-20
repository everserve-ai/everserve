// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! One writer, fenced (decision 0036): the lease's term travels on every coordinator message; a
//! daemon refuses a term older than the newest it has seen, answering a fenced delivery at once,
//! and reports its newest term in its hello; a coordinator that hears a newer term than its own
//! knows it is deposed and claims no change.

mod common;

use std::sync::Arc;
use std::time::Duration;

use everserve_controller::coord::{Cause, ChangeError, Coordinator};
use everserve_controller::node::Daemon;
use everserve_controller::policy;
use everserve_controller::proto::everserve::coord::v1::{
    CoordToDaemon, DaemonHello, DaemonToCoord, Deliver, EngineState, coord_to_daemon,
    daemon_to_coord,
};
use everserve_controller::proto::everserve::signals::v1::{
    Change, CoverPlan, Member, Notice, Reason, Register, Role, Signal, signal,
};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::{Address, Listener, Stream, connect};
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

fn plan(term: u64, generation: u64) -> CoordToDaemon {
    CoordToDaemon {
        term,
        kind: Some(coord_to_daemon::Kind::CoverPlan(CoverPlan {
            engine: "k3-a".into(),
            generation,
            ..CoverPlan::default()
        })),
    }
}

async fn hello(listener: &Listener) -> (Stream, DaemonHello) {
    tokio::time::timeout(Duration::from_secs(5), hello_unbounded(listener))
        .await
        .expect("a daemon's hello within 5 s")
}

async fn hello_unbounded(listener: &Listener) -> (Stream, DaemonHello) {
    let mut link = listener.accept().await.expect("accept").stream;
    let msg: DaemonToCoord = frame::recv(&mut link).await.expect("hello");
    let Some(daemon_to_coord::Kind::Hello(h)) = msg.kind else {
        panic!("not a hello: {msg:?}");
    };
    (link, h)
}

#[tokio::test]
async fn a_daemon_refuses_an_older_term_and_answers_a_fenced_delivery() {
    let dir = tempfile::tempdir().expect("tmp");
    let address = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    let cancel = CancellationToken::new();
    let (d, _rx) = Daemon::new("node-a", &Telemetry::new());
    let sock = dir.path().join("node-a.sock");
    tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel.clone()));
    tokio::spawn(Arc::clone(&d).link(address, None, "coordinator".into(), cancel.clone()));
    let (mut link, h) = hello(&listener).await;
    assert_eq!(h.term, 0, "no term seen yet");
    for _ in 0..200 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mut w = UnixStream::connect(&sock).await.expect("connect");
    let reg = Signal {
        kind: Some(signal::Kind::Register(Register {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            engine: "k3-a".into(),
            rank: 0,
            member: Some(Member {
                node: "n".into(),
                device: 0,
            }),
            role: Role::Worker as i32,
            identity: Some(common::identity()),
            ..Register::default()
        })),
    };
    frame::send(&mut w, &reg).await.expect("register");

    // term 5 is heard; a message at term 3 after it moves nothing
    frame::send(&mut link, &plan(5, 1)).await.expect("plan");
    assert_eq!(common::next_plan(&mut w).await.generation, 1);
    frame::send(&mut link, &plan(3, 2))
        .await
        .expect("stale plan");
    // a fenced delivery is answered at once, so a deposed coordinator's change fails fast
    let deliver = CoordToDaemon {
        term: 3,
        kind: Some(coord_to_daemon::Kind::Deliver(Deliver {
            request_id: 42,
            engine: "k3-a".into(),
            change: Some(Change {
                epoch: 1,
                ..Change::default()
            }),
            to_ranks: vec![0],
            ..Deliver::default()
        })),
    };
    frame::send(&mut link, &deliver)
        .await
        .expect("stale delivery");
    let answer = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let msg: DaemonToCoord = frame::recv(&mut link).await.expect("frame");
            if let Some(daemon_to_coord::Kind::Delivered(d)) = msg.kind {
                return d;
            }
        }
    })
    .await
    .expect("answered");
    assert_eq!(answer.request_id, 42);
    assert!(answer.error.contains("term 3 refused"), "{}", answer.error);
    // the next plan the worker sees is term 5's generation 3, never term 3's generation 2
    frame::send(&mut link, &plan(5, 3)).await.expect("plan");
    assert_eq!(common::next_plan(&mut w).await.generation, 3);

    // the next hello reports the newest term
    drop(link);
    let (_link, h) = hello(&listener).await;
    assert_eq!(h.term, 5);
    cancel.cancel();
}

#[tokio::test]
async fn a_coordinator_stamps_its_term_and_one_that_hears_a_newer_term_claims_nothing() {
    let dir = tempfile::tempdir().expect("tmp");
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-pilot.json");
    let (coord, _rx) = Coordinator::new(policy::load(&path).expect("policy"), &Telemetry::new());
    coord.act_under(7, CancellationToken::new()).expect("act");
    let address = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    let cancel = CancellationToken::new();
    tokio::spawn(Arc::clone(&coord).serve(listener, cancel.clone()));
    let daemon_hello = |node: &str, term: u64| DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::Hello(DaemonHello {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            node: node.into(),
            engines: vec![EngineState {
                engine: "k3-a".into(),
                workers: vec![0, 1],
                world_size: 2,
                ..EngineState::default()
            }],
            term,
            ..DaemonHello::default()
        })),
    };
    let mut a = connect(&address, None, "coordinator")
        .await
        .expect("connect");
    frame::send(&mut a, &daemon_hello("node-a", 7))
        .await
        .expect("hello");
    for _ in 0..300 {
        if coord.engine("k3-a").await.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // a notice goes out stamped with term 7
    let notice = Notice {
        rank: 1,
        deadline_ms: everserve_controller::clock::now_ms() + 600_000,
        reason: Reason::Operator as i32,
    };
    tokio::spawn(Arc::clone(&coord).notice("k3-a", notice));
    let msg = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let m: CoordToDaemon = frame::recv(&mut a).await.expect("frame");
            if matches!(m.kind, Some(coord_to_daemon::Kind::ForwardNotice(_))) {
                return m;
            }
        }
    })
    .await
    .expect("a notice");
    assert_eq!(msg.term, 7);
    assert!(!coord.deposed());

    // a daemon that has seen term 9 says hello: another coordinator holds the lease
    let mut b = connect(&address, None, "coordinator")
        .await
        .expect("connect");
    frame::send(&mut b, &daemon_hello("node-b", 9))
        .await
        .expect("hello");
    for _ in 0..300 {
        if coord.deposed() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(coord.deposed());
    match coord.rank_lost("k3-a", 0, Cause::Fault).await {
        Err(ChangeError::Deposed { term, newer }) => assert_eq!((term, newer), (7, 9)),
        other => panic!("{other:?}"),
    }
    cancel.cancel();
}

#[tokio::test]
async fn a_coordinator_whose_hold_ended_sends_nothing_and_claims_nothing() {
    let dir = tempfile::tempdir().expect("tmp");
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-pilot.json");
    let (coord, _rx) = Coordinator::new(policy::load(&path).expect("policy"), &Telemetry::new());
    let ended = CancellationToken::new();
    coord.act_under(7, ended.clone()).expect("act");
    assert!(
        coord.act_under(8, CancellationToken::new()).is_err(),
        "one hold for a coordinator's life"
    );
    let address = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    let cancel = CancellationToken::new();
    tokio::spawn(Arc::clone(&coord).serve(listener, cancel.clone()));
    let mut a = connect(&address, None, "coordinator")
        .await
        .expect("connect");
    let hello = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::Hello(DaemonHello {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            node: "node-a".into(),
            engines: vec![EngineState {
                engine: "k3-a".into(),
                workers: vec![0, 1],
                world_size: 2,
                ..EngineState::default()
            }],
            term: 7,
            ..DaemonHello::default()
        })),
    };
    frame::send(&mut a, &hello).await.expect("hello");
    for _ in 0..300 {
        if coord.engine("k3-a").await.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // the hold ends: the coordinator acts under its token, with no relay between
    ended.cancel();
    assert!(coord.deposed());
    match coord.rank_lost("k3-a", 0, Cause::Fault).await {
        Err(ChangeError::LeaseLost { term }) => assert_eq!(term, 7),
        other => panic!("{other:?}"),
    }
    // a notice it is asked to forward is never sent: the link ends at it instead
    let notice = Notice {
        rank: 1,
        deadline_ms: everserve_controller::clock::now_ms() + 600_000,
        reason: Reason::Operator as i32,
    };
    tokio::spawn(Arc::clone(&coord).notice("k3-a", notice));
    let ended = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match frame::recv::<_, CoordToDaemon>(&mut a).await {
                Ok(m) => assert!(
                    !matches!(m.kind, Some(coord_to_daemon::Kind::ForwardNotice(_))),
                    "a coordinator whose hold ended sent {m:?}"
                ),
                Err(_) => return,
            }
        }
    })
    .await;
    assert!(ended.is_ok(), "the link ended");
    cancel.cancel();
}

// ---------------------------------------------------------------------------------------------
// Following the lease (decision 0036): daemons learn the active coordinator and its term from the
// lease's record.

use everserve_controller::lease::{FileLease, FileLeaseWatch, Lease};
use everserve_controller::proto::everserve::coord::v1::LeaseRecord;

/// A daemon following the lease at `lease` (read every 20 ms), with a worker registered on it.
async fn following(
    dir: &std::path::Path,
    lease: &std::path::Path,
    cancel: &CancellationToken,
) -> (Arc<Daemon>, UnixStream) {
    let (d, _rx) = Daemon::new("node-a", &Telemetry::new());
    let sock = dir.join("node-a.sock");
    tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel.clone()));
    let (target, followed) = tokio::sync::watch::channel(None);
    tokio::spawn(Arc::clone(&d).watch_lease(
        Arc::new(FileLeaseWatch::new(lease)),
        target,
        Duration::from_millis(20),
        cancel.clone(),
    ));
    tokio::spawn(Arc::clone(&d).link_to(followed, None, "coordinator".into(), cancel.clone()));
    for _ in 0..200 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mut w = UnixStream::connect(&sock).await.expect("connect");
    let reg = Signal {
        kind: Some(signal::Kind::Register(Register {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            engine: "k3-a".into(),
            rank: 0,
            member: Some(Member {
                node: "n".into(),
                device: 0,
            }),
            role: Role::Worker as i32,
            identity: Some(common::identity()),
            ..Register::default()
        })),
    };
    frame::send(&mut w, &reg).await.expect("register");
    (d, w)
}

fn record(address: &Address) -> LeaseRecord {
    LeaseRecord {
        holder: "test".into(),
        address: address.to_string(),
        ..LeaseRecord::default()
    }
}

#[tokio::test]
async fn a_daemon_links_to_the_coordinator_the_lease_names_and_moves_with_it() {
    let dir = tempfile::tempdir().expect("tmp");
    let lease_path = dir.path().join("coord.lease");
    let a_addr = Address::Unix(dir.path().join("a.sock"));
    let b_addr = Address::Unix(dir.path().join("b.sock"));
    let a_listener = Listener::bind(&a_addr, None).await.expect("bind");
    let b_listener = Listener::bind(&b_addr, None).await.expect("bind");
    let cancel = CancellationToken::new();
    let a = FileLease::new(&lease_path)
        .try_acquire()
        .await
        .expect("acquire")
        .expect("free");
    a.publish(record(&a_addr)).await.expect("publish");
    let (d, _w) = following(dir.path(), &lease_path, &cancel).await;
    let (mut a_link, h) = hello(&a_listener).await;
    assert_eq!(
        h.term, 1,
        "the daemon read the lease's term before it linked"
    );

    // the lease moves to b: the daemon leaves a and links to b, at b's term
    a.release().await.expect("release");
    let b = FileLease::new(&lease_path)
        .try_acquire()
        .await
        .expect("acquire")
        .expect("free");
    b.publish(record(&b_addr)).await.expect("publish");
    let (_b_link, h) = hello(&b_listener).await;
    assert_eq!(h.term, 2);
    assert_eq!(d.term(), 2);
    // what the daemon sent a before it moved is still there to read; then a's link ends
    let ended = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let next: Result<DaemonToCoord, _> = frame::recv(&mut a_link).await;
            if next.is_err() {
                return;
            }
        }
    })
    .await;
    assert!(ended.is_ok(), "the daemon left the deposed coordinator");
    cancel.cancel();
}

#[tokio::test]
async fn the_lease_fences_the_old_holder_before_any_link_moves() {
    // a virtual address: the new holder answers on the old one's address, so no link moves, and
    // only the lease's term tells the daemon the old holder is deposed
    let dir = tempfile::tempdir().expect("tmp");
    let lease_path = dir.path().join("coord.lease");
    let addr = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&addr, None).await.expect("bind");
    let cancel = CancellationToken::new();
    let a = FileLease::new(&lease_path)
        .try_acquire()
        .await
        .expect("acquire")
        .expect("free");
    a.publish(record(&addr)).await.expect("publish");
    let (d, mut w) = following(dir.path(), &lease_path, &cancel).await;
    let (mut link, _) = hello(&listener).await;
    frame::send(&mut link, &plan(1, 1)).await.expect("plan");
    assert_eq!(common::next_plan(&mut w).await.generation, 1);

    a.release().await.expect("release");
    let b = FileLease::new(&lease_path)
        .try_acquire()
        .await
        .expect("acquire")
        .expect("free");
    b.publish(record(&addr)).await.expect("publish");
    for _ in 0..300 {
        if d.term() == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(d.term(), 2, "the daemon read the new term");
    // the old holder, still linked, sends at term 1: refused; term 2 passes
    frame::send(&mut link, &plan(1, 2))
        .await
        .expect("stale plan");
    frame::send(&mut link, &plan(2, 3)).await.expect("plan");
    assert_eq!(
        common::next_plan(&mut w).await.generation,
        3,
        "term 1's plan never reached the worker"
    );
    cancel.cancel();
}

#[tokio::test]
async fn a_lease_let_go_leaves_its_daemons_holding_their_reports_not_chasing_it() {
    let dir = tempfile::tempdir().expect("tmp");
    let lease_path = dir.path().join("coord.lease");
    let addr = Address::Unix(dir.path().join("a.sock"));
    let listener = Listener::bind(&addr, None).await.expect("bind");
    let cancel = CancellationToken::new();
    let a = FileLease::new(&lease_path)
        .try_acquire()
        .await
        .expect("acquire")
        .expect("free");
    a.publish(record(&addr)).await.expect("publish");
    let (d, _w) = following(dir.path(), &lease_path, &cancel).await;
    let (mut link, _) = hello(&listener).await;
    // a shuts down in order: no holder, the term kept, then the lock let go
    a.publish(LeaseRecord::default()).await.expect("publish");
    a.release().await.expect("release");
    let ended = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let next: Result<DaemonToCoord, _> = frame::recv(&mut link).await;
            if next.is_err() {
                return;
            }
        }
    })
    .await;
    assert!(ended.is_ok(), "the daemon left a's link");
    // and does not link again while no one holds the lease
    let again = tokio::time::timeout(Duration::from_millis(300), listener.accept()).await;
    assert!(again.is_err(), "no new link to an address no one holds");
    assert_eq!(d.term(), 1, "the fence keeps its term");
    cancel.cancel();
}

#[tokio::test]
async fn a_deposed_coordinator_writes_no_spec() {
    // decision 0036: the store refuses a write under a term older than the newest that wrote to
    // it, as the daemons refuse such a coordinator's messages; a coordinator that knows its hold
    // ended writes nothing at all
    use everserve_controller::coord::NodeSpecError;
    use everserve_controller::spec::{FileStore, SpecError};
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("specs.pb");
    let policy = policy::load(&common::example("policy-pilot.json")).expect("policy");
    // each takes the store under its term as it starts, as a coordinator taking the lease does
    let open = |term: u64, ended: CancellationToken| {
        let (policy, path) = (policy.clone(), path.clone());
        async move {
            let (c, _rx) = Coordinator::with_store(
                policy,
                &Telemetry::new(),
                Arc::new(FileStore::new(&path)),
                term,
            )
            .await
            .expect("store");
            c.act_under(term, ended).expect("act");
            c
        }
    };
    let cordon = || everserve_controller::coord::NodeWant {
        unschedulable: true,
        ..everserve_controller::coord::NodeWant::default()
    };
    let a = open(7, CancellationToken::new()).await;
    a.put_node_spec("node-x", "operator", cordon(), None)
        .await
        .expect("term 7 writes");
    // another coordinator takes the lease at term 8 and takes the store: term 7's write is in
    // what it read, and from now on the store refuses term 7
    let b = open(8, CancellationToken::new()).await;
    assert!(
        b.node_spec("node-x")
            .await
            .is_some_and(|s| s.generation == 1),
        "the deposed holder's write before the take is read"
    );
    // the old one, deposed without noticing, is refused by the store before term 8 writes
    // anything, and stops acting
    match a.put_node_spec("node-y", "operator", cordon(), None).await {
        Err(NodeSpecError::Store(SpecError::Fenced {
            term: 7, newer: 8, ..
        })) => {}
        other => panic!("{other:?}"),
    }
    assert!(a.deposed(), "a refused write ends the hold it acted under");
    // a spec other than term 7's, so the write reaches the store and is not taken as asked
    // already
    let rewritten = everserve_controller::coord::NodeWant {
        reason: "term 8".into(),
        ..cordon()
    };
    let written = b
        .put_node_spec("node-x", "operator", rewritten, None)
        .await
        .expect("term 8 writes");
    assert_eq!(written.generation, 2, "{written:?}");
    // and one that knows its hold ended does not try
    let ended = CancellationToken::new();
    let c = open(9, ended.clone()).await;
    ended.cancel();
    match c.put_node_spec("node-z", "operator", cordon(), None).await {
        Err(NodeSpecError::Store(SpecError::Deposed { .. })) => {}
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn the_coordinator_s_messages_are_applied_in_the_order_they_came() {
    // a commit then the delivery of the next epoch, back to back: the delivery finds the daemon
    // at the committed epoch, whatever task runs it
    use everserve_controller::proto::everserve::coord::v1::Committed;
    use everserve_controller::proto::everserve::signals::v1::MembershipCommitted;
    let dir = tempfile::tempdir().expect("tmp");
    let address = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    let cancel = CancellationToken::new();
    let (d, _rx) = Daemon::new("node-a", &Telemetry::new());
    let sock = dir.path().join("node-a.sock");
    tokio::spawn(Arc::clone(&d).serve(sock.clone(), cancel.clone()));
    tokio::spawn(Arc::clone(&d).link(address, None, "coordinator".into(), cancel.clone()));
    let (mut link, _) = hello(&listener).await;
    for _ in 0..200 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mut w = common::register(&sock, 0, 0, Role::Worker).await;
    for _ in 0..200 {
        if d.registered("k3-a").await.len() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    for epoch in 1..=20u64 {
        let committed = CoordToDaemon {
            term: 1,
            kind: Some(coord_to_daemon::Kind::Committed(Committed {
                engine: "k3-a".into(),
                membership: Some(MembershipCommitted {
                    epoch,
                    active_ranks: vec![0],
                    world_size: 1,
                    ..MembershipCommitted::default()
                }),
                ..Committed::default()
            })),
        };
        let deliver = CoordToDaemon {
            term: 1,
            kind: Some(coord_to_daemon::Kind::Deliver(Deliver {
                request_id: epoch,
                engine: "k3-a".into(),
                change: Some(Change {
                    epoch: epoch + 1,
                    ..Change::default()
                }),
                to_ranks: vec![0],
                ..Deliver::default()
            })),
        };
        frame::send(&mut link, &committed).await.expect("committed");
        frame::send(&mut link, &deliver).await.expect("deliver");
        // the worker sees the change (a commit goes to the engine's clients, not its workers),
        // which it acknowledges; the delivery is answered without a refusal
        let sig: Signal = common::recv(&mut w).await.expect("change");
        assert!(
            matches!(sig.kind, Some(signal::Kind::Change(ref c)) if c.epoch == epoch + 1),
            "epoch {epoch}: {sig:?}"
        );
        frame::send(
            &mut w,
            &Signal {
                kind: Some(signal::Kind::Ack(
                    everserve_controller::proto::everserve::signals::v1::Ack {
                        rank: 0,
                        epoch: epoch + 1,
                        error: String::new(),
                        received_at_us: 0,
                    },
                )),
            },
        )
        .await
        .expect("ack");
        let answer = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let msg: DaemonToCoord = frame::recv(&mut link).await.expect("frame");
                if let Some(daemon_to_coord::Kind::Delivered(a)) = msg.kind {
                    return a;
                }
            }
        })
        .await
        .expect("answered");
        assert_eq!(answer.request_id, epoch);
        assert_eq!(
            answer.error, "",
            "epoch {epoch}: the delivery was not refused as stale"
        );
    }
    cancel.cancel();
}
