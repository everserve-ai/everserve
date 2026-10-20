// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0

//! An engine's facts go to its audience only (`coord::audience`): a node with no process of an
//! engine hears nothing of it, and the first process of the engine to reach the coordinator from a
//! node, by asking or by registering, brings the node the engine's facts ahead of its answer.

use std::sync::Arc;
use std::time::Duration;

use everserve_controller::coord::Coordinator;
use everserve_controller::policy;
use everserve_controller::proto::everserve::coord::v1::{
    CoordToDaemon, DaemonHello, DaemonToCoord, Delivered, EngineState, RoleAsked, WorkerEvent,
    coord_to_daemon, daemon_to_coord, worker_event,
};
use everserve_controller::proto::everserve::policy::v1::{Engine, Policy, SparePool, spare_pool};
use everserve_controller::proto::everserve::signals::v1::{Ack, Member, Outcome};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::{Address, Listener, Stream, connect};
use tokio_util::sync::CancellationToken;

/// Engines e0 (on n0 and n1) and e1 (on n2 and n3), eight devices a node.
fn two_engines() -> Policy {
    let engine = |name: &str, nodes: [&str; 2]| Engine {
        name: name.into(),
        members: nodes
            .iter()
            .flat_map(|n| {
                (0..8).map(move |device| Member {
                    node: (*n).to_owned(),
                    device,
                })
            })
            .collect(),
        tensor_parallel: 8,
        data_parallel: 2,
        pipeline_parallel: 1,
        placement_independent_model: true,
        ..Engine::default()
    };
    let policy = Policy {
        version: "audience".into(),
        outcomes_in_order: vec![Outcome::Shrink as i32],
        max_concurrent_losses: 1,
        spares: Some(SparePool {
            loading: spare_pool::Loading::Preloaded as i32,
            ..SparePool::default()
        }),
        engines: vec![engine("e0", ["n0", "n1"]), engine("e1", ["n2", "n3"])],
        ..Policy::default()
    };
    policy::validate(&policy).expect("valid");
    policy
}

fn hello(node: &str, engine: Option<(&str, u32)>) -> DaemonToCoord {
    DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::Hello(DaemonHello {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            node: node.into(),
            engines: engine
                .map(|(engine, first)| EngineState {
                    engine: engine.into(),
                    workers: (first..first + 8).collect(),
                    world_size: 16,
                    ..EngineState::default()
                })
                .into_iter()
                .collect(),
            ..DaemonHello::default()
        })),
    }
}

async fn link(address: &Address, msg: &DaemonToCoord) -> Stream {
    let mut s = connect(address, None, "coordinator")
        .await
        .expect("connect");
    frame::send(&mut s, msg).await.expect("hello");
    s
}

/// The kinds of what `link` is sent within `within`, as short names with their engine.
async fn heard(link: &mut Stream, within: Duration) -> Vec<String> {
    let mut out = Vec::new();
    let _ = tokio::time::timeout(within, async {
        loop {
            let Ok(m): Result<CoordToDaemon, _> = frame::recv(link).await else {
                return;
            };
            out.push(match m.kind {
                Some(coord_to_daemon::Kind::EnginePolicy(p)) => format!("policy {}", p.engine),
                Some(coord_to_daemon::Kind::EnginePlacement(p)) => {
                    format!("placement {}", p.engine)
                }
                Some(coord_to_daemon::Kind::InstanceState(s)) => format!("state {}", s.engine),
                Some(coord_to_daemon::Kind::RoleAnswer(a)) => format!("answer {}", a.engine),
                other => format!("{other:?}"),
            });
        }
    })
    .await;
    out
}

/// Answer every delivery on `link` with an ack from each rank it names, until the link ends.
fn answering(mut link: Stream) {
    tokio::spawn(async move {
        loop {
            let Ok(m): Result<CoordToDaemon, _> = frame::recv(&mut link).await else {
                return;
            };
            let Some(coord_to_daemon::Kind::Deliver(d)) = m.kind else {
                continue;
            };
            let epoch = d.change.as_ref().map_or(0, |c| c.epoch);
            let delivered = DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::Delivered(Delivered {
                    request_id: d.request_id,
                    engine: d.engine,
                    epoch,
                    acks: d
                        .to_ranks
                        .iter()
                        .map(|&rank| Ack {
                            rank,
                            epoch,
                            ..Ack::default()
                        })
                        .collect(),
                    error: String::new(),
                })),
            };
            if frame::send(&mut link, &delivered).await.is_err() {
                return;
            }
        }
    });
}

/// A coordinator with both engines' nodes linked, answering, and each engine placed.
async fn placed(dir: &std::path::Path, cancel: &CancellationToken) -> (Arc<Coordinator>, Address) {
    let (coord, _rx) = Coordinator::new(two_engines(), &Telemetry::new());
    let address = Address::Unix(dir.join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    tokio::spawn(Arc::clone(&coord).serve(listener, cancel.clone()));
    for (node, engine, first) in [
        ("n0", "e0", 0),
        ("n1", "e0", 8),
        ("n2", "e1", 0),
        ("n3", "e1", 8),
    ] {
        answering(link(&address, &hello(node, Some((engine, first)))).await);
    }
    for _ in 0..500 {
        let mut both = true;
        for e in ["e0", "e1"] {
            both &= coord.engine(e).await.is_some_and(|v| v.placement.is_some());
        }
        if both {
            return (coord, address);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("never placed");
}

#[tokio::test]
async fn a_node_with_no_process_of_an_engine_hears_nothing_of_it() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (coord, address) = placed(dir.path(), &cancel).await;
    let mut nx = link(&address, &hello("nx", None)).await;
    // e0 changes while nx is linked: its placement goes to n0 and n1, not to nx
    coord
        .rank_lost("e0", 3, everserve_controller::coord::Cause::Fault)
        .await
        .expect("committed");
    let got = heard(&mut nx, Duration::from_millis(500)).await;
    assert!(
        got.iter().all(|h| !h.contains(" e0") && !h.contains(" e1")),
        "{got:?}"
    );
    cancel.cancel();
}

#[tokio::test]
async fn the_first_ask_from_a_node_brings_its_engines_facts_ahead_of_the_answer() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (_coord, address) = placed(dir.path(), &cancel).await;
    let mut nx = link(&address, &hello("nx", None)).await;
    let ask = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::RoleAsked(RoleAsked {
            request_id: 1,
            engine: "e0".into(),
            rank: 0,
            ..RoleAsked::default()
        })),
    };
    frame::send(&mut nx, &ask).await.expect("ask");
    let got = heard(&mut nx, Duration::from_secs(2)).await;
    let at = |what: &str| {
        got.iter()
            .position(|h| h == what)
            .unwrap_or_else(|| panic!("no {what}: {got:?}"))
    };
    assert!(at("policy e0") < at("answer e0"), "{got:?}");
    assert!(at("placement e0") < at("answer e0"), "{got:?}");
    // and nothing of the engine it has no process of
    assert!(got.iter().all(|h| !h.contains(" e1")), "{got:?}");
    cancel.cancel();
}

#[tokio::test]
async fn a_spare_that_registers_without_asking_brings_its_node_the_engines_facts() {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let (_coord, address) = placed(dir.path(), &cancel).await;
    let mut ny = link(&address, &hello("ny", None)).await;
    // a pool spare (decision 0031) registers for e1 straight away, on a node with nothing of it
    let registered = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::WorkerEvent(WorkerEvent {
            engine: "e1".into(),
            kind: worker_event::Kind::SpareRegistered as i32,
            rank: 2,
            world_size: 16,
            device: 0,
            ..WorkerEvent::default()
        })),
    };
    frame::send(&mut ny, &registered).await.expect("registered");
    let got = heard(&mut ny, Duration::from_secs(1)).await;
    assert!(got.iter().any(|h| h == "policy e1"), "{got:?}");
    assert!(got.iter().any(|h| h == "placement e1"), "{got:?}");
    assert!(got.iter().all(|h| !h.contains(" e0")), "{got:?}");
    cancel.cancel();
}
