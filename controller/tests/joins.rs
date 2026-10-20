// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Joins and replays: a join payload routed to the rank's process and its answer back to the
//! clients; a spare's replay ask reaching the head and the offer that asker alone (decision 0044),
//! an offer from a node without the head refused, and a spare not yet registered asking on a
//! connection of its own.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use everserve_controller::coord::Cause;
use everserve_controller::proto::everserve::signals::v1::{Join, Role, Signal, signal};
use everserve_wire::frame;
use tokio::net::UnixStream;

use common::cluster::{Cluster, cluster, expect_change, two_node_engine, within};

/// Read frames from `w` until a join arrives; a client hears the claimed change and the commit
/// first.
async fn expect_join(w: &mut UnixStream) -> Join {
    loop {
        let sig: Signal = common::recv(w).await.expect("frame");
        if let Some(signal::Kind::Join(j)) = sig.kind {
            return j;
        }
    }
}

#[tokio::test]
async fn a_join_payload_is_routed_to_the_ranks_process_and_its_answer_back_to_the_clients() {
    // decision 0024: after node-b's group moves to node-c's spares, the executor's client on
    // node-a joins rank 4, whose position node-c's former spare now holds
    let cl = cluster("policy-two-node.json").await;
    let (mut on_a, _on_b, mut spares, (sock_a, _, _)) = two_node_engine(&cl).await;
    let mut client = common::register(&sock_a, 0, 0, Role::Client).await;
    let change = {
        let c = Arc::clone(&cl.coord);
        tokio::spawn(async move { c.rank_lost("k3-a", 5, Cause::Fault).await })
    };
    for (rank, w) in (0u32..).zip(on_a.iter_mut()) {
        expect_change(w, rank, 1, 1).await;
    }
    for (rank, sp) in (4u32..).zip(spares.iter_mut()) {
        expect_change(sp, rank, 1, 1).await;
    }
    change.await.expect("join").expect("the group moves");
    let join = Join {
        engine: "k3-a".into(),
        epoch: 1,
        rank: 4,
        payload: b"ring handle".to_vec(),
    };
    frame::send(
        &mut client,
        &Signal {
            kind: Some(signal::Kind::Join(join.clone())),
        },
    )
    .await
    .expect("join");
    assert_eq!(
        expect_join(&mut spares[0]).await,
        join,
        "rank 4's process hears the join"
    );
    let answer = Join {
        engine: "k3-a".into(),
        epoch: 1,
        rank: 4,
        payload: b"response handle".to_vec(),
    };
    frame::send(
        &mut spares[0],
        &Signal {
            kind: Some(signal::Kind::Join(answer.clone())),
        },
    )
    .await
    .expect("answer");
    assert_eq!(
        expect_join(&mut client).await,
        answer,
        "the client hears the answer"
    );
    // readiness's record of the change carries both payloads through the coordinator
    let c = cl
        .coord
        .readiness()
        .await
        .engines
        .remove(0)
        .last_change
        .expect("the committed change's record");
    assert_eq!((c.epoch, c.outcome.as_str()), (1, "replace"));
    assert!(
        c.reported_at_us <= c.delivered_at_us
            && c.delivered_at_us <= c.acknowledged_at_us
            && c.acknowledged_at_us <= c.committed_at_us,
        "{c:?}"
    );
    assert!(!c.acks.is_empty(), "{c:?}");
    let sizes: Vec<(bool, u64)> = c.joins.iter().map(|j| (j.from_client, j.bytes)).collect();
    assert_eq!(
        sizes,
        vec![(true, 11), (false, 15)],
        "the join, then its answer"
    );
    for j in &c.joins {
        assert!(c.committed_at_us <= j.received_at_us, "{j:?}");
        assert!(!j.forwarded_at_us.is_empty(), "{j:?}");
        assert!(
            j.forwarded_at_us.values().all(|&at| j.received_at_us <= at),
            "{j:?}"
        );
    }
    assert!(
        c.joins[0].forwarded_at_us.contains_key("node-c"),
        "the join went to the node holding rank 4: {c:?}"
    );
    cl.cancel.cancel();
}

/// The next `ReplayAsk` `w` hears within `ms`, if any.
async fn replay_ask_within(
    w: &mut UnixStream,
    ms: u64,
) -> Option<everserve_controller::proto::everserve::signals::v1::ReplayAsk> {
    tokio::time::timeout(std::time::Duration::from_millis(ms), async {
        loop {
            let sig: Signal = common::recv(w).await.expect("frame");
            if let Some(signal::Kind::ReplayAsk(a)) = sig.kind {
                return a;
            }
        }
    })
    .await
    .ok()
}

/// The next `ReplayOffer` `w` hears within `ms`, if any.
async fn replay_offer_within(
    w: &mut UnixStream,
    ms: u64,
) -> Option<everserve_controller::proto::everserve::signals::v1::ReplayOffer> {
    tokio::time::timeout(std::time::Duration::from_millis(ms), async {
        loop {
            let sig: Signal = common::recv(w).await.expect("frame");
            if let Some(signal::Kind::ReplayOffer(o)) = sig.kind {
                return o;
            }
        }
    })
    .await
    .ok()
}

/// A connection to the daemon at `sock` whose first frame asks for `rank`'s replay.
async fn ask_alone(sock: &Path, rank: u32) -> UnixStream {
    let mut asking = UnixStream::connect(sock).await.expect("connect");
    frame::send(
        &mut asking,
        &Signal {
            kind: Some(signal::Kind::ReplayAsk(
                everserve_controller::proto::everserve::signals::v1::ReplayAsk {
                    engine: "k3-a".into(),
                    rank,
                    asker: None,
                },
            )),
        },
    )
    .await
    .expect("ask");
    asking
}

/// An offer of `rank`'s replay queue, addressed to `to`.
fn offer_to(
    rank: u32,
    to: everserve_controller::proto::everserve::signals::v1::ProcessRef,
) -> everserve_controller::proto::everserve::signals::v1::ReplayOffer {
    everserve_controller::proto::everserve::signals::v1::ReplayOffer {
        engine: "k3-a".into(),
        rank,
        handle: format!("rank {rank}'s replay queue"),
        to: Some(to),
    }
}

/// A cell with its head registered on node-b: the cell, node-b's workers, and node-c's and
/// node-b's sockets, with the head's connection.
async fn cell_with_head() -> (Cluster, Vec<UnixStream>, PathBuf, PathBuf, UnixStream) {
    let cl = cluster("policy-two-node.json").await;
    let (_on_a, on_b, _spares, (_, sock_b, sock_c)) = two_node_engine(&cl).await;
    let head = common::register_head(&sock_b, std::process::id()).await;
    within("the head", async {
        while cl
            .coord
            .engine("k3-a")
            .await
            .unwrap_or_default()
            .head
            .is_none()
        {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;
    (cl, on_b, sock_c, sock_b, head)
}

#[tokio::test]
async fn a_spares_replay_ask_reaches_the_head_and_its_offer_that_asker_alone() {
    // decision 0038: a spare asks the core for what it sent its rank at its start; the ask
    // reaches the head with the asker named by its daemon, and the head's offer reaches that one
    // asking connection, never another, and never from a process that is not the head
    let (cl, mut on_b, sock_c, _sock_b, mut head) = cell_with_head().await;
    let mut first = ask_alone(&sock_c, 4).await;
    let _ = replay_ask_within(&mut head, 3000)
        .await
        .expect("the head hears the first ask");
    let mut second = ask_alone(&sock_c, 5).await;
    let at_head = replay_ask_within(&mut head, 3000)
        .await
        .expect("the head hears the second ask");
    let asker = at_head.asker.clone().expect("the ask names its asker");
    assert_eq!((at_head.rank, asker.node.as_str()), (5, "node-c"));
    let offer = |to| Signal {
        kind: Some(signal::Kind::ReplayOffer(offer_to(5, to))),
    };
    // a worker on the head's own node is not the head: its daemon drops its offer
    frame::send(&mut on_b[0], &offer(asker.clone()))
        .await
        .expect("a worker's offer");
    assert!(
        replay_offer_within(&mut second, 500).await.is_none(),
        "an offer from a worker reached the asker"
    );
    frame::send(&mut head, &offer(asker))
        .await
        .expect("the head's offer");
    let got = replay_offer_within(&mut second, 3000)
        .await
        .expect("the asker hears the head's offer");
    assert_eq!(got.handle, "rank 5's replay queue");
    assert!(
        replay_offer_within(&mut first, 500).await.is_none(),
        "another asking connection heard an offer meant for one"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_replay_offer_from_a_node_without_the_head_is_refused_by_the_coordinator() {
    // a daemon that is not the head's passes an offer up (as a daemon run by someone else
    // might): the coordinator takes offers from the head's node alone
    use everserve_controller::proto::everserve::coord::v1::{
        DaemonHello, DaemonToCoord, EngineState, daemon_to_coord,
    };
    let (cl, _on_b, sock_c, _sock_b, mut head) = cell_with_head().await;
    let mut asking = ask_alone(&sock_c, 4).await;
    let asker = replay_ask_within(&mut head, 3000)
        .await
        .and_then(|a| a.asker)
        .expect("the asker");
    let mut rogue = everserve_wire::transport::connect(&cl.address, None, "coordinator")
        .await
        .expect("connect");
    let hello = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::Hello(DaemonHello {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            node: "rogue".into(),
            engines: vec![EngineState {
                engine: "k3-a".into(),
                ..EngineState::default()
            }],
            ..DaemonHello::default()
        })),
    };
    frame::send(&mut rogue, &hello).await.expect("hello");
    let offer = DaemonToCoord {
        kind: Some(daemon_to_coord::Kind::ReplayOffer(offer_to(4, asker))),
    };
    frame::send(&mut rogue, &offer).await.expect("offer");
    assert!(
        replay_offer_within(&mut asking, 800).await.is_none(),
        "an offer from a node without the head reached the asker"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_spare_that_has_not_registered_asks_on_a_connection_of_its_own_and_hears_its_offer_there()
{
    // a spare's build needs its replay before it can register: it asks on a connection whose first
    // frame is the ask, registers nothing, and hears the head's offer on that same connection
    let (cl, _on_b, sock_c, _sock_b, mut head) = cell_with_head().await;
    let mut asking = ask_alone(&sock_c, 6).await;
    let asker = replay_ask_within(&mut head, 3000)
        .await
        .and_then(|a| a.asker)
        .expect("the head hears the ask, its asker named");
    assert_eq!(asker.node, "node-c");
    let offer = Signal {
        kind: Some(signal::Kind::ReplayOffer(offer_to(6, asker))),
    };
    frame::send(&mut head, &offer).await.expect("offer");
    let got = replay_offer_within(&mut asking, 3000)
        .await
        .expect("the asking connection hears its offer");
    assert_eq!(got.handle, "rank 6's replay queue");
    cl.cancel.cancel();
}
