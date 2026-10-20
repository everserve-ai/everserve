// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What a spare runs (decision 0034): one that runs something else or says nothing is kept out
//! of the pool, one admitted before its position registered leaves when the position differs,
//! and a refused spare joins once its position runs what it runs.

mod common;

use std::path::Path;

use everserve_controller::coord::EngineView;
use everserve_controller::proto::everserve::signals::v1::{Member, Register, Role, Signal, signal};
use everserve_wire::frame;
use tokio::net::UnixStream;

use common::cluster::{cluster, wait_view};

/// Register `rank` as a spare on `device` of the daemon at `path`, reporting `identity`.
async fn register_spare_running(
    path: &Path,
    rank: u32,
    device: u32,
    identity: Option<everserve_controller::proto::everserve::signals::v1::SoftwareIdentity>,
) -> UnixStream {
    register_running(path, rank, device, Role::Spare, identity).await
}

/// Register `rank` as `role` on the daemon at `path`, on its device `device`, running `identity`.
async fn register_running(
    path: &Path,
    rank: u32,
    device: u32,
    role: Role,
    identity: Option<everserve_controller::proto::everserve::signals::v1::SoftwareIdentity>,
) -> UnixStream {
    let mut s = UnixStream::connect(path).await.expect("connect");
    let reg = Signal {
        kind: Some(signal::Kind::Register(Register {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            engine: "k3-a".into(),
            rank,
            member: Some(Member {
                node: "n".into(),
                device,
            }),
            role: role as i32,
            identity,
            ..Register::default()
        })),
    };
    frame::send(&mut s, &reg).await.expect("register");
    s
}

#[tokio::test]
async fn a_spare_that_runs_something_else_or_says_nothing_is_kept_out_of_the_pool() {
    // decision 0034: capability before movement; a spare is held to what its position runs
    let cl = cluster("policy-pilot.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    common::wait_members(&cl.coord, 2).await;
    let refused = |reason: &'static str| {
        move |v: &EngineView| {
            v.refused_spares
                .get(&("node-a".to_owned(), 2))
                .is_some_and(|(rank, why)| *rank == Some(1) && why.contains(reason))
        }
    };

    let other_checkpoint = everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
        checkpoint_digest: "another-checkpoint".into(),
        ..common::identity()
    };
    let _s = register_spare_running(&sock_a, 1, 2, Some(other_checkpoint)).await;
    let v = wait_view(&cl.coord, "the refusal", refused("checkpoint_digest")).await;
    assert!(!v.spares.contains_key(&1), "{v:?}");
    let e = cl.coord.readiness().await.engines.remove(0);
    assert!(
        e.refused_spares
            .iter()
            .any(|r| r.device == 2 && r.reason.contains("checkpoint_digest")),
        "{e:?}"
    );

    let _s = register_spare_running(&sock_a, 1, 2, None).await;
    wait_view(
        &cl.coord,
        "a refusal for no identity",
        refused("no software identity"),
    )
    .await;

    let _s = register_spare_running(&sock_a, 1, 2, Some(common::identity())).await;
    let v = wait_view(&cl.coord, "the spare admitted", |v| {
        v.spares.contains_key(&1)
    })
    .await;
    assert!(v.refused_spares.is_empty(), "{v:?}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_spare_admitted_before_its_position_registered_leaves_when_the_position_differs() {
    // decision 0036: admission is derived from current facts; a spare that came up first was
    // held to nothing, and is held to its position once the position says what it runs
    let cl = cluster("policy-pilot.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    common::wait_members(&cl.coord, 1).await;
    let other = everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
        checkpoint_digest: "another-checkpoint".into(),
        ..common::identity()
    };
    let _s = register_spare_running(&sock_a, 1, 2, Some(other)).await;
    wait_view(
        &cl.coord,
        "the spare admitted: rank 1 runs nothing yet",
        |v| v.spares.contains_key(&1),
    )
    .await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let v = wait_view(&cl.coord, "the spare out of the pool", |v| {
        !v.spares.contains_key(&1)
    })
    .await;
    let (rank, why) = v
        .refused_spares
        .get(&("node-a".to_owned(), 2))
        .expect("refused");
    assert_eq!(*rank, Some(1));
    assert!(why.contains("checkpoint_digest"), "{why}");
    let events = cl.coord.engine_events("k3-a").await.expect("events");
    assert!(
        events
            .iter()
            .any(|e| e.reason == "SpareRefused" && e.message.contains("left the pool")),
        "{events:?}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_refused_spare_joins_once_its_position_runs_what_it_runs() {
    // the upgrade case (decision 0033): a spare prepared on the new checkpoint is refused while
    // its position runs the old one, and joins without registering again once the position's
    // process runs the new one
    let cl = cluster("policy-pilot.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    common::wait_members(&cl.coord, 2).await;
    let upgraded = everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
        checkpoint_digest: "the-new-checkpoint".into(),
        ..common::identity()
    };
    let _s = register_spare_running(&sock_a, 1, 2, Some(upgraded.clone())).await;
    wait_view(&cl.coord, "the spare refused", |v| {
        v.refused_spares.contains_key(&("node-a".to_owned(), 2))
    })
    .await;
    // rank 1's process restarts on the new checkpoint
    drop(w1);
    wait_view(&cl.coord, "rank 1 departed", |v| {
        !v.members.contains_key(&1)
    })
    .await;
    let _w1 = register_running(&sock_a, 1, 1, Role::Worker, Some(upgraded)).await;
    let v = wait_view(&cl.coord, "the spare admitted", |v| {
        v.spares.contains_key(&1)
    })
    .await;
    assert!(v.refused_spares.is_empty(), "{v:?}");
    let events = cl.coord.engine_events("k3-a").await.expect("events");
    assert!(
        events.iter().any(|e| e.reason == "SpareAdmitted"),
        "{events:?}"
    );
    cl.cancel.cancel();
}
