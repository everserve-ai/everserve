// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Protocol checks (decision 0027): a hello from another protocol is refused, a registration
//! from a newer build that serves this one is taken, and one from another protocol is refused.

mod common;

use everserve_controller::proto::everserve::signals::v1::{Member, Register, Role, Signal, signal};
use everserve_wire::frame;
use tokio::net::UnixStream;

use common::cluster::cluster;

#[tokio::test]
async fn a_hello_from_another_protocol_is_refused() {
    // decision 0027: a daemon built against another protocol gets no link, and the coordinator
    // says which protocols met
    let cl = cluster("policy-rig-4gpu.json").await;
    let mut stream = everserve_wire::transport::connect(&cl.address, None, "coordinator")
        .await
        .expect("connect");
    let hello = everserve_wire::proto::everserve::coord::v1::DaemonToCoord {
        kind: Some(
            everserve_wire::proto::everserve::coord::v1::daemon_to_coord::Kind::Hello(
                everserve_wire::proto::everserve::coord::v1::DaemonHello {
                    node: "node-x".into(),
                    protocol: everserve_wire::PROTOCOL_VERSION + 1,
                    protocol_oldest: everserve_wire::PROTOCOL_VERSION + 1,
                    ..Default::default()
                },
            ),
        ),
    };
    frame::send(&mut stream, &hello).await.expect("send");
    let closed: Result<everserve_wire::proto::everserve::coord::v1::CoordToDaemon, _> =
        frame::recv(&mut stream).await;
    assert!(
        closed.is_err(),
        "the coordinator kept a link from another protocol"
    );
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(
        cl.coord_telemetry
            .render()
            .contains("everserve_daemons_connected 0"),
        "{}",
        cl.coord_telemetry.render()
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_registration_from_a_newer_build_that_serves_this_one_is_taken() {
    // decision 0040: a rolling upgrade puts a newer engine process beside an older daemon; it
    // serves the daemon's version, so the link opens, at the older of the two
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let _w = common::register_as(
        &sock_a,
        Register {
            protocol: everserve_wire::PROTOCOL_VERSION + 1,
            protocol_oldest: everserve_wire::PROTOCOL_VERSION,
            ..common::registration(0, 0, Role::Worker)
        },
    )
    .await;
    common::wait_members(&cl.coord, 1).await;
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_registration_from_another_protocol_is_refused() {
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let mut w = UnixStream::connect(&sock_a).await.expect("connect");
    let reg = Signal {
        kind: Some(signal::Kind::Register(Register {
            engine: "k3-a".into(),
            rank: 0,
            member: Some(Member {
                node: "node-a".into(),
                device: 0,
            }),
            role: Role::Worker as i32,
            slot: None,
            protocol: everserve_wire::PROTOCOL_VERSION + 1,
            protocol_oldest: everserve_wire::PROTOCOL_VERSION + 1,
            compile_cache: None,
            any_rank: false,
            launcher_pid: 0,
            identity: Some(common::identity()),
            stage: Vec::new(),
            host: String::new(),
        })),
    };
    frame::send(&mut w, &reg).await.expect("send");
    let closed: Result<Signal, _> = frame::recv(&mut w).await;
    assert!(
        closed.is_err(),
        "the daemon kept a worker from another protocol"
    );
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(
        cl.coord
            .engine("k3-a")
            .await
            .is_none_or(|v| v.members.is_empty())
    );
    cl.cancel.cancel();
}
