// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The engine's head (decision 0032): an engine is down from its head's goodbye until a head
//! registers again, and a second head on another node is refused and never takes it down.

mod common;

use everserve_controller::coord::{Cause, ChangeError};
use everserve_controller::proto::everserve::signals::v1::{
    Goodbye, Role, Signal, StopCause, signal,
};
use everserve_wire::frame;

use common::cluster::{cluster, wait_view};

#[tokio::test]
async fn an_engine_is_down_from_its_heads_goodbye_until_a_head_registers_again() {
    // decision 0032: a head that stops on a failure says so, and no change runs for the engine
    let cl = cluster("policy-pilot.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock_a, 1, 1, Role::Worker).await;
    let _spare = common::register(&sock_a, 1, 2, Role::Spare).await;
    let mut head = common::register_head(&sock_a, 4242).await;
    let v = wait_view(&cl.coord, "the head", |v| v.head.is_some()).await;
    let h = v.head.expect("head");
    assert_eq!((h.process.node.as_str(), h.launcher_pid), ("node-a", 4242));

    frame::send(
        &mut head,
        &Signal {
            kind: Some(signal::Kind::Goodbye(Goodbye {
                cause: StopCause::Failed as i32,
                detail: "pipeline refused".into(),
            })),
        },
    )
    .await
    .expect("goodbye");
    drop(head);
    let v = wait_view(&cl.coord, "the engine down", |v| v.down.is_some()).await;
    let d = v.down.expect("down");
    assert_eq!((d.cause, d.detail.as_str()), ("failed", "pipeline refused"));
    let err = cl
        .coord
        .rank_lost("k3-a", 1, Cause::Fault)
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            ChangeError::Down {
                cause: "failed",
                ..
            }
        ),
        "{err}"
    );
    let text = cl.coord_telemetry.render();
    assert!(
        text.contains("everserve_engine_down{engine=\"k3-a\"} 1"),
        "{text}"
    );
    assert!(text.contains("result=\"engine_down\"} 1"), "{text}");

    let _head = common::register_head(&sock_a, 4242).await; // the engine relaunched
    wait_view(&cl.coord, "the engine back", |v| {
        v.down.is_none() && v.head.is_some()
    })
    .await;
    let text = cl.coord_telemetry.render();
    assert!(
        text.contains("everserve_engine_down{engine=\"k3-a\"} 0"),
        "{text}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_second_head_on_another_node_is_refused_and_never_takes_the_engine_down() {
    // a node's own process registers as a head too: a loss of that node must not read as the
    // engine down, whatever order the nodes started in
    let cl = cluster("policy-pilot.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let (_db, sock_b) =
        common::daemon(cl.dir.path(), "node-b", &cl.address, Vec::new(), &cl.cancel).await;
    let head_a = common::register_head(&sock_a, 4242).await;
    wait_view(&cl.coord, "the head", |v| v.head.is_some()).await;
    let head_b = common::register_head(&sock_b, 4242).await;
    let v = wait_view(&cl.coord, "the conflict", |v| v.head_conflict.is_some()).await;
    assert_eq!(v.head.expect("head").process.node, "node-a");
    assert_eq!(v.head_conflict.expect("conflict").process.node, "node-b");
    let text = cl.coord_telemetry.render();
    assert!(
        text.contains("everserve_head_conflicts_total{engine=\"k3-a\"} 1"),
        "{text}"
    );
    let e = cl.coord.readiness().await.engines.remove(0);
    assert_eq!(e.head_conflict.map(|h| h.node).as_deref(), Some("node-b"));

    drop(head_b); // the refused head goes: the engine is not down
    let v = wait_view(&cl.coord, "the conflict gone", |v| {
        v.head_conflict.is_none()
    })
    .await;
    assert!(v.down.is_none(), "{v:?}");
    assert_eq!(v.head.expect("head").process.node, "node-a");

    drop(head_a); // the engine's own head goes: now it is
    let v = wait_view(&cl.coord, "the engine down", |v| v.down.is_some()).await;
    assert_eq!(v.down.expect("down").cause, "lost");
    cl.cancel.cancel();
}
