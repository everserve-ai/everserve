// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A cluster of one coordinator and node daemons on one machine, for the membership tests: the
//! coordinator under a policy file, the engines its tests form, the waits they share, and fakes
//! of a daemon and of a node's store.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use everserve_controller::coord::{Coordinator, EngineView};
use everserve_controller::node::Daemon;
use everserve_controller::policy;
use everserve_controller::proto::everserve::signals::v1::{
    Ack, Change, CompileCache, Notice, Register, Role, Signal, signal,
};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::Address;
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

/// `register_on`, reporting `compile_cache` (decision 0029).
pub async fn register_with(
    path: &Path,
    rank: u32,
    device: u32,
    role: Role,
    compile_cache: Option<CompileCache>,
) -> UnixStream {
    super::register_as(
        path,
        Register {
            compile_cache,
            ..super::registration(rank, device, role)
        },
    )
    .await
}

/// Wait until the coordinator's view of `k3-a` satisfies `pred`.
pub async fn wait_view(
    c: &Coordinator,
    what: &str,
    pred: impl Fn(&EngineView) -> bool,
) -> EngineView {
    for _ in 0..500 {
        if let Some(v) = c.engine("k3-a").await
            && pred(&v)
        {
            return v;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("never saw {what}: {:?}", c.engine("k3-a").await)
}

pub fn cache(digest: &str) -> CompileCache {
    cache_under("760834052d", digest)
}

pub fn cache_under(key: &str, digest: &str) -> CompileCache {
    CompileCache {
        key: key.into(),
        digest: digest.into(),
    }
}

pub async fn engine_readiness(cl: &Cluster) -> everserve_controller::readiness::EngineReadiness {
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let r = cl.coord.readiness().await;
    r.engines
        .iter()
        .find(|e| e.engine == "k3-a")
        .cloned()
        .expect("engine")
}

/// Read a `Change` for `rank` at `epoch` and acknowledge it with `ack_epoch`.
pub async fn expect_change(w: &mut UnixStream, rank: u32, epoch: u64, ack_epoch: u64) -> Change {
    let sig: Signal = super::recv(w).await.expect("change");
    let Some(signal::Kind::Change(c)) = sig.kind else {
        panic!("rank {rank}: expected a change, got {sig:?}")
    };
    assert_eq!(c.epoch, epoch, "rank {rank}");
    frame::send(
        w,
        &Signal {
            kind: Some(signal::Kind::Ack(Ack {
                rank,
                epoch: ack_epoch,
                error: String::new(),
                received_at_us: 0,
            })),
        },
    )
    .await
    .expect("ack");
    c
}

pub struct Cluster {
    pub coord: Arc<Coordinator>,
    pub coord_telemetry: Telemetry,
    pub cancel: CancellationToken,
    pub address: Address,
    pub dir: tempfile::TempDir,
}

pub async fn cluster(policy_file: &str) -> Cluster {
    cluster_of(policy::load(&super::example(policy_file)).expect("policy")).await
}

pub async fn cluster_of(
    policy: everserve_controller::proto::everserve::policy::v1::Policy,
) -> Cluster {
    let dir = tempfile::tempdir().expect("tmp");
    let coord_telemetry = Telemetry::new();
    let cancel = CancellationToken::new();
    let (coord, address) = super::coordinator(policy, dir.path(), &coord_telemetry, &cancel).await;
    Cluster {
        coord,
        coord_telemetry,
        cancel,
        address,
        dir,
    }
}

/// A daemon on a node whose devices are `n` of `class`, with a store at `store`, linked.
pub async fn node_with_devices(
    cl: &Cluster,
    node: &str,
    n: u32,
    class: &str,
    store: Option<(PathBuf, &str)>,
) -> (Arc<Daemon>, PathBuf) {
    let d = super::new_daemon(node, super::devices(node, n, class)).await;
    if let Some((socket, replica)) = store {
        d.set_mirror(socket, Some(replica.to_owned())).await;
    }
    let sock = super::start(&d, cl.dir.path(), &cl.address, &cl.cancel).await;
    (d, sock)
}

/// The two-node engine: ranks 0-3 on node-a, 4-7 on node-b, spares for 4-7 on node-c. No daemon
/// reports devices, so each node is its own domain, as the topology falls back to. Returns the
/// connections in rank order and the daemons' sockets.
#[allow(clippy::type_complexity)]
pub async fn two_node_engine(
    cl: &Cluster,
) -> (
    Vec<UnixStream>,
    Vec<UnixStream>,
    Vec<UnixStream>,
    (PathBuf, PathBuf, PathBuf),
) {
    let mut socks = Vec::new();
    for node in ["node-a", "node-b", "node-c"] {
        let (_d, sock) =
            super::daemon(cl.dir.path(), node, &cl.address, Vec::new(), &cl.cancel).await;
        socks.push(sock);
    }
    let (sock_a, sock_b, sock_c) = (socks[0].clone(), socks[1].clone(), socks[2].clone());
    let mut on_a = Vec::new();
    let mut on_b = Vec::new();
    for r in 0..4 {
        on_a.push(super::register(&sock_a, r, r, Role::Worker).await);
        on_b.push(super::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    super::wait_members(&cl.coord, 8).await;
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(super::register(&sock_c, 4 + r, r, Role::Spare).await);
    }
    for _ in 0..300 {
        if cl.coord.pool("k3-a").await.len() == 4 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(cl.coord.pool("k3-a").await.len(), 4);
    (on_a, on_b, spares, (sock_a, sock_b, sock_c))
}

/// A bare link to the coordinator posing as `node`'s daemon: a hello with `workers` of `k3-a`
/// and a heartbeat promise, then whatever the test does with the stream.
pub async fn fake_daemon(
    cl: &Cluster,
    node: &str,
    workers: Vec<u32>,
    heartbeat_ms: u32,
) -> everserve_wire::transport::Stream {
    fake_daemon_with_notices(cl, node, workers, heartbeat_ms, Vec::new()).await
}

/// [`fake_daemon`], its hello carrying `notices` the node's processes reported.
pub async fn fake_daemon_with_notices(
    cl: &Cluster,
    node: &str,
    workers: Vec<u32>,
    heartbeat_ms: u32,
    notices: Vec<Notice>,
) -> everserve_wire::transport::Stream {
    use everserve_controller::proto::everserve::coord::v1::{DaemonHello, EngineState};
    let mut s = everserve_wire::transport::connect(&cl.address, None, "coordinator")
        .await
        .expect("connect");
    let hello = everserve_controller::proto::everserve::coord::v1::DaemonToCoord {
        kind: Some(
            everserve_controller::proto::everserve::coord::v1::daemon_to_coord::Kind::Hello(
                DaemonHello {
                    protocol: everserve_wire::PROTOCOL_VERSION,
                    protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                    node: node.into(),
                    engines: vec![EngineState {
                        engine: "k3-a".into(),
                        epoch: 0,
                        workers,
                        spares: vec![],
                        any_spares: vec![],
                        world_size: 4,
                        lost: vec![],
                        departed: vec![],
                        slot: None,
                        notices,
                        ..EngineState::default()
                    }],
                    devices: vec![],
                    mirror_replica_address: String::new(),
                    heartbeat_ms,
                    ..DaemonHello::default()
                },
            ),
        ),
    };
    frame::send(&mut s, &hello).await.expect("hello");
    for _ in 0..300 {
        if cl.coord.daemons().await.iter().any(|n| n == node) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    s
}

/// The next change on `w`, skipping every other signal (a commit, a notice).
pub async fn next_change(w: &mut UnixStream) -> Change {
    loop {
        let sig: Signal = super::recv(w).await.expect("frame");
        if let Some(signal::Kind::Change(c)) = sig.kind {
            return c;
        }
    }
}

/// Acknowledge `epoch` for `rank` on `w`.
pub async fn ack(w: &mut UnixStream, rank: u32, epoch: u64) {
    frame::send(
        w,
        &Signal {
            kind: Some(signal::Kind::Ack(Ack {
                rank,
                epoch,
                error: String::new(),
                received_at_us: 0,
            })),
        },
    )
    .await
    .expect("ack");
}

/// `f`, failing the test as `what` if it takes 5 s.
pub async fn within<T>(what: &str, f: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(std::time::Duration::from_secs(5), f)
        .await
        .unwrap_or_else(|_| panic!("{what}: not within 5 s"))
}
