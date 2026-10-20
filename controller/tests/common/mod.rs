// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The tests' shared harness: a coordinator on a Unix socket, daemons linked to it, and the
//! processes that register with them, each reporting what it runs (decision 0034: every process
//! of this protocol reports an identity, and a spare that reports none is refused). Each test
//! file keeps its own `Cluster`, since what a test needs around the coordinator differs; the
//! parts are built here once.
// each test crate compiles this module and uses a part of it
#![allow(dead_code)]

pub mod api;
pub mod cluster;
pub mod etcd;
pub mod kube;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use everserve_controller::coord::Coordinator;
use everserve_controller::node::Daemon;
use everserve_controller::proto::everserve::coord::v1::Device as WireDevice;
use everserve_controller::proto::everserve::policy::v1::Policy;
use everserve_controller::proto::everserve::signals::v1::{
    Ack, CoverPlan, KvFormat, Member, ParallelLayout, Reason, Register, Role, Signal,
    SoftwareIdentity, signal,
};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::{Address, Listener, Tls};
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

/// The engine every test registers with.
pub const ENGINE: &str = "k3-a";

/// The example document `name` in `docs/examples`.
pub fn example(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/examples")
        .join(name)
}

/// `n` devices of `class` on `node`, 80 GiB each.
pub fn devices(node: &str, n: u32, class: &str) -> Vec<WireDevice> {
    (0..n)
        .map(|index| WireDevice {
            index,
            uuid: format!("GPU-{node}-{index}"),
            class: class.into(),
            memory_bytes: 80 << 30,
            domain: String::new(),
        })
        .collect()
}

/// A coordinator under `policy`, serving at `coord.sock` in `dir` until `cancel` fires.
pub async fn coordinator(
    policy: Policy,
    dir: &Path,
    telemetry: &Telemetry,
    cancel: &CancellationToken,
) -> (Arc<Coordinator>, Address) {
    let (coord, _rx) = Coordinator::new(policy, telemetry);
    let address = serve(&coord, dir, cancel).await;
    (coord, address)
}

/// Serve `coord` at `coord.sock` in `dir` until `cancel` fires; the address daemons link to.
pub async fn serve(coord: &Arc<Coordinator>, dir: &Path, cancel: &CancellationToken) -> Address {
    let address = Address::Unix(dir.join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    tokio::spawn(Arc::clone(coord).serve(listener, cancel.clone()));
    address
}

/// `node`'s daemon with `devices`, serving its socket in `dir` and linked to the coordinator at
/// `address` until `cancel` fires; its socket's path.
pub async fn daemon(
    dir: &Path,
    node: &str,
    address: &Address,
    devices: Vec<WireDevice>,
    cancel: &CancellationToken,
) -> (Arc<Daemon>, PathBuf) {
    let d = new_daemon(node, devices).await;
    let sock = start(&d, dir, address, cancel).await;
    (d, sock)
}

/// `node`'s daemon with `devices`, not yet serving: for a test that sets more before [`start`].
pub async fn new_daemon(node: &str, devices: Vec<WireDevice>) -> Arc<Daemon> {
    let (d, _rx) = Daemon::new(node, &Telemetry::new());
    if !devices.is_empty() {
        d.set_devices(devices).await;
    }
    d
}

/// Serve `d`'s socket in `dir` and link it to the coordinator at `address` until `cancel` fires;
/// its socket's path. Fails the test if it does not link in 3 s.
pub async fn start(
    d: &Arc<Daemon>,
    dir: &Path,
    address: &Address,
    cancel: &CancellationToken,
) -> PathBuf {
    let node = d.node().to_owned();
    let sock = dir.join(format!("{node}.sock"));
    tokio::spawn(Arc::clone(d).serve(sock.clone(), cancel.clone()));
    tokio::spawn(Arc::clone(d).link(address.clone(), None, "coordinator".into(), cancel.clone()));
    for _ in 0..300 {
        if sock.exists() && d.linked().await {
            return sock;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("daemon {node} did not link");
}

/// What a process of `rank` on `device` registers as `role`, with the shared identity.
pub fn registration(rank: u32, device: u32, role: Role) -> Register {
    Register {
        protocol: everserve_wire::PROTOCOL_VERSION,
        protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
        engine: ENGINE.into(),
        rank,
        member: Some(Member {
            node: "n".into(),
            device,
        }),
        role: role as i32,
        identity: Some(identity()),
        ..Register::default()
    }
}

/// Connect to the daemon at `path` and send `reg`.
pub async fn register_as(path: &Path, reg: Register) -> UnixStream {
    let mut s = UnixStream::connect(path).await.expect("connect");
    let sig = Signal {
        kind: Some(signal::Kind::Register(reg)),
    };
    frame::send(&mut s, &sig).await.expect("register");
    s
}

/// Register `rank` as `role` on `device` with the daemon at `path`.
pub async fn register(path: &Path, rank: u32, device: u32, role: Role) -> UnixStream {
    register_as(path, registration(rank, device, role)).await
}

/// Register the engine's head (decision 0032), its launcher `launcher_pid`, driving its engine's
/// steps through Everserve's hook, as an adapter's head says.
pub async fn register_head(path: &Path, launcher_pid: u32) -> UnixStream {
    register_as(
        path,
        Register {
            role: Role::Head as i32,
            launcher_pid,
            identity: Some(SoftwareIdentity {
                serves_changes: true,
                ..identity()
            }),
            ..registration(0, 0, Role::Head)
        },
    )
    .await
}

/// Wait until the coordinator holds `n` members of the engine; fails the test after 3 s.
pub async fn wait_members(c: &Coordinator, n: usize) {
    for _ in 0..300 {
        if c.engine(ENGINE).await.is_some_and(|v| v.members.len() == n) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("never saw {n} members: {:?}", c.engine(ENGINE).await);
}

/// Whether process `pid` is alive.
pub fn alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .is_ok_and(|s| s.success())
}

/// The TLS material `everserve-controller certs` wrote for `name` in `dir`.
pub fn tls(dir: &Path, name: &str) -> Tls {
    Tls::load(
        &dir.join(format!("{name}.crt")),
        &dir.join(format!("{name}.key")),
        &dir.join("ca.crt"),
    )
    .expect("load")
}

/// One identity every test process shares, so every spare matches every position.
pub fn identity() -> SoftwareIdentity {
    SoftwareIdentity {
        engine: "vllm".into(),
        engine_commit: "6e517b15".into(),
        adapter_commit: "test".into(),
        cuda_version: "12.8".into(),
        nccl_version: "2.27.3".into(),
        checkpoint_digest: "test-checkpoint".into(),
        model_dtype: "bfloat16".into(),
        kv_format: Some(KvFormat {
            dtype: "bfloat16".into(),
            block_size: 16,
            layout_hash: "test-layout".into(),
        }),
        layout: Some(ParallelLayout {
            tensor: 1,
            pipeline: 1,
            data: 1,
            expert: 1,
            context: 1,
        }),
        step: Some(step(&[("max_num_seqs", "256"), ("block_size", "16")])),
        guarantee: "exact".into(),
        ..SoftwareIdentity::default()
    }
}

/// A step config of `fields`, its digest standing for theirs as an adapter's would.
pub fn step(
    fields: &[(&str, &str)],
) -> everserve_controller::proto::everserve::signals::v1::StepConfig {
    let fields: std::collections::BTreeMap<String, String> = fields
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    everserve_controller::proto::everserve::signals::v1::StepConfig {
        digest: format!("{fields:?}"),
        fields,
    }
}

/// Read `n` notices of `reason` then one change for `rank` at `epoch`, acknowledging the change;
/// the ranks the notices named.
pub async fn expect_notices_then_change(
    w: &mut UnixStream,
    rank: u32,
    n: usize,
    epoch: u64,
    reason: Reason,
) -> Vec<u32> {
    let mut notices = Vec::new();
    loop {
        let sig: Signal = recv(w).await.expect("signal");
        match sig.kind {
            Some(signal::Kind::Notice(notice)) => {
                assert_eq!(notice.reason, reason as i32, "rank {rank}");
                notices.push(notice.rank);
            }
            Some(signal::Kind::Change(c)) => {
                assert_eq!(notices.len(), n, "rank {rank}: {notices:?}");
                assert_eq!(c.epoch, epoch, "rank {rank}");
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
                return c.lost;
            }
            other => panic!("rank {rank}: unexpected {other:?}"),
        }
    }
}

/// The next cover plan `process` is sent (decision 0035), skipping anything else; fails the test
/// after 5 s.
pub async fn next_plan(process: &mut UnixStream) -> CoverPlan {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let sig: Signal = frame::recv(process).await.expect("signal");
            if let Some(signal::Kind::CoverPlan(p)) = sig.kind {
                return p;
            }
        }
    })
    .await
    .expect("a plan within 5 s")
}

/// Bound every test: a lease test that hangs is a failure, not a slow pass.
pub async fn bounded(f: std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>) {
    tokio::time::timeout(Duration::from_secs(30), f)
        .await
        .expect("the test finished in 30 s");
}

/// A child process killed when dropped, so a test that fails leaves nothing running.
pub struct Spawned(pub std::process::Child);

impl Drop for Spawned {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// `child`'s exit code once it exits.
pub async fn exited(Spawned(child): &mut Spawned) -> Option<i32> {
    loop {
        if let Some(status) = child.try_wait().expect("wait") {
            return status.code();
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// A TCP relay to a test's server that can go silent: once cut, it drops every byte both ways and keeps the
/// connections open, as a partition does, so requests hang rather than fail.
pub struct Relay {
    /// Where clients connect to reach the server through it.
    pub endpoint: String,
    cut: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl Relay {
    /// A relay to `upstream` (`host:port`).
    pub async fn start(upstream: &str) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let endpoint = listener.local_addr().expect("addr").to_string();
        let cut = Arc::new(AtomicBool::new(false));
        let (upstream, flag) = (upstream.to_owned(), Arc::clone(&cut));
        let task = tokio::spawn(async move {
            while let Ok((down, _)) = listener.accept().await {
                let Ok(up) = tokio::net::TcpStream::connect(&upstream).await else {
                    continue;
                };
                let (down_r, down_w) = down.into_split();
                let (up_r, up_w) = up.into_split();
                tokio::spawn(pump(down_r, up_w, Arc::clone(&flag)));
                tokio::spawn(pump(up_r, down_w, Arc::clone(&flag)));
            }
        });
        Self {
            endpoint,
            cut,
            task,
        }
    }

    /// Go silent, both ways, from now on.
    pub fn cut(&self) {
        self.cut.store(true, Ordering::SeqCst);
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn pump(
    mut from: tokio::net::tcp::OwnedReadHalf,
    mut to: tokio::net::tcp::OwnedWriteHalf,
    cut: Arc<AtomicBool>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = vec![0; 16 * 1024];
    loop {
        let n = match from.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        if cut.load(Ordering::SeqCst) {
            continue;
        }
        if to.write_all(&buf[..n]).await.is_err() {
            return;
        }
    }
}

/// A process's next signal on `s`, past the engine policies, placements and instance states
/// its daemon hands every process (decision 0041), which a test that is not about them reads
/// past.
pub async fn recv(s: &mut UnixStream) -> Result<Signal, everserve_wire::frame::WireError> {
    loop {
        let sig: Signal = frame::recv(s).await?;
        if !matches!(
            sig.kind,
            Some(
                signal::Kind::EnginePolicy(_)
                    | signal::Kind::EnginePlacement(_)
                    | signal::Kind::InstanceState(_)
            )
        ) {
            return Ok(sig);
        }
    }
}
