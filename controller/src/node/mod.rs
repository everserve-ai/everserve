// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The per-node daemon: the coordinator's agent on a node (decision 0014).
//!
//! Workers, the spare and clients connect over a Unix socket and register. The daemon keeps those
//! links, tells the coordinator what it holds (a hello on every connection, then worker events),
//! forwards what the links report (a lost rank, a notice), delivers the coordinator's changes to
//! the ranks it names and returns their acknowledgements, and applies a committed membership
//! (promoting its spare, telling its clients). It decides nothing: the policy and the epoch live
//! in the coordinator.
//!
//! Without a coordinator the daemon serves its links and holds every report until a link comes
//! up, so a fault during a coordinator restart waits; it is never lost. A delivery for an epoch
//! that is not the next one is refused, which is the fence that keeps a partition from splitting
//! an engine. Framing comes from `everserve_wire::frame`.
//!
//! The daemon is one object across this module's files: `link` keeps the coordinator link and
//! takes its messages, `deliver` delivers changes and applies commits, `forward` passes the
//! coordinator's other word on, `cover` holds the cover plan and the groups, `head` the engine's
//! head, `stacks` a process's stacks, and `conns` serves the worker links.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::UnixListener;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::proto::everserve::coord::v1::RoleAnswer;

mod cache;
mod conns;
mod cover;
mod deliver;
pub mod devices;
mod facts;
mod forward;
mod head;
mod link;
pub mod process_signal;
mod replay;
pub mod spares;
mod stacks;
pub mod store;
mod undecided;

pub use stacks::MOST_STACKS_BYTES;

use crate::discovery::{self, FileDiscovery};
use crate::metrics::DaemonMetrics;
use crate::proto::everserve::coord::v1::{
    DaemonToCoord, HeldGroup, LaunchSpare, NodeNotice, WorkerEvent, daemon_to_coord,
};
use crate::proto::everserve::signals::v1::{
    Ack, CompileCache, CoverPlan, Goodbye, MembershipCommitted, Notice, Signal, SlotChoice,
    SoftwareIdentity, StacksDumped,
};

/// How long a survivor has to acknowledge a change before the daemon reports it missing.
pub const ACK_TIMEOUT: Duration = Duration::from_mins(2);

/// How long the daemon waits before trying the coordinator again after a failed connection, at
/// first; it doubles up to [`RECONNECT_MAX`].
pub const RECONNECT_MIN: Duration = Duration::from_millis(200);
/// The longest wait between reconnection attempts.
pub const RECONNECT_MAX: Duration = Duration::from_secs(5);

/// A wait of between half of `wait` and all of it, drawn afresh each call: daemons that lost
/// their coordinator together do not try the next one together. Without it every daemon of a
/// cell retries in step, and the cell's whole fleet meets a listener's backlog at once
/// (`bench/runs/2026-10-14-cell-s0`: 15,182 refused connects linking 2,500 nodes).
#[must_use]
pub fn jittered(wait: Duration) -> Duration {
    use std::hash::{BuildHasher, RandomState};
    // a fresh RandomState is seeded per process and per call: the spread needs no other source
    let draw = RandomState::new().hash_one(std::time::Instant::now());
    let half = wait / 2;
    let span = u64::try_from(half.as_nanos()).unwrap_or(u64::MAX);
    half + Duration::from_nanos(draw % span.max(1))
}

/// A signal to deliver to a connection and, when it expects one, where its acknowledgement goes.
/// Changes are acknowledged; notices and committed memberships are not.
type Outbound = (Signal, Option<oneshot::Sender<Ack>>);

/// A connected worker, spare or client. `id` tells this connection from a later one that takes
/// over the same rank (a spare after REPLACE), so a closing connection removes only itself.
struct Conn {
    id: u64,
    tx: mpsc::Sender<Outbound>,
    /// The process at the other end, from the socket's peer credentials; None when the system
    /// did not say.
    pid: Option<u32>,
}

/// What a registration or departure says about the process: the rank (unused for a spare standing
/// for any position), the device, the compile cache it reported, and whether it stands for any
/// position (decision 0030).
struct Presence {
    rank: u32,
    device: u32,
    compile_cache: Option<CompileCache>,
    any_rank: bool,
    pid: Option<u32>,
    /// The engine's epoch on this node when the event happened.
    epoch: u64,
    /// What the process runs, as it registered; None on a departure (decision 0034).
    identity: Option<SoftwareIdentity>,
    /// The connection a registration came on; None on a departure.
    conn: Option<u64>,
    /// The address the process registered (decision 0041); empty on a departure.
    host: String,
    /// The process's goodbye, on a departure that had one.
    goodbye: Option<Goodbye>,
}

/// The engine's head on this node (decision 0032): its connection, its launcher, and the
/// goodbye it sent, if any, before its connection ended.
struct Head {
    id: u64,
    launcher_pid: u32,
    goodbye: Option<Goodbye>,
    /// Its pid as this daemon sees it, what it runs and the address it registered: what its
    /// registration and every hello report it with.
    pid: Option<u32>,
    identity: Option<SoftwareIdentity>,
    host: String,
}

/// The daemon's view of one engine on this node.
#[derive(Default)]
struct Engine {
    workers: BTreeMap<u32, Conn>,
    /// The pre-loaded spares on this node, by the rank each stands in for.
    spares: BTreeMap<u32, Conn>,
    /// The spares on this node standing for any position (decision 0030), by device.
    any_spares: BTreeMap<u32, Conn>,
    /// Which of those is taking which rank in the change in flight (`Deliver.take`), so the
    /// join for that rank reaches it before the commit promotes it.
    taking: BTreeMap<u32, u32>,
    clients: Vec<Conn>,
    /// The engine's head, when it registered on this node; it is also one of `clients`.
    head: Option<Head>,
    /// The last committed epoch heard from the coordinator, or the last one an abandoned change
    /// spent (decision 0032).
    epoch: u64,
    lost: Vec<u32>,
    /// Workers whose connection dropped without a change: a killed worker's socket closes
    /// before anyone reports its loss, and the report must still find it.
    departed: BTreeSet<u32>,
    world_size: u32,
    /// The engine's slot-collective choice, from its client's registration.
    slot: Option<SlotChoice>,
    /// The newest cover plan the coordinator sent (decision 0035), forwarded to every worker and
    /// spare of the engine here and to each that registers after it.
    plan: Option<CoverPlan>,
    /// The groups each connection reported formed and has not released, by connection id: what
    /// the hello reports as held, and what the daemon releases for a process whose connection
    /// ends.
    held: BTreeMap<u64, Vec<HeldGroup>>,
    /// Each spare's registration as it was sent up, by connection id, until it departs: the
    /// hello carries them, so a coordinator that starts or takes the lease holds each spare to
    /// what it registered with.
    standing: BTreeMap<u64, WorkerEvent>,
    /// Each serving process's registration, by connection id, with the rank it serves (a spare
    /// that took a rank keeps its own registration under that rank): the hello carries them.
    serving: BTreeMap<u64, WorkerEvent>,
    /// A process's goodbye, by connection id, until it departs: a refused one says why, and its
    /// departure carries it.
    goodbyes: BTreeMap<u64, Goodbye>,
    /// The notices this node's processes reported, by rank, the newest per rank: the hello
    /// carries those still ahead for ranks still held, so a coordinator that takes over arms
    /// their deadlines again.
    notices: BTreeMap<u32, Notice>,
}

/// Shared daemon state.
pub struct Daemon {
    node: String,
    engines: Mutex<BTreeMap<String, Engine>>,
    committed_tx: mpsc::UnboundedSender<MembershipCommitted>,
    in_flight: Mutex<usize>,
    next_id: std::sync::atomic::AtomicU64,
    /// Drawn at random when the daemon starts: with a connection's id, the name of an undecided
    /// process, unique across the daemon's restarts (decision 0041).
    instance: u64,
    /// The stacks requests waiting for their process's answer, by request.
    stacks_answers: Mutex<BTreeMap<u64, oneshot::Sender<StacksDumped>>>,
    next_stacks_request: std::sync::atomic::AtomicU64,
    /// The undecided processes' questions waiting for the coordinator's answer, by request.
    role_answers: Mutex<BTreeMap<u64, oneshot::Sender<RoleAnswer>>>,
    next_role_request: std::sync::atomic::AtomicU64,
    /// The undecided processes connected, by engine and connection id, with each one's rank and
    /// name: what a hello reports.
    undecided: Mutex<undecided::ByEngine>,
    /// The connections that asked for a replay and registered nothing, by connection id, each
    /// with its queue for the offer (decision 0044).
    replay_askers: Mutex<BTreeMap<u64, mpsc::Sender<Signal>>>,
    /// What the coordinator said of each engine for its processes (decision 0041): its policy,
    /// where it runs, its instance's state; watched by the undecided connections, which have no
    /// queue of their own.
    facts: Mutex<BTreeMap<String, tokio::sync::watch::Sender<facts::Facts>>>,
    published: Mutex<
        Option<(
            FileDiscovery,
            everserve_wire::proto::everserve::discovery::v1::DaemonRecord,
        )>,
    >,
    /// The link to the coordinator's outbound queue while one is up.
    up: Mutex<Option<mpsc::Sender<DaemonToCoord>>>,
    /// Reports held while no link is up, sent in order when one comes up.
    held: Mutex<Vec<DaemonToCoord>>,
    /// This node's store: its worker socket (for control requests the coordinator routes here)
    /// and the address other stores replicate into it through, advertised in the hello.
    mirror: Mutex<Option<(PathBuf, Option<String>)>>,
    /// The GPUs this node has, as reported in the hello.
    devices: Mutex<Vec<crate::proto::everserve::coord::v1::Device>>,
    /// How often the link carries a heartbeat; zero for never (decision 0023).
    heartbeat: Mutex<Duration>,
    /// The spare processes this daemon launched (decision 0024); set when it serves.
    spares: Mutex<Option<Arc<spares::Supervisor>>>,
    /// Launches that arrived before the daemon served, started then.
    pending_launches: Mutex<Vec<LaunchSpare>>,
    /// The newest coordinator term this daemon has seen (decision 0036): a message from an older
    /// one is refused. Kept in memory: a restarted daemon learns the current term from the first
    /// message of the coordinator it links to.
    term: std::sync::atomic::AtomicU64,
    metrics: DaemonMetrics,
}

/// The heartbeat interval unless the daemon was told otherwise (`--heartbeat-ms`).
pub const DEFAULT_HEARTBEAT: Duration = Duration::from_millis(500);

impl Daemon {
    /// The daemon of `node`, reporting through `telemetry`. `committed` receives every committed
    /// membership this daemon hears.
    #[must_use]
    pub fn new(
        node: &str,
        telemetry: &everserve_telemetry::Telemetry,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<MembershipCommitted>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Arc::new(Self {
                node: node.to_owned(),
                engines: Mutex::new(BTreeMap::new()),
                committed_tx: tx,
                in_flight: Mutex::new(0),
                next_id: std::sync::atomic::AtomicU64::new(1),
                instance: undecided::draw_instance(),
                role_answers: Mutex::new(BTreeMap::new()),
                next_role_request: std::sync::atomic::AtomicU64::new(1),
                undecided: Mutex::new(BTreeMap::new()),
                replay_askers: Mutex::new(BTreeMap::new()),
                facts: Mutex::new(BTreeMap::new()),
                stacks_answers: Mutex::new(BTreeMap::new()),
                next_stacks_request: std::sync::atomic::AtomicU64::new(1),
                published: Mutex::new(None),
                up: Mutex::new(None),
                held: Mutex::new(Vec::new()),
                mirror: Mutex::new(None),
                devices: Mutex::new(devices::list(node)),
                heartbeat: Mutex::new(DEFAULT_HEARTBEAT),
                spares: Mutex::new(None),
                pending_launches: Mutex::new(Vec::new()),
                term: std::sync::atomic::AtomicU64::new(0),
                metrics: DaemonMetrics::register(telemetry),
            }),
            rx,
        )
    }

    /// This daemon's node name.
    #[must_use]
    pub fn node(&self) -> &str {
        &self.node
    }

    /// Name this node's store: its socket, and the address peers replicate into it through,
    /// which the hello advertises so the coordinator can route other ranks' rows here.
    pub async fn set_mirror(&self, socket: PathBuf, replica_address: Option<String>) {
        *self.mirror.lock().await = Some((socket, replica_address));
    }

    /// How often the link carries a heartbeat; zero for never.
    pub async fn set_heartbeat(&self, every: Duration) {
        *self.heartbeat.lock().await = every;
    }

    /// Replace the devices this daemon reports (a test on a machine without a driver).
    pub async fn set_devices(&self, devices: Vec<crate::proto::everserve::coord::v1::Device>) {
        *self.devices.lock().await = devices;
    }

    /// Publish this daemon's record in `dir` once [`Daemon::serve`] is listening; the record's
    /// engine list follows registrations and the record is withdrawn on stop.
    ///
    /// # Errors
    ///
    /// The directory could not be created.
    pub async fn publish_to(
        &self,
        dir: impl Into<PathBuf>,
        socket: &std::path::Path,
        mirror: Option<&std::path::Path>,
        host: Option<&str>,
    ) -> Result<(), discovery::DiscoveryError> {
        let fd = FileDiscovery::new(dir)?;
        let rec = discovery::record(&self.node, socket, mirror, host);
        *self.published.lock().await = Some((fd, rec));
        Ok(())
    }

    async fn republish(&self) {
        let mut guard = self.published.lock().await;
        if let Some((fd, rec)) = guard.as_mut() {
            rec.engines = self.engines.lock().await.keys().cloned().collect();
            if let Err(e) = fd.publish(rec) {
                warn!("could not publish the daemon record: {e}");
            }
        }
    }

    async fn withdraw(&self) {
        if let Some((fd, rec)) = self.published.lock().await.take()
            && let Err(e) = fd.withdraw(&rec.node)
        {
            warn!("could not withdraw the daemon record: {e}");
        }
    }

    /// Bind `path` and accept workers until `cancel` fires, then remove the socket.
    ///
    /// Returns once the listener has stopped. Deliveries in flight finish on their own tasks;
    /// wait for them with [`Daemon::drained`].
    ///
    /// # Errors
    ///
    /// The bind error, if the socket cannot be created.
    pub async fn serve(
        self: Arc<Self>,
        path: PathBuf,
        cancel: CancellationToken,
    ) -> std::io::Result<()> {
        {
            // the spares the coordinator places here register through this socket (0024)
            let supervisor = Arc::new(spares::Supervisor::new(&self.node, &path, cancel.clone()));
            *self.spares.lock().await = Some(Arc::clone(&supervisor));
            for l in std::mem::take(&mut *self.pending_launches.lock().await) {
                if let Err(e) = supervisor.launch(l).await {
                    warn!("spare not launched: {e}");
                }
            }
        }
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        let listener = UnixListener::bind(&path)?;
        info!("daemon listening on {}", path.display());
        self.republish().await;
        loop {
            tokio::select! {
                () = cancel.cancelled() => break,
                accepted = listener.accept() => {
                    let (stream, _) = accepted?;
                    let me = Arc::clone(&self);
                    let worker_cancel = cancel.clone();
                    tokio::spawn(async move {
                        tokio::select! {
                            () = worker_cancel.cancelled() => {}
                            result = me.handle(stream) => {
                                if let Err(e) = result {
                                    warn!("worker connection ended: {e}");
                                }
                            }
                        }
                    });
                }
            }
        }
        self.withdraw().await;
        // the socket's path stays: a daemon started again may have bound it already, and its
        // next bind replaces a leftover (wire's Listener says why)
        info!("daemon stopped listening");
        Ok(())
    }

    /// Resolves once no delivery is in flight.
    pub async fn drained(&self) {
        loop {
            if *self.in_flight.lock().await == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The ranks currently registered for `engine`.
    pub async fn registered(&self, engine: &str) -> Vec<u32> {
        self.engines
            .lock()
            .await
            .get(engine)
            .map(|e| e.workers.keys().copied().collect())
            .unwrap_or_default()
    }

    /// The last committed epoch heard for `engine`.
    pub async fn epoch(&self, engine: &str) -> u64 {
        self.engines.lock().await.get(engine).map_or(0, |e| e.epoch)
    }

    /// A notice source says this node leaves at the notice's deadline: tell the coordinator, which
    /// drains the node. Held until a link is up, like a report.
    pub async fn node_notice(&self, notice: NodeNotice) {
        info!(
            "node {}: notice from {} at {} ms: {}",
            self.node, notice.source, notice.deadline_ms, notice.detail
        );
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::NodeNotice(notice)),
            },
            true,
        )
        .await;
    }

    /// Whether a coordinator link is up.
    pub async fn linked(&self) -> bool {
        self.up.lock().await.is_some()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::time::Duration;

    use super::jittered;

    #[test]
    fn reconnect_waits_spread_between_half_and_all_of_the_wait() {
        let wait = Duration::from_secs(4);
        let draws: BTreeSet<Duration> = (0..64).map(|_| jittered(wait)).collect();
        assert!(
            draws.iter().all(|d| *d >= wait / 2 && *d <= wait),
            "{draws:?}"
        );
        // daemons that lost their coordinator together do not all try again together
        assert!(draws.len() > 32, "{} distinct of 64", draws.len());
        assert_eq!(jittered(Duration::ZERO), Duration::ZERO);
    }
}
