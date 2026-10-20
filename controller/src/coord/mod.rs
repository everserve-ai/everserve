// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The coordinator: the one writer of every engine's epoch (decision 0014).
//!
//! Node daemons connect to it (`proto/everserve/proto/coord.proto`), say what they hold, report
//! what their workers say, deliver its changes and return the acknowledgements. The coordinator
//! holds the view of each engine across nodes, chooses the outcome by policy, drives a change as
//! one delivery per node, and commits it to every daemon of the engine. Observed state it keeps
//! nowhere: on start, or on taking the lease, it rebuilds every engine from the daemons' hellos.
//! Desired state, what the operator asked of each node, it reads from its spec store (decision
//! 0036).
//!
//! Single-node is the case where every daemon is the same daemon.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::metrics::CoordMetrics;
use crate::policy;
use crate::proto::everserve::coord::v1::{CoordToDaemon, Delivered, StacksSent, coord_to_daemon};
use crate::proto::everserve::policy::v1::Policy;
use crate::proto::everserve::signals::v1::MembershipCommitted;
use crate::proto::everserve::spec::v1::{
    EngineStatus, ResourceSpec, SpecSnapshot, resource_spec, resource_status,
};
use crate::readiness::SpareReadiness;
use crate::spec::{MemoryStore, SpecError, SpecStore};
use everserve_wire::transport::Listener;

use crate::topology::Topology;

mod audience;
mod capability;
mod capacity;
mod conditions;
mod controller;
mod cover;
mod engine_placement;
mod fleet;
mod leaks;
mod link;
mod nodes;
mod outbox;

use outbox::Outbox;
mod placement;
mod pools;
mod queue;
mod reads;
mod replay;
mod resume;
mod roles;
mod takeover;
mod transition;
mod view;
pub use reads::NodeView;

pub use capacity::{Label, NodeLabels};
pub use cover::SETTLE;
pub use link::DEFAULT_NODE_LOSS_MISSES;
pub use nodes::{DrainState, NodeSpecError, NodeWant};
pub use pools::{PoolSpecError, PoolView};
pub use takeover::{UNREACHABLE_AFTER, default_allowance};
pub use view::{Down, EngineView, HeadProcess, Process, Processes};

/// How long the coordinator waits for a node's delivery before giving up on the change: the
/// daemon's own wait for its processes' acknowledgements ([`crate::node::ACK_TIMEOUT`]) and the
/// transit either way, so the daemon's answer that its process never acknowledged arrives before
/// this gives up, and is read as the change's failure rather than as an answer that came late.
pub const DELIVERY_TIMEOUT: Duration = crate::node::ACK_TIMEOUT.saturating_add(DELIVERY_TRANSIT);

/// The margin [`DELIVERY_TIMEOUT`] allows past the daemon's wait for a delivery's answer to
/// cross the link.
pub const DELIVERY_TRANSIT: Duration = Duration::from_secs(15);

/// How long a write to the spec store may take before the coordinator stops acting, for a
/// coordinator that holds no lease (tests, one host). Under a lease the deadline is the hold's
/// own, [`store_write_deadline_for`] its time to live, set when the hold is taken. A write
/// slower than the hold outlives it, and a coordinator that cannot write what it decides cannot
/// act: it ends its hold, and its successor reads what the store holds, whether the write landed
/// or not (decision 0038).
pub const STORE_WRITE_DEADLINE: Duration = Duration::from_secs(10);

/// How long a store write may take under a lease of `ttl`: as long as the hold itself lasts past
/// a renewal, three quarters of the time to live (`lease::renewal`), so no write made under a
/// hold can land after the hold has ended.
#[must_use]
pub fn store_write_deadline_for(ttl: Duration) -> Duration {
    ttl * 3 / 4
}

/// Why a change is being driven: the labels that tell a fault's REPLACE from a notice's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// A worker reported the rank lost.
    Fault,
    /// A notice's deadline arrived while the rank was still alive.
    Notice,
}

impl Cause {
    /// The metric label value.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Fault => "fault",
            Self::Notice => "notice",
        }
    }
}

/// A daemon's answer to a request the coordinator made of it, matched by request id.
#[derive(Debug)]
enum Reply {
    Delivered(Delivered),
    Stacks(StacksSent),
}

/// The requests in flight on one daemon link, by request id.
type Pending = Arc<Mutex<BTreeMap<u64, oneshot::Sender<Reply>>>>;

/// How long a daemon has to answer a stacks request.
const STACKS_TIMEOUT: Duration = Duration::from_secs(10);

/// Which process of an engine a stacks request names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StacksAt {
    /// The serving rank, or the spare preloaded for it or taking it.
    Rank(u32),
    /// The spare standing for any position on `node`'s `device` (decision 0030).
    Spare {
        /// Its node.
        node: String,
        /// Its device.
        device: u32,
    },
    /// The engine's head, and its launcher when that is another process (decision 0032).
    Head,
}

/// A stacks request done: the process asked on its link, on which node, and when.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct StacksTaken {
    /// The node holding the process.
    pub node: String,
    /// The process asked for its stacks; absent when its daemon cannot see its id, as another
    /// pod's on Kubernetes.
    #[serde(default)]
    pub pid: Option<u32>,
    /// Whether the process answered with its stacks in time; one that does not writes them to
    /// its own log only.
    #[serde(default)]
    pub answered: bool,
    /// Its every thread's stack, as it answered; at most 1 MiB.
    #[serde(default)]
    pub stacks: String,
    /// When the daemon was asked, Unix milliseconds.
    pub at_ms: i64,
}

/// Why a stacks request was not done.
#[derive(Debug, thiserror::Error)]
pub enum StacksError {
    /// No process of the engine stands where the request names.
    #[error("{0}")]
    NotFound(String),
    /// The node's daemon could not do it: not linked, no answer, or the signal failed.
    #[error("node {node}: {reason}")]
    Node {
        /// The node.
        node: String,
        /// Why.
        reason: String,
    },
}

/// The hold of the lease a coordinator acts under (decision 0036).
#[derive(Debug)]
struct Acting {
    /// The hold's term.
    term: u64,
    /// Cancelled when the hold ends: from then on the coordinator sends no daemon anything and
    /// claims no change.
    ended: CancellationToken,
}

/// Why a coordinator may not act under a hold of the lease (decision 0036).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ActError {
    /// It acts under one hold for its life, and already acts under this term's.
    #[error("already acting under lease term {0}")]
    AlreadyActing(u64),
    /// It took its spec store under another term: what it read is not what this term's holder
    /// may act on.
    #[error("the spec store was taken under lease term {taken}, not {term}")]
    TakenUnder {
        /// The term the store was taken under.
        taken: u64,
        /// The term asked to act under.
        term: u64,
    },
}

/// What went wrong while driving a change.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ChangeError {
    /// The rank is not a member of this engine.
    #[error("rank {0} is not a member")]
    Unknown(u32),
    /// Ranks of the engine's kept membership that this coordinator knows no holder of, after the
    /// takeover's allowance: a change would not reach them.
    #[error("ranks {ranks:?} have no holder this coordinator knows: no change is claimed")]
    Unheard {
        /// The ranks with no known holder.
        ranks: Vec<u32>,
    },
    /// The policy chose REBUILD, which the executor performs, not the runtime.
    #[error("policy chose REBUILD for rank {0}; the executor must rebuild")]
    Rebuild(u32),
    /// A survivor failed to apply the change.
    #[error("rank {rank} failed to apply the change: {error}")]
    Survivor {
        /// The survivor that failed.
        rank: u32,
        /// Its reported error.
        error: String,
    },
    /// The engine is down (decision 0032): its head departed, or a change failed after its
    /// delivery. No change runs until its head registers again.
    #[error("the engine is down ({cause}): {detail}")]
    Down {
        /// Why: `stopped`, `failed`, `lost` or `change_failed`.
        cause: &'static str,
        /// The head's goodbye, or the failed change's error.
        detail: String,
    },
    /// A survivor acknowledged the change and then left before it committed: its process is
    /// gone, so the membership the change would commit no longer exists, and committing it would
    /// commit an engine with no process left.
    #[error("ranks {ranks:?} left after acknowledging the change; it is not committed")]
    Departed {
        /// The survivors no longer registered.
        ranks: Vec<u32>,
    },
    /// A node did not deliver in time, refused the delivery, or is not connected.
    #[error("node {node} did not deliver the change: {reason}")]
    Node {
        /// The node.
        node: String,
        /// Why.
        reason: String,
    },
    /// A survivor acknowledged a different epoch than the one being committed.
    #[error("rank {rank} acknowledged epoch {got}, the change is epoch {expected}")]
    Epoch {
        /// The survivor.
        rank: u32,
        /// The epoch it acknowledged.
        got: u64,
        /// The epoch of the change.
        expected: u64,
    },
    /// A replacement would put a captured collective group across domains, and the group cannot
    /// move whole (decision 0022): a rank of it has no spare, or the spares are not in one domain.
    #[error("rank {rank}: {reason}")]
    Domain {
        /// The reported rank.
        rank: u32,
        /// Why the group cannot move.
        reason: String,
    },
    /// A daemon reported a newer lease term than this coordinator's: another coordinator holds
    /// the lease, and this one claims nothing (decision 0036).
    #[error("deposed: term {term} is ours, a daemon has seen term {newer}")]
    Deposed {
        /// This coordinator's term.
        term: u64,
        /// The newer term a daemon reported.
        newer: u64,
    },
    /// This coordinator's hold of the lease ended: it claims nothing (decision 0036).
    #[error("lease lost: term {term} is no longer ours")]
    LeaseLost {
        /// The term this coordinator held.
        term: u64,
    },
    /// The engine's head did not say it serves membership changes: it runs without Everserve's
    /// step hook, and a change would reach an engine core that cannot make it.
    #[error("the engine's head on {node} does not serve membership changes: {detail}")]
    NotServing {
        /// The head's node.
        node: String,
        /// Why, as the head's identity says.
        detail: String,
    },
    /// The store did not keep the change's log (decision 0038): before its delivery, nothing went
    /// out and the change did not happen; at its commit, the change fails after its delivery.
    #[error("the change at epoch {epoch} was not logged: {reason}")]
    Unlogged {
        /// The change's epoch.
        epoch: u64,
        /// Why the store did not keep it.
        reason: String,
    },
    /// Cover is planned and no covered set answers the loss (decision 0035): the change is
    /// refused at once rather than wait for a set to form, unless the policy lets it form one on
    /// the critical path.
    #[error("no covered set answers the loss of ranks {ranks:?}: {why}")]
    Uncovered {
        /// The ranks lost.
        ranks: Vec<u32>,
        /// What is missing: no set planned for the loss, or one planned and not yet formed.
        why: String,
    },
    /// Every reported rank has changed since the epoch the reporter saw it in: the report is a
    /// second sighting of a death already acted on (decision 0021), and nothing is changed.
    #[error("rank {rank} changed at epoch {changed}, after the report's epoch {seen}")]
    Stale {
        /// The first reported rank.
        rank: u32,
        /// The epoch the report saw it in.
        seen: u64,
        /// The epoch it last changed at.
        changed: u64,
    },
}

/// One spare in the pool, as the readiness view reports it.
pub type SpareView = SpareReadiness;

/// One connected daemon.
struct DaemonLink {
    /// What is queued for the daemon; its link's writer sends it.
    tx: Outbox,
    pending: Pending,
    /// Which link this is, of the daemon's links over time: a link that ended is forgotten, and
    /// its node taken as lost, only while it is still the one the daemon holds. A daemon that
    /// relinked before its old link's reader timed out replaced it, and is not lost.
    connection: u64,
    /// Fired when a newer link from the same daemon replaced this one: its handler ends quietly.
    replaced: CancellationToken,
}

/// Shared coordinator state.
pub struct Coordinator {
    policy: Policy,
    /// Each engine of the policy by name: its index in the policy's engines.
    engine_index: BTreeMap<String, usize>,
    /// The policy without its engines: one engine's policy is this with that engine.
    policy_frame: Policy,
    /// For each node, the engines the policy places a member, a spare or a replica of on it.
    placed_on: BTreeMap<String, BTreeSet<String>>,
    engines: Mutex<BTreeMap<String, EngineView>>,
    daemons: Mutex<BTreeMap<String, DaemonLink>>,
    /// Every device the daemons reported.
    topology: Mutex<Topology>,
    /// Where each node's store takes replication, from its daemon's hello; absent when it does
    /// not.
    replica_addresses: Mutex<BTreeMap<String, String>>,
    /// Where each node's processes rendezvous, from its daemon's hello (decision 0035).
    advertise_hosts: Mutex<BTreeMap<String, String>>,
    /// Missed heartbeats after which a daemon's node is lost (decision 0023).
    node_loss_misses: AtomicU64,
    /// The nodes' capacity, as the cluster's Nodes say it; None while they are not followed.
    capacity: Mutex<Option<capacity::Capacity>>,
    /// The nodes whose preemption the cluster's Nodes announced, each drained once.
    preempted: Mutex<BTreeSet<String>>,
    /// How long a store write may take, in milliseconds, before this coordinator stops acting
    /// ([`STORE_WRITE_DEADLINE`] unless set).
    store_deadline_ms: AtomicU64,
    /// The hold of the lease this coordinator acts under (decision 0036): its term, stamped on
    /// every message to a daemon, and its end. Unset without a lease: term 0.
    hold: std::sync::OnceLock<Acting>,
    /// The lease term its spec store was taken under, if it was taken: the one term it may act
    /// under.
    taken: std::sync::OnceLock<u64>,
    /// The newest lease term any daemon reported: above `term`, another coordinator holds the
    /// lease, and this one claims no change.
    newest_seen: AtomicU64,
    committed_tx: mpsc::UnboundedSender<MembershipCommitted>,
    /// Changes being driven now: a shutdown waits for them (`drained`), never for a deadline
    /// still ahead, which the daemons' hellos carry to a successor.
    in_flight: Mutex<usize>,
    /// The notice deadlines waiting to drive a change, by engine, rank and deadline: a notice
    /// heard again (a hello after a relink carries the node's standing notices) is not armed twice.
    armed: Mutex<BTreeSet<(String, Vec<u32>, i64)>>,
    next_request: AtomicU64,
    /// Numbers each daemon link as it is accepted (`DaemonLink::connection`).
    next_link: AtomicU64,
    /// When this coordinator started: it plans cover only once the daemons it expects relinked,
    /// or [`cover::SETTLE`] after this.
    started: std::time::Instant,
    metrics: CoordMetrics,
    /// The engine controller's queue and pending reports (decision 0036).
    controller: controller::Controller,
    /// How this coordinator heard the cell again after it started (decision 0036).
    takeover: takeover::Takeover,
    /// This coordinator, for the controller's worker and transitions to run as their own tasks.
    me: Weak<Self>,
    /// Desired state: the store, this coordinator's copy of it, and the drains it set going
    /// (decision 0036).
    specs: Specs,
}

/// A pool's launched spares, by node and device, with when each was launched.
type Launches = BTreeMap<(String, u32), std::time::Instant>;

/// The spec store, this coordinator's copy of what it holds, and where each node's drain stands.
struct Specs {
    store: Arc<dyn SpecStore>,
    /// Every resource's spec by key, as last read from or written to the store; a reconcile
    /// reads this, never the store.
    cache: Mutex<BTreeMap<String, ResourceSpec>>,
    /// Nodes whose spec asks for a drain, by node.
    drains: Mutex<BTreeMap<String, DrainState>>,
    /// The node controller's queue: nodes to reconcile. Its own queue, apart from the engine
    /// controller's: a drain's changes go through the engine controller, whose outcomes only
    /// wake this queue.
    queue: queue::WorkQueue<String>,
    /// Each node's conditions and event log.
    status: Mutex<BTreeMap<String, crate::status::Status>>,
    /// Each pool's conditions and event log (decision 0031).
    pool_status: Mutex<BTreeMap<String, crate::status::Status>>,
    /// The spares this coordinator launched for each pool and when, by node and device, until
    /// they stand or stop counting as warming.
    launched: Mutex<BTreeMap<String, Launches>>,
    /// Each engine's status as the store keeps it (decision 0038), read at the start and
    /// written whole; each engine's lock is held across its write, so its writes land in order.
    kept: Mutex<BTreeMap<String, Arc<Mutex<EngineStatus>>>>,
    /// Each engine's status as last written, for readers: a reader never waits behind a write
    /// in flight, which holds the engine's lock for as long as the store takes.
    written: std::sync::RwLock<BTreeMap<String, EngineStatus>>,
    /// Engines whose last change a previous coordinator logged, which this one finishes before
    /// it claims any other change of them (decision 0038).
    unfinished: Mutex<BTreeSet<String>>,
    /// Engines with a wake set for when [`cover::SETTLE`] has passed.
    settle_wakes: Mutex<BTreeSet<String>>,
}

/// The drains the spec store holds, each waiting for its node's hello.
fn drains_of(snapshot: &SpecSnapshot) -> BTreeMap<String, DrainState> {
    snapshot
        .resources
        .values()
        .filter_map(|r| match &r.kind {
            Some(resource_spec::Kind::Node(n)) if n.drain_by_ms != 0 => Some((
                n.node.clone(),
                DrainState::Waiting(
                    "read from the spec store; its node has not said hello".to_owned(),
                ),
            )),
            _ => None,
        })
        .collect()
}

impl Coordinator {
    /// A coordinator for `policy` reporting through `telemetry`, keeping desired state in
    /// memory. `committed` receives every committed membership.
    #[must_use]
    pub fn new(
        policy: Policy,
        telemetry: &everserve_telemetry::Telemetry,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<MembershipCommitted>) {
        Self::build(
            policy,
            telemetry,
            Arc::new(MemoryStore::default()),
            SpecSnapshot::default(),
        )
    }

    /// A coordinator for `policy` reporting through `telemetry`, keeping desired state in
    /// `store`, which it takes now under lease term `term` ([`SpecStore::take`]): the term it
    /// then acts under ([`Coordinator::act_under`]). A node the store says is drained resumes
    /// its drain once its positions are known.
    ///
    /// # Errors
    ///
    /// The store could not be taken: a newer term took it, or it could not be read or written.
    pub async fn with_store(
        policy: Policy,
        telemetry: &everserve_telemetry::Telemetry,
        store: Arc<dyn SpecStore>,
        term: u64,
    ) -> Result<(Arc<Self>, mpsc::UnboundedReceiver<MembershipCommitted>), SpecError> {
        let snapshot = store.take(term).await?;
        info!(
            "spec store: {}, taken under term {term}: {} resources, {} engines' status",
            store.describe(),
            snapshot.resources.len(),
            snapshot.statuses.len()
        );
        let built = Self::build(policy, telemetry, store, snapshot);
        built
            .0
            .taken
            .set(term)
            .unwrap_or_else(|_| unreachable!("a coordinator just built took no store"));
        Ok(built)
    }

    fn build(
        policy: Policy,
        telemetry: &everserve_telemetry::Telemetry,
        store: Arc<dyn SpecStore>,
        snapshot: SpecSnapshot,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<MembershipCommitted>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let metrics = CoordMetrics::register(telemetry);
        let controller = controller::Controller::new(&metrics);
        let queue = queue::WorkQueue::new(&metrics, "node");
        let drains = drains_of(&snapshot);
        (
            Arc::new_cyclic(|me| Self {
                policy_frame: Policy {
                    engines: Vec::new(),
                    ..policy.clone()
                },
                placed_on: audience::placed_on(&policy),
                engine_index: policy
                    .engines
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (e.name.clone(), i))
                    .collect(),
                policy,
                engines: Mutex::new(BTreeMap::new()),
                daemons: Mutex::new(BTreeMap::new()),
                topology: Mutex::new(Topology::default()),
                replica_addresses: Mutex::new(BTreeMap::new()),
                advertise_hosts: Mutex::new(BTreeMap::new()),
                node_loss_misses: AtomicU64::new(DEFAULT_NODE_LOSS_MISSES),
                capacity: Mutex::new(None),
                preempted: Mutex::new(BTreeSet::new()),
                store_deadline_ms: AtomicU64::new(
                    u64::try_from(STORE_WRITE_DEADLINE.as_millis()).unwrap_or(u64::MAX),
                ),
                hold: std::sync::OnceLock::new(),
                taken: std::sync::OnceLock::new(),
                newest_seen: AtomicU64::new(0),
                committed_tx: tx,
                in_flight: Mutex::new(0),
                armed: Mutex::new(BTreeSet::new()),
                next_request: AtomicU64::new(1),
                next_link: AtomicU64::new(1),
                started: std::time::Instant::now(),
                controller,
                takeover: takeover::Takeover::default(),
                me: me.clone(),
                metrics,
                specs: Specs {
                    store,
                    cache: Mutex::new(snapshot.resources.into_iter().collect()),
                    drains: Mutex::new(drains),
                    queue,
                    status: Mutex::new(BTreeMap::new()),
                    pool_status: Mutex::new(BTreeMap::new()),
                    launched: Mutex::new(BTreeMap::new()),
                    settle_wakes: Mutex::new(BTreeSet::new()),
                    unfinished: Mutex::new(
                        snapshot
                            .statuses
                            .values()
                            .filter_map(|s| match &s.kind {
                                Some(resource_status::Kind::Engine(e)) if e.change.is_some() => {
                                    Some(e.engine.clone())
                                }
                                _ => None,
                            })
                            .collect(),
                    ),
                    written: std::sync::RwLock::new(
                        snapshot
                            .statuses
                            .values()
                            .filter_map(|s| {
                                s.kind.as_ref().map(|resource_status::Kind::Engine(e)| {
                                    (e.engine.clone(), e.clone())
                                })
                            })
                            .collect(),
                    ),
                    kept: Mutex::new(
                        snapshot
                            .statuses
                            .into_values()
                            .filter_map(|s| match s.kind {
                                Some(resource_status::Kind::Engine(e)) => {
                                    Some((e.engine.clone(), Arc::new(Mutex::new(e))))
                                }
                                None => None,
                            })
                            .collect(),
                    ),
                },
            }),
            rx,
        )
    }

    /// Accept daemons on `listener` until `cancel` fires.
    ///
    /// # Errors
    ///
    /// The accept error.
    pub async fn serve(
        self: Arc<Self>,
        listener: Listener,
        cancel: CancellationToken,
    ) -> std::io::Result<()> {
        info!(
            "coordinator listening on {}",
            listener
                .local_address()
                .map_or_else(|_| "?".to_owned(), |a| a.to_string())
        );
        // the clocks of a start or a takeover: a rank never heard again, a cell never heard
        self.watch_takeover(&cancel);
        // cover planned at last for an engine whose daemons did not all relink
        if let Some(me) = self.me.upgrade() {
            tokio::spawn(async move {
                tokio::time::sleep(cover::SETTLE).await;
                let engines: Vec<String> = me.engines.lock().await.keys().cloned().collect();
                for engine in engines {
                    me.observed(&engine).await;
                }
            });
        }
        loop {
            tokio::select! {
                () = cancel.cancelled() => break,
                accepted = listener.accept() => {
                    let accepted = match accepted {
                        Ok(a) => a,
                        Err(e) => {
                            warn!("coordinator accept: {e}");
                            continue;
                        }
                    };
                    let me = Arc::clone(&self);
                    let conn_cancel = cancel.clone();
                    tokio::spawn(async move {
                        tokio::select! {
                            () = conn_cancel.cancelled() => {}
                            result = me.handle(accepted) => {
                                if let Err(e) = result {
                                    warn!("daemon connection ended: {e}");
                                }
                            }
                        }
                    });
                }
            }
        }
        info!("coordinator stopped listening");
        Ok(())
    }

    /// Resolves once no change is in flight.
    pub async fn drained(&self) {
        loop {
            if *self.in_flight.lock().await == 0 && self.controller.backlog().await == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The policy's `engine`, found by name without a walk of every engine.
    #[must_use]
    pub fn engine_spec(
        &self,
        engine: &str,
    ) -> Option<&crate::proto::everserve::policy::v1::Engine> {
        self.engine_index
            .get(engine)
            .and_then(|&i| self.policy.engines.get(i))
    }

    /// The view of `engine`, for tests and the readiness API.
    pub async fn engine(&self, engine: &str) -> Option<EngineView> {
        self.engines.lock().await.get(engine).cloned()
    }

    /// Nodes whose daemon is connected.
    pub async fn daemons(&self) -> Vec<String> {
        self.daemons.lock().await.keys().cloned().collect()
    }

    /// Whether `engine` is held to the exact class (decision 0030); an engine the policy does not
    /// cover is held to it.
    fn exact(&self, engine: &str) -> bool {
        self.engine_spec(engine).is_none_or(policy::exact)
    }

    /// One message to `node`'s daemon, if it is linked.
    async fn tell(&self, node: &str, kind: coord_to_daemon::Kind) {
        let tx = self.daemons.lock().await.get(node).map(|d| d.tx.clone());
        if let Some(tx) = tx {
            let _ = tx.send(CoordToDaemon {
                term: 0,
                kind: Some(kind),
            });
        }
    }

    /// The daemons holding a member or a spare of `engine`, standing for one position or any, by
    /// node.
    async fn daemons_of(&self, engine: &str) -> BTreeMap<String, Outbox> {
        let nodes: BTreeSet<String> = {
            let engines = self.engines.lock().await;
            engines
                .get(engine)
                .map(|v| {
                    v.members
                        .values()
                        .cloned()
                        .chain(v.spares.values().cloned())
                        .chain(v.any_spares.iter().map(|(n, _)| n.clone()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let daemons = self.daemons.lock().await;
        nodes
            .into_iter()
            .filter_map(|n| daemons.get(&n).map(|d| (n.clone(), d.tx.clone())))
            .collect()
    }
}

/// Hand a daemon's answer to the request waiting for it, or say it came twice or late.
async fn answer(pending: &Pending, node: &str, id: u64, reply: Reply) {
    if let Some(waiting) = pending.lock().await.remove(&id) {
        let _ = waiting.send(reply);
    } else {
        warn!("node {node}: request {id} answered twice or late");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_store_write_may_take_as_long_as_the_hold_lasts_past_a_renewal() {
        // the hold ends three quarters of the time to live after the last answered renewal
        // (lease::renewal): a write bounded the same way never lands after the hold has ended
        assert_eq!(
            store_write_deadline_for(Duration::from_secs(10)),
            Duration::from_millis(7_500)
        );
        assert_eq!(
            store_write_deadline_for(Duration::from_secs(4)),
            Duration::from_secs(3)
        );
    }
}
