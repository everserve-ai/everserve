// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The coordinator link (decision 0014): connect, say hello, send what was held, serve
//! the link, and on any failure wait and connect again; the lease names which coordinator
//! (decision 0036), and a message from an older term is refused.

use super::{Daemon, RECONNECT_MAX, RECONNECT_MIN, facts, jittered};
use crate::proto::everserve::coord::v1::{
    Bye, CoordToDaemon, DaemonHello, DaemonToCoord, Delivered, EngineSlot, EngineState, Heartbeat,
    LeaseRecord, coord_to_daemon, daemon_to_coord, worker_event,
};
use crate::proto::everserve::signals::v1::SlotChoice;
use everserve_wire::frame as wire;
use everserve_wire::transport::{Address, Tls, connect};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::info;
use tracing::warn;

/// What a refused message is answered with, when its sender waits on one: a delivery is told it
/// was fenced, so the deposed coordinator's change fails at once instead of timing out.
fn fenced_answer(msg: &CoordToDaemon, refusal: &str) -> Option<DaemonToCoord> {
    match msg.kind.as_ref()? {
        coord_to_daemon::Kind::Deliver(d) => Some(DaemonToCoord {
            kind: Some(daemon_to_coord::Kind::Delivered(Delivered {
                request_id: d.request_id,
                engine: d.engine.clone(),
                epoch: d.change.as_ref().map_or(0, |c| c.epoch),
                acks: Vec::new(),
                error: refusal.to_owned(),
            })),
        }),
        _ => None,
    }
}

/// The next heartbeat, or never when the daemon sends none.
async fn heartbeat_due(tick: Option<&mut tokio::time::Interval>) {
    match tick {
        Some(t) => {
            t.tick().await;
        }
        None => std::future::pending::<()>().await,
    }
}

impl Daemon {
    /// Keep a link to the coordinator at `address` until `cancel` fires: connect, say hello,
    /// send what was held, serve the link; on any failure wait and connect again.
    pub async fn link(
        self: Arc<Self>,
        address: Address,
        tls: Option<Tls>,
        server_name: String,
        cancel: CancellationToken,
    ) {
        let (_fixed, target) = tokio::sync::watch::channel(Some(address));
        self.link_to(target, tls, server_name, cancel).await;
    }

    /// Keep a link to the coordinator `target` names until `cancel` fires, as [`Daemon::link`]
    /// does; when `target` names another, the link moves there at once (decision 0036: the lease
    /// says which coordinator is active). While it names none, the daemon holds its reports.
    pub async fn link_to(
        self: Arc<Self>,
        mut target: tokio::sync::watch::Receiver<Option<Address>>,
        tls: Option<Tls>,
        server_name: String,
        cancel: CancellationToken,
    ) {
        let mut wait = RECONNECT_MIN;
        loop {
            if cancel.is_cancelled() {
                return;
            }
            let address = target.borrow_and_update().clone();
            let Some(address) = address else {
                tokio::select! {
                    () = cancel.cancelled() => return,
                    _ = target.changed() => continue,
                }
            };
            match connect(&address, tls.as_ref(), &server_name).await {
                Ok(stream) => {
                    wait = RECONNECT_MIN;
                    info!("linked to the coordinator at {address}");
                    let moved = tokio::select! {
                        result = Arc::clone(&self).serve_link(stream, &cancel) => {
                            match result {
                                Ok(()) => {
                                    *self.up.lock().await = None;
                                    self.metrics.coordinator_linked.set(0);
                                    return; // cancelled
                                }
                                Err(e) => {
                                    warn!("coordinator link ended: {e}; reconnecting");
                                    false
                                }
                            }
                        }
                        _ = target.changed() => {
                            info!("the lease names another coordinator: leaving {address}");
                            true
                        }
                    };
                    *self.up.lock().await = None;
                    self.metrics.coordinator_linked.set(0);
                    if moved {
                        continue;
                    }
                }
                Err(e) => warn!("coordinator at {address} unreachable: {e}; retrying in {wait:?}"),
            }
            tokio::select! {
                () = cancel.cancelled() => return,
                () = tokio::time::sleep(jittered(wait)) => {}
                _ = target.changed() => continue,
            }
            wait = (wait * 2).min(RECONNECT_MAX);
        }
    }

    /// Watch the coordinator lease's record until `cancel` fires (decision 0036): its term raises
    /// this daemon's fence the moment it is read, so a deposed coordinator's messages are refused
    /// while its link still stands; then its address, sent on `target`, moves the link to the
    /// coordinator that holds the lease. Read every `every`; a record that cannot be read leaves
    /// both as they are.
    pub async fn watch_lease(
        self: Arc<Self>,
        watch: Arc<dyn crate::lease::LeaseWatch>,
        target: tokio::sync::watch::Sender<Option<Address>>,
        every: Duration,
        cancel: CancellationToken,
    ) {
        info!("following the coordinator lease at {}", watch.describe());
        loop {
            match watch.read().await {
                Ok(Some(record)) => self.follow(&record, &target),
                Ok(None) => {}
                Err(e) => warn!("{e}"),
            }
            tokio::select! {
                () = cancel.cancelled() => return,
                () = tokio::time::sleep(every) => {}
            }
        }
    }

    /// Take `record` as the lease's: its term first, then its address.
    fn follow(&self, record: &LeaseRecord, target: &tokio::sync::watch::Sender<Option<Address>>) {
        use std::sync::atomic::Ordering;
        let before = self.term.fetch_max(record.term, Ordering::SeqCst);
        if record.term > before {
            info!(
                "the lease is at term {} held by {} at {}",
                record.term, record.holder, record.address
            );
        }
        if record.address.is_empty() {
            // no one holds the lease: hold reports until someone does, rather than chase the
            // address of a coordinator that let it go
            target.send_if_modified(|current| current.take().is_some());
            return;
        }
        match Address::parse(&record.address) {
            Ok(address) => {
                target.send_if_modified(|current| {
                    let changed = current.as_ref() != Some(&address);
                    if changed {
                        *current = Some(address.clone());
                    }
                    changed
                });
            }
            Err(e) => warn!("the lease record names no address daemons can reach: {e}"),
        }
    }

    /// The newest coordinator term this daemon has seen (decision 0036).
    #[must_use]
    pub fn term(&self) -> u64 {
        self.term.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn serve_link(
        self: Arc<Self>,
        stream: everserve_wire::transport::Stream,
        cancel: &CancellationToken,
    ) -> Result<(), wire::WireError> {
        let (rd, mut wr) = tokio::io::split(stream);
        // read through a FrameReader: the heartbeat tick and the outbound queue drop the read
        // below, and a frame half-read when that happens must not be lost
        let mut rd = wire::FrameReader::new(rd);
        let (tx, mut rx) = mpsc::channel::<DaemonToCoord>(64);
        // the coordinator's messages are taken in the order they came, on a task of their own so
        // the reader keeps up with the heartbeats; only a delivery and a stacks request, which
        // wait on processes, run beside the rest
        let (ordered_tx, ordered_rx) = mpsc::channel::<CoordToDaemon>(256);
        let _in_order = AbortOnDropHandle::new(tokio::spawn(
            Arc::clone(&self).take_in_order(ordered_rx, tx.clone()),
        ));
        let heartbeat = *self.heartbeat.lock().await;
        let hello = DaemonToCoord {
            kind: Some(daemon_to_coord::Kind::Hello(DaemonHello {
                node: self.node.clone(),
                engines: self.state().await,
                devices: self.devices.lock().await.clone(),
                mirror_replica_address: self
                    .mirror
                    .lock()
                    .await
                    .as_ref()
                    .and_then(|(_, r)| r.clone())
                    .unwrap_or_default(),
                heartbeat_ms: u32::try_from(heartbeat.as_millis()).unwrap_or(u32::MAX),
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                term: self.term.load(std::sync::atomic::Ordering::SeqCst),
                advertise_host: self
                    .published
                    .lock()
                    .await
                    .as_ref()
                    .map(|(_, rec)| rec.host.clone())
                    .unwrap_or_default(),
            })),
        };
        wire::send(&mut wr, &hello).await?;
        let held: Vec<DaemonToCoord> = std::mem::take(&mut *self.held.lock().await);
        for msg in held {
            wire::send(&mut wr, &msg).await?;
        }
        *self.up.lock().await = Some(tx.clone());
        self.metrics.coordinator_linked.set(1);
        // a stopped daemon says goodbye, so the coordinator keeps its ranks for its return; a
        // daemon that promised heartbeats sends them, so a lost node is seen in time
        let mut tick = (heartbeat > Duration::ZERO).then(|| {
            let mut t = tokio::time::interval(heartbeat);
            t.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            t
        });
        let pulse = DaemonToCoord {
            kind: Some(daemon_to_coord::Kind::Heartbeat(Heartbeat {})),
        };
        loop {
            tokio::select! {
                () = cancel.cancelled() => {
                    let bye = DaemonToCoord {
                        kind: Some(daemon_to_coord::Kind::Bye(Bye {})),
                    };
                    let _ = wire::send(&mut wr, &bye).await;
                    let _ = wr.shutdown().await; // close_notify: an orderly end, not a lost peer
                    return Ok(());
                }
                frame = rd.recv::<CoordToDaemon>() => {
                    let msg = frame?;
                    // checked here, in the order the messages came, before any is handed off
                    if let Some(refusal) = self.fence(&msg) {
                        warn!("{refusal}");
                        self.metrics.fenced_total.inc();
                        if let Some(answer) = fenced_answer(&msg, &refusal) {
                            wire::send(&mut wr, &answer).await?;
                        }
                        continue;
                    }
                    if ordered_tx.send(msg).await.is_err() {
                        return Err(wire::WireError::Io(std::io::Error::other(
                            "the link's message queue ended",
                        )));
                    }
                }
                Some(out) = rx.recv() => {
                    wire::send(&mut wr, &out).await?;
                }
                () = heartbeat_due(tick.as_mut()) => {
                    wire::send(&mut wr, &pulse).await?;
                }
            }
        }
    }

    /// Why `msg` is refused: its coordinator's term is older than the newest this daemon has
    /// seen (decision 0036). A newer term becomes the newest; a coordinator without a lease sends
    /// 0, refused only once a coordinator with one was heard.
    fn fence(&self, msg: &CoordToDaemon) -> Option<String> {
        use std::sync::atomic::Ordering;
        let newest = self.term.fetch_max(msg.term, Ordering::SeqCst);
        (msg.term < newest).then(|| {
            format!(
                "a coordinator message at term {} refused: term {newest} holds the lease",
                msg.term
            )
        })
    }

    /// The engine's client said which slot-collective variant the engine runs: tell the
    /// coordinator now. The hello carries it too, so it is not held across a missing link.
    pub(super) async fn report_slot(&self, engine: &str, slot: SlotChoice) {
        info!(
            "{engine}: slot collective {:?} by {:?}: {}",
            slot.variant(),
            slot.how(),
            slot.detail
        );
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::EngineSlot(EngineSlot {
                    engine: engine.to_owned(),
                    slot: Some(slot),
                })),
            },
            false,
        )
        .await;
    }

    /// What this node holds, for a hello.
    async fn state(&self) -> Vec<EngineState> {
        // read first, apart: an undecided process may be all this node holds of an engine
        let undecided = self.undecided_now().await;
        let now = crate::clock::now_ms();
        let engines = self.engines.lock().await;
        let mut states: Vec<EngineState> = engines
            .iter()
            .map(|(name, e)| EngineState {
                engine: name.clone(),
                epoch: e.epoch,
                workers: e.workers.keys().copied().collect(),
                spares: e.spares.keys().copied().collect(),
                any_spares: e.any_spares.keys().copied().collect(),
                world_size: e.world_size,
                lost: e.lost.clone(),
                departed: e.departed.iter().copied().collect(),
                slot: e.slot.clone(),
                plan_generation: e.plan.as_ref().map_or(0, |p| p.generation),
                plan: e.plan.clone(),
                held: e.held.values().flatten().cloned().collect(),
                standing: e.standing.values().cloned().collect(),
                heads: e
                    .head
                    .iter()
                    .map(|h| self.head_registration(name, h, worker_event::Kind::HeadRegistered))
                    .collect(),
                serving: e.serving.values().cloned().collect(),
                undecided: Vec::new(),
                notices: e
                    .notices
                    .values()
                    .filter(|n| n.deadline_ms > now && e.workers.contains_key(&n.rank))
                    .copied()
                    .collect(),
            })
            .collect();
        for (engine, processes) in undecided {
            match states.iter_mut().find(|s| s.engine == engine) {
                Some(s) => s.undecided = processes,
                None => states.push(EngineState {
                    engine,
                    undecided: processes,
                    ..EngineState::default()
                }),
            }
        }
        states
    }

    /// Send to the coordinator, or hold the message until a link is up when `hold` is set.
    /// Worker events are not held: the next hello carries the state they would have changed.
    pub(super) async fn send_up(&self, msg: DaemonToCoord, hold: bool) {
        let tx = self.up.lock().await.clone();
        match tx {
            Some(tx) if tx.send(msg.clone()).await.is_ok() => {}
            _ if hold => {
                warn!("no coordinator link: holding a report until one comes up");
                self.held.lock().await.push(msg);
            }
            _ => {}
        }
    }

    /// Take the coordinator's messages from `queue` in the order they came: each is applied
    /// before the next is read, so a commit is applied before the delivery that follows it,
    /// except a delivery or a stacks request, which waits on processes and runs on a task of
    /// its own. Ends when the link's reader ends.
    async fn take_in_order(
        self: Arc<Self>,
        mut queue: mpsc::Receiver<CoordToDaemon>,
        link: mpsc::Sender<DaemonToCoord>,
    ) {
        while let Some(msg) = queue.recv().await {
            match msg.kind {
                Some(coord_to_daemon::Kind::Deliver(_) | coord_to_daemon::Kind::Stacks(_)) => {
                    let (me, link) = (Arc::clone(&self), link.clone());
                    tokio::spawn(async move { me.on_coord_message(msg, link).await });
                }
                _ => Arc::clone(&self).on_coord_message(msg, link.clone()).await,
            }
        }
    }

    /// Take one message from the coordinator; a request is answered on `link`, the link it came
    /// on. Answered on a later link it would reach another coordinator, which numbers its
    /// requests afresh, as the answer to a request of its own (decision 0038: a successor
    /// delivering a change again was answered by its predecessor's failed delivery).
    async fn on_coord_message(
        self: Arc<Self>,
        msg: CoordToDaemon,
        link: mpsc::Sender<DaemonToCoord>,
    ) {
        match msg.kind {
            Some(coord_to_daemon::Kind::CoverPlan(p)) => self.cover_plan(p).await,
            Some(coord_to_daemon::Kind::Deliver(d)) => {
                *self.in_flight.lock().await += 1;
                let delivered = self.deliver(d).await;
                *self.in_flight.lock().await -= 1;
                let _ = link
                    .send(DaemonToCoord {
                        kind: Some(daemon_to_coord::Kind::Delivered(delivered)),
                    })
                    .await;
            }
            Some(coord_to_daemon::Kind::Committed(c)) => self.apply_committed(c).await,
            Some(coord_to_daemon::Kind::ForwardNotice(f)) => self.forward_notice(f).await,
            Some(coord_to_daemon::Kind::ForwardChange(f)) => self.forward_change(f).await,
            Some(coord_to_daemon::Kind::ReplicateTargets(t)) => self.replicate_targets(t).await,
            Some(coord_to_daemon::Kind::LaunchSpare(l)) => self.launch_spare(l).await,
            Some(coord_to_daemon::Kind::ForwardJoin(f)) => self.forward_join(f).await,
            Some(coord_to_daemon::Kind::ReplayAsk(a)) => self.replay_ask_to_head(a).await,
            Some(coord_to_daemon::Kind::ReplayOffer(o)) => self.replay_offer_to_asker(o).await,
            Some(coord_to_daemon::Kind::ExpectReplication(e)) => self.expect_replication(e).await,
            Some(coord_to_daemon::Kind::Abandon(a)) => self.abandon(a).await,
            Some(coord_to_daemon::Kind::RoleAnswer(a)) => self.role_answered(a).await,
            Some(coord_to_daemon::Kind::EnginePolicy(p)) => {
                self.engine_fact(facts::Fact::Policy(p)).await;
            }
            Some(coord_to_daemon::Kind::EnginePlacement(p)) => {
                self.engine_fact(facts::Fact::Placement(p)).await;
            }
            Some(coord_to_daemon::Kind::InstanceState(s)) => {
                self.engine_fact(facts::Fact::State(s)).await;
            }
            Some(coord_to_daemon::Kind::GroupId(id)) => {
                self.engine_fact(facts::Fact::GroupId(id)).await;
            }
            Some(coord_to_daemon::Kind::Stacks(s)) => {
                let sent = self.stacks(&s).await;
                let _ = link
                    .send(DaemonToCoord {
                        kind: Some(daemon_to_coord::Kind::StacksSent(sent)),
                    })
                    .await;
            }
            None => warn!("empty message from the coordinator"),
        }
    }
}
