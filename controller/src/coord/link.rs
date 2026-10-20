// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The daemons' links: hellos, worker and head events, reports and replies, the joins routed between processes, and a node's loss (layer 4's contract with the coordinator, decision 0034).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{error, info, warn};

use everserve_wire::frame as wire;

use crate::clock::now_ms;
use crate::clock::now_us;
use crate::proto::everserve::coord::v1::{
    Committed, CoordToDaemon, DaemonHello, DaemonToCoord, ForwardJoin, JoinReport, WorkerEvent,
    coord_to_daemon, daemon_to_coord, worker_event,
};
use crate::proto::everserve::signals::v1::{Join, Notice, StopCause};
use crate::readiness::JoinStamp;
use crate::status::Severity;
use everserve_wire::transport::{Accepted, Stream};

use super::fleet::place_spare;
use super::outbox::Outbox;
use super::view::{Down, EngineView, HeadProcess, Process};
use super::{
    ActError, Acting, Cause, ChangeError, Coordinator, DaemonLink, NodeSpecError, Pending, Reply,
    answer,
};
use crate::topology::Device;

/// The engine and rank another engine's member on `node`'s `device` serves at a committed epoch,
/// if one does: a spare of `engine` whose registration closes there took that position, and
/// left `engine`'s pool by withdrawing (decision 0031), not by going away. The process closes its
/// other registrations only after the take's commit is applied, and the coordinator records a
/// commit before sending it, so the take is already in the view when the close arrives.
fn serving_elsewhere(
    engines: &BTreeMap<String, EngineView>,
    engine: &str,
    node: &str,
    device: u32,
) -> Option<(String, u32)> {
    engines
        .iter()
        .filter(|(name, _)| name.as_str() != engine)
        .find_map(|(name, v)| {
            v.members
                .iter()
                .find(|(r, n)| n.as_str() == node && v.devices.get(r) == Some(&device))
                .map(|(r, _)| (name.clone(), *r))
        })
}

/// Record a spare's leaving `view`'s pool: a withdrawal to the position it took in another engine,
/// or a departure.
fn spare_left(view: &mut EngineView, node: &str, device: u32, withdrew_to: Option<(String, u32)>) {
    let standing = view.standing.contains_key(&(node.to_owned(), device));
    let (reason, message) = match withdrew_to {
        Some((engine, rank)) => (
            "SpareWithdrew",
            format!("spare on {node} device {device} took rank {rank} of {engine}"),
        ),
        None => (
            "SpareDeparted",
            format!("spare on {node} device {device} went away"),
        ),
    };
    if standing {
        view.status
            .record(now_ms(), Severity::Normal, reason, message);
    }
}

impl Coordinator {
    pub(super) async fn handle(self: Arc<Self>, accepted: Accepted) -> Result<(), wire::WireError> {
        let Accepted { stream, peer } = accepted;
        let (rd, wr) = tokio::io::split(stream);
        // read through a FrameReader: the read below is dropped by a send or by the heartbeat
        // timeout, and a frame half-read when that happens must not be lost
        let mut rd = wire::FrameReader::new(rd);
        let first: DaemonToCoord = rd.recv().await?;
        let Some(daemon_to_coord::Kind::Hello(hello)) = first.kind else {
            warn!("first message from a daemon was not a hello; closing");
            return Ok(());
        };
        if let Err(e) = everserve_wire::check_protocol(hello.protocol, hello.protocol_oldest) {
            warn!("daemon {}: {e}; closing", hello.node);
            return Ok(());
        }
        // a peer is who its certificate says (decision 0026): a hello for a node the
        // certificate is not valid for is refused, whatever CA signed it
        if !peer.is_named(&hello.node) {
            warn!(
                "daemon {}: hello from a certificate not valid for that name; closing",
                hello.node
            );
            return Ok(());
        }
        let node = hello.node.clone();
        // a daemon that promised heartbeats is lost after the policy's count of misses; one that
        // promised none is lost only when its link closes (decision 0023)
        let dead_after = (hello.heartbeat_ms > 0).then(|| {
            Duration::from_millis(
                u64::from(hello.heartbeat_ms) * self.node_loss_misses.load(Ordering::Relaxed),
            )
        });
        let mut orderly = false;
        let outbox = Outbox::default();
        let pending = Arc::new(Mutex::new(BTreeMap::new()));
        let (connection, replaced) = self.register_link(&node, &outbox, &pending).await;
        self.heard(&node);
        // the facts of the engines it holds or is placed for, before the hello's own changes go
        // to their audiences: it is one of them from now (`audience`)
        self.link_facts(&node, &hello).await;
        // the group ids its processes' groups meet with (decision 0035, amended)
        self.send_group_ids(&node).await;
        self.merge_hello(&hello).await;
        self.send_launches(&node).await;
        info!(
            "daemon {node} connected with {} engines",
            hello.engines.len()
        );
        // the engines this node's store may hold or replicate a rank of: those it holds and
        // those the policy places on it, not the cell's
        let mut touched: BTreeSet<String> =
            hello.engines.iter().map(|e| e.engine.clone()).collect();
        touched.extend(self.placed_on.get(&node).into_iter().flatten().cloned());
        self.refresh_replication(&touched).await;
        // the writer is the only task that waits on the daemon's socket (see `outbox`). It holds
        // the socket's write half, so it ends with this task, however this task ends: a
        // coordinator that stops drops its links, and each daemon must see its link close
        let mut writer = AbortOnDropHandle::new(tokio::spawn(
            Arc::clone(&self).write_link(outbox.clone(), wr),
        ));
        let result = loop {
            let next = async {
                match dead_after {
                    Some(limit) => tokio::time::timeout(limit, rd.recv::<DaemonToCoord>())
                        .await
                        .unwrap_or_else(|_| {
                            Err(wire::WireError::Io(std::io::Error::new(
                                std::io::ErrorKind::TimedOut,
                                "no heartbeat",
                            )))
                        }),
                    None => rd.recv::<DaemonToCoord>().await,
                }
            };
            tokio::select! {
                frame = next => {
                    let msg = match frame {
                        Ok(m) => m,
                        Err(e) => break Err(e),
                    };
                    match msg.kind {
                        Some(daemon_to_coord::Kind::Heartbeat(_)) => {}
                        Some(daemon_to_coord::Kind::Bye(_)) => {
                            orderly = true;
                        }
                        _ => self.on_message(&node, msg, &pending).await,
                    }
                }
                ended = &mut writer => {
                    break ended.unwrap_or_else(|e| {
                        Err(wire::WireError::Io(std::io::Error::other(format!(
                            "the link's writer ended: {e}"
                        ))))
                    });
                }
                () = replaced.cancelled() => {
                    break Ok(());
                }
            }
        };
        outbox.close();
        drop(writer);
        self.link_ended(&node, connection, orderly, result).await
    }

    /// Send what `outbox` queues to the daemon over `wr` until the link ends: Ok once this
    /// coordinator's hold ended, else why the link cannot go on.
    async fn write_link(
        self: Arc<Self>,
        outbox: Outbox,
        mut wr: tokio::io::WriteHalf<Stream>,
    ) -> Result<(), wire::WireError> {
        loop {
            let batch = outbox
                .next()
                .await
                .map_err(|why| wire::WireError::Io(std::io::Error::other(why)))?;
            for mut out in batch {
                if self.hold_ended() {
                    // a coordinator whose hold ended sends nothing; the daemon links to the
                    // lease's next holder
                    return Ok(());
                }
                // every message carries the term this coordinator holds the lease under
                out.term = self.term();
                wire::send(&mut wr, &out).await?;
            }
        }
    }

    /// Keep `node`'s new link, with `outbox` and `pending` as its queue and its requests: its
    /// number, and the token fired when a newer link replaces it. A daemon that relinks before
    /// its old link's reader has timed out replaces it: the old handler ends quietly, and nothing
    /// of the node is taken as lost.
    async fn register_link(
        &self,
        node: &str,
        outbox: &Outbox,
        pending: &Pending,
    ) -> (u64, CancellationToken) {
        let connection = self.next_link.fetch_add(1, Ordering::Relaxed);
        let replaced = CancellationToken::new();
        let mut daemons = self.daemons.lock().await;
        if let Some(old) = daemons.insert(
            node.to_owned(),
            DaemonLink {
                tx: outbox.clone(),
                pending: Arc::clone(pending),
                connection,
                replaced: replaced.clone(),
            },
        ) {
            info!("daemon {node} relinked; its earlier link is replaced");
            old.replaced.cancel();
            old.tx.close();
        }
        self.metrics
            .daemons_connected
            .set(i64::try_from(daemons.len()).unwrap_or(i64::MAX));
        (connection, replaced)
    }

    /// The link `connection` to `node`'s daemon ended: forget it, and unless it said goodbye,
    /// take its node as lost (decision 0023). A link the daemon has already replaced by a newer
    /// one ends with nothing lost: the node is still linked. A closed link and a missed heartbeat
    /// are not errors here.
    pub(super) async fn link_ended(
        self: &Arc<Self>,
        node: &str,
        connection: u64,
        orderly: bool,
        result: Result<(), wire::WireError>,
    ) -> Result<(), wire::WireError> {
        {
            let mut daemons = self.daemons.lock().await;
            match daemons.get(node) {
                Some(current) if current.connection == connection => {
                    daemons.remove(node);
                }
                _ => {
                    info!("daemon {node}: an earlier link ended; the daemon is linked anew");
                    return Ok(());
                }
            }
            self.metrics
                .daemons_connected
                .set(i64::try_from(daemons.len()).unwrap_or(i64::MAX));
        }
        self.unreached(node).await;
        let gone = self.node_gone(node).await;
        if orderly {
            info!("daemon {node} said goodbye; its ranks wait for it");
        } else {
            let why = match &result {
                Err(wire::WireError::Io(e)) if e.kind() == std::io::ErrorKind::TimedOut => {
                    "no heartbeat".to_owned()
                }
                Err(wire::WireError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    "link closed".to_owned()
                }
                Err(e) => e.to_string(),
                Ok(()) => "link ended".to_owned(),
            };
            info!("daemon {node} lost ({why}); its ranks are lost");
            self.node_lost(node, gone, &why);
        }
        match result {
            Err(wire::WireError::Io(e))
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::TimedOut
                ) =>
            {
                Ok(())
            }
            other => other,
        }
    }

    /// Act under the hold of the lease at `term` (decision 0036), until `ended` is cancelled: the
    /// term is stamped on every message to a daemon, which refuses an older one, and once the
    /// hold ends this coordinator sends no daemon anything and claims no change. A coordinator
    /// acts under one hold for its life; a second is refused.
    ///
    /// # Errors
    ///
    /// [`ActError::AlreadyActing`] with the term already held; [`ActError::TakenUnder`] when
    /// its spec store was taken under another term.
    pub fn act_under(&self, term: u64, ended: CancellationToken) -> Result<(), ActError> {
        if let Some(&taken) = self.taken.get()
            && taken != term
        {
            return Err(ActError::TakenUnder { taken, term });
        }
        let watch = ended.clone();
        self.hold
            .set(Acting { term, ended })
            .map_err(|_| ActError::AlreadyActing(self.term()))?;
        info!("acting under lease term {term}");
        let deposed = self.metrics.control.deposed.clone();
        tokio::spawn(async move {
            watch.cancelled().await;
            error!(
                "the hold of the lease under term {term} has ended: claiming nothing from now on"
            );
            deposed.set(1);
        });
        Ok(())
    }

    /// The term this coordinator acts under; 0 without a lease.
    #[must_use]
    pub fn term(&self) -> u64 {
        self.hold.get().map_or(0, |h| h.term)
    }

    fn hold_ended(&self) -> bool {
        self.hold.get().is_some_and(|h| h.ended.is_cancelled())
    }

    /// Whether this coordinator may no longer act (decision 0036): its hold of the lease ended,
    /// or a daemon reported a newer lease term than its own.
    #[must_use]
    pub fn deposed(&self) -> bool {
        self.hold_ended() || self.newest_seen.load(Ordering::SeqCst) > self.term()
    }

    /// Refuse to claim a change once deposed: a coordinator that knows another may hold the lease
    /// moves nothing, whether or not the daemons would refuse it.
    ///
    /// # Errors
    ///
    /// [`ChangeError::LeaseLost`] once the hold has ended; [`ChangeError::Deposed`] with both
    /// terms once a daemon reported a newer one.
    pub(super) fn refuse_if_deposed(&self) -> Result<(), ChangeError> {
        let (term, newer) = (self.term(), self.newest_seen.load(Ordering::SeqCst));
        if self.hold_ended() {
            return Err(ChangeError::LeaseLost { term });
        }
        if newer > term {
            return Err(ChangeError::Deposed { term, newer });
        }
        Ok(())
    }

    /// The missed heartbeats after which a daemon's node is lost (decision 0023).
    pub fn set_node_loss_misses(&self, misses: u64) {
        self.node_loss_misses
            .store(misses.max(1), Ordering::Relaxed);
    }

    /// How long a store write may take before this coordinator stops acting; at least a
    /// millisecond. [`super::STORE_WRITE_DEADLINE`] unless set.
    pub fn set_store_write_deadline(&self, deadline: std::time::Duration) {
        self.store_deadline_ms.store(
            u64::try_from(deadline.as_millis())
                .unwrap_or(u64::MAX)
                .max(1),
            Ordering::Relaxed,
        );
    }

    /// Stop acting: end this coordinator's hold of the lease, as its loss would, so its
    /// supervisor restarts it to contend again. Nothing without a lease.
    pub(super) fn step_down(&self, why: &str) {
        if let Some(h) = self.hold.get() {
            error!("ending the hold of the lease under term {}: {why}", h.term);
            h.ended.cancel();
        }
    }

    /// A node's daemon is gone without a goodbye: every rank it held is lost, one change per
    /// engine for the set (decision 0023). The changes run on their own tasks.
    pub(super) fn node_lost(
        self: &Arc<Self>,
        node: &str,
        gone: BTreeMap<String, Vec<u32>>,
        why: &str,
    ) {
        for (engine, ranks) in gone {
            if ranks.is_empty() {
                continue;
            }
            let me = Arc::clone(self);
            let node = node.to_owned();
            let why = why.to_owned();
            tokio::spawn(async move {
                match me.ranks_lost(&engine, &ranks, Cause::Fault).await {
                    Ok(_) => info!("{engine}: node {node} lost ({why}): ranks {ranks:?} changed"),
                    Err(e) => warn!(
                        "{engine}: node {node} lost ({why}): change for ranks {ranks:?} failed: {e}"
                    ),
                }
            });
        }
    }

    /// A daemon's hello: take its engines' state as the truth for the ranks on that node and the
    /// highest epoch heard.
    /// What a hello says of its node itself: its devices, where its store takes replication,
    /// and where its processes rendezvous.
    async fn merge_node_facts(&self, hello: &DaemonHello) {
        let mut topo = self.topology.lock().await;
        topo.set_node(
            &hello.node,
            hello
                .devices
                .iter()
                .map(|d| Device::from_wire(&hello.node, d))
                .collect(),
        );
        let mut addresses = self.replica_addresses.lock().await;
        if hello.mirror_replica_address.is_empty() {
            addresses.remove(&hello.node);
        } else {
            addresses.insert(hello.node.clone(), hello.mirror_replica_address.clone());
        }
        let mut hosts = self.advertise_hosts.lock().await;
        if hello.advertise_host.is_empty() {
            hosts.remove(&hello.node);
        } else {
            hosts.insert(hello.node.clone(), hello.advertise_host.clone());
        }
    }

    pub(super) async fn merge_hello(&self, hello: &DaemonHello) {
        let ours = self.term();
        let before = self.newest_seen.fetch_max(hello.term, Ordering::SeqCst);
        if hello.term > ours && before <= ours {
            error!(
                "deposed: node {}'s daemon has seen lease term {}, this coordinator holds {ours}; \
                 it claims no change from now on",
                hello.node, hello.term
            );
            self.metrics.control.deposed.set(1);
        }
        self.merge_node_facts(hello).await;
        // the highest plan generation the store keeps, which a restarted coordinator numbers its
        // first plan above; the plan itself it adopts from the hellos (decision 0038)
        let stored: BTreeMap<String, u64> = hello
            .engines
            .iter()
            .filter_map(|state| {
                let kept = self.kept_status(&state.engine)?.plan_generation;
                Some((state.engine.clone(), kept))
            })
            .collect();
        let mut stale: Vec<(String, Vec<u32>, Committed)> = Vec::new();
        // the notices the node's processes reported whose deadlines are ahead: armed again, once
        let mut standing_notices: Vec<(String, Notice)> = Vec::new();
        let mut engines = self.engines.lock().await;
        for state in &hello.engines {
            let view = engines.entry(state.engine.clone()).or_default();
            // a node behind the engine's committed epoch missed a commit
            let behind = state.epoch < view.committed_epoch;
            self.merge_heads(view, &hello.node, &state.heads);
            if let Some(&generation) = stored.get(&state.engine) {
                view.cover.reserved = view.cover.reserved.max(generation);
            }
            view.epoch = view.epoch.max(state.epoch);
            view.committed_epoch = view.committed_epoch.max(state.epoch);
            let left_out = merge_workers(view, &hello.node, state, behind);
            if behind {
                stale.push((
                    state.engine.clone(),
                    left_out,
                    standing_commit(&state.engine, view),
                ));
            }
            merge_spares(view, &hello.node, state);
            // the serving processes' registrations: each position's device, identity and pair
            for ev in &state.serving {
                view.devices.insert(ev.rank, ev.device);
                view.hosts.insert(ev.rank, ev.host.clone());
                if let Some(id) = ev.identity.as_ref() {
                    view.identities.insert(ev.rank, id.clone());
                    // a successor holds later askers to the serving ranks' step config, not to
                    // whichever process asks it first
                    if view.step.is_none() {
                        view.step = id.step.clone().filter(|s| !s.digest.is_empty());
                    }
                }
                if let Some(c) = ev.compile_cache.as_ref() {
                    view.hold_member_cache(&state.engine, &hello.node, ev.rank, c);
                }
            }
            // the spares' registrations, from which their admission is derived again
            view.standing.retain(|(n, _), _| n != &hello.node);
            for ev in &state.standing {
                view.standing
                    .insert((hello.node.clone(), ev.device), ev.clone());
            }
            super::cover::merge_held(view, &hello.node, state);
            standing_notices.extend(
                state
                    .notices
                    .iter()
                    .filter(|n| n.deadline_ms > now_ms())
                    .map(|n| (state.engine.clone(), *n)),
            );
            view.world_size = view.world_size.max(state.world_size);
            if let Some(slot) = &state.slot {
                view.slots.insert(hello.node.clone(), slot.clone());
            }
            if view.lost.is_empty() {
                view.lost.clone_from(&state.lost);
            }
            view.departed.extend(state.departed.iter().copied());
        }
        drop(engines);
        self.tell_missed(&hello.node, stale).await;
        // decision 0041: a process told to serve a rank on this node that the node no longer
        // names is gone
        self.release_gone(hello).await;
        for state in &hello.engines {
            self.observed(&state.engine).await;
            // a change waiting for this coordinator to hear from the engine's daemons (a logged
            // one, a loss reported, a head lost) goes once they are back
            self.wake(&state.engine).await;
        }
        self.positions_known(&hello.node).await;
        // the node's standing notices: a coordinator that starts or takes the lease arms each
        // deadline from here; one that heard the notice already arms nothing twice
        if let Some(me) = self.me.upgrade() {
            for (engine, notice) in standing_notices {
                Arc::clone(&me).notice(&engine, notice).await;
            }
        }
    }

    /// A daemon disconnected: its ranks are unreachable until it is back; a report naming them
    /// must still find them.
    pub(super) async fn node_gone(&self, node: &str) -> BTreeMap<String, Vec<u32>> {
        self.topology.lock().await.remove_node(node);
        self.replica_addresses.lock().await.remove(node);
        self.advertise_hosts.lock().await.remove(node);
        let mut engines = self.engines.lock().await;
        let mut out = BTreeMap::new();
        for (name, view) in engines.iter_mut() {
            super::cover::forget_node_cover(view, node);
            let gone: Vec<u32> = view
                .members
                .iter()
                .filter(|(_, n)| n.as_str() == node)
                .map(|(r, _)| *r)
                .collect();
            for r in &gone {
                view.members.remove(r);
                view.departed.insert(*r);
            }
            view.spares.retain(|_, n| n != node);
            view.standing.retain(|(n, _), _| n != node);
            view.refused_spares.retain(|(n, _), _| n != node);
            view.slots.remove(node);
            out.insert(name.clone(), gone);
        }
        let names: Vec<String> = engines.keys().cloned().collect();
        drop(engines);
        for name in names {
            // a process told to serve a rank on the lost node will never register (0041)
            self.release_assignments(&name, "its node is lost", |a| a.node == node)
                .await;
            self.observed(&name).await;
        }
        out
    }

    /// `node` missed a commit of each engine in `stale`: it hears the membership as it stands,
    /// so its daemon stops the workers it leaves out, and the engine says which ranks those were.
    async fn tell_missed(&self, node: &str, stale: Vec<(String, Vec<u32>, Committed)>) {
        for (engine, ranks, c) in stale {
            self.tell(node, coord_to_daemon::Kind::Committed(c)).await;
            if !ranks.is_empty() {
                let why = format!(
                    "ranks {ranks:?} on {node} registered before the engine's last commit, which \
                     does not place them there: not its members, and told the membership"
                );
                warn!("{engine}: {why}");
                self.engine_event(&engine, Severity::Warning, "StaleRanks", why)
                    .await;
            }
        }
    }

    /// The head on `node` as it registered, from its daemon's hello (decisions 0032 and 0036): a
    /// coordinator that starts or takes the lease knows it from nothing else.
    fn merge_heads(&self, view: &mut EngineView, node: &str, heads: &[WorkerEvent]) {
        for ev in heads {
            if ev.kind == worker_event::Kind::HeadRegistered as i32 {
                self.on_head_event(view, node, ev);
            }
        }
    }

    /// The engine's head registered on `node`, or departed from it (decision 0032): a head
    /// registering brings a down engine back; a head departing puts it down, with its cause.
    pub(super) fn on_head_event(&self, view: &mut EngineView, node: &str, ev: &WorkerEvent) {
        let down = |v: i64| {
            self.metrics
                .engines_down
                .with_label_values(&[&ev.engine])
                .set(v);
        };
        let process = || Process {
            node: node.to_owned(),
            pid: (ev.pid > 0).then_some(ev.pid),
            registered_at_ms: now_ms(),
            identity: ev.identity.clone(),
        };
        if ev.kind == worker_event::Kind::HeadRegistered as i32 {
            // one head per engine: while it is alive on another node, a second is a conflict,
            // kept for readiness and never taken for the engine's head, so a loss of the second's
            // node never reads as the engine down, whatever order the nodes started in
            if let Some(h) = view.head.as_ref().filter(|h| h.process.node != node) {
                warn!(
                    "{}: a second head registered on {node} while its head on {} is alive; refused",
                    ev.engine, h.process.node
                );
                self.metrics
                    .head_conflicts_total
                    .with_label_values(&[&ev.engine])
                    .inc();
                view.head_conflict = Some(HeadProcess {
                    process: process(),
                    launcher_pid: ev.launcher_pid,
                    host: ev.host.clone(),
                });
                return;
            }
            view.head = Some(HeadProcess {
                process: process(),
                launcher_pid: ev.launcher_pid,
                host: ev.host.clone(),
            });
            if let Some(d) = view.down.take() {
                info!(
                    "{}: back: its head registered on {node} (it was down: {})",
                    ev.engine, d.cause
                );
            }
            down(0);
        } else if view
            .head_conflict
            .as_ref()
            .is_some_and(|h| h.process.node == node)
        {
            view.head_conflict = None;
        } else if view.head.as_ref().is_some_and(|h| h.process.node == node) {
            view.head = None;
            let cause = match StopCause::try_from(ev.cause) {
                Ok(StopCause::Stopped) => "stopped",
                Ok(StopCause::Failed) => "failed",
                _ => "lost",
            };
            warn!("{}: down: its head {cause}: {}", ev.engine, ev.detail);
            view.down = Some(Down {
                at_ms: now_ms(),
                cause,
                detail: ev.detail.clone(),
            });
            down(1);
        }
    }

    /// A worker or spare of `node` registered or departed; then the engine's conditions.
    pub(super) async fn on_worker_event(&self, node: &str, ev: WorkerEvent) {
        let engine = ev.engine.clone();
        // a process that registers without asking (a pool spare, a process started from a
        // policy file, a launched spare) is the engine's first contact on its node, as an ask is
        if matches!(
            worker_event::Kind::try_from(ev.kind),
            Ok(worker_event::Kind::WorkerRegistered
                | worker_event::Kind::SpareRegistered
                | worker_event::Kind::HeadRegistered)
        ) {
            self.reached(&engine, node).await;
        }
        // a process a guard refused said why as it left: the engine's event, beside its exit
        if ev.cause == crate::proto::everserve::signals::v1::StopCause::Refused as i32 {
            warn!(
                "{engine}: rank {}'s process on {node} left, refused: {}",
                ev.rank, ev.detail
            );
            self.engine_event(
                &engine,
                crate::status::Severity::Warning,
                "ProcessRefused",
                format!("rank {}'s process on {node}: {}", ev.rank, ev.detail),
            )
            .await;
        }
        // decision 0041: a rank assigned to a process on another node is that process's; a
        // worker registering for it elsewhere holds nothing
        if ev.kind == worker_event::Kind::WorkerRegistered as i32
            && self
                .assignments(&engine)
                .iter()
                .any(|a| a.rank == ev.rank && a.node != node)
        {
            warn!(
                "{engine}: a worker for rank {} registered on {node}, but a process on another node \
                 was told to serve it; not taken as its holder",
                ev.rank
            );
            return;
        }
        let registered_for = (ev.kind == worker_event::Kind::WorkerRegistered as i32
            || (ev.kind == worker_event::Kind::SpareRegistered as i32 && !ev.any_rank))
            .then_some(ev.rank);
        let worker = ev.kind == worker_event::Kind::WorkerRegistered as i32;
        let undecided_left = (ev.kind == worker_event::Kind::UndecidedDeparted as i32)
            .then_some((ev.rank, ev.connection));
        if let Some((rank, seen)) = self.apply_worker_event(node, ev).await {
            self.departed_lost(&engine, node, rank, seen);
        }
        // decision 0041: a worker's registration spends its rank's assignment, and an undecided
        // process that leaves first releases its own
        if let Some(rank) = registered_for.filter(|_| worker) {
            self.release_assignments(&engine, "its worker registered", |a| {
                a.rank == rank && a.node == node
            })
            .await;
        }
        // the connection that asked for a stage holds every rank of it no worker took yet
        if let Some((_, connection)) = undecided_left {
            self.release_assignments(&engine, "its process left before its workers loaded", |a| {
                a.node == node && super::roles::same(a.connection.as_ref(), connection.as_ref())
            })
            .await;
        }
        self.observed(&engine).await;
    }

    /// A spare registered: kept as standing, then admitted to the pool or refused with the
    /// reason (decision 0034); its admission is derived again as what it is held to changes.
    async fn register_spare(&self, view: &mut EngineView, node: &str, ev: &WorkerEvent) {
        let key = (node.to_owned(), ev.device);
        view.standing.insert(key.clone(), ev.clone());
        let verdict = {
            let topo = self.topology.lock().await;
            self.spare_verdict(view, node, ev, &topo)
        };
        match verdict {
            Ok(()) => {
                view.refused_spares.remove(&key);
                view.processes.record(node, ev);
                place_spare(view, node, ev);
                info!(
                    "{}: spare {} on {node} device {} joins the pool",
                    ev.engine,
                    if ev.any_rank {
                        "for any rank".to_owned()
                    } else {
                        format!("for rank {}", ev.rank)
                    },
                    ev.device
                );
            }
            Err(why) => {
                warn!(
                    "{}: spare on {node} device {} not admitted: {why}",
                    ev.engine, ev.device
                );
                view.status.record(
                    now_ms(),
                    Severity::Warning,
                    "SpareRefused",
                    format!("spare on {node} device {}: {why}", ev.device),
                );
                view.refused_spares
                    .insert(key, ((!ev.any_rank).then_some(ev.rank), why));
            }
        }
    }

    /// Take `ev` from `node` into its engine's view; the rank and epoch of a serving worker that
    /// departed outside any change, which is the rank's loss.
    async fn apply_worker_event(&self, node: &str, ev: WorkerEvent) -> Option<(u32, u64)> {
        let mut engines = self.engines.lock().await;
        let withdrew_to = (ev.kind == worker_event::Kind::SpareDeparted as i32)
            .then(|| serving_elsewhere(&engines, &ev.engine, node, ev.device))
            .flatten();
        let view = engines.entry(ev.engine.clone()).or_default();
        if ev.kind == worker_event::Kind::SpareDeparted as i32 {
            spare_left(view, node, ev.device, withdrew_to);
        }
        view.world_size = view.world_size.max(ev.world_size);
        // the engine's step config, from its first rank's process that reports one, asking or
        // registering: every later one is held to it
        let rank_event = matches!(
            worker_event::Kind::try_from(ev.kind),
            Ok(worker_event::Kind::WorkerRegistered | worker_event::Kind::SpareRegistered)
        );
        if view.step.is_none() && rank_event {
            view.step = ev
                .identity
                .as_ref()
                .and_then(|i| i.step.clone())
                .filter(|s| !s.digest.is_empty());
        }
        if let Some(changed) = view.stale_departure(&ev) {
            info!(
                "{}: rank {} departed on {node} at epoch {}, before its change at epoch {changed}; the rank's holder stays",
                ev.engine, ev.rank, ev.epoch
            );
            return None;
        }
        if ev.kind == worker_event::Kind::SpareRegistered as i32 {
            self.register_spare(view, node, &ev).await;
            return None;
        }
        view.processes.record(node, &ev);
        match worker_event::Kind::try_from(ev.kind).unwrap_or(worker_event::Kind::Unspecified) {
            worker_event::Kind::HeadRegistered | worker_event::Kind::HeadDeparted => {
                self.on_head_event(view, node, &ev);
            }
            worker_event::Kind::WorkerRegistered => {
                view.members.insert(ev.rank, node.to_owned());
                view.hosts.insert(ev.rank, ev.host.clone());
                view.devices.insert(ev.rank, ev.device);
                if let Some(id) = ev.identity.as_ref() {
                    view.identities.insert(ev.rank, id.clone());
                }
                view.departed.remove(&ev.rank);
                if let Some(c) = ev.compile_cache.as_ref() {
                    view.hold_member_cache(&ev.engine, node, ev.rank, c);
                }
                if let Some(d) = self.topology.lock().await.device(node, ev.device) {
                    view.classes.insert(ev.rank, d.class.clone());
                }
                drop(engines);
                self.refresh_replication(&BTreeSet::from([ev.engine.clone()]))
                    .await;
            }
            worker_event::Kind::WorkerDeparted => {
                if view.members.get(&ev.rank).is_some_and(|n| n == node) {
                    view.members.remove(&ev.rank);
                    // the position keeps its pair: a spare taking it is held to it (0029)
                    view.departed.insert(ev.rank);
                    view.forget_compile_key_if_empty();
                    // a serving worker gone outside any change, a goodbye or none (a pod's
                    // graceful deletion, a kill, an out-of-memory exit): its rank is lost. A
                    // change taking it would have removed it from the members already, and a
                    // down engine takes no change
                    if view.down.is_none() {
                        return Some((ev.rank, ev.epoch));
                    }
                }
            }
            worker_event::Kind::SpareDeparted if ev.any_rank => {
                view.standing.remove(&(node.to_owned(), ev.device));
                view.refused_spares.remove(&(node.to_owned(), ev.device));
                view.forget_any_spare(node, ev.device);
                drop(engines);
            }
            worker_event::Kind::SpareDeparted => {
                view.standing.remove(&(node.to_owned(), ev.device));
                view.refused_spares.remove(&(node.to_owned(), ev.device));
                if view.spares.get(&ev.rank).is_some_and(|n| n == node) {
                    view.spares.remove(&ev.rank);
                    view.spare_caches.remove(&ev.rank);
                    view.forget_compile_key_if_empty();
                }
                drop(engines);
            }
            // a spare's registration was taken above, by register_spare; an undecided process's
            // departure releases its assignment, after
            worker_event::Kind::SpareRegistered
            | worker_event::Kind::UndecidedDeparted
            | worker_event::Kind::Unspecified => {}
        }
        None
    }

    /// Rank `rank`'s serving worker on `node` departed outside any change: report the rank
    /// lost, as a worker's report would be, on a task of its own, since the change it starts is
    /// delivered over the links this one's messages come on.
    fn departed_lost(&self, engine: &str, node: &str, rank: u32, seen: u64) {
        warn!(
            "{engine}: rank {rank}'s worker on {node} departed outside any change: the rank is lost"
        );
        let Some(me) = self.me.upgrade() else { return };
        let engine = engine.to_owned();
        tokio::spawn(async move {
            match me
                .ranks_reported(&engine, &[rank], seen, Cause::Fault)
                .await
            {
                Ok(_) => {}
                Err(e @ ChangeError::Stale { .. }) => info!("{engine}: report folded: {e}"),
                Err(e) => warn!("{engine}: change for rank {rank}, departed, failed: {e}"),
            }
        });
    }

    pub(super) async fn on_message(
        self: &Arc<Self>,
        node: &str,
        msg: DaemonToCoord,
        pending: &Pending,
    ) {
        match msg.kind {
            Some(daemon_to_coord::Kind::Hello(hello)) => self.merge_hello(&hello).await,
            Some(daemon_to_coord::Kind::WorkerEvent(ev)) => self.on_worker_event(node, ev).await,
            Some(daemon_to_coord::Kind::RoleAsked(asked)) => self.answer_role(node, asked).await,
            Some(daemon_to_coord::Kind::GroupId(id)) => self.group_id(node, id).await,
            Some(daemon_to_coord::Kind::LostReport(report)) => {
                let Some(lost) = report.lost else { return };
                info!(
                    "{}: ranks {:?} reported lost from node {node}: {}",
                    report.engine, lost.ranks, lost.detail
                );
                let me = Arc::clone(self);
                tokio::spawn(async move {
                    match me
                        .ranks_reported(&report.engine, &lost.ranks, lost.epoch, Cause::Fault)
                        .await
                    {
                        Ok(_) => {}
                        Err(e @ ChangeError::Stale { .. }) => {
                            info!("{}: report folded: {e}", report.engine);
                        }
                        Err(e) => warn!(
                            "{}: change for ranks {:?} failed: {e}",
                            report.engine, lost.ranks
                        ),
                    }
                });
            }
            Some(daemon_to_coord::Kind::NoticeReport(report)) => {
                if let Some(notice) = report.notice {
                    Arc::clone(self).notice(&report.engine, notice).await;
                }
            }
            // handled on the link itself, before dispatch
            Some(daemon_to_coord::Kind::Heartbeat(_) | daemon_to_coord::Kind::Bye(_)) => {}
            Some(daemon_to_coord::Kind::JoinReport(r)) => self.route_join(node, r).await,
            Some(daemon_to_coord::Kind::ReplayAsk(a)) => self.route_replay_ask(node, a).await,
            Some(daemon_to_coord::Kind::ReplayOffer(o)) => self.route_replay_offer(node, o).await,
            Some(daemon_to_coord::Kind::EngineSlot(e)) => {
                if let Some(slot) = e.slot {
                    info!(
                        "{}: slot collective {:?} by {:?} on node {node}",
                        e.engine,
                        slot.variant(),
                        slot.how()
                    );
                    self.engines
                        .lock()
                        .await
                        .entry(e.engine)
                        .or_default()
                        .slots
                        .insert(node.to_owned(), slot);
                }
            }
            Some(daemon_to_coord::Kind::NodeNotice(n)) => {
                info!(
                    "node {node}: {} says it leaves at {} ms ({}); draining",
                    n.source, n.deadline_ms, n.detail
                );
                let Some(me) = self.me.upgrade() else { return };
                let writer = format!("node {node} ({})", n.source);
                match me.drain(node, n.deadline_ms, &writer, &n.detail).await {
                    Ok(_) => info!("node {node}: drain set"),
                    Err(NodeSpecError::NoMembers(_)) => {
                        info!("node {node}: holds no member; nothing to drain");
                    }
                    Err(e) => warn!("node {node}: drain not set: {e}"),
                }
            }
            Some(daemon_to_coord::Kind::Delivered(d)) => {
                let id = d.request_id;
                answer(pending, node, id, Reply::Delivered(d)).await;
            }
            Some(daemon_to_coord::Kind::ReleaseLeaked(at)) => self.release_leaked(node, at).await,
            Some(daemon_to_coord::Kind::CoverReport(r)) => self.cover_report(node, r).await,
            Some(daemon_to_coord::Kind::StacksSent(s)) => {
                let id = s.request_id;
                answer(pending, node, id, Reply::Stacks(s)).await;
            }
            None => warn!("node {node}: empty message"),
        }
    }

    /// A join payload (decision 0024): from a client, to the daemon holding the rank's position;
    /// from the process in that position, to every daemon of the engine and the head's for their
    /// clients.
    pub(super) async fn route_join(&self, node: &str, report: JoinReport) {
        let received_at_us = now_us();
        let Some(join) = report.join else { return };
        let engine = report.engine;
        // timestamped, so a run's record brackets the control plane's share of a join's round
        // trip between the executor's own stamps (measured between a commit and the boundary on
        // that path: 15 ms to 0.3 s)
        info!(
            "{engine}: join for rank {} at epoch {} received from node {node}{}",
            join.rank,
            join.epoch,
            if report.from_client {
                " (a client's)"
            } else {
                ""
            }
        );
        self.forward_join(&engine, join, report.from_client, received_at_us)
            .await;
    }

    /// Send `join` on: a client's to the daemon holding its rank, or the one whose process was
    /// told to serve it, or the one whose spare stands for it, dropped when none does (the
    /// offering side's bounded wait says so); a process's to the daemons of the engine and the
    /// head's. Stamped in the change's record once sent.
    async fn forward_join(&self, engine: &str, join: Join, from_client: bool, received_at_us: u64) {
        let targets: Vec<(String, Outbox)> = if from_client {
            let assigned = self
                .assignments(engine)
                .into_iter()
                .find(|a| a.rank == join.rank)
                .map(|a| a.node);
            // a process told to serve the rank holds it while it loads, until its worker
            // registers (0041), ahead of any spare standing for the rank
            let holder = self.engines.lock().await.get(engine).and_then(|view| {
                view.members
                    .get(&join.rank)
                    .cloned()
                    .or(assigned)
                    .or_else(|| view.spare_node_for(join.rank).cloned())
            });
            let Some(holder) = holder else {
                warn!(
                    "{engine}: join for rank {} at epoch {} reaches no process: none holds the \
                     rank, is told to serve it, or stands for it; dropped",
                    join.rank, join.epoch
                );
                return;
            };
            let daemons = self.daemons.lock().await;
            daemons
                .get(&holder)
                .map(|d| vec![(holder.clone(), d.tx.clone())])
                .unwrap_or_default()
        } else {
            self.answer_targets(engine).await.into_iter().collect()
        };
        let mut forwarded_at_us = BTreeMap::new();
        for (to, tx) in targets {
            let _ = tx.send(CoordToDaemon {
                term: 0,
                kind: Some(coord_to_daemon::Kind::ForwardJoin(ForwardJoin {
                    engine: engine.to_owned(),
                    join: Some(join.clone()),
                    to_clients: !from_client,
                })),
            });
            info!("{engine}: join for rank {} routed to {to}", join.rank);
            forwarded_at_us.insert(to, now_us());
        }
        if let Some(view) = self.engines.lock().await.get_mut(engine) {
            view.record_join(JoinStamp {
                rank: join.rank,
                epoch: join.epoch,
                from_client,
                bytes: u64::try_from(join.payload.len()).unwrap_or(u64::MAX),
                received_at_us,
                forwarded_at_us,
            });
        }
    }

    /// The daemons a process's join goes to for their clients: every daemon of the engine, and
    /// the head's, should its node hold no rank of it.
    async fn answer_targets(&self, engine: &str) -> BTreeMap<String, Outbox> {
        let mut targets = self.daemons_of(engine).await;
        let head = self
            .engines
            .lock()
            .await
            .get(engine)
            .and_then(|v| v.head.as_ref().map(|h| h.process.node.clone()));
        if let Some(node) = head
            && let Some(d) = self.daemons.lock().await.get(&node)
        {
            targets.entry(node).or_insert_with(|| d.tx.clone());
        }
        targets
    }
}

/// Missed heartbeats after which a daemon's node is lost, unless the coordinator was told
/// otherwise (`--node-loss-misses`).
pub const DEFAULT_NODE_LOSS_MISSES: u64 = 3;

/// Take `state`'s workers on `node` as the engine's members; the ranks left out. A node
/// `behind` the engine's committed epoch holds workers of an older membership: a rank the
/// committed membership holds on this node is a member still (a node that acknowledged the
/// change and missed only the commit's message, since a commit places a rank only on a node that
/// acknowledged it); any other is not, whoever holds it now.
fn merge_workers(
    view: &mut EngineView,
    node: &str,
    state: &crate::proto::everserve::coord::v1::EngineState,
    behind: bool,
) -> Vec<u32> {
    let held_here: BTreeSet<u32> = view
        .members
        .iter()
        .filter(|(_, n)| n.as_str() == node)
        .map(|(r, _)| *r)
        .collect();
    view.members.retain(|_, n| n != node);
    let mut left_out = Vec::new();
    for &r in &state.workers {
        if behind && !held_here.contains(&r) {
            left_out.push(r);
            continue;
        }
        view.members.insert(r, node.to_owned());
        view.departed.remove(&r);
    }
    left_out
}

/// Take `state`'s spares on `node`, for one position and for any, as the engine's.
fn merge_spares(
    view: &mut EngineView,
    node: &str,
    state: &crate::proto::everserve::coord::v1::EngineState,
) {
    view.spares.retain(|_, n| n != node);
    for &r in &state.spares {
        view.spares.insert(r, node.to_owned());
    }
    view.any_spares.retain(|(n, _)| n != node);
    for &d in &state.any_spares {
        view.any_spares.insert((node.to_owned(), d));
    }
}

/// `engine`'s membership as `view` holds it at its committed epoch, as a commit a node that
/// missed one is sent: it moves no rank, and states where every rank runs.
pub(super) fn standing_commit(engine: &str, view: &EngineView) -> Committed {
    let active: Vec<u32> = view.members.keys().copied().collect();
    Committed {
        engine: engine.to_owned(),
        membership: Some(crate::proto::everserve::signals::v1::MembershipCommitted {
            epoch: view.committed_epoch,
            members: super::transition::placement(view, &active),
            active_ranks: active,
            world_size: view.world_size,
            lost: view.lost.clone(),
            ..Default::default()
        }),
        ..Committed::default()
    }
}

#[cfg(test)]
mod tests {
    use super::merge_workers;
    use crate::coord::view::EngineView;
    use crate::proto::everserve::coord::v1::EngineState;

    #[test]
    fn a_node_behind_the_committed_epoch_keeps_only_the_ranks_it_holds_there() {
        let state = EngineState {
            workers: vec![0, 4, 5],
            ..EngineState::default()
        };
        // acknowledged the change and missed only its commit's message: rank 4 and 5 are held
        // here, rank 0 moved to node-c
        let mut view = EngineView::default();
        view.members.insert(0, "node-c".into());
        view.members.insert(4, "node-b".into());
        view.members.insert(5, "node-b".into());
        let left_out = merge_workers(&mut view, "node-b", &state, true);
        assert_eq!(left_out, [0]);
        assert_eq!(view.members.get(&0).map(String::as_str), Some("node-c"));
        assert_eq!(view.members.get(&4).map(String::as_str), Some("node-b"));
        // never heard the change, and never known: none of its ranks is a member
        let mut view = EngineView::default();
        view.members.insert(0, "node-c".into());
        let left_out = merge_workers(&mut view, "node-b", &state, true);
        assert_eq!(left_out, [0, 4, 5]);
        assert_eq!(view.members.len(), 1);
        // a node at the committed epoch takes its ranks as it names them
        let mut view = EngineView::default();
        assert!(merge_workers(&mut view, "node-b", &state, false).is_empty());
        assert_eq!(view.members.len(), 3);
    }
}
