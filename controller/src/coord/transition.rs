// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The transition machine: a change claimed, delivered, acknowledged and committed, or abandoned with the engine down (layer 5, decisions 0014, 0032, 0033).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::clock::now_ms;
use crate::clock::now_us;
use crate::policy;
use crate::policy::cover::{Member, set_key};
use crate::proto::everserve::coord::v1::{
    Abandon, Committed, CoordToDaemon, Deliver, ForwardChange, ForwardNotice, coord_to_daemon,
};
use crate::proto::everserve::policy::v1::Engine as EngineSpec;
use crate::proto::everserve::signals::v1::{
    Ack, Change, MembershipCommitted, Notice, Outcome, RankPlacement, Take,
};
use crate::proto::everserve::spec::v1::{ChangeLog, NodeDelivery};
use crate::readiness::{AckStamp, LastChange};
use crate::status::Severity;

use super::outbox::Outbox;
use super::view::{Down, EngineView};
use super::{Cause, ChangeError, Coordinator, DELIVERY_TIMEOUT, Reply};
use crate::topology::{Position, Topology};

impl Coordinator {
    /// One rank of `engine` is gone; see [`Coordinator::ranks_lost`].
    ///
    /// # Errors
    ///
    /// See [`ChangeError`].
    pub async fn rank_lost(
        &self,
        engine: &str,
        dead: u32,
        cause: Cause,
    ) -> Result<MembershipCommitted, ChangeError> {
        self.report_loss(engine, &[dead], None, cause).await
    }

    /// A set of ranks of `engine` is gone (one rank, or every rank of a node): choose the
    /// outcome once for the set, claim one epoch, drive the change through every node that holds
    /// a survivor or a spare, and commit the membership to every daemon of the engine.
    ///
    /// # Errors
    ///
    /// See [`ChangeError`]. On an error the membership is not committed.
    pub async fn ranks_lost(
        &self,
        engine: &str,
        lost: &[u32],
        cause: Cause,
    ) -> Result<MembershipCommitted, ChangeError> {
        self.report_loss(engine, lost, None, cause).await
    }

    /// `lost` of `engine` were reported gone by a process that saw them at epoch `seen`: as
    /// [`Coordinator::ranks_lost`], except that a rank changed after `seen` is dropped from the
    /// report. One death is seen several times, by the slot kernel on every survivor and by the
    /// executor's process sentinel; the first report moves it and the others fold (decision 0021).
    ///
    /// # Errors
    ///
    /// [`ChangeError::Stale`] when every reported rank has changed since `seen`; otherwise as
    /// [`Coordinator::ranks_lost`].
    pub async fn ranks_reported(
        &self,
        engine: &str,
        lost: &[u32],
        seen: u64,
        cause: Cause,
    ) -> Result<MembershipCommitted, ChangeError> {
        self.report_loss(engine, lost, Some(seen), cause).await
    }

    /// One transition for `lost` of `engine`, reported first at `reported_at_us`, `faulted` the
    /// ranks a fault report named: the outcome policy's answer driven to its commit or its
    /// abandon, counted and timed. Its cause is a fault if any rank faulted. The controller calls
    /// it once per fold; nothing else does.
    pub(super) async fn change(
        &self,
        engine: &str,
        lost: &[u32],
        faulted: &BTreeSet<u32>,
        reported_at_us: u64,
    ) -> Result<MembershipCommitted, ChangeError> {
        let cause = if faulted.is_empty() {
            Cause::Notice
        } else {
            Cause::Fault
        };
        *self.in_flight.lock().await += 1;
        self.metrics.changes_in_flight.inc();
        let started = std::time::Instant::now();
        let result = self
            .drive_change(engine, lost, faulted, reported_at_us)
            .await;
        self.metrics.changes_in_flight.dec();
        *self.in_flight.lock().await -= 1;
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let summary = result
            .as_ref()
            .map(|(outcome, committed)| (*outcome, committed.epoch))
            .map_err(Clone::clone);
        self.change_event(engine, lost, &summary, elapsed_ms).await;
        let (outcome, verdict) = match &result {
            Ok((outcome, _)) => (outcome_label(*outcome), "committed"),
            Err(ChangeError::Rebuild(_)) => ("rebuild", "refused_rebuild"),
            Err(ChangeError::Unknown(_)) => ("none", "unknown_rank"),
            Err(ChangeError::Unheard { .. }) => ("none", "unheard_ranks"),
            Err(ChangeError::Survivor { .. }) => ("unknown", "survivor_error"),
            Err(ChangeError::Departed { .. }) => ("unknown", "survivor_departed"),
            Err(ChangeError::Down { .. }) => ("none", "engine_down"),
            Err(ChangeError::Node { .. }) => ("unknown", "no_delivery"),
            Err(ChangeError::Epoch { .. }) => ("unknown", "epoch_mismatch"),
            Err(ChangeError::Stale { .. }) => ("none", "folded"),
            Err(ChangeError::Domain { .. }) => ("unknown", "domain"),
            Err(ChangeError::Uncovered { .. }) => ("none", "refused_uncovered"),
            Err(ChangeError::Deposed { .. }) => ("none", "deposed"),
            Err(ChangeError::LeaseLost { .. }) => ("none", "lease_lost"),
            Err(ChangeError::Unlogged { .. }) => ("unknown", "unlogged"),
            Err(ChangeError::NotServing { .. }) => ("none", "head_not_serving"),
        };
        self.metrics
            .changes_total
            .with_label_values(&[engine, cause.label(), outcome, verdict])
            .inc();
        if result.is_ok() {
            let elapsed = started.elapsed();
            self.metrics
                .change_seconds
                .with_label_values(&[cause.label(), outcome])
                .observe(elapsed.as_secs_f64());
            if let Some(view) = self.engines.lock().await.get_mut(engine) {
                view.last_pause_ms.insert(
                    outcome.to_owned(),
                    u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
                );
            }
        }
        self.observed(engine).await;
        result.map(|(_, committed)| committed)
    }

    /// Drive the change for `lost`, `faulted` the ranks a fault report named; on success, the
    /// outcome taken and the committed membership.
    pub(super) async fn drive_change(
        &self,
        engine: &str,
        reported: &[u32],
        faulted: &BTreeSet<u32>,
        reported_at_us: u64,
    ) -> Result<(Outcome, MembershipCommitted), ChangeError> {
        // the change's moments on this clock, for the record: the first report it folded, the
        // delivery, the last acknowledgement, the commit
        let decided = self.claim(engine, reported).await?;
        let (epoch, outcome) = (decided.epoch, decided.outcome);
        let take: Vec<Take> = decided
            .by_node
            .values()
            .flat_map(|(_, _, takes)| takes.iter().cloned())
            .collect();
        // a rank leaves alive unless a report says it faulted: a drained rank, or one moved whole
        // with its group (decision 0022)
        let departing: Vec<u32> = decided
            .lost
            .iter()
            .copied()
            .filter(|r| !faulted.contains(r))
            .collect();
        let delivered_at_us = now_us();
        let change = Change {
            lost: decided.lost.clone(),
            outcome: outcome as i32,
            epoch,
            delivered_at_us,
            take,
            departing,
        };
        // the store holds the change before any process hears of it (decision 0038): a successor
        // finishes it rather than claiming another under its epoch
        let log = decided.log(&change);
        if let Err(e) = self
            .keep_status(engine, |s| s.change = Some(log.clone()))
            .await
        {
            if let Some(view) = self.engines.lock().await.get_mut(engine)
                && view.change.as_ref().is_some_and(|(at, _)| *at == epoch)
            {
                view.change = None;
            }
            return Err(ChangeError::Unlogged {
                epoch,
                reason: e.to_string(),
            });
        }
        if let Some(view) = self.engines.lock().await.get_mut(engine) {
            apply_claim(view, &log);
        }
        self.metrics.change_decision_seconds.observe(
            Duration::from_micros(delivered_at_us.saturating_sub(reported_at_us)).as_secs_f64(),
        );
        self.finish_change(engine, &log, reported_at_us).await
    }

    /// Deliver the change `log` records, which the engine's view has taken, and commit it once
    /// every acknowledgement is in; on success, the outcome and the committed membership. The
    /// one way a change goes out, whether this coordinator claimed it or took it from the log of
    /// one that died (decision 0038).
    pub(super) async fn finish_change(
        &self,
        engine: &str,
        log: &ChangeLog,
        reported_at_us: u64,
    ) -> Result<(Outcome, MembershipCommitted), ChangeError> {
        let change = log.change.clone().unwrap_or_default();
        let (epoch, delivered_at_us, world_size) =
            (change.epoch, change.delivered_at_us, log.world_size);
        let outcome = Outcome::try_from(change.outcome).unwrap_or(Outcome::Unspecified);
        let shrunk = log.shrunk.clone();
        let lost = change.lost.as_slice();
        self.engine_event(
            engine,
            Severity::Normal,
            "ChangeStarted",
            format!(
                "epoch {epoch}: {} for ranks {lost:?}",
                outcome.as_str_name()
            ),
        )
        .await;
        self.observed(engine).await;
        if outcome == Outcome::Replace {
            self.stage_restores(engine, lost).await;
        }
        let active: Vec<u32> = (0..world_size).filter(|r| !shrunk.contains(r)).collect();
        self.forward_to_departing(engine, &change).await;
        let all = self.deliver_all(engine, &change, by_node_of(log)).await;
        let delivered = match all {
            Ok(acks) => self
                .survivors_present(engine, &active, lost)
                .await
                .map(|()| acks),
            Err(e) => Err(e),
        };
        let acks = match delivered {
            Ok(acks) => acks,
            Err(e) => {
                self.unstage_restores(engine, lost, epoch).await;
                if let Some(view) = self.engines.lock().await.get_mut(engine) {
                    view.unassign_any_spares(lost);
                    view.change = None;
                }
                self.abandon(engine, epoch, &e).await;
                return Err(e);
            }
        };
        let acknowledged_at_us = now_us();
        let committed = MembershipCommitted {
            epoch,
            active_ranks: active,
            world_size,
            pause_ms: 0,
            lost: shrunk,
            reported_at_us,
            delivered_at_us,
            acknowledged_at_us,
            committed_at_us: now_us(),
            acks,
            members: Vec::new(),
        };
        let committed = match self.commit(engine, lost, outcome, committed).await {
            Ok(c) => c,
            Err(e) => {
                self.abandon(engine, epoch, &e).await;
                return Err(e);
            }
        };
        info!("{engine}: membership committed at epoch {epoch} ({outcome:?}, lost {lost:?})");
        Ok((outcome, committed))
    }

    /// A change of `engine` at `epoch` failed after its delivery (decision 0032): the processes
    /// that applied it hold a membership that will not commit, so the engine is down, unless its
    /// head's departure already said why, and every daemon of the engine takes `epoch` as spent
    /// and tells the engine's processes on its node.
    pub(super) async fn abandon(&self, engine: &str, epoch: u64, error: &ChangeError) {
        let detail = error.to_string();
        if let Some(view) = self.engines.lock().await.get_mut(engine)
            && view.down.is_none()
        {
            view.down = Some(Down {
                at_ms: now_ms(),
                cause: "change_failed",
                detail: detail.clone(),
            });
        }
        self.metrics
            .engines_down
            .with_label_values(&[engine])
            .set(1);
        warn!(
            "{engine}: the change at epoch {epoch} failed after its delivery; the engine is down: {detail}"
        );
        let abandon = CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::Abandon(Abandon {
                engine: engine.to_owned(),
                epoch,
                detail,
            })),
        };
        // every node with a process of the engine: its members', its spares' (those standing
        // for any position and taking included) and its head's
        let nodes: BTreeSet<String> = self
            .engines
            .lock()
            .await
            .get(engine)
            .map(|v| {
                v.members
                    .values()
                    .chain(v.spares.values())
                    .chain(v.any_spares.iter().map(|(n, _)| n))
                    .chain(v.taking.values().map(|(n, _)| n))
                    .chain(v.head.iter().map(|h| &h.process.node))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let links: Vec<Outbox> = {
            let daemons = self.daemons.lock().await;
            nodes
                .iter()
                .filter_map(|n| daemons.get(n).map(|l| l.tx.clone()))
                .collect()
        };
        for tx in links {
            let _ = tx.send(abandon.clone());
        }
    }

    /// Every survivor of a change (an active rank the change does not replace) still has its
    /// process registered once the last acknowledgement is in. An acknowledgement says a survivor
    /// applied the change, not that it is still there to serve under it: a process can
    /// acknowledge and then exit while a spare's take is still running.
    pub(super) async fn survivors_present(
        &self,
        engine: &str,
        active: &[u32],
        lost: &[u32],
    ) -> Result<(), ChangeError> {
        let engines = self.engines.lock().await;
        let Some(view) = engines.get(engine) else {
            return Ok(());
        };
        if let Some(d) = &view.down {
            return Err(ChangeError::Down {
                cause: d.cause,
                detail: d.detail.clone(),
            });
        }
        let ranks: Vec<u32> = active
            .iter()
            .copied()
            .filter(|r| !lost.contains(r) && !view.members.contains_key(r))
            .collect();
        if ranks.is_empty() {
            Ok(())
        } else {
            Err(ChangeError::Departed { ranks })
        }
    }

    /// Decide the change for `reported` under the lock, on a copy of the engine's view, and mark
    /// its epoch claimed on the view itself so no other change starts; the change, which the view
    /// takes only once the store holds its log (decision 0038).
    pub(super) async fn claim(
        &self,
        engine: &str,
        reported: &[u32],
    ) -> Result<Decided, ChangeError> {
        let placement_independent = self
            .engine_spec(engine)
            .is_some_and(|e| e.placement_independent_model);
        let exact = self.exact(engine);
        let first = reported.first().copied().ok_or(ChangeError::Unknown(0))?;
        self.refuse_if_deposed()?;
        // read before the engines: a cordoned node's spares take no position (decision 0036)
        let cordoned = self.cordoned().await;
        let mut engines = self.engines.lock().await;
        let real = engines.get_mut(engine).ok_or(ChangeError::Unknown(first))?;
        if let Some(d) = &real.down {
            return Err(ChangeError::Down {
                cause: d.cause,
                detail: d.detail.clone(),
            });
        }
        // a head that did not say it serves changes is told none: a head whose engine core still
        // steps as vLLM's own dies at a change's first delivery
        if let Some(why) = not_serving(real) {
            return Err(why);
        }
        // decided on a copy: the view takes the change once it is logged, from the log
        let mut scratch = real.clone();
        let view = &mut scratch;
        // the controller folded the reports and dropped what changed since each saw it
        let lost = reported.to_vec();
        // a replacement outside its captured group's domain moves the whole group (0022)
        let lost = match self.engine_spec(engine) {
            Some(spec) => {
                let topo = self.topology.lock().await;
                widen_to_domain(spec, view, &topo, lost)?
            }
            None => lost,
        };
        // REPLACE needs a spare for every lost rank: its own, or one standing for any position
        // (decision 0030), registered and in the pool
        let chosen = policy::choose_outcome(
            &self.policy,
            view.spares_cover(&lost, &cordoned),
            placement_independent,
        );
        if chosen == Outcome::Rebuild {
            return Err(ChangeError::Rebuild(first));
        }
        // with cover planned, a swap goes only through a covered set, which names its spares
        // (decision 0035)
        let through = self.cover_gate(view, &lost)?;
        let outcome = match &through {
            Some(set) if set.iter().any(|m| matches!(m, Member::Spare { .. })) => Outcome::Replace,
            Some(_) => Outcome::Shrink,
            None => chosen,
        };
        view.change_cover = Some(match &through {
            Some(set) => format!("through covered set {}", set_key(set)),
            None if view.cover.plan.is_some() => "formed in the change".to_owned(),
            None => String::new(),
        });
        for &r in &lost {
            if !view.members.contains_key(&r)
                && !view.lost.contains(&r)
                && !view.departed.remove(&r)
            {
                return Err(ChangeError::Unknown(r));
            }
        }
        view.change_from = lost
            .iter()
            .filter_map(|r| view.members.get(r).cloned())
            .collect();
        for r in &lost {
            view.members.remove(r);
        }
        view.epoch += 1;
        view.change = Some((view.epoch, lost.clone()));
        for &r in &lost {
            view.changed_at.insert(r, view.epoch);
        }
        if outcome == Outcome::Shrink {
            view.lost.extend(lost.iter().copied());
        }
        let mut by_node: ByNode = BTreeMap::new();
        for (r, n) in &view.members {
            by_node.entry(n.clone()).or_default().0.push(*r);
        }
        // the head hears every change as it is claimed (its node's daemon hands a delivery to the
        // engine's clients), whatever its node still holds: a head beside ranks that all depart,
        // or apart from its GPUs, holds no member, and a core that never heard the claim cannot
        // apply its commit (decision 0038)
        if let Some(h) = view.head.as_ref() {
            by_node.entry(h.process.node.clone()).or_default();
        }
        if outcome == Outcome::Replace {
            match &through {
                // the covered set's own spares, never the pool's first
                Some(set) => view.assign_from_set(&lost, set),
                // spares_cover held, so every lost rank has its own spare or gets one from the
                // pool
                None => view.assign_any_spares(&lost, &cordoned),
            }
            for r in &lost {
                if let Some(n) = view.spares.get(r).filter(|n| !cordoned.contains(*n)) {
                    by_node.entry(n.clone()).or_default().1.push(*r);
                } else if let Some((n, d)) = view.taking.get(r) {
                    by_node.entry(n.clone()).or_default().2.push(Take {
                        rank: *r,
                        node: n.clone(),
                        device: *d,
                        // held to the position's pair under the exact class only (0030)
                        compile_cache: exact.then(|| view.caches.get(r).cloned()).flatten(),
                    });
                }
            }
        }
        let decided = Decided::of(view, lost, outcome, by_node);
        real.change = Some((decided.epoch, decided.lost.clone()));
        Ok(decided)
    }

    /// The covered set a change of `view` through the loss of `lost` must use, or None when cover
    /// is not in force for it: the policy's cover is off, no plan was delivered yet, or the
    /// policy lets a change form its set on the critical path.
    ///
    /// # Errors
    ///
    /// [`ChangeError::Uncovered`] when cover is in force and no covered set answers the loss.
    fn cover_gate(
        &self,
        view: &EngineView,
        lost: &[u32],
    ) -> Result<Option<Vec<Member>>, ChangeError> {
        let Some(cover) = self.policy.cover.as_ref().filter(|c| c.enabled) else {
            return Ok(None);
        };
        if view.cover.plan.is_none() {
            return Ok(None);
        }
        if let Some(set) = view.cover.covered_answer(lost) {
            return Ok(Some(set.set.clone()));
        }
        if cover.form_on_critical_path {
            return Ok(None);
        }
        let planned = view
            .cover
            .sets
            .iter()
            .find(|s| policy::cover::answers(&s.set, lost));
        let why = match planned {
            Some(s) => format!("set {} is planned and not yet formed", set_key(&s.set)),
            None => view
                .cover
                .uncovered
                .iter()
                .find(|u| u.lost == lost)
                .map_or_else(
                    || "the plan holds no set for this loss".to_owned(),
                    |u| format!("the plan holds no set for it: {}", u.why),
                ),
        };
        Err(ChangeError::Uncovered {
            ranks: lost.to_vec(),
            why,
        })
    }

    /// Send the change to every node that holds a survivor, then collect every delivery.
    ///
    /// Every delivery goes out before any is awaited: a worker's shrink is a collective that
    /// completes only when all survivors take part.
    pub(super) async fn deliver_all(
        &self,
        engine: &str,
        change: &Change,
        by_node: ByNode,
    ) -> Result<Vec<Ack>, ChangeError> {
        let epoch = change.epoch;
        let mut all = Vec::new();
        let mut pending = Vec::new();
        {
            let daemons = self.daemons.lock().await;
            for (node, (ranks, to_spares, take)) in by_node {
                let link = daemons.get(&node).ok_or_else(|| ChangeError::Node {
                    node: node.clone(),
                    reason: "not connected".into(),
                })?;
                let request_id = self.next_request.fetch_add(1, Ordering::Relaxed);
                let (tx, rx) = oneshot::channel();
                link.pending.lock().await.insert(request_id, tx);
                let deliver = CoordToDaemon {
                    term: 0,
                    kind: Some(coord_to_daemon::Kind::Deliver(Deliver {
                        request_id,
                        engine: engine.to_owned(),
                        change: Some(change.clone()),
                        to_ranks: ranks,
                        to_spares,
                        take,
                    })),
                };
                if link.tx.send(deliver).is_err() {
                    return Err(ChangeError::Node {
                        node,
                        reason: "link closed".into(),
                    });
                }
                pending.push((node, rx));
            }
        }
        for (node, rx) in pending {
            let reply = tokio::time::timeout(DELIVERY_TIMEOUT, rx)
                .await
                .map_err(|_| ChangeError::Node {
                    node: node.clone(),
                    reason: "timed out".into(),
                })?
                .map_err(|_| ChangeError::Node {
                    node: node.clone(),
                    reason: "link closed".into(),
                })?;
            let Reply::Delivered(delivered) = reply else {
                return Err(ChangeError::Node {
                    node,
                    reason: "answered a delivery with something else".into(),
                });
            };
            if !delivered.error.is_empty() {
                return Err(ChangeError::Node {
                    node,
                    reason: delivered.error,
                });
            }
            for ack in delivered.acks {
                if !ack.error.is_empty() {
                    return Err(ChangeError::Survivor {
                        rank: ack.rank,
                        error: ack.error,
                    });
                }
                if ack.epoch != epoch {
                    return Err(ChangeError::Epoch {
                        rank: ack.rank,
                        got: ack.epoch,
                        expected: epoch,
                    });
                }
                all.push(ack);
            }
        }
        Ok(all)
    }

    /// After every acknowledgement: promote the spare to the dead rank's place under REPLACE in
    /// the view, tell every daemon of the engine, and the committed-membership channel.
    pub(super) async fn commit(
        &self,
        engine: &str,
        lost: &[u32],
        outcome: Outcome,
        mut committed: MembershipCommitted,
    ) -> Result<MembershipCommitted, ChangeError> {
        let targets = self.commit_targets(engine).await;
        let exact = self.exact(engine);
        let mut take = Vec::new();
        {
            let mut engines = self.engines.lock().await;
            if let Some(view) = engines.get_mut(engine) {
                view.committed_epoch = view.committed_epoch.max(committed.epoch);
                view.change = None;
                let joins = std::mem::take(&mut view.joins)
                    .into_iter()
                    .filter(|j| j.epoch == committed.epoch)
                    .collect();
                view.last_change = Some(LastChange {
                    epoch: committed.epoch,
                    outcome: outcome_label(outcome).to_owned(),
                    lost: lost.to_vec(),
                    reported_at_us: committed.reported_at_us,
                    delivered_at_us: committed.delivered_at_us,
                    acknowledged_at_us: committed.acknowledged_at_us,
                    committed_at_us: committed.committed_at_us,
                    cover: view.change_cover.take().unwrap_or_default(),
                    acks: committed
                        .acks
                        .iter()
                        .map(|a| AckStamp {
                            rank: a.rank,
                            received_at_us: a.received_at_us,
                        })
                        .collect(),
                    joins,
                });
            }
            if let Some(view) = engines.get_mut(engine)
                && outcome == Outcome::Replace
            {
                for r in lost {
                    if let Some(node) = view.spares.remove(r) {
                        // a spare that took a position stands no longer: it holds the rank, on
                        // the device it registered with
                        let at = view
                            .standing
                            .iter()
                            .find(|((n, _), ev)| n == &node && !ev.any_rank && ev.rank == *r)
                            .map(|(at, _)| at.clone());
                        if let Some(at) = at {
                            view.devices.insert(*r, at.1);
                            if let Some(ev) = view.standing.remove(&at) {
                                view.hosts.insert(*r, ev.host);
                            }
                        }
                        view.processes.promote(*r, None);
                        view.members.insert(*r, node);
                    } else if let Some((node, device)) = view.taking.remove(r) {
                        if let Some(ev) = view.standing.remove(&(node.clone(), device)) {
                            view.hosts.insert(*r, ev.host);
                        }
                        view.devices.insert(*r, device);
                        view.processes.promote(*r, Some(&(node.clone(), device)));
                        take.push(Take {
                            rank: *r,
                            node: node.clone(),
                            device,
                            compile_cache: exact.then(|| view.caches.get(r).cloned()).flatten(),
                        });
                        view.members.insert(*r, node);
                    }
                }
            }
            if let Some(view) = engines.get(engine) {
                committed.members = placement(view, &committed.active_ranks);
            }
        }
        let sent = Committed {
            engine: engine.to_owned(),
            membership: Some(committed.clone()),
            outcome: outcome as i32,
            lost: lost.to_vec(),
            take,
        };
        // the store holds the commit before any daemon hears it (decision 0038): a successor
        // sends it on to every daemon below its epoch rather than deciding the change again
        self.log_commit(engine, &sent).await?;
        let msg = CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::Committed(sent)),
        };
        for tx in targets.into_values() {
            let _ = tx.send(msg.clone());
        }
        let _ = self.committed_tx.send(committed.clone());
        // the spares restored before they acknowledged, so the staged replications are whole:
        // the replica stops carrying the rank to its new holder, which is told to expect nobody
        self.unstage_restores(engine, lost, committed.epoch).await;
        // a replaced rank now lives on its spare's node: targets follow it, the old holder's
        // store is told it replicates the rank nowhere
        self.refresh_replication(&BTreeSet::from([engine.to_owned()]))
            .await;
        Ok(committed)
    }

    /// Write `sent` into its change's log in the store, trying again twice, 100 and 200 ms
    /// apart, while the store fails for a reason other than a newer term.
    ///
    /// # Errors
    ///
    /// [`ChangeError::Unlogged`] when the store did not keep it.
    async fn log_commit(&self, engine: &str, sent: &Committed) -> Result<(), ChangeError> {
        let epoch = sent.membership.as_ref().map_or(0, |m| m.epoch);
        let mut wait = Duration::from_millis(100);
        let mut attempts = 0;
        loop {
            let kept = self
                .keep_status(engine, |s| {
                    if let Some(log) = s.change.as_mut()
                        && log.change.as_ref().is_some_and(|c| c.epoch == epoch)
                    {
                        log.committed = Some(sent.clone());
                    }
                })
                .await;
            match kept {
                Ok(()) => return Ok(()),
                Err(e) if attempts < 2 && !e.is_fenced() => {
                    warn!("{engine}: the commit at epoch {epoch} not logged yet: {e}");
                    tokio::time::sleep(wait).await;
                    wait *= 2;
                    attempts += 1;
                }
                Err(e) => {
                    return Err(ChangeError::Unlogged {
                        epoch,
                        reason: e.to_string(),
                    });
                }
            }
        }
    }

    /// Pass `change` to the processes of its departing ranks (protocol 9): still running, they
    /// leave at the change, refusing their groups as a survivor does, then draining and releasing.
    /// Nothing waits on them; they are outside the change's barrier. A node whose daemon is not
    /// linked learns they left from the commit's membership.
    async fn forward_to_departing(&self, engine: &str, change: &Change) {
        if change.departing.is_empty() {
            return;
        }
        let by_node: BTreeMap<String, Vec<u32>> = {
            let engines = self.engines.lock().await;
            let Some(view) = engines.get(engine) else {
                return;
            };
            let mut by_node: BTreeMap<String, Vec<u32>> = BTreeMap::new();
            // the processes still registered as the ranks: the claim already took the ranks out
            // of the membership
            for r in &change.departing {
                if let Some(p) = view.processes.members.get(r) {
                    by_node.entry(p.node.clone()).or_default().push(*r);
                }
            }
            by_node
        };
        let daemons = self.daemons.lock().await;
        for (node, ranks) in by_node {
            let Some(link) = daemons.get(&node) else {
                warn!("{engine}: departing ranks {ranks:?} on {node}: its daemon is not linked");
                continue;
            };
            let forward = CoordToDaemon {
                term: 0,
                kind: Some(coord_to_daemon::Kind::ForwardChange(ForwardChange {
                    engine: engine.to_owned(),
                    change: Some(change.clone()),
                    to_ranks: ranks,
                })),
            };
            let _ = link.tx.send(forward);
        }
    }

    /// Every daemon a commit of `engine` goes to: those with a member or a spare of it after the
    /// change, those the change took ranks from, the head's, and the rest of its audience
    /// (`audience`), a node whose members were all lost without notice among them. A node left
    /// holding nothing of the engine still hears the change that emptied it, which is how its
    /// node role learns its positions left, and its daemon's epoch stays the engine's.
    pub(super) async fn commit_targets(&self, engine: &str) -> BTreeMap<String, Outbox> {
        let mut targets = self.daemons_of(engine).await;
        // and the engine's audience: a node whose every member was lost without notice took no
        // part in the change, and a daemon that never hears a commit stays at its epoch and
        // refuses the next change through it as stale
        for (node, tx) in self.audience_of(engine).await {
            targets.entry(node).or_insert(tx);
        }
        let extra: BTreeSet<String> = self
            .engines
            .lock()
            .await
            .get_mut(engine)
            .map(|v| {
                let mut nodes = std::mem::take(&mut v.change_from);
                nodes.extend(v.head.as_ref().map(|h| h.process.node.clone()));
                nodes
            })
            .unwrap_or_default();
        let daemons = self.daemons.lock().await;
        for node in extra {
            if let Some(d) = daemons.get(&node) {
                targets.entry(node).or_insert_with(|| d.tx.clone());
            }
        }
        targets
    }

    /// A rank of `engine` will leave at the notice's deadline: tell every daemon of the engine
    /// now, and at the deadline drive the same change a fault would, while the rank is still
    /// alive, so the engine never observes a fault.
    pub async fn notice(self: Arc<Self>, engine: &str, notice: Notice) {
        let deadline_ms = notice.deadline_ms;
        let rank = notice.rank;
        self.forward_notice(engine, notice).await;
        self.change_at(engine.to_owned(), vec![rank], deadline_ms)
            .await;
    }

    /// Tell every daemon of `engine` about `notice`, warning when its lead is below the policy's.
    pub(super) async fn forward_notice(&self, engine: &str, notice: Notice) {
        let lead = notice.deadline_ms.saturating_sub(now_ms());
        if lead < i64::from(self.policy.notice_min_ms) {
            warn!(
                "{engine}: notice for rank {} gives {lead} ms, below the policy's {} ms",
                notice.rank, self.policy.notice_min_ms
            );
        }
        let forward = CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::ForwardNotice(ForwardNotice {
                engine: engine.to_owned(),
                notice: Some(notice),
            })),
        };
        for tx in self.daemons_of(engine).await.into_values() {
            let _ = tx.send(forward.clone());
        }
        info!(
            "{engine}: notice for rank {} at {} ms",
            notice.rank, notice.deadline_ms
        );
    }

    /// At `deadline_ms`, drive one change for `ranks` of `engine` as a notice. The deadline is
    /// armed once: the same notice heard again (a daemon's hello after a relink carries its
    /// node's standing notices) changes nothing. A shutdown does not wait for it: the change
    /// counts as in flight once it runs, and a successor arms the deadline again from the
    /// hellos.
    pub(super) async fn change_at(
        self: Arc<Self>,
        engine: String,
        ranks: Vec<u32>,
        deadline_ms: i64,
    ) {
        let key = (engine.clone(), ranks.clone(), deadline_ms);
        if !self.armed.lock().await.insert(key.clone()) {
            info!("{engine}: the change at {deadline_ms} ms for ranks {ranks:?} is already armed");
            return;
        }
        let lead = deadline_ms.saturating_sub(now_ms());
        let wait = Duration::from_millis(u64::try_from(lead.max(0)).unwrap_or(0));
        tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            self.armed.lock().await.remove(&key);
            if let Err(e) = self.ranks_lost(&engine, &ranks, Cause::Notice).await {
                warn!("{engine}: change at the notice deadline for ranks {ranks:?} failed: {e}");
            }
        });
    }
}

/// What a claim decides: the ranks lost (widened to their groups where 0022 requires), the new
/// epoch, the outcome, per node the ranks to deliver to, the spares preloaded for lost ranks, and
/// the spares standing for any position that take one (decision 0030); the world size; and the
/// ranks shrunk away.
/// A change decided and not yet logged: what the coordinator writes to the store before anything
/// goes out, and what the engine's view takes from that log (decision 0038).
pub(super) struct Decided {
    /// The ranks the change takes, and the epoch it claims.
    pub lost: Vec<u32>,
    pub epoch: u64,
    pub outcome: Outcome,
    /// What each node is sent.
    pub by_node: ByNode,
    pub world_size: u32,
    /// The ranks lost under SHRINK once the change commits.
    pub shrunk: Vec<u32>,
    /// The nodes the change takes ranks from.
    pub from_nodes: Vec<String>,
    /// How the change used cover, for its record.
    pub cover: String,
    /// The spares standing for any position that take the lost ranks.
    pub taking: Vec<Take>,
}

impl Decided {
    /// The change `view` took when it claimed `lost` with `outcome`, sent as `by_node` says.
    fn of(view: &EngineView, lost: Vec<u32>, outcome: Outcome, by_node: ByNode) -> Self {
        Self {
            epoch: view.epoch,
            outcome,
            by_node,
            world_size: view.world_size,
            shrunk: view.lost.clone(),
            from_nodes: view.change_from.iter().cloned().collect(),
            cover: view.change_cover.clone().unwrap_or_default(),
            taking: lost
                .iter()
                .filter_map(|r| {
                    view.taking.get(r).map(|(node, device)| Take {
                        rank: *r,
                        node: node.clone(),
                        device: *device,
                        compile_cache: None,
                    })
                })
                .collect(),
            lost,
        }
    }

    /// The log of this change, delivered as `change`.
    pub(super) fn log(&self, change: &Change) -> ChangeLog {
        ChangeLog {
            change: Some(change.clone()),
            deliveries: self
                .by_node
                .iter()
                .map(|(node, (to_ranks, to_spares, take))| NodeDelivery {
                    node: node.clone(),
                    to_ranks: to_ranks.clone(),
                    to_spares: to_spares.clone(),
                    take: take.clone(),
                })
                .collect(),
            world_size: self.world_size,
            shrunk: self.shrunk.clone(),
            from_nodes: self.from_nodes.clone(),
            cover: self.cover.clone(),
            taking: self.taking.clone(),
            committed: None,
        }
    }
}

/// What each node is sent of the change `log` names.
pub(super) fn by_node_of(log: &ChangeLog) -> ByNode {
    log.deliveries
        .iter()
        .map(|d| {
            (
                d.node.clone(),
                (d.to_ranks.clone(), d.to_spares.clone(), d.take.clone()),
            )
        })
        .collect()
}

/// Make `view` the engine's after the claim `log` records: the lost ranks leave the membership,
/// the epoch is the change's, and the spares standing for any position that take a rank are
/// assigned it. The one way a view takes a claim, the live coordinator's and a successor's.
pub(super) fn apply_claim(view: &mut EngineView, log: &ChangeLog) {
    let Some(change) = log.change.as_ref() else {
        return;
    };
    for r in &change.lost {
        view.members.remove(r);
        view.departed.remove(r);
        view.changed_at.insert(*r, change.epoch);
    }
    view.epoch = view.epoch.max(change.epoch);
    view.change = Some((change.epoch, change.lost.clone()));
    view.lost.clone_from(&log.shrunk);
    for t in &log.taking {
        let at = (t.node.clone(), t.device);
        view.any_spares.remove(&at);
        view.taking.insert(t.rank, at);
    }
    view.change_from = log.from_nodes.iter().cloned().collect();
    view.change_cover = Some(log.cover.clone());
}

/// Per node, what a change goes to: the serving ranks there, the spares preloaded for lost ranks
/// there, and the spares standing for any position there that take one (decision 0030).
pub(super) type ByNode = BTreeMap<String, (Vec<u32>, Vec<u32>, Vec<Take>)>;

/// The position of `rank` of `engine` as the fit sees it: its class and its peers' domain.
pub(super) fn position(engine: &str, rank: u32, class: &str, node: &str) -> Position {
    Position {
        engine: engine.to_owned(),
        rank,
        memory_bytes: 0,
        class: class.to_owned(),
        domain: node.to_owned(),
    }
}

/// Decision 0022: a replacement must sit in its captured collective group's domain unless the
/// engine's collectives can span the fabric. The tensor-parallel group of rank `r` is the block
/// of `tensor_parallel` consecutive ranks holding it. For each lost rank whose spare is outside
/// the domain of the group's survivors, the whole group is lost: its alive ranks join `lost` and
/// are handed over the way a notice hands a rank over. Every rank of a moved group needs a spare,
/// and the group's spares must share one domain, or the change is refused.
pub(super) fn widen_to_domain(
    spec: &EngineSpec,
    view: &EngineView,
    topo: &Topology,
    mut lost: Vec<u32>,
) -> Result<Vec<u32>, ChangeError> {
    let tp = spec.tensor_parallel.max(1);
    if spec.fabric_collectives || tp == 1 {
        return Ok(lost);
    }
    let mut widened = false;
    let mut i = 0;
    while i < lost.len() {
        let r = lost[i];
        i += 1;
        let Some(spare_node) = view.spare_node_for(r) else {
            continue; // no spare: REPLACE is not available for it anyway
        };
        let spare_domain = topo.domain_of(spare_node);
        let group: Vec<u32> = (r / tp * tp..r / tp * tp + tp).collect();
        let split = group.iter().any(|g| {
            !lost.contains(g)
                && view
                    .members
                    .get(g)
                    .is_some_and(|n| topo.domain_of(n) != spare_domain)
        });
        if !split {
            continue;
        }
        for g in &group {
            if lost.contains(g) {
                continue;
            }
            let Some(n) = view.spare_node_for(*g) else {
                return Err(ChangeError::Domain {
                    rank: r,
                    reason: format!(
                        "its spare on {spare_node} is outside its group's domain, and rank {g} \
                         of the group has no spare to move with it"
                    ),
                });
            };
            if topo.domain_of(n) != spare_domain {
                return Err(ChangeError::Domain {
                    rank: r,
                    reason: format!(
                        "its group's spares are not in one domain ({spare_node} and {n})"
                    ),
                });
            }
            lost.push(*g);
            widened = true;
        }
    }
    if widened {
        lost.sort_unstable();
        lost.dedup();
        info!(
            "a replacement outside its group's domain: the whole group moves, ranks {lost:?} \
             (decision 0022)"
        );
    }
    Ok(lost)
}

pub(super) fn outcome_label(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Replace => "replace",
        Outcome::Shrink => "shrink",
        Outcome::Rebuild => "rebuild",
        Outcome::Unspecified => "unspecified",
    }
}

/// Where each of `active` runs in `view`, as a commit carries it (protocol 9): every active rank
/// whose node and device are known.
pub(super) fn placement(view: &EngineView, active: &[u32]) -> Vec<RankPlacement> {
    active
        .iter()
        .filter_map(|r| {
            let node = view.members.get(r)?;
            let device = view.devices.get(r)?;
            Some(RankPlacement {
                rank: *r,
                node: node.clone(),
                device: Some(*device),
                host: view.hosts.get(r).cloned().unwrap_or_default(),
                // a commit carries where ranks run; their caches go with the engine's placement
                ..RankPlacement::default()
            })
        })
        .collect()
}

/// Why `view`'s registered head cannot take a change: it did not say it serves membership
/// changes. None for one that did, and for an engine with no head registered.
pub(super) fn not_serving(view: &EngineView) -> Option<ChangeError> {
    let head = view.head.as_ref()?;
    let serves = head
        .process
        .identity
        .as_ref()
        .is_some_and(|i| i.serves_changes);
    (!serves).then(|| ChangeError::NotServing {
        node: head.process.node.clone(),
        detail: if head.process.identity.is_some() {
            "it registered without Everserve's step hook".to_owned()
        } else {
            "it registered no identity".to_owned()
        },
    })
}
