// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Deliveries and commits: a change delivered to the ranks it names on this node and their
//! acknowledgements returned, a committed membership applied, and a change abandoned after
//! its delivery (decision 0032).

use super::{ACK_TIMEOUT, Daemon, Engine, Outbound};
use crate::clock::now_us;
use crate::proto::everserve::coord::v1::{Abandon, Committed, Deliver, Delivered, worker_event};
use crate::proto::everserve::signals::v1::{Abandoned, Ack, Outcome, Signal, signal};
use std::collections::BTreeSet;
use tokio::sync::{mpsc, oneshot};
use tracing::debug;
use tracing::info;
use tracing::warn;

/// The spare on connection `id` of `e` took `rank`: its registration moves from the standing to
/// the serving, under that rank.
fn promote(e: &mut Engine, id: u64, rank: u32) {
    if let Some(mut ev) = e.standing.remove(&id) {
        ev.rank = rank;
        ev.any_rank = false;
        ev.kind = worker_event::Kind::WorkerRegistered as i32;
        e.serving.insert(id, ev);
    }
}

fn missing(rank: u32, why: &str) -> Ack {
    Ack {
        received_at_us: now_us(),
        rank,
        epoch: 0,
        error: why.to_owned(),
    }
}

impl Daemon {
    /// A change of `a.engine` failed after its delivery (decision 0032): take its epoch as spent,
    /// forget which spare was taking which rank for it, and tell every process of the engine on
    /// this node, each of which acts on it (a head fails the engine, a taking spare exits for its
    /// supervisor to launch it standing again).
    pub(super) async fn abandon(&self, a: Abandon) {
        let targets: Vec<mpsc::Sender<Outbound>> = {
            let mut engines = self.engines.lock().await;
            let Some(e) = engines.get_mut(&a.engine) else {
                return;
            };
            e.epoch = e.epoch.max(a.epoch);
            e.taking.clear();
            e.workers
                .values()
                .chain(e.spares.values())
                .chain(e.any_spares.values())
                .chain(e.clients.iter())
                .map(|c| c.tx.clone())
                .collect()
        };
        warn!(
            "{}: the change at epoch {} was abandoned: {}",
            a.engine, a.epoch, a.detail
        );
        let sig = Signal {
            kind: Some(signal::Kind::Abandoned(Abandoned {
                epoch: a.epoch,
                detail: a.detail,
            })),
        };
        for tx in targets {
            let _ = tx.send((sig.clone(), None)).await;
        }
    }

    /// Deliver `d.change` to the ranks it names on this node and collect their acknowledgements.
    ///
    /// Refused, with the reason in `error`, when the change's epoch is not the next one for the
    /// engine or a named rank is not held here. Public so a test can drive it without a link.
    pub async fn deliver(&self, d: Deliver) -> Delivered {
        let epoch = d.change.as_ref().map_or(0, |c| c.epoch);
        let refuse = |error: String| Delivered {
            request_id: d.request_id,
            engine: d.engine.clone(),
            epoch,
            acks: Vec::new(),
            error,
        };
        let Some(change) = d.change.clone() else {
            return refuse("no change".into());
        };
        let targets: Vec<(u32, mpsc::Sender<Outbound>)> = {
            let mut engines = self.engines.lock().await;
            let Some(e) = engines.get_mut(&d.engine) else {
                return refuse(format!("engine {} unknown here", d.engine));
            };
            if change.epoch != e.epoch + 1 {
                return refuse(format!(
                    "stale epoch: change is {}, this node is at {}",
                    change.epoch, e.epoch
                ));
            }
            let mut targets = Vec::new();
            for r in &d.to_ranks {
                let Some(w) = e.workers.get(r) else {
                    return refuse(format!("rank {r} is not on this node"));
                };
                targets.push((*r, w.tx.clone()));
            }
            for r in &d.to_spares {
                let Some(sp) = e.spares.get(r) else {
                    return refuse(format!("no spare for rank {r} on this node"));
                };
                targets.push((*r, sp.tx.clone()));
            }
            // a spare standing for any position takes the rank the coordinator named (0030)
            for t in &d.take {
                let Some(sp) = e.any_spares.get(&t.device) else {
                    return refuse(format!(
                        "no spare for any rank on device {} of this node to take rank {}",
                        t.device, t.rank
                    ));
                };
                e.taking.insert(t.rank, t.device);
                targets.push((t.rank, sp.tx.clone()));
            }
            targets
        };
        let sig = Signal {
            kind: Some(signal::Kind::Change(change.clone())),
        };
        // the engine's clients hear the change as it is claimed: an executor learns a remote
        // rank's loss from the coordinator's word, since it has no local sight of it
        let clients: Vec<mpsc::Sender<Outbound>> = {
            let engines = self.engines.lock().await;
            engines
                .get(&d.engine)
                .map(|e| e.clients.iter().map(|c| c.tx.clone()).collect())
                .unwrap_or_default()
        };
        for tx in clients {
            let _ = tx.send((sig.clone(), None)).await;
        }
        let mut pending = Vec::new();
        for (rank, tx) in targets {
            let (ack_tx, ack_rx) = oneshot::channel();
            if tx.send((sig.clone(), Some(ack_tx))).await.is_err() {
                pending.push((rank, None));
                continue;
            }
            pending.push((rank, Some(ack_rx)));
        }
        let mut acks = Vec::new();
        for (rank, rx) in pending {
            let ack = match rx {
                Some(rx) => match tokio::time::timeout(ACK_TIMEOUT, rx).await {
                    Ok(Ok(ack)) => ack,
                    Ok(Err(_)) => missing(rank, "connection closed"),
                    Err(_) => missing(rank, "no acknowledgement in time"),
                },
                None => missing(rank, "connection closed"),
            };
            acks.push(ack);
        }
        Delivered {
            request_id: d.request_id,
            engine: d.engine,
            epoch: change.epoch,
            acks,
            error: String::new(),
        }
    }

    /// A committed membership from the coordinator: record the epoch, hand the dead rank's place
    /// to the spare under REPLACE, tell the committed-membership channel and every client. The
    /// membership as it stands (no outcome: a node that missed commits, or one new to the engine)
    /// moves the epoch and tells the workers it displaces, never the clients: no client claimed a
    /// change for it, and an executor completes every commit it hears as its own change.
    pub(super) async fn apply_committed(&self, c: Committed) {
        let Some(membership) = c.membership else {
            return;
        };
        let outcome = Outcome::try_from(c.outcome).unwrap_or(Outcome::Unspecified);
        let clients = {
            let mut engines = self.engines.lock().await;
            let e = engines.entry(c.engine.clone()).or_default();
            // a commit sent again after a coordinator's restart, or one overtaken by a later
            // epoch, is already applied: applied twice, it would drop the rank's new holder
            // (decision 0038)
            if membership.epoch <= e.epoch {
                debug!(
                    "{}: commit at epoch {} ignored, this node is at {}",
                    c.engine, membership.epoch, e.epoch
                );
                return;
            }
            e.epoch = membership.epoch;
            e.lost.clone_from(&membership.lost);
            e.world_size = e.world_size.max(membership.world_size);
            for r in &c.lost {
                e.workers.remove(r);
                e.departed.remove(r);
                // the process that held the rank serves it no longer, whether or not it is gone
                e.serving.retain(|_, ev| ev.rank != *r);
                if outcome == Outcome::Replace
                    && let Some(conn) = e.spares.remove(r)
                {
                    // a spare that took a position stands no longer; it serves the rank
                    promote(e, conn.id, *r);
                    e.workers.insert(*r, conn);
                }
            }
            // the spares standing for any position that took a rank serve as it now (0030)
            for t in &c.take {
                if outcome == Outcome::Replace
                    && let Some(conn) = e.any_spares.remove(&t.device)
                {
                    promote(e, conn.id, t.rank);
                    e.workers.insert(t.rank, conn);
                }
            }
            e.taking.clear();
            // a worker the membership does not place here has left, in the commit's words: it
            // registered before a change it never heard (a node that missed it), and its groups
            // are an older epoch's. It is the engine's no longer here, and is told the membership
            // so it stops and asks anew
            let placed: BTreeSet<u32> = membership
                .members
                .iter()
                .filter(|p| p.node == self.node)
                .map(|p| p.rank)
                .collect();
            // a commit that states no placements (a sender before them) says nothing of where
            // ranks run, and displaces nobody
            let displaced: Vec<u32> = if membership.members.is_empty() {
                Vec::new()
            } else {
                e.workers
                    .keys()
                    .copied()
                    .filter(|r| !placed.contains(r))
                    .collect()
            };
            let mut told: Vec<_> = Vec::new();
            for r in &displaced {
                if let Some(conn) = e.workers.remove(r) {
                    told.push(conn.tx.clone());
                }
                e.serving.retain(|_, ev| ev.rank != *r);
            }
            if !displaced.is_empty() {
                warn!(
                    "{}: ranks {displaced:?} here are not in the membership committed at epoch {}: \
                     no longer the engine's on this node",
                    c.engine, membership.epoch
                );
            }
            self.metrics
                .workers_registered
                .with_label_values(&[&c.engine])
                .set(i64::try_from(e.workers.len()).unwrap_or(i64::MAX));
            e.clients
                .iter()
                .filter(|_| outcome != Outcome::Unspecified)
                .map(|c| c.tx.clone())
                .chain(told)
                .collect::<Vec<_>>()
        };
        info!(
            "{}: membership committed at epoch {} ({outcome:?})",
            c.engine, membership.epoch
        );
        let _ = self.committed_tx.send(membership.clone());
        let sig = Signal {
            kind: Some(signal::Kind::MembershipCommitted(membership)),
        };
        for tx in clients {
            let _ = tx.send((sig.clone(), None)).await;
        }
    }
}
