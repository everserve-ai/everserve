// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The change in flight survives the coordinator (decision 0038). A coordinator that starts, or
//! takes the lease, finds each engine's last change in the store's log, and finishes it before it
//! claims any other change of the engine: a logged commit is sent to every daemon of the engine
//! again, and a logged claim is delivered again, the same change to the same receivers, and
//! committed. A logged claim whose receivers are not all there cannot be sent again, and fails as
//! a change does after its delivery (0032).

use std::collections::BTreeSet;

use tracing::{info, warn};

use crate::proto::everserve::coord::v1::{CoordToDaemon, coord_to_daemon};
use crate::proto::everserve::signals::v1::Outcome;
use crate::proto::everserve::spec::v1::ChangeLog;
use crate::status::Severity;

use super::cover::{SETTLE, expected_nodes};
use super::transition::{apply_claim, by_node_of};
use super::view::EngineView;
use super::{ChangeError, Coordinator};

impl Coordinator {
    /// Whether `engine` has a logged change this coordinator has not finished yet.
    pub(super) async fn unfinished(&self, engine: &str) -> bool {
        self.specs.unfinished.lock().await.contains(engine)
    }

    /// Whether `engine`'s logged change can be finished now: every daemon its policy, its plan
    /// and the change name has linked again, or [`SETTLE`] has passed since this coordinator
    /// started, after which a receiver still missing fails the change.
    pub(super) async fn ready_to_finish(&self, engine: &str) -> bool {
        if self.started.elapsed() >= SETTLE {
            return true;
        }
        let Some(spec) = self.engine_spec(engine) else {
            return true;
        };
        let named: BTreeSet<String> = {
            let engines = self.engines.lock().await;
            let Some(view) = engines.get(engine) else {
                return false;
            };
            expected_nodes(spec, view)
        };
        let logged: BTreeSet<String> = self
            .kept_status(engine)
            .and_then(|s| s.change)
            .map(|log| log.deliveries.into_iter().map(|d| d.node).collect())
            .unwrap_or_default();
        let linked = self.daemons.lock().await;
        named.iter().chain(&logged).all(|n| linked.contains_key(n))
    }

    /// Whether this coordinator may claim a new change of `engine`: it knows every rank of the
    /// membership kept for the engine's incarnation, held, departed outside a change, or shrunk
    /// away. A coordinator that starts or takes the lease learns the holders from the hellos, and
    /// a change's deliveries are made from them: claimed before they came, a change is delivered
    /// to part of the engine, or to none of it, and commits without the ranks that never heard
    /// it. A rank no daemon relinks is known at last as lost, by the takeover's allowance
    /// (`takeover`), never by waiting alone. An engine with no membership kept never held every
    /// rank, and is claimed on what is known, as it always was.
    pub(super) async fn may_claim(&self, engine: &str) -> bool {
        let Some(kept) = self.kept_members(engine) else {
            return true;
        };
        let engines = self.engines.lock().await;
        let Some(view) = engines.get(engine) else {
            return false;
        };
        kept.keys().all(|r| {
            view.members.contains_key(r) || view.departed.contains(r) || view.lost.contains(r)
        })
    }

    /// Wake `engine` once [`SETTLE`] has passed since this coordinator started, when no daemon
    /// relinking wakes it first: one wake per engine.
    pub(super) async fn wake_at_settle(&self, engine: &str) {
        if !self
            .specs
            .settle_wakes
            .lock()
            .await
            .insert(engine.to_owned())
        {
            return;
        }
        let (Some(me), wait) = (
            self.me.upgrade(),
            SETTLE.saturating_sub(self.started.elapsed()),
        ) else {
            return;
        };
        tokio::spawn(wake_after(me, engine.to_owned(), wait));
    }

    /// Finish `engine`'s logged change: send its logged commit again, or deliver its logged claim
    /// again and commit it. Run by the controller before any other change of the engine.
    pub(super) async fn finish_logged(&self, engine: &str) {
        let log = self.kept_status(engine).and_then(|s| s.change);
        self.specs.unfinished.lock().await.remove(engine);
        let Some(log) = log else { return };
        let Some(change) = log.change.clone() else {
            return;
        };
        let epoch = change.epoch;
        let committed_here = self
            .engines
            .lock()
            .await
            .get(engine)
            .map_or(0, |v| v.committed_epoch);
        if log.committed.is_some() {
            self.send_logged_commit(engine, &log).await;
            return;
        }
        if committed_here >= epoch {
            // a daemon holds a commit at or past it that was never logged: not one of ours
            warn!(
                "{engine}: the logged change at epoch {epoch} has no logged commit, and a daemon is at \
                 {committed_here}; nothing to finish"
            );
            return;
        }
        info!("{engine}: finishing the change at epoch {epoch} a previous coordinator claimed");
        if let Err(e) = self.receivers_present(engine, &log).await {
            if let Some(view) = self.engines.lock().await.get_mut(engine) {
                view.epoch = view.epoch.max(epoch);
            }
            self.abandon(engine, epoch, &e).await;
            return;
        }
        if let Some(view) = self.engines.lock().await.get_mut(engine) {
            apply_claim(view, &log);
        }
        self.engine_event(
            engine,
            Severity::Normal,
            "ChangeResumed",
            format!("epoch {epoch}: delivered again, as the previous coordinator logged it"),
        )
        .await;
        if let Err(e) = self
            .finish_change(engine, &log, change.delivered_at_us)
            .await
        {
            warn!("{engine}: the resumed change at epoch {epoch} did not commit: {e}");
        }
    }

    /// Send `log`'s commit to every daemon of the engine, the view taking the membership it
    /// commits; a daemon that already applied it ignores it.
    async fn send_logged_commit(&self, engine: &str, log: &ChangeLog) {
        let Some(committed) = log.committed.clone() else {
            return;
        };
        let Some(membership) = committed.membership.clone() else {
            return;
        };
        let epoch = membership.epoch;
        info!(
            "{engine}: the change at epoch {epoch} was committed; its commit goes again to every \
             daemon of the engine"
        );
        if let Some(view) = self.engines.lock().await.get_mut(engine) {
            apply_logged_commit(view, log);
        }
        let targets = self.commit_targets(engine).await;
        let msg = CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::Committed(committed)),
        };
        for tx in targets.into_values() {
            let _ = tx.send(msg.clone());
        }
        let _ = self.committed_tx.send(membership);
        let lost = log
            .change
            .as_ref()
            .map(|c| c.lost.clone())
            .unwrap_or_default();
        self.unstage_restores(engine, &lost, epoch).await;
        self.refresh_replication(&BTreeSet::from([engine.to_owned()]))
            .await;
        self.engine_event(
            engine,
            Severity::Normal,
            "ChangeResumed",
            format!("epoch {epoch}: its logged commit sent again"),
        )
        .await;
        self.observed(engine).await;
    }

    /// Every receiver `log` names is here: each node linked, each serving rank registered on its
    /// node, each spare standing on its node.
    ///
    /// # Errors
    ///
    /// [`ChangeError::Node`] for a node not linked; [`ChangeError::Departed`] for ranks or spares
    /// not registered where the log sent them.
    async fn receivers_present(&self, engine: &str, log: &ChangeLog) -> Result<(), ChangeError> {
        {
            let daemons = self.daemons.lock().await;
            if let Some(d) = log
                .deliveries
                .iter()
                .find(|d| !daemons.contains_key(&d.node))
            {
                return Err(ChangeError::Node {
                    node: d.node.clone(),
                    reason: "not linked since the coordinator restarted".into(),
                });
            }
        }
        let engines = self.engines.lock().await;
        let view = engines.get(engine).ok_or(ChangeError::Unknown(0))?;
        let missing: Vec<u32> = by_node_of(log)
            .into_iter()
            .flat_map(|(node, (ranks, spares, takes))| {
                let ranks = ranks
                    .into_iter()
                    .filter(|r| view.members.get(r) != Some(&node))
                    .collect::<Vec<_>>();
                let spares = spares
                    .into_iter()
                    .filter(|r| view.spares.get(r) != Some(&node))
                    .collect::<Vec<_>>();
                let takes = takes
                    .into_iter()
                    .filter(|t| !view.any_spares.contains(&(t.node.clone(), t.device)))
                    .map(|t| t.rank)
                    .collect::<Vec<_>>();
                ranks.into_iter().chain(spares).chain(takes)
            })
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(ChangeError::Departed { ranks: missing })
        }
    }
}

/// Wake `engine`'s reconcile after `wait`. Its own function, typed as `Send`: the wake reconciles,
/// and a reconcile may set another wake.
fn wake_after(
    me: std::sync::Arc<Coordinator>,
    engine: String,
    wait: std::time::Duration,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(async move {
        tokio::time::sleep(wait).await;
        me.wake(&engine).await;
    })
}

/// Make `view` the engine's after the commit `log` records, from the membership it committed:
/// each rank's holder is the one the commit names, whatever mix of the daemons' hellos before and
/// after it the view was built from. A spare that took a rank stands no longer.
pub(super) fn apply_logged_commit(view: &mut EngineView, log: &ChangeLog) {
    let (Some(change), Some(committed)) = (log.change.as_ref(), log.committed.as_ref()) else {
        return;
    };
    let Some(membership) = committed.membership.as_ref() else {
        return;
    };
    let replace = Outcome::try_from(committed.outcome) == Ok(Outcome::Replace);
    for r in &change.lost {
        view.changed_at.insert(*r, change.epoch);
        view.departed.remove(r);
    }
    for p in &membership.members {
        // a commit places every rank on its device: a placement without one names no holder
        let Some(device) = p.device else { continue };
        let at = (p.node.clone(), device);
        if replace && change.lost.contains(&p.rank) {
            if view.spares.get(&p.rank) == Some(&p.node) {
                view.spares.remove(&p.rank);
                view.processes.promote(p.rank, None);
            } else if view.any_spares.remove(&at) {
                view.processes.promote(p.rank, Some(&at));
            }
            if let Some(ev) = view.standing.remove(&at) {
                view.hosts.insert(p.rank, ev.host);
            }
            view.taking.remove(&p.rank);
        }
        view.members.insert(p.rank, p.node.clone());
        view.devices.insert(p.rank, device);
        if !p.host.is_empty() {
            view.hosts.insert(p.rank, p.host.clone());
        }
    }
    let held: BTreeSet<u32> = membership.members.iter().map(|p| p.rank).collect();
    for r in change.lost.iter().filter(|r| !held.contains(r)) {
        view.members.remove(r);
    }
    view.lost.clone_from(&log.shrunk);
    view.epoch = view.epoch.max(change.epoch);
    view.committed_epoch = view.committed_epoch.max(change.epoch);
    view.change = None;
}

#[cfg(test)]
mod tests {
    use super::apply_logged_commit;
    use crate::coord::view::EngineView;
    use crate::proto::everserve::coord::v1::Committed;
    use crate::proto::everserve::signals::v1::{
        Change, MembershipCommitted, Outcome, RankPlacement,
    };
    use crate::proto::everserve::spec::v1::ChangeLog;

    /// The log of node-b's ranks 4 to 7 moving to node-c's spares at epoch 1, committed.
    fn moved_to_node_c() -> ChangeLog {
        let placed = |rank: u32, node: &str, device: u32| RankPlacement {
            rank,
            node: node.into(),
            device: Some(device),
            ..RankPlacement::default()
        };
        ChangeLog {
            change: Some(Change {
                lost: vec![4, 5, 6, 7],
                outcome: Outcome::Replace as i32,
                epoch: 1,
                ..Change::default()
            }),
            world_size: 8,
            committed: Some(Committed {
                engine: "k3-a".into(),
                outcome: Outcome::Replace as i32,
                lost: vec![4, 5, 6, 7],
                membership: Some(MembershipCommitted {
                    epoch: 1,
                    world_size: 8,
                    active_ranks: (0..8).collect(),
                    members: (0..4)
                        .map(|r| placed(r, "node-a", r))
                        .chain((4..8).map(|r| placed(r, "node-c", r - 4)))
                        .collect(),
                    ..MembershipCommitted::default()
                }),
                ..Committed::default()
            }),
            ..ChangeLog::default()
        }
    }

    /// A view as hellos built it: node-a's ranks, and node-b's and node-c's as each daemon last
    /// held them, before the commit (`behind`) or after it.
    fn view(old_holder_behind: bool, spare_node_behind: bool) -> EngineView {
        let mut v = EngineView::default();
        for r in 0..4 {
            v.members.insert(r, "node-a".into());
            v.devices.insert(r, r);
        }
        for r in 4..8 {
            if old_holder_behind {
                v.members.insert(r, "node-b".into());
                v.devices.insert(r, r - 4);
            }
            if spare_node_behind {
                v.spares.insert(r, "node-c".into());
            } else {
                v.members.insert(r, "node-c".into());
                v.devices.insert(r, r - 4);
            }
        }
        v.epoch = u64::from(!spare_node_behind);
        v.committed_epoch = v.epoch;
        v
    }

    #[test]
    fn a_logged_commit_names_every_ranks_holder_whatever_hellos_the_view_was_built_from() {
        // decision 0038: a successor's view is built from hellos sent before the commit and after
        // it, in any order; the logged commit says who holds each rank
        for (b_behind, c_behind) in [(true, true), (true, false), (false, false)] {
            let mut v = view(b_behind, c_behind);
            apply_logged_commit(&mut v, &moved_to_node_c());
            for r in 4..8 {
                assert_eq!(
                    (v.members.get(&r).map(String::as_str), v.devices.get(&r)),
                    (Some("node-c"), Some(&(r - 4))),
                    "rank {r}, node-b behind {b_behind}, node-c behind {c_behind}"
                );
            }
            assert!(
                v.spares.is_empty(),
                "a spare that took its rank stands no longer"
            );
            assert_eq!((v.epoch, v.committed_epoch), (1, 1));
            assert!(v.change.is_none());
        }
    }
}
