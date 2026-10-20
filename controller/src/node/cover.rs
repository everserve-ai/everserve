// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The cover plan and the groups it forms (decision 0035): the newest plan kept and forwarded
//! to every worker and spare here, each group a connection reports formed held until it is
//! released, and released for it when its connection ends.

use super::{Daemon, Outbound};
use crate::proto::everserve::coord::v1::{
    CoverReport, DaemonToCoord, HeldGroup, cover_report, daemon_to_coord,
};
use crate::proto::everserve::signals::v1::{CoverPlan, GroupFormed, GroupReleased, Signal, signal};
use tokio::sync::mpsc;
use tracing::info;

impl Daemon {
    /// The coordinator's newest cover plan for an engine (decision 0035): kept, and forwarded to
    /// every worker and spare of it here. A plan at or below one already held is ignored, so
    /// generations only move forward.
    pub(super) async fn cover_plan(&self, plan: CoverPlan) {
        let targets: Vec<mpsc::Sender<Outbound>> = {
            let mut engines = self.engines.lock().await;
            let e = engines.entry(plan.engine.clone()).or_default();
            if let Some(held) = e.plan.as_ref()
                && plan.generation <= held.generation
            {
                info!(
                    "{}: cover plan {} ignored: {} is held",
                    plan.engine, plan.generation, held.generation
                );
                return;
            }
            e.plan = Some(plan.clone());
            e.workers
                .values()
                .chain(e.spares.values())
                .chain(e.any_spares.values())
                .map(|c| c.tx.clone())
                .collect()
        };
        info!(
            "{}: cover plan {} for epoch {}: {} groups, to {} processes",
            plan.engine,
            plan.generation,
            plan.epoch,
            plan.groups.len(),
            targets.len()
        );
        for tx in targets {
            let _ = tx
                .send((
                    Signal {
                        kind: Some(signal::Kind::CoverPlan(plan.clone())),
                    },
                    None,
                ))
                .await;
        }
    }

    /// The engine's standing plan, to a process that registers while it stands: it forms its
    /// part now, not at the next generation (decision 0035).
    pub(super) async fn plan_to(&self, engine: &str, to: &mpsc::Sender<Outbound>) {
        let plan = self
            .engines
            .lock()
            .await
            .get(engine)
            .and_then(|e| e.plan.clone());
        if let Some(plan) = plan {
            let _ = to
                .send((
                    Signal {
                        kind: Some(signal::Kind::CoverPlan(plan)),
                    },
                    None,
                ))
                .await;
        }
    }

    /// Connection `id` of `engine` formed a group: held until it is released, and reported.
    pub(super) async fn formed(&self, engine: &str, id: u64, f: GroupFormed) {
        if let Some(e) = self.engines.lock().await.get_mut(engine) {
            let held = e.held.entry(id).or_default();
            held.retain(|h| !(h.spans == f.spans && h.scope == f.scope));
            held.push(HeldGroup {
                member: f.member.clone(),
                spans: f.spans.clone(),
                scope: f.scope,
                generation: f.generation,
            });
        }
        self.report_cover(cover_report::Kind::Formed(f)).await;
    }

    /// Connection `id` of `engine` released a group: no longer held, and reported.
    pub(super) async fn released(&self, engine: &str, id: u64, r: GroupReleased) {
        if let Some(e) = self.engines.lock().await.get_mut(engine)
            && let Some(held) = e.held.get_mut(&id)
        {
            held.retain(|h| !(h.spans == r.spans && h.scope == r.scope));
        }
        self.report_cover(cover_report::Kind::Released(r)).await;
    }

    /// Connection `id` of `engine` closed: every group it held is released with it, reported for
    /// its member, so a spare that comes back on the same device inherits nothing and a port its
    /// group held returns to its host (decision 0035).
    pub(super) async fn release_held(&self, engine: &str, id: u64) {
        let held = self
            .engines
            .lock()
            .await
            .get_mut(engine)
            .and_then(|e| e.held.remove(&id))
            .unwrap_or_default();
        for h in held {
            self.report_cover(cover_report::Kind::Released(GroupReleased {
                engine: engine.to_owned(),
                generation: h.generation,
                spans: h.spans,
                scope: h.scope,
                member: h.member,
            }))
            .await;
        }
    }

    /// Send a process's report of a group up. Not held across a missing link: the next hello
    /// carries what is held.
    pub(super) async fn report_cover(&self, kind: cover_report::Kind) {
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::CoverReport(CoverReport {
                    kind: Some(kind),
                })),
            },
            false,
        )
        .await;
    }
}
