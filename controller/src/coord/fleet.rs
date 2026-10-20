// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Spare pools: which spares an engine has, their admission, their launches on a node, and the outcome the policy would take (layer 6, decision 0034).

use std::collections::BTreeSet;

use tracing::{info, warn};

use crate::clock::now_ms;
use crate::policy;
use crate::proto::everserve::coord::v1::{
    CoordToDaemon, LaunchSpare, WorkerEvent, coord_to_daemon,
};
use crate::proto::everserve::policy::v1::spare_pool::Loading;
use crate::proto::everserve::signals::v1::Outcome;
use crate::readiness::CompileCacheReadiness;
use crate::status::Severity;

use super::capability::{destination_refusal, engine_difference, position_difference};
use super::transition::position;
use super::view::{EngineView, cache_text, process_view};
use super::{Coordinator, SpareView};
use crate::topology::{self, Topology};

impl Coordinator {
    /// The spares standing behind `engine` right now, with their expected speed and state.
    pub async fn pool(&self, engine: &str) -> Vec<SpareView> {
        let cordoned = self.cordoned().await;
        let engines = self.engines.lock().await;
        let topo = self.topology.lock().await;
        engines
            .get(engine)
            .map(|view| self.spare_views(engine, view, &topo, &cordoned))
            .unwrap_or_default()
    }

    /// The pool of `view` with each spare's fit to its position.
    pub(super) fn spare_views(
        &self,
        engine: &str,
        view: &EngineView,
        topo: &Topology,
        cordoned: &BTreeSet<String>,
    ) -> Vec<SpareView> {
        let loading = self
            .policy
            .spares
            .as_ref()
            .and_then(|p| Loading::try_from(p.loading).ok())
            .unwrap_or(Loading::Preloaded);
        view.spares
            .iter()
            .map(|(rank, node)| {
                let fit = view.classes.get(rank).and_then(|class| {
                    topo.devices()
                        .find(|d| &d.node == node)
                        .map(|d| topology::fit(&position(engine, *rank, class, node), d))
                });
                let (speed, same_class) = fit.map_or((1.0, true), |f| (f.speed, f.same_class));
                let state = match (same_class, loading) {
                    _ if cordoned.contains(node) => "cordoned",
                    (false, _) => "mismatched",
                    (true, Loading::OnNotice) => "on_demand",
                    (true, Loading::Preloaded | Loading::Unspecified) => "warm",
                };
                SpareView {
                    rank: *rank,
                    node: node.clone(),
                    speed,
                    state: state.to_owned(),
                    compile_cache: view
                        .spare_caches
                        .get(rank)
                        .map(CompileCacheReadiness::from_wire),
                    process: process_view(
                        view.processes.spares.get(rank).filter(|p| &p.node == node),
                    ),
                }
            })
            .collect()
    }

    /// Set the `everserve_spares` gauge for `engine` from its pool.
    pub(super) async fn gauge_spares(&self, engine: &str) {
        let pool = self.pool(engine).await;
        for state in ["warm", "on_demand", "mismatched", "cordoned"] {
            let n = pool.iter().filter(|s| s.state == state).count();
            self.metrics
                .spares
                .with_label_values(&[engine, state])
                .set(i64::try_from(n).unwrap_or(i64::MAX));
        }
    }

    /// Tell `node`'s daemon which spares the policy places on it (decision 0024): one
    /// `LaunchSpare` per slot of every engine with a launch command.
    pub(super) async fn send_launches(&self, node: &str) {
        let mut launches = Vec::new();
        for engine in &self.policy.engines {
            let Some(launch) = engine.spare_launch.as_ref() else {
                continue;
            };
            for slot in engine.spares.iter().filter(|s| s.node == node) {
                launches.push(LaunchSpare {
                    engine: engine.name.clone(),
                    rank: slot.rank,
                    device: slot.device,
                    command: launch.command.clone(),
                    env: launch.env.clone(),
                    pool: String::new(),
                    engines: vec![engine.name.clone()],
                });
            }
        }
        if launches.is_empty() {
            return;
        }
        let tx = self.daemons.lock().await.get(node).map(|d| d.tx.clone());
        let Some(tx) = tx else { return };
        for l in launches {
            info!(
                "{}: spare {} kept on {node} (device {}): its daemon launches it unless it runs",
                l.engine,
                l.rank
                    .map_or_else(|| "for any rank".to_owned(), |r| format!("for rank {r}")),
                l.device
            );
            let _ = tx.send(CoordToDaemon {
                term: 0,
                kind: Some(coord_to_daemon::Kind::LaunchSpare(l)),
            });
        }
    }

    /// The outcome the policy would take for losing `ranks` of `engine` now.
    pub(super) fn expected_outcome(
        &self,
        engine: &str,
        view: &EngineView,
        ranks: &[u32],
        cordoned: &BTreeSet<String>,
    ) -> Outcome {
        let placement_independent = self
            .engine_spec(engine)
            .is_some_and(|e| e.placement_independent_model);
        policy::choose_outcome(
            &self.policy,
            view.spares_cover(ranks, cordoned),
            placement_independent,
        )
    }
}

impl Coordinator {
    /// Whether the spare `ev` registered on `node` may stand in `view`'s pool now, or why not:
    /// what it runs, held to its position or the engine (decision 0034); under the exact class,
    /// its compile pair, held to its rank's once that rank reported one, and the first pair
    /// reported setting the engine's key (0029); and its device's fit to its position, a class
    /// other than the position's only where the policy allows mismatch within its degrade
    /// (docs/MULTINODE.md 5). A spare standing for any position is held at its join for the
    /// rest.
    pub(super) fn spare_verdict(
        &self,
        view: &mut EngineView,
        node: &str,
        ev: &WorkerEvent,
        topo: &Topology,
    ) -> Result<(), String> {
        if let Some(why) = self.identity_refusal(view, ev) {
            return Err(format!("{why} (decision 0034)"));
        }
        if ev.any_rank {
            return Ok(());
        }
        if self.exact(&ev.engine) {
            // a spare reporting no pair is admitted only while the engine has no key to hold it
            // to (an engine that compiles nothing keeps its spares; it never reads consistent)
            let same = match ev.compile_cache.as_ref() {
                Some(c) => {
                    if view.compile_key.is_none() {
                        view.compile_key = Some(c.key.clone());
                    }
                    view.holds_for(ev.rank, c)
                }
                None => view.compile_key.is_none(),
            };
            if !same {
                return Err(format!(
                    "it runs on compile cache {}, rank {}'s is {} under the engine's key {} \
                     (decision 0029)",
                    cache_text(ev.compile_cache.as_ref()),
                    ev.rank,
                    cache_text(view.caches.get(&ev.rank)),
                    view.compile_key.as_deref().unwrap_or("none")
                ));
            }
        }
        let (Some(class), Some(device)) =
            (view.classes.get(&ev.rank), topo.device(node, ev.device))
        else {
            return Ok(());
        };
        let f = topology::fit(&position(&ev.engine, ev.rank, class, node), device);
        let allowed = f.same_class
            || self.policy.spares.as_ref().is_some_and(|p| {
                p.allow_mismatched && topology::within_degrade(&f, p.max_degrade_percent)
            });
        if allowed && f.ok {
            Ok(())
        } else {
            Err(format!("{} (speed {:.2})", f.why, f.speed))
        }
    }

    /// Derive again which of `engine`'s standing spares its pool admits, from their
    /// registrations (decision 0036): after any change to the view, what a spare is held to may
    /// have arrived or changed (a position's identity or compile pair, the engine's key). A spare
    /// that no longer qualifies leaves the pool, refused with the reason; a refused one that now
    /// does joins it, without registering again. A position keeps the spare it has; nothing is
    /// derived while a change is in flight, since its spares are spoken for.
    pub(super) async fn recheck_spares(&self, engine: &str) {
        let mut engines = self.engines.lock().await;
        let Some(view) = engines.get_mut(engine) else {
            return;
        };
        if view.change.is_some() {
            return;
        }
        let topo = self.topology.lock().await;
        let standing: Vec<((String, u32), WorkerEvent)> = view
            .standing
            .iter()
            .map(|(k, ev)| (k.clone(), ev.clone()))
            .collect();
        for ((node, device), ev) in standing {
            let placed = if ev.any_rank {
                view.any_spares.contains(&(node.clone(), device))
            } else {
                view.spares.get(&ev.rank) == Some(&node)
            };
            let key = (node.clone(), device);
            match (placed, self.spare_verdict(view, &node, &ev, &topo)) {
                (true, Err(why)) => {
                    warn!("{engine}: spare on {node} device {device} leaves the pool: {why}");
                    if ev.any_rank {
                        view.any_spares.remove(&key);
                    } else {
                        view.spares.remove(&ev.rank);
                        view.spare_caches.remove(&ev.rank);
                    }
                    view.status.record(
                        now_ms(),
                        Severity::Warning,
                        "SpareRefused",
                        format!("spare on {node} device {device} left the pool: {why}"),
                    );
                    view.refused_spares
                        .insert(key, ((!ev.any_rank).then_some(ev.rank), why));
                }
                (false, Ok(())) if ev.any_rank || !view.spares.contains_key(&ev.rank) => {
                    info!(
                        "{engine}: spare on {node} device {device} now qualifies; it joins the pool"
                    );
                    view.refused_spares.remove(&key);
                    view.processes.record(&node, &ev);
                    place_spare(view, &node, &ev);
                    view.status.record(
                        now_ms(),
                        Severity::Normal,
                        "SpareAdmitted",
                        format!("spare on {node} device {device} joins the pool"),
                    );
                }
                (false, Err(why)) => {
                    view.refused_spares
                        .insert(key, ((!ev.any_rank).then_some(ev.rank), why));
                }
                _ => {}
            }
        }
    }
}

/// Put the admitted spare `ev` on `node` in `view`'s pool: standing for any position by node and
/// device, or for its rank with its compile pair.
pub(super) fn place_spare(view: &mut EngineView, node: &str, ev: &WorkerEvent) {
    if ev.any_rank {
        view.admit_any_spare(&ev.engine, node, ev.device);
        return;
    }
    view.spares.insert(ev.rank, node.to_owned());
    match ev.compile_cache.as_ref() {
        Some(c) => {
            view.spare_caches.insert(ev.rank, c.clone());
        }
        None => {
            view.spare_caches.remove(&ev.rank);
        }
    }
}

impl Coordinator {
    /// Why a spare that registered may not join the engine's pool for what it runs (decision
    /// 0034): no identity or a dirty adapter, or a field that differs from the position it stands
    /// for, or, for a spare standing for any position, from the engine's. None when it may, or
    /// when there is nothing yet to hold it to.
    pub(super) fn identity_refusal(&self, view: &EngineView, ev: &WorkerEvent) -> Option<String> {
        let id = ev.identity.as_ref();
        if let Some(why) = destination_refusal(id) {
            return Some(why);
        }
        let id = id?;
        let exact = self.exact(&ev.engine);
        if ev.any_rank {
            engine_difference(id, view.identities.values().next()?, exact)
        } else {
            position_difference(id, view.identities.get(&ev.rank)?, exact)
        }
    }
}
