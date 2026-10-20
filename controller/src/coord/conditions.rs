// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! An engine's conditions, derived from its observed status (decision 0036) in one place: after
//! every change to its view, with the spare gauge.
//!
//! Until the engine has a spec of its own, what it should be is what the policy says: every
//! position held, a spare for every node's loss, one head, no spare refused.

use std::collections::{BTreeMap, BTreeSet};

use crate::clock::now_ms;
use crate::policy;
use crate::proto::everserve::coord::v1::{CoordToDaemon, coord_to_daemon};
use crate::proto::everserve::signals::v1::{InstancePhase, InstanceState, Outcome};
use crate::status::{Observed, Polarity, Severity};

use super::view::EngineView;
use super::{ChangeError, Coordinator};

impl Coordinator {
    /// `engine`'s observed status changed: derive again which spares its pool admits, then
    /// refresh its spare gauge and its conditions.
    pub(super) async fn observed(&self, engine: &str) {
        // a process that left a release behind goes once a spare can take its rank, before the
        // conditions below say where it stands
        self.retire_leaked(engine).await;
        self.recheck_spares(engine).await;
        self.replan(engine).await;
        self.gauge_spares(engine).await;
        let cordoned = self.cordoned().await;
        let capacity = self.capacity.lock().await.clone();
        let mut engines = self.engines.lock().await;
        let Some(view) = engines.get_mut(engine) else {
            return;
        };
        let mut observed = self.engine_conditions(engine, view, &cordoned);
        if let Some(head) = self.head_on_spot(engine, view, capacity.as_ref()) {
            observed.push(head);
        }
        view.status.observe(observed, now_ms());
        crate::metrics::ControlMetrics::set_conditions(
            &self.metrics.control.engine_condition,
            engine,
            &view.status.conditions(),
        );
        let world = self
            .engine_spec(engine)
            .map_or(view.world_size, policy::world_size);
        let state = instance_state(engine, view, world);
        let changed = view.instance_state.as_ref() != Some(&state);
        if changed {
            view.instance_state = Some(state.clone());
        }
        drop(engines);
        // an engine's spares are its pools' too (decision 0031)
        self.observe_pools().await;
        // its processes learn where it runs, and the state their framework is told, from here
        // (decision 0041): a change's state goes out before the change's first delivery
        self.push_placement(engine).await;
        if changed {
            self.push_instance_state(state).await;
        }
        self.keep_members(engine).await;
    }

    /// Send `state` to its engine's audience, each daemon of which hands it to the engine's
    /// processes on its node.
    async fn push_instance_state(&self, state: InstanceState) {
        let audience = self.audience_of(&state.engine).await;
        let msg = CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::InstanceState(state)),
        };
        for tx in audience.into_values() {
            let _ = tx.send(msg.clone());
        }
    }

    /// Record an event in `engine`'s log; an engine the policy names has one before any process
    /// of it registered.
    pub(super) async fn engine_event(
        &self,
        engine: &str,
        severity: Severity,
        reason: &str,
        message: String,
    ) {
        let mut engines = self.engines.lock().await;
        // an engine of the policy has its log before any process of it registered: a process
        // refused at its first ask is its first event
        let view = if self.engine_spec(engine).is_some() {
            Some(engines.entry(engine.to_owned()).or_default())
        } else {
            engines.get_mut(engine)
        };
        if let Some(view) = view {
            view.status.record(now_ms(), severity, reason, message);
        }
    }

    /// Record a change's outcome in `engine`'s log: committed, folded away, or refused or failed
    /// with the reason.
    pub(super) async fn change_event(
        &self,
        engine: &str,
        lost: &[u32],
        result: &Result<(Outcome, u64), ChangeError>,
        elapsed_ms: u64,
    ) {
        let (severity, reason, message) = match result {
            Ok((outcome, epoch)) => (
                Severity::Normal,
                "ChangeCommitted",
                format!(
                    "epoch {epoch}: {} for ranks {lost:?}, {elapsed_ms} ms",
                    outcome.as_str_name()
                ),
            ),
            Err(e @ ChangeError::Stale { .. }) => (Severity::Normal, "ReportFolded", e.to_string()),
            Err(e) => (
                Severity::Warning,
                change_error_reason(e),
                format!("ranks {lost:?}: {e}"),
            ),
        };
        self.engine_event(engine, severity, reason, message).await;
    }

    /// The conditions `view` of `engine` shows, with `cordoned` nodes' spares taking no position.
    pub(super) fn engine_conditions(
        &self,
        engine: &str,
        view: &EngineView,
        cordoned: &BTreeSet<String>,
    ) -> Vec<Observed> {
        let down = match &view.down {
            Some(d) => Observed::new(
                "Down",
                Polarity::Trouble,
                true,
                down_reason(d.cause),
                d.detail.clone(),
            ),
            None => Observed::new("Down", Polarity::Trouble, false, "Up", ""),
        };
        let serving = if view.down.is_some() {
            Observed::new(
                "Serving",
                Polarity::Healthy,
                false,
                "Down",
                "the engine is down",
            )
        } else if view.members.is_empty() {
            Observed::new(
                "Serving",
                Polarity::Healthy,
                false,
                "NoMembers",
                "no position is held",
            )
        } else {
            Observed::new(
                "Serving",
                Polarity::Healthy,
                true,
                "Members",
                format!(
                    "{} of {} positions held",
                    view.members.len(),
                    view.world_size
                ),
            )
        };
        let progressing = match &view.change {
            Some((epoch, lost)) => Observed::new(
                "Progressing",
                Polarity::Neutral,
                true,
                "ChangeInFlight",
                format!("epoch {epoch} takes ranks {lost:?}"),
            ),
            None => Observed::new("Progressing", Polarity::Neutral, false, "Idle", ""),
        };
        vec![
            serving,
            progressing,
            self.covered(engine, view, cordoned),
            degraded(view),
            down,
        ]
    }

    /// `HeadOnSpot`, for an engine whose policy wants its head on on-demand capacity (decision
    /// 0041): true when the head's node is spot, unknown while the head or its node's capacity is
    /// not known. None for an engine that does not ask.
    fn head_on_spot(
        &self,
        engine: &str,
        view: &EngineView,
        capacity: Option<&super::capacity::Capacity>,
    ) -> Option<Observed> {
        if !self.engine_spec(engine).is_some_and(|e| e.head_on_demand) {
            return None;
        }
        let kind = "HeadOnSpot";
        let Some(head) = view.head.as_ref() else {
            return Some(Observed::new(
                kind,
                Polarity::Trouble,
                crate::status::Truth::Unknown,
                "NoHead",
                "no head is registered",
            ));
        };
        let node = &head.process.node;
        Some(match super::capacity::is_spot(capacity, node) {
            Some(true) => Observed::new(
                kind,
                Polarity::Trouble,
                true,
                "SpotNode",
                format!("the head runs on {node}, spot capacity a reclaim can take"),
            ),
            Some(false) => Observed::new(
                kind,
                Polarity::Trouble,
                false,
                "OnDemandNode",
                format!("the head runs on {node}, on-demand capacity"),
            ),
            None => Observed::new(
                kind,
                Polarity::Trouble,
                crate::status::Truth::Unknown,
                "CapacityUnknown",
                format!(
                    "{node}'s capacity is not known: the cluster's Nodes are not followed, or do not list it"
                ),
            ),
        })
    }

    /// Covered: a loss of any one node holding members is replaced, not rebuilt. The head's
    /// node's loss restarts the engine (decision 0032), named as such rather than counted.
    fn covered(&self, engine: &str, view: &EngineView, cordoned: &BTreeSet<String>) -> Observed {
        if view.cover.plan.is_some() {
            return covered_by_plan(view);
        }
        if view.members.is_empty() {
            return Observed::new(
                "Covered",
                Polarity::Healthy,
                crate::status::Truth::Unknown,
                "NoMembers",
                "no position is held",
            );
        }
        let mut by_node: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
        for (rank, node) in &view.members {
            by_node.entry(node).or_default().push(*rank);
        }
        let uncovered: Vec<String> = by_node
            .iter()
            .filter(|(_, ranks)| {
                self.expected_outcome(engine, view, ranks, cordoned) == Outcome::Rebuild
            })
            .map(|(node, ranks)| format!("{node} (ranks {ranks:?})"))
            .collect();
        if uncovered.is_empty() {
            Observed::new(
                "Covered",
                Polarity::Healthy,
                true,
                "EveryNodeCovered",
                "a loss of any one node is replaced",
            )
        } else {
            Observed::new(
                "Covered",
                Polarity::Healthy,
                false,
                "NodeUncovered",
                format!("a loss of {} would rebuild", uncovered.join(", ")),
            )
        }
    }
}

/// Covered while cover is planned (decision 0035): a loss is covered only through a set every
/// member of whose groups reported it, not by a spare standing.
fn covered_by_plan(view: &EngineView) -> Observed {
    let missing = super::cover::losses_without_cover(view);
    // the head's node is named, its ranks not counted: their loss restarts the engine (decision
    // 0032), and a condition that no engine with a head could ever meet would say nothing
    let head = super::cover::head_node_loss(view)
        .map(|r| format!("; the head's node holds {r:?}, whose loss restarts the engine"))
        .unwrap_or_default();
    if missing.is_empty() {
        return Observed::new(
            "Covered",
            Polarity::Healthy,
            true,
            "EveryLossCovered",
            format!("every node's and position's loss has a formed set{head}"),
        );
    }
    let planned = |l: &Vec<u32>| {
        view.cover
            .sets
            .iter()
            .any(|s| policy::cover::answers(&s.set, l))
    };
    let forming = missing.iter().filter(|l| planned(l)).count();
    Observed::new(
        "Covered",
        Polarity::Healthy,
        false,
        if forming == missing.len() {
            "Forming"
        } else {
            "LossUncovered"
        },
        format!(
            "{} losses without a formed set ({forming} planned and forming): {missing:?}{head}",
            missing.len()
        ),
    )
}

/// `engine`'s instance state in a framework's terms (decision 0041), of a `world` of positions:
/// the first that holds of failed (down), recovering (a change in flight), degraded (ranks lost
/// under SHRINK), serving (every position held) and starting.
pub(super) fn instance_state(engine: &str, view: &EngineView, world: u32) -> InstanceState {
    let held = u32::try_from(view.members.len()).unwrap_or(u32::MAX);
    let (phase, reason) = if let Some(d) = &view.down {
        (
            InstancePhase::InstanceFailed,
            format!("{}: {}", d.cause, d.detail),
        )
    } else if let Some((epoch, lost)) = &view.change {
        (
            InstancePhase::InstanceRecovering,
            format!("epoch {epoch} takes ranks {lost:?}"),
        )
    } else if !view.lost.is_empty() {
        (
            InstancePhase::InstanceDegraded,
            format!("ranks {:?} lost under SHRINK", view.lost),
        )
    } else if world > 0 && held >= world {
        (InstancePhase::InstanceServing, String::new())
    } else {
        (
            InstancePhase::InstanceStarting,
            format!("{held} of {world} positions held"),
        )
    };
    InstanceState {
        engine: engine.to_owned(),
        phase: phase as i32,
        positions_held: held,
        world,
        epoch: view.committed_epoch,
        reason,
    }
}

/// Degraded: serving, short of what the policy says, with the first reason found.
/// How two of `view`'s members differ in what they run (decision 0034's compared fields), the
/// lowest-ranked one that differs from rank-order's first named; None when they all run the same.
/// A spare is held to its position's identity before it is admitted; the members are held to one
/// another here, as they register, so an engine whose ranks run different software, or the same
/// weights read as different ones, says so at bring-up rather than at its first replacement.
fn members_differ(view: &EngineView) -> Option<String> {
    let mut members = view
        .members
        .keys()
        .filter_map(|r| view.identities.get(r).map(|i| (*r, i)));
    let (first, reference) = members.next()?;
    members.find_map(|(rank, identity)| {
        super::capability::engine_difference(identity, reference, false)
            .map(|why| format!("rank {rank} and rank {first} run different software: {why}"))
    })
}

fn degraded(view: &EngineView) -> Observed {
    let trouble =
        |reason, message| Observed::new("Degraded", Polarity::Trouble, true, reason, message);
    if view.down.is_some() {
        Observed::new(
            "Degraded",
            Polarity::Trouble,
            false,
            "Down",
            "the engine is down",
        )
    } else if !view.lost.is_empty() {
        trouble("Shrunk", format!("ranks {:?} lost under SHRINK", view.lost))
    } else if let Some(h) = &view.head_conflict {
        trouble(
            "SecondHead",
            format!(
                "a second head registered on {}{}",
                h.process.node,
                h.process
                    .pid
                    .map(|p| format!(" (pid {p})"))
                    .unwrap_or_default()
            ),
        )
    } else if let Some(why) = members_differ(view) {
        trouble("MembersDiffer", why)
    } else if let Some(why) = super::transition::not_serving(view) {
        trouble("NotServingChanges", why.to_string())
    } else if let Some((rank, leaks)) = view.leaks.iter().next() {
        trouble(
            "ReleaseLeaked",
            format!(
                "{} processes left communicator releases behind; rank {rank}'s on {}: {}",
                view.leaks.len(),
                leaks.node,
                leaks.state()
            ),
        )
    } else if let Some(((node, device), (_, why))) = view.refused_spares.iter().next() {
        trouble(
            "SpareRefused",
            format!(
                "{} spares refused; {node} device {device}: {why}",
                view.refused_spares.len()
            ),
        )
    } else {
        Observed::new("Degraded", Polarity::Trouble, false, "AsPolicy", "")
    }
}

/// The Down condition's reason for a cause of 0032's.
fn down_reason(cause: &str) -> &'static str {
    match cause {
        "stopped" => "HeadStopped",
        "failed" => "HeadFailed",
        "change_failed" => "ChangeFailed",
        _ => "HeadLost",
    }
}

/// An event's reason for a change that did not commit.
fn change_error_reason(e: &ChangeError) -> &'static str {
    match e {
        ChangeError::Rebuild(_) => "RefusedRebuild",
        ChangeError::Unknown(_) => "UnknownRank",
        ChangeError::Unheard { .. } => "UnheardRanks",
        ChangeError::Survivor { .. } => "SurvivorError",
        ChangeError::Departed { .. } => "SurvivorDeparted",
        ChangeError::Down { .. } => "EngineDown",
        ChangeError::Node { .. } => "NoDelivery",
        ChangeError::Epoch { .. } => "EpochMismatch",
        ChangeError::Stale { .. } => "ReportFolded",
        ChangeError::Domain { .. } => "Domain",
        ChangeError::Uncovered { .. } => "RefusedUncovered",
        ChangeError::Deposed { .. } => "Deposed",
        ChangeError::LeaseLost { .. } => "LeaseLost",
        ChangeError::Unlogged { .. } => "Unlogged",
        ChangeError::NotServing { .. } => "HeadNotServing",
    }
}

#[cfg(test)]
mod tests {
    use super::instance_state;
    use crate::coord::view::{Down, EngineView};
    use crate::proto::everserve::signals::v1::InstancePhase;

    fn phase(view: &EngineView) -> InstancePhase {
        instance_state("e", view, 4).phase()
    }

    #[test]
    fn an_instance_state_is_the_first_that_holds() {
        let mut view = EngineView::default();
        view.members.insert(0, "node-a".into());
        assert_eq!(phase(&view), InstancePhase::InstanceStarting, "bring-up");
        for r in 1..4 {
            view.members.insert(r, "node-a".into());
        }
        assert_eq!(
            phase(&view),
            InstancePhase::InstanceServing,
            "every position held"
        );
        // a shrink took rank 3: serving, with its capacity down
        view.members.remove(&3);
        view.lost = vec![3];
        assert_eq!(phase(&view), InstancePhase::InstanceDegraded);
        // a change in flight goes before degraded: the framework holds its recovery off
        view.change = Some((2, vec![2]));
        assert_eq!(phase(&view), InstancePhase::InstanceRecovering);
        // and down before everything: the framework's own recovery takes over
        view.down = Some(Down {
            at_ms: 0,
            cause: "lost",
            detail: "its node is gone".into(),
        });
        let failed = instance_state("e", &view, 4);
        assert_eq!(failed.phase(), InstancePhase::InstanceFailed);
        assert_eq!(failed.reason, "lost: its node is gone");
        assert_eq!((failed.positions_held, failed.world), (3, 4));
    }
}
