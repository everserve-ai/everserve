// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Which process serves a rank is decided once (decision 0041). A process of a framework's group
//! registers undecided for its rank before it loads anything, and is told whether it serves the
//! rank or stands as its spare. Exactly one process is told to serve a rank: the answer is an
//! assignment, written to the engine's status in the store before it goes out, so a coordinator's
//! restart never gives it twice. It holds the rank while its process loads, minutes perhaps, and is
//! spent when the process registers as the rank's worker; it is released if the process leaves
//! first or its node is lost. A rank that has had a holder is never served by a registration: its
//! holder changes only through a committed change, so every later asker stands as its spare.

use tracing::{info, warn};

use std::collections::BTreeSet;

use crate::proto::everserve::coord::v1::{
    ConnectionRef, CoordToDaemon, DaemonHello, EngineState, RoleAnswer, RoleAsked, coord_to_daemon,
};
use crate::proto::everserve::signals::v1::Role;
use crate::proto::everserve::spec::v1::Assignment;

use super::Coordinator;
use super::cover::SETTLE;

impl Coordinator {
    /// An engine that is down (decision 0032: its head failed or was lost) and holds no process
    /// of its old incarnation starts anew when a process of it asks: its framework restarted it.
    /// The old incarnation's departed and lost ranks, its change, its step config and its
    /// assignments go with it, so no rank of the new one counts as having had a holder, the
    /// asker is decided as at bring-up, and its step config is the new one's. While a process of the old incarnation still serves, nothing restarts: two
    /// incarnations never mix, and the restart comes with the last one's departure. Its epochs
    /// keep rising, so no daemon sees one go back.
    async fn restart_if_down(&self, engine: &str) {
        let was = {
            let mut engines = self.engines.lock().await;
            let Some(view) = engines.get_mut(engine) else {
                return;
            };
            if view.down.is_none() || !view.members.is_empty() {
                return;
            }
            let was = view.down.take();
            view.departed.clear();
            view.lost.clear();
            view.changed_at.clear();
            view.change = None;
            view.step = None;
            was
        };
        let Some(was) = was else { return };
        info!(
            "{engine}: down ({}: {}) and holding no process of it, starts anew as a process of it \
             asks",
            was.cause, was.detail
        );
        self.engine_event(
            engine,
            crate::status::Severity::Normal,
            "EngineRestarted",
            format!(
                "down ({}: {}), its processes all gone; a process of it asked, and it starts anew",
                was.cause, was.detail
            ),
        )
        .await;
        self.metrics
            .engines_down
            .with_label_values(&[engine])
            .set(0);
        self.release_assignments(engine, "the engine starts anew", |_| true)
            .await;
    }

    /// Why `asked`'s process may not serve its engine, and the event's reason: it was started for
    /// another guarantee class than its engine's in the policy (`GuaranteeRefused`), or its step
    /// config differs from the engine's, the first one a process of the engine reported, which
    /// this one becomes when it is the first (`StepRefused`). A process whose collectives would
    /// not match its engine's is refused before it loads rather than left to hang in them.
    async fn refusal(&self, asked: &RoleAsked) -> Option<(&'static str, String)> {
        if let Some(why) = self.guarantee_refusal(asked) {
            return Some(("GuaranteeRefused", why));
        }
        self.step_refusal(asked)
            .await
            .map(|why| ("StepRefused", why))
    }

    /// Why `asked`'s process was started for another guarantee class than its engine's.
    fn guarantee_refusal(&self, asked: &RoleAsked) -> Option<String> {
        let wanted = match self.engine_spec(&asked.engine) {
            Some(e) if crate::policy::exact(e) => "exact",
            Some(_) => "continuous",
            None => return None,
        };
        let here = asked.identity.as_ref().map_or("", |i| i.guarantee.as_str());
        if here.is_empty() {
            Some(format!(
                "the guarantee is empty here, {wanted:?} in the policy"
            ))
        } else if here != wanted {
            Some(format!(
                "the guarantee is {here:?} here, {wanted:?} in the policy"
            ))
        } else {
            None
        }
    }

    /// Why `asked`'s step config differs from its engine's.
    async fn step_refusal(&self, asked: &RoleAsked) -> Option<String> {
        let here = asked.identity.as_ref().and_then(|i| i.step.as_ref());
        let mut engines = self.engines.lock().await;
        let view = engines.entry(asked.engine.clone()).or_default();
        if let Some(engine) = &view.step {
            return super::capability::step_difference(here, Some(engine));
        }
        let refused = super::capability::step_difference(here, here);
        if refused.is_none() {
            view.step = here.cloned();
        }
        refused
    }

    /// Send `answer` to `node`'s daemon.
    async fn send_answer(&self, node: &str, answer: RoleAnswer) {
        let Some(tx) = self.daemons.lock().await.get(node).map(|d| d.tx.clone()) else {
            return;
        };
        let _ = tx.send(CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::RoleAnswer(answer)),
        });
    }

    /// An undecided process on `node` asked what it is: answered on `node`'s link, unless its
    /// assignment could not be written, when it is not answered and the process asks again.
    pub(super) async fn answer_role(&self, node: &str, asked: RoleAsked) {
        // the daemon holds the engine's facts before the answer: it hands the process its policy
        // as it tells it what it is
        self.reached(&asked.engine, node).await;
        self.restart_if_down(&asked.engine).await;
        if let Some((reason, why)) = self.refusal(&asked).await {
            warn!(
                "{}: rank {}'s process on {node} is refused: {why}",
                asked.engine, asked.rank
            );
            // an operator reads it in the engine's events, beside the process's own exit
            self.engine_event(
                &asked.engine,
                crate::status::Severity::Warning,
                reason,
                format!("rank {}'s process on {node}: {why}", asked.rank),
            )
            .await;
            self.send_answer(
                node,
                RoleAnswer {
                    request_id: asked.request_id,
                    engine: asked.engine.clone(),
                    rank: asked.rank,
                    refusal: why,
                    ..RoleAnswer::default()
                },
            )
            .await;
            return;
        }
        let Some((epoch, role)) = self.decide_role(node, &asked).await else {
            return;
        };
        info!(
            "{}: rank {}'s process {} on {node} is told to {}",
            asked.engine,
            asked.rank,
            pid_of(asked.connection.as_ref()),
            if role == Role::Worker {
                "serve it"
            } else {
                "stand as its spare"
            }
        );
        self.send_answer(
            node,
            RoleAnswer {
                request_id: asked.request_id,
                engine: asked.engine.clone(),
                rank: asked.rank,
                epoch,
                role: role as i32,
                refusal: String::new(),
            },
        )
        .await;
        if role == Role::Worker {
            // an assignment may place the last rank: the engine's processes learn where it runs
            self.push_placement(&asked.engine).await;
        }
    }

    /// What `asked`'s process is, and the engine's committed epoch; None when it is to serve and
    /// the store did not keep its assignment. Decided under the engine's status's lock, against
    /// the assignments the store keeps, so two askers never both serve.
    async fn decide_role(&self, node: &str, asked: &RoleAsked) -> Option<(u64, Role)> {
        // a stage's ranks move together (decision 0022): one answer covers them all
        let ranks: Vec<u32> = if asked.stage.is_empty() {
            vec![asked.rank]
        } else {
            asked.stage.clone()
        };
        let (held, epoch) = self
            .engines
            .lock()
            .await
            .get(&asked.engine)
            .map_or((false, 0), |v| {
                (ranks.iter().any(|r| v.held_ever(*r)), v.committed_epoch)
            });
        let linked: BTreeSet<String> = self.daemons.lock().await.keys().cloned().collect();
        let settled = self.started.elapsed() >= SETTLE;
        let decided = self
            .update_status(&asked.engine, |s| {
                // a process on a node that has not linked since this coordinator started, once
                // the daemons have had their time to: its node is gone
                if settled {
                    s.assignments.retain(|a| linked.contains(&a.node));
                }
                let assigned: Vec<&Assignment> = s
                    .assignments
                    .iter()
                    .filter(|a| ranks.contains(&a.rank))
                    .collect();
                let mine = |a: &&Assignment| {
                    a.node == node && same(a.connection.as_ref(), asked.connection.as_ref())
                };
                if !assigned.is_empty() {
                    // the same process asking again for the whole stage: its answer stands
                    return if assigned.len() == ranks.len() && assigned.iter().all(mine) {
                        Role::Worker
                    } else {
                        Role::Spare
                    };
                }
                if held {
                    return Role::Spare;
                }
                for rank in &ranks {
                    s.assignments.push(Assignment {
                        rank: *rank,
                        epoch,
                        node: node.to_owned(),
                        connection: asked.connection,
                        host: asked.host.clone(),
                    });
                }
                Role::Worker
            })
            .await;
        match decided {
            Ok(role) => Some((epoch, role)),
            Err(e) => {
                warn!(
                    "{}: rank {}'s process {} on {node} not answered: the store did not keep its \
                     assignment: {e}",
                    asked.engine,
                    asked.rank,
                    pid_of(asked.connection.as_ref())
                );
                None
            }
        }
    }

    /// Release `engine`'s assignments that `gone` names.
    pub(super) async fn release_assignments(
        &self,
        engine: &str,
        why: &str,
        gone: impl Fn(&Assignment) -> bool + Send + Sync,
    ) {
        let released = self
            .update_status(engine, |s| {
                let released: Vec<Assignment> =
                    s.assignments.iter().filter(|a| gone(a)).cloned().collect();
                s.assignments.retain(|a| !gone(a));
                released
            })
            .await;
        match released {
            Ok(released) => {
                for a in released {
                    info!(
                        "{engine}: rank {}'s assignment to process {} on {} released: {why}",
                        a.rank,
                        pid_of(a.connection.as_ref()),
                        a.node
                    );
                }
            }
            // kept until a later write; a restart then finds its process gone
            Err(e) => warn!("{engine}: released assignments not written: {e}"),
        }
    }

    /// `engine`'s processes told to serve a rank and not yet its worker.
    #[must_use]
    pub fn assignments(&self, engine: &str) -> Vec<Assignment> {
        self.kept_status(engine)
            .map(|s| s.assignments)
            .unwrap_or_default()
    }

    /// `hello`'s node holds, of the processes told to serve a rank there, only those it names
    /// as undecided or as the rank's worker: the others are gone.
    pub(super) async fn release_gone(&self, hello: &DaemonHello) {
        let engines: Vec<String> = self
            .specs
            .written
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .cloned()
            .collect();
        for engine in engines {
            let state = hello.engines.iter().find(|s| s.engine == engine);
            self.release_assignments(&engine, "its process is gone from its node", |a| {
                a.node == hello.node && !named(state, a)
            })
            .await;
        }
    }
}

/// Whether `state` names `a`'s process: undecided on its connection, or the rank's worker.
fn named(state: Option<&EngineState>, a: &Assignment) -> bool {
    state.is_some_and(|s| {
        s.workers.contains(&a.rank)
            || s.undecided
                .iter()
                .any(|u| u.rank == a.rank && same(u.connection.as_ref(), a.connection.as_ref()))
    })
}

/// Whether `a` and `b` name one connection: the same daemon instance and serial. The pid is for
/// people only.
pub(super) fn same(a: Option<&ConnectionRef>, b: Option<&ConnectionRef>) -> bool {
    matches!((a, b), (Some(a), Some(b)) if a.instance == b.instance && a.serial == b.serial)
}

/// The pid a connection reported, for a log line.
fn pid_of(c: Option<&ConnectionRef>) -> u32 {
    c.map_or(0, |c| c.pid)
}
