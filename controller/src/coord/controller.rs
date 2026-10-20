// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The engine controller (decision 0036): losses are recorded against their engine, a work queue
//! keyed by engine wakes its reconcile, and the reconcile folds every loss it finds into one
//! transition, starts it, and returns. The transition runs as its own task; its outcome is an
//! event that answers every report it folded and wakes the engine again.
//!
//! The node and the executor detect a loss, fence the dead process and refuse its groups without
//! waiting for any of this (0025, 0023): the controller only chooses what replaces it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use std::time::Instant;

use tokio::sync::{Mutex, oneshot};
use tracing::{info, warn};

use crate::clock::now_us;
use crate::proto::everserve::signals::v1::MembershipCommitted;
use crate::status::Severity;

use super::queue::WorkQueue;
use super::{Cause, ChangeError, Coordinator};
use crate::metrics::CoordMetrics;

/// What a report of a loss gets back: the outcome of the transition that handled it.
pub type Answer = Result<MembershipCommitted, ChangeError>;

/// One report of lost ranks, held until a reconcile folds it into a transition.
#[derive(Debug)]
pub(super) struct Report {
    /// The ranks reported lost.
    pub(super) ranks: Vec<u32>,
    /// The epoch the reporter saw them in; a rank changed since is folded away (decision 0021).
    pub(super) seen: Option<u64>,
    /// A fault, or a notice's deadline.
    pub(super) cause: Cause,
    /// When the report reached the coordinator, on its clock: the change's decision latency
    /// is measured from here.
    pub(super) reported_at_us: u64,
    /// The reporter waiting for the outcome.
    pub(super) answer: oneshot::Sender<Answer>,
}

/// The controller's state: losses not yet folded, and the queue of engines to reconcile.
pub(super) struct Controller {
    /// Reports per engine, in the order they arrived.
    pending: Mutex<BTreeMap<String, Vec<Report>>>,
    /// Engines with a transition running: at most one per engine.
    running: Mutex<BTreeSet<String>>,
    /// Engines queued and not yet taken by the worker; a key queued twice runs once.
    queue: WorkQueue<String>,
}

impl Controller {
    pub(super) fn new(metrics: &CoordMetrics) -> Self {
        Self {
            pending: Mutex::new(BTreeMap::new()),
            running: Mutex::new(BTreeSet::new()),
            queue: WorkQueue::new(metrics, "engine"),
        }
    }

    /// Wake `engine`'s reconcile, once however often it is asked before it runs.
    pub(super) async fn enqueue(&self, engine: &str) {
        self.queue.enqueue(engine.to_owned()).await;
    }

    /// Losses waiting to be folded, across engines: what a shutdown waits for.
    pub(super) async fn backlog(&self) -> usize {
        self.pending
            .lock()
            .await
            .values()
            .map(Vec::len)
            .sum::<usize>()
            + self.running.lock().await.len()
    }
}

impl Coordinator {
    /// Record that `ranks` of `engine` are lost, wake its reconcile, and wait for the outcome of
    /// the transition that handles them. Every rank reported before that transition starts is
    /// folded into it: a whole node's loss is one change, however many reports announce it.
    pub(super) async fn report_loss(
        &self,
        engine: &str,
        ranks: &[u32],
        seen: Option<u64>,
        cause: Cause,
    ) -> Answer {
        let (tx, rx) = oneshot::channel();
        self.controller
            .pending
            .lock()
            .await
            .entry(engine.to_owned())
            .or_default()
            .push(Report {
                ranks: ranks.to_vec(),
                seen,
                cause,
                reported_at_us: now_us(),
                answer: tx,
            });
        self.start_controller().await;
        self.controller.enqueue(engine).await;
        rx.await.unwrap_or(Err(ChangeError::Unknown(u32::MAX)))
    }

    /// Wake `engine`'s reconcile, starting the controller's worker if it is not running yet.
    pub(super) async fn wake(&self, engine: &str) {
        self.start_controller().await;
        self.controller.enqueue(engine).await;
    }

    /// Start the controller's worker on first use, once: it takes each queued engine and
    /// reconciles it. A reconcile never waits on I/O, so one worker serves every engine.
    async fn start_controller(&self) {
        let Some(mut rx) = self.controller.queue.receiver().await else {
            return;
        };
        let Some(me) = self.me.upgrade() else { return };
        tokio::spawn(async move {
            while let Some(engine) = rx.recv().await {
                me.controller.queue.take(&engine).await;
                let started = Instant::now();
                let outcome = Arc::clone(&me).reconcile(&engine).await;
                me.metrics
                    .control
                    .reconcile_seconds
                    .with_label_values(&["engine"])
                    .observe(started.elapsed().as_secs_f64());
                me.metrics
                    .control
                    .reconciles_total
                    .with_label_values(&["engine", outcome])
                    .inc();
            }
        });
    }

    /// One reconcile of `engine`: if no transition of it is running, fold every pending loss
    /// into one, start it as its own task, and return. A transition already running leaves the
    /// losses pending; its outcome wakes the engine again.
    /// Returns what it did: `running` (a transition is, so the losses wait), `idle` (none
    /// pending) or `started`.
    async fn reconcile(self: Arc<Self>, engine: &str) -> &'static str {
        if self.controller.running.lock().await.contains(engine) {
            return "running";
        }
        // a change a previous coordinator logged goes first: no other change of the engine is
        // claimed until it is finished (decision 0038); the losses reported meanwhile wait
        if self.unfinished(engine).await {
            if !self.ready_to_finish(engine).await {
                self.wake_at_settle(engine).await;
                return "waiting";
            }
            self.controller
                .running
                .lock()
                .await
                .insert(engine.to_owned());
            let me = Arc::clone(&self);
            let engine = engine.to_owned();
            tokio::spawn(async move {
                me.finish_logged(&engine).await;
                me.controller.running.lock().await.remove(&engine);
                me.controller.enqueue(&engine).await;
            });
            return "resumed";
        }
        let reported = self
            .controller
            .pending
            .lock()
            .await
            .get(engine)
            .is_some_and(|r| !r.is_empty());
        if !reported {
            return "idle";
        }
        if !self.may_claim(engine).await && !self.deposed() {
            // ranks a kept membership names and no daemon has relinked yet: the takeover's
            // allowance hears them or counts them lost, and wakes the engine either way
            if self.awaits_unheard() {
                return "waiting";
            }
            // past the allowance with ranks still unknown (an engine down meanwhile): nothing will
            // come, so nothing waits, and a report is answered so
            self.refuse_unheard(engine).await;
            return "refused";
        }
        let reports = self
            .controller
            .pending
            .lock()
            .await
            .remove(engine)
            .unwrap_or_default();
        self.controller
            .running
            .lock()
            .await
            .insert(engine.to_owned());
        let me = Arc::clone(&self);
        let engine = engine.to_owned();
        tokio::spawn(async move {
            me.transition(&engine, reports).await;
            me.controller.running.lock().await.remove(&engine);
            me.controller.enqueue(&engine).await;
        });
        "started"
    }

    /// Answer every report of `engine` waiting for ranks no daemon will bring: refused, counted
    /// and recorded, naming them.
    async fn refuse_unheard(&self, engine: &str) {
        let ranks = self.unheard_ranks(engine).await;
        let reports = self
            .controller
            .pending
            .lock()
            .await
            .remove(engine)
            .unwrap_or_default();
        let refused = ChangeError::Unheard { ranks };
        for r in reports {
            self.metrics
                .changes_total
                .with_label_values(&[engine, r.cause.label(), "none", "unheard_ranks"])
                .inc();
            let _ = r.answer.send(Err(refused.clone()));
        }
        warn!("{engine}: a change refused: {refused}");
        self.engine_event(
            engine,
            Severity::Warning,
            "UnheardRanks",
            refused.to_string(),
        )
        .await;
    }

    /// The transition for `reports` of `engine`: one change for every rank among them, its
    /// outcome given to every report it folded. A report whose ranks all changed since the epoch
    /// it saw folds away with its own answer, as it would alone (decision 0021).
    async fn transition(&self, engine: &str, reports: Vec<Report>) {
        let (ranks, answers, faulted, reported_at_us) = self.fold(engine, reports).await;
        if ranks.is_empty() {
            return;
        }
        if answers.len() > 1 {
            info!(
                "{engine}: {} reports folded into one change for ranks {ranks:?}",
                answers.len()
            );
        }
        let answer = self.change(engine, &ranks, &faulted, reported_at_us).await;
        if let Err(e) = &answer {
            warn!("{engine}: the change for ranks {ranks:?} did not commit: {e}");
        }
        for tx in answers {
            let _ = tx.send(answer.clone());
        }
        // a drain may have emptied its node, and one waiting its turn may start
        self.wake_drains().await;
    }

    /// Fold `reports` into the ranks one change takes: each report's ranks less those changed
    /// since the epoch it saw. A report with no rank, or left with none, is answered at once, as
    /// it would be alone. With them, the ranks a fault report named, which are dead; every other
    /// rank the change takes leaves alive. The report time is the earliest.
    async fn fold(
        &self,
        engine: &str,
        reports: Vec<Report>,
    ) -> (Vec<u32>, Vec<oneshot::Sender<Answer>>, BTreeSet<u32>, u64) {
        let changed_at = self
            .engines
            .lock()
            .await
            .get(engine)
            .map(|v| v.changed_at.clone())
            .unwrap_or_default();
        let mut ranks = BTreeSet::new();
        let mut answers = Vec::new();
        let mut faulted = BTreeSet::new();
        let mut reported_at_us = u64::MAX;
        for r in reports {
            let Some(&first) = r.ranks.first() else {
                let _ = r.answer.send(Err(ChangeError::Unknown(0)));
                continue;
            };
            let live: Vec<u32> = r
                .ranks
                .iter()
                .copied()
                .filter(|rank| {
                    r.seen
                        .is_none_or(|seen| changed_at.get(rank).is_none_or(|&c| c <= seen))
                })
                .collect();
            if live.is_empty() {
                let _ = r.answer.send(Err(ChangeError::Stale {
                    rank: first,
                    seen: r.seen.unwrap_or_default(),
                    changed: changed_at.get(&first).copied().unwrap_or_default(),
                }));
                continue;
            }
            if r.cause == Cause::Fault {
                faulted.extend(live.iter().copied());
            }
            ranks.extend(live);
            reported_at_us = reported_at_us.min(r.reported_at_us);
            answers.push(r.answer);
        }
        (
            ranks.into_iter().collect(),
            answers,
            faulted,
            reported_at_us,
        )
    }
}
