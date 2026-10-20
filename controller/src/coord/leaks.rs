// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0

//! Releases a serving process left behind (decision 0025, amended 2026-10-15). A refused
//! communicator's release still running past its bound holds that communicator's memory and a
//! thread until the process ends. Serving is unaffected, so the process is not killed: the
//! coordinator counts the leak, says the engine is degraded, and retires the process by a notice
//! for its rank once a spare can take it with no gap, through the same planned change a drain
//! makes. While no spare can, the process stays, degraded and counted, and the retirement is
//! tried again whenever the engine's status moves, a plan that covers the rank among it. A
//! second leak while the notice is out adds to the count, never a second notice.

use tracing::{info, warn};

use crate::clock::now_ms;
use crate::proto::everserve::coord::v1::ReleaseLeakedAt;
use crate::proto::everserve::signals::v1::{Notice, Reason};
use crate::status::Severity;

use super::Coordinator;

/// What leaks to the notice's deadline beyond the policy's minimum lead: the notice's way to
/// every daemon of the engine.
const NOTICE_TRANSIT_MS: i64 = 1000;

/// The releases one rank's process left behind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Leaks {
    /// The node its process runs on: a later holder of the rank is another process.
    pub node: String,
    /// Every release it left behind, by name, in the order reported.
    pub releases: Vec<String>,
    /// Its own count, as it last reported it.
    pub total: u32,
    /// The notice retiring it went out, at this deadline.
    pub retiring_at_ms: Option<i64>,
}

impl Leaks {
    /// Where its retirement stands, in words.
    #[must_use]
    pub fn state(&self) -> String {
        let last = self.releases.last().map_or("", String::as_str);
        match self.retiring_at_ms {
            Some(at) => format!(
                "{} left behind, the last {last}; retired by notice at {at}",
                self.total
            ),
            None => format!(
                "{} left behind, the last {last}; waiting for a spare that can take the rank",
                self.total
            ),
        }
    }
}

impl Coordinator {
    /// `node`'s daemon forwarded a process's leaked release: counted, recorded, and the process
    /// retired when a spare can take its rank now. A report about a rank `node` does not hold is
    /// a process that left it, and changes nothing.
    pub(super) async fn release_leaked(&self, node: &str, at: ReleaseLeakedAt) {
        let Some(leak) = at.leak else { return };
        let engine = leak.engine.clone();
        let held = {
            let mut engines = self.engines.lock().await;
            let Some(view) = engines.get_mut(&engine) else {
                return;
            };
            let held = view.members.get(&leak.rank).is_some_and(|n| n == node);
            if held {
                let entry = view.leaks.entry(leak.rank).or_insert_with(|| Leaks {
                    node: node.to_owned(),
                    releases: Vec::new(),
                    total: 0,
                    retiring_at_ms: None,
                });
                entry.releases.push(leak.release.clone());
                entry.total = entry.total.max(leak.leaked_total);
            }
            held
        };
        if !held {
            info!(
                "{engine}: a leaked release from rank {} on {node}, which holds it no longer",
                leak.rank
            );
            return;
        }
        self.metrics
            .releases_leaked_total
            .with_label_values(&[&engine])
            .inc();
        let why = format!(
            "rank {} on {node} left release {} behind after {} ms ({} in all): its communicator's \
             memory is held until the process ends, so the process is retired once a spare can \
             take the rank",
            leak.rank, leak.release, leak.run_ms, leak.leaked_total
        );
        warn!("{engine}: {why}");
        self.engine_event(&engine, Severity::Warning, "ReleaseLeaked", why)
            .await;
        self.observed(&engine).await;
    }

    /// Retire each process of `engine` that left a release behind, by a notice for its rank,
    /// when a spare can take the rank now: a set that answers its loss is covered, or, with no
    /// cover planned, a spare stands for it. One notice per process; ranks whose holder moved are
    /// forgotten.
    pub(super) async fn retire_leaked(&self, engine: &str) {
        let lead_ms = i64::from(self.policy.notice_min_ms) + NOTICE_TRANSIT_MS;
        let planned = self.policy.cover.as_ref().is_some_and(|c| c.enabled);
        let due: Vec<(u32, i64)> = {
            let mut engines = self.engines.lock().await;
            let Some(view) = engines.get_mut(engine) else {
                return;
            };
            // a rank another process holds now: its leaks were the process that left
            let members = view.members.clone();
            view.leaks
                .retain(|r, l| members.get(r).is_some_and(|n| *n == l.node));
            if view.change.is_some() || view.down.is_some() {
                return;
            }
            let ready: Vec<u32> = view
                .leaks
                .iter()
                .filter(|(_, l)| l.retiring_at_ms.is_none())
                .map(|(r, _)| *r)
                .filter(|r| {
                    if planned {
                        view.cover.covered_answer(&[*r]).is_some()
                    } else {
                        view.spares.contains_key(r)
                    }
                })
                .collect();
            let deadline = now_ms() + lead_ms;
            for r in &ready {
                if let Some(l) = view.leaks.get_mut(r) {
                    l.retiring_at_ms = Some(deadline);
                }
            }
            ready.into_iter().map(|r| (r, deadline)).collect()
        };
        let Some(me) = self.me.upgrade() else { return };
        for (rank, deadline_ms) in due {
            info!(
                "{engine}: rank {rank}'s process retired by notice at {deadline_ms}, a spare taking it"
            );
            self.engine_event(
                engine,
                Severity::Normal,
                "ProcessRetired",
                format!("rank {rank}'s process, which left a release behind, leaves by notice at {deadline_ms}"),
            )
            .await;
            let notice = Notice {
                rank,
                deadline_ms,
                reason: Reason::Retired as i32,
            };
            tokio::spawn(retire(
                std::sync::Arc::clone(&me),
                engine.to_owned(),
                notice,
            ));
        }
    }
}

/// Send `notice` for `engine`. Its own function, typed as `Send`: the notice's change ends in a
/// reconcile, which may retire again.
fn retire(
    me: std::sync::Arc<Coordinator>,
    engine: String,
    notice: Notice,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(async move {
        me.notice(&engine, notice).await;
    })
}
