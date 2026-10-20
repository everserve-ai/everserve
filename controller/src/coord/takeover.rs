// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0

//! A coordinator that starts or takes the lease (decision 0036, amended 2026-10-15) knows an
//! engine's ranks only from the daemons' hellos. It claims no change until it knows every rank,
//! and never counts a rank lost for being unheard alone: a coordinator's restart must not become
//! a change of serving ranks. A rank is lost by the node-loss rule, applied to a node it has
//! never heard: a daemon that can reach it relinks within its longest backoff, so one not heard
//! within that and the node-loss allowance after the first daemon's hello cannot reach it. The
//! first hello starts the clock, not the listening, since it shows the daemons' whole path
//! (Service, DNS, TLS, hello handling) carries. Only ranks of a membership kept for the engine's
//! incarnation are counted: an engine with none kept never served, and its ranks still
//! registering are not lost. A coordinator no daemon reaches is the one cut off, says so, and
//! changes nothing.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::proto::everserve::spec::v1::KeptMember;
use crate::status::Severity;

use super::{Cause, ChangeError, Coordinator};

/// How long after no daemon has reached it a coordinator, with engines that served before it,
/// says it is unreachable.
pub const UNREACHABLE_AFTER: Duration = Duration::from_mins(1);

/// How long after the first daemon's hello a rank still unheard is lost: the longest a daemon's
/// reconnect waits, jitter included, and then the node-loss allowance at the default heartbeat.
#[must_use]
pub fn default_allowance() -> Duration {
    let backoff = crate::node::RECONNECT_MAX + crate::node::RECONNECT_MAX / 4;
    let misses = u32::try_from(super::DEFAULT_NODE_LOSS_MISSES).unwrap_or(u32::MAX);
    backoff + crate::node::DEFAULT_HEARTBEAT * misses
}

/// What a coordinator knows of hearing the cell again.
pub(super) struct Takeover {
    /// The stop of this coordinator's serving, which ends the clocks below with it.
    serving: std::sync::Mutex<Option<CancellationToken>>,
    first_hello: std::sync::Mutex<Option<Instant>>,
    heard: std::sync::Mutex<std::collections::BTreeSet<String>>,
    allowance_ms: AtomicU64,
    unreachable_after_ms: AtomicU64,
}

impl Default for Takeover {
    fn default() -> Self {
        Self {
            serving: std::sync::Mutex::new(None),
            first_hello: std::sync::Mutex::new(None),
            heard: std::sync::Mutex::new(std::collections::BTreeSet::new()),
            allowance_ms: AtomicU64::new(millis(default_allowance())),
            unreachable_after_ms: AtomicU64::new(millis(UNREACHABLE_AFTER)),
        }
    }
}

fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

impl Coordinator {
    /// Count a rank of a kept membership lost when no daemon holds it `allowance` after the first
    /// daemon's hello; [`default_allowance`] unless set. Set before the first hello.
    pub fn set_takeover_allowance(&self, allowance: Duration) {
        self.takeover
            .allowance_ms
            .store(millis(allowance), Ordering::Relaxed);
    }

    /// Say this coordinator is unreachable when no daemon reached it `after` it started serving;
    /// [`UNREACHABLE_AFTER`] unless set. Set before it serves.
    pub fn set_unreachable_after(&self, after: Duration) {
        self.takeover
            .unreachable_after_ms
            .store(millis(after), Ordering::Relaxed);
    }

    /// Start watching for the cell's daemons as this coordinator serves, until `cancel`.
    pub(super) fn watch_takeover(&self, cancel: &CancellationToken) {
        if let Ok(mut s) = self.takeover.serving.lock() {
            *s = Some(cancel.clone());
        }
        let Some(me) = self.me.upgrade() else { return };
        let cancel = cancel.clone();
        let after =
            Duration::from_millis(self.takeover.unreachable_after_ms.load(Ordering::Relaxed));
        tokio::spawn(async move {
            tokio::select! {
                () = cancel.cancelled() => {}
                () = tokio::time::sleep(after) => me.unreachable_if_unheard(after),
            }
        });
    }

    fn unreachable_if_unheard(&self, after: Duration) {
        let heard = self.takeover.first_hello.lock().is_ok_and(|f| f.is_some());
        let served = self
            .engine_names()
            .iter()
            .any(|e| self.kept_members(e).is_some());
        if !heard && served {
            error!(
                "no daemon has reached this coordinator {after:?} after it started serving: it is \
                 the one cut off, and changes nothing until one does"
            );
            self.metrics.control.takeover.unreachable.set(1);
        }
    }

    /// `node`'s daemon said hello: the first starts the clock after which a rank still unheard
    /// is lost; each node's first is measured from it.
    pub(super) fn heard(&self, node: &str) {
        let now = Instant::now();
        let first = {
            let Ok(mut f) = self.takeover.first_hello.lock() else {
                return;
            };
            if let Some(at) = *f {
                Err(at)
            } else {
                *f = Some(now);
                Ok(now)
            }
        };
        let newly = self
            .takeover
            .heard
            .lock()
            .is_ok_and(|mut h| h.insert(node.to_owned()));
        if newly {
            let since = first.unwrap_or_else(|at| at);
            self.metrics
                .control
                .takeover
                .relink_seconds
                .observe(now.duration_since(since).as_secs_f64());
        }
        if first.is_err() {
            return;
        }
        self.metrics.control.takeover.unreachable.set(0);
        let Some(me) = self.me.upgrade() else { return };
        let allowance = Duration::from_millis(self.takeover.allowance_ms.load(Ordering::Relaxed));
        let cancel = self
            .takeover
            .serving
            .lock()
            .ok()
            .and_then(|s| s.clone())
            .unwrap_or_default();
        tokio::spawn(async move {
            tokio::select! {
                () = cancel.cancelled() => {}
                () = tokio::time::sleep(allowance) => me.lose_unheard(allowance).await,
            }
        });
    }

    /// The allowance after the first hello has passed: every rank of a kept membership that no
    /// daemon holds is lost, as a node gone is.
    async fn lose_unheard(&self, allowance: Duration) {
        // every engine looks again at what waited for this, lost ranks or none
        for engine in self.engine_names() {
            self.wake(&engine).await;
        }
        for engine in self.engine_names() {
            let Some(kept) = self.kept_members(&engine) else {
                continue;
            };
            let lost: Vec<(u32, String, u64)> = {
                let mut engines = self.engines.lock().await;
                let Some(view) = engines.get_mut(&engine) else {
                    continue;
                };
                if view.down.is_some() {
                    continue;
                }
                let unheard: Vec<(u32, String)> = kept
                    .into_iter()
                    .filter(|(r, _)| {
                        !view.members.contains_key(r)
                            && !view.departed.contains(r)
                            && !view.lost.contains(r)
                    })
                    .collect();
                for (r, _) in &unheard {
                    view.departed.insert(*r);
                }
                unheard
                    .into_iter()
                    .map(|(r, n)| (r, n, view.epoch))
                    .collect()
            };
            if lost.is_empty() {
                continue;
            }
            let ranks: Vec<u32> = lost.iter().map(|(r, _, _)| *r).collect();
            let nodes: std::collections::BTreeSet<&str> =
                lost.iter().map(|(_, n, _)| n.as_str()).collect();
            let seen = lost.first().map_or(0, |(_, _, e)| *e);
            let why = format!(
                "ranks {ranks:?} on {nodes:?} not heard {allowance:?} after the first daemon \
                 relinked to this coordinator: lost, as a node gone is"
            );
            warn!("{engine}: {why}");
            self.metrics
                .control
                .takeover
                .unheard_lost_total
                .with_label_values(&[&engine])
                .inc_by(u64::try_from(ranks.len()).unwrap_or(u64::MAX));
            self.engine_event(&engine, Severity::Warning, "RanksUnheard", why)
                .await;
            let Some(me) = self.me.upgrade() else { return };
            tokio::spawn(async move {
                match me.ranks_reported(&engine, &ranks, seen, Cause::Fault).await {
                    Ok(_) => {}
                    Err(e @ ChangeError::Stale { .. }) => info!("{engine}: report folded: {e}"),
                    Err(e) => warn!("{engine}: change for unheard ranks {ranks:?} failed: {e}"),
                }
            });
        }
    }

    /// Whether a claim waiting on ranks of a kept membership waits for the takeover's allowance:
    /// the allowance after the first daemon's hello (or, with none yet, the unreachable bound
    /// after this coordinator started) has not passed.
    pub(super) fn awaits_unheard(&self) -> bool {
        let allowance = Duration::from_millis(self.takeover.allowance_ms.load(Ordering::Relaxed));
        let unreachable =
            Duration::from_millis(self.takeover.unreachable_after_ms.load(Ordering::Relaxed));
        let first = self.takeover.first_hello.lock().ok().and_then(|f| *f);
        match first {
            Some(at) => at.elapsed() < allowance,
            None => self.started.elapsed() < unreachable,
        }
    }

    /// The ranks of `engine`'s kept membership this coordinator knows no holder of.
    pub(super) async fn unheard_ranks(&self, engine: &str) -> Vec<u32> {
        let kept = self.kept_members(engine).unwrap_or_default();
        let engines = self.engines.lock().await;
        let Some(view) = engines.get(engine) else {
            return Vec::new();
        };
        kept.into_keys()
            .filter(|r| {
                !view.members.contains_key(r)
                    && !view.departed.contains(r)
                    && !view.lost.contains(r)
            })
            .collect()
    }

    /// Keep `engine`'s membership for its incarnation: every rank's holder once it holds every
    /// rank and whenever a commit moved one, nothing while it is down. Written only when it
    /// differs from what is kept.
    pub(super) async fn keep_members(&self, engine: &str) {
        let world = self.engine_spec(engine).map(crate::policy::world_size);
        let wanted: Option<Vec<KeptMember>> = {
            let engines = self.engines.lock().await;
            let Some(view) = engines.get(engine) else {
                return;
            };
            let world = world.unwrap_or(view.world_size);
            if view.down.is_some() {
                Some(Vec::new())
            } else if world > 0
                && (0..world).all(|r| view.members.contains_key(&r) || view.lost.contains(&r))
            {
                Some(
                    view.members
                        .iter()
                        .map(|(r, n)| KeptMember {
                            rank: *r,
                            node: n.clone(),
                        })
                        .collect(),
                )
            } else {
                None
            }
        };
        let Some(wanted) = wanted else { return };
        let kept = self
            .kept_status(engine)
            .map(|s| s.members)
            .unwrap_or_default();
        if kept == wanted {
            return;
        }
        if let Err(e) = self.keep_status(engine, |s| s.members = wanted).await {
            warn!("{engine}: its membership not kept: {e}");
        }
    }

    /// `engine`'s kept membership, rank to node; none when nothing is kept.
    pub(super) fn kept_members(&self, engine: &str) -> Option<BTreeMap<u32, String>> {
        let kept = self.kept_status(engine)?.members;
        (!kept.is_empty()).then(|| kept.into_iter().map(|m| (m.rank, m.node)).collect())
    }

    fn engine_names(&self) -> Vec<String> {
        self.policy.engines.iter().map(|e| e.name.clone()).collect()
    }
}
