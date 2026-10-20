// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Nodes as resources (decision 0036): a node's spec says whether it is cordoned and whether it
//! is drained by a deadline; the node's reconcile turns that into what the engines on it do. A
//! cordoned node's spares take no position; a drained node's positions move off it at the
//! deadline, as many nodes at once as the policy's concurrent losses allow, the rest waiting
//! their turn in deadline order. A drain whose change was refused is blocked: its node keeps
//! serving and its Degraded condition says why. A blocked drain is tried again, level-triggered:
//! at once when one of an engine's sets becomes covered (the refusal most often waits on one),
//! otherwise after a backoff, until its change commits or its spec is withdrawn or given a new
//! deadline.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::clock::now_ms;
use crate::lease::BoxFuture;
use crate::proto::everserve::signals::v1::{Notice, Reason};
use crate::proto::everserve::spec::v1::{
    EngineStatus, NodeSpec, ResourceSpec, ResourceStatus, resource_spec, resource_status,
};
use crate::spec::{SpecError, SpecStore, engine_key, node_key};
use crate::status::{Event, Observed, Polarity, Severity, Status};

use super::{Cause, ChangeError, Coordinator};
use crate::metrics::ControlMetrics;

/// Where a node's drain stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrainState {
    /// Its notices went out and its changes are set for the deadline: ranks per engine.
    Scheduled(BTreeMap<String, Vec<u32>>),
    /// It waits for its turn: a drain with an earlier deadline goes first, and the policy's
    /// concurrent losses would be exceeded by both at once.
    Waiting(String),
    /// Its change at `deadline_ms` was refused; its positions stay where they are until it is
    /// tried again at `retry_at`, or sooner when a set becomes covered.
    Blocked {
        /// The deadline it was blocked at.
        deadline_ms: i64,
        /// Why, from the refused change.
        why: String,
        /// How many times its change was refused.
        attempts: u32,
        /// When it is tried again.
        retry_at: Instant,
    },
}

impl DrainState {
    /// How the node list names it.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Scheduled(_) => "scheduled",
            Self::Waiting(_) => "waiting",
            Self::Blocked { .. } => "blocked",
        }
    }
}

/// Why a node's spec could not be written.
#[derive(Debug, thiserror::Error)]
pub enum NodeSpecError {
    /// The store refused or failed the write.
    #[error(transparent)]
    Store(#[from] SpecError),
    /// The drain's notices already went out with its deadline; the engines on the node are
    /// preparing to leave it then, and nothing tells them to stay or to leave at another time.
    #[error("{0}: its drain is underway (the notices went out); it cannot be withdrawn or moved")]
    DrainUnderway(String),
    /// A drain asked of a node that holds no member of any engine.
    #[error("node {0} holds no member of any engine: there is nothing to drain")]
    NoMembers(String),
    /// The spec asks what cannot be.
    #[error("{0}")]
    Invalid(String),
    /// The write was made against a generation of the spec that is no longer the one held.
    #[error("{node}: written against generation {expected}, and generation {held} is held")]
    Stale {
        /// The node.
        node: String,
        /// The generation the write was made against.
        expected: u64,
        /// The generation held now.
        held: u64,
    },
}

/// A node's spec as an operator or a notice asks for it (decision 0042): every field the writer
/// sets, the rest of what the store holds kept.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeWant {
    /// Cordoned: its spares take no position.
    pub unschedulable: bool,
    /// Drained by this time, Unix milliseconds; 0 for no drain.
    pub drain_by_ms: i64,
    /// Why, for people.
    pub reason: String,
}

impl Coordinator {
    /// Write `resource` as `key`'s over generation `read`, under this coordinator's lease term
    /// (decision 0036). A coordinator that knows it is deposed writes nothing; the store refuses a
    /// write under a term older than the newest that wrote to it.
    ///
    /// # Errors
    ///
    /// [`SpecError::Deposed`] when this coordinator's hold ended or a daemon reported a newer
    /// term; otherwise as [`SpecStore::put`](crate::spec::SpecStore::put).
    pub(super) async fn put_spec(
        &self,
        key: &str,
        resource: ResourceSpec,
        read: u64,
    ) -> Result<ResourceSpec, SpecError> {
        self.write_store(key, move |store, key, term| {
            Box::pin(async move { store.put(key, resource, read, term).await })
        })
        .await
    }

    /// Change `engine`'s kept status with `change` and write it whole to the store, fenced by
    /// this coordinator's term (decision 0038); kept only once written. An engine's writes go one
    /// at a time, so they land in the order they were made.
    pub(super) async fn keep_status(
        &self,
        engine: &str,
        change: impl FnOnce(&mut EngineStatus) + Send,
    ) -> Result<(), SpecError> {
        self.update_status(engine, change).await
    }

    /// As [`Coordinator::keep_status`], with what `decide` returns: `decide` reads and changes
    /// the engine's status under its lock, so two callers never decide on the same status.
    pub(super) async fn update_status<T: Send>(
        &self,
        engine: &str,
        decide: impl FnOnce(&mut EngineStatus) -> T + Send,
    ) -> Result<T, SpecError> {
        let slot = Arc::clone(
            self.specs
                .kept
                .lock()
                .await
                .entry(engine.to_owned())
                .or_insert_with(|| {
                    Arc::new(Mutex::new(EngineStatus {
                        engine: engine.to_owned(),
                        ..EngineStatus::default()
                    }))
                }),
        );
        let mut kept = slot.lock().await;
        let mut next = kept.clone();
        let decided = decide(&mut next);
        if next == *kept {
            return Ok(decided);
        }
        let status = ResourceStatus {
            kind: Some(resource_status::Kind::Engine(next.clone())),
        };
        self.write_store(&engine_key(engine), move |store, key, term| {
            Box::pin(async move { store.put_status(key, status, term).await })
        })
        .await?;
        self.specs
            .written
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(engine.to_owned(), next.clone());
        *kept = next;
        Ok(decided)
    }

    /// `engine`'s status as the store keeps it.
    pub(super) fn kept_status(&self, engine: &str) -> Option<EngineStatus> {
        self.specs
            .written
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(engine)
            .cloned()
    }

    /// Run `write` against the store for `key` under this coordinator's term, within the store
    /// write deadline; nothing once it is deposed. Every write this coordinator makes goes
    /// through here, and one the store fences (a newer holder took it or wrote to it) ends this
    /// coordinator's hold, as a write that outruns the deadline does.
    async fn write_store<T: Send + 'static>(
        &self,
        key: &str,
        write: impl FnOnce(Arc<dyn SpecStore>, String, u64) -> BoxFuture<'static, Result<T, SpecError>>,
    ) -> Result<T, SpecError> {
        if self.deposed() {
            return Err(SpecError::Deposed {
                key: key.to_owned(),
            });
        }
        let (store, term, put_key) = (Arc::clone(&self.specs.store), self.term(), key.to_owned());
        let deadline = Duration::from_millis(
            self.store_deadline_ms
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        if let Ok(written) = tokio::time::timeout(deadline, write(store, put_key, term)).await {
            if let Err(e) = &written
                && matches!(e, SpecError::Fenced { .. } | SpecError::Superseded { .. })
            {
                // another holder took the store: this one acts no more
                self.step_down(&format!("the spec store refused {key}: {e}"));
            }
            written
        } else {
            // the write may still land: nothing this coordinator does from now on may rest on
            // whether it did, so it acts no more, and its successor reads the store
            self.step_down(&format!(
                "the spec store took longer than {deadline:?} to write {key}"
            ));
            Err(SpecError::TimedOut {
                key: key.to_owned(),
                after: deadline,
            })
        }
    }

    /// `node`'s spec as this coordinator last read or wrote it.
    pub async fn node_spec(&self, node: &str) -> Option<ResourceSpec> {
        self.specs.cache.lock().await.get(&node_key(node)).cloned()
    }

    /// Where `node`'s drain stands, if one was asked for.
    pub async fn drain_state(&self, node: &str) -> Option<DrainState> {
        self.specs.drains.lock().await.get(node).cloned()
    }

    /// Count a `kind` spec write by `writer` that ended as `result`. A writer is the operator API
    /// or a node's notice; the label holds which, never the node, so its values stay few.
    pub(super) fn count_spec_write(&self, kind: &str, writer: &str, result: &str) {
        let writer = if writer.starts_with("node ") {
            "node_notice"
        } else {
            "operator_api"
        };
        self.metrics
            .control
            .spec_writes_total
            .with_label_values(&[kind, writer, result])
            .inc();
    }

    /// Every node's spec this coordinator holds, by node.
    pub(super) async fn node_specs(&self) -> BTreeMap<String, NodeSpec> {
        self.specs
            .cache
            .lock()
            .await
            .values()
            .filter_map(|r| match &r.kind {
                Some(resource_spec::Kind::Node(n)) => Some((n.node.clone(), n.clone())),
                _ => None,
            })
            .collect()
    }

    /// The cordoned nodes: their spares take no position.
    pub(super) async fn cordoned(&self) -> BTreeSet<String> {
        self.specs
            .cache
            .lock()
            .await
            .values()
            .filter_map(|r| match &r.kind {
                Some(resource_spec::Kind::Node(n)) if n.unschedulable => Some(n.node.clone()),
                _ => None,
            })
            .collect()
    }

    /// Make `node`'s spec `want`, written by `writer`, only over generation `expected` when one
    /// is given; a want the spec already holds is no write. A drain cordons its node, and drains
    /// only a node holding members.
    ///
    /// # Errors
    ///
    /// [`NodeSpecError::Invalid`] for a drain on a node left schedulable;
    /// [`NodeSpecError::NoMembers`] for a new drain on a node holding no member;
    /// [`NodeSpecError::Stale`] when the generation held is not `expected`;
    /// [`NodeSpecError::DrainUnderway`] when the write moves or clears a drain whose notices went
    /// out; [`NodeSpecError::Store`] when the store refused or failed the write.
    pub async fn put_node_spec(
        self: &Arc<Self>,
        node: &str,
        writer: &str,
        want: NodeWant,
        expected: Option<u64>,
    ) -> Result<ResourceSpec, NodeSpecError> {
        if want.drain_by_ms != 0 && !want.unschedulable {
            return Err(NodeSpecError::Invalid(format!(
                "{node}: a node being drained is cordoned; a drain needs unschedulable"
            )));
        }
        let held = self.node_spec(node).await;
        let (generation, spec) = match &held {
            Some(ResourceSpec {
                generation,
                kind: Some(resource_spec::Kind::Node(n)),
                ..
            }) => (*generation, Some(n)),
            _ => (0, None),
        };
        // a stale write is refused even when it asks what is held; write_node_spec checks again
        // at the generation it writes over
        if let Some(expected) = expected
            && expected != generation
        {
            self.count_spec_write("node", writer, "refused");
            return Err(NodeSpecError::Stale {
                node: node.to_owned(),
                expected,
                held: generation,
            });
        }
        if let (Some(held), Some(spec)) = (&held, spec)
            && spec.unschedulable == want.unschedulable
            && spec.drain_by_ms == want.drain_by_ms
            && spec.reason == want.reason
        {
            return Ok(held.clone());
        }
        let drained = spec.map_or(0, |s| s.drain_by_ms);
        if want.drain_by_ms != 0
            && want.drain_by_ms != drained
            && self.positions_on(node).await.is_empty()
        {
            return Err(NodeSpecError::NoMembers(node.to_owned()));
        }
        self.write_node_spec(node, writer, expected, move |s| {
            s.unschedulable = want.unschedulable;
            s.drain_by_ms = want.drain_by_ms;
            s.reason = want.reason;
        })
        .await
    }

    /// Write `node`'s spec: `change` applied to the spec this coordinator holds, at its
    /// generation, which must be `expected` when one is given, recorded as written by `writer`;
    /// then reconcile the node. The store takes the write only at the generation read, so a write
    /// landing between the read and this one refuses this one.
    ///
    /// # Errors
    ///
    /// [`NodeSpecError::Stale`] when the generation held is not `expected`;
    /// [`NodeSpecError::DrainUnderway`] when the change clears or moves the deadline of a drain
    /// whose notices went out;
    /// [`NodeSpecError::Store`] when the store refused the write (another write landed since, or
    /// a newer lease term wrote to it), this coordinator is deposed, or the store failed.
    pub(super) async fn write_node_spec(
        self: &Arc<Self>,
        node: &str,
        writer: &str,
        expected: Option<u64>,
        change: impl FnOnce(&mut NodeSpec) + Send,
    ) -> Result<ResourceSpec, NodeSpecError> {
        let key = node_key(node);
        let (mut spec, read) = match self.node_spec(node).await {
            Some(ResourceSpec {
                generation,
                kind: Some(resource_spec::Kind::Node(n)),
                ..
            }) => (n, generation),
            _ => (
                NodeSpec {
                    node: node.to_owned(),
                    ..NodeSpec::default()
                },
                0,
            ),
        };
        if let Some(expected) = expected
            && expected != read
        {
            self.count_spec_write("node", writer, "refused");
            return Err(NodeSpecError::Stale {
                node: node.to_owned(),
                expected,
                held: read,
            });
        }
        let deadline = spec.drain_by_ms;
        change(&mut spec);
        if spec.drain_by_ms != deadline
            && matches!(self.drain_state(node).await, Some(DrainState::Scheduled(_)))
        {
            self.count_spec_write("node", writer, "refused");
            return Err(NodeSpecError::DrainUnderway(node.to_owned()));
        }
        let resource = ResourceSpec {
            generation: read,
            written_by: writer.to_owned(),
            written_at_ms: now_ms(),
            kind: Some(resource_spec::Kind::Node(spec)),
        };
        let put = self.put_spec(&key, resource, read).await;
        self.count_spec_write("node", writer, written(&put));
        let stored = put?;
        self.specs.cache.lock().await.insert(key, stored.clone());
        info!(
            "{node}: spec written by {writer} (generation {})",
            stored.generation
        );
        if let Some(resource_spec::Kind::Node(n)) = &stored.kind {
            self.node_event(
                node,
                Severity::Normal,
                "SpecWritten",
                format!(
                    "generation {} by {writer}: {}{}{}",
                    stored.generation,
                    if n.unschedulable {
                        "cordoned"
                    } else {
                        "schedulable"
                    },
                    if n.drain_by_ms == 0 {
                        String::new()
                    } else {
                        format!(", drained by {} ms", n.drain_by_ms)
                    },
                    if n.reason.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", n.reason)
                    }
                ),
            )
            .await;
        }
        self.start_node_controller().await;
        self.reconcile_node(node).await;
        // a cordon changes what every engine's status means: its spares' states, its spare
        // gauge and its Covered condition
        let engines: Vec<String> = self.engines.lock().await.keys().cloned().collect();
        for engine in engines {
            self.observed(&engine).await;
        }
        Ok(stored)
    }

    /// One reconcile of `node`: start its drain when its spec asks for one, no drain with an
    /// earlier deadline waits, and the policy's concurrent losses allow it; otherwise leave it
    /// waiting. Forget a drain that emptied its node or was withdrawn. A scheduled drain, and a
    /// blocked one whose deadline has not changed, stay as they are. Never waits on a change.
    pub(super) async fn reconcile_node(self: &Arc<Self>, node: &str) -> Option<DrainState> {
        let specs = self.node_specs().await;
        let drain_by_ms = specs.get(node).map_or(0, |n| n.drain_by_ms);
        let on_node = self.positions_on(node).await;
        let mut drains = self.specs.drains.lock().await;
        if drain_by_ms == 0 || on_node.is_empty() {
            // nothing asked, or nothing left to move: a drain that emptied the node is done, any
            // other is withdrawn (a scheduled one cannot be: its spec write was refused)
            let was = drains.get(node).cloned();
            if !matches!(was, Some(DrainState::Scheduled(_))) || on_node.is_empty() {
                drains.remove(node);
            }
            let now = drains.get(node).cloned();
            drop(drains);
            if let Some(was) = was
                && now.is_none()
            {
                let (reason, message) = if on_node.is_empty() {
                    ("DrainDone", "every position left the node".to_owned())
                } else {
                    (
                        "DrainWithdrawn",
                        format!("the {} drain was withdrawn", was.label()),
                    )
                };
                self.node_event(node, Severity::Normal, reason, message)
                    .await;
            }
            self.node_observe(node).await;
            return now;
        }
        // a blocked drain is tried again once due, counting its refusals; one given a new
        // deadline starts over
        let retry = match drains.get(node) {
            Some(s @ DrainState::Scheduled(_)) => return Some(s.clone()),
            Some(
                s @ DrainState::Blocked {
                    deadline_ms,
                    attempts,
                    retry_at,
                    ..
                },
            ) if *deadline_ms == drain_by_ms => {
                if Instant::now() < *retry_at {
                    return Some(s.clone());
                }
                Some(*attempts)
            }
            _ => None,
        };
        let wait = self
            .drain_turn(node, drain_by_ms, &on_node, &drains, &specs)
            .await;
        if let Some(why) = wait {
            // a drain that waits for another reason than before is news
            let news = !matches!(drains.get(node), Some(DrainState::Waiting(w)) if *w == why);
            let state = DrainState::Waiting(why.clone());
            drains.insert(node.to_owned(), state.clone());
            drop(drains);
            if news {
                info!("{node}: its drain waits: {why}");
                self.node_event(node, Severity::Normal, "DrainWaiting", why)
                    .await;
            }
            self.node_observe(node).await;
            return Some(state);
        }
        drains.insert(node.to_owned(), DrainState::Scheduled(on_node.clone()));
        drop(drains);
        let (reason, message) = match retry {
            None => (
                "DrainStarted",
                format!("ranks {on_node:?} leave at {drain_by_ms} ms"),
            ),
            Some(n) => (
                "DrainRetried",
                format!("ranks {on_node:?}: the change tried again after {n} refusals"),
            ),
        };
        self.node_event(node, Severity::Normal, reason, message)
            .await;
        self.node_observe(node).await;
        for (engine, ranks) in &on_node {
            // a retry's notices went out at the drain's start, with the same deadline
            for rank in ranks.iter().filter(|_| retry.is_none()) {
                self.forward_notice(
                    engine,
                    Notice {
                        rank: *rank,
                        deadline_ms: drain_by_ms,
                        reason: Reason::Operator as i32,
                    },
                )
                .await;
            }
            info!("{engine}: drain of {node} releases ranks {ranks:?} at {drain_by_ms} ms");
            Arc::clone(self).drain_at(node, engine, ranks.clone(), drain_by_ms, retry.unwrap_or(0));
        }
        Some(DrainState::Scheduled(on_node))
    }

    /// Why `node`'s drain, due at `drain_by_ms`, must wait, if it must. Engines are independent:
    /// only drains of nodes holding a position of an engine `node` holds are weighed. One of
    /// those with an earlier deadline (then the lesser node name) that still waits goes first;
    /// and no engine may have more of its positions moving at once than the policy's concurrent
    /// losses allow, counting every scheduled drain's.
    async fn drain_turn(
        &self,
        node: &str,
        drain_by_ms: i64,
        on_node: &BTreeMap<String, Vec<u32>>,
        drains: &BTreeMap<String, DrainState>,
        specs: &BTreeMap<String, NodeSpec>,
    ) -> Option<String> {
        let mut others = Vec::new();
        for (other, state) in drains {
            if other == node {
                continue;
            }
            let theirs = self.positions_on(other).await;
            // a drain of a node holding none of this node's engines is no concern of this one.
            // Its engines are the positions it holds, or, while those are unknown (its daemon
            // has not said hello to this coordinator), the engines the policy places on it: a
            // stored drain due first keeps its turn across a restart
            let placed = self.placed_on.get(other.as_str());
            let shares = theirs.keys().any(|e| on_node.contains_key(e))
                || placed.is_some_and(|p| p.iter().any(|e| on_node.contains_key(e)));
            if shares {
                others.push((other, state, theirs));
            }
        }
        // drains of the same engines start in deadline order: an earlier one waiting goes first
        let mine = (drain_by_ms, node);
        if let Some((other, _, _)) = others.iter().find(|(other, state, _)| {
            matches!(state, DrainState::Waiting(_))
                && (
                    specs.get(*other).map_or(0, |n| n.drain_by_ms),
                    other.as_str(),
                ) < mine
        }) {
            return Some(format!("{other}'s drain is due first"));
        }
        let max = usize::try_from(self.policy.max_concurrent_losses).unwrap_or(usize::MAX);
        if max == 0 {
            return None;
        }
        // each engine's positions every scheduled drain still has to move, beside this node's
        for (engine, ranks) in on_node {
            let moving = ranks.len()
                + others
                    .iter()
                    .filter(|(_, state, _)| matches!(state, DrainState::Scheduled(_)))
                    .filter_map(|(_, _, theirs)| theirs.get(engine).map(Vec::len))
                    .sum::<usize>();
            if moving > max {
                return Some(format!(
                    "{moving} of {engine}'s positions would move at once, the policy allows {max}"
                ));
            }
        }
        None
    }

    /// The change that moves `ranks` of `engine` off `node` at `deadline_ms`, as its own task,
    /// refused `attempts` times before; a refusal blocks the drain until a backoff, and the next
    /// drain in line may start. A shutdown does not wait for the deadline: the drain is in the
    /// spec store, and a successor schedules it again.
    fn drain_at(
        self: Arc<Self>,
        node: &str,
        engine: &str,
        ranks: Vec<u32>,
        deadline_ms: i64,
        attempts: u32,
    ) {
        let wait =
            Duration::from_millis(u64::try_from(deadline_ms.saturating_sub(now_ms())).unwrap_or(0));
        let (node, engine) = (node.to_owned(), engine.to_owned());
        tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            let moved = if ranks.is_empty() {
                Ok(())
            } else {
                self.ranks_lost(&engine, &ranks, Cause::Notice)
                    .await
                    .map(|_| ())
            };
            match moved {
                Ok(()) | Err(ChangeError::Stale { .. }) => {}
                Err(e) => {
                    warn!("{node}: drain blocked: {engine}'s change for ranks {ranks:?}: {e}");
                    let why = format!("{engine}'s change for ranks {ranks:?}: {e}");
                    let wait = super::cover::backoff(attempts);
                    let mut drains = self.specs.drains.lock().await;
                    if matches!(drains.get(&node), Some(DrainState::Scheduled(_))) {
                        drains.insert(
                            node.clone(),
                            DrainState::Blocked {
                                deadline_ms,
                                why,
                                attempts: attempts + 1,
                                retry_at: Instant::now() + wait,
                            },
                        );
                    }
                    drop(drains);
                    self.node_observe(&node).await;
                    // tried again when due, if nothing makes it due sooner
                    let me = Arc::clone(&self);
                    let again = node.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(wait).await;
                        me.specs.queue.enqueue(again).await;
                    });
                    // a blocked drain moves nothing: the next in line may start
                    self.wake_drains().await;
                }
            }
        });
    }

    /// Record an event in `node`'s log.
    pub(super) async fn node_event(
        &self,
        node: &str,
        severity: Severity,
        reason: &str,
        message: String,
    ) {
        self.specs
            .status
            .lock()
            .await
            .entry(node.to_owned())
            .or_default()
            .record(now_ms(), severity, reason, message);
    }

    /// Derive `node`'s conditions from its drain.
    async fn node_observe(&self, node: &str) {
        let drain = self.drain_state(node).await;
        let draining = match &drain {
            Some(DrainState::Scheduled(ranks)) => Observed::new(
                "Draining",
                Polarity::Neutral,
                true,
                "Scheduled",
                format!("ranks {ranks:?} leave at the deadline"),
            ),
            Some(DrainState::Waiting(why)) => {
                Observed::new("Draining", Polarity::Neutral, true, "Waiting", why.clone())
            }
            Some(DrainState::Blocked { .. }) => Observed::new(
                "Draining",
                Polarity::Neutral,
                true,
                "Blocked",
                "its change was refused",
            ),
            None => Observed::new("Draining", Polarity::Neutral, false, "NotDraining", ""),
        };
        let degraded = match &drain {
            Some(DrainState::Blocked { why, .. }) => Observed::new(
                "Degraded",
                Polarity::Trouble,
                true,
                "DrainBlocked",
                why.clone(),
            ),
            _ => Observed::new("Degraded", Polarity::Trouble, false, "AsSpecified", ""),
        };
        self.specs
            .status
            .lock()
            .await
            .entry(node.to_owned())
            .or_default()
            .observe(vec![draining, degraded], now_ms());
        let conditions = self.node_status(node).await.conditions();
        ControlMetrics::set_conditions(&self.metrics.control.node_condition, node, &conditions);
    }

    /// `node`'s conditions and events, as the node list and the events route show them.
    pub(super) async fn node_status(&self, node: &str) -> Status {
        self.specs
            .status
            .lock()
            .await
            .get(node)
            .cloned()
            .unwrap_or_default()
    }

    /// Why `node` may not be given back now, each reason in words; empty when it may: it holds
    /// no position (no rank, no head, no spare taking one), nothing is staged onto or from it, and
    /// no change in flight touches it. A spare standing on it, taking nothing,
    /// holds no position.
    pub async fn release_blockers(&self, node: &str) -> Vec<String> {
        let mut why = Vec::new();
        for (engine, view) in self.engines.lock().await.iter() {
            let ranks: Vec<u32> = view
                .members
                .iter()
                .filter(|(_, n)| *n == node)
                .map(|(r, _)| *r)
                .collect();
            if !ranks.is_empty() {
                why.push(format!("it holds ranks {ranks:?} of {engine}"));
            }
            if view.head.as_ref().is_some_and(|h| h.process.node == node) {
                why.push(format!("it holds {engine}'s head"));
            }
            let taking: Vec<u32> = view
                .taking
                .iter()
                .filter(|(_, (n, _))| n == node)
                .map(|(r, _)| *r)
                .collect();
            if !taking.is_empty() {
                why.push(format!("a spare on it takes ranks {taking:?} of {engine}"));
            }
            // a lost rank's rows staged from a replica here, or onto a spare here
            let target = |r: &u32| {
                view.spares
                    .get(r)
                    .map(String::as_str)
                    .or_else(|| view.taking.get(r).map(|(n, _)| n.as_str()))
            };
            let staged: Vec<u32> = view
                .restoring
                .iter()
                .filter(|(r, from)| *from == node || target(r) == Some(node))
                .map(|(r, _)| *r)
                .collect();
            if !staged.is_empty() {
                why.push(format!(
                    "rows of {engine}'s ranks {staged:?} are staged onto or from it"
                ));
            }
            if let Some((epoch, _)) = &view.change {
                let touched =
                    view.change_from.contains(node) || view.spares.values().any(|n| n == node);
                if touched {
                    why.push(format!("{engine}'s change at epoch {epoch} touches it"));
                }
            }
        }
        why
    }

    /// The events kept for `node`, oldest first.
    pub async fn node_events(&self, node: &str) -> Vec<Event> {
        self.node_status(node).await.events()
    }

    /// Make every blocked drain due now and queue it: one of an engine's sets became covered,
    /// and a refusal most often waits on one.
    pub(super) async fn retry_blocked_drains(&self) {
        let mut due = Vec::new();
        for (node, state) in self.specs.drains.lock().await.iter_mut() {
            if let DrainState::Blocked { retry_at, .. } = state {
                *retry_at = Instant::now();
                due.push(node.clone());
            }
        }
        for node in due {
            self.specs.queue.enqueue(node).await;
        }
    }

    /// Queue every node whose drain is waiting or scheduled: after a change ends, one may have
    /// finished and another may now start.
    pub(super) async fn wake_drains(&self) {
        let nodes: Vec<String> = self.specs.drains.lock().await.keys().cloned().collect();
        for node in nodes {
            self.specs.queue.enqueue(node).await;
        }
    }

    /// Start the node controller's worker on first use, once: it takes each queued node and
    /// reconciles it. Started where a drain begins (a spec written, or one resumed), never from
    /// the engine controller, whose transitions only wake it.
    async fn start_node_controller(&self) {
        let Some(mut rx) = self.specs.queue.receiver().await else {
            return;
        };
        let Some(me) = self.me.upgrade() else { return };
        tokio::spawn(async move {
            while let Some(node) = rx.recv().await {
                me.specs.queue.take(&node).await;
                let started = Instant::now();
                let state = me.reconcile_node(&node).await;
                me.metrics
                    .control
                    .reconcile_seconds
                    .with_label_values(&["node"])
                    .observe(started.elapsed().as_secs_f64());
                me.metrics
                    .control
                    .reconciles_total
                    .with_label_values(&["node", state.as_ref().map_or("none", DrainState::label)])
                    .inc();
            }
        });
    }

    /// `node`'s daemon said what it holds: a drain its spec asks for, set before this coordinator
    /// knew its positions (a restart, or a takeover of the lease), resumes.
    pub(super) async fn positions_known(&self, node: &str) {
        if !self.specs.drains.lock().await.contains_key(node) {
            return;
        }
        self.start_node_controller().await;
        if let Some(me) = self.me.upgrade() {
            me.reconcile_node(node).await;
        }
    }

    /// The positions on `node`, by engine.
    pub(super) async fn positions_on(&self, node: &str) -> BTreeMap<String, Vec<u32>> {
        self.engines
            .lock()
            .await
            .iter()
            .filter_map(|(name, view)| {
                let ranks: Vec<u32> = view
                    .members
                    .iter()
                    .filter(|(_, n)| n.as_str() == node)
                    .map(|(r, _)| *r)
                    .collect();
                (!ranks.is_empty()).then(|| (name.clone(), ranks))
            })
            .collect()
    }
}

/// How a spec write ended, as the write counter labels it.
pub(super) fn written(put: &Result<ResourceSpec, SpecError>) -> &'static str {
    match put {
        Ok(_) => "written",
        Err(SpecError::Conflict { .. }) => "conflict",
        Err(
            SpecError::Fenced { .. } | SpecError::Superseded { .. } | SpecError::Deposed { .. },
        ) => "fenced",
        Err(_) => "failed",
    }
}
