// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Pools as resources (decision 0031): spare capacity for the engines of one shape. A pool's spec
//! names its engines, how many spares it wants standing and what may join it; its status counts
//! the spares standing for any of its engines, each process once whatever engines it registered
//! with, and the deficit a scheduler keeps at zero. The shape its engines share (the identity's
//! engine-compatible fields and the compile key) is checked when a spec is written, and derived
//! again as the engines' processes report what they run.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::info;

use crate::clock::now_ms;
use crate::proto::everserve::coord::v1::{CoordToDaemon, LaunchSpare, coord_to_daemon};
use crate::proto::everserve::spec::v1::{PoolSpec, ResourceSpec, resource_spec};
use crate::spec::{SpecError, pool_key};
use crate::status::{Condition, Observed, Polarity, Severity, Truth};

use super::Coordinator;
use super::capability::engine_difference;
use super::outbox::Outbox;

/// How long a spare the coordinator launched counts as warming if it never stands: its process
/// loads a checkpoint and compiles, minutes on a large model, and the daemon restarts it if it
/// exits.
pub const WARMING: Duration = Duration::from_mins(15);

/// Why a pool's spec could not be written.
#[derive(Debug, thiserror::Error)]
pub enum PoolSpecError {
    /// The store refused or failed the write.
    #[error(transparent)]
    Store(#[from] SpecError),
    /// An engine the spec names is not in the policy.
    #[error("{0}: no engine of that name in the policy")]
    UnknownEngine(String),
    /// The engines' known shapes differ: one spare cannot stand for all of them (0031).
    #[error("the pool's engines differ in shape: {0}")]
    Shape(String),
    /// The write was made against a generation of the spec that is no longer the one held.
    #[error("{pool}: written against generation {expected}, and generation {held} is held")]
    Stale {
        /// The pool.
        pool: String,
        /// The generation the write was made against.
        expected: u64,
        /// The generation held now.
        held: u64,
    },
}

/// A pool as `GET /v1/pools` shows it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PoolView {
    /// Its name.
    pub name: String,
    /// Its spec's generation.
    pub generation: u64,
    /// Who wrote its spec last.
    pub written_by: String,
    /// When, Unix milliseconds.
    pub written_at_ms: i64,
    /// Why, as its writer said.
    pub reason: String,
    /// The engines it holds spares for.
    pub engines: Vec<String>,
    /// How many spares it wants standing.
    pub wants: u32,
    /// The spares standing for any of its engines on a node that is not cordoned and that its
    /// selector admits, each process counted once.
    pub standing: u32,
    /// Spares taking a position in a change in flight.
    pub taking: u32,
    /// Spares this coordinator launched for the pool that have not yet registered.
    pub warming: u32,
    /// `wants` less `standing` and `warming`: what the pool's launches, or a scheduler, keep at
    /// zero.
    pub deficit: u32,
    /// The device class a spare must stand on; empty for any.
    pub device_class: String,
    /// The network domain a spare must stand in; empty for any.
    pub domain: String,
    /// `Satisfied` and `ShapeAgrees`.
    pub conditions: Vec<Condition>,
}

impl Coordinator {
    /// Every pool's spec this coordinator holds, by name.
    pub(super) async fn pool_specs(&self) -> BTreeMap<String, (u64, PoolSpec)> {
        self.pool_resources()
            .await
            .into_iter()
            .filter_map(|(name, r)| match r.kind {
                Some(resource_spec::Kind::Pool(p)) => Some((name, (r.generation, p))),
                _ => None,
            })
            .collect()
    }

    /// Every pool's spec as the store holds it, its writer with it, by name.
    async fn pool_resources(&self) -> BTreeMap<String, ResourceSpec> {
        self.specs
            .cache
            .lock()
            .await
            .values()
            .filter_map(|r| match &r.kind {
                Some(resource_spec::Kind::Pool(p)) => Some((p.name.clone(), r.clone())),
                _ => None,
            })
            .collect()
    }

    /// `pool`'s spec and its generation, as this coordinator last read or wrote it.
    pub async fn pool_spec(&self, pool: &str) -> Option<(u64, PoolSpec)> {
        self.pool_specs().await.remove(pool)
    }

    /// Write `pool`'s spec as `spec`, recorded as written by `writer`, over the generation this
    /// coordinator holds, or only over `expected` when one is given; then derive every pool's
    /// status again.
    ///
    /// # Errors
    ///
    /// [`PoolSpecError::UnknownEngine`] for an engine the policy does not name;
    /// [`PoolSpecError::Shape`] when the engines' known shapes differ; [`PoolSpecError::Stale`]
    /// when the generation held is not `expected`; [`PoolSpecError::Store`] when the store
    /// refused the write (another landed since) or failed.
    pub async fn write_pool_spec(
        self: &Arc<Self>,
        writer: &str,
        mut spec: PoolSpec,
        expected: Option<u64>,
    ) -> Result<ResourceSpec, PoolSpecError> {
        if let Some(e) = spec.engines.iter().find(|e| self.engine_spec(e).is_none()) {
            return Err(PoolSpecError::UnknownEngine(e.clone()));
        }
        spec.engines.sort();
        spec.engines.dedup();
        if let Some(why) = self.shape_difference(&spec.engines).await {
            self.count_spec_write("pool", writer, "refused");
            return Err(PoolSpecError::Shape(why));
        }
        let key = pool_key(&spec.name);
        let read = self
            .specs
            .cache
            .lock()
            .await
            .get(&key)
            .map_or(0, |r| r.generation);
        if let Some(expected) = expected
            && expected != read
        {
            self.count_spec_write("pool", writer, "refused");
            return Err(PoolSpecError::Stale {
                pool: spec.name,
                expected,
                held: read,
            });
        }
        let name = spec.name.clone();
        let resource = ResourceSpec {
            generation: read,
            written_by: writer.to_owned(),
            written_at_ms: now_ms(),
            kind: Some(resource_spec::Kind::Pool(spec)),
        };
        let put = self.put_spec(&key, resource, read).await;
        self.count_spec_write("pool", writer, super::nodes::written(&put));
        let stored = put?;
        self.specs.cache.lock().await.insert(key, stored.clone());
        info!(
            "pool {name}: spec written by {writer} (generation {})",
            stored.generation
        );
        self.pool_event(
            &name,
            Severity::Normal,
            "SpecWritten",
            format!("generation {} by {writer}", stored.generation),
        )
        .await;
        self.observe_pools().await;
        Ok(stored)
    }

    /// Why `engines` cannot share one pool, by what their processes reported: the first known
    /// engine's identity against each other known one's, under the identity's engine-compatible
    /// comparison, and their compile keys. None when they agree or too little is known yet.
    async fn shape_difference(&self, engines: &[String]) -> Option<String> {
        let views = self.engines.lock().await;
        let known: Vec<(&String, &super::view::EngineView)> = engines
            .iter()
            .filter_map(|e| views.get(e).map(|v| (e, v)))
            .filter(|(_, v)| !v.identities.is_empty())
            .collect();
        let ((first, a), rest) = known.split_first()?;
        let a_id = a.identities.values().next()?;
        for (e, b) in rest {
            let b_id = b.identities.values().next()?;
            if let Some(why) = engine_difference(b_id, a_id, false) {
                return Some(format!("{e} against {first}: {why}"));
            }
            if let (Some(ka), Some(kb)) = (&a.compile_key, &b.compile_key)
                && ka != kb
            {
                return Some(format!(
                    "{e} against {first}: compile key {kb}, not {ka}; a spare would compile for each"
                ));
            }
        }
        None
    }

    /// Every pool, its spec and its status.
    pub async fn pools(&self) -> Vec<PoolView> {
        let specs = self.pool_resources().await;
        let cordoned = self.cordoned().await;
        let mut out = Vec::new();
        for (name, resource) in specs {
            let Some(resource_spec::Kind::Pool(spec)) = &resource.kind else {
                continue;
            };
            let (standing, taking) = self.pool_counts(spec, &cordoned).await;
            let warming = self.warming(&name).await;
            let conditions = self
                .specs
                .pool_status
                .lock()
                .await
                .get(&name)
                .map(crate::status::Status::conditions)
                .unwrap_or_default();
            out.push(PoolView {
                name,
                generation: resource.generation,
                written_by: resource.written_by.clone(),
                written_at_ms: resource.written_at_ms,
                reason: spec.reason.clone(),
                engines: spec.engines.clone(),
                wants: spec.wants,
                standing,
                taking,
                warming,
                deficit: spec.wants.saturating_sub(standing + warming),
                device_class: spec.device_class.clone(),
                domain: spec.domain.clone(),
                conditions,
            });
        }
        out
    }

    /// The spares standing for any of `spec`'s engines that it admits, and those taking a
    /// position, each process (node and device) counted once.
    async fn pool_counts(&self, spec: &PoolSpec, cordoned: &BTreeSet<String>) -> (u32, u32) {
        let views = self.engines.lock().await;
        let topo = self.topology.lock().await;
        let admits = |node: &String, device: u32| {
            if cordoned.contains(node) {
                return false;
            }
            let d = topo.device(node, device);
            (spec.device_class.is_empty() || d.is_some_and(|d| d.class == spec.device_class))
                && (spec.domain.is_empty() || d.is_some_and(|d| d.domain == spec.domain))
        };
        let mut standing: BTreeSet<(String, u32)> = BTreeSet::new();
        let mut taking: BTreeSet<(String, u32)> = BTreeSet::new();
        for v in spec.engines.iter().filter_map(|e| views.get(e)) {
            for ((node, device), ev) in &v.standing {
                let placed = if ev.any_rank {
                    v.any_spares.contains(&(node.clone(), *device))
                } else {
                    v.spares.get(&ev.rank) == Some(node)
                };
                if placed && admits(node, *device) {
                    standing.insert((node.clone(), *device));
                }
            }
            taking.extend(v.taking.values().cloned());
        }
        let n = |s: BTreeSet<(String, u32)>| u32::try_from(s.len()).unwrap_or(u32::MAX);
        (n(standing), n(taking))
    }

    /// Derive every pool's conditions and deficit gauge again: after any change to an engine's
    /// view or a pool's spec.
    pub(super) async fn observe_pools(&self) {
        let specs = self.pool_specs().await;
        let cordoned = self.cordoned().await;
        for (name, (_, spec)) in specs {
            let (standing, _) = self.pool_counts(&spec, &cordoned).await;
            self.refill(&name, &spec, &cordoned).await;
            let deficit = spec
                .wants
                .saturating_sub(standing + self.warming(&name).await);
            let satisfied = if deficit == 0 {
                Observed::new(
                    "Satisfied",
                    Polarity::Healthy,
                    true,
                    "Satisfied",
                    format!("{standing} of {} standing", spec.wants),
                )
            } else {
                Observed::new(
                    "Satisfied",
                    Polarity::Healthy,
                    false,
                    "Deficit",
                    format!("{standing} of {} standing: {deficit} short", spec.wants),
                )
            };
            let shape = match self.shape_difference(&spec.engines).await {
                Some(why) => {
                    Observed::new("ShapeAgrees", Polarity::Healthy, false, "ShapeDiffers", why)
                }
                None => Observed::new(
                    "ShapeAgrees",
                    Polarity::Healthy,
                    Truth::True,
                    "Agrees",
                    "the engines seen so far share one shape",
                ),
            };
            self.specs
                .pool_status
                .lock()
                .await
                .entry(name.clone())
                .or_default()
                .observe(vec![satisfied, shape], now_ms());
            self.metrics
                .control
                .pool_deficit
                .with_label_values(&[&name])
                .set(i64::from(deficit));
        }
    }

    /// Launch `pool`'s spares on idle nodes until its deficit is met (decision 0031): a node is
    /// idle for the pool while its daemon is connected, it is not cordoned, it holds no position of
    /// any engine, and the pool's selector admits its device. A spare launched counts as warming
    /// until it stands for one of the pool's engines, or for [`WARMING`] if it never does; nothing
    /// is launched for a pool without a launch command.
    async fn refill(&self, pool: &str, spec: &PoolSpec, cordoned: &BTreeSet<String>) {
        if spec.command.is_empty() {
            return;
        }
        let (standing, _) = self.pool_counts(spec, cordoned).await;
        let now = Instant::now();
        // what stands no longer warms, and a launch that never stood stops counting after a while
        let standing_at = self.pool_standing(spec).await;
        let warming = {
            let mut launched = self.specs.launched.lock().await;
            let mine = launched.entry(pool.to_owned()).or_default();
            mine.retain(|at, since| {
                !standing_at.contains(at) && now.duration_since(*since) < WARMING
            });
            u32::try_from(mine.len()).unwrap_or(u32::MAX)
        };
        let short = spec.wants.saturating_sub(standing + warming);
        if short == 0 {
            return;
        }
        let serving: BTreeSet<String> = self
            .engines
            .lock()
            .await
            .values()
            .flat_map(|v| v.members.values().cloned())
            .collect();
        let connected: BTreeMap<String, Outbox> = self
            .daemons
            .lock()
            .await
            .iter()
            .map(|(n, d)| (n.clone(), d.tx.clone()))
            .collect();
        let candidates: Vec<(String, u32)> = {
            let topo = self.topology.lock().await;
            let launched = self.specs.launched.lock().await;
            let taken = |at: &(String, u32)| {
                standing_at.contains(at) || launched.values().any(|m| m.contains_key(at))
            };
            topo.devices()
                .filter(|d| connected.contains_key(&d.node))
                .filter(|d| !cordoned.contains(&d.node) && !serving.contains(&d.node))
                .filter(|d| spec.device_class.is_empty() || d.class == spec.device_class)
                .filter(|d| spec.domain.is_empty() || d.domain == spec.domain)
                .map(|d| (d.node.clone(), d.index))
                .filter(|at| !taken(at))
                .take(usize::try_from(short).unwrap_or(usize::MAX))
                .collect()
        };
        for (node, device) in candidates {
            let Some(tx) = connected.get(&node) else {
                continue;
            };
            info!("pool {pool}: a spare launches on {node} device {device}");
            let launch = CoordToDaemon {
                term: 0,
                kind: Some(coord_to_daemon::Kind::LaunchSpare(LaunchSpare {
                    engine: String::new(),
                    rank: None,
                    device,
                    command: spec.command.clone(),
                    env: spec.env.clone(),
                    pool: pool.to_owned(),
                    engines: spec.engines.clone(),
                })),
            };
            if tx.send(launch).is_ok() {
                self.specs
                    .launched
                    .lock()
                    .await
                    .entry(pool.to_owned())
                    .or_default()
                    .insert((node.clone(), device), now);
                self.pool_event(
                    pool,
                    Severity::Normal,
                    "SpareLaunched",
                    format!("on {node} device {device}"),
                )
                .await;
            }
        }
    }

    /// The pool's spares warming: launched by this coordinator, not yet standing.
    async fn warming(&self, pool: &str) -> u32 {
        self.specs
            .launched
            .lock()
            .await
            .get(pool)
            .map_or(0, |m| u32::try_from(m.len()).unwrap_or(u32::MAX))
    }

    /// Where the spares of `spec`'s engines stand, by node and device, whatever the selector.
    async fn pool_standing(&self, spec: &PoolSpec) -> BTreeSet<(String, u32)> {
        let views = self.engines.lock().await;
        spec.engines
            .iter()
            .filter_map(|e| views.get(e))
            .flat_map(|v| v.standing.keys().cloned())
            .collect()
    }

    /// The events kept for `pool`, oldest first.
    pub async fn pool_events(&self, pool: &str) -> Vec<crate::status::Event> {
        self.specs
            .pool_status
            .lock()
            .await
            .get(pool)
            .map(crate::status::Status::events)
            .unwrap_or_default()
    }

    /// Record an event in `pool`'s log.
    async fn pool_event(&self, pool: &str, severity: Severity, reason: &str, message: String) {
        self.specs
            .pool_status
            .lock()
            .await
            .entry(pool.to_owned())
            .or_default()
            .record(now_ms(), severity, reason, message);
    }
}
