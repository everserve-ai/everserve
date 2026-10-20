// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The objects the operator API shows, and their changes (decision 0042). Every object is
//! computed from the coordinator's state in one place and compared with the form it was last
//! shown in; each difference is a change, numbered once for the coordinator's term and kept, the
//! last [`Pacing::kept`] of them, for watches. A read takes the objects as they are now, a write's
//! answer as they are after it, and a watch the changes after the version it names, so every
//! version a client sees means the same object. A field added to an object is watched with no
//! call site to remember: what the API shows is what is compared.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;
use tracing::warn;

use super::objects::{
    Engine, EventType, Metadata, Node, NodeEngine, NodeSpec, NodeStatus, Pool, PoolSpec,
    PoolStatus, WatchEvent,
};
use crate::coord::Coordinator;
use crate::proto::everserve::spec::v1::resource_spec;

/// How the feed keeps pace with the coordinator, and with its watches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pacing {
    /// How often the objects are compared with their last shown form while no request does it.
    pub refresh: Duration,
    /// How often an idle watch is told the version it is current at.
    pub bookmark: Duration,
    /// How many changes are kept for watches: a watch from an older version lists again.
    pub kept: usize,
}

impl Pacing {
    /// A coordinator's: a change seen within a quarter second, a bookmark every 30 s as
    /// Kubernetes' watch cache sends one, and the last 4096 changes kept.
    pub const SERVED: Self = Self {
        refresh: Duration::from_millis(250),
        bookmark: Duration::from_secs(30),
        kept: 4096,
    };
}

/// A kind of object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    /// Engines.
    Engines,
    /// Nodes.
    Nodes,
    /// Pools.
    Pools,
}

/// Why a watch cannot start from a version.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Stale {
    /// Older than the changes kept, or of another term: list again.
    #[error("resourceVersion {0} is older than the changes kept, or of another term: list again")]
    Gone(String),
    /// Not a version this API wrote.
    #[error("resourceVersion {0} is not one this API gave")]
    Invalid(String),
}

/// A change: its number in the term, its object's kind, and the event.
#[derive(Debug, Clone)]
struct Change {
    n: u64,
    kind: Kind,
    event: WatchEvent,
}

/// An object as the coordinator's state gives it, compared as it is: only one that changed is
/// serialized, so a refresh that finds nothing new costs the objects, not their JSON.
#[derive(Debug, Clone, PartialEq)]
enum Object {
    Engine(Box<Engine>),
    Node(Box<Node>),
    Pool(Box<Pool>),
}

impl Object {
    fn value(&self) -> Result<Value, serde_json::Error> {
        match self {
            Self::Engine(e) => serde_json::to_value(e),
            Self::Node(n) => serde_json::to_value(n),
            Self::Pool(p) => serde_json::to_value(p),
        }
    }
}

/// An object as last shown: as compared, as serialized without its version, and its version's
/// number.
#[derive(Debug)]
struct Shown {
    object: Object,
    value: Value,
    n: u64,
}

#[derive(Debug, Default)]
struct State {
    /// The term the versions are numbered in; a new one starts the numbers again.
    term: u64,
    /// The last change's number.
    last: u64,
    /// Each object as last shown.
    shown: BTreeMap<(Kind, String), Shown>,
    /// The last changes, oldest first.
    changes: VecDeque<Change>,
}

impl State {
    fn version(&self, n: u64) -> String {
        format!("{}.{n}", self.term)
    }

    /// The number of `from`, a version a watch can start after: one of this term, no newer than
    /// the last change, and with every change after it kept.
    fn watchable(&self, from: &str) -> Result<u64, Stale> {
        let (term, n) = parse(from).ok_or_else(|| Stale::Invalid(from.to_owned()))?;
        if term != self.term {
            return Err(Stale::Gone(from.to_owned()));
        }
        if n > self.last {
            return Err(Stale::Invalid(from.to_owned()));
        }
        // the oldest change kept must be the one right after `n`, or older
        let oldest = self.changes.front().map_or(self.last + 1, |c| c.n);
        if n + 1 < oldest {
            return Err(Stale::Gone(from.to_owned()));
        }
        Ok(n)
    }

    /// `object` as shown, with the version of its number.
    fn stamped(&self, object: &Value, n: u64) -> Value {
        let mut object = object.clone();
        object["metadata"]["resourceVersion"] = Value::String(self.version(n));
        object
    }
}

/// The objects, as the API shows them, and their changes.
pub struct Feed {
    coord: Arc<Coordinator>,
    pacing: Pacing,
    /// Each engine's spec, the policy's, serialized once: the policy does not change.
    specs: BTreeMap<String, Value>,
    /// Held from computing the objects to numbering their changes, so two refreshes never apply
    /// their snapshots out of order.
    refreshing: Mutex<()>,
    state: Mutex<State>,
    /// The last change's number, sent on each change and on a new term.
    changed: watch::Sender<(u64, u64)>,
    /// Fired when the coordinator stops: every watch ends.
    cancel: CancellationToken,
}

impl Feed {
    /// The feed over `coord`'s state, kept at `pacing` until `cancel` fires.
    pub fn start(coord: Arc<Coordinator>, pacing: Pacing, cancel: &CancellationToken) -> Arc<Self> {
        let specs = coord
            .policy()
            .engines
            .iter()
            .filter_map(|e| Some((e.name.clone(), serde_json::to_value(e).ok()?)))
            .collect();
        let feed = Arc::new(Self {
            coord,
            pacing,
            specs,
            refreshing: Mutex::new(()),
            state: Mutex::new(State::default()),
            changed: watch::channel((0, 0)).0,
            cancel: cancel.clone(),
        });
        let running = Arc::clone(&feed);
        let cancel = cancel.clone();
        tokio::spawn(async move {
            let mut every = tokio::time::interval(pacing.refresh);
            every.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    () = cancel.cancelled() => return,
                    _ = every.tick() => running.refresh().await,
                }
            }
        });
        feed
    }

    /// The coordinator the objects are computed from.
    #[must_use]
    pub fn coordinator(&self) -> &Arc<Coordinator> {
        &self.coord
    }

    /// Compare every object with its last shown form, and number each difference.
    pub async fn refresh(&self) {
        let _one = self.refreshing.lock().await;
        let now = objects(&self.coord, &self.specs).await;
        let present: BTreeSet<(Kind, String)> = now.keys().cloned().collect();
        let term = self.coord.term();
        let mut state = self.state.lock().await;
        if state.term != term {
            // a new term numbers its versions again: a watch of the old one lists again
            *state = State {
                term,
                ..State::default()
            };
        }
        let mut changes = Vec::new();
        for ((kind, name), object) in now {
            let event = match state.shown.get(&(kind, name.clone())) {
                Some(shown) if shown.object == object => continue,
                Some(_) => EventType::Modified,
                None => EventType::Added,
            };
            match object.value() {
                Ok(value) => changes.push((kind, name, event, Some(object), value)),
                Err(e) => warn!("operator API: {name} cannot be shown: {e}"),
            }
        }
        let gone: Vec<(Kind, String)> = state
            .shown
            .keys()
            .filter(|k| !present.contains(*k))
            .cloned()
            .collect();
        for (kind, name) in gone {
            if let Some(shown) = state.shown.get(&(kind, name.clone())) {
                let value = shown.value.clone();
                changes.push((kind, name, EventType::Deleted, None, value));
            }
        }
        if changes.is_empty() {
            self.changed.send_if_modified(|v| {
                let was = *v;
                *v = (term, state.last);
                was != *v
            });
            return;
        }
        for (kind, name, event, object, value) in changes {
            state.last += 1;
            let n = state.last;
            let stamped = state.stamped(&value, n);
            match object {
                Some(object) => {
                    state.shown.insert((kind, name), Shown { object, value, n });
                }
                None => {
                    state.shown.remove(&(kind, name));
                }
            }
            state.changes.push_back(Change {
                n,
                kind,
                event: WatchEvent {
                    kind: event,
                    object: stamped,
                },
            });
            if state.changes.len() > self.pacing.kept {
                state.changes.pop_front();
            }
        }
        let _ = self.changed.send((term, state.last));
    }

    /// Every object of `kind` as it is now, in name order, and the version they were read at.
    pub async fn list(&self, kind: Kind) -> (String, Vec<Value>) {
        self.refresh().await;
        let state = self.state.lock().await;
        let items = state
            .shown
            .iter()
            .filter(|((k, _), _)| *k == kind)
            .map(|(_, shown)| state.stamped(&shown.value, shown.n))
            .collect();
        (state.version(state.last), items)
    }

    /// Up to `limit` objects of `kind` as they are now, in name order, after the page
    /// `continued` ends (from the first when None): Kubernetes' paged list. Every page carries the
    /// first page's version, so a watch from it after the last page replays whatever changed
    /// while the pages were read; a page whose first version is no longer watchable is
    /// [`Stale::Gone`], and the client lists again.
    ///
    /// # Errors
    ///
    /// [`Stale::Gone`] once the first page's version can no longer be watched from;
    /// [`Stale::Invalid`] for a token this API did not give.
    pub async fn page(
        &self,
        kind: Kind,
        limit: usize,
        continued: Option<&str>,
    ) -> Result<Page, Stale> {
        let at = continued
            .map(|token| {
                let (version, name) = token
                    .split_once(':')
                    .and_then(|(v, hex)| Some((v.to_owned(), unhex(hex)?)))
                    .ok_or_else(|| Stale::Invalid(token.to_owned()))?;
                Ok::<_, Stale>((version, name))
            })
            .transpose()?;
        self.refresh().await;
        let state = self.state.lock().await;
        let (version, after) = match at {
            Some((version, after)) => {
                state.watchable(&version)?;
                (version, Some(after))
            }
            None => (state.version(state.last), None),
        };
        let from = (kind, after.clone().unwrap_or_default());
        let mut items = Vec::new();
        let mut last: Option<&str> = None;
        let mut next = None;
        for ((k, name), shown) in state.shown.range(from..) {
            if *k != kind {
                break;
            }
            if after.as_deref() == Some(name.as_str()) {
                continue;
            }
            if let Some(last) = last
                && items.len() == limit.max(1)
            {
                next = Some(format!("{version}:{}", hex(last)));
                break;
            }
            items.push(state.stamped(&shown.value, shown.n));
            last = Some(name);
        }
        Ok(Page {
            version,
            items,
            next,
        })
    }

    /// `name`'s object of `kind` as it is now.
    pub async fn get(&self, kind: Kind, name: &str) -> Option<Value> {
        self.refresh().await;
        let state = self.state.lock().await;
        state
            .shown
            .get(&(kind, name.to_owned()))
            .map(|shown| state.stamped(&shown.value, shown.n))
    }

    /// The changes of `kind` after the version `from`, and the number of the last change.
    ///
    /// # Errors
    ///
    /// [`Stale::Gone`] for a version of another term or older than the changes kept;
    /// [`Stale::Invalid`] for one this API did not give.
    pub async fn since(&self, kind: Kind, from: &str) -> Result<(Vec<WatchEvent>, u64), Stale> {
        let state = self.state.lock().await;
        let n = state.watchable(from)?;
        let events = state
            .changes
            .iter()
            .filter(|c| c.n > n && c.kind == kind)
            .map(|c| c.event.clone())
            .collect();
        Ok((events, state.last))
    }

    /// The version a number of the current term names.
    pub async fn version(&self, n: u64) -> String {
        self.state.lock().await.version(n)
    }

    /// How the feed keeps pace.
    #[must_use]
    pub fn pacing(&self) -> Pacing {
        self.pacing
    }

    /// Fired when the coordinator stops.
    #[must_use]
    pub fn stopped(&self) -> &CancellationToken {
        &self.cancel
    }

    /// The term and the last change's number, as each change sends them.
    #[must_use]
    pub fn changes(&self) -> watch::Receiver<(u64, u64)> {
        self.changed.subscribe()
    }
}

/// One page of a list: the first page's version, the objects, and the token for the next page
/// when there is one.
#[derive(Debug)]
pub struct Page {
    /// The version the first page was read at: a watch from it misses no change.
    pub version: String,
    /// The objects, in name order.
    pub items: Vec<Value>,
    /// The `continue` token for the next page; None on the last.
    pub next: Option<String>,
}

/// `name` in lowercase hex: a token carries any name without escaping.
fn hex(name: &str) -> String {
    use std::fmt::Write as _;
    name.bytes().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// The name `hex` wrote, or None for text it could not have written.
fn unhex(text: &str) -> Option<String> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    let bytes = (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect::<Option<Vec<u8>>>()?;
    String::from_utf8(bytes).ok()
}

/// A version's term and number.
#[must_use]
pub fn parse(version: &str) -> Option<(u64, u64)> {
    let (term, n) = version.split_once('.')?;
    Some((term.parse().ok()?, n.parse().ok()?))
}

/// Every object `coord` holds, as the API shows it, without its version.
async fn objects(
    coord: &Arc<Coordinator>,
    specs: &BTreeMap<String, Value>,
) -> BTreeMap<(Kind, String), Object> {
    let readiness = coord.readiness().await;
    let mut out = BTreeMap::new();
    let mut on_node: BTreeMap<String, Vec<NodeEngine>> = BTreeMap::new();
    for status in readiness.engines {
        for n in &status.nodes {
            on_node
                .entry(n.node.clone())
                .or_default()
                .push(NodeEngine::of(&status.engine, n));
        }
        let spec = specs.get(&status.engine).cloned().unwrap_or(Value::Null);
        let engine = Engine {
            metadata: Metadata {
                name: status.engine.clone(),
                ..Metadata::default()
            },
            spec,
            status,
        };
        out.insert(
            (Kind::Engines, engine.metadata.name.clone()),
            Object::Engine(Box::new(engine)),
        );
    }
    for view in coord.node_views().await {
        let (metadata, spec) = match &view.spec {
            Some(r) => match &r.kind {
                Some(resource_spec::Kind::Node(n)) => (
                    Metadata {
                        name: view.node.clone(),
                        generation: r.generation,
                        written_by: r.written_by.clone(),
                        written_at_ms: r.written_at_ms,
                        ..Metadata::default()
                    },
                    NodeSpec {
                        unschedulable: n.unschedulable,
                        drain: (n.drain_by_ms != 0).then_some(super::objects::Drain {
                            deadline_ms: n.drain_by_ms,
                        }),
                        reason: n.reason.clone(),
                    },
                ),
                _ => continue,
            },
            None => (
                Metadata {
                    name: view.node.clone(),
                    ..Metadata::default()
                },
                NodeSpec::default(),
            ),
        };
        let node = Node {
            metadata,
            spec,
            status: NodeStatus {
                linked: view.linked,
                devices: view.devices,
                replica_address: view.replica_address,
                engines: on_node.remove(&view.node).unwrap_or_default(),
                conditions: view.conditions,
            },
        };
        out.insert((Kind::Nodes, view.node), Object::Node(Box::new(node)));
    }
    for p in coord.pools().await {
        let pool = Pool {
            metadata: Metadata {
                name: p.name.clone(),
                generation: p.generation,
                written_by: p.written_by,
                written_at_ms: p.written_at_ms,
                ..Metadata::default()
            },
            spec: PoolSpec {
                engines: p.engines,
                wants: p.wants,
                device_class: p.device_class,
                domain: p.domain,
                reason: p.reason,
            },
            status: PoolStatus {
                standing: p.standing,
                taking: p.taking,
                warming: p.warming,
                deficit: p.deficit,
                conditions: p.conditions,
            },
        };
        out.insert((Kind::Pools, p.name), Object::Pool(Box::new(pool)));
    }
    out
}
