// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator API's objects (decision 0042): an engine, a node and a pool, each its
//! `metadata`, its `spec` (what is wanted, written through the API) and its `status` (what the
//! coordinator observes, with conditions); the lists of them, the writes, and a watch's events.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::readiness::{EngineReadiness, NodeReadiness};
use crate::status::Condition;
use crate::topology::Device;

/// An object's name and versions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Metadata {
    /// The object's name: the engine's in the policy, the node's, the pool's.
    pub name: String,
    /// Raised by each write of its spec; 0 for a spec never written through the API.
    pub generation: u64,
    /// Raised by any change of the object as the API shows it, spec or status: what a write's
    /// precondition and a watch name. Opaque: `<term>.<n>`, comparable only for equality.
    pub resource_version: String,
    /// Who wrote its spec last, as the API authenticated them, or the notice that did.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub written_by: String,
    /// When its spec was last written, Unix milliseconds; 0 for a spec never written.
    #[serde(default)]
    pub written_at_ms: i64,
}

/// An engine: its spec is its entry in the policy, read-only here; its status, what a
/// framework or a scheduler reads before it acts on the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct Engine {
    /// Its name and versions.
    pub metadata: Metadata,
    /// Its entry in the policy, in the policy's protobuf JSON.
    #[schema(value_type = Object)]
    pub spec: serde_json::Value,
    /// What the coordinator observes of it.
    pub status: EngineReadiness,
}

/// A node's spec: what an operator, a cloud's notice or a Node's label asks of it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeSpec {
    /// Cordoned: no spare takes a position on it; what it holds stays.
    #[serde(default)]
    pub unschedulable: bool,
    /// Drained: every rank on it moved to a spare by the deadline, then the node released. A
    /// drained node is cordoned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drain: Option<Drain>,
    /// Why, for people.
    #[serde(default)]
    pub reason: String,
}

/// A drain's deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Drain {
    /// When the node goes, Unix milliseconds: a time, never a lead, so the same spec written
    /// twice asks the same thing.
    pub deadline_ms: i64,
}

/// One engine's part of a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeEngine {
    /// The engine.
    pub engine: String,
    /// Its ranks on the node.
    pub ranks: Vec<u32>,
    /// Whether a loss of the node now is replaced, not rebuilt.
    pub reclaimable: bool,
    /// The outcome a loss of the node now would take.
    pub outcome: String,
    /// Why, when it is not replaced.
    pub why: String,
}

impl NodeEngine {
    /// `engine`'s part of a node, from its readiness.
    #[must_use]
    pub fn of(engine: &str, n: &NodeReadiness) -> Self {
        Self {
            engine: engine.to_owned(),
            ranks: n.ranks.clone(),
            reclaimable: n.reclaimable,
            outcome: n.outcome.clone(),
            why: n.why.clone(),
        }
    }
}

/// What the coordinator observes of a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeStatus {
    /// Its daemon is linked to the coordinator.
    pub linked: bool,
    /// Its devices, as its daemon reported them.
    pub devices: Vec<Device>,
    /// Where its store takes replicas; absent when it takes none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replica_address: Option<String>,
    /// Each engine with ranks on it.
    pub engines: Vec<NodeEngine>,
    /// `Draining` (reason `Scheduled`: its notices went out; `Waiting`: other drains move as
    /// many positions as the policy allows at once; `Blocked`: its change was refused; or
    /// `NotDraining`) and `Degraded` (`DrainBlocked`). A cordon is the spec's `unschedulable`.
    pub conditions: Vec<Condition>,
}

/// A node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct Node {
    /// Its name and versions.
    pub metadata: Metadata,
    /// What is asked of it.
    pub spec: NodeSpec,
    /// What the coordinator observes of it.
    pub status: NodeStatus,
}

/// A pool's spec (decision 0031). Its launcher is written apart, under a grant of its own.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PoolSpec {
    /// The engines it holds spares for, of one shape.
    pub engines: Vec<String>,
    /// How many spares it wants standing.
    pub wants: u32,
    /// The device class a spare must stand on; empty for any.
    #[serde(default)]
    pub device_class: String,
    /// The network domain a spare must stand in; empty for any.
    #[serde(default)]
    pub domain: String,
    /// Why, for people.
    #[serde(default)]
    pub reason: String,
}

/// What the coordinator observes of a pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct PoolStatus {
    /// Spares standing for any of its engines on an uncordoned node its selector admits, each
    /// process counted once.
    pub standing: u32,
    /// Spares taking a position in a change in flight.
    pub taking: u32,
    /// Spares launched for it that have not yet registered.
    pub warming: u32,
    /// `wants` less `standing` and `warming`: what its launches, or a scheduler, keep at zero.
    pub deficit: u32,
    /// `Satisfied` and `ShapeAgrees`.
    pub conditions: Vec<Condition>,
}

/// A pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct Pool {
    /// Its name and versions.
    pub metadata: Metadata,
    /// What is asked of it.
    pub spec: PoolSpec,
    /// What the coordinator observes of it.
    pub status: PoolStatus,
}

/// How a node's daemon starts one of a pool's spares: a command every node of the pool runs,
/// so writing it is a grant apart from the pool's (`update pools/launcher`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Launcher {
    /// The command and its arguments; empty, the daemons launch nothing.
    #[serde(default)]
    pub command: Vec<String>,
    /// The spares' environment.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

/// The precondition a write may carry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct WriteMetadata {
    /// The object's `resourceVersion` as the writer read it: the write is refused, 409
    /// `Conflict`, if the object changed since. Absent, the write applies as it stands.
    #[serde(default)]
    pub resource_version: Option<String>,
}

/// A write of an object's spec. An object read and sent back is a valid write: its `status` and
/// its metadata other than `resourceVersion` are ignored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct Write<S> {
    /// The write's precondition.
    #[serde(default)]
    pub metadata: WriteMetadata,
    /// The spec as wanted.
    pub spec: S,
}

/// A list's version.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ListMetadata {
    /// The version the list was read at: a watch from it misses no change. On every page of a
    /// paged list, the first page's.
    pub resource_version: String,
    /// The token for the next page of a paged list (`?limit=N`); absent on the last page.
    #[serde(rename = "continue", default, skip_serializing_if = "Option::is_none")]
    pub continue_token: Option<String>,
}

/// Every object of a kind, in name order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct List<T> {
    /// The list's version.
    pub metadata: ListMetadata,
    /// The objects.
    pub items: Vec<T>,
}

/// What a watch's event says happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "UPPERCASE")]
pub enum EventType {
    /// The object appeared.
    Added,
    /// The object changed.
    Modified,
    /// The object went.
    Deleted,
    /// No change: the version the watch is current at, its object's metadata carrying it.
    Bookmark,
}

/// One line of a watch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct WatchEvent {
    /// What happened.
    #[serde(rename = "type")]
    pub kind: EventType,
    /// The object as it is after the change; as it was last shown, for a deletion.
    #[schema(value_type = Object)]
    pub object: serde_json::Value,
}

/// What a stacks request names: a serving rank (or the spare preloaded for it or taking it), a
/// spare standing for any position by its node and device, or the engine's head.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StacksRequest {
    /// The serving rank.
    #[serde(default)]
    pub rank: Option<u32>,
    /// A spare standing for any position: its node.
    #[serde(default)]
    pub node: Option<String>,
    /// A spare standing for any position: its device.
    #[serde(default)]
    pub device: Option<u32>,
    /// The engine's head, its launcher asked too (decision 0032).
    #[serde(default)]
    pub head: bool,
}

/// Whether a node may be given back now: it holds no position, nothing is staged
/// onto or from it, and no change in flight touches it. What maintenance and reclaim tooling asks
/// before it returns a node, rather than reading conditions and waiting for `DrainDone`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeRelease {
    /// The node.
    pub node: String,
    /// Whether it may be given back.
    pub releasable: bool,
    /// Why not, each reason in words; empty when it may.
    pub why: Vec<String>,
}
