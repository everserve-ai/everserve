// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! An engine's status as the operator API shows it (decision 0042): what a framework, a
//! scheduler or an operator reads before it acts on an engine or reclaims a node
//! (docs/MULTINODE.md section 8). Every figure is state the coordinator already holds for
//! driving changes; nothing here is a second source of truth. An expected pause is the last
//! pause this coordinator measured for that outcome on that engine, and absent until one was.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::proto::everserve::signals::v1::{
    CompileCache, InstancePhase, SlotChoice, SoftwareIdentity,
};
use crate::status::Condition;

/// Every engine's readiness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Readiness {
    /// Per engine, in name order.
    pub engines: Vec<EngineReadiness>,
    /// Changes the coordinator is driving right now, across engines.
    pub changes_in_flight: usize,
}

/// An instance's state in a framework's terms (decision 0041's second obligation), derived from
/// its engine's view in one place, the first that holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum InstanceState {
    /// Down (decision 0032): its head is lost or stopped, or a change failed. The framework's
    /// own recovery, Dynamo's migration, takes over.
    Failed,
    /// A change is in flight: a rank replaced, a drain's ranks moved, a shrink. The framework
    /// holds its own recovery off while Everserve's runs.
    Recovering,
    /// Serving with ranks lost under SHRINK: its capacity is down.
    Degraded,
    /// Every position of its world is held.
    Serving,
    /// None of these: bring-up, before every position is held.
    Starting,
}

impl InstanceState {
    /// The state a phase on the wire names; starting for one this build does not know.
    #[must_use]
    pub fn from_phase(phase: InstancePhase) -> Self {
        match phase {
            InstancePhase::InstanceFailed => Self::Failed,
            InstancePhase::InstanceRecovering => Self::Recovering,
            InstancePhase::InstanceDegraded => Self::Degraded,
            InstancePhase::InstanceServing => Self::Serving,
            InstancePhase::InstanceStarting | InstancePhase::Unspecified => Self::Starting,
        }
    }
}

/// One engine as the scheduler sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct EngineReadiness {
    /// The engine's name in the policy.
    pub engine: String,
    /// The instance's state in a framework's terms (decision 0041): what an integration reports
    /// to the framework that routes to the engine.
    pub state: InstanceState,
    /// The last committed membership epoch. A change in flight shows in `change`, never here:
    /// between a change's claim and its commit the members it takes are gone from `members` and
    /// its spares are not yet in it.
    pub epoch: u64,
    /// The change in flight, if one is (decision 0032).
    #[serde(default)]
    pub change: Option<ChangeReadiness>,
    /// Ranks the engine started with.
    pub world_size: u32,
    /// Ranks serving now.
    pub members: Vec<MemberReadiness>,
    /// Spares standing behind the engine, each for one rank.
    pub spares: Vec<SpareReadiness>,
    /// Spares standing for any position of the engine (decision 0030): they hold the model in host
    /// memory and take the shard of the rank a change names, so REPLACE is available for any rank
    /// while one is in the pool.
    pub any_spares: Vec<AnySpareReadiness>,
    /// Ranks lost to SHRINK, still counted against the policy's concurrent losses.
    pub lost: Vec<u32>,
    /// Every node holding a member: whether it can be reclaimed, and how.
    pub nodes: Vec<NodeReadiness>,
    /// The slot-collective variant the engine runs on each node and how it was decided, keyed by
    /// node: an engine node's from its client, a spare node's from its spare (decision 0024);
    /// empty until one registered. `host` is the slower path a scheduler may want to know about.
    pub slots: BTreeMap<String, SlotReadiness>,
    /// The engine's guarantee class (decision 0030): `exact` or `continuous`.
    pub guarantee: String,
    /// The compile cache key the engine's first registered process reported (decision 0029): one
    /// engine, one compiled artifact, which every later process must report; absent until one
    /// did, and again once every process of the engine has gone, so a re-formed engine sets its
    /// own.
    pub compile_key: Option<String>,
    /// The engine has a key, every serving rank reported it, and every spare in the pool reported
    /// the pair of the rank it stands in for (digests differ by rank on a pipeline engine, since
    /// each stage compiles its own graph). False means a spare's rows would not match its
    /// position's, or a process reported none, so nothing holds it: a scheduler does not run an
    /// exact change on the engine either way.
    pub compile_cache_consistent: bool,
    /// The engine's head (decision 0032): the process that owns its scheduler and executor, and
    /// its launcher. Absent until it registered, and while the engine is down.
    #[serde(default)]
    pub head: Option<HeadReadiness>,
    /// Why the engine is down and since when (decision 0032); absent while it is not. A down
    /// engine takes no change until its head registers again.
    #[serde(default)]
    pub down: Option<DownReadiness>,
    /// The last committed change's record: what it cost, and where the time went.
    #[serde(default)]
    pub last_change: Option<LastChange>,
    /// A second head that registered on another node while the engine's head was alive, refused:
    /// two processes believe they own the engine, which an operator must resolve.
    #[serde(default)]
    pub head_conflict: Option<HeadReadiness>,
    /// Spares kept out of the pool for what they run (decision 0034), with the field that
    /// differed: a spare an operator expects in the pool and does not find is here.
    #[serde(default)]
    pub refused_spares: Vec<RefusedSpareReadiness>,
    /// The cover plan delivered for the engine and how much of it is formed (decision 0035);
    /// absent while cover is not planned.
    #[serde(default)]
    pub cover: Option<CoverReadiness>,
    /// What its status means (decision 0036): Serving, Progressing, Covered, Degraded and Down,
    /// each with its reason and when it last changed.
    #[serde(default)]
    pub conditions: Vec<Condition>,
}

/// A committed change's record. Its moments are on the coordinator's clock, microseconds since
/// the Unix epoch, except each acknowledgement's, which is its daemon's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct LastChange {
    /// The epoch it committed.
    pub epoch: u64,
    /// `replace` or `shrink`.
    pub outcome: String,
    /// The ranks it took.
    pub lost: Vec<u32>,
    /// The report that moved it.
    pub reported_at_us: u64,
    /// Its delivery to the survivors and spares.
    pub delivered_at_us: u64,
    /// The last acknowledgement.
    pub acknowledged_at_us: u64,
    /// The commit.
    pub committed_at_us: u64,
    /// How it used cover (decision 0035): through which covered set, or formed in the change;
    /// empty while cover is not planned.
    #[serde(default)]
    pub cover: String,
    /// Every acknowledgement it waited on: which process the change waited for longest.
    pub acks: Vec<AckStamp>,
    /// The join payloads routed for its epoch (decision 0024), at most 64, before or after its
    /// commit: the control plane's share of a take's join and its answers.
    pub joins: Vec<JoinStamp>,
}

/// An acknowledgement and when its daemon received it, on that daemon's clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct AckStamp {
    /// The rank that acknowledged.
    pub rank: u32,
    /// When its daemon received it.
    pub received_at_us: u64,
}

/// A join payload through the coordinator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct JoinStamp {
    /// The rank whose position it joins.
    pub rank: u32,
    /// The epoch it is for.
    pub epoch: u64,
    /// True for the executor's payload to the process, false for the process's answer back.
    pub from_client: bool,
    /// Its payload's size.
    pub bytes: u64,
    /// When the coordinator read it from its node's daemon.
    pub received_at_us: u64,
    /// Each node it was forwarded to, and when its forward was queued to that node's link.
    pub forwarded_at_us: BTreeMap<String, u64>,
}

/// A change in flight: the epoch it claimed and the ranks it takes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ChangeReadiness {
    /// The epoch the change claimed; readiness's `epoch` reaches it at the commit.
    pub epoch: u64,
    /// The ranks it takes.
    pub lost: Vec<u32>,
}

/// An engine's head as its node's daemon saw it register (decision 0032).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct HeadReadiness {
    /// The node it runs on.
    pub node: String,
    /// The engine core's process id, from the daemon socket's peer credentials; absent when the
    /// daemon cannot see it, as a process in another pod on Kubernetes.
    #[serde(default)]
    pub pid: Option<u32>,
    /// The process the operator launched, as the head reported it in its own pid namespace; the
    /// engine core's own when it runs in that process. For people: nothing is signalled by it.
    pub launcher_pid: u32,
    /// When it registered, Unix milliseconds on the coordinator's clock.
    pub registered_at_ms: i64,
    /// The address its engine core registered; empty when it named none. Front ends find the
    /// core through their framework's own discovery; this is for people.
    #[serde(default)]
    pub host: String,
}

/// Why an engine is down (decision 0032).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct DownReadiness {
    /// Since when, Unix milliseconds on the coordinator's clock.
    pub at_ms: i64,
    /// `stopped`, `failed` or `lost` from the head's departure, or `change_failed` when a change
    /// failed after its delivery.
    pub cause: String,
    /// The head's goodbye, or the failed change's error.
    pub detail: String,
}

/// The compiled-kernel cache a process reported running with (decision 0029).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CompileCacheReadiness {
    /// vLLM's `torch_compile_cache` key.
    pub key: String,
    /// The digest over every file under it, the autotune results included.
    pub digest: String,
}

impl CompileCacheReadiness {
    /// The wire pair as the view reports it.
    #[must_use]
    pub fn from_wire(c: &CompileCache) -> Self {
        Self {
            key: c.key.clone(),
            digest: c.digest.clone(),
        }
    }
}

/// The engine's slot-collective choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct SlotReadiness {
    /// `p2p` or `host`.
    pub variant: String,
    /// `override`, `nvml`, `cache` or `probe`.
    pub how: String,
    /// What the probe found, for a human.
    pub detail: String,
    /// Seconds the decision took.
    pub seconds: f64,
}

impl SlotReadiness {
    /// The wire choice as the view reports it.
    #[must_use]
    pub fn from_wire(slot: &SlotChoice) -> Self {
        use crate::proto::everserve::signals::v1::slot_choice::{How, Variant};
        Self {
            variant: match slot.variant() {
                Variant::P2p => "p2p",
                Variant::Host => "host",
                Variant::Unspecified => "unspecified",
            }
            .to_owned(),
            how: match slot.how() {
                How::Override => "override",
                How::Nvml => "nvml",
                How::Cache => "cache",
                How::Probe => "probe",
                How::Unspecified => "unspecified",
            }
            .to_owned(),
            detail: slot.detail.clone(),
            seconds: slot.seconds,
        }
    }
}

/// One serving rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemberReadiness {
    /// The rank.
    pub rank: u32,
    /// Its node.
    pub node: String,
    /// Its device class, when the node reported devices.
    pub class: Option<String>,
    /// The replica addresses its store sends every commit to.
    pub replicates_to: Vec<String>,
    /// The compile cache it reported (decision 0029); absent when it reported none.
    pub compile_cache: Option<CompileCacheReadiness>,
    /// The process serving the rank, as its node's daemon saw it register.
    pub process: Option<ProcessReadiness>,
}

/// A process the control plane serves, as its node's daemon saw it register: the id is the
/// socket's peer credential, never the process's own word.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessReadiness {
    /// The process id on its node; absent when the daemon cannot see it, as a process in another
    /// pod on Kubernetes.
    #[serde(default)]
    pub pid: Option<u32>,
    /// When it registered, Unix milliseconds on the coordinator's clock.
    pub registered_at_ms: i64,
    /// What it runs, as it registered (decision 0034); absent when it reported nothing.
    #[serde(default)]
    pub identity: Option<IdentityReadiness>,
}

/// What a process runs (decision 0034), as readiness reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct IdentityReadiness {
    /// The engine, `vllm`.
    pub engine: String,
    /// The engine's source commit.
    pub engine_commit: String,
    /// The adapter's commit.
    pub adapter_commit: String,
    /// Whether the adapter was built from a tree with uncommitted changes.
    pub adapter_dirty: bool,
    /// The torch version.
    pub torch_version: String,
    /// The CUDA runtime, major.minor.
    pub cuda_version: String,
    /// The GPU driver.
    pub driver_version: String,
    /// The NCCL version.
    pub nccl_version: String,
    /// The model as served.
    pub model: String,
    /// The checkpoint's snapshot commit.
    pub model_revision: String,
    /// The digest over the checkpoint's snapshot.
    pub checkpoint_digest: String,
    /// The quantization method, empty for none.
    pub quantization: String,
    /// The model's compute dtype.
    pub model_dtype: String,
    /// The KV's dtype.
    pub kv_dtype: String,
    /// The KV's block size.
    pub kv_block_size: u32,
    /// The hash of the KV's layout for the position; empty for a spare standing for any position.
    pub kv_layout_hash: String,
    /// The parallel widths: tensor, pipeline, data, expert, context.
    pub layout: [u32; 5],
}

impl IdentityReadiness {
    /// From the wire message.
    #[must_use]
    pub fn from_wire(i: &SoftwareIdentity) -> Self {
        let kv = i.kv_format.clone().unwrap_or_default();
        let l = i.layout.unwrap_or_default();
        Self {
            engine: i.engine.clone(),
            engine_commit: i.engine_commit.clone(),
            adapter_commit: i.adapter_commit.clone(),
            adapter_dirty: i.adapter_dirty,
            torch_version: i.torch_version.clone(),
            cuda_version: i.cuda_version.clone(),
            driver_version: i.driver_version.clone(),
            nccl_version: i.nccl_version.clone(),
            model: i.model.clone(),
            model_revision: i.model_revision.clone(),
            checkpoint_digest: i.checkpoint_digest.clone(),
            quantization: i.quantization.clone(),
            model_dtype: i.model_dtype.clone(),
            kv_dtype: kv.dtype,
            kv_block_size: kv.block_size,
            kv_layout_hash: kv.layout_hash,
            layout: [l.tensor, l.pipeline, l.data, l.expert, l.context],
        }
    }
}

/// A spare kept out of an engine's pool for what it runs (decision 0034).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct RefusedSpareReadiness {
    /// Its node.
    pub node: String,
    /// Its device.
    pub device: u32,
    /// The rank it stood for; absent for a spare standing for any position.
    pub rank: Option<u32>,
    /// Why: the field that differed, or the identity it did not report.
    pub reason: String,
}

/// One spare standing for any position (decision 0030).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct AnySpareReadiness {
    /// Its node.
    pub node: String,
    /// Its device.
    pub device: u32,
    /// `standing` in the pool, or `taking` a rank in the change in flight.
    pub state: String,
    /// The spare's process, as its node's daemon saw it register.
    pub process: Option<ProcessReadiness>,
}

/// One spare in the pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct SpareReadiness {
    /// The rank it stands in for.
    pub rank: u32,
    /// Its node.
    pub node: String,
    /// Expected speed relative to the position's class (1.0: the same class).
    pub speed: f64,
    /// `warm`, `on_demand` or `mismatched` (docs/MULTINODE.md section 5).
    pub state: String,
    /// The compile cache it reported (decision 0029); absent when it reported none.
    pub compile_cache: Option<CompileCacheReadiness>,
    /// The spare's process, as its node's daemon saw it register.
    pub process: Option<ProcessReadiness>,
}

/// Whether one node can be reclaimed now and what the engine would do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeReadiness {
    /// The node.
    pub node: String,
    /// The engine's ranks on it.
    pub ranks: Vec<u32>,
    /// Losing every rank on it keeps the engine serving.
    pub reclaimable: bool,
    /// The outcome the policy would take: `replace`, `shrink` or `rebuild`.
    pub outcome: String,
    /// Why it is not reclaimable; empty when it is.
    pub why: String,
    /// The shortest notice the policy treats as a notice rather than a fault.
    pub min_notice_ms: u32,
    /// The last pause this coordinator measured for that outcome on this engine; absent until one
    /// was.
    pub expected_pause_ms: Option<u64>,
}

/// An engine's cover plan, as readiness shows it (decision 0035).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CoverReadiness {
    /// The plan's generation.
    pub generation: u64,
    /// The committed epoch it is for.
    pub epoch: u64,
    /// Its sets, in plan order.
    pub sets: Vec<CoverSetReadiness>,
    /// The losses it holds no set for, with why.
    pub uncovered: Vec<UncoveredReadiness>,
    /// Groups a member could not form under it.
    pub failed: Vec<FailedGroupReadiness>,
}

/// One set of a cover plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CoverSetReadiness {
    /// Its members, joined by `-`: a rank, or `R<rank>@<node>:<device>` for a spare.
    pub set: String,
    /// What it is for: `current`, or the loss it answers and how.
    pub purpose: String,
    /// Every member of every one of its groups reported it formed.
    pub covered: bool,
    /// Its groups.
    pub groups: usize,
    /// Of those, the ones every member reported.
    pub formed: usize,
}

/// A loss a cover plan holds no set for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UncoveredReadiness {
    /// The ranks lost together.
    pub lost: Vec<u32>,
    /// Why no set covers it.
    pub why: String,
}

/// A group a member could not form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct FailedGroupReadiness {
    /// The members it spans, in rank order: a group is shared by every set that needs it.
    pub group: String,
    /// `domain` or `fabric`.
    pub scope: String,
    /// The member that failed.
    pub member: String,
    /// Its error.
    pub error: String,
}
