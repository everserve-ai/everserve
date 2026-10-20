// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Cover plans as the coordinator holds them (decision 0035): the plan it delivered for each
//! engine, the groups each process reported formed, and the groups an older plan still holds a
//! port with.
//!
//! The plan is derived again after every change to an engine's view, as the spares' admission is
//! (0036): when the planned groups differ from the delivered ones, a new generation goes to every
//! daemon with a process of the engine, above every generation any daemon reported. A process
//! reports, at each plan's generation, every group of it that it holds, formed then or kept, so
//! coverage is rebuilt from current reports: a report at another generation counts toward
//! nothing. A set is covered when every member of each of its groups reported it.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use crate::clock::now_ms;
use crate::policy;
use crate::policy::cover::{
    GroupKey, Input, Layout, Member, PlannedSet, StandingSpare, Uncovered, answers, losses_of,
    set_key,
};
use crate::proto::everserve::coord::v1::{
    CoordToDaemon, CoverReport, EngineState, coord_to_daemon, cover_report,
};
use crate::proto::everserve::signals::v1::{
    CoverPlan, CoverSet, GroupId, Holder, Outcome, Scope, SetMember,
};
use crate::readiness::{
    CoverReadiness, CoverSetReadiness, FailedGroupReadiness, UncoveredReadiness,
};
use crate::status::Severity;

use super::Coordinator;
use super::outbox::Outbox;
use super::view::EngineView;

/// One engine's cover, as this coordinator holds it.
#[derive(Debug, Clone, Default)]
pub struct CoverState {
    /// The plan delivered last.
    pub plan: Option<CoverPlan>,
    /// Its sets, with what each is for, as the planner gave them.
    pub sets: Vec<PlannedSet>,
    /// The losses the planner could not cover, with why.
    pub uncovered: Vec<Uncovered>,
    /// The highest generation delivered or reported by any daemon: the next plan's is above it.
    pub generation: u64,
    /// The members that reported each group of the plan formed or kept, at its generation.
    pub formed: BTreeMap<GroupKey, BTreeSet<Member>>,
    /// The members that reported each group of the plan ready, at its generation: they hold the
    /// plan and wait for the group's id, which is relayed once every member is here (decision
    /// 0035, amended), so none starts a blocking join a member never answers.
    pub ready: BTreeMap<GroupKey, BTreeSet<Member>>,
    /// The groups the daemons' hellos said their processes hold, by member: they stand as cover
    /// until the plan's own reports replace them (a restarted coordinator, or a standby).
    pub held: BTreeMap<GroupKey, BTreeSet<Member>>,
    /// Groups a member could not form under the plan, with the error.
    pub failed: BTreeMap<GroupKey, (Member, String)>,
    /// The newest id each group's server member made for it (decision 0035, amended), which the
    /// group's other members form it with: relayed to their daemons as it comes, and to a daemon
    /// as it links.
    pub ids: BTreeMap<GroupKey, GroupId>,
    /// The next retry of a failed group: how many plans were delivered for retries since the
    /// engine was last fully covered, and when the next is due (decision 0036's backoff).
    pub retry: Option<(u32, Instant)>,
    /// A plan is being written to the store: no other is made meanwhile, and `again` asks for
    /// one more after it.
    pub planning: bool,
    pub again: bool,
    /// The highest generation a plan was numbered with, delivered or not: the next plan's is
    /// above it and `generation`, so no two plans share one.
    pub reserved: u64,
    /// The plan generation each daemon of the engine holds, as its hello said or as this
    /// coordinator delivered it: a daemon behind the delivered plan is sent it again.
    pub reached: BTreeMap<String, u64>,
    /// A plan could not be written to the store, so it was not delivered, and the engine is
    /// woken to plan again after [`UNWRITTEN_RETRY`]: one wake waits at a time.
    pub unwritten: bool,
}

/// How long a planned group's members have to form it when the policy says nothing
/// (`cover.formTimeoutMs` of 0).
pub const DEFAULT_FORM_TIMEOUT: Duration = Duration::from_secs(30);

/// What the coordinator allows beyond the members' own formation timeout before it counts a
/// group as failed: the plan's way to each member and each member's report back, over a daemon's
/// link each.
pub const FORM_TRANSIT: Duration = Duration::from_secs(2);

/// How long a restarted coordinator waits for the daemons it expects before it plans cover with
/// whichever relinked: until then it keeps the plan they hold (decision 0035).
pub const SETTLE: Duration = Duration::from_secs(10);

/// A plan made and not yet delivered: the store must hold it first (decision 0038).
pub(super) struct NextPlan {
    plan: CoverPlan,
    /// Its sets and the losses it leaves uncovered, the engine's once it is delivered.
    sets: Vec<PlannedSet>,
    uncovered: Vec<Uncovered>,
    /// Whether it is a failed group's retry (decision 0036's backoff).
    retry: bool,
}

/// How long an engine whose plan could not be written to the store waits before it plans again.
pub const UNWRITTEN_RETRY: Duration = Duration::from_secs(1);

/// How long a failed group waits before the plan is delivered again, after `attempts` retries:
/// one second, doubling, at most a minute.
#[must_use]
pub fn backoff(attempts: u32) -> Duration {
    Duration::from_secs(1u64 << attempts.min(6)).min(Duration::from_mins(1))
}

impl CoverState {
    /// Whether every member of `key`'s group reported it, or held it at the last hello.
    fn group_covered(&self, key: &GroupKey) -> bool {
        key.spans.iter().all(|m| {
            self.formed.get(key).is_some_and(|f| f.contains(m))
                || self.held.get(key).is_some_and(|h| h.contains(m))
        })
    }

    /// Whether every group the plan's `set` needs is covered; a set with no group of two
    /// members needs none.
    #[must_use]
    pub fn set_covered(&self, set: &[Member]) -> bool {
        let (Some(plan), Some(planned)) =
            (self.plan.as_ref(), self.sets.iter().find(|s| s.set == set))
        else {
            return false;
        };
        planned
            .groups
            .iter()
            .filter_map(|&i| plan.groups.get(i))
            .all(|g| self.group_covered(&GroupKey::of(g)))
    }

    /// A covered set that answers the loss of `lost`, if the plan holds one.
    #[must_use]
    pub fn covered_answer(&self, lost: &[u32]) -> Option<&PlannedSet> {
        self.sets
            .iter()
            .filter(|s| s.purpose != "current")
            .find(|s| answers(&s.set, lost) && self.set_covered(&s.set))
    }

    /// The plan's group over `spans` in `scope`, when `member` is one of them.
    fn group_of(&self, spans: &[SetMember], scope: i32, member: &Member) -> Option<GroupKey> {
        let key = GroupKey::from_parts(scope, spans);
        (key.spans.contains(member)
            && self
                .plan
                .as_ref()?
                .groups
                .iter()
                .any(|g| GroupKey::of(g) == key))
        .then_some(key)
    }
}

impl Coordinator {
    /// Plan `engine`'s cover again from its view, and deliver a new generation when the planned
    /// groups differ from the delivered ones. Nothing is planned while the policy's cover is off,
    /// while a change is in flight, or before every position has a holder. One plan of an engine
    /// is made at a time: asked for while one is written to the store, another follows it.
    pub(super) async fn replan(&self, engine: &str) {
        while self.plan_once(engine).await {}
    }

    /// One pass of [`Coordinator::replan`]; whether another was asked for while it wrote.
    async fn plan_once(&self, engine: &str) -> bool {
        if !self.policy.cover.as_ref().is_some_and(|c| c.enabled) {
            return false;
        }
        let Some(spec) = self.engine_spec(engine) else {
            return false;
        };
        let cordoned = self.cordoned().await;
        let linked: BTreeSet<String> = self.daemons.lock().await.keys().cloned().collect();
        let settled = self.started.elapsed() >= SETTLE;
        let targets = self.daemons_of(engine).await;
        let next = {
            let mut engines = self.engines.lock().await;
            let elsewhere = held_elsewhere(&engines, engine);
            let Some(view) = engines.get_mut(engine) else {
                return false;
            };
            if view.cover.planning {
                view.cover.again = true;
                return false;
            }
            // a restarted coordinator plans only from the whole picture: planned from the first
            // daemons to relink, each relink would deliver another plan, with other ports, and
            // undo the cover the processes hold
            if !settled && !expected_nodes(spec, view).is_subset(&linked) {
                return false;
            }
            let world = view.world_size.max(policy::world_size(spec));
            // planned only once every position's holder is known, by node and device
            if view.change.is_some()
                || world == 0
                || (0..world)
                    .any(|r| !view.members.contains_key(&r) || !view.devices.contains_key(&r))
            {
                return false;
            }
            let around = Around {
                cordoned: &cordoned,
                elsewhere: &elsewhere,
            };
            let Some(next) = self.plan_for(engine, spec, view, world, &around) else {
                // nothing new to plan: a daemon behind the delivered plan (one that relinked
                // after a restart, or missed a delivery) is sent it again, ports and all
                let behind = behind(view, targets);
                drop(engines);
                if let Some((plan, to)) = behind {
                    info!(
                        "{engine}: cover plan {} sent again to {:?}, behind it",
                        plan.generation,
                        to.keys().collect::<Vec<_>>()
                    );
                    self.send_plan(engine, plan, to).await;
                }
                return false;
            };
            view.cover.planning = true;
            next
        };
        let plan = next.plan.clone();
        // the store holds the plan's generation before any daemon holds the plan (decision
        // 0038): a successor numbers its first plan above it, so no generation is used twice,
        // even one that reached only a daemon lost since. The plan itself is the daemons': each
        // reports the one it holds, and a successor adopts the newest
        let generation = plan.generation;
        let kept = self
            .keep_status(engine, |s| {
                s.plan_generation = s.plan_generation.max(generation);
            })
            .await;
        if let Err(e) = kept {
            if let Some(v) = self.engines.lock().await.get_mut(engine) {
                v.cover.planning = false;
                v.cover.again = false;
            }
            warn!(
                "{engine}: cover plan {} not delivered: the store did not keep it: {e}",
                plan.generation
            );
            self.replan_later(engine).await;
            return false;
        }
        let Some(again) = self.make_delivered(engine, &plan, next).await else {
            return false;
        };
        info!(
            "{engine}: cover plan {} for epoch {}: {} groups",
            plan.generation,
            plan.epoch,
            plan.groups.len()
        );
        let to = self.daemons_of(engine).await;
        self.send_plan(engine, plan.clone(), to).await;
        self.watch_formation(engine, plan.generation);
        again
    }

    /// How long a planned group's members have to form it (`cover.formTimeoutMs`).
    fn form_timeout(&self) -> Duration {
        self.policy
            .cover
            .as_ref()
            .map(|c| c.form_timeout_ms)
            .filter(|ms| *ms > 0)
            .map_or(DEFAULT_FORM_TIMEOUT, |ms| {
                Duration::from_millis(u64::from(ms))
            })
    }

    /// Once `engine`'s plan `generation` has been out for the formation timeout and its transit,
    /// count every group of it not yet formed by all its members as failed.
    fn watch_formation(&self, engine: &str, generation: u64) {
        let Some(me) = self.me.upgrade() else { return };
        let engine = engine.to_owned();
        let after = self.form_timeout() + FORM_TRANSIT;
        tokio::spawn(async move {
            tokio::time::sleep(after).await;
            me.formation_due(&engine, generation).await;
        });
    }

    /// `engine`'s plan `generation` reached its formation deadline. A group some member has not
    /// reported formed is failed, as if a member had said so: the next plan leaves it out, so
    /// every member lets it go (one whose formation reported success without its peers holds a
    /// dead communicator otherwise), and the retry names it again. Nothing is counted when a
    /// later plan or a change has come since: during a change members only record plans, and
    /// the plan after its commit has a deadline of its own.
    async fn formation_due(&self, engine: &str, generation: u64) {
        let timeout = self.form_timeout();
        let retry_in = {
            let mut engines = self.engines.lock().await;
            let Some(view) = engines.get_mut(engine) else {
                return;
            };
            let current = view
                .cover
                .plan
                .as_ref()
                .is_some_and(|p| p.generation == generation && p.epoch == view.committed_epoch);
            if !current || view.change.is_some() {
                return;
            }
            // the groups the engine serves on are not the plan's to form: their members keep them
            let serving: BTreeSet<GroupKey> = view
                .cover
                .sets
                .iter()
                .filter(|s| s.purpose == "current")
                .flat_map(|s| s.groups.iter())
                .filter_map(|&i| view.cover.plan.as_ref()?.groups.get(i))
                .map(GroupKey::of)
                .collect();
            let unformed: Vec<(GroupKey, Member)> = view
                .cover
                .plan
                .iter()
                .flat_map(|p| p.groups.iter().map(GroupKey::of))
                .filter(|k| {
                    !serving.contains(k)
                        && !view.cover.group_covered(k)
                        && !view.cover.failed.contains_key(k)
                })
                .filter_map(|k| {
                    let silent = k
                        .spans
                        .iter()
                        .find(|m| !view.cover.formed.get(&k).is_some_and(|f| f.contains(m)))?
                        .clone();
                    Some((k, silent))
                })
                .collect();
            if unformed.is_empty() {
                return;
            }
            let mut retry_in = None;
            for (key, member) in unformed {
                let why = format!("not formed by every member within {timeout:?}");
                retry_in = failed(view, key, member, why).or(retry_in);
            }
            retry_in
        };
        warn!("{engine}: cover plan {generation}: groups not formed within {timeout:?}: failed");
        if let Some(wait) = retry_in
            && let Some(me) = self.me.upgrade()
        {
            let engine = engine.to_owned();
            tokio::spawn(async move {
                tokio::time::sleep(wait).await;
                me.observed(&engine).await;
            });
        }
        // the plan without them now, so their members let them go before the retry
        self.observed(engine).await;
    }

    /// Make `next`, which the store now holds, `engine`'s delivered plan, with its sets; whether
    /// another plan was asked for meanwhile, or None when a change claimed since, whose commit
    /// plans again.
    async fn make_delivered(&self, engine: &str, plan: &CoverPlan, next: NextPlan) -> Option<bool> {
        let mut engines = self.engines.lock().await;
        let view = engines.get_mut(engine)?;
        view.cover.planning = false;
        let again = std::mem::take(&mut view.cover.again);
        if view.change.is_some() {
            return None;
        }
        supersede(&mut view.cover, plan);
        view.cover.sets = next.sets;
        view.cover.uncovered = next.uncovered;
        if next.retry
            && let Some((n, due)) = view.cover.retry
        {
            view.cover.retry = Some((n + 1, due));
            self.metrics
                .control
                .cover_retries_total
                .with_label_values(&[engine])
                .inc();
        }
        view.status.record(
            now_ms(),
            Severity::Normal,
            "PlanDelivered",
            format!(
                "cover plan {} for epoch {}: {} sets, {} groups; {} losses uncovered",
                plan.generation,
                plan.epoch,
                view.cover.sets.len(),
                plan.groups.len(),
                view.cover.uncovered.len()
            ),
        );
        Some(again)
    }

    /// Send `plan` to the daemons `to`, and note that each holds it.
    async fn send_plan(&self, engine: &str, plan: CoverPlan, to: BTreeMap<String, Outbox>) {
        let generation = plan.generation;
        let msg = CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::CoverPlan(plan)),
        };
        let mut sent = Vec::new();
        for (node, tx) in to {
            if tx.send(msg.clone()).is_ok() {
                sent.push(node);
            }
        }
        if let Some(view) = self.engines.lock().await.get_mut(engine) {
            for node in sent {
                let at = view.cover.reached.entry(node).or_default();
                *at = (*at).max(generation);
            }
        }
    }

    /// Wake `engine` to plan again after [`UNWRITTEN_RETRY`], unless a wake already waits.
    async fn replan_later(&self, engine: &str) {
        let wake = self
            .engines
            .lock()
            .await
            .get_mut(engine)
            .is_some_and(|v| !std::mem::replace(&mut v.cover.unwritten, true));
        if let Some(me) = self.me.upgrade().filter(|_| wake) {
            tokio::spawn(wake_later(me, engine.to_owned()));
        }
    }

    /// The plan for `view` now, its generation taken; None when it names the same groups as the
    /// plan already delivered for this epoch. It is made the delivered one, with its sets, only
    /// once the store holds it.
    fn plan_for(
        &self,
        engine: &str,
        spec: &crate::proto::everserve::policy::v1::Engine,
        view: &mut EngineView,
        world: u32,
        around: &Around<'_>,
    ) -> Option<NextPlan> {
        let pp = spec.pipeline_parallel.max(1);
        let spares = standing_spares(view, around.cordoned);
        let outcomes: Vec<Outcome> = self
            .policy
            .outcomes_in_order
            .iter()
            .filter_map(|o| Outcome::try_from(*o).ok())
            .collect();
        let placement = placement_of(view);
        let planned = policy::cover::plan(&Input {
            layout: Layout {
                tensor_parallel: world / pp,
                pipeline_parallel: pp,
            },
            placement: &view.members,
            spares: &spares,
            replace: outcomes.contains(&Outcome::Replace),
            shrink: outcomes.contains(&Outcome::Shrink) && spec.placement_independent_model,
            budget: self
                .policy
                .cover
                .as_ref()
                .map(|c| c.max_groups_per_gpu)
                .filter(|b| *b > 0),
            elsewhere: around.elsewhere,
        });
        // a failed group is retried by delivering the plan again once its backoff is due; the
        // process never retries on its own (decision 0035)
        let retry = !view.cover.failed.is_empty()
            && view
                .cover
                .retry
                .is_some_and(|(_, due)| due <= Instant::now());
        // until a failed group's retry is due, plans leave it out, so every member lets go of
        // what it holds of it and the retry forms it afresh: a member that kept a group named
        // again would keep one whose formation never completed
        let waiting = !view.cover.failed.is_empty() && !retry;
        let planned = if waiting {
            let dropped: BTreeSet<GroupKey> = view.cover.failed.keys().cloned().collect();
            policy::cover::without(planned, &dropped)
        } else {
            planned
        };
        let unchanged = !retry
            && view.cover.plan.as_ref().is_some_and(|p| {
                p.epoch == view.committed_epoch
                    && p.groups == planned.groups
                    && p.sets == wire_sets(&planned.sets)
            });
        if unchanged {
            view.cover.sets = planned.sets;
            view.cover.uncovered = planned.uncovered;
            return None;
        }
        let plan = CoverPlan {
            engine: engine.to_owned(),
            generation: view.cover.generation.max(view.cover.reserved) + 1,
            epoch: view.committed_epoch,
            placement: placement.into_iter().collect(),
            sets: wire_sets(&planned.sets),
            groups: planned.groups,
        };
        // taken now, so no other plan is numbered the same, delivered or not
        view.cover.reserved = plan.generation;
        Some(NextPlan {
            plan,
            sets: planned.sets,
            uncovered: planned.uncovered,
            retry,
        })
    }

    /// A process's report of a group, forwarded by `node`'s daemon: counted against the plan
    /// delivered for its engine when it is at that plan's generation; then what is derived from
    /// the engine's status (its Covered condition) is derived again.
    pub(super) async fn cover_report(&self, node: &str, r: CoverReport) {
        let ready = match &r.kind {
            Some(cover_report::Kind::Ready(g)) => Some(GroupKey::from_parts(g.scope, &g.spans)),
            _ => None,
        };
        let Some((engine, retry_in, covered)) = self.count_report(node, r).await else {
            return;
        };
        if let Some(key) = ready {
            self.relay_if_ready(&engine, &key).await;
            return;
        }
        if covered {
            // a drain refused for want of this set may go now
            self.retry_blocked_drains().await;
        }
        if let Some(wait) = retry_in
            && let Some(me) = self.me.upgrade()
        {
            // the retry is a plan delivered again when it is due: wake the engine then
            let engine = engine.clone();
            tokio::spawn(async move {
                tokio::time::sleep(wait).await;
                me.observed(&engine).await;
            });
        }
        self.observed(&engine).await;
    }

    /// Count `r` against its engine's plan: the engine, when the report was for one it knows,
    /// how long until a retry when the report scheduled one, and whether it completed a set.
    async fn count_report(
        &self,
        node: &str,
        r: CoverReport,
    ) -> Option<(String, Option<Duration>, bool)> {
        let Some(kind) = r.kind else {
            warn!("{node}: an empty cover report");
            return None;
        };
        let engine = fields(&kind).0.to_owned();
        let mut engines = self.engines.lock().await;
        let view = engines.get_mut(&engine)?;
        let (retry_in, covered) = count(view, node, kind)?;
        Some((engine, retry_in, covered))
    }

    /// `view`'s cover as readiness shows it, or None when no plan was delivered.
    pub(super) fn cover_readiness(view: &EngineView) -> Option<CoverReadiness> {
        let cover = &view.cover;
        let plan = cover.plan.as_ref()?;
        Some(CoverReadiness {
            generation: plan.generation,
            epoch: plan.epoch,
            sets: cover
                .sets
                .iter()
                .map(|s| {
                    let groups: Vec<GroupKey> = s
                        .groups
                        .iter()
                        .filter_map(|&i| plan.groups.get(i))
                        .map(GroupKey::of)
                        .collect();
                    CoverSetReadiness {
                        set: set_key(&s.set),
                        purpose: s.purpose.clone(),
                        covered: cover.set_covered(&s.set),
                        groups: groups.len(),
                        formed: groups.iter().filter(|k| cover.group_covered(k)).count(),
                    }
                })
                .collect(),
            uncovered: cover
                .uncovered
                .iter()
                .map(|u| UncoveredReadiness {
                    lost: u.lost.clone(),
                    why: u.why.clone(),
                })
                .collect(),
            failed: cover
                .failed
                .iter()
                .map(|(k, (m, e))| FailedGroupReadiness {
                    group: set_key(&k.spans),
                    scope: scope_text(k.scope).to_owned(),
                    member: m.to_string(),
                    error: e.clone(),
                })
                .collect(),
        })
    }
}

/// The engine, plan generation, group (the members it spans), scope and member a report names.
fn fields(kind: &cover_report::Kind) -> (&str, u64, &[SetMember], i32, Option<&SetMember>) {
    match kind {
        cover_report::Kind::Formed(f) => (
            &f.engine,
            f.generation,
            &f.spans,
            f.scope,
            f.member.as_ref(),
        ),
        cover_report::Kind::Failed(f) => (
            &f.engine,
            f.generation,
            &f.spans,
            f.scope,
            f.member.as_ref(),
        ),
        cover_report::Kind::Released(f) => (
            &f.engine,
            f.generation,
            &f.spans,
            f.scope,
            f.member.as_ref(),
        ),
        cover_report::Kind::Ready(f) => (
            &f.engine,
            f.generation,
            &f.spans,
            f.scope,
            f.member.as_ref(),
        ),
    }
}

/// Count `kind`, a report `node`'s daemon forwarded, against `view`'s delivered plan: how long
/// until a retry when the report scheduled one, and whether it completed a set; None when it
/// counts toward nothing.
fn count(
    view: &mut EngineView,
    node: &str,
    kind: cover_report::Kind,
) -> Option<(Option<Duration>, bool)> {
    let (engine, generation, spans, scope, member) = fields(&kind);
    let (engine, spans) = (engine.to_owned(), spans.to_vec());
    let Some(member) = member.and_then(Member::from_wire) else {
        warn!("{node}: a cover report naming no member");
        return None;
    };
    view.cover.generation = view.cover.generation.max(generation);
    if let cover_report::Kind::Released(_) = kind {
        release(&mut view.cover, &spans, scope, &member);
        return Some((None, false));
    }
    let current = view.cover.plan.as_ref().map(|p| p.generation);
    let Some(key) = view.cover.group_of(&spans, scope, &member) else {
        info!(
            "{engine}: {member} on {node} reported {} ({}) under plan {generation}, which \
             names no such group: counted toward nothing",
            key_text(&spans),
            scope_text(scope)
        );
        return None;
    };
    if current != Some(generation) {
        return None; // another plan's report: the process reports again at the current one
    }
    // a commit moved the epoch past the plan: its sets name positions the commit changed,
    // and one the change went through now serves. A report under it, a group that could not
    // form again on its serving port among them, counts toward nothing; the next plan is
    // for the new epoch; counted, a clean REPLACE would read as a failed group
    if view
        .cover
        .plan
        .as_ref()
        .is_some_and(|p| p.epoch < view.committed_epoch)
    {
        info!(
            "{engine}: {member} on {node} reported {} ({}) under plan {generation}, which a \
             commit at epoch {} superseded: counted toward nothing",
            key_text(&spans),
            scope_text(scope),
            view.committed_epoch
        );
        return None;
    }
    debug!(
        "{engine}: {member} on {node} counted for {} ({}) under plan {generation}",
        set_key(&key.spans),
        scope_text(scope)
    );
    let (retry_in, covered) = match kind {
        cover_report::Kind::Formed(_) => (None, formed(view, key, member, generation)),
        cover_report::Kind::Failed(f) => {
            warn!(
                "{engine}: {member} on {node} could not form {} ({}) under plan \
                 {generation}: {}",
                set_key(&key.spans),
                scope_text(scope),
                f.error
            );
            (failed(view, key, member, f.error), false)
        }
        cover_report::Kind::Released(_) => (None, false),
        cover_report::Kind::Ready(_) => {
            view.cover.ready.entry(key).or_default().insert(member);
            (None, false)
        }
    };
    Some((retry_in, covered))
}

/// `member` reported `key`'s group formed at the plan's `generation`: counted, and each set it
/// completes is an event (what a change through the set waits for); a group shared by several
/// sets may complete several. A plan fully covered starts the retry backoff over. Whether the
/// report completed a set.
fn formed(view: &mut EngineView, key: GroupKey, member: Member, generation: u64) -> bool {
    view.cover.failed.remove(&key);
    let sets = view.cover.sets.clone();
    let before: Vec<bool> = sets
        .iter()
        .map(|s| view.cover.set_covered(&s.set))
        .collect();
    view.cover.formed.entry(key).or_default().insert(member);
    let mut completed = false;
    for (s, was) in sets.iter().zip(before) {
        if !was && view.cover.set_covered(&s.set) {
            completed = true;
            view.status.record(
                now_ms(),
                Severity::Normal,
                "SetCovered",
                format!(
                    "set {} ({}) covered under plan {generation}",
                    set_key(&s.set),
                    s.purpose
                ),
            );
        }
    }
    if sets.iter().all(|s| view.cover.set_covered(&s.set)) {
        view.cover.retry = None;
    }
    completed
}

/// `member` could not form `key`'s group: recorded, an event, and a retry scheduled unless one
/// is already due later; how long until it, when this report scheduled it.
fn failed(view: &mut EngineView, key: GroupKey, member: Member, error: String) -> Option<Duration> {
    view.status.record(
        now_ms(),
        Severity::Warning,
        "GroupFailed",
        format!(
            "{member} could not form {} ({}): {error}",
            set_key(&key.spans),
            scope_text(key.scope)
        ),
    );
    view.cover.failed.insert(key, (member, error));
    let attempts = view.cover.retry.map_or(0, |(n, _)| n);
    if view
        .cover
        .retry
        .is_none_or(|(_, due)| due <= Instant::now())
    {
        let wait = backoff(attempts);
        view.cover.retry = Some((attempts, Instant::now() + wait));
        return Some(wait);
    }
    None
}

/// What `node`'s hello says its processes hold for an engine, and the highest generation its
/// daemon saw: the next plan is numbered above it, and the groups stand as cover until the
/// plan's own reports replace them. A plan newer than this coordinator's is adopted, with its
/// sets: the processes hold it, and a restart must change nothing they hold.
pub(super) fn merge_held(view: &mut EngineView, node: &str, state: &EngineState) {
    let members = view.members.clone();
    let cover = &mut view.cover;
    cover.generation = cover.generation.max(state.plan_generation);
    cover.reached.insert(node.to_owned(), state.plan_generation);
    if let Some(plan) = state.plan.as_ref() {
        adopt(cover, plan, &format!("{node}'s daemon"));
    }
    for holders in cover.held.values_mut() {
        holders.retain(|m| !on_node(m, node, &members));
    }
    for h in &state.held {
        let Some(member) = h.member.as_ref().and_then(Member::from_wire) else {
            continue;
        };
        // the hello names the group by its scope and spans, as the plan does
        let key = GroupKey::from_parts(h.scope, &h.spans);
        cover.held.entry(key).or_default().insert(member);
    }
    cover.held.retain(|_, holders| !holders.is_empty());
}

/// The plan `view` delivered and the daemons of `targets` behind it, if any are.
fn behind(
    view: &EngineView,
    targets: BTreeMap<String, Outbox>,
) -> Option<(CoverPlan, BTreeMap<String, Outbox>)> {
    let plan = view.cover.plan.clone()?;
    let to: BTreeMap<_, _> = targets
        .into_iter()
        .filter(|(n, _)| view.cover.reached.get(n).copied().unwrap_or(0) < plan.generation)
        .collect();
    (!to.is_empty()).then_some((plan, to))
}

/// Wake `engine` after [`UNWRITTEN_RETRY`] to plan again. Its own function, typed as `Send`: the
/// wake plans, and planning may ask for another wake.
fn wake_later(
    me: std::sync::Arc<Coordinator>,
    engine: String,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(async move {
        tokio::time::sleep(UNWRITTEN_RETRY).await;
        if let Some(v) = me.engines.lock().await.get_mut(&engine) {
            v.cover.unwritten = false;
        }
        me.observed(&engine).await;
    })
}

/// Take `plan` as the delivered one, with its sets, when it is newer than the one `cover` has:
/// a restarted coordinator adopts the newest plan of the store's and the daemons' (decisions
/// 0035, 0038), and sends it to every daemon behind it.
pub(super) fn adopt(cover: &mut CoverState, plan: &CoverPlan, from: &str) {
    cover.generation = cover.generation.max(plan.generation);
    if cover
        .plan
        .as_ref()
        .is_some_and(|p| p.generation >= plan.generation)
    {
        return;
    }
    info!(
        "{}: adopted cover plan {} for epoch {} from {from}",
        plan.engine, plan.generation, plan.epoch
    );
    cover.sets = sets_of(plan);
    cover.uncovered.clear();
    cover.formed.clear();
    cover.plan = Some(plan.clone());
}

/// `node` is gone: what its processes held is gone with them.
pub(super) fn forget_node_cover(view: &mut EngineView, node: &str) {
    let members = view.members.clone();
    let cover = &mut view.cover;
    cover.reached.remove(node);
    for holders in cover.formed.values_mut().chain(cover.held.values_mut()) {
        holders.retain(|m| !on_node(m, node, &members));
    }
}

/// The nodes a restarted coordinator waits for before it plans `spec`'s cover: those its policy
/// places members and spares on, and those its current plan names.
pub(super) fn expected_nodes(
    spec: &crate::proto::everserve::policy::v1::Engine,
    view: &EngineView,
) -> BTreeSet<String> {
    // a pinned engine's nodes, from its policy; otherwise, its holders' as the plan placed them
    let mut nodes: BTreeSet<String> = spec.members.iter().map(|m| m.node.clone()).collect();
    nodes.extend(spec.spare_nodes.iter().cloned());
    nodes.extend(spec.spares.iter().map(|s| s.node.clone()));
    if let Some(plan) = &view.cover.plan {
        nodes.extend(plan.placement.values().map(|h| h.node.clone()));
        for g in &plan.groups {
            for m in g.spans.iter().filter_map(Member::from_wire) {
                if let Member::Spare { node, .. } = m {
                    nodes.insert(node);
                }
            }
        }
    }
    nodes.retain(|n| !n.is_empty());
    nodes
}

/// The sets an adopted `plan` names, in plan order, each with what it is for and the groups it
/// needs, as the plan carries them.
fn sets_of(plan: &CoverPlan) -> Vec<PlannedSet> {
    plan.sets
        .iter()
        .map(|s| PlannedSet {
            set: s.members.iter().filter_map(Member::from_wire).collect(),
            purpose: s.purpose.clone(),
            groups: s
                .groups
                .iter()
                .map(|&i| usize::try_from(i).unwrap_or(usize::MAX))
                .collect(),
        })
        .collect()
}

/// Every loss of `view`'s placement that no covered set answers: what the Covered condition
/// reports while cover is planned. The head's node's ranks are left out: their loss restarts the
/// engine (decision 0032), which no set answers.
pub(super) fn losses_without_cover(view: &EngineView) -> Vec<Vec<u32>> {
    let heads = head_node_loss(view);
    losses_of(&view.members)
        .into_iter()
        .filter(|l| {
            !heads
                .as_ref()
                .is_some_and(|h| l.iter().all(|r| h.contains(r)))
        })
        .filter(|l| view.cover.covered_answer(l).is_none())
        .collect()
}

/// The ranks on the node the engine's head runs on, when it has one. A loss of any of them is the
/// head's, which no spare answers (decision 0032): the head's engine core steps every rank on its
/// node, and dies with any of them, so the engine restarts. It is no question of cover.
pub(super) fn head_node_loss(view: &EngineView) -> Option<Vec<u32>> {
    let node = &view.head.as_ref()?.process.node;
    let ranks: Vec<u32> = view
        .members
        .iter()
        .filter(|(_, n)| *n == node)
        .map(|(r, _)| *r)
        .collect();
    (!ranks.is_empty()).then_some(ranks)
}

/// What an engine's plan is made within, beside its own view: the cordoned nodes and the groups
/// its spares hold in other engines' plans.
struct Around<'a> {
    cordoned: &'a BTreeSet<String>,
    elsewhere: &'a BTreeMap<(String, u32), u32>,
}

/// The groups each spare process holds in the plans of every engine but `engine`, by node and
/// device: a pool's spare stands for all its engines at once (decision 0031). Engines plan in
/// turn, so one planned earlier holds what its plan took, and the next is planned within what is
/// left.
fn held_elsewhere(
    engines: &BTreeMap<String, EngineView>,
    engine: &str,
) -> BTreeMap<(String, u32), u32> {
    let mut out: BTreeMap<(String, u32), u32> = BTreeMap::new();
    for (_, v) in engines.iter().filter(|(name, _)| name.as_str() != engine) {
        let Some(plan) = v.cover.plan.as_ref() else {
            continue;
        };
        for g in &plan.groups {
            for m in g.spans.iter().filter_map(Member::from_wire) {
                if let Member::Spare { node, device, .. } = m {
                    *out.entry((node, device)).or_default() += 1;
                }
            }
        }
    }
    out
}

/// The spares of `view` standing in its pool on nodes that are not cordoned, as the planner
/// takes them: each with its device, from its registration.
fn standing_spares(view: &EngineView, cordoned: &BTreeSet<String>) -> Vec<StandingSpare> {
    view.standing
        .iter()
        .filter(|((node, _), _)| !cordoned.contains(node))
        .filter(|((node, device), ev)| {
            if ev.any_rank {
                view.any_spares.contains(&(node.clone(), *device))
            } else {
                view.spares.get(&ev.rank) == Some(node)
            }
        })
        .map(|((node, device), ev)| StandingSpare {
            node: node.clone(),
            device: *device,
            for_rank: (!ev.any_rank).then_some(ev.rank),
            host: ev.host.clone(),
        })
        .collect()
}

/// Make `plan` the delivered one: coverage starts again from the reports at the new generation.
fn supersede(cover: &mut CoverState, plan: &CoverPlan) {
    // the first plan after a coordinator starts is at the epoch the hellos reported, and their
    // held groups stand until its reports replace them; only a commit moves the epoch
    let epoch_moved = cover.plan.as_ref().is_some_and(|p| p.epoch != plan.epoch);
    // a group the plan names again over the same processes keeps its communicator and its id;
    // any other group is formed anew, under a new id from its server
    let named: BTreeSet<GroupKey> = plan.groups.iter().map(GroupKey::of).collect();
    if let Some(old) = cover.plan.as_ref() {
        cover
            .ids
            .retain(|k, _| named.contains(k) && same_processes(k, &old.placement, &plan.placement));
    }
    // a plan for another epoch names positions some other processes hold: what the hellos said
    // was held under the old placement stands as cover no longer (a group naming a lost rank
    // meant the process that left)
    if epoch_moved {
        cover.held.clear();
    }
    cover.formed.clear();
    cover.ready.clear();
    // a failed group the plan names again is formed afresh; one it leaves out waits for its
    // retry, still failed, so plans keep leaving it out until the backoff is due
    cover.failed.retain(|k, _| !named.contains(k));
    cover.generation = plan.generation;
    cover.plan = Some(plan.clone());
}

/// `member` released the group over `spans` in `scope`: it holds it no longer.
fn release(cover: &mut CoverState, spans: &[SetMember], scope: i32, member: &Member) {
    let key = GroupKey::from_parts(scope, spans);
    let matches = |k: &GroupKey| *k == key && k.spans.contains(member);
    for (k, holders) in cover.formed.iter_mut().chain(cover.held.iter_mut()) {
        if matches(k) {
            holders.remove(member);
        }
    }
}

/// The placement `view` holds now: each rank's holding process by node and device.
fn placement_of(view: &EngineView) -> BTreeMap<u32, Holder> {
    view.members
        .iter()
        .map(|(r, node)| {
            let holder = Holder {
                node: node.clone(),
                device: view.devices.get(r).copied().unwrap_or_default(),
            };
            (*r, holder)
        })
        .collect()
}

/// Whether `key`'s group spans the same processes under `new` as under `old`: every rank it
/// spans held by the same node and device. A spare is named by its own node and device already.
fn same_processes(
    key: &GroupKey,
    old: &std::collections::HashMap<u32, Holder>,
    new: &impl PlacementLookup,
) -> bool {
    key.spans.iter().all(|m| match m {
        Member::Rank(r) => old.get(r).is_some_and(|h| new.holder(*r) == Some(h)),
        Member::Spare { .. } => true,
    })
}

/// A placement, looked up by rank: the plan's map on the wire, or the one being planned.
trait PlacementLookup {
    fn holder(&self, rank: u32) -> Option<&Holder>;
}

impl PlacementLookup for std::collections::HashMap<u32, Holder> {
    fn holder(&self, rank: u32) -> Option<&Holder> {
        self.get(&rank)
    }
}

impl PlacementLookup for BTreeMap<u32, Holder> {
    fn holder(&self, rank: u32) -> Option<&Holder> {
        self.get(&rank)
    }
}

/// Whether `member` runs on `node`: a rank by its holder's node, a spare by its own.
fn on_node(member: &Member, node: &str, members: &BTreeMap<u32, String>) -> bool {
    match member {
        Member::Rank(r) => members.get(r).is_some_and(|n| n == node),
        Member::Spare { node: n, .. } => n == node,
    }
}

fn key_text(set: &[SetMember]) -> String {
    set_key(&set.iter().filter_map(Member::from_wire).collect::<Vec<_>>())
}

fn scope_text(s: i32) -> &'static str {
    match Scope::try_from(s) {
        Ok(Scope::Domain) => "domain",
        Ok(Scope::Fabric) => "fabric",
        _ => "no scope",
    }
}

/// The node `member` runs on, by `members` for a serving rank.
fn node_of(member: &Member, members: &BTreeMap<u32, String>) -> Option<String> {
    match member {
        Member::Rank(r) => members.get(r).cloned(),
        Member::Spare { node, .. } => Some(node.clone()),
    }
}

impl Coordinator {
    /// `node`'s daemon passed on `id`, a group's id from its server member: kept as the group's
    /// newest, and sent to the daemons of the group's other members (decision 0035, amended). An
    /// id for a group the delivered plan does not name, or from a node other than its server's,
    /// is refused: a group's id comes from its server alone.
    pub(super) async fn group_id(&self, node: &str, id: GroupId) {
        let key = GroupKey::from_parts(id.scope, &id.spans);
        let relay = {
            let mut engines = self.engines.lock().await;
            let Some(view) = engines.get_mut(&id.engine) else {
                return;
            };
            let server = view.cover.plan.as_ref().and_then(|p| {
                p.groups
                    .iter()
                    .find(|g| GroupKey::of(g) == key)
                    .and_then(|g| g.server.as_ref())
                    .and_then(Member::from_wire)
            });
            let Some(server) = server else {
                warn!(
                    "{}: an id for {} ({}), which plan {} names no group of: refused",
                    id.engine,
                    set_key(&key.spans),
                    scope_text(key.scope),
                    id.generation
                );
                return;
            };
            if node_of(&server, &view.members).as_deref() != Some(node) {
                warn!(
                    "{}: an id for {} ({}) from {node}, which is not its server's node: refused",
                    id.engine,
                    set_key(&key.spans),
                    scope_text(key.scope)
                );
                return;
            }
            if view
                .cover
                .ids
                .get(&key)
                .is_some_and(|kept| kept.generation > id.generation)
            {
                return;
            }
            view.cover.ids.insert(key.clone(), id.clone());
            // held until every member is ready, the server included
            relay_of(view, &key)
        };
        if let Some((id, to)) = relay {
            self.relay(id, &to).await;
        }
    }

    /// Relay `key`'s id of `engine` to every member, once every member reported it ready at the
    /// plan's generation and the server's id for that generation is here: each member's go.
    async fn relay_if_ready(&self, engine: &str, key: &GroupKey) {
        let relay = self
            .engines
            .lock()
            .await
            .get(engine)
            .and_then(|view| relay_of(view, key));
        if let Some((id, to)) = relay {
            self.relay(id, &to).await;
        }
    }

    /// Send `id` to the daemons of `to`.
    async fn relay(&self, id: GroupId, to: &BTreeSet<String>) {
        let msg = CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::GroupId(id)),
        };
        let links: Vec<_> = {
            let daemons = self.daemons.lock().await;
            to.iter()
                .filter_map(|n| daemons.get(n).map(|d| d.tx.clone()))
                .collect()
        };
        for tx in links {
            let _ = tx.send(msg.clone());
        }
    }

    /// Give `node`'s daemon, newly linked, the id of every group a process on it is a member of.
    pub(super) async fn send_group_ids(&self, node: &str) {
        let Some(tx) = self.daemons.lock().await.get(node).map(|d| d.tx.clone()) else {
            return;
        };
        let ids: Vec<GroupId> = self
            .engines
            .lock()
            .await
            .values()
            .flat_map(|view| ids_for_node(view, node))
            .collect();
        for id in ids {
            let msg = CoordToDaemon {
                term: 0,
                kind: Some(coord_to_daemon::Kind::GroupId(id)),
            };
            if tx.send(msg).is_err() {
                return;
            }
        }
    }
}

/// The ids `view` may give a daemon of `node` as it links: of the groups a process on it is a
/// member of, each whose members are all ready. A relink is no way round the gate.
fn ids_for_node(view: &EngineView, node: &str) -> Vec<GroupId> {
    view.cover
        .ids
        .keys()
        .filter(|k| k.spans.iter().any(|m| on_node(m, node, &view.members)))
        .filter_map(|k| relay_of(view, k).map(|(id, _)| id))
        .collect()
}

/// `key`'s id and the nodes of every member, the server included, when the id may go: the
/// server's id is here for the delivered plan's generation, and every member reported ready at
/// it. None while the group waits.
fn relay_of(view: &EngineView, key: &GroupKey) -> Option<(GroupId, BTreeSet<String>)> {
    let plan = view.cover.plan.as_ref()?;
    let id = view.cover.ids.get(key)?;
    if id.generation != plan.generation {
        return None;
    }
    let ready = view.cover.ready.get(key)?;
    if !key.spans.iter().all(|m| ready.contains(m)) {
        return None;
    }
    let to = key
        .spans
        .iter()
        .filter_map(|m| node_of(m, &view.members))
        .collect();
    Some((id.clone(), to))
}

/// The plan's sets as the wire carries them: each with the groups it needs by index.
fn wire_sets(sets: &[PlannedSet]) -> Vec<CoverSet> {
    sets.iter()
        .map(|s| CoverSet {
            members: s.set.iter().map(Member::to_wire).collect(),
            groups: s
                .groups
                .iter()
                .map(|&i| u32::try_from(i).unwrap_or(u32::MAX))
                .collect(),
            purpose: s.purpose.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{CoverState, count, supersede};
    use crate::coord::view::EngineView;
    use crate::policy::cover::{GroupKey, Member};
    use crate::proto::everserve::coord::v1::cover_report;
    use crate::proto::everserve::signals::v1::{
        CoverGroup, CoverPlan, GroupFailed, Scope, SetMember, set_member,
    };

    fn plan(generation: u64, epoch: u64) -> CoverPlan {
        CoverPlan {
            engine: "k3-a".into(),
            generation,
            epoch,
            ..CoverPlan::default()
        }
    }

    fn held() -> CoverState {
        let key = GroupKey {
            scope: Scope::Domain as i32,
            spans: vec![Member::Rank(0), Member::Rank(1)],
        };
        let mut c = CoverState::default();
        c.held
            .insert(key, BTreeSet::from([Member::Rank(0), Member::Rank(1)]));
        c
    }

    /// A view whose delivered plan, generation 5 at epoch 0, has one group of ranks 0 and 1;
    /// `committed` is the view's committed epoch.
    fn delivered(committed: u64) -> EngineView {
        let pair: Vec<SetMember> = (0..2)
            .map(|r| SetMember {
                kind: Some(set_member::Kind::Rank(r)),
            })
            .collect();
        let mut view = EngineView {
            committed_epoch: committed,
            ..EngineView::default()
        };
        view.cover.plan = Some(CoverPlan {
            groups: vec![CoverGroup {
                scope: Scope::Domain as i32,
                spans: pair,
                ..CoverGroup::default()
            }],
            ..plan(5, 0)
        });
        view
    }

    fn failure() -> cover_report::Kind {
        cover_report::Kind::Failed(GroupFailed {
            engine: "k3-a".into(),
            generation: 5,
            spans: delivered(0).cover.plan.expect("plan").groups[0]
                .spans
                .clone(),
            scope: Scope::Domain as i32,
            error: "Address already in use".into(),
            member: Some(SetMember {
                kind: Some(set_member::Kind::Rank(0)),
            }),
        })
    }

    #[test]
    fn a_failure_under_a_plan_a_commit_superseded_counts_toward_nothing() {
        // under the current epoch's plan, a failure is the group's: recorded, a retry scheduled
        let mut view = delivered(0);
        assert!(count(&mut view, "node-a", failure()).is_some());
        assert_eq!(view.cover.failed.len(), 1);
        // the plan made at epoch 0 is still the delivered one when the REPLACE commits epoch 1,
        // and rank 0 cannot form a group of it again on the port the change made serving. The
        // next plan is for epoch 1; this one's failure is nobody's
        let mut view = delivered(1);
        assert!(count(&mut view, "node-a", failure()).is_none());
        assert!(view.cover.failed.is_empty());
        assert!(view.cover.retry.is_none());
    }

    #[test]
    fn the_hellos_held_groups_stand_until_a_commit_moves_the_epoch() {
        // the first plan after a coordinator starts is at the hellos' epoch: they stand
        let mut c = held();
        supersede(&mut c, &plan(5, 3));
        assert_eq!(c.held.len(), 1);
        // another plan at that epoch: they still stand
        supersede(&mut c, &plan(6, 3));
        assert_eq!(c.held.len(), 1);
        // a commit moved the epoch: a held group naming a lost rank meant the process that left
        supersede(&mut c, &plan(7, 4));
        assert!(c.held.is_empty());
    }

    #[test]
    fn a_groups_id_goes_only_once_every_member_is_ready_and_a_relink_is_no_way_round() {
        use super::{ids_for_node, relay_of};
        use crate::proto::everserve::signals::v1::GroupId;
        let key = GroupKey {
            scope: Scope::Domain as i32,
            spans: vec![Member::Rank(0), Member::Rank(1)],
        };
        let mut view = EngineView::default();
        view.members.insert(0, "node-a".into());
        view.members.insert(1, "node-b".into());
        view.cover.plan = Some(plan(3, 0));
        let id = |generation| GroupId {
            engine: "k3-a".into(),
            generation,
            id: vec![7; 4],
            ..GroupId::default()
        };
        view.cover.ids.insert(key.clone(), id(3));
        // no member ready: held, and no relink is given it
        assert!(relay_of(&view, &key).is_none());
        assert!(ids_for_node(&view, "node-b").is_empty());
        // one ready: still held
        view.cover
            .ready
            .insert(key.clone(), BTreeSet::from([Member::Rank(1)]));
        assert!(relay_of(&view, &key).is_none());
        // every member ready: to both nodes, the server's included
        view.cover.ready.insert(
            key.clone(),
            BTreeSet::from([Member::Rank(0), Member::Rank(1)]),
        );
        let (_, to) = relay_of(&view, &key).expect("relayed");
        assert_eq!(
            to,
            BTreeSet::from(["node-a".to_owned(), "node-b".to_owned()])
        );
        assert_eq!(ids_for_node(&view, "node-b").len(), 1);
        // an id made under an earlier plan is not this plan's go
        view.cover.ids.insert(key.clone(), id(2));
        assert!(relay_of(&view, &key).is_none());
    }
}
