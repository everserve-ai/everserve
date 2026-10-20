// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The cover planner (decision 0035): which member sets an engine pre-forms groups for, and where
//! each group meets.
//!
//! A REPLACE or a SHRINK is a pointer swap to a group pre-formed for the new member set (0005).
//! The planner chooses those sets from the engine's placement and its standing spares, within a
//! budget of groups per GPU, in order: the current membership's set always, then each node's loss,
//! then each position's, then a second spare for a position's loss already covered. It is a pure
//! function of its input; the coordinator holds the state it reads (the placement, the spares,
//! the groups other engines' plans put on a spare) and delivers what it returns. A group meets on
//! the control plane's links (`GroupId`), so the planner hands out no address or port.

use std::collections::{BTreeMap, BTreeSet};

use crate::proto::everserve::signals::v1::{CoverGroup, Scope, SetMember, SpareRef, set_member};

/// An engine's tensor-parallel and pipeline-parallel widths, and the mapping they imply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Ranks in a stage.
    pub tensor_parallel: u32,
    /// Stages.
    pub pipeline_parallel: u32,
}

impl Layout {
    /// Ranks in the engine.
    #[must_use]
    pub fn world(self) -> u32 {
        self.tensor_parallel * self.pipeline_parallel
    }

    fn stage(self, position: u32) -> u32 {
        position / self.tensor_parallel.max(1)
    }

    fn column(self, position: u32) -> u32 {
        position % self.tensor_parallel.max(1)
    }
}

/// A member of a set: a serving rank, or a spare standing in for a position.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Member {
    /// The process serving the rank.
    Rank(u32),
    /// A spare, standing in for `for_rank`, on `node`'s device `device`.
    Spare {
        /// The position it stands in for in this set.
        for_rank: u32,
        /// Its node.
        node: String,
        /// Its device there.
        device: u32,
    },
}

impl Member {
    /// The rank position the member holds in a set.
    #[must_use]
    pub fn position(&self) -> u32 {
        match self {
            Self::Rank(r) | Self::Spare { for_rank: r, .. } => *r,
        }
    }

    /// The process this member is: a spare counted once whatever position it stands for.
    fn process(&self) -> Process {
        match self {
            Self::Rank(r) => Process::Rank(*r),
            Self::Spare { node, device, .. } => Process::Spare(node.clone(), *device),
        }
    }

    /// The wire form.
    #[must_use]
    pub fn to_wire(&self) -> SetMember {
        SetMember {
            kind: Some(match self {
                Self::Rank(r) => set_member::Kind::Rank(*r),
                Self::Spare {
                    for_rank,
                    node,
                    device,
                } => set_member::Kind::Spare(SpareRef {
                    for_rank: *for_rank,
                    node: node.clone(),
                    device: *device,
                }),
            }),
        }
    }

    /// From the wire form; None for an empty member.
    #[must_use]
    pub fn from_wire(m: &SetMember) -> Option<Self> {
        Some(match m.kind.as_ref()? {
            set_member::Kind::Rank(r) => Self::Rank(*r),
            set_member::Kind::Spare(s) => Self::Spare {
                for_rank: s.for_rank,
                node: s.node.clone(),
                device: s.device,
            },
        })
    }
}

impl std::fmt::Display for Member {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rank(r) => write!(f, "{r}"),
            Self::Spare {
                for_rank,
                node,
                device,
            } => write!(f, "R{for_rank}@{node}:{device}"),
        }
    }
}

/// A set's name, as logs and readiness show it: its members joined by `-`.
#[must_use]
pub fn set_key(set: &[Member]) -> String {
    set.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("-")
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Process {
    Rank(u32),
    Spare(String, u32),
}

/// A spare standing for the engine, admitted to its pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandingSpare {
    /// Its node.
    pub node: String,
    /// Its device there.
    pub device: u32,
    /// The rank it was launched for; None for one standing for any position (decision 0030).
    pub for_rank: Option<u32>,
    /// The address its process registered, which its peers reach it at (a pod's IP); empty for
    /// none.
    pub host: String,
}

/// What the planner plans from.
#[derive(Debug, Clone)]
pub struct Input<'a> {
    /// The engine's widths.
    pub layout: Layout,
    /// Each rank's node at the committed epoch: ranks `0..world` after every commit (0035's
    /// normalized ids).
    pub placement: &'a BTreeMap<u32, String>,
    /// The spares admitted to the engine's pool, standing on nodes that are not cordoned.
    pub spares: &'a [StandingSpare],
    /// Whether the policy may answer a loss with REPLACE.
    pub replace: bool,
    /// Whether it may answer one with SHRINK.
    pub shrink: bool,
    /// The most groups any one process may hold, counted over both scopes; None for no bound.
    pub budget: Option<u32>,
    /// The groups each spare process (by node and device) holds in other engines' plans: a
    /// pool's spare stands for all its engines at once, so the budget counts across them
    /// (decision 0031).
    pub elsewhere: &'a BTreeMap<(String, u32), u32>,
}

/// A group's identity across plans and sets: its scope and the members it spans (decision 0035,
/// amended). Two sets that need a group of one scope over the same members need one
/// communicator.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GroupKey {
    /// The scope, by its wire value.
    pub scope: i32,
    /// The members it spans, in rank order.
    pub spans: Vec<Member>,
}

impl GroupKey {
    /// The key of a planned group.
    #[must_use]
    pub fn of(g: &CoverGroup) -> Self {
        Self::from_parts(g.scope, &g.spans)
    }

    /// The key of a group named by its scope and spans on the wire.
    #[must_use]
    pub fn from_parts(scope: i32, spans: &[SetMember]) -> Self {
        Self {
            scope,
            spans: spans.iter().filter_map(Member::from_wire).collect(),
        }
    }
}

/// A loss the plan holds no set for, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uncovered {
    /// The ranks lost together.
    pub lost: Vec<u32>,
    /// Why no set covers it.
    pub why: String,
}

/// One set the plan names, with why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedSet {
    /// Its members, by position (a survivor set lacks the lost positions).
    pub set: Vec<Member>,
    /// What it is for: `current`, or the loss it answers and how.
    pub purpose: String,
    /// The groups it needs, both scopes, by their index in the plan's groups.
    pub groups: Vec<usize>,
}

/// What the planner returns.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// The sets, in plan order.
    pub sets: Vec<PlannedSet>,
    /// Every group any set needs, each once, with where it meets.
    pub groups: Vec<CoverGroup>,
    /// Losses no set covers.
    pub uncovered: Vec<Uncovered>,
}

/// `plan` without the groups in `dropped`, so their members let them go: each set that needs one
/// is left out with it, and the others keep their groups, renumbered. The current set keeps every
/// group it needs: a process never lets go of the group it serves on.
#[must_use]
pub fn without(plan: Plan, dropped: &BTreeSet<GroupKey>) -> Plan {
    let serving: BTreeSet<usize> = plan
        .sets
        .iter()
        .filter(|s| s.purpose == "current")
        .flat_map(|s| s.groups.iter().copied())
        .collect();
    let kept: Vec<bool> = plan
        .groups
        .iter()
        .enumerate()
        .map(|(i, g)| serving.contains(&i) || !dropped.contains(&GroupKey::of(g)))
        .collect();
    let mut renumbered = vec![None; plan.groups.len()];
    let mut groups = Vec::new();
    for (i, g) in plan.groups.into_iter().enumerate() {
        if kept[i] {
            renumbered[i] = Some(groups.len());
            groups.push(g);
        }
    }
    let sets = plan
        .sets
        .into_iter()
        .filter_map(|s| {
            let needs: Option<Vec<usize>> = s.groups.iter().map(|&i| renumbered[i]).collect();
            Some(PlannedSet {
                groups: needs?,
                ..s
            })
        })
        .collect();
    Plan {
        sets,
        groups,
        uncovered: plan.uncovered,
    }
}

/// Plan the cover for `input`.
#[must_use]
pub fn plan(input: &Input<'_>) -> Plan {
    let world = input.layout.world();
    let current: Vec<Member> = (0..world).map(Member::Rank).collect();
    let mut out = Plan::default();
    let mut held: BTreeMap<Process, u32> = input
        .elsewhere
        .iter()
        .map(|((node, device), n)| (Process::Spare(node.clone(), *device), *n))
        .collect();
    let mut named: BTreeSet<Vec<Member>> = BTreeSet::new();
    // the current membership's set, always: its groups are the serving ones, and a plan that
    // omitted them would ask every process to release what it serves on
    admit(
        input,
        &mut out,
        &mut held,
        &mut named,
        current,
        "current".to_owned(),
    );
    let mut over_budget: BTreeSet<Vec<u32>> = BTreeSet::new();
    let all = losses(input);
    // each loss's set is chosen as the plan stands, so a spare standing for any position is the
    // one holding fewest groups so far: the spares share the losses, and no one spare's budget
    // caps the plan
    let mut consider = |out: &mut Plan,
                        held: &mut BTreeMap<Process, u32>,
                        set: Vec<Member>,
                        purpose: String,
                        lost: &[u32]| {
        if named.contains(&set) {
            return;
        }
        if input
            .budget
            .is_none_or(|max| within(input.layout, &set, out, held, max))
        {
            admit(input, out, held, &mut named, set, purpose);
        } else {
            over_budget.insert(lost.to_vec());
        }
    };
    for lost in &all {
        if input.replace
            && let Some(set) = replacement(input, lost, &BTreeSet::new(), &held)
        {
            consider(
                &mut out,
                &mut held,
                set,
                format!("REPLACE of ranks {lost:?}"),
                lost,
            );
        }
        if input.shrink {
            let set = (0..world)
                .filter(|r| !lost.contains(r))
                .map(Member::Rank)
                .collect();
            consider(
                &mut out,
                &mut held,
                set,
                format!("SHRINK of ranks {lost:?}"),
                lost,
            );
        }
    }
    if input.replace {
        // a second spare for a position's loss already covered, never the one covering it
        for lost in all.iter().filter(|l| l.len() == 1) {
            let used: BTreeSet<(String, u32)> = out
                .sets
                .iter()
                .filter(|s| answers(&s.set, lost))
                .flat_map(|s| s.set.iter())
                .filter_map(|m| match m {
                    Member::Spare { node, device, .. } => Some((node.clone(), *device)),
                    Member::Rank(_) => None,
                })
                .collect();
            if used.is_empty() {
                continue;
            }
            if let Some(set) = replacement(input, lost, &used, &held) {
                let purpose = format!("REPLACE of ranks {lost:?} through a second spare");
                consider(&mut out, &mut held, set, purpose, lost);
            }
        }
    }
    for lost in losses(input) {
        if covers(&out.sets, &lost) {
            continue;
        }
        let why = if over_budget.contains(&lost) {
            format!(
                "over the cover budget of {} groups per GPU",
                input.budget.unwrap_or(0)
            )
        } else if !input.replace && !input.shrink {
            "the policy allows neither REPLACE nor SHRINK".to_owned()
        } else {
            // with SHRINK allowed, a loss is covered or over the budget
            "no spare can take it and SHRINK is not allowed".to_owned()
        };
        out.uncovered.push(Uncovered { lost, why });
    }
    out
}

/// Whether every process in `set` stays within `max` groups once the groups the set needs and
/// `out` does not hold yet are added: a group another set already needs costs nothing more.
fn within(
    layout: Layout,
    set: &[Member],
    out: &Plan,
    held: &BTreeMap<Process, u32>,
    max: u32,
) -> bool {
    let mut adds: BTreeMap<Process, u32> = BTreeMap::new();
    for (scope, spans) in groups_of(layout, set) {
        let key = GroupKey {
            scope: scope as i32,
            spans: spans.clone(),
        };
        if index_of(out, &key).is_some() {
            continue;
        }
        for m in spans {
            *adds.entry(m.process()).or_default() += 1;
        }
    }
    adds.iter()
        .all(|(p, n)| held.get(p).copied().unwrap_or(0) + n <= max)
}

/// The index of the group `key` names in `out`, if a set already needs it.
fn index_of(out: &Plan, key: &GroupKey) -> Option<usize> {
    out.groups.iter().position(|g| GroupKey::of(g) == *key)
}

/// Whether a set in `sets` answers the loss of `lost`.
fn covers(sets: &[PlannedSet], lost: &[u32]) -> bool {
    sets.iter().any(|s| answers(&s.set, lost))
}

/// Whether `set` is a set the engine moves to on losing `lost`: no serving rank of `lost` in it,
/// every other rank serving, and every lost position either held by a spare or gone with the set.
#[must_use]
pub fn answers(set: &[Member], lost: &[u32]) -> bool {
    let positions: BTreeSet<u32> = set.iter().map(Member::position).collect();
    let survivors_serve = set
        .iter()
        .filter(|m| !lost.contains(&m.position()))
        .all(|m| matches!(m, Member::Rank(_)));
    let lost_replaced = set
        .iter()
        .filter(|m| lost.contains(&m.position()))
        .all(|m| matches!(m, Member::Spare { .. }));
    let replaced = lost.iter().all(|r| positions.contains(r));
    let shrunk = lost.iter().all(|r| !positions.contains(r));
    survivors_serve && lost_replaced && (replaced || shrunk) && !lost.is_empty()
}

/// Add `set` to the plan: the groups it needs that the plan holds already, and its others,
/// counted against each member.
fn admit(
    input: &Input<'_>,
    out: &mut Plan,
    held: &mut BTreeMap<Process, u32>,
    named: &mut BTreeSet<Vec<Member>>,
    set: Vec<Member>,
    purpose: String,
) {
    let mut groups = Vec::new();
    for (scope, spans) in groups_of(input.layout, &set) {
        let key = GroupKey {
            scope: scope as i32,
            spans: spans.clone(),
        };
        if let Some(i) = index_of(out, &key) {
            groups.push(i);
            continue;
        }
        for m in &spans {
            *held.entry(m.process()).or_default() += 1;
        }
        groups.push(out.groups.len());
        out.groups.push(CoverGroup {
            scope: scope as i32,
            server: Some(spans[0].to_wire()),
            spans: spans.iter().map(Member::to_wire).collect(),
        });
    }
    named.insert(set.clone());
    out.sets.push(PlannedSet {
        set,
        purpose,
        groups,
    });
}

/// A set's groups: a stage's in the domain scope, a column's in the fabric scope, each spanning
/// the set's members of that stage or column in set order. A group of one member is none.
fn groups_of(layout: Layout, set: &[Member]) -> Vec<(Scope, Vec<Member>)> {
    let mut out = Vec::new();
    for stage in 0..layout.pipeline_parallel {
        let spans: Vec<Member> = set
            .iter()
            .filter(|m| layout.stage(m.position()) == stage)
            .cloned()
            .collect();
        if spans.len() > 1 {
            out.push((Scope::Domain, spans));
        }
    }
    if layout.pipeline_parallel > 1 {
        for column in 0..layout.tensor_parallel {
            let spans: Vec<Member> = set
                .iter()
                .filter(|m| layout.column(m.position()) == column)
                .cloned()
                .collect();
            if spans.len() > 1 {
                out.push((Scope::Fabric, spans));
            }
        }
    }
    out
}

/// The losses the plan answers, in order: each node's ranks, then each position on a node that
/// holds more than one.
fn losses(input: &Input<'_>) -> Vec<Vec<u32>> {
    losses_of(input.placement)
}

/// The losses a placement can suffer that cover answers, in order: each node's ranks together,
/// then each position alone on a node that holds more than one.
#[must_use]
pub fn losses_of(placement: &BTreeMap<u32, String>) -> Vec<Vec<u32>> {
    let mut by_node: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for (rank, node) in placement {
        by_node.entry(node).or_default().push(*rank);
    }
    let mut out: Vec<Vec<u32>> = by_node.values().cloned().collect();
    for ranks in by_node.values().filter(|r| r.len() > 1) {
        out.extend(ranks.iter().map(|r| vec![*r]));
    }
    out
}

/// The set the engine moves to when `lost` is replaced: each lost position held by its own spare
/// or, failing that, the one standing for any position that `held` counts fewest groups for, in
/// pool order among equals, never one in `exclude` nor one on a node the loss takes whole (a
/// process lost on a living node may be replaced on it); None when the spares cannot take every
/// lost position.
fn replacement(
    input: &Input<'_>,
    lost: &[u32],
    exclude: &BTreeSet<(String, u32)>,
    held: &BTreeMap<Process, u32>,
) -> Option<Vec<Member>> {
    let lost_nodes: BTreeSet<&String> = lost
        .iter()
        .filter_map(|r| input.placement.get(r))
        .filter(|node| {
            input
                .placement
                .iter()
                .filter(|(_, n)| n == node)
                .all(|(r, _)| lost.contains(r))
        })
        .collect();
    let usable = |s: &&StandingSpare| {
        !lost_nodes.contains(&s.node) && !exclude.contains(&(s.node.clone(), s.device))
    };
    let mut taken: BTreeSet<(String, u32)> = BTreeSet::new();
    let mut by_rank: BTreeMap<u32, &StandingSpare> = BTreeMap::new();
    for r in lost {
        let own = input
            .spares
            .iter()
            .filter(usable)
            .find(|s| s.for_rank == Some(*r) && !taken.contains(&(s.node.clone(), s.device)));
        let any = || {
            input
                .spares
                .iter()
                .filter(usable)
                .filter(|s| s.for_rank.is_none() && !taken.contains(&(s.node.clone(), s.device)))
                .min_by_key(|s| {
                    held.get(&Process::Spare(s.node.clone(), s.device))
                        .copied()
                        .unwrap_or(0)
                })
        };
        let s = own.or_else(any)?;
        taken.insert((s.node.clone(), s.device));
        by_rank.insert(*r, s);
    }
    Some(
        (0..input.layout.world())
            .map(|p| match by_rank.get(&p) {
                Some(s) => Member::Spare {
                    for_rank: p,
                    node: s.node.clone(),
                    device: s.device,
                },
                None => Member::Rank(p),
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{Input, Layout, Member, StandingSpare, plan, set_key};
    use crate::proto::everserve::signals::v1::Scope;

    fn placement(nodes: &[(&str, &[u32])]) -> BTreeMap<u32, String> {
        nodes
            .iter()
            .flat_map(|(n, ranks)| ranks.iter().map(move |r| (*r, (*n).to_owned())))
            .collect()
    }

    fn spare(node: &str, device: u32, for_rank: Option<u32>) -> StandingSpare {
        StandingSpare {
            node: node.into(),
            device,
            for_rank,
            host: String::new(),
        }
    }

    struct Case {
        layout: Layout,
        placement: BTreeMap<u32, String>,
        spares: Vec<StandingSpare>,
        replace: bool,
        shrink: bool,
        budget: Option<u32>,
        elsewhere: BTreeMap<(String, u32), u32>,
    }

    impl Case {
        fn new(tp: u32, pp: u32, nodes: &[(&str, &[u32])]) -> Self {
            Self {
                layout: Layout {
                    tensor_parallel: tp,
                    pipeline_parallel: pp,
                },
                placement: placement(nodes),
                spares: Vec::new(),
                replace: true,
                shrink: false,
                budget: None,
                elsewhere: BTreeMap::new(),
            }
        }

        fn plan(&self) -> super::Plan {
            plan(&Input {
                layout: self.layout,
                placement: &self.placement,
                spares: &self.spares,
                replace: self.replace,
                shrink: self.shrink,
                budget: self.budget,
                elsewhere: &self.elsewhere,
            })
        }
    }

    fn keys(p: &super::Plan) -> Vec<String> {
        p.sets.iter().map(|s| set_key(&s.set)).collect()
    }

    #[test]
    fn the_current_set_comes_first_then_each_nodes_loss_then_each_positions() {
        // TP2 x PP2: node-a holds stage 0, node-b stage 1; spares for ranks 2 and 3 on node-c
        let mut c = Case::new(2, 2, &[("node-a", &[0, 1]), ("node-b", &[2, 3])]);
        c.spares = vec![spare("node-c", 0, Some(2)), spare("node-c", 1, Some(3))];
        let p = c.plan();
        assert_eq!(
            keys(&p),
            [
                "0-1-2-3",
                "0-1-R2@node-c:0-R3@node-c:1",
                "0-1-R2@node-c:0-3",
                "0-1-2-R3@node-c:1",
            ]
        );
        assert_eq!(p.sets[0].purpose, "current");
        // node-a's loss has no spare and SHRINK is not allowed: it is reported, not covered
        assert_eq!(p.uncovered.len(), 3, "{:?}", p.uncovered);
        assert_eq!(p.uncovered[0].lost, [0, 1]);
    }

    #[test]
    fn groups_span_a_stage_in_the_domain_and_a_column_in_the_fabric_in_set_order() {
        let c = Case::new(2, 2, &[("node-a", &[0, 1]), ("node-b", &[2, 3])]);
        let p = c.plan();
        let current: Vec<_> = p.groups.iter().collect();
        let spans = |scope: Scope| -> Vec<Vec<Member>> {
            current
                .iter()
                .filter(|g| g.scope == scope as i32)
                .map(|g| g.spans.iter().filter_map(Member::from_wire).collect())
                .collect()
        };
        assert_eq!(
            spans(Scope::Domain),
            [
                vec![Member::Rank(0), Member::Rank(1)],
                vec![Member::Rank(2), Member::Rank(3)]
            ]
        );
        assert_eq!(
            spans(Scope::Fabric),
            [
                vec![Member::Rank(0), Member::Rank(2)],
                vec![Member::Rank(1), Member::Rank(3)]
            ]
        );
        // the first member serves the store, on its node's host
        assert!(
            p.groups
                .iter()
                .all(|g| g.server.as_ref() == g.spans.first())
        );
    }

    #[test]
    fn a_spare_on_a_node_answers_its_processes_losses_but_never_the_nodes() {
        let mut c = Case::new(4, 1, &[("node-a", &[0, 1, 2, 3])]);
        c.spares = vec![spare("node-a", 4, None), spare("node-b", 0, None)];
        let p = c.plan();
        // node-a's loss needs four spares and only node-b's survives it: uncovered
        let node = p
            .uncovered
            .iter()
            .find(|u| u.lost == [0, 1, 2, 3])
            .expect("node-a");
        assert!(node.why.contains("no spare"), "{node:?}");
        let on = |r: u32, node: &str, device: u32| {
            p.sets.iter().any(|s| {
                s.set[r as usize]
                    == Member::Spare {
                        for_rank: r,
                        node: node.into(),
                        device,
                    }
            })
        };
        for r in 0..4u32 {
            // a process lost on a living node: the spare beside it first, in pool order, then
            // node-b's as the second spare
            assert!(
                on(r, "node-a", 4) && on(r, "node-b", 0),
                "rank {r}: {:?}",
                keys(&p)
            );
        }
    }

    #[test]
    fn a_second_spare_covers_a_position_already_covered_after_every_first_cover() {
        let mut c = Case::new(2, 1, &[("node-a", &[0, 1])]);
        c.spares = vec![spare("node-b", 0, None), spare("node-c", 0, None)];
        let p = c.plan();
        let second: Vec<_> = p
            .sets
            .iter()
            .filter(|s| s.purpose.contains("second spare"))
            .collect();
        assert_eq!(second.len(), 2, "{:?}", keys(&p));
        let first_second = p
            .sets
            .iter()
            .position(|s| s.purpose.contains("second"))
            .unwrap();
        assert!(
            p.sets[..first_second]
                .iter()
                .all(|s| !s.purpose.contains("second")),
            "first covers come before any second"
        );
    }

    #[test]
    fn the_budget_bounds_the_groups_any_process_holds_and_reports_what_it_left_out() {
        let mut c = Case::new(4, 1, &[("node-a", &[0, 1]), ("node-b", &[2, 3])]);
        c.spares = vec![spare("node-c", 0, None), spare("node-c", 1, None)];
        c.budget = Some(2);
        let p = c.plan();
        // every process holds two groups at most: the current set and one node's loss each
        let mut held: BTreeMap<super::Process, u32> = BTreeMap::new();
        for g in &p.groups {
            for m in g.spans.iter().filter_map(Member::from_wire) {
                *held.entry(m.process()).or_default() += 1;
            }
        }
        assert!(held.values().all(|n| *n <= 2), "{held:?}");
        assert_eq!(p.sets.len(), 3, "{:?}", keys(&p));
        assert!(
            p.uncovered
                .iter()
                .any(|u| u.why.contains("budget of 2 groups per GPU")),
            "{:?}",
            p.uncovered
        );
    }

    #[test]
    fn shrink_sets_drop_the_lost_positions() {
        let mut c = Case::new(2, 1, &[("node-a", &[0]), ("node-b", &[1])]);
        c.replace = false;
        c.shrink = true;
        let p = c.plan();
        // each node holds one rank: its loss is a single set, and a group of one is none
        assert_eq!(keys(&p), ["0-1", "1", "0"]);
        assert_eq!(p.groups.len(), 1, "only the current set spans two members");
        assert!(p.uncovered.is_empty());
    }

    #[test]
    fn a_spares_groups_in_other_engines_count_against_its_budget() {
        // decision 0031: node-b's spare already holds two groups in another engine's plan
        let mut c = Case::new(2, 1, &[("node-a", &[0, 1])]);
        c.spares = vec![spare("node-b", 0, None)];
        c.budget = Some(2);
        c.elsewhere.insert(("node-b".into(), 0), 2);
        let p = c.plan();
        assert_eq!(keys(&p), ["0-1"], "no set may name node-b's spare");
        // each position's loss would go through it, and is left out for the budget; node-a's
        // needs two spares and has one
        for lost in [vec![0], vec![1]] {
            let u = p
                .uncovered
                .iter()
                .find(|u| u.lost == lost)
                .expect("uncovered");
            assert!(u.why.contains("budget"), "{u:?}");
        }
        c.elsewhere.clear();
        assert!(
            c.plan().sets.len() > 1,
            "without the other engine's groups it is planned"
        );
    }

    #[test]
    fn the_current_set_is_named_even_over_budget() {
        let mut c = Case::new(4, 1, &[("node-a", &[0, 1, 2, 3])]);
        c.budget = Some(0);
        let p = c.plan();
        assert_eq!(keys(&p), ["0-1-2-3"]);
    }

    /// A plan of `tp` x `pp` ranks, `tp` to a node, with one node of `tp` spares standing for any
    /// position, under `budget`.
    fn wide(tp: u32, pp: u32, budget: Option<u32>) -> super::Plan {
        let placement: BTreeMap<u32, String> = (0..tp * pp)
            .map(|r| (r, format!("node-{}", r / tp)))
            .collect();
        let spares: Vec<StandingSpare> = (0..tp)
            .map(|d| spare(&format!("node-{pp}"), d, None))
            .collect();
        plan(&Input {
            layout: Layout {
                tensor_parallel: tp,
                pipeline_parallel: pp,
            },
            placement: &placement,
            spares: &spares,
            replace: true,
            shrink: false,
            budget,
            elsewhere: &BTreeMap::new(),
        })
    }

    #[test]
    fn a_plans_groups_grow_with_its_world_and_the_spares_share_the_losses() {
        // decision 0035, amended: a group is formed once for every set that needs it, so a
        // plan grows with its world (with sets re-forming their groups it was 350, 828 and
        // 2,192 groups, about 730 KB at 64 GPUs); and each loss takes the spare standing for any
        // position that holds fewest groups, so under a budget the spares share the losses
        let groups = |pp| wide(8, pp, None).groups.len();
        assert_eq!([groups(2), groups(4)], [76, 144]);
        let sixty_four = wide(8, 8, None);
        let bytes: usize = sixty_four
            .groups
            .iter()
            .map(prost::Message::encoded_len)
            .sum();
        assert_eq!(sixty_four.groups.len(), 280);
        // about 15 KB: a group carries no address since it meets on the control plane's links
        assert!((12_000..18_000).contains(&bytes), "{bytes} bytes");
        // with one spare taking every loss, a budget of 8 planned 9 sets at 16 GPUs and left 12
        // losses uncovered; shared among the spares, 13 sets leave 6
        let budgeted = wide(8, 2, Some(8));
        assert_eq!((budgeted.sets.len(), budgeted.uncovered.len()), (13, 6));
    }

    #[test]
    fn a_plan_without_failed_groups_drops_the_sets_needing_them_and_never_the_current_one() {
        use super::{GroupKey, Plan, PlannedSet, without};
        use crate::proto::everserve::signals::v1::CoverGroup;
        use std::collections::BTreeSet;
        let g = |spans: &[u32]| CoverGroup {
            scope: 1,
            spans: spans.iter().map(|r| Member::Rank(*r).to_wire()).collect(),
            ..CoverGroup::default()
        };
        let set = |purpose: &str, groups: Vec<usize>| PlannedSet {
            set: Vec::new(),
            purpose: purpose.into(),
            groups,
        };
        let plan = Plan {
            groups: vec![g(&[0, 1]), g(&[2, 3]), g(&[4, 5])],
            sets: vec![
                set("current", vec![0]),
                set("loss of 2", vec![0, 1]),
                set("loss of 4", vec![2]),
            ],
            uncovered: Vec::new(),
        };
        // the current set's group stays, though failed; the set needing [2, 3] goes with it
        let dropped: BTreeSet<GroupKey> = [GroupKey::of(&g(&[0, 1])), GroupKey::of(&g(&[2, 3]))]
            .into_iter()
            .collect();
        let out = without(plan, &dropped);
        assert_eq!(out.groups, vec![g(&[0, 1]), g(&[4, 5])]);
        let kept: Vec<(&str, Vec<usize>)> = out
            .sets
            .iter()
            .map(|s| (s.purpose.as_str(), s.groups.clone()))
            .collect();
        assert_eq!(kept, vec![("current", vec![0]), ("loss of 4", vec![1])]);
    }
}
