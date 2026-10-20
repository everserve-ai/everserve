// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The membership view of each engine across nodes: its members, spares, processes, head, epochs and the records readiness reads (layer 5, decision 0034).

use std::collections::{BTreeMap, BTreeSet};

use tracing::{info, warn};

use crate::clock::now_ms;
use crate::policy::cover::Member;
use crate::proto::everserve::coord::v1::{WorkerEvent, worker_event};
use crate::proto::everserve::signals::v1::{CompileCache, SlotChoice, SoftwareIdentity};
use crate::readiness::{IdentityReadiness, JoinStamp, LastChange, ProcessReadiness};
use crate::status::Status;

/// The coordinator's view of one engine across nodes.
#[derive(Debug, Default, Clone)]
pub struct EngineView {
    /// The last claimed epoch: the last committed one, or one a change in flight claimed or a
    /// failed change spent (decision 0032). The next change claims the one after it.
    pub epoch: u64,
    /// The last committed epoch: what readiness reports, so a scheduler never reads an epoch
    /// whose membership is not yet in place.
    pub committed_epoch: u64,
    /// The change in flight: the epoch it claimed and the ranks it takes.
    pub change: Option<(u64, Vec<u32>)>,
    /// The last committed change's record, which readiness reports.
    pub last_change: Option<LastChange>,
    /// Join payloads routed for an epoch not yet committed, carried into its record at the
    /// commit.
    pub joins: Vec<JoinStamp>,
    /// Each serving rank's host, as its process registered it (decision 0041).
    pub hosts: BTreeMap<u32, String>,
    /// The step config the engine's processes must share (decision 0034's identity): the first
    /// a process of the engine reported, asking or registering.
    pub step: Option<crate::proto::everserve::signals::v1::StepConfig>,
    /// The instance state last sent to the engine's processes (decision 0041).
    pub instance_state: Option<crate::proto::everserve::signals::v1::InstanceState>,
    /// The placement last sent to the engine's processes (decision 0041).
    pub placement: Option<crate::proto::everserve::signals::v1::EnginePlacement>,
    /// Rank to the node its worker is registered on.
    pub members: BTreeMap<u32, String>,
    /// The pre-loaded spares: the rank each stands in for, to its node.
    pub spares: BTreeMap<u32, String>,
    /// Ranks lost under SHRINK, in order.
    pub lost: Vec<u32>,
    /// Ranks whose worker went away without a change; a report must still find them.
    pub departed: BTreeSet<u32>,
    /// The world size the policy or the registrations imply.
    pub world_size: u32,
    /// The device class each rank runs on, when its node reported devices.
    pub classes: BTreeMap<u32, String>,
    /// The replication targets last sent for each rank, so a change in the spare nodes' stores
    /// is sent once.
    pub replication: BTreeMap<u32, Vec<String>>,
    /// The node each rank's replication targets were last sent to: when a rank moves, that
    /// node's store is told the rank replicates nowhere, so it stops draining a backlog into a
    /// replica that has moved on, every step refused as stale at each retry, 40 ms apart.
    pub replicating_from: BTreeMap<u32, String>,
    /// The nodes told to expect each rank's rows, and from whom, so a target no longer one is
    /// told to expect nobody and a new holder's targets are told the holder (decision 0026).
    pub expected_at: BTreeMap<u32, (String, BTreeSet<String>)>,
    /// The last pause measured for each outcome taken on this engine, in milliseconds, by the
    /// outcome's label; what the readiness API reports as the expected pause.
    pub last_pause_ms: BTreeMap<String, u64>,
    /// The engine's slot-collective choice on each node, as its client (an engine node) or its
    /// spare (a spare node, decision 0024) registered it; absent until one did.
    pub slots: BTreeMap<String, SlotChoice>,
    /// The epoch at which each rank last left or was replaced; a loss report from an earlier
    /// epoch is folded (decision 0021).
    pub changed_at: BTreeMap<u32, u64>,
    /// The compile cache key the engine's first registered process reported (decision 0029): one
    /// engine, one compiled artifact; every later process must report it.
    pub compile_key: Option<String>,
    /// The compile cache pair each serving rank reported. Digests differ by rank on a pipeline
    /// engine, since each stage compiles its own graph, so a position's pair is its rank's.
    pub caches: BTreeMap<u32, CompileCache>,
    /// The compile cache pair each admitted spare reported.
    pub spare_caches: BTreeMap<u32, CompileCache>,
    /// Lost ranks whose rows are being staged onto their spare's node for the change in flight,
    /// to the node holding a replica of them (decision 0030); cleared at the commit.
    pub restoring: BTreeMap<u32, String>,
    /// Spares standing for any position (decision 0030), by node and device.
    pub any_spares: BTreeSet<(String, u32)>,
    /// Which of them takes which lost rank in the change in flight; moved into `members` at the
    /// commit, returned to `any_spares` when the delivery fails.
    pub taking: BTreeMap<u32, (String, u32)>,
    /// The processes behind the members and spares, as their daemons saw them register.
    pub processes: Processes,
    /// Nodes a process of the engine reached the coordinator from, or whose daemon's hello named
    /// it, since their link came up: part of the engine's audience (`audience`).
    pub reached: BTreeSet<String>,
    /// The engine's head (decision 0032), as its daemon saw it register; None until it did, and
    /// once it departed.
    pub head: Option<HeadProcess>,
    /// Releases each rank's process left behind (decision 0025, amended), while that process
    /// holds the rank: it is retired by notice once a spare can take it.
    pub leaks: BTreeMap<u32, super::leaks::Leaks>,
    /// Why the engine is down, since when; None while it is not (decision 0032).
    pub down: Option<Down>,
    /// A second head that registered on another node while the engine's head was alive: refused,
    /// kept until it departs.
    pub head_conflict: Option<HeadProcess>,
    /// Each position's software identity, as its serving process registered it; kept when that
    /// process departs, so the spare that takes the position is held to it (decision 0034).
    pub identities: BTreeMap<u32, SoftwareIdentity>,
    /// The device each rank's holder registered on, on its node: with `members`, the process
    /// holding the rank (decision 0035's placement).
    pub devices: BTreeMap<u32, u32>,
    /// Spares kept out of the pool for what they run, by node and device: the rank each stood
    /// for (None for any position) and the field that differed.
    pub refused_spares: BTreeMap<(String, u32), (Option<u32>, String)>,
    /// Every spare standing for the engine, admitted or refused, by node and device: its
    /// registration, from which its admission is derived again whenever what it is held to
    /// changes (decision 0036). Dropped when it departs or takes a position.
    pub standing: BTreeMap<(String, u32), WorkerEvent>,
    /// Its conditions and event log (decision 0036), derived after every change to this view.
    pub status: Status,
    /// The cover plan delivered for the engine and the groups reported against it (0035).
    pub cover: super::cover::CoverState,
    /// How the change in flight used cover, for its record: through which covered set, or
    /// formed in the change; empty while cover is not planned.
    pub change_cover: Option<String>,
    /// The nodes the change in flight takes ranks from: their daemons hear its commit even when
    /// the node holds nothing of the engine after it (a drained node, whose node role ends on
    /// it).
    pub change_from: BTreeSet<String>,
}

/// An engine's head: its process (the engine core) and the process its operator launched.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadProcess {
    /// The engine core.
    pub process: Process,
    /// The launcher's pid, as the head reported it in its own pid namespace; the engine core's
    /// when it runs in it. For people: nothing is signalled by it.
    pub launcher_pid: u32,
    /// The address its engine core registered (`Register.host`); empty when it named none.
    pub host: String,
}

/// Why an engine is down (decision 0032).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Down {
    /// Since when, Unix milliseconds on this clock.
    pub at_ms: i64,
    /// `stopped`, `failed` or `lost` from the head's departure; `change_failed` for a change that
    /// failed after its delivery.
    pub cause: &'static str,
    /// The head's goodbye, or the failed change's error.
    pub detail: String,
}

/// One registered process: its node, its id there, and when it registered.
#[derive(Debug, Clone, PartialEq)]
pub struct Process {
    /// The node it runs on.
    pub node: String,
    /// Its id on that node, from the daemon socket's peer credentials; None when the daemon
    /// cannot see it, as a process in another pod, in another pid namespace, on Kubernetes. A
    /// process is named by its connection, never by its pid (decision 0041).
    pub pid: Option<u32>,
    /// When it registered, Unix milliseconds on this clock.
    pub registered_at_ms: i64,
    /// What it runs, as it registered (decision 0034); None when it reported none.
    pub identity: Option<SoftwareIdentity>,
}

/// The processes behind an engine's members and spares, kept beside them so readiness and a
/// stacks request name the process, never a guess at it.
#[derive(Debug, Clone, Default)]
pub struct Processes {
    /// Serving ranks.
    pub members: BTreeMap<u32, Process>,
    /// Spares preloaded for one rank.
    pub spares: BTreeMap<u32, Process>,
    /// Spares standing for any position, by node and device (decision 0030).
    pub any: BTreeMap<(String, u32), Process>,
}

impl Processes {
    /// Keep what a worker event says about its process: a registration records it, a departure
    /// from the node that holds the entry drops it.
    pub(super) fn record(&mut self, node: &str, ev: &WorkerEvent) {
        let process = || Process {
            node: node.to_owned(),
            pid: (ev.pid > 0).then_some(ev.pid),
            registered_at_ms: now_ms(),
            identity: ev.identity.clone(),
        };
        let here = |p: &Process| p.node == node;
        match worker_event::Kind::try_from(ev.kind).unwrap_or(worker_event::Kind::Unspecified) {
            worker_event::Kind::WorkerRegistered => {
                self.members.insert(ev.rank, process());
            }
            worker_event::Kind::SpareRegistered if ev.any_rank => {
                self.any.insert((node.to_owned(), ev.device), process());
            }
            worker_event::Kind::SpareRegistered => {
                self.spares.insert(ev.rank, process());
            }
            worker_event::Kind::WorkerDeparted if self.members.get(&ev.rank).is_some_and(here) => {
                self.members.remove(&ev.rank);
            }
            worker_event::Kind::SpareDeparted if ev.any_rank => {
                self.any.remove(&(node.to_owned(), ev.device));
            }
            worker_event::Kind::SpareDeparted if self.spares.get(&ev.rank).is_some_and(here) => {
                self.spares.remove(&ev.rank);
            }
            _ => {}
        }
    }

    /// The spare that took `rank` at a commit serves it now: its process becomes the rank's.
    pub(super) fn promote(&mut self, rank: u32, taken_from: Option<&(String, u32)>) {
        let process = match taken_from {
            Some(at) => self.any.remove(at),
            None => self.spares.remove(&rank),
        };
        if let Some(p) = process {
            self.members.insert(rank, p);
        }
    }
}

/// A process as the readiness view reports it.
pub(super) fn process_view(p: Option<&Process>) -> Option<ProcessReadiness> {
    p.map(|p| ProcessReadiness {
        pid: p.pid,
        registered_at_ms: p.registered_at_ms,
        identity: p.identity.as_ref().map(IdentityReadiness::from_wire),
    })
}

/// Join payloads kept per change: enough for every position of a large engine, bounded so a
/// stream of joins cannot grow the view.
pub(super) const MAX_JOINS: usize = 64;

impl EngineView {
    /// The epoch of the rank's last change, when `ev` is a departure its node saw before it: a
    /// process that no longer holds the rank, such as the worker a spare on the same node
    /// replaced, whose departure reaches this view after the commit that promoted the spare.
    pub(super) fn stale_departure(&self, ev: &WorkerEvent) -> Option<u64> {
        if ev.kind != worker_event::Kind::WorkerDeparted as i32 {
            return None;
        }
        self.changed_at
            .get(&ev.rank)
            .copied()
            .filter(|&changed| changed > ev.epoch)
    }

    /// Keep a join's stamps with the change of its epoch: the last committed one's record, or
    /// held for a change not yet committed. A join for an older epoch is dropped.
    /// Whether `rank` has had a holder: a worker registered now, or one that departed, was lost,
    /// or was replaced, or a change in flight takes it. Such a rank's holder changes only through
    /// a committed change, never through a registration (decision 0041).
    pub(super) fn held_ever(&self, rank: u32) -> bool {
        self.members.contains_key(&rank)
            || self.departed.contains(&rank)
            || self.lost.contains(&rank)
            || self.changed_at.contains_key(&rank)
            || self
                .change
                .as_ref()
                .is_some_and(|(_, ranks)| ranks.contains(&rank))
    }

    pub(super) fn record_join(&mut self, stamp: JoinStamp) {
        match self.last_change.as_mut() {
            Some(c) if c.epoch == stamp.epoch => {
                if c.joins.len() < MAX_JOINS {
                    c.joins.push(stamp);
                }
            }
            _ => {
                if stamp.epoch > self.committed_epoch && self.joins.len() < MAX_JOINS {
                    self.joins.push(stamp);
                }
            }
        }
    }

    /// The engine has a compile key, every serving rank reported it, and every admitted spare's
    /// pair equals the pair of the serving rank it stands in for (decision 0029). A process that
    /// reported none, or a spare whose rank has no serving pair yet, makes the engine
    /// inconsistent: a check that no process answers is no check.
    pub(super) fn compile_cache_consistent(&self) -> bool {
        let Some(key) = self.compile_key.as_deref() else {
            return false;
        };
        self.members
            .keys()
            .all(|r| self.caches.get(r).is_some_and(|c| c.key == key))
            && self.spares.keys().all(|r| {
                self.spare_caches
                    .get(r)
                    .is_some_and(|s| self.caches.get(r) == Some(s))
            })
    }

    /// Whether `c` is the pair a spare for `rank` must report: the engine's key, and the serving
    /// rank's pair when that rank has reported one.
    pub(super) fn holds_for(&self, rank: u32, c: &CompileCache) -> bool {
        self.compile_key.as_deref() == Some(c.key.as_str())
            && self.caches.get(&rank).is_none_or(|m| m == c)
    }

    /// The compile key belongs to the engine's processes, not its name: when the last of them
    /// has gone, the next one to register sets it, so a re-formed engine's ranks are not held to
    /// the key of the engine that ran before it under the same name on the same control plane.
    pub(super) fn forget_compile_key_if_empty(&mut self) {
        if self.members.is_empty() && self.spares.is_empty() && self.any_spares.is_empty() {
            self.compile_key = None;
            self.caches.clear();
        }
    }

    /// The node whose spare takes `rank` in the change in flight: the one preloaded for it, or
    /// the one standing for any position that was assigned it (decision 0030).
    pub(super) fn spare_node_for(&self, rank: u32) -> Option<&String> {
        self.spares
            .get(&rank)
            .or_else(|| self.taking.get(&rank).map(|(n, _)| n))
    }

    /// Give every rank of `lost` without a spare of its own one standing for any position, in
    /// pool order (decision 0030), recording the assignment, while the pool lasts.
    pub(super) fn assign_any_spares(&mut self, lost: &[u32], cordoned: &BTreeSet<String>) {
        for &r in lost {
            if self.usable_spare(r, cordoned) || self.taking.contains_key(&r) {
                continue;
            }
            let Some(at) = self
                .any_spares
                .iter()
                .find(|(node, _)| !cordoned.contains(node))
                .cloned()
            else {
                return;
            };
            self.any_spares.remove(&at);
            self.taking.insert(r, at);
        }
    }

    /// A serving rank reported its compile cache (decision 0029): the engine's key is set or
    /// checked, the rank's pair recorded, and a spare admitted for the rank while its pair was
    /// unknown is held to it now.
    pub(super) fn hold_member_cache(
        &mut self,
        engine: &str,
        node: &str,
        rank: u32,
        c: &CompileCache,
    ) {
        if !same_key(&mut self.compile_key, &c.key) {
            warn!(
                "{engine}: rank {rank} on {node} runs on compile cache {}, the engine's key is {}: \
                 its rows will not match the engine's (decision 0029)",
                cache_text(Some(c)),
                self.compile_key.as_deref().unwrap_or("none")
            );
        }
        self.caches.insert(rank, c.clone());
        if self.spare_caches.get(&rank).is_some_and(|s| s != c) {
            warn!(
                "{engine}: spare for rank {rank} leaves the pool: it runs on compile cache {}, \
                 rank {rank}'s is {} (decision 0029)",
                cache_text(self.spare_caches.get(&rank)),
                cache_text(Some(c))
            );
            self.spares.remove(&rank);
            self.spare_caches.remove(&rank);
        }
    }

    /// A spare standing for any position joins the pool (decision 0030). It holds no shard and
    /// has compiled nothing yet, so it is held to its position's compile pair at the join, not
    /// here.
    pub(super) fn admit_any_spare(&mut self, engine: &str, node: &str, device: u32) {
        self.any_spares.insert((node.to_owned(), device));
        info!("{engine}: spare for any rank on {node} device {device} joins the pool");
    }

    /// A spare standing for any position left: out of the pool, and out of any take in flight.
    pub(super) fn forget_any_spare(&mut self, node: &str, device: u32) {
        self.any_spares.remove(&(node.to_owned(), device));
        self.taking.retain(|_, (n, d)| !(n == node && *d == device));
        self.forget_compile_key_if_empty();
    }

    /// Give every rank of `lost` the spare `set` names for it (decision 0035): a spare standing
    /// for any position is assigned the rank, as the pool would assign it, and leaves the pool; a
    /// rank's own spare needs no assignment.
    pub(super) fn assign_from_set(&mut self, lost: &[u32], set: &[Member]) {
        for m in set {
            let Member::Spare {
                for_rank,
                node,
                device,
            } = m
            else {
                continue;
            };
            if !lost.contains(for_rank) || self.spares.get(for_rank) == Some(node) {
                continue;
            }
            let at = (node.clone(), *device);
            if self.any_spares.remove(&at) {
                self.taking.insert(*for_rank, at);
            }
        }
    }

    /// Return the assignments of `lost` to the pool: the change did not happen.
    pub(super) fn unassign_any_spares(&mut self, lost: &[u32]) {
        for r in lost {
            if let Some(at) = self.taking.remove(r) {
                self.any_spares.insert(at);
            }
        }
    }

    /// Whether REPLACE has a spare for every rank of `lost` now, by its own spare or one standing
    /// for any position, without assigning anything.
    pub(super) fn spares_cover(&self, lost: &[u32], cordoned: &BTreeSet<String>) -> bool {
        let unmatched = lost
            .iter()
            .filter(|r| !self.usable_spare(**r, cordoned) && !self.taking.contains_key(r))
            .count();
        unmatched
            <= self
                .any_spares
                .iter()
                .filter(|(node, _)| !cordoned.contains(node))
                .count()
    }

    /// Whether `rank` has its own spare on a node that is not cordoned (decision 0036).
    pub(super) fn usable_spare(&self, rank: u32, cordoned: &BTreeSet<String>) -> bool {
        self.spares
            .get(&rank)
            .is_some_and(|node| !cordoned.contains(node))
    }
}

/// Whether `key` is the engine's compile key, which the first key reported becomes (decision
/// 0029).
pub(super) fn same_key(engine: &mut Option<String>, key: &str) -> bool {
    match engine {
        None => {
            *engine = Some(key.to_owned());
            true
        }
        Some(have) => have == key,
    }
}

/// A compile cache pair for a log line.
pub(super) fn cache_text(c: Option<&CompileCache>) -> String {
    c.map_or_else(|| "none".to_owned(), |c| format!("{}/{}", c.key, c.digest))
}

#[cfg(test)]
mod cover_tests {
    use super::EngineView;
    use crate::policy::cover::Member;

    #[test]
    fn a_covered_set_assigns_the_spare_it_names_not_the_pools_first() {
        let mut v = EngineView::default();
        v.any_spares.insert(("node-c".into(), 0));
        v.any_spares.insert(("node-d".into(), 0));
        // the set names node-d's spare for rank 1; the pool's first would be node-c's
        let set = vec![
            Member::Rank(0),
            Member::Spare {
                for_rank: 1,
                node: "node-d".into(),
                device: 0,
            },
        ];
        v.assign_from_set(&[1], &set);
        assert_eq!(v.taking.get(&1), Some(&("node-d".to_owned(), 0)));
        assert!(
            v.any_spares.contains(&("node-c".to_owned(), 0)),
            "node-c's stays in the pool"
        );
        assert!(!v.any_spares.contains(&("node-d".to_owned(), 0)));
    }
}
