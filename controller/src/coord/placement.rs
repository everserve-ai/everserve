// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Where each rank's KV lives: replication targets, and the staging of a lost rank's rows onto its spare's node before a change (decision 0030).

use std::collections::{BTreeMap, BTreeSet};

use tracing::{info, warn};

use crate::policy;
use crate::proto::everserve::coord::v1::{
    CoordToDaemon, ExpectReplication, ReplicateTargets, coord_to_daemon,
};

use super::Coordinator;

impl Coordinator {
    /// A lost rank's rows live in a replica's store, not on its spare's node (decision 0030): before
    /// the change goes out, point that replica at the spare's node for the rank, and tell the
    /// spare's daemon to expect the rank from it. The carrier's snapshot carries the rows; the
    /// spare's restore waits for its own store to hold the rank caught up, then reads it locally,
    /// before it acknowledges the change.
    ///
    /// When the spare's node is itself one of the rank's replicas, its store already holds the
    /// rows, and nothing is staged.
    pub(super) async fn stage_restores(&self, engine: &str, lost: &[u32]) {
        let addresses = self.replica_addresses.lock().await.clone();
        let staged: Vec<(u32, String, String)> = {
            let mut engines = self.engines.lock().await;
            let Some(view) = engines.get_mut(engine) else {
                return;
            };
            let mut out = Vec::new();
            for r in lost {
                let Some(spare_node) = view.spare_node_for(*r).cloned() else {
                    continue;
                };
                let held_at = view.expected_at.get(r).map(|(_, at)| at);
                if held_at.is_some_and(|at| at.contains(&spare_node)) {
                    // the spare's own store already takes the rank's rows: staging them from
                    // another replica would add nothing and point that store at a new source
                    // mid-stream, which it then waits on to catch up (measured: 1.48 s)
                    info!(
                        "{engine}: rank {r}'s rows are held on {spare_node}: its spare restores there"
                    );
                    continue;
                }
                let Some(replica) = held_at
                    .and_then(|at| {
                        at.iter()
                            .find(|n| **n != spare_node && addresses.contains_key(*n))
                    })
                    .cloned()
                else {
                    warn!(
                        "{engine}: rank {r} has no reachable replica to stage its rows from: its \
                         spare on {spare_node} restores from its own store"
                    );
                    continue;
                };
                view.restoring.insert(*r, replica.clone());
                out.push((*r, replica, spare_node));
            }
            out
        };
        for (r, replica, spare_node) in staged {
            let Some(target) = addresses.get(&spare_node).cloned() else {
                continue;
            };
            info!("{engine}: rank {r}'s rows staged from {replica}'s store onto {spare_node}");
            self.tell(
                &spare_node,
                coord_to_daemon::Kind::ExpectReplication(ExpectReplication {
                    engine: engine.to_owned(),
                    rank: r,
                    from_node: replica.clone(),
                    // the staged rows are the lost rank's last committed ones, from before this
                    // change's epoch: a staging moves no fence, or the spare's store refuses the
                    // very rows it is being given. The commit fences the rank at the new epoch
                    // when it releases the staging.
                    epoch: 0,
                }),
            )
            .await;
            self.tell(
                &replica,
                coord_to_daemon::Kind::ReplicateTargets(ReplicateTargets {
                    engine: engine.to_owned(),
                    rank: r,
                    targets: vec![target],
                }),
            )
            .await;
        }
    }

    /// End the staged replications of `lost` (decision 0030): the replica carries the rank
    /// nowhere, and the node it staged onto expects nobody. After a commit, when the spares have
    /// restored; after a failed delivery, when nobody will.
    pub(super) async fn unstage_restores(&self, engine: &str, lost: &[u32], epoch: u64) {
        let staged: Vec<(u32, String, Option<String>)> = {
            let mut engines = self.engines.lock().await;
            let Some(view) = engines.get_mut(engine) else {
                return;
            };
            lost.iter()
                .filter_map(|r| {
                    let replica = view.restoring.remove(r)?;
                    // the spare's node: the member after a commit, the pool's entry before one
                    let at = view
                        .members
                        .get(r)
                        .filter(|n| **n != replica)
                        .or_else(|| view.spare_node_for(*r))
                        .cloned();
                    Some((*r, replica, at))
                })
                .collect()
        };
        for (r, replica, at) in staged {
            self.tell(
                &replica,
                coord_to_daemon::Kind::ReplicateTargets(ReplicateTargets {
                    engine: engine.to_owned(),
                    rank: r,
                    targets: Vec::new(),
                }),
            )
            .await;
            if let Some(at) = at {
                self.tell(
                    &at,
                    coord_to_daemon::Kind::ExpectReplication(ExpectReplication {
                        engine: engine.to_owned(),
                        rank: r,
                        from_node: String::new(),
                        epoch,
                    }),
                )
                .await;
            }
        }
    }

    /// Tell every member's daemon, of each engine in `engines`, where its store replicates the
    /// rank's rows: the replica addresses of the engine's replica nodes (decision 0030: its own
    /// member nodes unless the policy names others), other than the rank's own node. Sent only
    /// when the targets changed since the last time.
    pub(super) async fn refresh_replication(&self, engines: &BTreeSet<String>) {
        let addresses = self.replica_addresses.lock().await.clone();
        let mut to_send: Vec<(String, ReplicateTargets)> = Vec::new();
        let mut to_expect: Vec<(String, ExpectReplication)> = Vec::new();
        {
            let mut views = self.engines.lock().await;
            for (name, view) in views.iter_mut().filter(|(name, _)| engines.contains(*name)) {
                let Some(spec) = self.engine_spec(name) else {
                    continue;
                };
                // siblings are the nodes holding members now (decision 0030), not as placed
                let member_nodes: BTreeSet<String> = view.members.values().cloned().collect();
                for (rank, node) in &view.members {
                    // a rank that moved: its old holder's store replicates it nowhere from now,
                    // and the new holder is told its targets whatever was sent before
                    if let Some(old) = view.replicating_from.get(rank)
                        && old != node
                    {
                        to_send.push((
                            old.clone(),
                            ReplicateTargets {
                                engine: name.clone(),
                                rank: *rank,
                                targets: Vec::new(),
                            },
                        ));
                        view.replication.remove(rank);
                    }
                    view.replicating_from.insert(*rank, node.clone());
                    // the rank's replica nodes (decision 0030) whose stores are reachable
                    let target_nodes: BTreeSet<String> =
                        policy::replica_targets(spec, node, &member_nodes)
                            .into_iter()
                            .filter(|n| n != node && addresses.contains_key(n))
                            .collect();
                    let targets: Vec<String> = target_nodes
                        .iter()
                        .filter_map(|n| addresses.get(n).cloned())
                        .collect();
                    // a rank is fenced at the epoch its holder took it, never at the engine's: a
                    // rank that did not move keeps committing at its own epoch, and fencing it
                    // at a later one it would be refused every step it sends after a change
                    let fence = view.changed_at.get(rank).copied().unwrap_or(0);
                    to_expect.extend(expectations(
                        &mut view.expected_at,
                        name,
                        *rank,
                        node,
                        target_nodes,
                        fence,
                    ));
                    // a store replicates nowhere until told: an empty first word is not sent
                    let sent = view.replication.get(rank);
                    if sent == Some(&targets) || (sent.is_none() && targets.is_empty()) {
                        view.replication.entry(*rank).or_default();
                        continue;
                    }
                    view.replication.insert(*rank, targets.clone());
                    to_send.push((
                        node.clone(),
                        ReplicateTargets {
                            engine: name.clone(),
                            rank: *rank,
                            targets,
                        },
                    ));
                }
            }
        }
        let daemons = self.daemons.lock().await;
        for (node, msg) in to_expect {
            if let Some(link) = daemons.get(&node) {
                let _ = link.tx.send(CoordToDaemon {
                    term: 0,
                    kind: Some(coord_to_daemon::Kind::ExpectReplication(msg)),
                });
            }
        }
        for (node, msg) in to_send {
            if let Some(link) = daemons.get(&node) {
                let _ = link.tx.send(CoordToDaemon {
                    term: 0,
                    kind: Some(coord_to_daemon::Kind::ReplicateTargets(msg)),
                });
            }
        }
    }
}

/// What each target's daemon is told about who `rank` of `engine` comes from (decision 0026):
/// every current target expects `holder`, a node that stopped being a target expects nobody,
/// and nothing when nothing changed since the last refresh.
pub(super) fn expectations(
    expected_at: &mut BTreeMap<u32, (String, BTreeSet<String>)>,
    engine: &str,
    rank: u32,
    holder: &str,
    target_nodes: BTreeSet<String>,
    epoch: u64,
) -> Vec<(String, ExpectReplication)> {
    let want = (holder.to_owned(), target_nodes);
    if expected_at.get(&rank) == Some(&want) {
        return Vec::new();
    }
    let before = expected_at
        .get(&rank)
        .map(|(_, at)| at.clone())
        .unwrap_or_default();
    let expect = |at: &String, from_node: String| {
        (
            at.clone(),
            ExpectReplication {
                engine: engine.to_owned(),
                rank,
                from_node,
                epoch,
            },
        )
    };
    let mut out: Vec<(String, ExpectReplication)> = before
        .difference(&want.1)
        .map(|gone| expect(gone, String::new()))
        .collect();
    out.extend(want.1.iter().map(|at| expect(at, holder.to_owned())));
    expected_at.insert(rank, want);
    out
}
