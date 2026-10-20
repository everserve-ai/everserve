// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What the operator API and a scheduler read: readiness, the nodes, the topology, and the
//! requests for stacks and a drain (layer 7, decision 0034).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::oneshot;
use tracing::info;

use crate::clock::now_ms;
use crate::policy;
use crate::proto::everserve::coord::v1::{CoordToDaemon, Stacks, coord_to_daemon};
use crate::proto::everserve::signals::v1::Outcome;
use crate::readiness::{
    AnySpareReadiness, ChangeReadiness, CompileCacheReadiness, DownReadiness, EngineReadiness,
    HeadReadiness, InstanceState, MemberReadiness, NodeReadiness, Readiness, RefusedSpareReadiness,
    SlotReadiness,
};

use super::nodes::{NodeSpecError, NodeWant};
use super::transition::outcome_label;
use super::view::{EngineView, process_view};
use super::{Coordinator, Reply, STACKS_TIMEOUT, StacksAt, StacksError, StacksTaken};
use crate::proto::everserve::spec::v1::{ResourceSpec, resource_spec};
use crate::status::Event;
use crate::topology::Topology;

/// The spares of `view` standing for any position, and those taking one; a standing spare on a
/// cordoned node is shown as such.
fn any_spare_views(view: &EngineView, cordoned: &BTreeSet<String>) -> Vec<AnySpareReadiness> {
    view.any_spares
        .iter()
        .map(|at| ("standing", at))
        .chain(view.taking.values().map(|at| ("taking", at)))
        .map(|(state, (node, device))| AnySpareReadiness {
            node: node.clone(),
            device: *device,
            state: if state == "standing" && cordoned.contains(node) {
                "cordoned"
            } else {
                state
            }
            .to_owned(),
            process: process_view(view.processes.any.get(&(node.clone(), *device))),
        })
        .collect()
}

impl Coordinator {
    /// The policy this coordinator runs under.
    #[must_use]
    pub fn policy(&self) -> &crate::proto::everserve::policy::v1::Policy {
        &self.policy
    }

    /// Every node the coordinator knows: those whose daemons are linked, those with a spec,
    /// and those holding a member of an engine, in name order.
    pub async fn node_views(&self) -> Vec<NodeView> {
        let linked: BTreeSet<String> = self.daemons().await.into_iter().collect();
        let specs = self.specs.cache.lock().await.clone();
        let specs: BTreeMap<String, ResourceSpec> = specs
            .into_values()
            .filter_map(|r| match &r.kind {
                Some(resource_spec::Kind::Node(n)) => Some((n.node.clone(), r.clone())),
                _ => None,
            })
            .collect();
        let holding: BTreeSet<String> = self
            .engines
            .lock()
            .await
            .values()
            .flat_map(|v| v.members.values().cloned())
            .collect();
        let names: BTreeSet<String> = linked
            .iter()
            .chain(specs.keys())
            .chain(&holding)
            .cloned()
            .collect();
        let drains = self.specs.drains.lock().await.clone();
        let mut conditions = BTreeMap::new();
        for node in &names {
            conditions.insert(node.clone(), self.node_status(node).await.conditions());
        }
        let topo = self.topology.lock().await;
        // grouped once: a walk of every device per node made the list cost nodes x devices
        let mut devices: BTreeMap<&str, Vec<_>> = BTreeMap::new();
        for d in topo.devices() {
            devices.entry(d.node.as_str()).or_default().push(d.clone());
        }
        let addresses = self.replica_addresses.lock().await;
        names
            .into_iter()
            .map(|node| NodeView {
                linked: linked.contains(&node),
                devices: devices.remove(node.as_str()).unwrap_or_default(),
                replica_address: addresses.get(&node).cloned(),
                spec: specs.get(&node).cloned(),
                drain: drains.get(&node).map(|d| d.label().to_owned()),
                conditions: conditions.remove(&node).unwrap_or_default(),
                node,
            })
            .collect()
    }

    /// What a scheduler reads before it reclaims a node (docs/MULTINODE.md section 8).
    pub async fn readiness(&self) -> Readiness {
        let changes_in_flight = *self.in_flight.lock().await;
        let cordoned = self.cordoned().await;
        let engines = self.engines.lock().await;
        let topo = self.topology.lock().await;
        let mut out = Vec::new();
        for (name, view) in engines.iter() {
            let members = view
                .members
                .iter()
                .map(|(rank, node)| MemberReadiness {
                    rank: *rank,
                    node: node.clone(),
                    class: view.classes.get(rank).cloned(),
                    replicates_to: view.replication.get(rank).cloned().unwrap_or_default(),
                    compile_cache: view.caches.get(rank).map(CompileCacheReadiness::from_wire),
                    process: process_view(
                        view.processes.members.get(rank).filter(|p| &p.node == node),
                    ),
                })
                .collect();
            let mut by_node: BTreeMap<String, Vec<u32>> = BTreeMap::new();
            for (rank, node) in &view.members {
                by_node.entry(node.clone()).or_default().push(*rank);
            }
            let nodes = by_node
                .into_iter()
                .map(|(node, ranks)| self.node_readiness(name, view, node, ranks, &cordoned))
                .collect();
            let world = self
                .engine_spec(name)
                .map_or(view.world_size, policy::world_size);
            let state = super::conditions::instance_state(name, view, world);
            out.push(EngineReadiness {
                engine: name.clone(),
                state: InstanceState::from_phase(state.phase()),
                epoch: view.committed_epoch,
                change: view.change.as_ref().map(|(epoch, lost)| ChangeReadiness {
                    epoch: *epoch,
                    lost: lost.clone(),
                }),
                world_size: view.world_size,
                members,
                spares: self.spare_views(name, view, &topo, &cordoned),
                any_spares: any_spare_views(view, &cordoned),
                lost: view.lost.clone(),
                nodes,
                slots: view
                    .slots
                    .iter()
                    .map(|(n, s)| (n.clone(), SlotReadiness::from_wire(s)))
                    .collect(),
                guarantee: if self.exact(name) {
                    "exact"
                } else {
                    "continuous"
                }
                .to_owned(),
                compile_key: view.compile_key.clone(),
                compile_cache_consistent: view.compile_cache_consistent(),
                head: view.head.as_ref().map(head_readiness),
                last_change: view.last_change.clone(),
                refused_spares: view
                    .refused_spares
                    .iter()
                    .map(|((node, device), (rank, reason))| RefusedSpareReadiness {
                        node: node.clone(),
                        device: *device,
                        rank: *rank,
                        reason: reason.clone(),
                    })
                    .collect(),
                head_conflict: view.head_conflict.as_ref().map(head_readiness),
                conditions: view.status.conditions(),
                cover: Self::cover_readiness(view),
                down: view.down.as_ref().map(|d| DownReadiness {
                    at_ms: d.at_ms,
                    cause: d.cause.to_owned(),
                    detail: d.detail.clone(),
                }),
            });
        }
        Readiness {
            engines: out,
            changes_in_flight,
        }
    }

    /// Whether `node` can be reclaimed from `engine` now, and how.
    pub(super) fn node_readiness(
        &self,
        engine: &str,
        view: &EngineView,
        node: String,
        ranks: Vec<u32>,
        cordoned: &BTreeSet<String>,
    ) -> NodeReadiness {
        let outcome = self.expected_outcome(engine, view, &ranks, cordoned);
        let max = usize::try_from(self.policy.max_concurrent_losses).unwrap_or(usize::MAX);
        let why = if outcome == Outcome::Rebuild {
            "no spare stands for every rank on the node and SHRINK is not allowed: the engine \
             would rebuild"
                .to_owned()
        } else if view.lost.len() + ranks.len() > max {
            format!(
                "{} ranks already lost; {} more exceeds the policy's {max} concurrent losses",
                view.lost.len(),
                ranks.len()
            )
        } else {
            String::new()
        };
        NodeReadiness {
            node,
            ranks,
            reclaimable: why.is_empty(),
            outcome: outcome_label(outcome).to_owned(),
            why,
            min_notice_ms: self.policy.notice_min_ms,
            expected_pause_ms: view.last_pause_ms.get(outcome_label(outcome)).copied(),
        }
    }

    /// Ask the process at `at` of `engine` to write every thread's stack to its own log, through
    /// the daemon of the node that holds it; the process goes on serving.
    ///
    /// # Errors
    ///
    /// [`StacksError::NotFound`] when no process of the engine stands there;
    /// [`StacksError::Node`] when that node's daemon is not linked, does not answer, or could not
    /// signal the process.
    pub async fn stacks(&self, engine: &str, at: StacksAt) -> Result<StacksTaken, StacksError> {
        let (node, rank, device, head) = {
            let engines = self.engines.lock().await;
            let view = engines
                .get(engine)
                .ok_or_else(|| StacksError::NotFound(format!("no engine {engine}")))?;
            match at {
                StacksAt::Rank(r) => {
                    let node = view
                        .members
                        .get(&r)
                        .or_else(|| view.spare_node_for(r))
                        .cloned()
                        .ok_or_else(|| {
                            StacksError::NotFound(format!("no process holds rank {r} of {engine}"))
                        })?;
                    (node, Some(r), None, false)
                }
                StacksAt::Spare { node, device } => {
                    if !view.any_spares.contains(&(node.clone(), device))
                        && !view.taking.values().any(|t| *t == (node.clone(), device))
                    {
                        return Err(StacksError::NotFound(format!(
                            "no spare of {engine} on {node} device {device}"
                        )));
                    }
                    (node, None, Some(device), false)
                }
                StacksAt::Head => {
                    let node = view
                        .head
                        .as_ref()
                        .map(|h| h.process.node.clone())
                        .ok_or_else(|| {
                            StacksError::NotFound(format!("no head of {engine} is registered"))
                        })?;
                    (node, None, None, true)
                }
            }
        };
        let request_id = self.next_request.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        let link_closed = |node: &str| StacksError::Node {
            node: node.to_owned(),
            reason: "not linked".into(),
        };
        {
            let daemons = self.daemons.lock().await;
            let link = daemons.get(&node).ok_or_else(|| link_closed(&node))?;
            link.pending.lock().await.insert(request_id, tx);
            let ask = CoordToDaemon {
                term: 0,
                kind: Some(coord_to_daemon::Kind::Stacks(Stacks {
                    request_id,
                    engine: engine.to_owned(),
                    rank,
                    device,
                    head,
                })),
            };
            link.tx.send(ask).map_err(|_| link_closed(&node))?;
        }
        let at_ms = now_ms();
        let reply = tokio::time::timeout(STACKS_TIMEOUT, rx)
            .await
            .map_err(|_| StacksError::Node {
                node: node.clone(),
                reason: "no answer".into(),
            })?
            .map_err(|_| link_closed(&node))?;
        let Reply::Stacks(sent) = reply else {
            return Err(StacksError::Node {
                node,
                reason: "answered a stacks request with something else".into(),
            });
        };
        if !sent.error.is_empty() {
            return Err(StacksError::Node {
                node,
                reason: sent.error,
            });
        }
        info!(
            "{engine}: stacks of a process on {node} asked (pid {})",
            sent.pid
        );
        Ok(StacksTaken {
            node,
            pid: (sent.pid > 0).then_some(sent.pid),
            answered: sent.answered,
            stacks: sent.text,
            at_ms,
        })
    }

    /// Drain `node` by `deadline_ms`, as `writer` asked for `reason`: a notice's or a Node
    /// label's drain, written as the node's spec as an operator's is (decision 0042), cordoned
    /// and drained; its reconcile sends a notice to every rank on it and sets one change per
    /// engine for the set of its ranks at the deadline, or leaves the drain waiting while other
    /// drains would take more positions at once than the policy allows.
    ///
    /// # Errors
    ///
    /// As [`Coordinator::put_node_spec`].
    pub async fn drain(
        self: &Arc<Self>,
        node: &str,
        deadline_ms: i64,
        writer: &str,
        reason: &str,
    ) -> Result<ResourceSpec, NodeSpecError> {
        let want = NodeWant {
            unschedulable: true,
            drain_by_ms: deadline_ms,
            reason: reason.to_owned(),
        };
        self.put_node_spec(node, writer, want, None).await
    }

    /// The events kept for `engine`, oldest first; None for an engine this coordinator does not
    /// know.
    pub async fn engine_events(&self, engine: &str) -> Option<Vec<Event>> {
        self.engines
            .lock()
            .await
            .get(engine)
            .map(|v| v.status.events())
    }

    /// The topology as reported so far.
    pub async fn topology(&self) -> Topology {
        self.topology.lock().await.clone()
    }
}

/// A node as the coordinator knows it: its spec as the store holds it, and what it observes.
#[derive(Debug, Clone)]
pub struct NodeView {
    /// The node.
    pub node: String,
    /// Its daemon is linked to this coordinator.
    pub linked: bool,
    /// Its devices, as its daemon reported them.
    pub devices: Vec<crate::topology::Device>,
    /// Where its store takes replicas; absent when it takes none.
    pub replica_address: Option<String>,
    /// Its spec, with its generation and writer; absent until one was written.
    pub spec: Option<ResourceSpec>,
    /// Where its drain stands: `scheduled`, `waiting` or `blocked`.
    pub drain: Option<String>,
    /// Its conditions (decision 0036).
    pub conditions: Vec<crate::status::Condition>,
}

/// The engine's head, as readiness shows it.
fn head_readiness(h: &super::view::HeadProcess) -> HeadReadiness {
    HeadReadiness {
        node: h.process.node.clone(),
        pid: h.process.pid,
        launcher_pid: h.launcher_pid,
        registered_at_ms: h.process.registered_at_ms,
        host: h.host.clone(),
    }
}
