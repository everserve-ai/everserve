// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0

//! Who hears an engine: its audience, the daemons of nodes with a process of it or a place for
//! one. An engine's policy, placement and instance state go to its audience only, so what a
//! change costs is its engine's, not the cell's.
//!
//! A node joins an engine's audience in one of four ways:
//!
//! - its daemon's hello names the engine;
//! - the policy places a member, a spare or a replica of the engine on it;
//! - an assignment or the view puts a member or a spare of the engine on it;
//! - a process of the engine first reaches the coordinator from it, by asking what it is or by
//!   registering (a pool spare, a head or node started from a policy file, a launched spare).
//!
//! The first time a process of an engine reaches the coordinator from a node, that node's daemon
//! is sent the engine's facts before anything that answers the process, on the same link, so the
//! daemon holds them when the process needs them. Among them is the membership as it stands, once
//! the engine has committed a change: a node that joins the audience after a commit (a spare
//! standing on a node new to the engine) heard none, and its daemon would refuse the next change
//! as stale. A node leaves the audience it reached when
//! its link ends; its next hello names what it still holds.
//!
//! Without an audience every change goes to every daemon in the cell
//! (`bench/runs/2026-10-14-cell-s0`: at 1,250 nodes, each change sent to all 1,250).

use std::collections::{BTreeMap, BTreeSet};

use prost::Message;

use crate::proto::everserve::coord::v1::{CoordToDaemon, DaemonHello, coord_to_daemon};
use crate::proto::everserve::policy::v1::Policy;
use crate::proto::everserve::signals::v1::EnginePolicy;

use super::Coordinator;
use super::outbox::Outbox;

/// For each node, the engines the policy places a member, a spare or a replica of on it.
pub(super) fn placed_on(policy: &Policy) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for engine in &policy.engines {
        let nodes = engine
            .members
            .iter()
            .map(|m| &m.node)
            .chain(engine.spares.iter().map(|s| &s.node))
            .chain(engine.replica_nodes.iter());
        for node in nodes {
            out.entry(node.clone())
                .or_default()
                .insert(engine.name.clone());
        }
    }
    out
}

impl Coordinator {
    /// The linked daemons of `engine`'s audience, by node.
    pub(super) async fn audience_of(&self, engine: &str) -> BTreeMap<String, Outbox> {
        let mut nodes: BTreeSet<String> = self
            .assignments(engine)
            .into_iter()
            .map(|a| a.node)
            .collect();
        if let Some(spec) = self.engine_spec(engine) {
            nodes.extend(spec.members.iter().map(|m| m.node.clone()));
            nodes.extend(spec.spares.iter().map(|s| s.node.clone()));
            nodes.extend(spec.replica_nodes.iter().cloned());
        }
        if let Some(v) = self.engines.lock().await.get(engine) {
            nodes.extend(v.members.values().cloned());
            nodes.extend(v.spares.values().cloned());
            nodes.extend(v.any_spares.iter().map(|(n, _)| n.clone()));
            nodes.extend(v.reached.iter().cloned());
        }
        let daemons = self.daemons.lock().await;
        nodes
            .into_iter()
            .filter_map(|n| daemons.get(&n).map(|d| (n, d.tx.clone())))
            .collect()
    }

    /// A process of `engine` reached the coordinator from `node`: the first time, `node`'s
    /// daemon is sent the engine's facts, ahead of whatever answers the process.
    pub(super) async fn reached(&self, engine: &str, node: &str) {
        let first = self
            .engines
            .lock()
            .await
            .entry(engine.to_owned())
            .or_default()
            .reached
            .insert(node.to_owned());
        if first {
            self.send_facts(engine, node).await;
        }
    }

    /// `node`'s daemon linked with `hello`: it joins the audience of each engine its hello
    /// names, and is sent the facts of those and of the engines the policy places on it.
    pub(super) async fn link_facts(&self, node: &str, hello: &DaemonHello) {
        let mut engines: BTreeSet<String> =
            hello.engines.iter().map(|e| e.engine.clone()).collect();
        {
            let mut views = self.engines.lock().await;
            for engine in &engines {
                views
                    .entry(engine.clone())
                    .or_default()
                    .reached
                    .insert(node.to_owned());
            }
        }
        engines.extend(self.placed_on.get(node).into_iter().flatten().cloned());
        for engine in &engines {
            self.send_facts(engine, node).await;
        }
    }

    /// `node`'s link ended: it leaves every audience it reached.
    pub(super) async fn unreached(&self, node: &str) {
        for view in self.engines.lock().await.values_mut() {
            view.reached.remove(node);
        }
    }

    /// Send `node`'s daemon what is known of `engine`: its policy, placement and instance state,
    /// and, once a change has committed, the membership as it stands, in that order. A daemon at
    /// that epoch already takes nothing from the membership.
    async fn send_facts(&self, engine: &str, node: &str) {
        let Some(tx) = self.daemons.lock().await.get(node).map(|d| d.tx.clone()) else {
            return;
        };
        let (placement, state, standing) = self
            .engines
            .lock()
            .await
            .get(engine)
            .map(|v| {
                (
                    v.placement.clone(),
                    v.instance_state.clone(),
                    (v.committed_epoch > 0).then(|| super::link::standing_commit(engine, v)),
                )
            })
            .unwrap_or_default();
        let kinds = [
            self.policy_of(engine)
                .map(coord_to_daemon::Kind::EnginePolicy),
            placement.map(coord_to_daemon::Kind::EnginePlacement),
            state.map(coord_to_daemon::Kind::InstanceState),
            standing.map(coord_to_daemon::Kind::Committed),
        ];
        for kind in kinds.into_iter().flatten() {
            if tx
                .send(CoordToDaemon {
                    term: 0,
                    kind: Some(kind),
                })
                .is_err()
            {
                return;
            }
        }
    }

    /// `engine`'s policy as its processes read it: the cell's, narrowed to that engine.
    fn policy_of(&self, engine: &str) -> Option<EnginePolicy> {
        let spec = self.engine_spec(engine)?;
        let narrowed = Policy {
            engines: vec![spec.clone()],
            ..self.policy_frame.clone()
        };
        Some(EnginePolicy {
            engine: engine.to_owned(),
            policy: narrowed.encode_to_vec(),
        })
    }
}
