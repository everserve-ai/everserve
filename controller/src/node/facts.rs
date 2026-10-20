// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What the coordinator says of an engine for its processes (decision 0041): its policy, where it
//! runs, its instance's state, and the ids its pre-formed groups are formed with (decision 0035,
//! amended), which a process takes for the groups it spans. The daemon keeps the last of each per engine, hands each to
//! every process of the engine on its node as it comes, and all of them to a process as it
//! registers, its policy first; an undecided process, which has no queue of its own, follows them
//! through the engine's watch until it registers. A process takes the last of each it was given.

use tokio::sync::{mpsc, watch};

use std::collections::BTreeMap;

use crate::proto::everserve::signals::v1::{
    EnginePlacement, EnginePolicy, GroupId, InstanceState, Signal, signal,
};

use super::{Daemon, Outbound};

/// The last of each the coordinator said of one engine.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Facts {
    /// Its policy, narrowed to it.
    pub policy: Option<EnginePolicy>,
    /// Where it runs.
    pub placement: Option<EnginePlacement>,
    /// Its instance's state.
    pub state: Option<InstanceState>,
    /// Each group's newest id, by its scope and its spans encoded.
    pub ids: BTreeMap<(i32, Vec<u8>), GroupId>,
}

impl Facts {
    /// What `self` holds that `before` does not, as the signals that carry it, in the order a
    /// process is handed them: its policy first.
    pub fn since(&self, before: &Self) -> Vec<Signal> {
        let mut out = Vec::new();
        if self.policy != before.policy
            && let Some(p) = &self.policy
        {
            out.push(signal::Kind::EnginePolicy(p.clone()));
        }
        if self.placement != before.placement
            && let Some(p) = &self.placement
        {
            out.push(signal::Kind::EnginePlacement(p.clone()));
        }
        if self.state != before.state
            && let Some(s) = &self.state
        {
            out.push(signal::Kind::InstanceState(s.clone()));
        }
        for (key, id) in &self.ids {
            if before.ids.get(key) != Some(id) {
                out.push(signal::Kind::GroupId(id.clone()));
            }
        }
        out.into_iter().map(|k| Signal { kind: Some(k) }).collect()
    }
}

/// One thing the coordinator said of an engine.
pub(super) enum Fact {
    Policy(EnginePolicy),
    Placement(EnginePlacement),
    State(InstanceState),
    GroupId(GroupId),
}

impl Fact {
    fn engine(&self) -> &str {
        match self {
            Self::Policy(p) => &p.engine,
            Self::Placement(p) => &p.engine,
            Self::State(s) => &s.engine,
            Self::GroupId(id) => &id.engine,
        }
    }
}

impl Daemon {
    /// The watch of what the coordinator said of `engine`.
    pub(super) async fn facts(&self, engine: &str) -> watch::Sender<Facts> {
        self.facts
            .lock()
            .await
            .entry(engine.to_owned())
            .or_insert_with(|| watch::channel(Facts::default()).0)
            .clone()
    }

    /// The coordinator said `fact` of its engine: kept, and handed to every process of the
    /// engine registered on this node; the undecided ones follow the watch.
    pub(super) async fn engine_fact(&self, fact: Fact) {
        let engine = fact.engine().to_owned();
        let watch = self.facts(&engine).await;
        let before = watch.borrow().clone();
        watch.send_modify(|f| match fact {
            Fact::Policy(p) => f.policy = Some(p),
            Fact::Placement(p) => f.placement = Some(p),
            Fact::State(s) => f.state = Some(s),
            Fact::GroupId(id) => {
                let key = (
                    id.scope,
                    id.spans
                        .iter()
                        .flat_map(prost::Message::encode_length_delimited_to_vec)
                        .collect(),
                );
                // the newest a group's server made: one re-formed after a failure has a newer
                if f.ids
                    .get(&key)
                    .is_none_or(|kept| kept.generation <= id.generation)
                {
                    f.ids.insert(key, id);
                }
            }
        });
        let signals = watch.borrow().since(&before);
        let targets: Vec<mpsc::Sender<Outbound>> = {
            let engines = self.engines.lock().await;
            engines
                .get(&engine)
                .map(|e| {
                    e.workers
                        .values()
                        .chain(e.spares.values())
                        .chain(e.any_spares.values())
                        .chain(e.clients.iter())
                        .map(|c| c.tx.clone())
                        .collect()
                })
                .unwrap_or_default()
        };
        for tx in targets {
            for sig in &signals {
                let _ = tx.send((sig.clone(), None)).await;
            }
        }
    }

    /// Hand a process of `engine` everything the coordinator said of it, its policy first.
    pub(super) async fn facts_to(&self, engine: &str, to: &mpsc::Sender<Outbound>) {
        let facts = self.facts(engine).await.borrow().clone();
        for sig in facts.since(&Facts::default()) {
            let _ = to.send((sig, None)).await;
        }
    }

    /// The policy of `engine`, as the coordinator last gave it.
    pub(super) async fn policy_of(&self, engine: &str) -> Option<EnginePolicy> {
        self.facts(engine).await.borrow().policy.clone()
    }
}
