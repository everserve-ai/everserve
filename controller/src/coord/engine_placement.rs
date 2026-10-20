// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Where an engine runs, as its processes learn it (decision 0041). An engine whose policy states
//! only its shape has no file naming each rank's node and host: its processes take them from the
//! coordinator. Once every rank of the world is placed (by the process told to serve it, before
//! anything loads, or by its registered worker), the coordinator sends the engine's placement,
//! every rank and every standing spare with its node, device and host, to every daemon, which
//! hands it to every process of the engine on its node, those that connect later included; and
//! again whenever it changes: a commit, a spare standing or leaving.

use std::collections::BTreeMap;

use crate::policy;
use crate::proto::everserve::coord::v1::{CoordToDaemon, coord_to_daemon};
use crate::proto::everserve::signals::v1::{
    CompileCache, EnginePlacement, RankPlacement, SparePlacement,
};
use crate::proto::everserve::spec::v1::Assignment;

use super::Coordinator;
use super::view::EngineView;

impl Coordinator {
    /// Send `engine`'s placement to its audience when it changed and every rank of its world is
    /// placed.
    pub(super) async fn push_placement(&self, engine: &str) {
        let Some(spec) = self.engine_spec(engine) else {
            return;
        };
        let world = policy::world_size(spec);
        let assignments = self.assignments(engine);
        let stores = self.replica_addresses.lock().await.clone();
        let placement = {
            let mut engines = self.engines.lock().await;
            // at bring-up the assignments may be all there is of the engine
            let view = engines.entry(engine.to_owned()).or_default();
            let Some(placement) = placement_of(engine, view, &assignments, world, &stores) else {
                return;
            };
            if view.placement.as_ref() == Some(&placement) {
                return;
            }
            view.placement = Some(placement.clone());
            placement
        };
        let msg = CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::EnginePlacement(placement)),
        };
        // to the engine's audience, not only the nodes holding a rank: a process told to stand
        // may be the only one of the engine on its node, and it reached the coordinator from
        // there before it stands
        for tx in self.audience_of(engine).await.into_values() {
            let _ = tx.send(msg.clone());
        }
    }
}

/// `engine`'s placement now: each rank of its `world` by its registered worker, or else by the
/// process told to serve it, with its compile cache and the `stores` (by node) that hold it, and
/// every admitted standing spare; None while a rank is placed by neither.
pub(super) fn placement_of(
    engine: &str,
    view: &EngineView,
    assignments: &[Assignment],
    world: u32,
    stores: &BTreeMap<String, String>,
) -> Option<EnginePlacement> {
    let mut members = Vec::new();
    for rank in 0..world.max(view.world_size) {
        let placed = if let Some(node) = view.members.get(&rank) {
            let compile_cache = view.caches.get(&rank).cloned();
            RankPlacement {
                rank,
                node: node.clone(),
                device: view.devices.get(&rank).copied(),
                host: view.hosts.get(&rank).cloned().unwrap_or_default(),
                cache_holders: cache_holders(view, rank, node, compile_cache.as_ref(), stores),
                compile_cache,
            }
        } else {
            let a = assignments.iter().find(|a| a.rank == rank)?;
            RankPlacement {
                rank,
                node: a.node.clone(),
                device: None,
                host: a.host.clone(),
                compile_cache: None,
                cache_holders: Vec::new(),
            }
        };
        members.push(placed);
    }
    let spares = view
        .standing
        .iter()
        .filter(|((node, device), ev)| {
            if ev.any_rank {
                view.any_spares.contains(&(node.clone(), *device))
            } else {
                view.spares.get(&ev.rank) == Some(node)
            }
        })
        .map(|((node, device), ev)| SparePlacement {
            for_rank: (!ev.any_rank).then_some(ev.rank),
            node: node.clone(),
            device: *device,
            host: ev.host.clone(),
        })
        .collect();
    Some(EnginePlacement {
        engine: engine.to_owned(),
        epoch: view.committed_epoch,
        members,
        spares,
    })
}

/// The stores that hold rank `rank`'s compile cache `pair` (decision 0043): its serving worker's
/// node's, then the node's of a spare admitted for the rank on the same pair; a node whose store
/// takes no replicas holds nothing a peer can fetch.
fn cache_holders(
    view: &EngineView,
    rank: u32,
    node: &str,
    pair: Option<&CompileCache>,
    stores: &BTreeMap<String, String>,
) -> Vec<String> {
    let Some(pair) = pair else {
        return Vec::new();
    };
    let spare = view
        .spares
        .get(&rank)
        .filter(|_| view.spare_caches.get(&rank) == Some(pair));
    std::iter::once(node)
        .chain(spare.map(String::as_str))
        .filter_map(|n| stores.get(n).cloned())
        .collect()
}
