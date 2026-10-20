// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The coordinator's word passed on: a join to the rank's process or the clients, a notice
//! and a change to the processes they concern, replication targets and expectations to this
//! node's store, and a spare launched where the coordinator placed it (decision 0024).

use super::{Daemon, Outbound, spares, store};
use crate::proto::everserve::coord::v1::{
    ExpectReplication, ForwardChange, ForwardJoin, ForwardNotice, LaunchSpare, ReplicateTargets,
};
use crate::proto::everserve::signals::v1::{Signal, signal};
use tokio::sync::mpsc;
use tracing::info;
use tracing::warn;

impl Daemon {
    /// The coordinator placed a spare on this node: launch it and keep it (decision 0024). Held
    /// until the daemon serves, since the spare registers through the worker socket.
    pub(super) async fn launch_spare(&self, l: LaunchSpare) {
        let supervisor = self.spares.lock().await.clone();
        match supervisor {
            Some(s) => {
                if let Err(e) = s.launch(l).await {
                    warn!("spare not launched: {e}");
                }
            }
            None => self.pending_launches.lock().await.push(l),
        }
    }

    /// The spares this daemon launched.
    pub async fn launched(&self) -> Vec<spares::Launched> {
        match self.spares.lock().await.clone() {
            Some(s) => s.launched().await,
            None => Vec::new(),
        }
    }

    /// A join payload from the coordinator: to the process in the rank's position, or to the
    /// engine's clients.
    pub(super) async fn forward_join(&self, f: ForwardJoin) {
        let Some(join) = f.join else { return };
        let targets: Vec<mpsc::Sender<Outbound>> = {
            let engines = self.engines.lock().await;
            let Some(e) = engines.get(&f.engine) else {
                warn!("{}: join for an engine this node does not hold", f.engine);
                return;
            };
            if f.to_clients {
                e.clients.iter().map(|c| c.tx.clone()).collect()
            } else {
                e.workers
                    .get(&join.rank)
                    .or_else(|| e.spares.get(&join.rank))
                    .or_else(|| e.taking.get(&join.rank).and_then(|d| e.any_spares.get(d)))
                    .map(|c| vec![c.tx.clone()])
                    .unwrap_or_default()
            }
        };
        if targets.is_empty() {
            warn!(
                "{}: join for rank {} reached this node, which holds no process for it",
                f.engine, join.rank
            );
        }
        info!(
            "{}: join for rank {} at epoch {} delivered to {} process(es) on this node",
            f.engine,
            join.rank,
            join.epoch,
            targets.len()
        );
        let sig = Signal {
            kind: Some(signal::Kind::Join(join)),
        };
        for tx in targets {
            let _ = tx.send((sig.clone(), None)).await;
        }
    }

    /// The coordinator named where a rank's rows replicate: tell this node's store.
    pub(super) async fn replicate_targets(&self, t: ReplicateTargets) {
        let Some((socket, _)) = self.mirror.lock().await.clone() else {
            warn!(
                "{}: replication targets for rank {} but this daemon has no store socket",
                t.engine, t.rank
            );
            return;
        };
        match store::replicate(&socket, &self.node, &t.engine, t.rank, &t.targets).await {
            Ok(()) => info!(
                "{}: rank {} replicates to {:?}",
                t.engine, t.rank, t.targets
            ),
            Err(e) => warn!(
                "{}: rank {} replication targets refused: {e}",
                t.engine, t.rank
            ),
        }
    }

    /// The coordinator says this node's store takes a rank's rows from one node (decision 0026).
    pub(super) async fn expect_replication(&self, e: ExpectReplication) {
        let Some((socket, _)) = self.mirror.lock().await.clone() else {
            warn!(
                "{}: replication expected for rank {} but this daemon has no store socket",
                e.engine, e.rank
            );
            return;
        };
        match store::expect(&socket, &e.engine, e.rank, &e.from_node, e.epoch).await {
            Ok(()) if e.from_node.is_empty() => {
                info!("{}: rank {} replicates here from nobody", e.engine, e.rank);
            }
            Ok(()) => info!(
                "{}: rank {} replicates here from {}",
                e.engine, e.rank, e.from_node
            ),
            Err(err) => warn!(
                "{}: rank {} replication expectation refused: {err}",
                e.engine, e.rank
            ),
        }
    }

    /// A notice from the coordinator: tell every worker and the spare of the engine.
    pub(super) async fn forward_notice(&self, f: ForwardNotice) {
        let Some(notice) = f.notice else { return };
        let targets = {
            let engines = self.engines.lock().await;
            engines
                .get(&f.engine)
                .map(|e| {
                    e.workers
                        .values()
                        .chain(e.spares.values())
                        .map(|c| c.tx.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        if targets.is_empty() {
            warn!(
                "{}: notice for rank {} at {} ms: no process of the engine on this node",
                f.engine, notice.rank, notice.deadline_ms
            );
            return;
        }
        info!(
            "{}: notice for rank {} at {} ms forwarded to {} processes",
            f.engine,
            notice.rank,
            notice.deadline_ms,
            targets.len()
        );
        let sig = Signal {
            kind: Some(signal::Kind::Notice(notice)),
        };
        for tx in targets {
            let _ = tx.send((sig.clone(), None)).await;
        }
    }

    /// Pass a change to the processes of its departing ranks on this node (protocol 9): they
    /// leave at it, and nothing waits on their acknowledgement.
    pub(super) async fn forward_change(&self, f: ForwardChange) {
        let Some(change) = f.change else { return };
        let targets: Vec<(u32, mpsc::Sender<Outbound>)> = {
            let engines = self.engines.lock().await;
            let Some(e) = engines.get(&f.engine) else {
                warn!(
                    "{}: departing ranks {:?}: engine unknown here",
                    f.engine, f.to_ranks
                );
                return;
            };
            f.to_ranks
                .iter()
                .filter_map(|r| e.workers.get(r).map(|w| (*r, w.tx.clone())))
                .collect()
        };
        info!(
            "{}: epoch {}: ranks {:?} depart; the change forwarded to {} of them",
            f.engine,
            change.epoch,
            f.to_ranks,
            targets.len()
        );
        let sig = Signal {
            kind: Some(signal::Kind::Change(change)),
        };
        for (_, tx) in targets {
            let _ = tx.send((sig.clone(), None)).await;
        }
    }
}
