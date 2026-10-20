// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0

//! A spare's start-up replay, as the coordinator routes it (decision 0044). A spare's ask goes to
//! the engine's head; the head's offer goes to the one spare it names, and is taken from the
//! engine's head's node alone. The coordinator keeps nothing per ask: the offer names its asker.

use tracing::warn;

use crate::proto::everserve::coord::v1::{CoordToDaemon, coord_to_daemon};
use crate::proto::everserve::signals::v1::{ReplayAsk, ReplayOffer};

use super::Coordinator;

impl Coordinator {
    /// The node `engine`'s head runs on, if one stands.
    async fn head_node(&self, engine: &str) -> Option<String> {
        self.engines
            .lock()
            .await
            .get(engine)
            .and_then(|v| v.head.as_ref().map(|h| h.process.node.clone()))
    }

    /// A spare's ask, from `node`'s daemon: to the engine's head. With no head standing it is
    /// dropped, and the spare's bounded wait asks again.
    pub(super) async fn route_replay_ask(&self, node: &str, ask: ReplayAsk) {
        let Some(head) = self.head_node(&ask.engine).await else {
            warn!(
                "{}: a replay ask for rank {} from {node}, with no head standing; dropped",
                ask.engine, ask.rank
            );
            return;
        };
        let tx = self.daemons.lock().await.get(&head).map(|d| d.tx.clone());
        if let Some(tx) = tx {
            let _ = tx.send(CoordToDaemon {
                term: 0,
                kind: Some(coord_to_daemon::Kind::ReplayAsk(ask)),
            });
        }
    }

    /// The head's offer, from `node`'s daemon: to the asker's daemon, taken from the head's node
    /// only, since only the core knows what it sent a rank.
    pub(super) async fn route_replay_offer(&self, node: &str, offer: ReplayOffer) {
        if self.head_node(&offer.engine).await.as_deref() != Some(node) {
            warn!(
                "{}: a replay offer for rank {} from {node}, which is not the head's node; dropped",
                offer.engine, offer.rank
            );
            return;
        }
        let Some(to) = offer.to.as_ref().map(|t| t.node.clone()) else {
            return;
        };
        let tx = self.daemons.lock().await.get(&to).map(|d| d.tx.clone());
        if let Some(tx) = tx {
            let _ = tx.send(CoordToDaemon {
                term: 0,
                kind: Some(coord_to_daemon::Kind::ReplayOffer(offer)),
            });
        }
    }
}
