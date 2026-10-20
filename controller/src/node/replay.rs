// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0

//! A spare's start-up replay, as the daemon carries it (decision 0044). A spare asks the engine's
//! core for what the core sent its rank at its start, so that it builds as that rank's worker was
//! built rather than deriving any of it. It asks on a connection of its own, which registers
//! nothing: the daemon names it (its node, this daemon's instance and the connection's serial),
//! passes the ask up, hands an ask from the coordinator to the engine's head here, passes the
//! head's offer up, and delivers the offer on the asking connection alone.

use tokio::io::{ReadHalf, WriteHalf};
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tracing::{info, warn};

use everserve_wire::frame as wire;

use crate::proto::everserve::coord::v1::{DaemonToCoord, daemon_to_coord};
use crate::proto::everserve::signals::v1::{ProcessRef, ReplayAsk, ReplayOffer, Signal, signal};

use super::Daemon;

impl Daemon {
    /// A process on connection `id` asks for its rank's replay: named, and passed up.
    pub(super) async fn replay_ask_up(&self, id: u64, mut ask: ReplayAsk) {
        ask.asker = Some(ProcessRef {
            node: self.node.clone(),
            instance: self.instance,
            serial: id,
        });
        info!(
            "{}: rank {}'s spare on connection {id} asks for its replay",
            ask.engine, ask.rank
        );
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::ReplayAsk(ask)),
            },
            false,
        )
        .await;
    }

    /// A connection whose first frame asks for a replay (a spare that has not registered: a
    /// replay feeds its build): its ask goes up naming this connection, and the offer comes back
    /// on it. It registers nothing; the process closes it once it has its offer, and anything it
    /// sends ends it, as any connection that does not register is ended.
    pub(super) async fn serve_replay_ask(
        &self,
        id: u64,
        ask: ReplayAsk,
        rd: &mut wire::FrameReader<ReadHalf<UnixStream>>,
        wr: &mut WriteHalf<UnixStream>,
    ) -> Result<(), wire::WireError> {
        let (tx, mut rx) = mpsc::channel(4);
        self.replay_askers.lock().await.insert(id, tx);
        self.replay_ask_up(id, ask).await;
        let served = async {
            loop {
                tokio::select! {
                    Some(offer) = rx.recv() => wire::send(wr, &offer).await?,
                    frame = rd.recv::<Signal>() => {
                        if frame.is_ok() {
                            warn!("connection {id} asked for a replay and then sent more; closing");
                        }
                        return Ok(());
                    }
                }
            }
        }
        .await;
        self.replay_askers.lock().await.remove(&id);
        served
    }

    /// The engine's head on connection `id` answers one spare: passed up. An offer from any other
    /// connection is dropped, since only the core knows what it sent a rank.
    pub(super) async fn replay_offer_up(&self, engine: &str, id: u64, offer: ReplayOffer) {
        if !self.is_head(engine, id).await {
            warn!("{engine}: a replay offer from connection {id}, which is not the head; dropped");
            return;
        }
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::ReplayOffer(offer)),
            },
            false,
        )
        .await;
    }

    /// The coordinator hands the engine's head a spare's ask.
    pub(super) async fn replay_ask_to_head(&self, ask: ReplayAsk) {
        let head = {
            let engines = self.engines.lock().await;
            engines.get(&ask.engine).and_then(|e| {
                let id = e.head.as_ref()?.id;
                e.clients.iter().find(|c| c.id == id).map(|c| c.tx.clone())
            })
        };
        let Some(tx) = head else {
            warn!(
                "{}: a replay ask for rank {}, and the engine's head is not on this node",
                ask.engine, ask.rank
            );
            return;
        };
        let _ = tx
            .send((
                Signal {
                    kind: Some(signal::Kind::ReplayAsk(ask)),
                },
                None,
            ))
            .await;
    }

    /// The coordinator delivers the head's offer to the spare that asked, on this node. An offer
    /// for another instance of this daemon (one restarted since the ask), or for a connection
    /// that has ended, is dropped: the spare's bounded wait asks again.
    pub(super) async fn replay_offer_to_asker(&self, offer: ReplayOffer) {
        let Some(to) = offer.to.clone() else {
            return;
        };
        if to.instance != self.instance {
            info!(
                "{}: a replay offer for rank {} named another instance of this daemon; dropped",
                offer.engine, offer.rank
            );
            return;
        }
        let asking = self.replay_askers.lock().await.get(&to.serial).cloned();
        let Some(tx) = asking else {
            info!(
                "{}: a replay offer for rank {} reached no asking connection {} here; dropped",
                offer.engine, offer.rank, to.serial
            );
            return;
        };
        let _ = tx
            .send(Signal {
                kind: Some(signal::Kind::ReplayOffer(offer)),
            })
            .await;
    }
}
