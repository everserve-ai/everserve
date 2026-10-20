// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The worker links: each connection served from its registration to its close, recorded
//! under its role, its signals sent up and the signals for it delivered, and what departed
//! said when it closes.

use super::{Conn, Daemon, Head, Outbound, Presence};
use crate::clock::now_us;
use crate::proto::everserve::coord::v1::{
    DaemonToCoord, JoinReport, LostReport, NoticeReport, WorkerEvent, cover_report,
    daemon_to_coord, worker_event,
};
use crate::proto::everserve::signals::v1::{Ack, Register, ReleaseLeaked, Role, Signal, signal};
use everserve_wire::frame as wire;
use std::sync::Arc;
use tokio::io::{ReadHalf, WriteHalf};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tracing::info;
use tracing::warn;

/// What a closing connection took with it: the event that says so, the engine's world size as
/// the event reports it, and the process as a departure names it.
struct Departure {
    kind: worker_event::Kind,
    world_size: u32,
    who: Presence,
}

/// Which connection a served link is: its engine, rank, role and id.
#[derive(Clone, Copy)]
struct Link<'a> {
    engine: &'a str,
    rank: u32,
    role: Role,
    id: u64,
}

/// The process at the other end of `stream`, as the system says, never as the process says of
/// itself; None when the system does not say.
fn peer_pid(stream: &UnixStream) -> Option<u32> {
    stream
        .peer_cred()
        .ok()
        .and_then(|c| c.pid())
        .and_then(|p| u32::try_from(p).ok())
        // the kernel gives 0 for a peer outside this process's pid namespace: another pod's
        .filter(|p| *p > 0)
}

/// The registration a connection opens with, or None with the reason logged: not a
/// registration, or one from another protocol (decision 0027).
pub(super) fn registration(first: Signal) -> Option<Register> {
    let Some(signal::Kind::Register(reg)) = first.kind else {
        warn!("first message was not a registration; closing");
        return None;
    };
    if let Err(e) = everserve_wire::check_protocol(reg.protocol, reg.protocol_oldest) {
        warn!("{}: rank {}: {e}; closing", reg.engine, reg.rank);
        return None;
    }
    Some(reg)
}

impl Daemon {
    pub(super) async fn handle(self: Arc<Self>, stream: UnixStream) -> Result<(), wire::WireError> {
        let pid = peer_pid(&stream);
        let (rd, mut wr) = tokio::io::split(stream);
        let mut rd = wire::FrameReader::new(rd);
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Some(reg) = self.served_as(id, pid, &mut rd, &mut wr).await? else {
            return Ok(());
        };
        let role = Role::try_from(reg.role).unwrap_or(Role::Worker);
        let (tx, mut rx) = mpsc::channel::<Outbound>(8);
        let device = reg.member.as_ref().map_or(0, |m| m.device);
        let to_process = tx.clone();
        let event = self.admit(&reg, role, Conn { id, tx, pid }, device).await;
        self.facts_to(&reg.engine, &to_process).await;
        if matches!(role, Role::Worker | Role::Spare) {
            self.plan_to(&reg.engine, &to_process).await;
        }
        self.head_registered(&reg, role, id).await;
        if matches!(role, Role::Client | Role::Head | Role::Spare)
            && let Some(slot) = reg.slot.clone()
        {
            self.report_slot(&reg.engine, slot).await;
        }
        if let Some((kind, world_size)) = event {
            let epoch = self
                .engines
                .lock()
                .await
                .get(&reg.engine)
                .map_or(0, |e| e.epoch);
            self.worker_event(
                &reg.engine,
                kind,
                world_size,
                Presence {
                    rank: reg.rank,
                    device,
                    compile_cache: reg.compile_cache.clone(),
                    any_rank: reg.any_rank,
                    pid,
                    epoch,
                    identity: reg.identity.clone(),
                    conn: Some(id),
                    host: reg.host.clone(),
                    goodbye: None,
                },
            )
            .await;
        }
        self.republish().await;
        let result = Arc::clone(&self)
            .serve_conn(
                Link {
                    engine: &reg.engine,
                    rank: reg.rank,
                    role,
                    id,
                },
                &mut rd,
                &mut wr,
                &mut rx,
            )
            .await;
        self.end_head(&reg.engine, id).await;
        self.release_held(&reg.engine, id).await;
        if let Some(gone) = self.departure(&reg.engine, id, device).await {
            self.worker_event(&reg.engine, gone.kind, gone.world_size, gone.who)
                .await;
        }
        result
    }

    /// Record a registered connection under its role; the worker event to send up, if any.
    async fn admit(
        &self,
        reg: &Register,
        role: Role,
        conn: Conn,
        device: u32,
    ) -> Option<(worker_event::Kind, u32)> {
        let (id, pid) = (conn.id, conn.pid);
        let mut engines = self.engines.lock().await;
        let engine = engines.entry(reg.engine.clone()).or_default();
        match role {
            Role::Worker => {
                engine.workers.insert(reg.rank, conn);
                engine.departed.remove(&reg.rank);
                engine.world_size = engine.world_size.max(reg.rank + 1);
                info!("rank {} of {} registered", reg.rank, reg.engine);
                self.metrics
                    .workers_registered
                    .with_label_values(&[&reg.engine])
                    .set(i64::try_from(engine.workers.len()).unwrap_or(i64::MAX));
                Some((worker_event::Kind::WorkerRegistered, engine.world_size))
            }
            Role::Spare => {
                if reg.any_rank {
                    engine.any_spares.insert(device, conn);
                    info!(
                        "spare for any rank of {} registered on device {device}",
                        reg.engine
                    );
                } else {
                    engine.spares.insert(reg.rank, conn);
                    info!("spare for rank {} of {} registered", reg.rank, reg.engine);
                }
                if let Some(slot) = reg.slot.clone() {
                    // a spare on a spare node decided the node's variant itself (0024)
                    engine.slot = Some(slot);
                }
                Some((worker_event::Kind::SpareRegistered, engine.world_size))
            }
            // taken by `decide` before any admission, which admits only what it was told to be
            Role::Undecided => {
                warn!(
                    "{}: rank {}: an undecided registration reached admission; ignored",
                    reg.engine, reg.rank
                );
                None
            }
            Role::Client => {
                engine.clients.push(conn);
                info!("client of {} registered", reg.engine);
                if let Some(slot) = reg.slot.clone() {
                    engine.slot = Some(slot);
                }
                None
            }
            Role::Head => {
                engine.clients.push(conn);
                engine.head = Some(Head {
                    id,
                    launcher_pid: reg.launcher_pid,
                    goodbye: None,
                    pid,
                    identity: reg.identity.clone(),
                    host: reg.host.clone(),
                });
                info!(
                    "head of {} registered (pid {}, launcher {})",
                    reg.engine,
                    pid.unwrap_or(0),
                    reg.launcher_pid
                );
                if let Some(slot) = reg.slot.clone() {
                    engine.slot = Some(slot);
                }
                None
            }
        }
    }

    /// Decision 0025, amended: a process on connection `id` left a release behind; up to the
    /// coordinator, named by this connection, so it retires the process that holds it.
    async fn release_leaked(&self, engine: &str, rank: u32, id: u64, leak: ReleaseLeaked) {
        warn!(
            "{engine}: rank {rank} left release {} behind after {} ms",
            leak.release, leak.run_ms
        );
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::ReleaseLeaked(
                    crate::proto::everserve::coord::v1::ReleaseLeakedAt {
                        connection: Some(crate::proto::everserve::coord::v1::ConnectionRef {
                            instance: self.instance,
                            serial: id,
                            pid: 0,
                        }),
                        leak: Some(leak),
                    },
                )),
            },
            true,
        )
        .await;
    }

    /// Connection `id` of `engine` closed, on `device`: forget what it registered and say what
    /// departed.
    async fn departure(&self, engine: &str, id: u64, device: u32) -> Option<Departure> {
        let mut engines = self.engines.lock().await;
        let state = engines.get_mut(engine)?;
        let goodbye = state.goodbyes.remove(&id);
        state.standing.remove(&id);
        state.serving.remove(&id);
        let epoch = state.epoch;
        // the event, the rank it names, and whether the process stood for any position
        let mut event: Option<(worker_event::Kind, u32, bool)> = None;
        let dropped: Vec<u32> = state
            .workers
            .iter()
            .filter(|(_, c)| c.id == id)
            .map(|(r, _)| *r)
            .collect();
        state.workers.retain(|_, c| c.id != id);
        for r in &dropped {
            state.departed.insert(*r);
            event = Some((worker_event::Kind::WorkerDeparted, *r, false));
        }
        state.clients.retain(|c| c.id != id);
        let gone: Vec<u32> = state
            .spares
            .iter()
            .filter(|(_, c)| c.id == id)
            .map(|(r, _)| *r)
            .collect();
        for r in gone {
            state.spares.remove(&r);
            event = Some((worker_event::Kind::SpareDeparted, r, false));
        }
        let gone_any: Vec<u32> = state
            .any_spares
            .iter()
            .filter(|(_, c)| c.id == id)
            .map(|(d, _)| *d)
            .collect();
        for d in gone_any {
            state.any_spares.remove(&d);
            state.taking.retain(|_, dev| *dev != d);
            event = Some((worker_event::Kind::SpareDeparted, 0, true));
        }
        self.metrics
            .workers_registered
            .with_label_values(&[engine])
            .set(i64::try_from(state.workers.len()).unwrap_or(i64::MAX));
        event.map(|(kind, rank, any_rank)| Departure {
            kind,
            world_size: state.world_size,
            who: Presence {
                rank,
                device,
                compile_cache: None,
                any_rank,
                pid: None,
                epoch,
                identity: None,
                conn: None,
                host: String::new(),
                goodbye,
            },
        })
    }

    /// Tell the coordinator a worker or spare registered or departed; a registration carries the
    /// compile cache the process reported (decision 0029).
    async fn worker_event(
        &self,
        engine: &str,
        kind: worker_event::Kind,
        world_size: u32,
        who: Presence,
    ) {
        let ev = WorkerEvent {
            engine: engine.to_owned(),
            kind: kind as i32,
            rank: who.rank,
            world_size,
            device: who.device,
            compile_cache: who.compile_cache,
            any_rank: who.any_rank,
            pid: who.pid.unwrap_or(0),
            epoch: who.epoch,
            identity: who.identity,
            host: who.host,
            cause: who.goodbye.as_ref().map_or(0, |g| g.cause),
            detail: who.goodbye.map(|g| g.detail).unwrap_or_default(),
            ..WorkerEvent::default()
        };
        if let Some(id) = who.conn
            && let Some(e) = self.engines.lock().await.get_mut(engine)
        {
            match kind {
                worker_event::Kind::SpareRegistered => {
                    e.standing.insert(id, ev.clone());
                }
                worker_event::Kind::WorkerRegistered => {
                    e.serving.insert(id, ev.clone());
                }
                _ => {}
            }
        }
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::WorkerEvent(ev)),
            },
            false,
        )
        .await;
    }

    /// A join of `role` for `rank`, sent up: a client's or the head's to the rank's process, a
    /// process's to the engine's clients.
    async fn join_up(
        &self,
        engine: &str,
        rank: u32,
        role: Role,
        join: crate::proto::everserve::signals::v1::Join,
    ) {
        let from_client = matches!(role, Role::Client | Role::Head);
        info!(
            "{engine}: join for rank {} at epoch {} from rank {rank} sent up",
            join.rank, join.epoch
        );
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::JoinReport(JoinReport {
                    engine: engine.to_owned(),
                    join: Some(join),
                    from_client,
                })),
            },
            false,
        )
        .await;
    }

    /// Serve one connection: deliver its outbound signals, route acknowledgements back to the
    /// delivery that awaits them, and send up what it reports (a lost rank, a notice).
    async fn serve_conn(
        self: Arc<Self>,
        link: Link<'_>,
        rd: &mut wire::FrameReader<ReadHalf<UnixStream>>,
        wr: &mut WriteHalf<UnixStream>,
        rx: &mut mpsc::Receiver<Outbound>,
    ) -> Result<(), wire::WireError> {
        let Link {
            engine,
            rank,
            role,
            id,
        } = link;
        let mut waiting: Option<oneshot::Sender<Ack>> = None;
        loop {
            tokio::select! {
                frame = rd.recv::<Signal>() => {
                    let sig = frame?;
                    match sig.kind {
                        Some(signal::Kind::Ack(ack)) => {
                            // by kind, not by position: a report can arrive between a Change
                            // and its acknowledgement (decision 0021)
                            if let Some(reply) = waiting.take() {
                                let _ = reply.send(Ack { received_at_us: now_us(), ..ack });
                            } else {
                                warn!(
                                    "{engine}: rank {rank} acknowledged epoch {} with no change pending",
                                    ack.epoch
                                );
                            }
                        }
                        Some(signal::Kind::RankLost(lost)) => {
                            info!("{engine}: ranks {:?} reported lost by rank {rank}: {}", lost.ranks, lost.detail);
                            self.send_up(DaemonToCoord {
                                kind: Some(daemon_to_coord::Kind::LostReport(LostReport {
                                    engine: engine.to_owned(),
                                    lost: Some(lost),
                                })),
                            }, true).await;
                        }
                        Some(signal::Kind::Notice(notice)) => {
                            info!("{engine}: notice for rank {} at {} ms", notice.rank, notice.deadline_ms);
                            self.engines
                                .lock()
                                .await
                                .entry(engine.to_owned())
                                .or_default()
                                .notices
                                .insert(notice.rank, notice);
                            self.send_up(DaemonToCoord {
                                kind: Some(daemon_to_coord::Kind::NoticeReport(NoticeReport {
                                    engine: engine.to_owned(),
                                    notice: Some(notice),
                                })),
                            }, true).await;
                        }
                        Some(signal::Kind::Join(join)) => self.join_up(engine, rank, role, join).await,
                        Some(signal::Kind::ReplayAsk(a)) => warn!("{engine}: a replay ask for rank {} on a registered connection {id}; asks go on a connection of their own", a.rank),
                        Some(signal::Kind::ReplayOffer(o)) => self.replay_offer_up(engine, id, o).await,
                        Some(signal::Kind::Goodbye(goodbye)) => {
                            self.goodbye(engine, id, goodbye).await;
                        }
                        Some(signal::Kind::GroupFormed(f)) => {
                            self.formed(engine, id, f).await;
                        }
                        Some(signal::Kind::GroupFailed(f)) => {
                            warn!("{engine}: rank {rank} could not form a group under plan {}: {}", f.generation, f.error);
                            self.report_cover(cover_report::Kind::Failed(f)).await;
                        }
                        Some(signal::Kind::GroupReady(r)) => {
                            self.report_cover(cover_report::Kind::Ready(r)).await;
                        }
                        Some(signal::Kind::GroupReleased(r)) => {
                            self.released(engine, id, r).await;
                        }
                        Some(signal::Kind::StacksDumped(d)) => self.stacks_dumped(d).await,
                        // decision 0025, amended: a release left behind, named by this
                        // connection, so the coordinator retires the process that holds it
                        Some(signal::Kind::ReleaseLeaked(leak)) => {
                            self.release_leaked(engine, rank, id, leak).await;
                        }
                        // a group's id, made by its server member: up to the coordinator, which
                        // relays it to the group's other members; held across a missing link,
                        // since the server makes it once
                        Some(signal::Kind::GroupId(id)) => {
                            self.send_up(
                                DaemonToCoord {
                                    kind: Some(daemon_to_coord::Kind::GroupId(id)),
                                },
                                true,
                            )
                            .await;
                        }
                        _ => warn!("{engine}: ignoring an unexpected signal from rank {rank}"),
                    }
                }
                Some((sig, reply)) = rx.recv() => {
                    wire::send(wr, &sig).await?;
                    // a signal that wants no answer (a notice, an abandon) never displaces the
                    // acknowledgement a delivery is still waiting on
                    if reply.is_some() {
                        waiting = reply;
                    }
                }
            }
        }
    }
}
