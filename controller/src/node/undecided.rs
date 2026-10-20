// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! An undecided process (decision 0041): a framework's group pod's process for a rank, asking,
//! before it loads anything, whether it serves the rank or stands as its spare. The daemon asks
//! the coordinator, naming the process by its connection, tells the process, and keeps the
//! connection while the process loads; the process then registers on it as what it was told, and
//! is served from there as any process is. A connection that ends first is reported, so the
//! coordinator releases the rank the process held.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tokio::io::{ReadHalf, WriteHalf};
use tokio::net::UnixStream;
use tokio::sync::oneshot;
use tracing::{info, warn};

use everserve_wire::frame as wire;

use crate::proto::everserve::coord::v1::{
    ConnectionRef, DaemonToCoord, RoleAsked, Undecided, WorkerEvent, daemon_to_coord, worker_event,
};
use crate::proto::everserve::signals::v1::{Register, Role, RoleAssigned, Signal, signal};

use super::Daemon;
use super::conns::registration;

/// How long an undecided process's question waits for the coordinator's answer before it is
/// asked again: no answer while the coordinator relinks, or one it could not log.
const ASK_AGAIN: Duration = Duration::from_secs(2);

/// The oldest protocol an undecided registration is served under.
const UNDECIDED_SINCE: u32 = 11;

/// A number drawn from the system's secure source when the daemon starts: with a connection's id,
/// the name of an undecided process, which no restart of the daemon gives again.
pub(super) fn draw_instance() -> u64 {
    let mut bytes = [0u8; 8];
    if rustls::crypto::ring::default_provider()
        .secure_random
        .fill(&mut bytes)
        .is_err()
    {
        // no secure source: the clock, which a restart does not turn back to the same nanosecond
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        return u64::try_from(now.as_nanos() & u128::from(u64::MAX)).unwrap_or(u64::MAX);
    }
    u64::from_le_bytes(bytes)
}

/// The undecided processes connected, by engine and connection id: the ranks each asked for and
/// its name.
pub(super) type ByEngine = BTreeMap<String, BTreeMap<u64, (Vec<u32>, ConnectionRef)>>;

/// What `reg`'s process on connection `me` asks the coordinator, as request `request_id`.
fn asked_of(reg: &Register, me: &ConnectionRef, request_id: u64) -> RoleAsked {
    RoleAsked {
        request_id,
        engine: reg.engine.clone(),
        rank: reg.rank,
        connection: Some(*me),
        stage: reg.stage.clone(),
        host: reg.host.clone(),
        identity: reg.identity.clone(),
    }
}

/// Every rank `reg`'s answer covers: its stage, or its rank alone.
fn stage_of(reg: &Register) -> Vec<u32> {
    if reg.stage.is_empty() {
        vec![reg.rank]
    } else {
        reg.stage.clone()
    }
}

/// How an undecided process's connection ended its first phase.
enum Decided {
    /// It registered as what it was told.
    Registered(Box<Register>),
    /// It left, or broke the protocol, first.
    Left,
}

impl Daemon {
    /// The registration connection `id` is served as: its first, or, for an undecided process,
    /// the one it makes once it is told what it is; None when the connection ends or breaks the
    /// protocol first.
    ///
    /// # Errors
    ///
    /// The connection could not be read or written.
    pub(super) async fn served_as(
        &self,
        id: u64,
        pid: Option<u32>,
        rd: &mut wire::FrameReader<ReadHalf<UnixStream>>,
        wr: &mut WriteHalf<UnixStream>,
    ) -> Result<Option<Register>, wire::WireError> {
        let first: Signal = rd.recv().await?;
        // a spare that has not registered asks for its rank's replay on a connection of its own
        // (decision 0044): served here, and never a registration
        if let Some(signal::Kind::ReplayAsk(ask)) = first.kind {
            self.serve_replay_ask(id, ask, rd, wr).await?;
            return Ok(None);
        }
        let Some(reg) = registration(first) else {
            return Ok(None);
        };
        if Role::try_from(reg.role) == Ok(Role::Undecided) {
            return self.decide(&reg, id, pid, rd, wr).await;
        }
        Ok(Some(reg))
    }

    /// The undecided processes connected now, by engine, as a hello names them.
    pub(super) async fn undecided_now(&self) -> BTreeMap<String, Vec<Undecided>> {
        self.undecided
            .lock()
            .await
            .iter()
            .map(|(engine, processes)| {
                (
                    engine.clone(),
                    processes
                        .values()
                        .flat_map(|(ranks, connection)| {
                            ranks.iter().map(|rank| Undecided {
                                rank: *rank,
                                connection: Some(*connection),
                            })
                        })
                        .collect(),
                )
            })
            .collect()
    }

    /// Serve the undecided process `reg` on connection `id` until it registers as what the
    /// coordinator told it; that registration, or None when its connection ended first.
    ///
    /// # Errors
    ///
    /// The connection could not be written to.
    pub(super) async fn decide(
        &self,
        reg: &Register,
        id: u64,
        pid: Option<u32>,
        rd: &mut wire::FrameReader<ReadHalf<UnixStream>>,
        wr: &mut WriteHalf<UnixStream>,
    ) -> Result<Option<Register>, wire::WireError> {
        let negotiated =
            everserve_wire::check_protocol(reg.protocol, reg.protocol_oldest).unwrap_or(0);
        if negotiated < UNDECIDED_SINCE {
            warn!(
                "{}: rank {}: an undecided registration needs protocol {UNDECIDED_SINCE}, this one \
                 speaks {negotiated}; closing",
                reg.engine, reg.rank
            );
            return Ok(None);
        }
        let me = ConnectionRef {
            instance: self.instance,
            serial: id,
            pid: pid.unwrap_or(0),
        };
        self.undecided
            .lock()
            .await
            .entry(reg.engine.clone())
            .or_default()
            .insert(id, (stage_of(reg), me));
        let decided = self.ask_and_wait(reg, &me, rd, wr).await;
        {
            let mut undecided = self.undecided.lock().await;
            if let Some(processes) = undecided.get_mut(&reg.engine) {
                processes.remove(&id);
                if processes.is_empty() {
                    undecided.remove(&reg.engine);
                }
            }
        }
        // a connection that ended while the daemon wrote to it (a placement relayed as the
        // process left) has ended as surely as one that closed: its departure is reported either
        // way, or the coordinator holds the ranks it was told to serve for a process that is gone
        let left = match decided {
            Ok(Decided::Registered(next)) => return Ok(Some(*next)),
            Ok(Decided::Left) => Ok(None),
            Err(e) => Err(e),
        };
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::WorkerEvent(WorkerEvent {
                    engine: reg.engine.clone(),
                    kind: worker_event::Kind::UndecidedDeparted as i32,
                    rank: reg.rank,
                    pid: me.pid,
                    connection: Some(me),
                    ..WorkerEvent::default()
                })),
            },
            false,
        )
        .await;
        left
    }

    /// Ask the coordinator what the process `me` is until it answers, tell the process, and wait
    /// for it to register as that.
    async fn ask_and_wait(
        &self,
        reg: &Register,
        me: &ConnectionRef,
        rd: &mut wire::FrameReader<ReadHalf<UnixStream>>,
        wr: &mut WriteHalf<UnixStream>,
    ) -> Result<Decided, wire::WireError> {
        let answer = loop {
            let request_id = self.next_role_request.fetch_add(1, Ordering::Relaxed);
            let (tx, rx) = oneshot::channel();
            self.role_answers.lock().await.insert(request_id, tx);
            self.send_up(
                DaemonToCoord {
                    kind: Some(daemon_to_coord::Kind::RoleAsked(asked_of(
                        reg, me, request_id,
                    ))),
                },
                false,
            )
            .await;
            tokio::select! {
                answered = rx => {
                    if let Ok(a) = answered {
                        break a;
                    }
                }
                () = tokio::time::sleep(ASK_AGAIN) => {
                    self.role_answers.lock().await.remove(&request_id);
                }
                // the process says nothing until it is told; its connection ending is its leaving
                frame = rd.recv::<Signal>() => {
                    self.role_answers.lock().await.remove(&request_id);
                    if frame.is_ok() {
                        warn!(
                            "{}: rank {}: an undecided process spoke before it was told what it is; \
                             closing",
                            reg.engine, reg.rank
                        );
                    }
                    return Ok(Decided::Left);
                }
            }
        };
        // refused: told why, and its connection ends; the process stops with the reason
        if !answer.refusal.is_empty() {
            warn!(
                "{}: rank {}'s undecided process {} is refused: {}",
                reg.engine, reg.rank, me.pid, answer.refusal
            );
            wire::send(
                wr,
                &Signal {
                    kind: Some(signal::Kind::RoleAssigned(RoleAssigned {
                        engine: reg.engine.clone(),
                        rank: reg.rank,
                        stage: reg.stage.clone(),
                        refusal: answer.refusal.clone(),
                        ..RoleAssigned::default()
                    })),
                },
            )
            .await?;
            return Ok(Decided::Left);
        }
        let role = Role::try_from(answer.role).unwrap_or(Role::Spare);
        // the engine's policy before the answer: the process decides with it (decision 0041)
        let policy = self.policy_of(&reg.engine).await;
        if let Some(p) = policy.clone() {
            wire::send(
                wr,
                &Signal {
                    kind: Some(signal::Kind::EnginePolicy(p)),
                },
            )
            .await?;
        }
        info!(
            "{}: rank {}'s undecided process {} is told: {}",
            reg.engine,
            reg.rank,
            me.pid,
            role.as_str_name()
        );
        wire::send(
            wr,
            &Signal {
                kind: Some(signal::Kind::RoleAssigned(RoleAssigned {
                    engine: reg.engine.clone(),
                    rank: reg.rank,
                    epoch: answer.epoch,
                    role: role as i32,
                    stage: reg.stage.clone(),
                    refusal: String::new(),
                })),
            },
        )
        .await?;
        let handed = super::facts::Facts {
            policy,
            ..super::facts::Facts::default()
        };
        self.wait_registered(reg, role, handed, rd, wr).await
    }

    /// After the answer, wait for the process `reg` asked for to register as `role`, handing it
    /// what the coordinator says of its engine meanwhile, beyond what it was `handed`; how its
    /// first phase ended.
    async fn wait_registered(
        &self,
        reg: &Register,
        role: Role,
        mut handed: super::facts::Facts,
        rd: &mut wire::FrameReader<ReadHalf<UnixStream>>,
        wr: &mut WriteHalf<UnixStream>,
    ) -> Result<Decided, wire::WireError> {
        // loading takes minutes: the connection stays the process's, and the assignment the
        // rank's. A process deciding for a stage registers nothing here: its workers register on
        // their own connections, and this one stays open, silent, as its liveness until it ends.
        // Meanwhile it is handed what the coordinator says of its engine: where it runs, which its
        // workers join by, and its instance's state (decision 0041)
        let mut facts = self.facts(&reg.engine).await.subscribe();
        facts.mark_changed();
        // a spare's asks for its ranks' compile caches (a stage's process asks for each of its
        // stage's ranks), answered in turn while its weights load (decision 0043)
        let mut wanted: std::collections::VecDeque<
            crate::proto::everserve::signals::v1::CacheWanted,
        > = std::collections::VecDeque::new();
        let mut fetching: Option<
            std::pin::Pin<Box<dyn std::future::Future<Output = Signal> + Send + '_>>,
        > = None;
        let next = loop {
            if fetching.is_none()
                && let Some(want) = wanted.pop_front()
            {
                fetching = Some(Box::pin(async move {
                    self.cache_for(&want, super::cache::CACHE_WAIT).await
                }));
            }
            tokio::select! {
                frame = rd.recv::<Signal>() => match frame {
                    Ok(Signal { kind: Some(signal::Kind::CacheWanted(want)) })
                        if role == Role::Spare
                            && want.engine == reg.engine
                            && (want.rank == reg.rank || reg.stage.contains(&want.rank)) =>
                    {
                        wanted.push_back(want);
                    }
                    other => break other,
                },
                answer = async { fetching.as_mut().expect("guarded").await }, if fetching.is_some() => {
                    fetching = None;
                    wire::send(wr, &answer).await?;
                }
                changed = facts.changed() => {
                    if changed.is_err() {
                        continue;
                    }
                    let now = facts.borrow_and_update().clone();
                    for sig in now.since(&handed) {
                        wire::send(wr, &sig).await?;
                    }
                    handed = now;
                }
            }
        };
        let Ok(next) = next else {
            return Ok(Decided::Left);
        };
        let Some(next) = registration(next) else {
            return Ok(Decided::Left);
        };
        if next.engine != reg.engine
            || next.rank != reg.rank
            || Role::try_from(next.role) != Ok(role)
        {
            warn!(
                "{}: rank {}: told {}, the process registered as {:?} for rank {}; closing",
                reg.engine,
                reg.rank,
                role.as_str_name(),
                Role::try_from(next.role),
                next.rank
            );
            return Ok(Decided::Left);
        }
        Ok(Decided::Registered(Box::new(next)))
    }

    /// The coordinator answered an undecided process's question.
    pub(super) async fn role_answered(
        &self,
        answer: crate::proto::everserve::coord::v1::RoleAnswer,
    ) {
        if let Some(tx) = self.role_answers.lock().await.remove(&answer.request_id) {
            let _ = tx.send(answer);
        }
    }
}
