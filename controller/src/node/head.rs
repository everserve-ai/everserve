// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The engine's head on this node (decision 0032): its registration told to the coordinator,
//! its goodbye kept until its connection ends, and its departure told with the cause.

use super::{Daemon, Head};
use crate::proto::everserve::coord::v1::{
    DaemonToCoord, WorkerEvent, daemon_to_coord, worker_event,
};
use crate::proto::everserve::signals::v1::{Goodbye, Register, Role, SoftwareIdentity, StopCause};
use tracing::info;

impl Daemon {
    /// Connection `id` of `engine` closed: if it was the engine's head, forget it and tell the
    /// coordinator why it went, its goodbye's cause or lost without one (decision 0032).
    pub(super) async fn end_head(&self, engine: &str, id: u64) {
        let head = {
            let mut engines = self.engines.lock().await;
            let Some(state) = engines.get_mut(engine) else {
                return;
            };
            if state.head.as_ref().is_none_or(|h| h.id != id) {
                return;
            }
            state.head.take()
        };
        let Some(head) = head else { return };
        let goodbye = head.goodbye.unwrap_or_else(|| Goodbye {
            cause: StopCause::Lost as i32,
            detail: "the head's connection ended without a goodbye".into(),
        });
        info!(
            "head of {engine} departed: {:?}: {}",
            StopCause::try_from(goodbye.cause).unwrap_or(StopCause::Lost),
            goodbye.detail
        );
        self.head_event(
            engine,
            worker_event::Kind::HeadDeparted,
            0,
            head.launcher_pid,
            Some(goodbye),
            None,
        )
        .await;
    }

    /// `reg`, on connection `id`, registered the engine's head (`role`): the coordinator is told,
    /// as the head stands recorded here (what it runs, which says whether it serves membership
    /// changes, and the address it registered), the same event every hello carries; any other
    /// role, nothing.
    pub(super) async fn head_registered(&self, reg: &Register, role: Role, id: u64) {
        if role != Role::Head {
            return;
        }
        let ev = {
            let engines = self.engines.lock().await;
            let Some(core) = engines
                .get(&reg.engine)
                .and_then(|e| e.head.as_ref())
                .filter(|h| h.id == id)
            else {
                return;
            };
            self.head_registration(&reg.engine, core, worker_event::Kind::HeadRegistered)
        };
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::WorkerEvent(ev)),
            },
            false,
        )
        .await;
    }

    /// `head` of `engine` as its registration and every hello report it.
    pub(super) fn head_registration(
        &self,
        engine: &str,
        head: &Head,
        kind: worker_event::Kind,
    ) -> WorkerEvent {
        WorkerEvent {
            engine: engine.to_owned(),
            kind: kind as i32,
            pid: head.pid.unwrap_or(0),
            launcher_pid: head.launcher_pid,
            identity: head.identity.clone(),
            connection: Some(crate::proto::everserve::coord::v1::ConnectionRef {
                instance: self.instance,
                serial: head.id,
                pid: head.pid.unwrap_or(0),
            }),
            host: head.host.clone(),
            ..WorkerEvent::default()
        }
    }

    /// The head of `engine` said goodbye on connection `id`; kept until its connection ends.
    pub(super) async fn goodbye(&self, engine: &str, id: u64, goodbye: Goodbye) {
        let mut engines = self.engines.lock().await;
        let Some(e) = engines.get_mut(engine) else {
            return;
        };
        match e.head.as_mut().filter(|h| h.id == id) {
            Some(head) => head.goodbye = Some(goodbye),
            None => {
                e.goodbyes.insert(id, goodbye);
            }
        }
    }

    /// Tell the coordinator the engine's head registered or departed (decision 0032).
    async fn head_event(
        &self,
        engine: &str,
        kind: worker_event::Kind,
        pid: u32,
        launcher_pid: u32,
        goodbye: Option<Goodbye>,
        identity: Option<SoftwareIdentity>,
    ) {
        let (cause, detail) = goodbye.map_or((0, String::new()), |g| (g.cause, g.detail));
        self.send_up(
            DaemonToCoord {
                kind: Some(daemon_to_coord::Kind::WorkerEvent(WorkerEvent {
                    engine: engine.to_owned(),
                    kind: kind as i32,
                    pid,
                    launcher_pid,
                    cause,
                    detail,
                    identity,
                    ..WorkerEvent::default()
                })),
            },
            false,
        )
        .await;
    }

    /// Whether connection `id` is `engine`'s head here now.
    pub(super) async fn is_head(&self, engine: &str, id: u64) -> bool {
        self.engines
            .lock()
            .await
            .get(engine)
            .and_then(|e| e.head.as_ref())
            .is_some_and(|h| h.id == id)
    }
}
