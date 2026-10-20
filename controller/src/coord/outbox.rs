// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0

//! What the coordinator has to send one daemon, written by that link's own writer task.
//!
//! A send never waits. A link's handler, a change, a placement push and a link coming up all put
//! messages here and go on. Only the writer waits, on its daemon's socket. A link whose own reader
//! task drains a bounded channel deadlocks at its hello, where a link coming up sends every
//! engine's policy, placement and state before it drains anything; once every link waits on
//! another's full channel the whole coordinator parks (a cell of 1,250 nodes under a policy
//! naming 33 engines: every link parked).
//!
//! What stays queued is bounded by what is current, not by how much was sent. An engine's
//! policy, placement, instance state and cover plan each say the whole of their fact, and the
//! daemon keeps only the newest: a newer one queued replaces the unsent older one, and takes its
//! place at the back, where a queue holding every version would have delivered it. The other
//! messages (a delivery, a commit, a forwarded notice) are delivered in order, each one; a daemon
//! that leaves [`MAX_ORDERED`] of them unread is not reading, and its link is closed.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::Notify;

use crate::proto::everserve::coord::v1::{CoordToDaemon, coord_to_daemon};

/// The most ordered messages a link holds unsent before it is taken as not reading.
pub const MAX_ORDERED: usize = 4096;

/// The fact a message states the whole of, which a newer one of the same fact replaces.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Fact {
    Policy(String),
    Placement(String),
    State(String),
    Cover(String),
}

impl Fact {
    fn of(msg: &CoordToDaemon) -> Option<Self> {
        match msg.kind.as_ref()? {
            coord_to_daemon::Kind::EnginePolicy(p) => Some(Self::Policy(p.engine.clone())),
            coord_to_daemon::Kind::EnginePlacement(p) => Some(Self::Placement(p.engine.clone())),
            coord_to_daemon::Kind::InstanceState(s) => Some(Self::State(s.engine.clone())),
            coord_to_daemon::Kind::CoverPlan(p) => Some(Self::Cover(p.engine.clone())),
            _ => None,
        }
    }
}

/// Why a message was not queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Closed {
    /// The link ended.
    #[error("the daemon's link ended")]
    Ended,
    /// The daemon left [`MAX_ORDERED`] messages unread; its link is closed.
    #[error("the daemon left {MAX_ORDERED} messages unread")]
    NotReading,
}

#[derive(Default)]
struct Queue {
    messages: VecDeque<(Option<Fact>, CoordToDaemon)>,
    ordered: usize,
    closed: Option<Closed>,
}

/// One daemon's outbox; clones share it.
#[derive(Clone, Default)]
pub struct Outbox {
    queue: Arc<Mutex<Queue>>,
    ready: Arc<Notify>,
}

impl Outbox {
    /// Queue `msg` for the daemon without waiting.
    ///
    /// # Errors
    ///
    /// [`Closed`] once the link ended, or when this message would leave [`MAX_ORDERED`]
    /// ordered messages unsent, which closes the link.
    pub fn send(&self, msg: CoordToDaemon) -> Result<(), Closed> {
        let fact = Fact::of(&msg);
        {
            let mut q = self.lock();
            if let Some(why) = q.closed {
                return Err(why);
            }
            if let Some(f) = &fact {
                q.messages.retain(|(g, _)| g.as_ref() != Some(f));
            } else if q.ordered >= MAX_ORDERED {
                q.closed = Some(Closed::NotReading);
                q.messages.clear();
                drop(q);
                self.ready.notify_one();
                return Err(Closed::NotReading);
            } else {
                q.ordered += 1;
            }
            q.messages.push_back((fact, msg));
        }
        self.ready.notify_one();
        Ok(())
    }

    /// Everything queued, oldest first, once there is any; `Err` once the link is closed.
    ///
    /// # Errors
    ///
    /// Why the link closed: what was queued then is not delivered.
    pub async fn next(&self) -> Result<Vec<CoordToDaemon>, Closed> {
        loop {
            // registered before the check, so a send between the check and the wait wakes it
            let ready = self.ready.notified();
            {
                let mut q = self.lock();
                if let Some(why) = q.closed {
                    return Err(why);
                }
                if !q.messages.is_empty() {
                    q.ordered = 0;
                    return Ok(q.messages.drain(..).map(|(_, m)| m).collect());
                }
            }
            ready.await;
        }
    }

    /// End the link: nothing more is queued, and the writer stops.
    pub fn close(&self) {
        {
            let mut q = self.lock();
            q.closed.get_or_insert(Closed::Ended);
            q.messages.clear();
        }
        self.ready.notify_one();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Queue> {
        // a panic while the lock is held leaves the queue whole: every change to it is one call
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::{Closed, MAX_ORDERED, Outbox};
    use crate::proto::everserve::coord::v1::{Abandon, CoordToDaemon, coord_to_daemon};
    use crate::proto::everserve::signals::v1::{EnginePlacement, EnginePolicy};

    fn placement(engine: &str, epoch: u64) -> CoordToDaemon {
        CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::EnginePlacement(EnginePlacement {
                engine: engine.into(),
                epoch,
                ..EnginePlacement::default()
            })),
        }
    }

    fn policy(engine: &str) -> CoordToDaemon {
        CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::EnginePolicy(EnginePolicy {
                engine: engine.into(),
                policy: Vec::new(),
            })),
        }
    }

    fn ordered(epoch: u64) -> CoordToDaemon {
        CoordToDaemon {
            term: 0,
            kind: Some(coord_to_daemon::Kind::Abandon(Abandon {
                epoch,
                ..Abandon::default()
            })),
        }
    }

    #[tokio::test]
    async fn a_newer_fact_replaces_the_unsent_one_and_goes_where_its_own_send_put_it() {
        let o = Outbox::default();
        o.send(placement("e0", 1)).expect("queued");
        o.send(ordered(7)).expect("queued");
        o.send(policy("e0")).expect("queued");
        o.send(placement("e1", 1)).expect("queued");
        o.send(placement("e0", 2)).expect("queued");
        let sent = o.next().await.expect("open");
        // e0's first placement is gone; its second follows everything sent before it
        assert_eq!(
            sent,
            vec![
                ordered(7),
                policy("e0"),
                placement("e1", 1),
                placement("e0", 2)
            ]
        );
    }

    #[tokio::test]
    async fn sends_never_wait_and_facts_stay_bounded_by_the_engines() {
        let o = Outbox::default();
        // far more sends than any channel held, with no writer draining them
        for epoch in 0..100_000 {
            o.send(placement(&format!("e{}", epoch % 40), epoch))
                .expect("queued");
        }
        assert_eq!(o.next().await.expect("open").len(), 40);
    }

    #[tokio::test]
    async fn a_daemon_that_leaves_too_many_ordered_messages_unread_is_closed() {
        let o = Outbox::default();
        for epoch in 0..MAX_ORDERED as u64 {
            o.send(ordered(epoch)).expect("queued");
        }
        assert_eq!(o.send(ordered(0)), Err(Closed::NotReading));
        assert_eq!(o.next().await, Err(Closed::NotReading));
        // read, the count starts again
        let o = Outbox::default();
        for epoch in 0..MAX_ORDERED as u64 {
            o.send(ordered(epoch)).expect("queued");
        }
        assert_eq!(o.next().await.expect("open").len(), MAX_ORDERED);
        o.send(ordered(0)).expect("queued after a read");
    }

    #[tokio::test]
    async fn a_closed_outbox_takes_nothing_and_wakes_its_writer() {
        let o = Outbox::default();
        let writer = tokio::spawn({
            let o = o.clone();
            async move { o.next().await }
        });
        tokio::task::yield_now().await;
        o.close();
        assert_eq!(writer.await.expect("join"), Err(Closed::Ended));
        assert_eq!(o.send(ordered(1)), Err(Closed::Ended));
    }
}
