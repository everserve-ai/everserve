// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A process's stacks, asked on its link (decision 0041): the operator API's request
//! reaches the process through its daemon, and its answer goes back the same way, cut at
//! a bound.

use super::Daemon;
use crate::proto::everserve::coord::v1::{Stacks, StacksSent};
use crate::proto::everserve::signals::v1::{DumpStacks, Signal, StacksDumped, signal};
use std::time::Duration;
use tokio::sync::oneshot;
use tracing::info;

/// How long a daemon waits for a process to answer with its stacks: inside the coordinator's
/// own wait for the daemon.
const PROCESS_STACKS_WAIT: Duration = Duration::from_secs(5);

/// The most of a process's stacks a daemon passes on.
pub const MOST_STACKS_BYTES: usize = 1024 * 1024;

/// `text`, cut to [`MOST_STACKS_BYTES`] on a character boundary, with a marker when it was cut.
fn bounded_stacks(mut text: String) -> String {
    if text.len() <= MOST_STACKS_BYTES {
        return text;
    }
    let mut at = MOST_STACKS_BYTES;
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    text.truncate(at);
    text.push_str("\n[stacks cut at 1 MiB]\n");
    text
}

impl Daemon {
    /// Ask a process of `s.engine` on this node for its stacks: the serving rank, or the spare
    /// preloaded for `s.rank` or taking it, or the spare standing for any position on `s.device`,
    /// or the engine's head, whose launcher is asked too when it is another process (0032).
    pub(super) async fn stacks(&self, s: &Stacks) -> StacksSent {
        // asked on the process's link: a daemon in another pod sees no pid of it to signal; the
        // process answers with its stacks, which go back to whoever asked (decision 0041)
        let target = {
            let engines = self.engines.lock().await;
            engines.get(&s.engine).and_then(|e| {
                if s.head {
                    let head = e.head.as_ref()?;
                    e.clients.iter().find(|c| c.id == head.id)
                } else {
                    match (s.rank, s.device) {
                        (Some(r), _) => e
                            .workers
                            .get(&r)
                            .or_else(|| e.spares.get(&r))
                            .or_else(|| e.taking.get(&r).and_then(|d| e.any_spares.get(d))),
                        (None, Some(d)) => e.any_spares.get(&d),
                        (None, None) => None,
                    }
                }
                .map(|c| (c.tx.clone(), c.pid.unwrap_or(0)))
            })
        };
        let Some((tx, pid)) = target else {
            return StacksSent {
                request_id: s.request_id,
                error: format!("no such process of {} on this node", s.engine),
                ..StacksSent::default()
            };
        };
        let request_id = self
            .next_stacks_request
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (answer_tx, answer) = oneshot::channel();
        self.stacks_answers
            .lock()
            .await
            .insert(request_id, answer_tx);
        let ask = Signal {
            kind: Some(signal::Kind::DumpStacks(DumpStacks {
                reason: "asked through the operator API".to_owned(),
                request_id,
            })),
        };
        let mut sent = StacksSent {
            request_id: s.request_id,
            pid,
            ..StacksSent::default()
        };
        if tx.send((ask, None)).await.is_err() {
            self.stacks_answers.lock().await.remove(&request_id);
            return StacksSent {
                error: "its link is closed".to_owned(),
                ..sent
            };
        }
        info!("{}: stacks asked of process {pid} on its link", s.engine);
        match tokio::time::timeout(PROCESS_STACKS_WAIT, answer).await {
            Ok(Ok(dumped)) => {
                sent.answered = true;
                sent.text = bounded_stacks(dumped.text);
            }
            // a build that writes its stacks only to its own log does not answer: asked all
            // the same
            _ => {
                self.stacks_answers.lock().await.remove(&request_id);
            }
        }
        sent
    }

    /// A process answered a stacks request.
    pub(super) async fn stacks_dumped(&self, dumped: StacksDumped) {
        if let Some(tx) = self.stacks_answers.lock().await.remove(&dumped.request_id) {
            let _ = tx.send(dumped);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MOST_STACKS_BYTES, bounded_stacks};

    #[test]
    fn a_processs_stacks_are_cut_at_their_bound_on_a_character_with_a_marker() {
        assert_eq!(bounded_stacks("short".into()), "short");
        // a two-byte character straddles the bound: the cut falls before it
        let long = format!("{}é{}", "a".repeat(MOST_STACKS_BYTES - 1), "b".repeat(10));
        let cut = bounded_stacks(long);
        assert!(
            cut.ends_with("[stacks cut at 1 MiB]\n"),
            "{}",
            &cut[cut.len() - 40..]
        );
        assert_eq!(
            cut.len(),
            MOST_STACKS_BYTES - 1 + "\n[stacks cut at 1 MiB]\n".len()
        );
    }
}
