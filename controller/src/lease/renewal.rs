// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The renewal every renewed backend's hold shares (decision 0036). A backend supplies one
//! renewal step ([`Renew`]); this owns the timing, so the rule that keeps one holder at a time is
//! written once: renew every quarter of the TTL, and end the hold three quarters of the TTL after
//! its last answered renewal was sent, by this process's monotonic clock, whether or not a
//! renewal is in flight. The store expires a lease no earlier than a full TTL after it received
//! that renewal, which is later still, so the hold has ended before a contender can take it.

use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;
use tracing::{error, warn};

use super::BoxFuture;

/// What one renewal found.
pub(super) enum Renewed {
    /// The lease lives this long more from when the renewal was sent, and is still the hold's.
    Until(Duration),
    /// The store says the lease is no longer the hold's: it expired, or another took it.
    Ended,
    /// The renewal failed; the deadline the last one set stands, and the next is tried.
    Failed(String),
}

/// One backend's renewal step.
pub(super) trait Renew: Send + 'static {
    /// Renew once. A backend resets, on [`Renewed::Failed`], whatever a late answer to this
    /// renewal could be mistaken for the answer to the next.
    fn renew(&mut self) -> BoxFuture<'_, Renewed>;

    /// The lease and the hold, for logs.
    fn describe(&self) -> String;
}

/// When a hold renewed by a request sent at `sent`, for `ttl`, stops acting.
pub(super) fn deadline(sent: Instant, ttl: Duration) -> Instant {
    sent + ttl * 3 / 4
}

/// Renew with `renew` every quarter of `ttl` until `ended` is cancelled, and cancel it when the
/// hold's deadline passes, first set by a grant of `granted` sent at `sent`, or when the store
/// says the lease is no longer the hold's. The task stops when dropped or aborted.
pub(super) fn start(
    mut renew: impl Renew,
    ttl: Duration,
    sent: Instant,
    granted: Duration,
    ended: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut until = deadline(sent, granted);
        let why = loop {
            let ends = tokio::time::Instant::from_std(until);
            tokio::select! {
                () = ended.cancelled() => return,
                () = tokio::time::sleep_until(ends) => break "no renewal was answered in time",
                () = tokio::time::sleep(ttl / 4) => {}
            }
            let sent = Instant::now();
            let renewed = tokio::select! {
                () = ended.cancelled() => return,
                () = tokio::time::sleep_until(ends) => break "no renewal was answered in time",
                renewed = renew.renew() => renewed,
            };
            match renewed {
                Renewed::Until(ttl) => until = deadline(sent, ttl),
                Renewed::Ended => break "the store says the lease is no longer this hold's",
                Renewed::Failed(e) => warn!("{}: renewing: {e}", renew.describe()),
            }
        };
        error!("{}: the hold has ended: {why}", renew.describe());
        ended.cancel();
    })
}
