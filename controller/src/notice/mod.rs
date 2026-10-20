// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Preemption notice sources, one module per cloud (docs/MULTINODE.md section 7).
//!
//! Every source answers one question for the node its daemon runs on: is this node leaving, and
//! when. The answer is a [`NodeNotice`] the daemon reports to the coordinator, which drains the
//! node: a notice to every rank on it and one change per engine at the deadline. The operator's
//! own reclaim is a drain set through the operator API (`PUT /v1/nodes/{node}`, docs/RUNBOOK.md)
//! and needs no source here.
//!
//! | source | where it reads | deadline |
//! |---|---|---|
//! | `aws` | `IMDSv2` `spot/instance-action`, polled | the action's `time` |
//! | `gcp` | `instance/preempted` and `instance/maintenance-event`, polled | now plus the lead for preemption (GCP's notice is 0 s by default, 120 s if the VM asked for it); 60 minutes for a host maintenance that terminates the VM |
//! | `azure` | Scheduled Events, polled, filtered to this VM's name | the event's `NotBefore`; now for an event already started |
//! | `kubernetes` | this node's taints, polled | the autoscaler's deletion time, or the lead for a cordon |
//!
//! Each source is a pure parser over the endpoint's body plus one small HTTP client; the parsers
//! are tested against the bodies the clouds document, and the sources against fake endpoints
//! (`docs/reviews/2026-10-06-neighbours-control-plane.md` records the sources read). The TLS
//! connection to a real Kubernetes API server is not exercised in CI.

use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::clock::now_ms;
use crate::node::Daemon;
pub use crate::proto::everserve::coord::v1::NodeNotice;
use crate::proto::everserve::signals::v1::Reason;

pub mod aws;
pub mod azure;
pub mod gcp;
pub mod http;
pub mod kubernetes;

/// Which cloud's signal the daemon watches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// No source: the operator's drain call is the only notice.
    None,
    /// AWS EC2 spot interruption notices.
    Aws,
    /// GCP preemption.
    Gcp,
    /// Azure Scheduled Events.
    Azure,
    /// Kubernetes node taints.
    Kubernetes,
}

impl FromStr for Source {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "aws" => Ok(Self::Aws),
            "gcp" => Ok(Self::Gcp),
            "azure" => Ok(Self::Azure),
            "kubernetes" | "k8s" => Ok(Self::Kubernetes),
            other => Err(format!(
                "unknown notice source {other:?}; one of none, aws, gcp, azure, kubernetes"
            )),
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Aws => "aws",
            Self::Gcp => "gcp",
            Self::Azure => "azure",
            Self::Kubernetes => "kubernetes",
        })
    }
}

/// How the daemon watches its source.
#[derive(Debug, Clone)]
pub struct Settings {
    /// The source.
    pub source: Source,
    /// The endpoint instead of the cloud's default, `http://host:port` or `https://...`.
    pub endpoint: Option<String>,
    /// Between polls.
    pub poll: Duration,
    /// The lead for a source without a deadline of its own.
    pub lead: Duration,
    /// This node's name (the Kubernetes node name).
    pub node: String,
    /// How the Kubernetes source reaches the API; the clouds' sources do not use it.
    pub kubernetes: Option<crate::kubernetes::Access>,
}

/// Why a source could not be read.
#[derive(Debug, thiserror::Error)]
pub enum NoticeError {
    /// The endpoint could not be reached or answered badly.
    #[error("{from}: {reason}")]
    Endpoint {
        /// The source.
        from: Source,
        /// What went wrong.
        reason: String,
    },
    /// The body was not what the cloud documents.
    #[error("{from}: cannot parse the endpoint's answer: {reason}")]
    Parse {
        /// The source.
        from: Source,
        /// What was wrong with it.
        reason: String,
    },
}

/// A notice with the lead the settings give a source that states no time of its own.
#[must_use]
pub fn in_lead(source: Source, lead: Duration, reason: Reason, detail: String) -> NodeNotice {
    NodeNotice {
        deadline_ms: now_ms().saturating_add(i64::try_from(lead.as_millis()).unwrap_or(i64::MAX)),
        reason: reason as i32,
        source: source.to_string(),
        detail,
    }
}

/// Unix milliseconds of an RFC 3339 time of the form `2017-09-18T08:22:00Z` (the only form the
/// clouds' notices use); `None` for anything else.
#[must_use]
pub fn rfc3339_ms(text: &str) -> Option<i64> {
    let (date, rest) = text.trim().split_once('T')?;
    let clock = rest.strip_suffix('Z')?;
    let mut date_parts = date.split('-');
    let year = date_parts.next()?.parse::<i64>().ok()?;
    let month = date_parts.next()?.parse::<u32>().ok()?;
    let day = date_parts.next()?.parse::<u32>().ok()?;
    if date_parts.next().is_some() {
        return None;
    }
    let mut clock_parts = clock.split(':');
    let hour = clock_parts.next()?.parse::<i64>().ok()?;
    let minute = clock_parts.next()?.parse::<i64>().ok()?;
    let second_text = clock_parts.next()?;
    if clock_parts.next().is_some() {
        return None;
    }
    let (second, fraction) = second_text.split_once('.').unwrap_or((second_text, ""));
    let second = second.parse::<i64>().ok()?;
    let millis = if fraction.is_empty() {
        0
    } else {
        let digits: String = fraction.chars().take(3).collect();
        let value = digits.parse::<i64>().ok()?;
        value * 10_i64.pow(3 - u32::try_from(digits.len()).ok()?)
    };
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let days = days_from_civil(year, month, day);
    Some((days * 86_400 + hour * 3600 + minute * 60 + second) * 1000 + millis)
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Watch the source in `settings` until `cancel` fires, reporting the node's deadline to `daemon`
/// once, and again only when it moves earlier. A source that keeps answering the same notice
/// reports it once: a fixed time (AWS, the autoscaler's taint) repeats equal, and a source without
/// a time of its own (GCP's flag, a cordon) computes a later deadline on every poll, which is the
/// same news. A source that fails is retried at the poll interval, with the failure logged.
pub async fn run(settings: Settings, daemon: Arc<Daemon>, cancel: CancellationToken) {
    if settings.source == Source::None {
        return;
    }
    info!(
        "watching {} for a notice (poll {:?}, lead {:?})",
        settings.source, settings.poll, settings.lead
    );
    let client = match http::Client::for_source(&settings) {
        Ok(c) => c,
        Err(e) => {
            warn!("notice source {} not started: {e}", settings.source);
            return;
        }
    };
    let mut last: Option<i64> = None;
    let mut vm_name: Option<String> = None;
    loop {
        let result = match settings.source {
            Source::None => return,
            Source::Aws => aws::poll(&client, &settings).await,
            Source::Gcp => gcp::poll(&client, &settings).await,
            Source::Azure => azure::poll(&client, &settings, &mut vm_name).await,
            Source::Kubernetes => kubernetes::poll(&client, &settings).await,
        };
        match result {
            Ok(Some(n)) if last.is_none_or(|reported| n.deadline_ms < reported) => {
                last = Some(n.deadline_ms);
                daemon.node_notice(n).await;
            }
            Ok(_) => {}
            Err(e) => warn!("{e}"),
        }
        tokio::select! {
            () = cancel.cancelled() => return,
            () = tokio::time::sleep(settings.poll) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::rfc3339_ms;

    #[test]
    fn rfc3339_times_become_unix_milliseconds() {
        assert_eq!(rfc3339_ms("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(rfc3339_ms("2017-09-18T08:22:00Z"), Some(1_505_722_920_000));
        assert_eq!(rfc3339_ms("2000-02-29T23:59:59.5Z"), Some(951_868_799_500));
        assert_eq!(rfc3339_ms("2026-10-06T11:41:18Z"), Some(1_791_286_878_000));
        for bad in [
            "",
            "2017-09-18",
            "2017-09-18T08:22:00",
            "2017-13-01T00:00:00Z",
            "x",
        ] {
            assert_eq!(rfc3339_ms(bad), None, "{bad}");
        }
    }
}
