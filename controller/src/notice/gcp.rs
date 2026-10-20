// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! GCP preemption and host maintenance, from the instance metadata server.
//!
//! Two keys under `computeMetadata/v1/instance/`, both with `Metadata-Flavor: Google`:
//!
//! - `preempted`: `FALSE` until the Spot VM is preempted, `TRUE` from then on. GCP states no
//!   time: the ACPI soft-off follows the flag at once by default, or after the preemption notice
//!   the VM was created with (120 s is the option), then up to 30 s of shutdown. The deadline is
//!   now plus the settings' lead, which the operator sets to the VM's notice.
//! - `maintenance-event`: `NONE`, `MIGRATE_ON_HOST_MAINTENANCE` or `TERMINATE_ON_HOST_MAINTENANCE`;
//!   GPU VMs cannot live-migrate, so theirs terminates, and the value changes 60 minutes before
//!   (GCP's figure). The deadline is now plus that.
//!
//! Both are polled at the poll interval rather than long-polled, so one request's wait never
//! delays the other key.

use std::time::Duration;

use super::http::Client;
use super::{NodeNotice, NoticeError, Settings, Source, in_lead};
use crate::proto::everserve::signals::v1::Reason;

/// The metadata server.
pub const DEFAULT_ENDPOINT: &str = "http://metadata.google.internal";
/// How long before a terminating host maintenance GCP flips `maintenance-event`.
pub const MAINTENANCE_LEAD: Duration = Duration::from_hours(1);

/// Poll once: the preemption if flagged, else the terminating maintenance if one is coming.
///
/// # Errors
///
/// [`NoticeError`] when the server cannot be reached or answers something else.
pub async fn poll(client: &Client, settings: &Settings) -> Result<Option<NodeNotice>, NoticeError> {
    let base = settings.endpoint.as_deref().unwrap_or(DEFAULT_ENDPOINT);
    let headers = [("Metadata-Flavor", "Google")];
    let preempted = client
        .get(
            &format!("{base}/computeMetadata/v1/instance/preempted"),
            &headers,
        )
        .await?;
    if preempted.status != 200 {
        return Err(NoticeError::Endpoint {
            from: Source::Gcp,
            reason: format!("preempted answered {}", preempted.status),
        });
    }
    if let Some(n) = parse_preempted(&preempted.body, settings)? {
        return Ok(Some(n));
    }
    let maintenance = client
        .get(
            &format!("{base}/computeMetadata/v1/instance/maintenance-event"),
            &headers,
        )
        .await?;
    if maintenance.status != 200 {
        return Err(NoticeError::Endpoint {
            from: Source::Gcp,
            reason: format!("maintenance-event answered {}", maintenance.status),
        });
    }
    parse_maintenance(&maintenance.body)
}

/// The notice in a `preempted` body: `TRUE` is a notice with the lead, `FALSE` none.
///
/// # Errors
///
/// [`NoticeError::Parse`] for any other body.
pub fn parse_preempted(body: &str, settings: &Settings) -> Result<Option<NodeNotice>, NoticeError> {
    match body.trim() {
        "TRUE" => Ok(Some(in_lead(
            Source::Gcp,
            settings.lead,
            Reason::Notice,
            "instance preempted".into(),
        ))),
        "FALSE" => Ok(None),
        other => Err(NoticeError::Parse {
            from: Source::Gcp,
            reason: format!("preempted: expected TRUE or FALSE, got {other:?}"),
        }),
    }
}

/// The notice in a `maintenance-event` body: a terminating maintenance is a notice 60 minutes
/// out; a migration or none is not.
///
/// # Errors
///
/// [`NoticeError::Parse`] for any other body.
pub fn parse_maintenance(body: &str) -> Result<Option<NodeNotice>, NoticeError> {
    match body.trim() {
        "TERMINATE_ON_HOST_MAINTENANCE" => Ok(Some(in_lead(
            Source::Gcp,
            MAINTENANCE_LEAD,
            Reason::Notice,
            "host maintenance terminates the instance".into(),
        ))),
        "NONE" | "MIGRATE_ON_HOST_MAINTENANCE" => Ok(None),
        other => Err(NoticeError::Parse {
            from: Source::Gcp,
            reason: format!("maintenance-event: unexpected {other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {

    use std::time::Duration;

    use super::{parse_maintenance, parse_preempted};
    use crate::clock::now_ms;
    use crate::notice::{Settings, Source};

    fn settings() -> Settings {
        Settings {
            source: Source::Gcp,
            endpoint: None,
            poll: Duration::from_secs(1),
            lead: Duration::from_mins(2),
            node: "n".into(),
            kubernetes: None,
        }
    }

    #[test]
    fn the_flag_gives_the_lead_and_a_terminating_maintenance_gives_an_hour() {
        let before = now_ms();
        let n = parse_preempted("TRUE\n", &settings())
            .unwrap()
            .expect("a notice");
        assert!(
            n.deadline_ms >= before + 120_000 && n.deadline_ms < before + 121_000,
            "{n:?}"
        );
        assert_eq!(parse_preempted("FALSE", &settings()).unwrap(), None);
        assert!(parse_preempted("maybe", &settings()).is_err());
        let n = parse_maintenance("TERMINATE_ON_HOST_MAINTENANCE")
            .unwrap()
            .expect("a notice");
        assert!(n.deadline_ms >= before + 3_600_000, "{n:?}");
        assert_eq!(parse_maintenance("NONE").unwrap(), None);
        assert_eq!(
            parse_maintenance("MIGRATE_ON_HOST_MAINTENANCE").unwrap(),
            None
        );
        assert!(parse_maintenance("").is_err());
    }
}
