// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! AWS EC2 spot interruption notices, from the instance metadata service (`IMDSv2`).
//!
//! `PUT /latest/api/token` gives a session token; `GET /latest/meta-data/spot/instance-action`
//! answers 404 while nothing is scheduled and, about two minutes before the interruption,
//! `{"action": "terminate" | "stop" | "hibernate", "time": "2017-09-18T08:22:00Z"}`. The time is
//! the deadline.

use super::http::Client;
use super::{NodeNotice, NoticeError, Settings, Source, rfc3339_ms};
use crate::proto::everserve::signals::v1::Reason;

/// The metadata service.
pub const DEFAULT_ENDPOINT: &str = "http://169.254.169.254";
/// Session token lifetime asked for.
const TOKEN_TTL_SECONDS: &str = "21600";

/// Poll once: the notice if one is scheduled.
///
/// # Errors
///
/// [`NoticeError`] when the service cannot be reached or answers something else.
pub async fn poll(client: &Client, settings: &Settings) -> Result<Option<NodeNotice>, NoticeError> {
    let base = settings.endpoint.as_deref().unwrap_or(DEFAULT_ENDPOINT);
    let token = client
        .put(
            &format!("{base}/latest/api/token"),
            &[("X-aws-ec2-metadata-token-ttl-seconds", TOKEN_TTL_SECONDS)],
        )
        .await?;
    if token.status != 200 {
        return Err(NoticeError::Endpoint {
            from: Source::Aws,
            reason: format!("token request answered {}", token.status),
        });
    }
    let answer = client
        .get(
            &format!("{base}/latest/meta-data/spot/instance-action"),
            &[("X-aws-ec2-metadata-token", token.body.trim())],
        )
        .await?;
    match answer.status {
        404 => Ok(None),
        200 => parse(&answer.body).map(Some),
        other => Err(NoticeError::Endpoint {
            from: Source::Aws,
            reason: format!("instance-action answered {other}"),
        }),
    }
}

/// The notice in an `instance-action` body.
///
/// # Errors
///
/// [`NoticeError::Parse`] when the body is not the documented JSON.
pub fn parse(body: &str) -> Result<NodeNotice, NoticeError> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| NoticeError::Parse {
        from: Source::Aws,
        reason: e.to_string(),
    })?;
    let action = v["action"].as_str().unwrap_or("");
    let time = v["time"].as_str().ok_or_else(|| NoticeError::Parse {
        from: Source::Aws,
        reason: "no time".into(),
    })?;
    let deadline_ms = rfc3339_ms(time).ok_or_else(|| NoticeError::Parse {
        from: Source::Aws,
        reason: format!("time {time:?} is not RFC 3339"),
    })?;
    Ok(NodeNotice {
        deadline_ms,
        reason: Reason::Notice as i32,
        source: Source::Aws.to_string(),
        detail: format!("spot {action} at {time}"),
    })
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn the_documented_body_parses_and_others_do_not() {
        let n = parse(r#"{"action": "terminate", "time": "2017-09-18T08:22:00Z"}"#).unwrap();
        assert_eq!(n.deadline_ms, 1_505_722_920_000);
        assert_eq!(n.source, "aws");
        assert!(n.detail.contains("terminate"));
        assert!(parse("{}").is_err());
        assert!(parse(r#"{"action":"stop","time":"soon"}"#).is_err());
        assert!(parse("not json").is_err());
    }
}
