// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Kubernetes: this node's taints, from the API server.
//!
//! `GET /api/v1/nodes/{name}` with the service account's bearer token. Two taints say the node
//! is leaving: `ToBeDeletedByClusterAutoscaler`, whose value is the Unix time of the deletion, and
//! `node.kubernetes.io/unschedulable` (a cordon, also `spec.unschedulable`), which gives the
//! settings' lead with reason `OPERATOR`: the operator cordons a node to drain it. A cloud's own
//! preemption on Kubernetes still arrives through that cloud's source; this one is the operator's
//! and the autoscaler's signal.

use super::http::Client;
use super::{NodeNotice, NoticeError, Settings, Source};
use crate::clock::now_ms;
use crate::proto::everserve::signals::v1::Reason;

/// The autoscaler's deletion taint; its value is Unix seconds.
pub const AUTOSCALER_TAINT: &str = "ToBeDeletedByClusterAutoscaler";
/// The cordon taint.
pub const CORDON_TAINT: &str = "node.kubernetes.io/unschedulable";

/// Poll once: the notice this node's taints carry, if any.
///
/// # Errors
///
/// [`NoticeError`] when the API server cannot be reached, refuses, or answers something else.
pub async fn poll(client: &Client, settings: &Settings) -> Result<Option<NodeNotice>, NoticeError> {
    let access = settings.kubernetes.as_ref().ok_or(NoticeError::Endpoint {
        from: Source::Kubernetes,
        reason: "no Kubernetes API access given".into(),
    })?;
    let answer = client
        .get(
            &access.url(&format!("/api/v1/nodes/{}", settings.node)),
            &[("Accept", "application/json")],
        )
        .await?;
    if answer.status != 200 {
        return Err(NoticeError::Endpoint {
            from: Source::Kubernetes,
            reason: format!("nodes/{} answered {}", settings.node, answer.status),
        });
    }
    parse(&answer.body, settings)
}

/// The notice in a Node object's taints.
///
/// # Errors
///
/// [`NoticeError::Parse`] when the body is not a Node.
pub fn parse(body: &str, settings: &Settings) -> Result<Option<NodeNotice>, NoticeError> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| NoticeError::Parse {
        from: Source::Kubernetes,
        reason: e.to_string(),
    })?;
    if v["kind"].as_str() != Some("Node") {
        return Err(NoticeError::Parse {
            from: Source::Kubernetes,
            reason: format!("kind {} is not Node", v["kind"]),
        });
    }
    let spec = &v["spec"];
    let taints = spec["taints"].as_array().cloned().unwrap_or_default();
    if let Some(t) = taints
        .iter()
        .find(|t| t["key"].as_str() == Some(AUTOSCALER_TAINT))
    {
        let value = t["value"].as_str().unwrap_or("");
        let secs: i64 = value.parse().map_err(|_| NoticeError::Parse {
            from: Source::Kubernetes,
            reason: format!("{AUTOSCALER_TAINT} value {value:?} is not Unix seconds"),
        })?;
        return Ok(Some(NodeNotice {
            deadline_ms: secs.saturating_mul(1000),
            reason: Reason::Notice as i32,
            source: Source::Kubernetes.to_string(),
            detail: format!("{AUTOSCALER_TAINT}={value}"),
        }));
    }
    let cordoned = spec["unschedulable"].as_bool() == Some(true)
        || taints
            .iter()
            .any(|t| t["key"].as_str() == Some(CORDON_TAINT));
    if cordoned {
        let lead_ms = i64::try_from(settings.lead.as_millis()).unwrap_or(i64::MAX);
        return Ok(Some(NodeNotice {
            deadline_ms: now_ms().saturating_add(lead_ms),
            reason: Reason::Operator as i32,
            source: Source::Kubernetes.to_string(),
            detail: "node cordoned".into(),
        }));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {

    use std::time::Duration;

    use super::parse;
    use crate::notice::{Settings, Source};
    use crate::proto::everserve::signals::v1::Reason;

    fn settings() -> Settings {
        Settings {
            source: Source::Kubernetes,
            endpoint: None,
            poll: Duration::from_secs(5),
            lead: Duration::from_secs(30),
            node: "node-a".into(),
            kubernetes: None,
        }
    }

    #[test]
    fn the_autoscaler_taint_gives_its_time_and_a_cordon_gives_the_lead() {
        let doc = r#"{"kind":"Node","metadata":{"name":"node-a"},"spec":{"taints":[
            {"key":"nvidia.com/gpu","value":"present","effect":"NoSchedule"},
            {"key":"ToBeDeletedByClusterAutoscaler","value":"1791286878","effect":"NoSchedule"}]}}"#;
        let n = parse(doc, &settings()).unwrap().expect("a notice");
        assert_eq!(n.deadline_ms, 1_791_286_878_000);
        assert_eq!(n.reason, Reason::Notice as i32);
        let doc = r#"{"kind":"Node","spec":{"unschedulable":true,"taints":[
            {"key":"node.kubernetes.io/unschedulable","effect":"NoSchedule"}]}}"#;
        let before = crate::clock::now_ms();
        let n = parse(doc, &settings()).unwrap().expect("a notice");
        assert!(n.deadline_ms >= before + 30_000, "{n:?}");
        assert_eq!(n.reason, Reason::Operator as i32);
        assert_eq!(
            parse(r#"{"kind":"Node","spec":{}}"#, &settings()).unwrap(),
            None
        );
        assert!(parse(r#"{"kind":"Status","code":403}"#, &settings()).is_err());
        let doc = r#"{"kind":"Node","spec":{"taints":[{"key":"ToBeDeletedByClusterAutoscaler","value":"soon"}]}}"#;
        assert!(parse(doc, &settings()).is_err());
    }
}
