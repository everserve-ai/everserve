// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Azure Scheduled Events, from the instance metadata service.
//!
//! `GET /metadata/scheduledevents?api-version=2020-07-01` with `Metadata: true` answers a document
//! of events. Events are delivered to every VM in the availability set or placement group, so
//! only those whose `Resources` name this VM count; the name comes from
//! `/metadata/instance/compute/name` once, and a VM that cannot learn it takes every event. A
//! `Preempt` (spot eviction, 30 s minimum), `Terminate`, `Reboot` or `Redeploy` is a notice:
//! `Scheduled` at its `NotBefore` (an HTTP date, `Mon, 19 Sep 2016 18:29:47 GMT`), `Started` now.
//! The earliest deadline wins. A `Freeze` pauses the VM for seconds and is not a notice.

use super::http::Client;
use super::{NodeNotice, NoticeError, Settings, Source};
use crate::clock::now_ms;
use crate::proto::everserve::signals::v1::Reason;

/// The metadata service.
pub const DEFAULT_ENDPOINT: &str = "http://169.254.169.254";
/// Event types that take the node away.
const LEAVING: [&str; 4] = ["Preempt", "Terminate", "Reboot", "Redeploy"];

/// Poll once: the earliest leaving event naming this VM, if any. `vm_name` is learned on the
/// first poll that can and kept.
///
/// # Errors
///
/// [`NoticeError`] when the service cannot be reached or answers something else.
pub async fn poll(
    client: &Client,
    settings: &Settings,
    vm_name: &mut Option<String>,
) -> Result<Option<NodeNotice>, NoticeError> {
    let base = settings.endpoint.as_deref().unwrap_or(DEFAULT_ENDPOINT);
    let headers = [("Metadata", "true")];
    if vm_name.is_none() {
        let name = client
            .get(
                &format!(
                    "{base}/metadata/instance/compute/name?api-version=2021-02-01&format=text"
                ),
                &headers,
            )
            .await;
        match name {
            Ok(a) if a.status == 200 && !a.body.trim().is_empty() => {
                *vm_name = Some(a.body.trim().to_owned());
            }
            Ok(a) => tracing::warn!(
                "azure: this VM's name is unknown (instance metadata answered {}): every event counts",
                a.status
            ),
            Err(e) => tracing::warn!("azure: this VM's name is unknown ({e}): every event counts"),
        }
    }
    let answer = client
        .get(
            &format!("{base}/metadata/scheduledevents?api-version=2020-07-01"),
            &headers,
        )
        .await?;
    if answer.status != 200 {
        return Err(NoticeError::Endpoint {
            from: Source::Azure,
            reason: format!("scheduledevents answered {}", answer.status),
        });
    }
    parse(&answer.body, settings, vm_name.as_deref())
}

/// The earliest leaving event in a Scheduled Events document that names `vm_name` (any event
/// when the name is unknown).
///
/// # Errors
///
/// [`NoticeError::Parse`] when the body is not the documented JSON.
pub fn parse(
    body: &str,
    settings: &Settings,
    vm_name: Option<&str>,
) -> Result<Option<NodeNotice>, NoticeError> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| NoticeError::Parse {
        from: Source::Azure,
        reason: e.to_string(),
    })?;
    let events = v["Events"].as_array().ok_or_else(|| NoticeError::Parse {
        from: Source::Azure,
        reason: "no Events".into(),
    })?;
    let lead_ms = i64::try_from(settings.lead.as_millis()).unwrap_or(i64::MAX);
    let mut best: Option<NodeNotice> = None;
    for e in events {
        let kind = e["EventType"].as_str().unwrap_or("");
        let status = e["EventStatus"].as_str().unwrap_or("Scheduled");
        if !LEAVING.contains(&kind) {
            continue;
        }
        if let Some(name) = vm_name
            && let Some(resources) = e["Resources"].as_array()
            && !resources.iter().any(|r| r.as_str() == Some(name))
        {
            continue;
        }
        let not_before = e["NotBefore"].as_str().unwrap_or("").trim();
        let deadline_ms = if status == "Started" {
            now_ms()
        } else if not_before.is_empty() {
            now_ms().saturating_add(lead_ms)
        } else {
            let t = httpdate::parse_http_date(not_before).map_err(|_| NoticeError::Parse {
                from: Source::Azure,
                reason: format!("NotBefore {not_before:?} is not an HTTP date"),
            })?;
            t.duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        };
        if best.as_ref().is_none_or(|b| deadline_ms < b.deadline_ms) {
            best = Some(NodeNotice {
                deadline_ms,
                reason: Reason::Notice as i32,
                source: Source::Azure.to_string(),
                detail: format!(
                    "{kind} {} {}",
                    e["EventId"].as_str().unwrap_or(""),
                    if status == "Started" {
                        "started".to_owned()
                    } else if not_before.is_empty() {
                        "now".to_owned()
                    } else {
                        format!("not before {not_before}")
                    }
                ),
            });
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {

    use std::time::Duration;

    use super::parse;
    use crate::notice::{Settings, Source};

    fn settings() -> Settings {
        Settings {
            source: Source::Azure,
            endpoint: None,
            poll: Duration::from_secs(5),
            lead: Duration::from_secs(30),
            node: "n".into(),
            kubernetes: None,
        }
    }

    const DOC: &str = r#"{
      "DocumentIncarnation": 2,
      "Events": [
        {"EventId": "a", "EventType": "Freeze", "ResourceType": "VirtualMachine", "Resources": ["vm0"],
         "EventStatus": "Scheduled", "NotBefore": "Mon, 19 Sep 2016 18:29:47 GMT", "Description": "", "EventSource": "Platform", "DurationInSeconds": 9},
        {"EventId": "b", "EventType": "Preempt", "ResourceType": "VirtualMachine", "Resources": ["vm0"],
         "EventStatus": "Scheduled", "NotBefore": "Mon, 19 Sep 2016 18:30:17 GMT", "Description": "", "EventSource": "Platform", "DurationInSeconds": -1},
        {"EventId": "c", "EventType": "Reboot", "ResourceType": "VirtualMachine", "Resources": ["vm0"],
         "EventStatus": "Started", "NotBefore": "", "Description": "", "EventSource": "User", "DurationInSeconds": -1}
      ]}"#;

    #[test]
    fn the_earliest_leaving_event_naming_this_vm_is_the_notice() {
        let n = parse(DOC, &settings(), Some("vm0"))
            .unwrap()
            .expect("a notice");
        assert_eq!(n.deadline_ms, 1_474_309_817_000, "{n:?}");
        assert!(n.detail.starts_with("Preempt b"), "{n:?}");
        // the same document reaches every VM in the placement group: another VM's events are not
        // this node's
        assert_eq!(parse(DOC, &settings(), Some("vm1")).unwrap(), None);
        // a VM that cannot learn its name takes every event
        assert!(parse(DOC, &settings(), None).unwrap().is_some());
        assert_eq!(
            parse(
                r#"{"DocumentIncarnation": 1, "Events": []}"#,
                &settings(),
                Some("vm0")
            )
            .unwrap(),
            None
        );
        assert!(parse("{}", &settings(), Some("vm0")).is_err());
    }

    #[test]
    fn an_event_without_a_time_gets_the_lead_and_a_started_one_is_now() {
        let doc = r#"{"Events": [{"EventId": "d", "EventType": "Terminate", "EventStatus": "Scheduled", "NotBefore": "", "Resources": ["vm0"]}]}"#;
        let before = crate::clock::now_ms();
        let n = parse(doc, &settings(), Some("vm0"))
            .unwrap()
            .expect("a notice");
        assert!(
            n.deadline_ms >= before + 30_000 && n.deadline_ms <= before + 31_000,
            "{n:?}"
        );
        // a hardware failure skips Scheduled: the event arrives Started, and the node is going now
        let doc = r#"{"Events": [{"EventId": "e", "EventType": "Reboot", "EventStatus": "Started", "NotBefore": "", "Resources": ["vm0"]}]}"#;
        let n = parse(doc, &settings(), Some("vm0"))
            .unwrap()
            .expect("a notice");
        assert!(
            n.deadline_ms >= before && n.deadline_ms <= before + 1_000,
            "{n:?}"
        );
        assert!(n.detail.ends_with("started"), "{n:?}");
    }
}
