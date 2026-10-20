// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The cluster's Nodes as the coordinator reads them (decision 0041): each node's labels and
//! whether it is cordoned, kept current the way client-go's reflector keeps a cache. List every
//! Node, then watch from the list's version; a watch the server ends is opened again from the last
//! version seen, bookmarks included; a version the server has forgotten (410 Gone) lists again.
//! Every change sends the whole set, so a reader never pieces it together from events.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::Access;
use crate::http::{Client, HttpError};

/// How long the server holds one watch open before it ends it, and the reflector opens another.
const WATCH_SECONDS: u64 = 290;

/// How long a watch may stay silent, with no event and no bookmark, before it is taken as broken.
const SILENT: Duration = Duration::from_secs(WATCH_SECONDS + 30);

/// Nodes listed per page.
const PAGE: u32 = 500;

/// The wait after a failed list or watch before the next try, doubled to a minute.
const BACKOFF: Duration = Duration::from_secs(1);

/// What the coordinator reads of a Node.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeFacts {
    /// Its labels.
    pub labels: BTreeMap<String, String>,
    /// Whether it is cordoned (`spec.unschedulable`).
    pub unschedulable: bool,
}

/// Every Node, by name.
pub type Nodes = BTreeMap<String, NodeFacts>;

/// Why a list or a watch ended.
#[derive(Debug, thiserror::Error)]
enum Ended {
    #[error(transparent)]
    Http(#[from] HttpError),
    #[error("{0}")]
    Answer(String),
    /// The version watched from is gone: list again.
    #[error("the watched version expired")]
    Expired,
}

/// Keep `tx` holding every Node of the cluster `access` reaches until `cancel` fires.
pub async fn reflect(access: Access, tx: watch::Sender<Nodes>, cancel: CancellationToken) {
    let client = match access.client() {
        Ok(c) => c,
        Err(e) => {
            warn!("kubernetes nodes: {e}; not watching nodes");
            return;
        }
    };
    let mut wait = BACKOFF;
    loop {
        let run = async {
            let (nodes, version) = match list(&client, &access).await {
                Ok(listed) => listed,
                Err(e) => return e,
            };
            info!(
                "kubernetes nodes: {} listed at version {version}",
                nodes.len()
            );
            tx.send_replace(nodes);
            watch_from(&client, &access, version, &tx).await
        };
        let ended = tokio::select! {
            () = cancel.cancelled() => return,
            ended = run => ended,
        };
        match ended {
            Ended::Expired => {
                debug!("kubernetes nodes: the watched version expired; listing again");
                wait = BACKOFF;
                continue;
            }
            e => warn!("kubernetes nodes: {e}; listing again in {wait:?}"),
        }
        tokio::select! {
            () = cancel.cancelled() => return,
            () = tokio::time::sleep(wait) => {}
        }
        wait = (wait * 2).min(Duration::from_mins(1));
    }
}

/// Every Node, page by page, and the version the list was taken at.
async fn list(client: &Client, access: &Access) -> Result<(Nodes, String), Ended> {
    let mut nodes = Nodes::new();
    let mut next = String::new();
    loop {
        let url = if next.is_empty() {
            access.url(&format!("/api/v1/nodes?limit={PAGE}"))
        } else {
            access.url(&format!("/api/v1/nodes?limit={PAGE}&continue={next}"))
        };
        let answer = client.get(&url, &[("Accept", "application/json")]).await?;
        if answer.status == 410 {
            // a page's continue token outlived the list it continues
            return Err(Ended::Expired);
        }
        if answer.status != 200 {
            return Err(Ended::Answer(format!(
                "listing nodes answered {}: {}",
                answer.status, answer.body
            )));
        }
        let page: Value = serde_json::from_str(&answer.body)
            .map_err(|e| Ended::Answer(format!("a node list that is not JSON: {e}")))?;
        for item in page["items"].as_array().into_iter().flatten() {
            if let Some((name, facts)) = facts_of(item) {
                nodes.insert(name, facts);
            }
        }
        page["metadata"]["continue"]
            .as_str()
            .unwrap_or_default()
            .clone_into(&mut next);
        if next.is_empty() {
            let version = page["metadata"]["resourceVersion"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            return Ok((nodes, version));
        }
    }
}

/// Watch every Node from `version`, applying each event to what `tx` holds, until the server
/// ends the watch or forgets the version; the watch is opened again from the last version seen.
async fn watch_from(
    client: &Client,
    access: &Access,
    mut version: String,
    tx: &watch::Sender<Nodes>,
) -> Ended {
    loop {
        let url = access.url(&format!(
            "/api/v1/nodes?watch=1&allowWatchBookmarks=true&timeoutSeconds={WATCH_SECONDS}\
             &resourceVersion={version}"
        ));
        let mut stream = match client.stream(&url, &[("Accept", "application/json")]).await {
            Ok(s) => s,
            Err(e) => return e.into(),
        };
        if stream.status == 410 {
            return Ended::Expired;
        }
        if stream.status != 200 {
            return Ended::Answer(format!("watching nodes answered {}", stream.status));
        }
        loop {
            let line = match tokio::time::timeout(SILENT, stream.line()).await {
                Err(_) => return Ended::Answer(format!("the watch was silent for {SILENT:?}")),
                Ok(Err(e)) => return e.into(),
                // the server ended the watch: open another from where this one got to
                Ok(Ok(None)) => break,
                Ok(Ok(Some(line))) => line,
            };
            if line.trim().is_empty() {
                continue;
            }
            let Ok(event) = serde_json::from_str::<Value>(&line) else {
                return Ended::Answer(format!("a watch event that is not JSON: {line}"));
            };
            let object = &event["object"];
            if let Some(v) = object["metadata"]["resourceVersion"].as_str() {
                v.clone_into(&mut version);
            }
            match event["type"].as_str().unwrap_or_default() {
                "ADDED" | "MODIFIED" => {
                    if let Some((name, facts)) = facts_of(object) {
                        tx.send_if_modified(|nodes| {
                            nodes.insert(name, facts.clone()).as_ref() != Some(&facts)
                        });
                    }
                }
                "DELETED" => {
                    if let Some(name) = object["metadata"]["name"].as_str() {
                        tx.send_if_modified(|nodes| nodes.remove(name).is_some());
                    }
                }
                "BOOKMARK" => {}
                "ERROR" if object["code"].as_u64() == Some(410) => return Ended::Expired,
                other => {
                    return Ended::Answer(format!("watch event {other}: {object}"));
                }
            }
        }
    }
}

/// A Node object's name and what the coordinator reads of it.
fn facts_of(node: &Value) -> Option<(String, NodeFacts)> {
    let name = node["metadata"]["name"].as_str()?.to_owned();
    let labels = node["metadata"]["labels"]
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_owned())))
                .collect()
        })
        .unwrap_or_default();
    let unschedulable = node["spec"]["unschedulable"].as_bool().unwrap_or(false);
    Some((
        name,
        NodeFacts {
            labels,
            unschedulable,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::facts_of;

    #[test]
    fn a_node_object_gives_its_name_labels_and_cordon() {
        let node = serde_json::json!({
            "metadata": {"name": "gpu-7", "labels": {"zone": "a", "n": 3}},
            "spec": {"unschedulable": true}
        });
        let (name, facts) = facts_of(&node).expect("facts");
        assert_eq!(name, "gpu-7");
        assert_eq!(facts.labels.get("zone").map(String::as_str), Some("a"));
        assert!(
            !facts.labels.contains_key("n"),
            "a label's value is a string"
        );
        assert!(facts.unschedulable);
        assert!(facts_of(&serde_json::json!({"metadata": {}})).is_none());
    }
}
