// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator API's resources (decision 0042): a watch from a list's version sees every change
//! after it and a bookmark while idle; a watch from no version starts with every object, and an
//! object that goes is deleted; a version too old or of another term is gone, a write against a
//! stale one conflicts; every error is one shape; and the published document is the one the
//! routes declare.

mod common;

use std::sync::Arc;
use std::time::Duration;

use everserve_controller::api::Pacing;
use everserve_controller::api::objects::{EventType, List, Node, WatchEvent};
use everserve_controller::coord::Coordinator;
use everserve_controller::http::Method;
use everserve_controller::policy;
use everserve_telemetry::Telemetry;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_util::sync::CancellationToken;

use common::api::OperatorApi;
use common::bounded;

/// A pacing a test can wait out: a change within 20 ms, a bookmark every 300 ms, `kept` changes.
fn quick(kept: usize) -> Pacing {
    Pacing {
        refresh: Duration::from_millis(20),
        bookmark: Duration::from_millis(300),
        kept,
    }
}

struct Rig {
    coord: Arc<Coordinator>,
    api: OperatorApi,
    address: everserve_wire::transport::Address,
    dir: tempfile::TempDir,
    cancel: CancellationToken,
}

async fn rig(pacing: Pacing) -> Rig {
    let dir = tempfile::tempdir().expect("tmp");
    let policy = policy::load(&common::example("policy-two-node.json")).expect("policy");
    let coord = Coordinator::new(policy, &Telemetry::new()).0;
    let cancel = CancellationToken::new();
    let address = common::serve(&coord, dir.path(), &cancel).await;
    let api = common::api::serve_paced(&coord, dir.path(), &cancel, pacing).await;
    Rig {
        coord,
        api,
        address,
        dir,
        cancel,
    }
}

impl Rig {
    /// A daemon for `node`, linked until `cancel` fires.
    async fn node(&self, node: &str, cancel: &CancellationToken) {
        let devices = common::devices(node, 2, "NVIDIA H100 80GB HBM3");
        common::daemon(self.dir.path(), node, &self.address, devices, cancel).await;
    }

    async fn call(&self, method: Method, path: &str, body: &str) -> (u16, String) {
        self.api.call(&self.api.operator, method, path, body).await
    }

    async fn nodes(&self) -> List<Node> {
        let (status, body) = self.call(Method::GET, "/v1/nodes", "").await;
        assert_eq!(status, 200, "{body}");
        serde_json::from_str(&body).expect("nodes")
    }

    /// Wait until the node list holds `n` nodes.
    async fn nodes_until(&self, n: usize) -> List<Node> {
        for _ in 0..500 {
            let list = self.nodes().await;
            if list.items.len() == n {
                return list;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("never {n} nodes: {:?}", self.nodes().await);
    }
}

/// A watch's lines as they come, its status line read first.
struct Watch {
    lines: tokio::io::Lines<BufReader<tokio_rustls::client::TlsStream<tokio::net::TcpStream>>>,
    status: u16,
}

/// Watch `path` as the operator, over HTTP/1.0 so the lines come unchunked.
async fn watch(api: &OperatorApi, path: &str) -> Watch {
    let mut conn = common::api::operator_connection(api).await;
    conn.write_all(format!("GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n").as_bytes())
        .await
        .expect("send");
    let mut lines = BufReader::new(conn).lines();
    let head = lines.next_line().await.expect("read").expect("a status");
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("a status");
    // the headers, to the blank line
    while let Some(l) = lines.next_line().await.expect("read") {
        if l.is_empty() {
            break;
        }
    }
    Watch { lines, status }
}

impl Watch {
    /// The next event, within five seconds.
    async fn next(&mut self) -> WatchEvent {
        let line = tokio::time::timeout(Duration::from_secs(5), self.lines.next_line())
            .await
            .expect("an event in time")
            .expect("read")
            .expect("a line, not the end");
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line}"))
    }

    /// The next event that is no bookmark.
    async fn next_change(&mut self) -> WatchEvent {
        loop {
            let e = self.next().await;
            if e.kind != EventType::Bookmark {
                return e;
            }
        }
    }

    /// Whether the server ends the watch within five seconds.
    async fn ends(&mut self) -> bool {
        tokio::time::timeout(Duration::from_secs(5), async {
            while let Ok(Some(_)) = self.lines.next_line().await {}
        })
        .await
        .is_ok()
    }
}

fn version(object: &serde_json::Value) -> &str {
    object["metadata"]["resourceVersion"]
        .as_str()
        .expect("a version")
}

#[tokio::test]
async fn a_watch_from_a_lists_version_sees_each_change_after_it_and_a_bookmark_while_idle() {
    bounded(Box::pin(async {
        let r = rig(quick(4096)).await;
        r.node("node-a", &r.cancel).await;
        let list = r.nodes_until(1).await;
        let mut w = watch(
            &r.api,
            &format!(
                "/v1/nodes?watch=true&resourceVersion={}",
                list.metadata.resource_version
            ),
        )
        .await;
        assert_eq!(w.status, 200);

        let (status, body) = r
            .call(
                Method::PUT,
                "/v1/nodes/node-a",
                r#"{"spec":{"unschedulable":true,"reason":"firmware"}}"#,
            )
            .await;
        assert_eq!(status, 200, "{body}");
        let written: serde_json::Value = serde_json::from_str(&body).expect("node");
        let e = w.next_change().await;
        assert_eq!(e.kind, EventType::Modified, "{e:?}");
        assert_eq!(e.object["spec"]["unschedulable"], true, "{e:?}");
        // the watch's object is the one the write answered, at the same version
        let (_, now) = r.call(Method::GET, "/v1/nodes/node-a", "").await;
        let now: serde_json::Value = serde_json::from_str(&now).expect("node");
        assert_eq!(version(&e.object), version(&now));
        assert_eq!(e.object, now);
        assert_eq!(
            e.object["metadata"]["generation"],
            written["metadata"]["generation"]
        );
        assert_ne!(version(&e.object), list.metadata.resource_version);

        // nothing changes: a bookmark names the version the watch is current at
        let b = w.next().await;
        assert_eq!(b.kind, EventType::Bookmark, "{b:?}");
        assert_eq!(version(&b.object), version(&now));
        r.cancel.cancel();
    }))
    .await;
}

#[tokio::test]
async fn a_watch_from_no_version_starts_with_every_object_and_a_node_that_goes_is_deleted() {
    bounded(Box::pin(async {
        let r = rig(quick(4096)).await;
        let b_cancel = r.cancel.child_token();
        r.node("node-a", &r.cancel).await;
        r.node("node-b", &b_cancel).await;
        r.nodes_until(2).await;
        let mut w = watch(&r.api, "/v1/nodes?watch=true").await;
        assert_eq!(w.status, 200);
        let mut added = Vec::new();
        for _ in 0..2 {
            let e = w.next_change().await;
            assert_eq!(e.kind, EventType::Added, "{e:?}");
            added.push(
                e.object["metadata"]["name"]
                    .as_str()
                    .expect("name")
                    .to_owned(),
            );
        }
        assert_eq!(added, ["node-a", "node-b"]);

        // node-b's daemon goes; it has no spec and holds nothing, so the node goes with it
        b_cancel.cancel();
        let e = w.next_change().await;
        assert_eq!(e.kind, EventType::Deleted, "{e:?}");
        assert_eq!(e.object["metadata"]["name"], "node-b");
        r.cancel.cancel();
    }))
    .await;
}

#[tokio::test]
async fn a_version_too_old_or_of_another_term_is_gone_and_a_stale_write_conflicts() {
    bounded(Box::pin(async {
        // two changes kept
        let r = rig(quick(2)).await;
        r.node("node-a", &r.cancel).await;
        let first = r.nodes_until(1).await.metadata.resource_version;
        for on in [true, false, true] {
            let body = format!(r#"{{"spec":{{"unschedulable":{on}}}}}"#);
            let (status, answer) = r.call(Method::PUT, "/v1/nodes/node-a", &body).await;
            assert_eq!(status, 200, "{answer}");
        }
        let w = watch(
            &r.api,
            &format!("/v1/nodes?watch=true&resourceVersion={first}"),
        )
        .await;
        assert_eq!(w.status, 410, "three changes since, two kept");
        let w = watch(&r.api, "/v1/nodes?watch=true&resourceVersion=99.1").await;
        assert_eq!(w.status, 410, "another term's version");
        let (status, body) = r
            .call(Method::GET, "/v1/nodes?watch=true&resourceVersion=x", "")
            .await;
        assert_eq!(status, 422, "{body}");

        // a write against the version it read applies; against an older one, it conflicts
        let (_, now) = r.call(Method::GET, "/v1/nodes/node-a", "").await;
        let now: serde_json::Value = serde_json::from_str(&now).expect("node");
        let stale = format!(
            r#"{{"metadata":{{"resourceVersion":"{first}"}},"spec":{{"unschedulable":false}}}}"#
        );
        let (status, body) = r.call(Method::PUT, "/v1/nodes/node-a", &stale).await;
        assert_eq!(status, 409, "{body}");
        assert!(body.contains(r#""reason":"Conflict""#), "{body}");
        let current = format!(
            r#"{{"metadata":{{"resourceVersion":"{}"}},"spec":{{"unschedulable":false}}}}"#,
            version(&now)
        );
        let (status, body) = r.call(Method::PUT, "/v1/nodes/node-a", &current).await;
        assert_eq!(status, 200, "{body}");
        r.cancel.cancel();
    }))
    .await;
}

/// `path` as a list, answered 200.
async fn page(r: &Rig, path: &str) -> List<serde_json::Value> {
    let (status, body) = r.call(Method::GET, path, "").await;
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).expect("list")
}

fn names(list: &List<serde_json::Value>) -> Vec<&str> {
    list.items
        .iter()
        .map(|o| o["metadata"]["name"].as_str().expect("name"))
        .collect()
}

#[tokio::test]
async fn a_paged_list_carries_its_first_version_and_a_watch_from_it_replays_what_changed_between() {
    bounded(Box::pin(async {
        let r = rig(quick(4096)).await;
        for node in ["node-a", "node-b", "node-c"] {
            r.node(node, &r.cancel).await;
        }
        r.nodes_until(3).await;
        let first = page(&r, "/v1/nodes?limit=2").await;
        assert_eq!(names(&first), ["node-a", "node-b"]);
        let token = first.metadata.continue_token.clone().expect("a next page");
        // between the pages, one node already read changes and one not yet read does
        for node in ["node-a", "node-c"] {
            let (status, body) = r
                .call(
                    Method::PUT,
                    &format!("/v1/nodes/{node}"),
                    r#"{"spec":{"unschedulable":true}}"#,
                )
                .await;
            assert_eq!(status, 200, "{body}");
        }
        let second = page(&r, &format!("/v1/nodes?limit=2&continue={token}")).await;
        assert_eq!(names(&second), ["node-c"]);
        assert_eq!(second.metadata.continue_token, None, "the last page");
        assert_eq!(
            second.metadata.resource_version, first.metadata.resource_version,
            "every page carries the first page's version"
        );
        // a watch from that version replays both changes: node-a's, which the first page missed
        let mut w = watch(
            &r.api,
            &format!(
                "/v1/nodes?watch=true&resourceVersion={}",
                first.metadata.resource_version
            ),
        )
        .await;
        assert_eq!(w.status, 200);
        let mut changed = Vec::new();
        for _ in 0..2 {
            let e = w.next_change().await;
            assert_eq!(e.kind, EventType::Modified, "{e:?}");
            changed.push(
                e.object["metadata"]["name"]
                    .as_str()
                    .expect("name")
                    .to_owned(),
            );
        }
        assert_eq!(changed, ["node-a", "node-c"]);
        r.cancel.cancel();
    }))
    .await;
}

#[tokio::test]
async fn a_page_after_its_first_version_expired_is_gone_and_a_token_not_given_is_refused() {
    bounded(Box::pin(async {
        // two changes kept
        let r = rig(quick(2)).await;
        for node in ["node-a", "node-b"] {
            r.node(node, &r.cancel).await;
        }
        r.nodes_until(2).await;
        let first = page(&r, "/v1/nodes?limit=1").await;
        let token = first.metadata.continue_token.expect("a next page");
        for on in [true, false, true] {
            let body = format!(r#"{{"spec":{{"unschedulable":{on}}}}}"#);
            let (status, answer) = r.call(Method::PUT, "/v1/nodes/node-a", &body).await;
            assert_eq!(status, 200, "{answer}");
        }
        let (status, body) = r
            .call(
                Method::GET,
                &format!("/v1/nodes?limit=1&continue={token}"),
                "",
            )
            .await;
        assert_eq!(
            status, 410,
            "three changes since, two kept: list again; {body}"
        );
        let (status, body) = r
            .call(Method::GET, "/v1/nodes?limit=1&continue=not-a-token", "")
            .await;
        assert_eq!(status, 422, "{body}");
        let (status, body) = r
            .call(Method::GET, "/v1/nodes?watch=true&limit=1", "")
            .await;
        assert_eq!(status, 422, "a watch is not paged; {body}");
        r.cancel.cancel();
    }))
    .await;
}

#[tokio::test]
async fn a_watch_ends_with_its_term_and_its_versions_are_gone_after() {
    bounded(Box::pin(async {
        let r = rig(quick(4096)).await;
        r.node("node-a", &r.cancel).await;
        let list = r.nodes_until(1).await;
        let mut w = watch(
            &r.api,
            &format!(
                "/v1/nodes?watch=true&resourceVersion={}",
                list.metadata.resource_version
            ),
        )
        .await;
        assert_eq!(w.status, 200);
        // the coordinator starts acting under a lease term: its versions start again
        r.coord
            .act_under(5, CancellationToken::new())
            .expect("acts");
        assert!(w.ends().await, "the watch of the old term ends");
        let w = watch(
            &r.api,
            &format!(
                "/v1/nodes?watch=true&resourceVersion={}",
                list.metadata.resource_version
            ),
        )
        .await;
        assert_eq!(w.status, 410);
        assert!(r.nodes().await.metadata.resource_version.starts_with("5."));
        r.cancel.cancel();
    }))
    .await;
}

#[tokio::test]
async fn every_error_is_one_shape() {
    bounded(Box::pin(async {
        let r = rig(quick(4096)).await;
        for (method, path, body, code, reason) in [
            (Method::GET, "/v1/nothing", "", 404, "NotFound"),
            (Method::GET, "/v1/engines/nope", "", 404, "NotFound"),
            (Method::PUT, "/v1/nodes/node-a", "{", 422, "Invalid"),
            (
                Method::PUT,
                "/v1/nodes/node-a",
                r#"{"spec":{"cordoned":true}}"#,
                422,
                "Invalid",
            ),
            (
                Method::PUT,
                "/v1/pools/p",
                r#"{"spec":{"engines":["nope"],"wants":1}}"#,
                422,
                "Invalid",
            ),
        ] {
            let (status, answer) = r.call(method.clone(), path, body).await;
            assert_eq!(status, code, "{method} {path}: {answer}");
            let error: serde_json::Value =
                serde_json::from_str(&answer).unwrap_or_else(|e| panic!("{e}: {answer}"));
            assert_eq!(error["code"], code, "{answer}");
            assert_eq!(error["reason"], reason, "{answer}");
            assert!(error["message"].is_string(), "{answer}");
        }
        let (status, answer) = r
            .api
            .call(&r.api.anonymous, Method::GET, "/v1/engines", "")
            .await;
        assert_eq!(status, 401);
        assert!(answer.contains(r#""reason":"Unauthorized""#), "{answer}");
        r.cancel.cancel();
    }))
    .await;
}

#[test]
fn the_published_document_is_the_one_the_routes_declare() {
    // `everserve-controller openapi > docs/api/openapi.json` after a route or a field changes
    let generated = everserve_controller::api::routes::document()
        .to_pretty_json()
        .expect("json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/api/openapi.json");
    let published = std::fs::read_to_string(&path).expect("docs/api/openapi.json");
    assert!(
        published.trim_end() == generated.trim_end(),
        "docs/api/openapi.json differs from what the routes declare: run \
         `cargo run -p everserve-controller -- openapi > docs/api/openapi.json`"
    );
}
