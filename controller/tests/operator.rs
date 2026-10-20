// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator's commands over the operator API (`everserve-controller get`, `watch`, `cordon`,
//! `drain`, `undrain`): a cordon and a drain written as a node's spec and read back, a drain of a
//! node holding nothing refused with the API's reason, a write never made over a change that
//! landed after the node was read, the objects as a table, a list read a page at a time, a watch
//! from every object, and no call without a credential.

mod common;

use std::sync::Arc;

use everserve_controller::api::objects::{List, Node};
use everserve_controller::config::{ApiAccessArgs, ObjectKind};
use everserve_controller::coord::Coordinator;
use everserve_controller::operator::{Operator, OperatorError, table};
use everserve_controller::policy;
use everserve_telemetry::Telemetry;
use tokio_util::sync::CancellationToken;

use common::bounded;

fn access(api: &common::api::OperatorApi) -> ApiAccessArgs {
    ApiAccessArgs {
        coordinator_api: api
            .base
            .trim_start_matches("https://")
            .replace("localhost", "127.0.0.1"),
        coordinator_name: "localhost".into(),
        ca: api.certs.join("ca.crt"),
        cert: Some(api.certs.join("operator-operator.crt")),
        key: Some(api.certs.join("operator-operator.key")),
        token_file: None,
    }
}

struct Rig {
    op: Operator,
    api: common::api::OperatorApi,
    cancel: CancellationToken,
    _dir: tempfile::TempDir,
}

async fn rig() -> Rig {
    let dir = tempfile::tempdir().expect("tmp");
    let policy = policy::load(&common::example("policy-two-node.json")).expect("policy");
    let coord: Arc<Coordinator> = Coordinator::new(policy, &Telemetry::new()).0;
    let cancel = CancellationToken::new();
    let api = common::api::serve(&coord, dir.path(), &cancel).await;
    let op = Operator::connect(&access(&api)).expect("credentials");
    Rig {
        op,
        api,
        cancel,
        _dir: dir,
    }
}

#[tokio::test]
async fn a_cordon_and_a_drain_are_the_nodes_spec_and_a_refusal_says_why() {
    bounded(Box::pin(async {
        let r = rig().await;
        let n =
            r.op.cordon("node-a", true, "firmware")
                .await
                .expect("cordoned");
        assert!(n.spec.unschedulable && n.spec.reason == "firmware", "{n:?}");
        assert_eq!(n.metadata.written_by, "operator api: operator");
        let n = r.op.cordon("node-a", false, "").await.expect("uncordoned");
        assert!(!n.spec.unschedulable);
        assert_eq!(n.metadata.generation, 2);
        // node-a holds no member: a drain of it is refused, and the reason is the API's
        match r.op.drain("node-a", 1, "maintenance").await {
            Err(OperatorError::Refused(e)) => {
                assert_eq!(e.code, 422);
                assert!(e.message.contains("holds no member"), "{e:?}");
            }
            other => panic!("{other:?}"),
        }
        let nodes: List<Node> = r.op.get("/v1/nodes").await.expect("nodes");
        assert_eq!(nodes.items.len(), 1);
        let printed = table(
            ObjectKind::Nodes,
            &[serde_json::to_value(&nodes.items[0]).expect("json")],
        );
        let lines: Vec<&str> = printed.lines().collect();
        assert!(
            lines[0].starts_with("NAME") && lines[0].contains("CORDONED"),
            "{printed}"
        );
        assert!(
            lines[1].starts_with("node-a") && lines[1].contains("false"),
            "{printed}"
        );
        r.cancel.cancel();
    }))
    .await;
}

// the other write runs while this test's task blocks inside the change: the server needs a
// thread of its own
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_is_never_made_over_a_change_that_landed_after_the_node_was_read() {
    bounded(Box::pin(async {
        let r = rig().await;
        r.op.cordon("node-a", true, "first")
            .await
            .expect("cordoned");
        let other = Operator::connect(&access(&r.api)).expect("credentials");
        // between this write's read and its write, another operator writes the node
        let written =
            r.op.write_node("node-a", |s| {
                std::thread::scope(|t| {
                    t.spawn(|| {
                        tokio::runtime::Runtime::new()
                            .expect("runtime")
                            .block_on(other.cordon("node-a", false, "second"))
                            .expect("the other write");
                    });
                });
                s.reason = "third".into();
            })
            .await;
        match written {
            Err(OperatorError::Refused(e)) => assert_eq!(e.code, 409, "{e:?}"),
            other => panic!("written over the other's: {other:?}"),
        }
        let n: Node = r.op.get("/v1/nodes/node-a").await.expect("node");
        assert_eq!(n.spec.reason, "second", "the other's write stands");
        r.cancel.cancel();
    }))
    .await;
}

#[tokio::test]
async fn a_watch_starts_with_every_object() {
    bounded(Box::pin(async {
        let r = rig().await;
        r.op.cordon("node-a", true, "").await.expect("cordoned");
        let mut changes = r.op.watch(ObjectKind::Nodes).await.expect("watch");
        let line = changes.line().await.expect("read").expect("a line");
        let event: serde_json::Value = serde_json::from_str(&line).expect("json");
        assert_eq!(event["type"], "ADDED");
        assert_eq!(event["object"]["metadata"]["name"], "node-a");
        r.cancel.cancel();
    }))
    .await;
}

#[tokio::test]
async fn a_list_is_read_a_page_at_a_time_into_one() {
    bounded(Box::pin(async {
        let r = rig().await;
        for node in ["node-a", "node-b", "node-c"] {
            r.op.cordon(node, true, "").await.expect("cordoned");
        }
        let list = r.op.list_by(ObjectKind::Nodes, 1).await.expect("listed");
        let names: Vec<&str> = list
            .items
            .iter()
            .map(|o| o["metadata"]["name"].as_str().expect("name"))
            .collect();
        assert_eq!(names, ["node-a", "node-b", "node-c"], "three pages of one");
        assert_eq!(list.metadata.continue_token, None);
        r.cancel.cancel();
    }))
    .await;
}

#[test]
fn no_call_is_made_without_a_credential() {
    let args = ApiAccessArgs {
        coordinator_api: "127.0.0.1:1".into(),
        coordinator_name: "coordinator".into(),
        ca: "ca.crt".into(),
        cert: None,
        key: None,
        token_file: None,
    };
    assert!(matches!(
        Operator::connect(&args),
        Err(OperatorError::Credentials(_))
    ));
}
