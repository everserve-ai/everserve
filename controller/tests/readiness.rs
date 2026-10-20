// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The readiness API beside `/metrics`: a node is reclaimable only when a spare stands for every
//! rank on it; a drain is a notice to every rank on the node and one change for the set at the
//! deadline; the spare gauge follows the pool; the last measured pause becomes the expected one.
//! A node's spec (decision 0036): a cordoned node's spares take no position, drains wait their
//! turn under the policy's concurrent losses, in deadline order across a restart, and a drain
//! whose change was refused is blocked, then withdrawn or given another deadline. An engine's
//! conditions and event log follow its changes.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use everserve_controller::api::objects::{Engine, List, Node, NodeEngine, NodeRelease, Pool};
use everserve_controller::coord::PoolView;
use everserve_controller::coord::{Cause, Coordinator};
use everserve_controller::policy;
use everserve_controller::proto::everserve::signals::v1::{
    Member, Reason, Register, Role, Signal, SlotChoice, signal, slot_choice,
};
use everserve_controller::readiness::EngineReadiness;
use everserve_controller::spec::{FileStore, MemoryStore, SpecStore};
use everserve_controller::status::{Condition, Event, Severity, Truth};
use everserve_controller::topology::Device;
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::{Address, Listener};
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

const H100: &str = "NVIDIA H100 80GB HBM3";

struct Cluster {
    coord: Arc<Coordinator>,
    telemetry: Telemetry,
    http: common::api::OperatorApi,
    /// Stops the daemons.
    cancel: CancellationToken,
    /// Stops the coordinator and its HTTP port, on their own so a test can restart them.
    coord_cancel: CancellationToken,
    address: Address,
    dir: tempfile::TempDir,
}

impl Cluster {
    fn stop(&self) {
        self.coord_cancel.cancel();
        self.cancel.cancel();
    }
}

async fn cluster(policy_file: &str) -> Cluster {
    cluster_with(policy_file, Arc::new(MemoryStore::default())).await
}

async fn cluster_with(policy_file: &str, store: Arc<dyn SpecStore>) -> Cluster {
    cluster_of(example(policy_file), store).await
}

fn example(policy_file: &str) -> everserve_controller::proto::everserve::policy::v1::Policy {
    policy::load(&common::example(policy_file)).expect("policy")
}

async fn cluster_of(
    policy: everserve_controller::proto::everserve::policy::v1::Policy,
    store: Arc<dyn SpecStore>,
) -> Cluster {
    let dir = tempfile::tempdir().expect("tmp");
    let telemetry = Telemetry::new();
    let (coord, _rx) = Coordinator::with_store(policy, &telemetry, store, 0)
        .await
        .expect("store");
    let cancel = CancellationToken::new();
    let coord_cancel = CancellationToken::new();
    let address = common::serve(&coord, dir.path(), &coord_cancel).await;
    // the operator API as an operator reaches it, over TLS behind its gate (decision 0039)
    let http = common::api::serve(&coord, dir.path(), &coord_cancel).await;
    Cluster {
        coord,
        telemetry,
        http,
        cancel,
        coord_cancel,
        address,
        dir,
    }
}

async fn node(cl: &Cluster, node: &str, devices: u32) -> PathBuf {
    node_until(cl, node, devices, cl.cancel.clone()).await
}

/// A daemon for `node` that stops when `cancel` fires.
async fn node_until(cl: &Cluster, node: &str, devices: u32, cancel: CancellationToken) -> PathBuf {
    let devices = common::devices(node, devices, H100);
    common::daemon(cl.dir.path(), node, &cl.address, devices, &cancel)
        .await
        .1
}

/// `method path` with a JSON `body`, as an operator: the status and the body.
async fn http(
    api: &common::api::OperatorApi,
    method: &str,
    path: &str,
    body: &str,
) -> (u16, String) {
    let method = everserve_controller::http::Method::from_bytes(method.as_bytes()).expect("method");
    api.call(&api.operator, method, path, body).await
}

async fn get<T: serde::de::DeserializeOwned>(cl: &Cluster, path: &str) -> T {
    let (status, body) = http(&cl.http, "GET", path, "").await;
    assert_eq!(status, 200, "{path}: {body}");
    serde_json::from_str(&body).unwrap_or_else(|e| panic!("{path}: {e}: {body}"))
}

/// Every engine's status, as the API lists the engines.
async fn engines(cl: &Cluster) -> Vec<EngineReadiness> {
    let list: List<Engine> = get(cl, "/v1/engines").await;
    list.items.into_iter().map(|e| e.status).collect()
}

/// Write `node`'s spec as `change` makes it from the spec the API shows (decision 0042): the
/// status and the body.
async fn write_node(
    cl: &Cluster,
    node: &str,
    change: impl FnOnce(&mut serde_json::Value),
) -> (u16, String) {
    let (status, body) = http(&cl.http, "GET", &format!("/v1/nodes/{node}"), "").await;
    let mut spec = if status == 200 {
        serde_json::from_str::<serde_json::Value>(&body).expect("node")["spec"].clone()
    } else {
        serde_json::json!({})
    };
    change(&mut spec);
    let write = serde_json::json!({ "spec": spec }).to_string();
    http(&cl.http, "PUT", &format!("/v1/nodes/{node}"), &write).await
}

/// Cordon `node`, or uncordon it, for `reason`.
async fn cordon(cl: &Cluster, node: &str, on: bool, reason: &str) -> (u16, String) {
    write_node(cl, node, |s| {
        s["unschedulable"] = on.into();
        s["reason"] = reason.into();
    })
    .await
}

/// Drain `node` by `lead_ms` from now.
async fn drain(cl: &Cluster, node: &str, lead_ms: i64) -> (u16, String) {
    let deadline = everserve_controller::clock::now_ms() + lead_ms;
    write_node(cl, node, |s| {
        s["unschedulable"] = true.into();
        s["drain"] = serde_json::json!({ "deadlineMs": deadline });
    })
    .await
}

/// Withdraw `node`'s drain.
async fn undrain(cl: &Cluster, node: &str) -> (u16, String) {
    write_node(cl, node, |s| {
        if let Some(s) = s.as_object_mut() {
            s.remove("drain");
        }
    })
    .await
}

/// What a drain's write answered, as the node it returned shows it.
#[derive(Debug)]
struct Drained {
    deadline_ms: i64,
    waiting: bool,
    engines: Vec<NodeEngine>,
}

fn drained(body: &str) -> Drained {
    let node: Node = serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"));
    Drained {
        deadline_ms: node.spec.drain.expect("a drain").deadline_ms,
        waiting: draining(&node).as_deref() == Some("waiting"),
        engines: node.status.engines,
    }
}

/// Where `node`'s drain stands, from its Draining condition: `scheduled`, `waiting`,
/// `blocked`, or None.
fn draining(node: &Node) -> Option<String> {
    node.status
        .conditions
        .iter()
        .find(|c| c.kind == "Draining" && c.status == Truth::True)
        .map(|c| c.reason.to_lowercase())
}

async fn wait_for(
    cl: &Cluster,
    what: &str,
    ok: impl Fn(&EngineReadiness) -> bool,
) -> EngineReadiness {
    for _ in 0..300 {
        if let Some(e) = cl.coord.readiness().await.engines.into_iter().next()
            && ok(&e)
        {
            return e;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("never {what}: {:?}", cl.coord.readiness().await);
}

#[tokio::test]
async fn a_node_is_reclaimable_when_a_spare_stands_for_every_rank_on_it() {
    let cl = cluster("policy-two-node.json").await;
    let sock_a = node(&cl, "node-a", 4).await;
    let sock_b = node(&cl, "node-b", 4).await;
    let sock_c = node(&cl, "node-c", 4).await;
    let mut workers = Vec::new();
    for r in 0..4 {
        workers.push(common::register(&sock_a, r, r, Role::Worker).await);
        workers.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    let e = wait_for(&cl, "saw eight members", |e| e.members.len() == 8).await;
    assert_eq!(e.engine, "k3-a");
    assert_eq!((e.epoch, e.world_size), (0, 8));
    assert_eq!(e.nodes.len(), 2, "{e:?}");
    for n in &e.nodes {
        assert!(!n.reclaimable, "no spares yet: {n:?}");
        assert_eq!(n.outcome, "rebuild");
        assert!(n.why.contains("no spare"), "{n:?}");
        assert_eq!(n.min_notice_ms, 2000);
        assert_eq!(n.expected_pause_ms, None);
    }
    assert_eq!(e.nodes[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(e.nodes[1].ranks, vec![4, 5, 6, 7]);
    assert!(
        e.members.iter().all(|m| m.class.as_deref() == Some(H100)),
        "{e:?}"
    );

    // four spares on the spare node, for node-a's ranks
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(common::register(&sock_c, r, r, Role::Spare).await);
    }
    let e = wait_for(&cl, "saw four spares", |e| e.spares.len() == 4).await;
    assert!(
        e.spares
            .iter()
            .all(|s| s.node == "node-c" && s.state == "warm"),
        "{e:?}"
    );
    let a = e.nodes.iter().find(|n| n.node == "node-a").expect("node-a");
    assert!(
        a.reclaimable && a.outcome == "replace" && a.why.is_empty(),
        "{a:?}"
    );
    let b = e.nodes.iter().find(|n| n.node == "node-b").expect("node-b");
    assert!(!b.reclaimable && b.outcome == "rebuild", "{b:?}");
    assert!(
        cl.telemetry
            .render()
            .contains("everserve_spares{engine=\"k3-a\",state=\"warm\"} 4"),
        "{}",
        cl.telemetry.render()
    );

    // over HTTP: every engine, one engine, a missing engine, the topology
    assert_eq!(engines(&cl).await, vec![e.clone()]);
    let one: Engine = get(&cl, "/v1/engines/k3-a").await;
    assert_eq!(one.status, e);
    assert_eq!(one.metadata.name, "k3-a");
    assert_eq!(
        one.spec["name"], "k3-a",
        "its spec is its entry in the policy"
    );
    let (status, body) = http(&cl.http, "GET", "/v1/engines/nope", "").await;
    assert_eq!(status, 404);
    assert!(body.contains(r#""reason":"NotFound""#), "{body}");
    let devices: Vec<Device> = get(&cl, "/v1/topology").await;
    assert_eq!(devices.len(), 12);
    assert!(
        devices
            .iter()
            .all(|d| d.class == H100 && d.domain == d.node)
    );
    let (status, _) = http(&cl.http, "GET", "/metrics", "").await;
    assert_eq!(
        status, 404,
        "metrics are served apart from the operator API (decision 0039)"
    );
    cl.stop();
}

/// What the controllers did through a drain's whole path, as metrics (decision 0036): its engine
/// reconcile, an empty engine queue, the engine's and the node's condition series, the operator's
/// spec write and the node controller's reconciles.
fn assert_controller_metrics(metrics: &str) {
    for line in [
        "everserve_reconciles_total{controller=\"engine\",outcome=\"started\"} 1",
        "everserve_queue_depth{controller=\"engine\"} 0",
        "everserve_engine_condition{engine=\"k3-a\",status=\"True\",type=\"Serving\"} 1",
        "everserve_engine_condition{engine=\"k3-a\",status=\"False\",type=\"Serving\"} 0",
        "everserve_spec_writes_total{kind=\"node\",result=\"written\",writer=\"operator_api\"} 1",
        "everserve_node_condition{node=\"node-a\",status=\"False\",type=\"Draining\"} 1",
    ] {
        assert!(metrics.contains(line), "{line}: {metrics}");
    }
    for series in [
        "everserve_reconcile_seconds_count{controller=\"engine\"}",
        "everserve_queue_wait_seconds_count{controller=\"engine\"}",
        "everserve_reconciles_total{controller=\"node\"",
    ] {
        assert!(metrics.contains(series), "{series}: {metrics}");
    }
}

#[tokio::test]
async fn a_drain_notices_every_rank_on_the_node_and_changes_them_as_one_set() {
    let cl = cluster("policy-two-node.json").await;
    let sock_a = node(&cl, "node-a", 4).await;
    let sock_b = node(&cl, "node-b", 4).await;
    let sock_c = node(&cl, "node-c", 4).await;
    let mut on_a = Vec::new();
    let mut on_b = Vec::new();
    for r in 0..4 {
        on_a.push(common::register(&sock_a, r, r, Role::Worker).await);
        on_b.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    wait_for(&cl, "saw eight members", |e| e.members.len() == 8).await;
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(common::register(&sock_c, r, r, Role::Spare).await);
    }
    wait_for(&cl, "saw four spares", |e| e.spares.len() == 4).await;

    // a drain with no time, one leaving its node schedulable, one of a node holding nothing,
    // then the real one
    let (status, body) = write_node(&cl, "node-a", |s| {
        s["unschedulable"] = true.into();
        s["drain"] = serde_json::json!({ "deadlineMs": 0 });
    })
    .await;
    assert_eq!(status, 422, "{body}");
    let (status, body) = write_node(&cl, "node-a", |s| {
        s["drain"] = serde_json::json!({ "deadlineMs": everserve_controller::clock::now_ms() });
    })
    .await;
    assert_eq!(status, 422, "a drained node is cordoned: {body}");
    let (status, body) = drain(&cl, "node-x", 10).await;
    assert_eq!(status, 422, "{body}");
    assert!(body.contains("holds no member"), "{body}");
    let before = everserve_controller::clock::now_ms();
    let (status, body) = drain(&cl, "node-a", 300).await;
    assert_eq!(status, 200, "{body}");
    let d = drained(&body);
    assert!(
        d.deadline_ms >= before + 300 && d.deadline_ms < before + 2000,
        "{d:?}"
    );
    assert_eq!(d.engines.len(), 1);
    assert_eq!(d.engines[0].engine, "k3-a");
    assert_eq!(d.engines[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(d.engines[0].outcome, "replace");
    assert!(!d.waiting, "no other drain moves anything: {d:?}");

    // every worker and spare hears four notices, then one change for the set at epoch 1
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        let lost = common::expect_notices_then_change(w, rank, 4, 1, Reason::Operator).await;
        assert_eq!(lost, vec![0, 1, 2, 3]);
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        let lost = common::expect_notices_then_change(s, rank, 4, 1, Reason::Operator).await;
        assert_eq!(lost, vec![0, 1, 2, 3]);
    }
    // node-a's workers heard the notices; they leave at the deadline and are not delivered to
    for (i, w) in on_a.iter_mut().enumerate() {
        for _ in 0..4 {
            let sig: Signal = common::recv(w).await.expect("notice");
            assert!(
                matches!(sig.kind, Some(signal::Kind::Notice(_))),
                "rank {i}: {sig:?}"
            );
        }
    }

    let e = wait_for(&cl, "committed epoch 1", |e| {
        e.epoch == 1 && e.spares.is_empty()
    })
    .await;
    assert_eq!(e.members.len(), 8);
    assert!(
        e.members
            .iter()
            .filter(|m| m.rank < 4)
            .all(|m| m.node == "node-c"),
        "node-a's ranks now serve from node-c: {e:?}"
    );
    assert!(e.nodes.iter().all(|n| n.node != "node-a"), "{e:?}");
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert!(view.last_pause_ms.contains_key("replace"), "{view:?}");
    assert!(
        cl.telemetry
            .render()
            .contains("everserve_spares{engine=\"k3-a\",state=\"warm\"} 0"),
        "{}",
        cl.telemetry.render()
    );
    assert!(engines(&cl).await.iter().all(|e| e.change.is_none()));
    assert_controller_metrics(&cl.telemetry.render());
    cl.stop();
}

/// `node` as the node list shows it: where its drain stands, its deadline, its cordon.
#[derive(Debug)]
struct Listed {
    drain: Option<String>,
    drain_by_ms: Option<i64>,
    cordoned: bool,
    conditions: Vec<Condition>,
}

async fn listed(cl: &Cluster, node: &str) -> Listed {
    let nodes: List<Node> = get(cl, "/v1/nodes").await;
    let n = nodes
        .items
        .into_iter()
        .find(|n| n.metadata.name == node)
        .unwrap_or_else(|| panic!("{node} not listed"));
    Listed {
        drain: draining(&n),
        drain_by_ms: n.spec.drain.map(|d| d.deadline_ms),
        cordoned: n.spec.unschedulable,
        conditions: n.status.conditions,
    }
}

/// Eight workers on node-a and node-b, and node-a's four spares on node-c, as the two-node policy
/// places them.
async fn two_node_engine(cl: &Cluster) -> (Vec<UnixStream>, Vec<UnixStream>, Vec<UnixStream>) {
    let sock_a = node(cl, "node-a", 4).await;
    let sock_b = node(cl, "node-b", 4).await;
    let sock_c = node(cl, "node-c", 4).await;
    let mut on_a = Vec::new();
    let mut on_b = Vec::new();
    for r in 0..4 {
        on_a.push(common::register(&sock_a, r, r, Role::Worker).await);
        on_b.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    wait_for(cl, "saw eight members", |e| e.members.len() == 8).await;
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(common::register(&sock_c, r, r, Role::Spare).await);
    }
    wait_for(cl, "saw four spares", |e| e.spares.len() == 4).await;
    (on_a, on_b, spares)
}

#[tokio::test]
async fn a_node_is_released_once_it_holds_no_position_and_a_waiting_question_hears_it() {
    // decision 0038: maintenance asks whether a node may be given back, and may wait for it
    let cl = cluster("policy-two-node.json").await;
    let (mut on_a, mut on_b, mut spares) = two_node_engine(&cl).await;
    let r: NodeRelease = get(&cl, "/v1/nodes/node-a/release").await;
    assert!(!r.releasable, "{r:?}");
    assert!(
        r.why
            .iter()
            .any(|w| w.contains("ranks [0, 1, 2, 3] of k3-a")),
        "{r:?}"
    );
    // spares standing for positions hold none: their node may go
    let r: NodeRelease = get(&cl, "/v1/nodes/node-c/release").await;
    assert!(r.releasable && r.why.is_empty(), "{r:?}");
    // a short wait on a node that stays held answers held, after the wait
    let started = std::time::Instant::now();
    let r: NodeRelease = get(&cl, "/v1/nodes/node-a/release?wait_ms=300").await;
    assert!(
        !r.releasable && started.elapsed() >= Duration::from_millis(300),
        "{r:?}"
    );
    // asked to wait, then node-a's workers go and node-c's spares take its ranks: the question
    // is answered releasable once the change commits
    let waiting = http(
        &cl.http,
        "GET",
        "/v1/nodes/node-a/release?wait_ms=20000",
        "",
    );
    let change = async {
        on_a.clear();
        let within = |f| tokio::time::timeout(Duration::from_secs(10), f);
        for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
            within(common::expect_notices_then_change(
                w,
                rank,
                0,
                1,
                Reason::Operator,
            ))
            .await
            .expect("the change");
        }
        for (rank, s) in (0u32..).zip(spares.iter_mut()) {
            within(common::expect_notices_then_change(
                s,
                rank,
                0,
                1,
                Reason::Operator,
            ))
            .await
            .expect("the change");
        }
    };
    let ((status, body), ()) = tokio::join!(waiting, change);
    assert_eq!(status, 200, "{body}");
    let r: NodeRelease = serde_json::from_str(&body).expect("release");
    assert!(r.releasable, "{r:?}");
    cl.stop();
}

#[tokio::test]
async fn a_cordoned_nodes_spares_take_no_position() {
    let cl = cluster("policy-two-node.json").await;
    let (_on_a, _on_b, _spares) = two_node_engine(&cl).await;

    let (status, body) = cordon(&cl, "node-c", true, "firmware").await;
    assert_eq!(status, 200, "{body}");
    let n: Node = serde_json::from_str(&body).expect("node");
    assert_eq!(
        (n.metadata.generation, n.spec.unschedulable, n.spec.drain),
        (1, true, None),
        "{n:?}"
    );
    assert_eq!(
        (n.spec.reason.as_str(), n.metadata.written_by.as_str()),
        ("firmware", "operator api: operator")
    );
    assert!(listed(&cl, "node-c").await.cordoned);
    let e = wait_for(&cl, "saw the cordon", |e| {
        e.spares.iter().all(|s| s.state == "cordoned")
    })
    .await;
    let a = e.nodes.iter().find(|n| n.node == "node-a").expect("node-a");
    assert!(!a.reclaimable && a.outcome == "rebuild", "{a:?}");
    // the cordon is a change to every engine's status: its gauge and its Covered condition
    // follow it at once, or the gauge reads warm spares on cordoned nodes
    let metrics = cl.telemetry.render();
    for line in [
        "everserve_spares{engine=\"k3-a\",state=\"cordoned\"} 4",
        "everserve_spares{engine=\"k3-a\",state=\"warm\"} 0",
    ] {
        assert!(metrics.contains(line), "{line}: {metrics}");
    }
    let covered = e
        .conditions
        .iter()
        .find(|c| c.kind == "Covered")
        .expect("Covered");
    assert!(covered.message.contains("node-a"), "{covered:?}");

    // uncordoned, the spares stand again
    let (status, body) = cordon(&cl, "node-c", false, "").await;
    assert_eq!(status, 200, "{body}");
    let n: Node = serde_json::from_str(&body).expect("node");
    assert_eq!((n.metadata.generation, n.spec.unschedulable), (2, false));
    let e = wait_for(&cl, "saw the uncordon", |e| {
        e.spares.iter().all(|s| s.state == "warm")
    })
    .await;
    let a = e.nodes.iter().find(|n| n.node == "node-a").expect("node-a");
    assert!(a.reclaimable && a.outcome == "replace", "{a:?}");

    // cordoned again, a change for node-a's ranks does not take them: the claim refuses, not
    // only the view
    let (status, body) = cordon(&cl, "node-c", true, "").await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = drain(&cl, "node-a", 50).await;
    assert_eq!(status, 200, "{body}");
    let d = drained(&body);
    assert_eq!(d.engines[0].outcome, "rebuild", "{d:?}");
    let refused = "everserve_changes_total{cause=\"notice\",engine=\"k3-a\",outcome=\"rebuild\",\
                   result=\"refused_rebuild\"} 1";
    for _ in 0..300 {
        if cl.telemetry.render().contains(refused) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        cl.telemetry.render().contains(refused),
        "{}",
        cl.telemetry.render()
    );
    let e = cl.coord.readiness().await.engines.remove(0);
    assert_eq!(e.epoch, 0, "{e:?}");
    assert!(e.members.iter().all(|m| m.node != "node-c"), "{e:?}");
    cl.stop();
}

#[tokio::test]
async fn a_drain_waits_its_turn_under_the_concurrent_losses_and_cannot_move_once_underway() {
    // the two-node policy allows four concurrent losses: node-a's four and node-b's four are
    // eight
    let cl = cluster("policy-two-node.json").await;
    let (mut on_a, mut on_b, mut spares) = two_node_engine(&cl).await;
    let (status, body) = drain(&cl, "node-a", 300).await;
    assert_eq!(status, 200, "{body}");
    let a = drained(&body);
    assert!(!a.waiting, "{a:?}");
    let (status, body) = drain(&cl, "node-b", 300).await;
    assert_eq!(status, 200, "{body}");
    let b = drained(&body);
    assert!(b.waiting, "four more would be eight at once: {b:?}");
    assert_eq!(
        listed(&cl, "node-a").await.drain.as_deref(),
        Some("scheduled")
    );
    let listed_b = listed(&cl, "node-b").await;
    assert_eq!(
        (
            listed_b.drain.as_deref(),
            listed_b.drain_by_ms,
            listed_b.cordoned
        ),
        (Some("waiting"), Some(b.deadline_ms), true)
    );

    // node-a's drain is underway: its deadline neither moves nor clears, and the same spec
    // written again changes nothing
    let (status, body) = drain(&cl, "node-a", 60_000).await;
    assert_eq!(status, 409, "{body}");
    assert!(body.contains(r#""reason":"DrainUnderway""#), "{body}");
    let (status, body) = undrain(&cl, "node-a").await;
    assert_eq!(status, 409, "{body}");
    let (status, body) = write_node(&cl, "node-a", |_| {}).await;
    assert_eq!(status, 200, "{body}");
    let n: Node = serde_json::from_str(&body).expect("node");
    assert_eq!(n.metadata.generation, 1, "no write: {n:?}");

    // node-b's workers and the spares hear node-a's four notices and nothing of node-b's before
    // node-a's change
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        assert_eq!(
            common::expect_notices_then_change(w, rank, 4, 1, Reason::Operator).await,
            vec![0, 1, 2, 3]
        );
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        assert_eq!(
            common::expect_notices_then_change(s, rank, 4, 1, Reason::Operator).await,
            vec![0, 1, 2, 3]
        );
    }
    for w in &mut on_a {
        for _ in 0..4 {
            let sig: Signal = common::recv(w).await.expect("notice");
            assert!(matches!(sig.kind, Some(signal::Kind::Notice(_))), "{sig:?}");
        }
    }
    // node-a emptied: its drain is done and node-b's starts, its notices to its own ranks
    wait_for(&cl, "committed epoch 1", |e| e.epoch == 1).await;
    // its deadline has passed, so its change runs at once, and no spare stands for node-b's
    // ranks: it is refused and the drain blocked
    wait_node(&cl, "node-b", "started", |n| {
        n.drain.as_deref() == Some("blocked")
    })
    .await;
    let reasons = event_reasons(&cl, "/v1/nodes/node-b/events").await;
    let waited = reasons
        .iter()
        .position(|r| r == "DrainWaiting")
        .expect("waited");
    let started = reasons
        .iter()
        .position(|r| r == "DrainStarted")
        .expect("started");
    assert!(waited < started, "{reasons:?}");
    assert_eq!(listed(&cl, "node-a").await.drain, None);
    let sig: Signal = common::recv(&mut on_b[0]).await.expect("notice");
    match sig.kind {
        Some(signal::Kind::Notice(n)) => {
            assert!((4..8).contains(&n.rank), "{n:?}");
            assert_eq!(n.deadline_ms, b.deadline_ms);
        }
        other => panic!("{other:?}"),
    }
    cl.stop();
}

#[tokio::test]
async fn a_stored_drain_resumes_when_a_restarted_coordinator_hears_its_node() {
    let dir = tempfile::tempdir().expect("tmp");
    let file = dir.path().join("specs.pb");
    let cl = cluster_with("policy-two-node.json", Arc::new(FileStore::new(&file))).await;
    let (_on_a, _on_b, _spares) = two_node_engine(&cl).await;
    let deadline = everserve_controller::clock::now_ms() + 600_000;
    let (status, body) = write_node(&cl, "node-a", |s| {
        s["unschedulable"] = true.into();
        s["drain"] = serde_json::json!({ "deadlineMs": deadline });
        s["reason"] = "kernel upgrade".into();
    })
    .await;
    assert_eq!(status, 200, "{body}");
    let d = drained(&body);

    // the coordinator dies; a new one reads the store and, once node-a's daemon says hello
    // again, the drain stands as it did
    cl.coord_cancel.cancel();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-two-node.json");
    let (coord2, _rx) = Coordinator::with_store(
        policy::load(&path).expect("policy"),
        &Telemetry::new(),
        Arc::new(FileStore::new(&file)),
        0,
    )
    .await
    .expect("store");
    let spec = coord2.node_spec("node-a").await.expect("the stored spec");
    assert_eq!(spec.written_by, "operator api: operator");
    let listener = Listener::bind(&cl.address, None).await.expect("rebind");
    let cancel2 = CancellationToken::new();
    tokio::spawn(Arc::clone(&coord2).serve(listener, cancel2.clone()));
    let mut resumed = None;
    for _ in 0..1200 {
        if let Some(n) = coord2
            .node_views()
            .await
            .into_iter()
            .find(|n| n.node == "node-a")
            && n.drain.as_deref() == Some("scheduled")
        {
            resumed = Some(n);
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let n = resumed.expect("node-a's drain resumed");
    let Some(everserve_controller::proto::everserve::spec::v1::resource_spec::Kind::Node(spec)) =
        n.spec.and_then(|s| s.kind)
    else {
        panic!("node-a's spec");
    };
    assert_eq!(
        (spec.drain_by_ms, spec.unschedulable),
        (d.deadline_ms, true)
    );
    cancel2.cancel();
    cl.stop();
}

fn condition<'a>(conditions: &'a [Condition], kind: &str) -> &'a Condition {
    conditions
        .iter()
        .find(|c| c.kind == kind)
        .unwrap_or_else(|| panic!("no {kind} in {conditions:?}"))
}

/// The reasons of the events `path` returns, oldest first.
async fn event_reasons(cl: &Cluster, path: &str) -> Vec<String> {
    let events: Vec<Event> = get(cl, path).await;
    events.into_iter().map(|e| e.reason).collect()
}

/// Wait until `node` lists `ok`.
async fn wait_node(cl: &Cluster, node: &str, what: &str, ok: impl Fn(&Listed) -> bool) {
    for _ in 0..300 {
        if ok(&listed(cl, node).await) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{node} never {what}: {:?}", listed(cl, node).await);
}

#[tokio::test]
async fn an_engines_conditions_and_events_follow_its_changes() {
    let cl = cluster("policy-two-node.json").await;
    let (mut on_a, mut on_b, mut spares) = two_node_engine(&cl).await;
    let e = cl.coord.readiness().await.engines.remove(0);
    let serving = condition(&e.conditions, "Serving");
    assert_eq!(
        (serving.status, serving.reason.as_str()),
        (Truth::True, "Members")
    );
    // the spares stand for node-a's ranks only: a loss of node-b would rebuild
    let covered = condition(&e.conditions, "Covered");
    assert_eq!(
        (covered.status, covered.reason.as_str()),
        (Truth::False, "NodeUncovered")
    );
    assert!(
        covered.message.contains("node-b") && !covered.message.contains("node-a"),
        "{covered:?}"
    );
    for (kind, status) in [
        ("Progressing", Truth::False),
        ("Degraded", Truth::False),
        ("Down", Truth::False),
    ] {
        assert_eq!(condition(&e.conditions, kind).status, status, "{kind}");
    }
    let (status, _) = http(&cl.http, "GET", "/v1/engines/nope/events", "").await;
    assert_eq!(status, 404);

    let (status, body) = drain(&cl, "node-a", 50).await;
    assert_eq!(status, 200, "{body}");
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        common::expect_notices_then_change(w, rank, 4, 1, Reason::Operator).await;
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        common::expect_notices_then_change(s, rank, 4, 1, Reason::Operator).await;
    }
    for w in &mut on_a {
        for _ in 0..4 {
            let _: Signal = common::recv(w).await.expect("notice");
        }
    }
    let e = wait_for(&cl, "committed epoch 1", |e| {
        e.epoch == 1 && condition(&e.conditions, "Progressing").status == Truth::False
    })
    .await;
    let reasons = event_reasons(&cl, "/v1/engines/k3-a/events").await;
    let started = reasons
        .iter()
        .position(|r| r == "ChangeStarted")
        .expect("started");
    let committed = reasons
        .iter()
        .position(|r| r == "ChangeCommitted")
        .expect("committed");
    assert!(started < committed, "{reasons:?}");
    // Progressing turned true at the claim and false at the commit: two events, after the start
    assert_eq!(
        reasons
            .iter()
            .filter(|r| *r == "ChangeInFlight" || *r == "Idle")
            .count(),
        2,
        "{reasons:?}"
    );
    let progressing = condition(&e.conditions, "Progressing");
    let events: Vec<Event> = get(&cl, "/v1/engines/k3-a/events").await;
    assert_eq!(
        progressing.last_transition_ms,
        events
            .iter()
            .rfind(|e| e.reason == "Idle")
            .expect("idle")
            .at_ms
    );
    // the spares were spent: node-c's loss now rebuilds too
    assert!(
        condition(&e.conditions, "Covered")
            .message
            .contains("node-c"),
        "{e:?}"
    );
    cl.stop();
}

#[tokio::test]
async fn a_refused_drain_is_blocked_then_withdrawn_or_given_another_deadline() {
    let cl = cluster("policy-two-node.json").await;
    let (mut on_a, mut on_b, mut spares) = two_node_engine(&cl).await;
    // node-c cordoned: node-a's change at the deadline is refused as REBUILD
    let (status, body) = cordon(&cl, "node-c", true, "").await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = drain(&cl, "node-a", 50).await;
    assert_eq!(status, 200, "{body}");
    wait_node(&cl, "node-a", "blocked", |n| {
        n.drain.as_deref() == Some("blocked")
    })
    .await;
    let n = listed(&cl, "node-a").await;
    let degraded = condition(&n.conditions, "Degraded");
    assert_eq!(
        (degraded.status, degraded.reason.as_str()),
        (Truth::True, "DrainBlocked")
    );
    assert!(degraded.message.contains("REBUILD"), "{degraded:?}");
    let events: Vec<Event> = get(&cl, "/v1/nodes/node-a/events").await;
    assert!(
        events
            .iter()
            .any(|e| e.reason == "DrainBlocked" && e.severity == Severity::Warning),
        "{events:?}"
    );
    let reasons = event_reasons(&cl, "/v1/engines/k3-a/events").await;
    assert!(reasons.iter().any(|r| r == "RefusedRebuild"), "{reasons:?}");

    // withdrawn: no drain, not degraded, still cordoned
    let (status, body) = undrain(&cl, "node-a").await;
    assert_eq!(status, 200, "{body}");
    let n = listed(&cl, "node-a").await;
    assert_eq!(
        (n.drain.as_deref(), n.drain_by_ms, n.cordoned),
        (None, None, true),
        "{n:?}"
    );
    assert_eq!(condition(&n.conditions, "Degraded").status, Truth::False);

    // node-c back in the pool and node-a drained again: its change commits
    let (status, body) = cordon(&cl, "node-c", false, "").await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = drain(&cl, "node-a", 50).await;
    assert_eq!(status, 200, "{body}");
    // two drains' notices, then the one change
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        common::expect_notices_then_change(w, rank, 8, 1, Reason::Operator).await;
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        common::expect_notices_then_change(s, rank, 8, 1, Reason::Operator).await;
    }
    for w in &mut on_a {
        for _ in 0..8 {
            let _: Signal = common::recv(w).await.expect("notice");
        }
    }
    wait_node(&cl, "node-a", "drained", |n| {
        n.drain.is_none() && n.drain_by_ms.is_some()
    })
    .await;
    let reasons = event_reasons(&cl, "/v1/nodes/node-a/events").await;
    let started = reasons
        .iter()
        .rposition(|r| r == "DrainStarted")
        .expect("started");
    assert!(
        reasons[started..].iter().any(|r| r == "DrainDone"),
        "{reasons:?}"
    );
    cl.stop();
}

#[tokio::test]
async fn a_blocked_drain_goes_by_itself_once_what_refused_it_clears() {
    let cl = cluster("policy-two-node.json").await;
    let (mut on_a, mut on_b, mut spares) = two_node_engine(&cl).await;
    // node-c cordoned: node-a's change at the deadline is refused as REBUILD, and blocked
    let (status, body) = cordon(&cl, "node-c", true, "").await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = drain(&cl, "node-a", 50).await;
    assert_eq!(status, 200, "{body}");
    wait_node(&cl, "node-a", "blocked", |n| {
        n.drain.as_deref() == Some("blocked")
    })
    .await;
    // node-c back in the pool, the drain left as it is: tried again after its backoff, the
    // change goes through, with no notice sent twice
    let (status, body) = cordon(&cl, "node-c", false, "").await;
    assert_eq!(status, 200, "{body}");
    let within = |f| tokio::time::timeout(Duration::from_secs(10), f);
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        within(common::expect_notices_then_change(
            w,
            rank,
            4,
            1,
            Reason::Operator,
        ))
        .await
        .expect("node-a's change, tried again");
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        within(common::expect_notices_then_change(
            s,
            rank,
            4,
            1,
            Reason::Operator,
        ))
        .await
        .expect("node-a's change, tried again");
    }
    for w in &mut on_a {
        for _ in 0..4 {
            let _: Signal = common::recv(w).await.expect("notice");
        }
    }
    wait_node(&cl, "node-a", "drained", |n| n.drain.is_none()).await;
    let reasons = event_reasons(&cl, "/v1/nodes/node-a/events").await;
    assert!(
        reasons.iter().any(|r| r == "DrainRetried") && reasons.iter().any(|r| r == "DrainDone"),
        "{reasons:?}"
    );
    cl.stop();
}

#[tokio::test]
async fn drains_of_two_engines_neither_wait_on_nor_count_against_each_other() {
    // k3-a on node-a and node-b, as the example; k3-b the same on node-d and node-e
    let mut policy = example("policy-two-node.json");
    let mut b = policy.engines[0].clone();
    b.name = "k3-b".into();
    for m in &mut b.members {
        m.node = if m.node == "node-a" {
            "node-d"
        } else {
            "node-e"
        }
        .into();
    }
    b.spare_nodes = vec!["node-f".into()];
    b.replica_nodes = vec!["node-f".into()];
    policy.engines.push(b);
    let cl = cluster_of(policy, Arc::new(MemoryStore::default())).await;
    let mut workers = Vec::new();
    for (engine, nodes) in [
        ("k3-a", ["node-a", "node-b"]),
        ("k3-b", ["node-d", "node-e"]),
    ] {
        for (i, n) in nodes.iter().enumerate() {
            let sock = node(&cl, n, 4).await;
            for d in 0..4u32 {
                let rank = u32::try_from(i).expect("index") * 4 + d;
                workers.push(
                    common::register_as(
                        &sock,
                        Register {
                            engine: engine.into(),
                            ..common::registration(rank, d, Role::Worker)
                        },
                    )
                    .await,
                );
            }
        }
    }
    for _ in 0..300 {
        let r = cl.coord.readiness().await;
        if r.engines.len() == 2 && r.engines.iter().all(|e| e.members.len() == 8) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // node-b's drain is scheduled first, four of k3-a's positions; the policy allows four at once
    let (status, body) = drain(&cl, "node-b", 600_000).await;
    assert_eq!(status, 200, "{body}");
    assert!(!drained(&body).waiting, "{body}");
    // node-e's drain, later, moves four of k3-b's: k3-a's drain is no concern of it
    let (status, body) = drain(&cl, "node-e", 900_000).await;
    assert_eq!(status, 200, "{body}");
    assert!(
        !drained(&body).waiting,
        "another engine's drain held it: {body}"
    );
    // node-a's, of k3-a again, waits: eight of k3-a's positions at once
    let (status, body) = drain(&cl, "node-a", 1_200_000).await;
    assert_eq!(status, 200, "{body}");
    assert!(drained(&body).waiting, "{body}");
    cl.stop();
}

#[tokio::test]
async fn stored_drains_resume_in_deadline_order_whichever_node_says_hello_first() {
    let dir = tempfile::tempdir().expect("tmp");
    let file = dir.path().join("specs.pb");
    let cl = cluster_with("policy-two-node.json", Arc::new(FileStore::new(&file))).await;
    let b_cancel = cl.cancel.child_token();
    let sock_a = node(&cl, "node-a", 4).await;
    let sock_b = node_until(&cl, "node-b", 4, b_cancel.clone()).await;
    let mut workers = Vec::new();
    for r in 0..4 {
        workers.push(common::register(&sock_a, r, r, Role::Worker).await);
        workers.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    wait_for(&cl, "saw eight members", |e| e.members.len() == 8).await;
    // node-b is due first and starts; node-a waits: eight positions at once, the policy allows 4
    let (status, body) = drain(&cl, "node-b", 600_000).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = drain(&cl, "node-a", 900_000).await;
    assert_eq!(status, 200, "{body}");
    let a = drained(&body);
    assert!(a.waiting, "{a:?}");

    // the coordinator restarts, and only node-a says hello to the new one: node-b's drain,
    // stored and due first, keeps its turn
    cl.coord_cancel.cancel();
    b_cancel.cancel();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-two-node.json");
    let (coord2, _rx) = Coordinator::with_store(
        policy::load(&path).expect("policy"),
        &Telemetry::new(),
        Arc::new(FileStore::new(&file)),
        0,
    )
    .await
    .expect("store");
    let listener = Listener::bind(&cl.address, None).await.expect("rebind");
    let cancel2 = CancellationToken::new();
    tokio::spawn(Arc::clone(&coord2).serve(listener, cancel2.clone()));
    let mut heard = false;
    for _ in 0..1200 {
        if coord2
            .node_views()
            .await
            .iter()
            .any(|n| n.node == "node-a" && n.linked)
            && !coord2.readiness().await.engines.is_empty()
        {
            heard = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(heard, "node-a re-helloed");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let n = coord2
        .node_views()
        .await
        .into_iter()
        .find(|n| n.node == "node-a")
        .expect("node-a");
    assert_eq!(n.drain.as_deref(), Some("waiting"), "{n:?}");
    let events = coord2.node_events("node-a").await;
    assert!(
        events
            .iter()
            .any(|e| e.reason == "DrainWaiting" && e.message.contains("node-b")),
        "{events:?}"
    );
    cancel2.cancel();
    cl.stop();
}

#[tokio::test]
async fn the_engine_clients_slot_choice_is_in_the_readiness_view_and_survives_a_coordinator_restart()
 {
    let cl = cluster("policy-two-node.json").await;
    let sock_a = node(&cl, "node-a", 4).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let e = wait_for(&cl, "saw a member", |e| e.members.len() == 1).await;
    assert!(e.slots.is_empty(), "no client yet");
    // the engine's client registers with the choice the executor made
    let mut c = UnixStream::connect(&sock_a).await.expect("connect");
    frame::send(
        &mut c,
        &Signal {
            kind: Some(signal::Kind::Register(Register {
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                engine: "k3-a".into(),
                rank: 0,
                member: Some(Member {
                    node: "n".into(),
                    device: 0,
                }),
                role: Role::Client as i32,
                slot: Some(SlotChoice {
                    variant: slot_choice::Variant::Host as i32,
                    how: slot_choice::How::Probe as i32,
                    detail: "no answer from devices [2] in 15.0s".into(),
                    seconds: 15.2,
                }),
                compile_cache: None,
                any_rank: false,
                launcher_pid: 0,
                identity: Some(common::identity()),
                stage: Vec::new(),
                host: String::new(),
            })),
        },
    )
    .await
    .expect("register");
    let e = wait_for(&cl, "saw the slot choice", |e| !e.slots.is_empty()).await;
    let slot = e.slots.get("node-a").cloned().expect("node-a's slot");
    assert_eq!(
        (slot.variant.as_str(), slot.how.as_str()),
        ("host", "probe")
    );
    assert!(slot.detail.contains("devices [2]"));
    assert!((slot.seconds - 15.2).abs() < 1e-9);
    let over_http: Engine = get(&cl, "/v1/engines/k3-a").await;
    assert_eq!(over_http.status.slots.get("node-a"), Some(&slot));

    // the coordinator dies; a new one on the same address learns the choice from the hello the
    // daemon sends when it reconnects (with backoff, up to five seconds between tries)
    cl.coord_cancel.cancel();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let telemetry = Telemetry::new();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-two-node.json");
    let (coord2, _rx) = Coordinator::new(policy::load(&path).expect("policy"), &telemetry);
    let listener = Listener::bind(&cl.address, None).await.expect("rebind");
    let cancel2 = CancellationToken::new();
    tokio::spawn(Arc::clone(&coord2).serve(listener, cancel2.clone()));
    for _ in 0..1200 {
        if coord2
            .readiness()
            .await
            .engines
            .first()
            .is_some_and(|e| !e.slots.is_empty())
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let e = coord2
        .readiness()
        .await
        .engines
        .into_iter()
        .next()
        .expect("the daemon re-helloed the new coordinator");
    assert_eq!(e.slots.get("node-a"), Some(&slot), "{e:?}");
    cancel2.cancel();
    cl.stop();
}

#[tokio::test]
async fn a_spares_slot_choice_is_the_spare_nodes_entry_in_the_view() {
    // decision 0024: a spare on a spare node decides that node's variant itself and registers it
    let cl = cluster("policy-two-node.json").await;
    let sock_a = node(&cl, "node-a", 4).await;
    let sock_c = node(&cl, "node-c", 4).await;
    let _w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    wait_for(&cl, "saw a member", |e| e.members.len() == 1).await;
    let mut s = UnixStream::connect(&sock_c).await.expect("connect");
    frame::send(
        &mut s,
        &Signal {
            kind: Some(signal::Kind::Register(Register {
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                engine: "k3-a".into(),
                rank: 4,
                member: Some(Member {
                    node: "n".into(),
                    device: 0,
                }),
                role: Role::Spare as i32,
                slot: Some(SlotChoice {
                    variant: slot_choice::Variant::P2p as i32,
                    how: slot_choice::How::Cache as i32,
                    detail: "kept from 2026-10-07".into(),
                    seconds: 0.01,
                }),
                compile_cache: None,
                any_rank: false,
                launcher_pid: 0,
                identity: Some(common::identity()),
                stage: Vec::new(),
                host: String::new(),
            })),
        },
    )
    .await
    .expect("register");
    let e = wait_for(&cl, "saw node-c's slot", |e| e.slots.contains_key("node-c")).await;
    let slot = &e.slots["node-c"];
    assert_eq!((slot.variant.as_str(), slot.how.as_str()), ("p2p", "cache"));
    assert!(
        !e.slots.contains_key("node-a"),
        "node-a's client has not registered: {e:?}"
    );
    cl.stop();
}

/// Answer the stacks request `s`'s process is asked on its link with `text`, as a process does.
async fn answer_stacks(s: &mut tokio::net::UnixStream, text: &str) {
    use everserve_controller::proto::everserve::signals::v1::{Signal, StacksDumped, signal};
    let request_id = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let sig: Signal = common::recv(s).await.expect("frame");
            if let Some(signal::Kind::DumpStacks(asked)) = sig.kind {
                return asked.request_id;
            }
        }
    })
    .await
    .expect("the process was asked for its stacks on its link");
    let answer = Signal {
        kind: Some(signal::Kind::StacksDumped(StacksDumped {
            request_id,
            text: text.to_owned(),
        })),
    };
    frame::send(s, &answer).await.expect("answer");
}

#[tokio::test]
async fn a_stacks_request_reaches_the_process_its_daemon_saw_register() {
    use everserve_controller::coord::{StacksAt, StacksError};

    // this test's own process registers as rank 0, which readiness names by the socket's peer
    // credentials; the request reaches it on its link (decision 0041), never as a signal
    let cl = cluster("policy-pilot.json").await;
    let sock = node(&cl, "node-a", 2).await;
    let mut w0 = common::register(&sock, 0, 0, Role::Worker).await;
    let e = wait_for(&cl, "rank 0's process", |e| {
        e.members.iter().any(|m| m.rank == 0 && m.process.is_some())
    })
    .await;
    let me = std::process::id();
    let m0 = e.members.iter().find(|m| m.rank == 0).expect("rank 0");
    assert_eq!(m0.process.as_ref().and_then(|p| p.pid), Some(me), "{e:?}");

    // the process answers with its stacks, which reach whoever asked: its standard error may
    // reach no log, as a vLLM worker's in a pod does not
    let asked = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.stacks("k3-a", StacksAt::Rank(0)).await }
    });
    answer_stacks(
        &mut w0,
        "Thread 0x1 (most recent call first):\n  File \"x.py\"",
    )
    .await;
    let taken = asked.await.expect("join").expect("stacks");
    assert_eq!((taken.node.as_str(), taken.pid), ("node-a", Some(me)));
    assert!(taken.answered);
    assert!(taken.stacks.starts_with("Thread 0x1"), "{taken:?}");

    assert!(matches!(
        cl.coord.stacks("k3-a", StacksAt::Rank(7)).await,
        Err(StacksError::NotFound(_))
    ));
    // over HTTP: a request names a rank, or a node and a device
    let (status, _) = http(&cl.http, "POST", "/v1/engines/k3-a/stacks", "{}").await;
    assert_eq!(status, 422);
    let (status, _) = http(&cl.http, "POST", "/v1/engines/nope/stacks", r#"{"rank":0}"#).await;
    assert_eq!(status, 404);
    cl.stop();
}

#[tokio::test]
async fn readiness_names_the_head_reaches_it_for_stacks_and_says_down_when_it_is_lost() {
    use everserve_controller::coord::StacksAt;

    // decision 0032: this test's own process registers as the engine's head, its own launcher
    let cl = cluster("policy-pilot.json").await;
    let sock = node(&cl, "node-a", 2).await;
    let _w0 = common::register(&sock, 0, 0, Role::Worker).await;
    let me = std::process::id();
    let mut head = common::register_head(&sock, me).await;
    let e = wait_for(&cl, "the head", |e| e.head.is_some()).await;
    let h = e.head.as_ref().expect("head");
    assert_eq!(
        (h.node.as_str(), h.pid, h.launcher_pid),
        ("node-a", Some(me), me),
        "{e:?}"
    );
    assert!(e.down.is_none(), "{e:?}");

    let asked = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.stacks("k3-a", StacksAt::Head).await }
    });
    answer_stacks(&mut head, "the head's stacks").await;
    let taken = asked.await.expect("join").expect("stacks");
    assert_eq!((taken.node.as_str(), taken.pid), ("node-a", Some(me)));
    assert_eq!(
        (taken.answered, taken.stacks.as_str()),
        (true, "the head's stacks")
    );
    // a process that does not answer, as a build that writes its stacks to its own log only:
    // asked all the same, and said so
    let (status, body) = http(
        &cl.http,
        "POST",
        "/v1/engines/k3-a/stacks",
        r#"{"head":true}"#,
    )
    .await;
    assert_eq!(status, 202);
    let taken: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert_eq!(
        (&taken["answered"], &taken["stacks"]),
        (&false.into(), &"".into()),
        "{body}"
    );
    let (status, _) = http(
        &cl.http,
        "POST",
        "/v1/engines/k3-a/stacks",
        r#"{"head":true,"rank":0}"#,
    )
    .await;
    assert_eq!(status, 422, "the head or a rank, not both");

    drop(head); // its connection ends with no goodbye
    let e = wait_for(&cl, "the engine down", |e| e.down.is_some()).await;
    let d = e.down.as_ref().expect("down");
    assert_eq!(d.cause, "lost", "{e:?}");
    assert!(e.head.is_none(), "{e:?}");
    cl.stop();
}

// ---------------------------------------------------------------------------------------------
// Pools (decision 0031)

/// `pool` as the coordinator's view of it, from the API's object.
fn pool_view(p: Pool) -> PoolView {
    PoolView {
        name: p.metadata.name,
        generation: p.metadata.generation,
        written_by: p.metadata.written_by,
        written_at_ms: p.metadata.written_at_ms,
        reason: p.spec.reason,
        engines: p.spec.engines,
        wants: p.spec.wants,
        standing: p.status.standing,
        taking: p.status.taking,
        warming: p.status.warming,
        deficit: p.status.deficit,
        device_class: p.spec.device_class,
        domain: p.spec.domain,
        conditions: p.status.conditions,
    }
}

fn pool_of(body: &str) -> PoolView {
    pool_view(serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}")))
}

async fn pools(cl: &Cluster) -> Vec<PoolView> {
    let list: List<Pool> = get(cl, "/v1/pools").await;
    list.items.into_iter().map(pool_view).collect()
}

/// Write `pool`'s spec, `spec` as JSON: the status and the body.
async fn put_pool(cl: &Cluster, pool: &str, spec: &str) -> (u16, String) {
    let write = format!(r#"{{"spec":{spec}}}"#);
    http(&cl.http, "PUT", &format!("/v1/pools/{pool}"), &write).await
}

fn pool_condition(p: &PoolView, kind: &str) -> (Truth, String) {
    let c = condition(&p.conditions, kind);
    (c.status, c.reason.clone())
}

#[tokio::test]
async fn a_pool_counts_its_engines_standing_spares_and_its_deficit_follows_them() {
    let cl = cluster("policy-two-node.json").await;
    let (_on_a, _on_b, _spares) = two_node_engine(&cl).await;
    let (status, body) = put_pool(
        &cl,
        "h100",
        r#"{"engines":["k3-a"],"wants":6,"reason":"one node's worth and two"}"#,
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let p = pool_of(&body);
    assert_eq!(
        (p.standing, p.taking, p.deficit, p.generation),
        (4, 0, 2, 1),
        "{p:?}"
    );
    assert_eq!(
        pool_condition(&p, "Satisfied"),
        (Truth::False, "Deficit".into())
    );
    assert_eq!(
        pool_condition(&p, "ShapeAgrees"),
        (Truth::True, "Agrees".into())
    );
    assert!(
        cl.telemetry
            .render()
            .contains("everserve_pool_deficit{pool=\"h100\"} 2"),
        "{}",
        cl.telemetry.render()
    );

    // a cordoned node's spares stand for no pool
    let (status, _) = cordon(&cl, "node-c", true, "").await;
    assert_eq!(status, 200);
    let p = pools(&cl).await.remove(0);
    assert_eq!((p.standing, p.deficit), (0, 6), "{p:?}");

    // uncordoned, and wanting what stands: satisfied
    let (status, _) = cordon(&cl, "node-c", false, "").await;
    assert_eq!(status, 200);
    let (status, body) = put_pool(&cl, "h100", r#"{"engines":["k3-a"],"wants":4}"#).await;
    assert_eq!(status, 200, "{body}");
    let p = pool_of(&body);
    assert_eq!((p.standing, p.deficit, p.generation), (4, 0, 2), "{p:?}");
    assert_eq!(
        pool_condition(&p, "Satisfied"),
        (Truth::True, "Satisfied".into())
    );
    cl.stop();
}

#[tokio::test]
async fn a_pools_selector_admits_only_the_spares_it_names() {
    let cl = cluster("policy-two-node.json").await;
    let (_on_a, _on_b, _spares) = two_node_engine(&cl).await;
    // the test's devices are H100s in their node's domain
    let (status, body) = put_pool(
        &cl,
        "other",
        r#"{"engines":["k3-a"],"wants":1,"deviceClass":"NVIDIA A100-SXM4-80GB"}"#,
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let p = pool_of(&body);
    assert_eq!((p.standing, p.deficit), (0, 1), "{p:?}");
    let (status, body) = put_pool(
        &cl,
        "same",
        &format!(r#"{{"engines":["k3-a"],"wants":1,"deviceClass":"{H100}","domain":"node-c"}}"#),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let p = pool_of(&body);
    assert_eq!(p.standing, 4, "{p:?}");
    cl.stop();
}

#[tokio::test]
async fn a_pool_of_engines_that_differ_in_shape_or_are_unknown_is_refused() {
    // a second engine like the first, whose process runs another checkpoint
    let mut policy = example("policy-two-node.json");
    let mut other = policy.engines[0].clone();
    other.name = "k3-b".into();
    policy.engines.push(other);
    let cl = cluster_of(policy, Arc::new(MemoryStore::default())).await;
    let (_on_a, _on_b, _spares) = two_node_engine(&cl).await;
    let sock = node(&cl, "node-d", 1).await;
    let mut s = UnixStream::connect(&sock).await.expect("connect");
    let reg = Signal {
        kind: Some(signal::Kind::Register(Register {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            engine: "k3-b".into(),
            rank: 0,
            member: Some(Member {
                node: "n".into(),
                device: 0,
            }),
            role: Role::Worker as i32,
            identity: Some(
                everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
                    checkpoint_digest: "another-checkpoint".into(),
                    ..common::identity()
                },
            ),
            ..Register::default()
        })),
    };
    frame::send(&mut s, &reg).await.expect("register");
    for _ in 0..300 {
        if cl
            .coord
            .engine("k3-b")
            .await
            .is_some_and(|v| !v.identities.is_empty())
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (status, body) = put_pool(&cl, "mixed", r#"{"engines":["k3-a","k3-b"],"wants":1}"#).await;
    assert_eq!(status, 422, "{body}");
    assert!(
        body.contains("differ in shape") && body.contains("checkpoint_digest"),
        "{body}"
    );
    let (status, body) = put_pool(&cl, "ghost", r#"{"engines":["k3-z"],"wants":1}"#).await;
    assert_eq!(status, 422, "{body}");
    assert!(pools(&cl).await.is_empty(), "nothing was written");
    cl.stop();
}

#[tokio::test]
async fn a_spare_registered_with_two_engines_of_a_pool_counts_once() {
    // decision 0031: a pool spare is one process registered once per engine of its pool
    let mut policy = example("policy-two-node.json");
    let mut other = policy.engines[0].clone();
    other.name = "k3-b".into();
    policy.engines.push(other);
    let cl = cluster_of(policy, Arc::new(MemoryStore::default())).await;
    let (_on_a, _on_b, _spares) = two_node_engine(&cl).await;
    let register_b = |path: std::path::PathBuf, rank: u32, device: u32, role: Role| async move {
        let mut s = UnixStream::connect(&path).await.expect("connect");
        let reg = Signal {
            kind: Some(signal::Kind::Register(Register {
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                engine: "k3-b".into(),
                rank,
                member: Some(Member {
                    node: "n".into(),
                    device,
                }),
                role: role as i32,
                identity: Some(common::identity()),
                ..Register::default()
            })),
        };
        frame::send(&mut s, &reg).await.expect("register");
        s
    };
    let d = node(&cl, "node-d", 1).await;
    let _w = register_b(d, 0, 0, Role::Worker).await;
    // the process standing for k3-a's rank 0 on node-c's device 0 registers with k3-b too
    let c = cl.dir.path().join("node-c.sock");
    let _s = register_b(c, 0, 0, Role::Spare).await;
    for _ in 0..300 {
        if cl
            .coord
            .engine("k3-b")
            .await
            .is_some_and(|v| v.spares.contains_key(&0))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        cl.coord
            .engine("k3-b")
            .await
            .is_some_and(|v| v.spares.contains_key(&0))
    );
    let (status, body) = put_pool(&cl, "both", r#"{"engines":["k3-a","k3-b"],"wants":4}"#).await;
    assert_eq!(status, 200, "{body}");
    let p = pool_of(&body);
    assert_eq!(
        (p.standing, p.deficit),
        (4, 0),
        "four processes, not five: {p:?}"
    );
    cl.stop();
}

#[tokio::test]
async fn a_spare_that_takes_one_engines_position_withdraws_from_the_others() {
    // decision 0031: a pool spare's other registrations close after its take's commit, and each
    // other engine reads that as a withdrawal, not a departure
    let mut policy = example("policy-two-node.json");
    let mut other = policy.engines[0].clone();
    other.name = "k3-b".into();
    policy.engines.push(other);
    let cl = cluster_of(policy, Arc::new(MemoryStore::default())).await;
    let (_on_a, mut on_b, mut spares) = two_node_engine(&cl).await;
    let register_b = |path: std::path::PathBuf, rank: u32, device: u32, role: Role| async move {
        let mut s = UnixStream::connect(&path).await.expect("connect");
        let reg = Signal {
            kind: Some(signal::Kind::Register(Register {
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                engine: "k3-b".into(),
                rank,
                member: Some(Member {
                    node: "n".into(),
                    device,
                }),
                role: role as i32,
                identity: Some(common::identity()),
                ..Register::default()
            })),
        };
        frame::send(&mut s, &reg).await.expect("register");
        s
    };
    let d = node(&cl, "node-d", 2).await;
    let _w = register_b(d.clone(), 0, 0, Role::Worker).await;
    // node-c's device 0 stands for rank 0 of both engines; node-d's device 1 only for k3-b
    let b_on_c = register_b(cl.dir.path().join("node-c.sock"), 0, 0, Role::Spare).await;
    let b_on_d = register_b(d, 1, 1, Role::Spare).await;
    for _ in 0..300 {
        if cl
            .coord
            .engine("k3-b")
            .await
            .is_some_and(|v| v.spares.len() == 2)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // k3-a loses rank 0: node-a's group moves to node-c's spares (0022)
    let change = tokio::spawn({
        let c = Arc::clone(&cl.coord);
        async move { c.rank_lost("k3-a", 0, Cause::Fault).await }
    });
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        common::expect_notices_then_change(w, rank, 0, 1, Reason::Operator).await;
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        common::expect_notices_then_change(s, rank, 0, 1, Reason::Operator).await;
    }
    change.await.expect("join").expect("committed");
    // the process closes its k3-b registration after the commit; node-d's spare simply goes
    drop(b_on_c);
    drop(b_on_d);
    let mut events = Vec::new();
    for _ in 0..300 {
        events = cl.coord.engine_events("k3-b").await.expect("events");
        if events
            .iter()
            .filter(|e| e.reason == "SpareWithdrew" || e.reason == "SpareDeparted")
            .count()
            == 2
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let withdrew = events
        .iter()
        .find(|e| e.reason == "SpareWithdrew")
        .unwrap_or_else(|| panic!("{events:?}"));
    assert!(
        withdrew.message.contains("took rank 0 of k3-a"),
        "{withdrew:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| e.reason == "SpareDeparted" && e.message.contains("node-d device 1")),
        "{events:?}"
    );
    cl.stop();
}

#[tokio::test]
async fn a_pool_short_of_spares_launches_them_on_idle_nodes_it_admits() {
    // decision 0031: a node that matches a pool and holds no position is the pool's capacity
    let cl = cluster("policy-two-node.json").await;
    let (_on_a, _on_b, _spares) = two_node_engine(&cl).await;
    let _e = node(&cl, "node-e", 2).await;
    let (status, _) = cordon(&cl, "node-e", true, "").await;
    assert_eq!(status, 200);
    let write = r#"{"engines":["k3-a"],"wants":6}"#;
    let (status, body) = put_pool(&cl, "h100", write).await;
    assert_eq!(status, 200, "{body}");
    // its launcher, written under a grant of its own; the pool's spec written again keeps it
    let command = r#"{"command":["sh","-c","exec sleep 30"]}"#;
    let (status, body) = http(&cl.http, "PUT", "/v1/pools/h100/launcher", command).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = put_pool(&cl, "h100", write).await;
    assert_eq!(status, 200, "{body}");
    let p = pool_of(&body);
    // node-a and node-b serve, node-c's spares stand, node-e is cordoned: nothing to launch on
    assert_eq!((p.standing, p.warming, p.deficit), (4, 0, 2), "{p:?}");

    let (status, _) = cordon(&cl, "node-e", false, "").await;
    assert_eq!(status, 200);
    let p = pools(&cl).await.remove(0);
    assert_eq!((p.standing, p.warming, p.deficit), (4, 2, 0), "{p:?}");
    let events: Vec<Event> = get(&cl, "/v1/pools/h100/events").await;
    let launched: Vec<&str> = events
        .iter()
        .filter(|e| e.reason == "SpareLaunched")
        .map(|e| e.message.as_str())
        .collect();
    assert_eq!(launched, ["on node-e device 0", "on node-e device 1"]);
    let (status, body) = http(&cl.http, "GET", "/v1/pools/h100/launcher", "").await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("exec sleep 30"), "{body}");
    // a deficit met by warming spares launches nothing more
    let (status, _) = put_pool(&cl, "h100", write).await;
    assert_eq!(status, 200);
    let events: Vec<Event> = get(&cl, "/v1/pools/h100/events").await;
    assert_eq!(
        events
            .iter()
            .filter(|e| e.reason == "SpareLaunched")
            .count(),
        2
    );
    cl.stop();
}

/// Read `s` until it is handed `engine`'s instance state in `phase`; that state.
async fn expect_phase(
    s: &mut tokio::net::UnixStream,
    phase: everserve_controller::proto::everserve::signals::v1::InstancePhase,
) -> everserve_controller::proto::everserve::signals::v1::InstanceState {
    use everserve_controller::proto::everserve::signals::v1::signal;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let sig: everserve_controller::proto::everserve::signals::v1::Signal =
                frame::recv(s).await.expect("frame");
            if let Some(signal::Kind::InstanceState(state)) = sig.kind
                && state.phase() == phase
            {
                return state;
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("no {phase:?} state"))
}

#[tokio::test]
async fn every_process_hears_its_instance_state_as_it_changes() {
    // decision 0041: the instance's state in a framework's terms, pushed to every process of the
    // engine, which an integration reports to the framework; readiness says the same
    use everserve_controller::proto::everserve::signals::v1::InstancePhase;
    use everserve_controller::readiness::InstanceState;
    let cl = cluster("policy-two-node.json").await;
    let sock_a = node(&cl, "node-a", 4).await;
    let sock_b = node(&cl, "node-b", 4).await;
    let mut w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    let starting = expect_phase(&mut w0, InstancePhase::InstanceStarting).await;
    assert_eq!((starting.positions_held, starting.world), (1, 8));
    let mut rest = Vec::new();
    for r in 1..4 {
        rest.push(common::register(&sock_a, r, r, Role::Worker).await);
    }
    for r in 0..4 {
        rest.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    let serving = expect_phase(&mut w0, InstancePhase::InstanceServing).await;
    assert_eq!((serving.positions_held, serving.world), (8, 8));
    // a process that registers now is handed the state at once
    expect_phase(&mut rest[6], InstancePhase::InstanceServing).await;
    let e = wait_for(&cl, "serving", |e| e.state == InstanceState::Serving).await;
    assert_eq!(e.members.len(), 8);

    // the head is lost: the engine is down, and the framework's own recovery takes over
    let head = common::register_head(&sock_a, std::process::id()).await;
    wait_for(&cl, "the head", |e| e.head.is_some()).await;
    drop(head);
    let failed = expect_phase(&mut w0, InstancePhase::InstanceFailed).await;
    assert!(failed.reason.starts_with("lost"), "{failed:?}");
    wait_for(&cl, "failed", |e| e.state == InstanceState::Failed).await;
    cl.stop();
}

#[tokio::test]
async fn a_serving_worker_that_leaves_outside_any_change_is_its_ranks_loss() {
    // a node pod deleted gracefully stops its worker and leaves its daemon no loss to report,
    // and the engine stalls on the rank with nothing replacing it. A serving worker gone outside
    // any change, with a goodbye or none, has lost its rank
    let cl = cluster("policy-two-node.json").await;
    let (mut on_a, mut on_b, mut spares) = two_node_engine(&cl).await;
    drop(on_a.remove(0)); // rank 0's worker, silently
    let within = |f| tokio::time::timeout(Duration::from_secs(10), f);
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        let lost = within(common::expect_notices_then_change(
            w,
            rank,
            0,
            1,
            Reason::Operator,
        ))
        .await
        .expect("the change for rank 0's loss");
        // rank 0's group is node-a's four ranks, and its spares stand on node-c: the group moves
        // as one (decision 0022)
        assert_eq!(lost, vec![0, 1, 2, 3]);
    }
    for (rank, w) in (1u32..).zip(on_a.iter_mut()) {
        within(common::expect_notices_then_change(
            w,
            rank,
            0,
            1,
            Reason::Operator,
        ))
        .await
        .expect("the change");
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        within(common::expect_notices_then_change(
            s,
            rank,
            0,
            1,
            Reason::Operator,
        ))
        .await
        .expect("the change");
    }
    let e = wait_for(&cl, "committed epoch 1", |e| e.epoch == 1).await;
    assert!(
        e.members
            .iter()
            .filter(|m| m.rank < 4)
            .all(|m| m.node == "node-c"),
        "the spares serve the group: {e:?}"
    );
    cl.stop();
}

#[tokio::test]
async fn a_node_that_lost_every_member_without_notice_still_hears_the_commit() {
    // a node whose only member crashed takes no part in the change that replaces it; one that
    // never hears its commit stays at epoch 0, the spare that stands there next is refused every
    // change as stale, and the engine goes down. The commit goes to the engine's audience, so a
    // node that lost every member hears it too
    let cl = cluster("policy-two-node.json").await;
    let (daemon_a, sock_a) = common::daemon(
        cl.dir.path(),
        "node-a",
        &cl.address,
        common::devices("node-a", 4, H100),
        &cl.cancel,
    )
    .await;
    let sock_b = node(&cl, "node-b", 4).await;
    let sock_c = node(&cl, "node-c", 4).await;
    let mut on_a = Vec::new();
    let mut on_b = Vec::new();
    for r in 0..4 {
        on_a.push(common::register(&sock_a, r, r, Role::Worker).await);
        on_b.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    wait_for(&cl, "saw eight members", |e| e.members.len() == 8).await;
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(common::register(&sock_c, r, r, Role::Spare).await);
    }
    wait_for(&cl, "saw four spares", |e| e.spares.len() == 4).await;
    // every worker on node-a crashes: the node holds no member of the change
    on_a.clear();
    let within = |f| tokio::time::timeout(Duration::from_secs(10), f);
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        within(common::expect_notices_then_change(
            w,
            rank,
            0,
            1,
            Reason::Operator,
        ))
        .await
        .expect("the change for node-a's ranks");
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        within(common::expect_notices_then_change(
            s,
            rank,
            0,
            1,
            Reason::Operator,
        ))
        .await
        .expect("the change");
    }
    wait_for(&cl, "committed epoch 1", |e| e.epoch == 1).await;
    for _ in 0..500 {
        if daemon_a.epoch(common::ENGINE).await == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        daemon_a.epoch(common::ENGINE).await,
        1,
        "node-a's daemon heard the commit of the change that replaced its members"
    );
    cl.stop();
}

#[tokio::test]
async fn an_engine_whose_head_does_not_serve_changes_is_changed_by_nothing_and_says_why() {
    // a head whose engine core still steps as vLLM's own dies at a change's first delivery. A
    // head that does not say it serves changes is sent none, and the engine says so before any
    // loss
    let cl = cluster("policy-two-node.json").await;
    let (on_a, mut on_b, _spares) = two_node_engine(&cl).await;
    let sock_a = cl.dir.path().join("node-a.sock");
    let _head = common::register_as(
        &sock_a,
        Register {
            role: Role::Head as i32,
            launcher_pid: std::process::id(),
            identity: Some(
                everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
                    serves_changes: false,
                    ..common::identity()
                },
            ),
            ..common::registration(0, 0, Role::Head)
        },
    )
    .await;
    let e = wait_for(&cl, "degraded by its head", |e| {
        condition(&e.conditions, "Degraded").reason == "NotServingChanges"
    })
    .await;
    assert_eq!(condition(&e.conditions, "Degraded").status, Truth::True);
    // a loss is reported and refused: no change reaches any process
    drop(on_b.remove(0));
    let refused = "result=\"head_not_serving\"";
    for _ in 0..300 {
        if cl.telemetry.render().contains(refused) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        cl.telemetry.render().contains(refused),
        "{}",
        cl.telemetry.render()
    );
    let e = cl.coord.readiness().await.engines.remove(0);
    assert_eq!(e.epoch, 0, "nothing changed: {e:?}");
    drop(on_a);
    cl.stop();
}

#[tokio::test]
async fn members_that_run_different_software_are_named_at_bring_up() {
    // decision 0034: a spare is held to its position's identity; the members are held to one
    // another as they register. Found on GPUs: two ranks registered two checkpoint digests for
    // the same weights and served together until a spare was refused for one of them
    let cl = cluster("policy-two-node.json").await;
    let a = node(&cl, "node-a", 4).await;
    let b = node(&cl, "node-b", 4).await;
    let mut held = Vec::new();
    for r in 0..4 {
        held.push(common::register(&a, r, r, Role::Worker).await);
    }
    for r in 0..4 {
        let rank = 4 + r;
        let identity = everserve_controller::proto::everserve::signals::v1::SoftwareIdentity {
            checkpoint_digest: if rank == 6 {
                "another-checkpoint".into()
            } else {
                common::identity().checkpoint_digest
            },
            ..common::identity()
        };
        held.push(
            common::register_as(
                &b,
                Register {
                    identity: Some(identity),
                    ..common::registration(rank, r, Role::Worker)
                },
            )
            .await,
        );
    }
    let e = wait_for(&cl, "the members' difference named", |e| {
        condition(&e.conditions, "Degraded").reason == "MembersDiffer"
    })
    .await;
    let d = condition(&e.conditions, "Degraded");
    assert!(
        d.message.contains("rank 6 and rank 0") && d.message.contains("checkpoint_digest"),
        "{d:?}"
    );
    cl.cancel.cancel();
}
