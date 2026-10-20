// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The cluster's Nodes as the coordinator reads them (decision 0041): followed from a real API
//! server, list then watch, so a label set, changed or a Node deleted reaches the coordinator; a
//! head on a spot node shows as `HeadOnSpot`; and a node marked for preemption is drained once.

mod common;

use std::time::Duration;

use everserve_controller::coord::{Coordinator, Label, NodeLabels};
use everserve_controller::kubernetes::nodes::{NodeFacts, Nodes, reflect};
use everserve_controller::policy;
use everserve_controller::proto::everserve::signals::v1::Role;
use everserve_controller::status::Truth;
use everserve_telemetry::Telemetry;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use common::kube::ApiServer;

/// `f`, which must finish within ten seconds.
async fn within<T>(what: &str, f: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), f)
        .await
        .unwrap_or_else(|_| panic!("{what}: not within 10 s"))
}

/// Wait until what `rx` holds satisfies `ok`.
async fn until(rx: &mut watch::Receiver<Nodes>, what: &str, ok: impl Fn(&Nodes) -> bool) {
    within(what, async {
        loop {
            if ok(&rx.borrow_and_update()) {
                return;
            }
            rx.changed().await.expect("the reflector runs");
        }
    })
    .await;
}

fn node_yaml(name: &str, label: &str) -> String {
    format!("apiVersion: v1\nkind: Node\nmetadata:\n  name: {name}\n  labels:\n    {label}\n")
}

#[tokio::test]
async fn a_nodes_labels_reach_the_coordinator_as_they_change() {
    let api = ApiServer::start().await;
    api.kubectl_ok(
        &["apply", "-f", "-"],
        Some(&node_yaml("gpu-1", "pool: spot")),
    );
    let (tx, mut rx) = watch::channel(Nodes::new());
    let cancel = CancellationToken::new();
    tokio::spawn(reflect(api.access.clone(), tx, cancel.clone()));
    until(&mut rx, "the listed node", |n| {
        n.get("gpu-1")
            .is_some_and(|f| f.labels.get("pool").map(String::as_str) == Some("spot"))
    })
    .await;
    // watched, not listed again: a change and a new node arrive as events
    api.kubectl_ok(
        &["label", "node", "gpu-1", "pool=on-demand", "--overwrite"],
        None,
    );
    api.kubectl_ok(
        &["apply", "-f", "-"],
        Some(&node_yaml("gpu-2", "pool: spot")),
    );
    until(&mut rx, "the relabelled node and the new one", |n| {
        n.get("gpu-1")
            .is_some_and(|f| f.labels.get("pool").map(String::as_str) == Some("on-demand"))
            && n.contains_key("gpu-2")
    })
    .await;
    api.kubectl_ok(&["cordon", "gpu-2"], None);
    until(&mut rx, "the cordon", |n| {
        n.get("gpu-2").is_some_and(|f| f.unschedulable)
    })
    .await;
    api.kubectl_ok(&["delete", "node", "gpu-1"], None);
    until(&mut rx, "the deletion", |n| !n.contains_key("gpu-1")).await;
    cancel.cancel();
}

/// A node as the reflector gives it, with `labels`.
fn facts(labels: &[(&str, &str)]) -> NodeFacts {
    NodeFacts {
        labels: labels
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
        unschedulable: false,
    }
}

#[tokio::test]
async fn a_head_on_a_spot_node_shows_and_a_preempted_node_is_drained_once() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut p = policy::load(&common::example("policy-pilot.json")).expect("policy");
    p.engines[0].head_on_demand = true;
    let cancel = CancellationToken::new();
    let telemetry = Telemetry::new();
    let (coord, address) = common::coordinator(p, dir.path(), &telemetry, &cancel).await;
    let (_d, sock) = common::daemon(dir.path(), "node-a", &address, Vec::new(), &cancel).await;
    let _w0 = common::register(&sock, 0, 0, Role::Worker).await;
    let _w1 = common::register(&sock, 1, 1, Role::Worker).await;
    let _head = common::register_head(&sock, 4242).await;
    let head_on_spot = |c: std::sync::Arc<Coordinator>| async move {
        c.readiness()
            .await
            .engines
            .into_iter()
            .next()
            .and_then(|e| e.conditions.into_iter().find(|c| c.kind == "HeadOnSpot"))
            .map(|c| (c.status, c.reason))
    };
    let wait_for = |want: (Truth, &'static str)| {
        let coord = std::sync::Arc::clone(&coord);
        async move {
            within("HeadOnSpot", async {
                loop {
                    if head_on_spot(std::sync::Arc::clone(&coord)).await
                        == Some((want.0, want.1.to_owned()))
                    {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await;
        }
    };
    // no Nodes followed yet: unknown
    wait_for((Truth::Unknown, "CapacityUnknown")).await;
    let (tx, rx) = watch::channel(Nodes::new());
    let labels = NodeLabels {
        spot: Some(Label {
            key: "pool".into(),
            value: "spot".into(),
        }),
        preemption: Some(Label {
            key: "preempt".into(),
            value: "true".into(),
        }),
        preemption_lead: Duration::from_mins(2),
    };
    tokio::spawn(std::sync::Arc::clone(&coord).follow_nodes(rx, labels, cancel.clone()));
    tx.send_replace([("node-a".to_owned(), facts(&[("pool", "spot")]))].into());
    wait_for((Truth::True, "SpotNode")).await;
    tx.send_replace([("node-a".to_owned(), facts(&[("pool", "on-demand")]))].into());
    wait_for((Truth::False, "OnDemandNode")).await;
    assert!(coord.drain_state("node-a").await.is_none());

    // the provider marks node-a: it is drained by the lead, once however often the label is seen
    let marked = [(
        "node-a".to_owned(),
        facts(&[("pool", "on-demand"), ("preempt", "true")]),
    )];
    tx.send_replace(marked.clone().into());
    within("the drain", async {
        while coord.drain_state("node-a").await.is_none() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    // the label seen again, and other Nodes changing, write nothing more, not even a refusal
    tx.send_replace(marked.clone().into());
    let mut more = marked.to_vec();
    more.push(("node-b".to_owned(), facts(&[("pool", "spot")])));
    tx.send_replace(more.into_iter().collect());
    tokio::time::sleep(Duration::from_millis(200)).await;
    let writes: f64 = telemetry
        .render()
        .lines()
        .filter(|l| {
            l.starts_with("everserve_spec_writes_total{") && l.contains("writer=\"node_notice\"")
        })
        .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
        .sum();
    assert!(
        (writes - 1.0).abs() < f64::EPSILON,
        "one drain written for one preemption: {writes} writes"
    );
    cancel.cancel();
}
