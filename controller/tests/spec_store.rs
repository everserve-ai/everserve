// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The spec store on Kubernetes (decision 0036, amended 2026-10-13), against a real API server:
//! a newer term takes every object and reads what was written before; the deposed holder's every
//! later write is refused, a new engine's included; an engine registered and not yet made is made
//! by the taker; a large engine's status fits its object.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use everserve_controller::proto::everserve::spec::v1::{
    Assignment, ChangeLog, EngineStatus, NodeDelivery, NodeSpec, ResourceSpec, ResourceStatus,
    resource_spec, resource_status,
};
use everserve_controller::spec::kubernetes::KubernetesStore;
use everserve_controller::spec::{SpecError, SpecStore, engine_key, node_key};

use common::bounded;
use common::kube::ApiServer;

const NAMESPACE: &str = "everserve";
const STORE: &str = "everserve";

/// An API server with the chart's `EverserveState` resource and the namespace.
async fn api_server() -> ApiServer {
    let api = ApiServer::start().await;
    let crd = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../deploy/kubernetes/everserve/crds/everservestates.yaml");
    let crd = std::fs::read_to_string(crd).expect("the CRD");
    api.kubectl_ok(&["apply", "-f", "-"], Some(&crd));
    api.kubectl_ok(
        &[
            "wait",
            "--for=condition=Established",
            "crd/everservestates.everserve.ai",
            "--timeout=20s",
        ],
        None,
    );
    api.kubectl_ok(&["create", "namespace", NAMESPACE], None);
    api
}

fn store(api: &ApiServer) -> KubernetesStore {
    KubernetesStore::new(&api.access, NAMESPACE, STORE, Duration::from_secs(5)).expect("store")
}

fn cordoned(node: &str) -> ResourceSpec {
    ResourceSpec {
        kind: Some(resource_spec::Kind::Node(NodeSpec {
            node: node.into(),
            unschedulable: true,
            ..NodeSpec::default()
        })),
        ..ResourceSpec::default()
    }
}

fn status(engine: &str, generation: u64) -> ResourceStatus {
    ResourceStatus {
        kind: Some(resource_status::Kind::Engine(EngineStatus {
            engine: engine.into(),
            plan_generation: generation,
            ..EngineStatus::default()
        })),
    }
}

/// Every object of the store, as the API server holds it: its name and the term on it.
fn terms(api: &ApiServer) -> Vec<(String, String)> {
    let list: serde_json::Value = serde_json::from_str(&api.kubectl_ok(
        &["get", "everservestates", "-n", NAMESPACE, "-o", "json"],
        None,
    ))
    .expect("json");
    list["items"]
        .as_array()
        .expect("items")
        .iter()
        .map(|o| {
            (
                o["metadata"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                o["spec"]["term"].as_str().unwrap_or_default().to_owned(),
            )
        })
        .collect()
}

#[tokio::test]
async fn a_newer_term_takes_every_object_and_the_deposed_holder_writes_nothing_more() {
    bounded(Box::pin(async {
        let api = api_server().await;
        let a = store(&api);
        let first = a.take(1).await.expect("term 1 takes");
        assert_eq!(
            (first.term, first.resources.len()),
            (1, 0),
            "an empty store, taken"
        );
        a.put(node_key("node-a"), cordoned("node-a"), 0, 1)
            .await
            .expect("a spec");
        // an engine named as no Kubernetes object may be: its object's name is the name's hex
        for engine in ["llama", "Llama 3.3/70B"] {
            a.put_status(engine_key(engine), status(engine, 3), 1)
                .await
                .expect("a status");
        }
        assert_eq!(terms(&api).len(), 3, "a root and an object per engine");

        // term 2 takes the store: every object carries its term, and what term 1 wrote is read
        let b = store(&api);
        let taken = b.take(2).await.expect("term 2 takes");
        assert_eq!(taken.term, 2);
        assert_eq!(taken.resources[&node_key("node-a")].generation, 1);
        for engine in ["llama", "Llama 3.3/70B"] {
            assert_eq!(taken.statuses[&engine_key(engine)], status(engine, 3));
        }
        assert!(
            terms(&api).iter().all(|(_, t)| t == "2"),
            "every object taken: {:?}",
            terms(&api)
        );

        // term 1, deposed without noticing: every write is refused, a new engine's included
        assert!(matches!(
            a.put(node_key("node-b"), cordoned("node-b"), 0, 1).await,
            Err(SpecError::Superseded { .. })
        ));
        assert!(matches!(
            a.put_status(engine_key("llama"), status("llama", 4), 1)
                .await,
            Err(SpecError::Superseded { .. })
        ));
        assert!(matches!(
            a.put_status(engine_key("mixtral"), status("mixtral", 1), 1)
                .await,
            Err(SpecError::Superseded { .. })
        ));
        assert_eq!(terms(&api).len(), 3, "and makes no object");
        // term 2 writes on
        b.put(node_key("node-b"), cordoned("node-b"), 0, 2)
            .await
            .expect("term 2 writes a spec");
        b.put_status(engine_key("mixtral"), status("mixtral", 1), 2)
            .await
            .expect("and a new engine's status");
        // nor does an older term take it back
        assert!(matches!(
            store(&api).take(1).await,
            Err(SpecError::Fenced {
                term: 1,
                newer: 2,
                ..
            })
        ));
    }))
    .await;
}

#[tokio::test]
async fn an_engine_registered_and_not_yet_made_is_made_by_the_taker() {
    bounded(Box::pin(async {
        let api = api_server().await;
        let a = store(&api);
        a.take(1).await.expect("term 1 takes");
        a.put_status(engine_key("llama"), status("llama", 1), 1)
            .await
            .expect("a status");
        // term 1 registered a second engine's object on the root and died before it made it
        let mut root: serde_json::Value = serde_json::from_str(&api.kubectl_ok(
            &[
                "get",
                "everservestate",
                STORE,
                "-n",
                NAMESPACE,
                "-o",
                "json",
            ],
            None,
        ))
        .expect("json");
        let pending = format!(
            "{STORE}-engine-{}",
            everserve_controller::issuer::hex(b"mixtral")
        );
        root["spec"]["engines"]
            .as_array_mut()
            .expect("engines")
            .push(pending.clone().into());
        api.kubectl_ok(&["replace", "-f", "-"], Some(&root.to_string()));

        // term 2's take makes it, empty, under term 2: a create of it under term 1 now fails
        let taken = store(&api).take(2).await.expect("term 2 takes");
        assert!(!taken.statuses.contains_key(&engine_key("mixtral")));
        assert!(
            terms(&api).contains(&(pending, "2".to_owned())),
            "{:?}",
            terms(&api)
        );
    }))
    .await;
}

#[tokio::test]
async fn a_large_engines_status_fits_its_object() {
    // a 128-GPU engine: every rank assigned and a change in flight to sixteen nodes, which the
    // plan, no longer kept, would have put far past what an object holds
    bounded(Box::pin(async {
        let api = api_server().await;
        let s = store(&api);
        s.take(1).await.expect("take");
        let status = ResourceStatus {
            kind: Some(resource_status::Kind::Engine(EngineStatus {
                members: Vec::new(),
                engine: "big".into(),
                plan_generation: 9,
                change: Some(ChangeLog {
                    deliveries: (0..16)
                        .map(|n| NodeDelivery {
                            node: format!("gpu-node-{n:04}"),
                            to_ranks: (n * 8..n * 8 + 8).collect(),
                            ..NodeDelivery::default()
                        })
                        .collect(),
                    world_size: 128,
                    ..ChangeLog::default()
                }),
                assignments: (0..128)
                    .map(|r| Assignment {
                        rank: r,
                        epoch: 7,
                        node: format!("gpu-node-{:04}", r / 8),
                        host: format!("10.0.{}.{}", r / 8, r % 8),
                        ..Assignment::default()
                    })
                    .collect(),
            })),
        };
        s.put_status(engine_key("big"), status.clone(), 1)
            .await
            .expect("fits");
        assert_eq!(
            store(&api).take(2).await.expect("take").statuses[&engine_key("big")],
            status
        );
    }))
    .await;
}

#[tokio::test]
async fn a_coordinator_whose_store_was_taken_stops_acting_at_its_next_write() {
    use everserve_controller::coord::{Coordinator, NodeSpecError};
    use everserve_telemetry::Telemetry;
    use tokio_util::sync::CancellationToken;
    bounded(Box::pin(async {
        let api = api_server().await;
        let policy = everserve_controller::policy::load(&common::example("policy-pilot.json"))
            .expect("policy");
        let open = |term: u64| {
            let (policy, store) = (policy.clone(), std::sync::Arc::new(store(&api)));
            async move {
                let (c, _rx) = Coordinator::with_store(policy, &Telemetry::new(), store, term)
                    .await
                    .expect("takes the store");
                c.act_under(term, CancellationToken::new()).expect("acts");
                c
            }
        };
        let cordon = || everserve_controller::coord::NodeWant {
            unschedulable: true,
            ..everserve_controller::coord::NodeWant::default()
        };
        let a = open(1).await;
        a.put_node_spec("node-x", "operator", cordon(), None)
            .await
            .expect("term 1 writes");
        let b = open(2).await;
        assert!(
            b.node_spec("node-x").await.is_some(),
            "term 1's write, read by term 2's take"
        );
        match a.put_node_spec("node-y", "operator", cordon(), None).await {
            Err(NodeSpecError::Store(SpecError::Superseded { .. })) => {}
            other => panic!("{other:?}"),
        }
        assert!(a.deposed(), "the refused write ended term 1's hold");
        b.put_node_spec("node-y", "operator", cordon(), None)
            .await
            .expect("term 2 writes");
    }))
    .await;
}
