// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The Kubernetes lease against a real API server (decision 0036): envtest's `kube-apiserver` over
//! a test etcd, started by each test and failing it when missing.

mod common;

use std::time::{Duration, Instant};

use everserve_controller::http::Method;
use everserve_controller::kubernetes::Access;
use everserve_controller::lease::{
    ADDRESS_ANNOTATION, Hold, KubernetesLease, KubernetesLeaseWatch, Lease, LeaseError, LeaseWatch,
    TERM_ANNOTATION,
};
use everserve_controller::proto::everserve::coord::v1::LeaseRecord;
use serde_json::{Value, json};

use common::kube::ApiServer;
use common::{Relay, bounded};

const NAMESPACE: &str = "default";

fn lease(access: &Access, name: &str, holder: &str, ttl_s: u64) -> KubernetesLease {
    KubernetesLease::new(access, NAMESPACE, name, holder, Duration::from_secs(ttl_s))
        .expect("lease")
}

fn watch(access: &Access, name: &str) -> KubernetesLeaseWatch {
    KubernetesLeaseWatch::new(access, NAMESPACE, name, Duration::from_secs(5)).expect("watch")
}

fn record(address: &str) -> LeaseRecord {
    LeaseRecord {
        address: address.into(),
        ..LeaseRecord::default()
    }
}

/// The `Lease` object as the API server holds it, read past the backend.
async fn object(access: &Access, name: &str) -> Option<Value> {
    let url = access.url(&format!(
        "/apis/coordination.k8s.io/v1/namespaces/{NAMESPACE}/leases/{name}"
    ));
    let answer = access
        .client()
        .expect("client")
        .get(&url, &[("Accept", "application/json")])
        .await
        .expect("get");
    (answer.status == 200).then(|| serde_json::from_str(&answer.body).expect("json"))
}

/// Write `lease` past the backend, as another writer would; the status the server answered.
async fn put(access: &Access, name: &str, lease: &Value) -> u16 {
    let url = access.url(&format!(
        "/apis/coordination.k8s.io/v1/namespaces/{NAMESPACE}/leases/{name}"
    ));
    access
        .client()
        .expect("client")
        .send(
            Method::PUT,
            &url,
            &[("Content-Type", "application/json")],
            Some(lease.to_string().into_bytes()),
        )
        .await
        .expect("put")
        .status
}

/// Change the object past the backend as another writer does: read it, change it, write it with
/// the version read, and on a conflict (the hold renewed in between) read it again; the object as
/// written.
async fn rewrite(access: &Access, name: &str, change: impl Fn(&mut Value)) -> Value {
    for _ in 0..20 {
        let mut o = object(access, name).await.expect("object");
        change(&mut o);
        match put(access, name, &o).await {
            200 => return o,
            409 => {}
            status => panic!("PUT answered {status}"),
        }
    }
    panic!("the object kept changing under another writer");
}

/// When `hold` ends, by the clock of the moment its token is cancelled.
fn ended_at(hold: &dyn Hold) -> tokio::task::JoinHandle<Instant> {
    let ended = hold.ended().clone();
    tokio::spawn(async move {
        ended.cancelled().await;
        Instant::now()
    })
}

#[tokio::test]
async fn the_kubernetes_lease_has_one_hold_at_a_time_each_at_a_higher_term() {
    bounded(Box::pin(async {
        let api = ApiServer::start().await;
        let a = lease(&api.access, "coordinator", "host-a:1", 10);
        let b = lease(&api.access, "coordinator", "host-b:2", 10);
        let held = a
            .try_acquire()
            .await
            .expect("a")
            .expect("no Lease yet: a creates it");
        assert_eq!(held.term(), 1);
        assert!(b.try_acquire().await.expect("b").is_none(), "held: refused");
        held.publish(record("10.0.0.1:7000"))
            .await
            .expect("publish");
        let read = watch(&api.access, "coordinator")
            .read()
            .await
            .expect("read")
            .expect("a record");
        assert_eq!(
            (read.holder.as_str(), read.term, read.address.as_str()),
            ("host-a:1", 1, "10.0.0.1:7000")
        );
        held.release().await.expect("release");
        let read = watch(&api.access, "coordinator")
            .read()
            .await
            .expect("read")
            .expect("the object stays");
        assert_eq!(
            (read.holder.as_str(), read.term, read.address.as_str()),
            ("", 1, ""),
            "released: no holder, the term kept"
        );
        let next = b
            .try_acquire()
            .await
            .expect("b")
            .expect("released, it can be taken");
        assert_eq!(
            next.term(),
            2,
            "the term rises in the write that takes the lease"
        );
        let o = object(&api.access, "coordinator").await.expect("object");
        assert_eq!(o["metadata"]["annotations"][TERM_ANNOTATION], "2");
    }))
    .await;
}

#[tokio::test]
async fn a_hold_cut_off_from_the_api_ends_before_a_contender_can_take_the_lease() {
    bounded(Box::pin(async {
        let api = ApiServer::start().await;
        let upstream = api
            .access
            .endpoint
            .trim_start_matches("https://")
            .to_owned();
        let relay = Relay::start(&upstream).await;
        let port = relay.endpoint.rsplit(':').next().expect("port").to_owned();
        let through_relay = Access {
            endpoint: format!("https://localhost:{port}"),
            ..api.access.clone()
        };
        let cut_off = lease(&through_relay, "partition", "host-a:1", 2);
        let contender = lease(&api.access, "partition", "host-b:2", 2);
        let asked = Instant::now();
        let a = cut_off.try_acquire().await.expect("a").expect("free");
        let a_ended = ended_at(a.as_ref());
        // a partition at once: every renewal hangs, so the take is the last answered write and
        // the hold ends three quarters of the 2 s duration after it was sent
        relay.cut();
        let b_took = loop {
            if let Some(b) = contender.try_acquire().await.expect("b") {
                assert!(b.term() > a.term());
                break Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        let a_ended = a_ended.await.expect("ended");
        assert!(
            a_ended < b_took,
            "the hold ended {:?} after the contender took the lease",
            a_ended.saturating_duration_since(b_took)
        );
        let held = a_ended - asked;
        assert!(
            held >= Duration::from_millis(1500) && held < Duration::from_millis(1750),
            "the hold ends three quarters of the duration after its last answered write was sent: \
             it held {held:?}"
        );
        assert!(
            b_took - a_ended >= Duration::from_millis(400),
            "the contender waited the whole duration from when it saw the last write"
        );
    }))
    .await;
}

#[tokio::test]
async fn a_hold_renews_past_its_first_deadline_and_ends_when_another_writer_takes_the_object() {
    bounded(Box::pin(async {
        let api = ApiServer::start().await;
        let a = lease(&api.access, "renewal", "host-a:1", 2)
            .try_acquire()
            .await
            .expect("a")
            .expect("free");
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(!a.ended().is_cancelled(), "renewal keeps the hold");
        // a write the hold did not make names another holder
        rewrite(&api.access, "renewal", |o| {
            o["spec"]["holderIdentity"] = json!("someone-else");
        })
        .await;
        tokio::time::timeout(Duration::from_secs(2), a.ended().cancelled())
            .await
            .expect(
                "a renewal reads a conflict as the store's word once the object is not its own",
            );
    }))
    .await;
}

#[tokio::test]
async fn a_write_another_writer_made_is_survived_while_the_object_is_still_the_holds() {
    bounded(Box::pin(async {
        let api = ApiServer::start().await;
        let a = lease(&api.access, "annotated", "host-a:1", 10)
            .try_acquire()
            .await
            .expect("a")
            .expect("free");
        // someone labels the object: its version moves, and it is still the hold's
        rewrite(&api.access, "annotated", |o| {
            o["metadata"]["labels"] = json!({"team": "inference"});
        })
        .await;
        a.publish(record("10.0.0.1:7000"))
            .await
            .expect("a conflict on a write that left the hold its own is written again");
        let o = object(&api.access, "annotated").await.expect("object");
        assert_eq!(
            o["metadata"]["annotations"][ADDRESS_ANNOTATION],
            "10.0.0.1:7000"
        );
        assert!(!a.ended().is_cancelled());
    }))
    .await;
}

#[tokio::test]
async fn a_hold_never_continues_on_a_recreated_object() {
    bounded(Box::pin(async {
        let api = ApiServer::start().await;
        let a = lease(&api.access, "recreated", "host-a:1", 10)
            .try_acquire()
            .await
            .expect("a")
            .expect("free");
        // deleted (which RBAC must deny) and created again with the same holder: another uid
        let url = api.access.url(&format!(
            "/apis/coordination.k8s.io/v1/namespaces/{NAMESPACE}/leases/recreated"
        ));
        let client = api.access.client().expect("client");
        let deleted = client
            .send(Method::DELETE, &url, &[], None)
            .await
            .expect("delete");
        assert_eq!(deleted.status, 200);
        let b = lease(&api.access, "recreated", "host-a:1", 10)
            .try_acquire()
            .await
            .expect("b")
            .expect("created again");
        assert_eq!(b.term(), 1, "a recreated object starts its term over");
        assert!(
            matches!(a.publish(record("x")).await, Err(LeaseError::Lost(_))),
            "the same holder and term on another uid is not the hold's object"
        );
        assert!(a.ended().is_cancelled());
    }))
    .await;
}

#[tokio::test]
async fn a_coordinator_holds_a_kubernetes_lease_and_exits_once_another_writer_takes_it() {
    bounded(Box::pin(async {
        let api = ApiServer::start().await;
        let dir = tempfile::tempdir().expect("tmp");
        let sock = dir.path().join("coord.sock");
        let policy = common::example("policy-two-node.json");
        let mut coord = common::Spawned(
            std::process::Command::new(env!("CARGO_BIN_EXE_everserve-controller"))
                .arg("coord")
                .arg("--policy")
                .arg(policy)
                .arg("--listen")
                .arg(format!("unix:{}", sock.display()))
                .args(["--lease", "kubernetes://default/coordinator"])
                .args(["--lease-ttl-secs", "2", "--graceful-shutdown-secs", "1"])
                .arg("--kubernetes-endpoint")
                .arg(&api.access.endpoint)
                .arg("--kubernetes-token-file")
                .arg(&api.access.token_file)
                .arg("--kubernetes-ca-file")
                .arg(&api.access.ca_file)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("spawn the coordinator"),
        );
        let address = format!("unix:{}", sock.display());
        let watch = watch(&api.access, "coordinator");
        loop {
            if let Ok(Some(r)) = watch.read().await
                && r.address == address
            {
                assert_eq!(r.term, 1);
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        rewrite(&api.access, "coordinator", |o| {
            o["spec"]["holderIdentity"] = json!("someone-else");
        })
        .await;
        assert_eq!(
            common::exited(&mut coord).await,
            Some(75),
            "the coordinator stops once its object is another's: EXIT_LEASE_LOST"
        );
    }))
    .await;
}
