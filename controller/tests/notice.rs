// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A notice source on the daemon, against a fake of each cloud's endpoint: the source reports the
//! node's deadline once, the coordinator drains the node, every rank on it hears a notice and the
//! survivors and spares one change for the set. A source that keeps saying the same thing reports
//! it once; a source that fails is retried; the Kubernetes source sends the bearer token.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, put};
use everserve_controller::coord::Coordinator;
use everserve_controller::node::Daemon;
use everserve_controller::notice::{self, Settings, Source};
use everserve_controller::policy;
use everserve_controller::proto::everserve::signals::v1::{Reason, Role, Signal, signal};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::Address;
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

const H100: &str = "NVIDIA H100 80GB HBM3";

struct Cluster {
    coord: Arc<Coordinator>,
    cancel: CancellationToken,
    address: Address,
    dir: tempfile::TempDir,
}

async fn cluster() -> Cluster {
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let policy = policy::load(&common::example("policy-two-node.json")).expect("policy");
    let (coord, address) =
        common::coordinator(policy, dir.path(), &Telemetry::new(), &cancel).await;
    Cluster {
        coord,
        cancel,
        address,
        dir,
    }
}

async fn node(cl: &Cluster, node: &str) -> (Arc<Daemon>, PathBuf) {
    let devices = common::devices(node, 4, H100);
    common::daemon(cl.dir.path(), node, &cl.address, devices, &cl.cancel).await
}

async fn wait_spares(cl: &Cluster, n: usize) {
    for _ in 0..300 {
        if cl.coord.pool("k3-a").await.len() == n {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("never saw {n} spares: {:?}", cl.coord.engine("k3-a").await);
}

/// A fake endpoint serving `router` on a free port; returns its base URL.
async fn fake(router: Router, cancel: &CancellationToken) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake");
    let addr = listener.local_addr().expect("addr");
    let cancel = cancel.clone();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move { cancel.cancelled().await })
            .await
            .expect("serve fake");
    });
    format!("http://{addr}")
}

fn settings(source: Source, endpoint: String, node: &str, dir: &Path) -> Settings {
    Settings {
        source,
        endpoint: Some(endpoint.clone()),
        poll: Duration::from_millis(50),
        lead: Duration::from_millis(400),
        node: node.into(),
        kubernetes: Some(everserve_controller::kubernetes::Access {
            endpoint,
            token_file: dir.join("token"),
            ca_file: dir.join("ca.crt"),
        }),
    }
}

/// The AWS fake: `IMDSv2` token, then 404 until `armed`, then the documented action with a time
/// `lead` from when the fake was built.
fn aws_fake(armed: Arc<AtomicUsize>, time: String) -> Router {
    #[derive(Clone)]
    struct St {
        armed: Arc<AtomicUsize>,
        time: String,
        hits: Arc<AtomicUsize>,
    }
    async fn token(headers: HeaderMap) -> (StatusCode, &'static str) {
        if headers
            .get("x-aws-ec2-metadata-token-ttl-seconds")
            .is_none()
        {
            return (StatusCode::BAD_REQUEST, "no ttl");
        }
        (StatusCode::OK, "tok-123")
    }
    async fn action(State(st): State<St>, headers: HeaderMap) -> (StatusCode, String) {
        if headers
            .get("x-aws-ec2-metadata-token")
            .map(axum::http::HeaderValue::as_bytes)
            != Some(b"tok-123")
        {
            return (StatusCode::UNAUTHORIZED, String::new());
        }
        st.hits.fetch_add(1, Ordering::SeqCst);
        if st.armed.load(Ordering::SeqCst) == 0 {
            return (StatusCode::NOT_FOUND, String::new());
        }
        (
            StatusCode::OK,
            format!(r#"{{"action": "terminate", "time": "{}"}}"#, st.time),
        )
    }
    Router::new()
        .route("/latest/api/token", put(token))
        .route("/latest/meta-data/spot/instance-action", get(action))
        .with_state(St {
            armed,
            time,
            hits: Arc::new(AtomicUsize::new(0)),
        })
}

/// An RFC 3339 time `ahead` from now, to the second.
fn rfc3339_in(ahead: Duration) -> (String, i64) {
    let ms = everserve_controller::clock::now_ms() + i64::try_from(ahead.as_millis()).unwrap();
    let secs = ms / 1000;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // civil from days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (
        format!(
            "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
            rem / 3600,
            (rem % 3600) / 60,
            rem % 60
        ),
        secs * 1000,
    )
}

#[tokio::test]
async fn an_aws_spot_notice_on_a_node_drains_it_through_the_coordinator() {
    let cl = cluster().await;
    let (da, sock_a) = node(&cl, "node-a").await;
    let (_db, sock_b) = node(&cl, "node-b").await;
    let (_dc, sock_c) = node(&cl, "node-c").await;
    let mut on_a = Vec::new();
    let mut on_b = Vec::new();
    for r in 0..4 {
        on_a.push(common::register(&sock_a, r, r, Role::Worker).await);
        on_b.push(common::register(&sock_b, 4 + r, r, Role::Worker).await);
    }
    common::wait_members(&cl.coord, 8).await;
    let mut spares = Vec::new();
    for r in 0..4 {
        spares.push(common::register(&sock_c, r, r, Role::Spare).await);
    }
    wait_spares(&cl, 4).await;

    // the fake says nothing for a while, then an interruption half a second out
    let armed = Arc::new(AtomicUsize::new(0));
    let (time, deadline_ms) = rfc3339_in(Duration::from_millis(1500));
    let base = fake(aws_fake(Arc::clone(&armed), time), &cl.cancel).await;
    tokio::spawn(notice::run(
        settings(Source::Aws, base, "node-a", cl.dir.path()),
        Arc::clone(&da),
        cl.cancel.clone(),
    ));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        cl.coord.engine("k3-a").await.expect("view").epoch,
        0,
        "nothing yet"
    );
    armed.store(1, Ordering::SeqCst);

    // every worker and spare hears four notices (one per rank on node-a) with the fake's time,
    // then one change for the set at epoch 1
    for (rank, w) in (4u32..).zip(on_b.iter_mut()) {
        let lost = common::expect_notices_then_change(w, rank, 4, 1, Reason::Operator).await;
        assert_eq!(lost, vec![0, 1, 2, 3]);
    }
    for (rank, s) in (0u32..).zip(spares.iter_mut()) {
        let lost = common::expect_notices_then_change(s, rank, 4, 1, Reason::Operator).await;
        assert_eq!(lost, vec![0, 1, 2, 3]);
    }
    for w in &mut on_a {
        for _ in 0..4 {
            let sig: Signal = common::recv(w).await.expect("notice");
            let Some(signal::Kind::Notice(n)) = sig.kind else {
                panic!("expected a notice: {sig:?}")
            };
            assert_eq!(
                n.deadline_ms, deadline_ms,
                "the deadline is the fake's time"
            );
        }
    }
    for _ in 0..300 {
        if cl
            .coord
            .engine("k3-a")
            .await
            .is_some_and(|v| v.epoch == 1 && v.spares.is_empty())
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let view = cl.coord.engine("k3-a").await.expect("view");
    assert_eq!(view.epoch, 1, "{view:?}");
    assert!(
        view.members
            .iter()
            .filter(|(r, _)| **r < 4)
            .all(|(_, n)| n == "node-c")
    );
    // the same answer keeps coming from the fake; it was reported once (epoch stays 1)
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(cl.coord.engine("k3-a").await.expect("view").epoch, 1);
    cl.cancel.cancel();
}

/// A cluster with one worker, rank 0 on node-a, and no spare.
async fn single() -> (Cluster, Arc<Daemon>, UnixStream) {
    let cl = cluster().await;
    let (da, sock_a) = node(&cl, "node-a").await;
    let w0 = common::register(&sock_a, 0, 0, Role::Worker).await;
    common::wait_members(&cl.coord, 1).await;
    (cl, da, w0)
}

/// Run `source` against `base` for `da` and return the first notice rank 0 hears.
async fn first_notice(
    cl: &Cluster,
    da: &Arc<Daemon>,
    w0: &mut UnixStream,
    source: Source,
    base: String,
) -> (
    everserve_controller::proto::everserve::signals::v1::Notice,
    CancellationToken,
) {
    let run_cancel = CancellationToken::new();
    tokio::spawn(notice::run(
        settings(source, base, "node-a", cl.dir.path()),
        Arc::clone(da),
        run_cancel.clone(),
    ));
    let sig: Signal = common::recv(w0).await.expect("notice");
    let Some(signal::Kind::Notice(n)) = sig.kind else {
        panic!("{sig:?}")
    };
    (n, run_cancel)
}

#[tokio::test]
async fn a_gcp_preemption_flag_gives_the_lead_and_is_reported_once_while_it_stays() {
    let (cl, da, mut w0) = single().await;
    // the flag flips after two polls and stays TRUE
    let polls = Arc::new(AtomicUsize::new(0));
    let base = fake(
        Router::new()
            .route(
                "/computeMetadata/v1/instance/maintenance-event",
                get(|| async { "NONE" }),
            )
            .route(
                "/computeMetadata/v1/instance/preempted",
                get({
                    let polls = Arc::clone(&polls);
                    move |headers: HeaderMap| {
                        let polls = Arc::clone(&polls);
                        async move {
                            assert_eq!(headers.get("metadata-flavor").unwrap(), "Google");
                            if polls.fetch_add(1, Ordering::SeqCst) < 2 {
                                "FALSE"
                            } else {
                                "TRUE"
                            }
                        }
                    }
                }),
            ),
        &cl.cancel,
    )
    .await;
    let before = everserve_controller::clock::now_ms();
    let (n, run_cancel) = first_notice(&cl, &da, &mut w0, Source::Gcp, base).await;
    assert!(
        n.deadline_ms >= before + 400 && n.deadline_ms < before + 3000,
        "{n:?}"
    );
    // the source keeps polling and the flag stays TRUE: every later poll computes a later
    // deadline, which is the same news, so the worker hears nothing more; and at the deadline
    // the drain's change is refused (no spare: REBUILD), so nothing else arrives either
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(polls.load(Ordering::SeqCst) >= 6, "the source kept polling");
    let quiet = tokio::time::timeout(
        Duration::from_millis(200),
        frame::recv::<_, Signal>(&mut w0),
    )
    .await;
    assert!(quiet.is_err(), "nothing more was heard: {quiet:?}");
    assert_eq!(cl.coord.engine("k3-a").await.expect("view").epoch, 0);
    run_cancel.cancel();
    cl.cancel.cancel();
}

#[tokio::test]
async fn an_azure_preempt_event_is_the_notice_after_failures_are_retried() {
    let (cl, da, mut w0) = single().await;
    let polls = Arc::new(AtomicUsize::new(0));
    let not_before = httpdate::fmt_http_date(std::time::SystemTime::now() + Duration::from_secs(5));
    let base = fake(
        Router::new()
            .route(
                "/metadata/instance/compute/name",
                get(|headers: HeaderMap| async move {
                    assert_eq!(headers.get("metadata").unwrap(), "true");
                    "vm"
                }),
            )
            .route(
            "/metadata/scheduledevents",
            get({
                let polls = Arc::clone(&polls);
                let not_before = not_before.clone();
                move |headers: HeaderMap| {
                    let polls = Arc::clone(&polls);
                    let not_before = not_before.clone();
                    async move {
                        assert_eq!(headers.get("metadata").unwrap(), "true");
                        if polls.fetch_add(1, Ordering::SeqCst) < 2 {
                            return (StatusCode::INTERNAL_SERVER_ERROR, String::new());
                        }
                        (
                            StatusCode::OK,
                            format!(
                                r#"{{"DocumentIncarnation":3,"Events":[{{"EventId":"e1","EventType":"Preempt","ResourceType":"VirtualMachine","Resources":["vm"],"EventStatus":"Scheduled","NotBefore":"{not_before}","Description":"","EventSource":"Platform","DurationInSeconds":-1}}]}}"#
                            ),
                        )
                    }
                }
            }),
        ),
        &cl.cancel,
    )
    .await;
    let (n, run_cancel) = first_notice(&cl, &da, &mut w0, Source::Azure, base).await;
    let expected = httpdate::parse_http_date(&not_before)
        .unwrap()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    assert_eq!(u128::try_from(n.deadline_ms).unwrap(), expected, "{n:?}");
    assert_eq!(
        n.reason,
        Reason::Operator as i32,
        "a drain's notices carry the operator's reason"
    );
    assert!(
        polls.load(Ordering::SeqCst) >= 3,
        "two failures were retried"
    );
    run_cancel.cancel();
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_kubernetes_autoscaler_taint_gives_its_time_with_the_bearer_token() {
    let (cl, da, mut w0) = single().await;
    std::fs::write(cl.dir.path().join("token"), "sa-token\n").expect("token");
    let base = fake(
        Router::new().route(
            "/api/v1/nodes/{name}",
            get(
                |axum::extract::Path(name): axum::extract::Path<String>, headers: HeaderMap| async move {
                    assert_eq!(name, "node-a");
                    assert_eq!(headers.get("authorization").unwrap(), "Bearer sa-token");
                    r#"{"kind":"Node","metadata":{"name":"node-a"},"spec":{"taints":[{"key":"ToBeDeletedByClusterAutoscaler","value":"1891286878","effect":"NoSchedule"}]}}"#
                },
            ),
        ),
        &cl.cancel,
    )
    .await;
    let (n, run_cancel) = first_notice(&cl, &da, &mut w0, Source::Kubernetes, base).await;
    assert_eq!(n.deadline_ms, 1_891_286_878_000);
    run_cancel.cancel();
    cl.cancel.cancel();
}
