// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0

//! A synthetic cell: one coordinator, N fake daemons over its real link, engines of K nodes of
//! eight devices each, and SHRINK changes driven through it. It records what a change costs at
//! cell size: report-to-commit time, the daemons a change sends to, and the bytes they receive;
//! and what the read side costs: one readiness and one node list.
//!
//! The measurements are `#[ignore]`d: each opens two descriptors per node, more than a default
//! descriptor limit allows. Run one with `ulimit -n 16384` first:
//!
//! ```text
//! cargo test -p everserve-controller --release --test cell -- --ignored --nocapture cell_1250
//! ```
//!
//! Each prints one JSON line. The run record under `bench/runs/` holds the numbers; no bound is
//! asserted here until a record exists to state it from.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use everserve_controller::api::feed::{Feed, Kind, Pacing};
use everserve_controller::coord::{Cause, Coordinator};
use everserve_controller::lease::BoxFuture;
use everserve_controller::node::{RECONNECT_MAX, RECONNECT_MIN, jittered};
use everserve_controller::policy;
use everserve_controller::proto::everserve::coord::v1::{
    CoordToDaemon, DaemonHello, DaemonToCoord, Delivered, EngineState, coord_to_daemon,
    daemon_to_coord,
};
use everserve_controller::proto::everserve::policy::v1::{Engine, Policy, SparePool, spare_pool};
use everserve_controller::proto::everserve::signals::v1::{Ack, Member, Outcome};
use everserve_controller::proto::everserve::spec::v1::{
    ResourceSpec, ResourceStatus, SpecSnapshot,
};
use everserve_controller::spec::{MemoryStore, SpecError, SpecStore};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::transport::{Address, Listener, Stream, connect};
use prost::Message;
use tokio_util::sync::CancellationToken;

const DEVICES: u32 = 8;

/// What one fake daemon received: frames and bytes, length prefixes included; and how often it
/// was refused before it linked.
#[derive(Default)]
struct Received {
    frames: AtomicU64,
    bytes: AtomicU64,
    /// Connections the coordinator's listener refused before this daemon linked.
    refused: AtomicU64,
}

impl Received {
    fn take(&self) -> (u64, u64) {
        (
            self.frames.load(Ordering::Relaxed),
            self.bytes.load(Ordering::Relaxed),
        )
    }
}

/// A coordinator of `policy` over `store`, taken under `term`. The store's writes take what its
/// delay says, as a real one's round trips do: a change that holds a lock other engines need
/// across its write makes them wait that long each.
async fn coordinator(policy: Policy, store: Arc<Delayed>, term: u64) -> Arc<Coordinator> {
    let coord = Coordinator::with_store(policy, &Telemetry::new(), store, term)
        .await
        .expect("a store taken")
        .0;
    // as a coordinator does once it holds the lease: it acts under the term it took the store at
    coord
        .act_under(term, CancellationToken::new())
        .expect("acting under the term");
    coord
}

/// The in-memory store, each write taking `delay` first.
struct Delayed {
    inner: MemoryStore,
    delay: Duration,
}

impl SpecStore for Delayed {
    fn take(&self, term: u64) -> BoxFuture<'_, Result<SpecSnapshot, SpecError>> {
        self.inner.take(term)
    }

    fn put(
        &self,
        key: String,
        spec: ResourceSpec,
        read: u64,
        term: u64,
    ) -> BoxFuture<'_, Result<ResourceSpec, SpecError>> {
        Box::pin(async move {
            tokio::time::sleep(self.delay).await;
            self.inner.put(key, spec, read, term).await
        })
    }

    fn put_status(
        &self,
        key: String,
        status: ResourceStatus,
        term: u64,
    ) -> BoxFuture<'_, Result<(), SpecError>> {
        Box::pin(async move {
            tokio::time::sleep(self.delay).await;
            self.inner.put_status(key, status, term).await
        })
    }

    fn describe(&self) -> String {
        format!(
            "{}, each write {:?} late",
            self.inner.describe(),
            self.delay
        )
    }
}

/// `engines` engines of `nodes_per_engine` nodes each, eight devices a node, every one SHRINK.
fn cell_policy(engines: usize, nodes_per_engine: usize) -> Policy {
    let engines = (0..engines)
        .map(|e| Engine {
            name: format!("e{e}"),
            members: (0..nodes_per_engine)
                .flat_map(|n| {
                    (0..DEVICES).map(move |device| Member {
                        node: format!("n{}", e * nodes_per_engine + n),
                        device,
                    })
                })
                .collect(),
            tensor_parallel: DEVICES,
            data_parallel: u32::try_from(nodes_per_engine).expect("nodes"),
            pipeline_parallel: 1,
            placement_independent_model: true,
            ..Engine::default()
        })
        .collect();
    let policy = Policy {
        version: "cell".into(),
        outcomes_in_order: vec![Outcome::Shrink as i32],
        max_concurrent_losses: u32::MAX,
        // a SHRINK engine runs no spares: the rule is stated, never exercised
        spares: Some(SparePool {
            loading: spare_pool::Loading::Preloaded as i32,
            ..SparePool::default()
        }),
        engines,
        ..Policy::default()
    };
    policy::validate(&policy).expect("a valid cell policy");
    policy
}

/// Node `node`'s fake daemon, as a daemon links: it says hello for its engine's ranks on it,
/// answers every delivery with an ack from each rank it names, and keeps each commit's epoch
/// and lost ranks; when its link ends it links again, with the daemon's jittered backoff, and its
/// hello says what it holds now. It counts what it receives, until `stop` fires.
async fn daemon(
    address: Address,
    node: usize,
    engine: usize,
    nodes_per_engine: usize,
    received: Arc<Received>,
    stop: CancellationToken,
) {
    let first = u32::try_from((node - engine * nodes_per_engine) * DEVICES as usize).expect("rank");
    let mut held = EngineState {
        engine: format!("e{engine}"),
        workers: (first..first + DEVICES).collect(),
        world_size: u32::try_from(nodes_per_engine).expect("nodes") * DEVICES,
        ..EngineState::default()
    };
    let mut wait = RECONNECT_MIN;
    while !stop.is_cancelled() {
        let Ok(mut link) = connect(&address, None, "coordinator").await else {
            received.refused.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(jittered(wait)).await;
            wait = (wait * 2).min(RECONNECT_MAX);
            continue;
        };
        wait = RECONNECT_MIN;
        let hello = DaemonToCoord {
            kind: Some(daemon_to_coord::Kind::Hello(DaemonHello {
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                node: format!("n{node}"),
                engines: vec![held.clone()],
                ..DaemonHello::default()
            })),
        };
        if frame::send(&mut link, &hello).await.is_ok() {
            tokio::select! {
                () = serve(&mut link, &mut held, &received) => {}
                () = stop.cancelled() => return,
            }
        }
        // the link ended: link again, as a daemon whose coordinator went does
        tokio::time::sleep(jittered(RECONNECT_MIN)).await;
    }
}

/// Serve one link until it ends: count each frame, ack each delivery, keep each commit.
async fn serve(link: &mut Stream, held: &mut EngineState, received: &Received) {
    loop {
        let Ok(msg): Result<CoordToDaemon, _> = frame::recv(link).await else {
            return;
        };
        received.frames.fetch_add(1, Ordering::Relaxed);
        received
            .bytes
            .fetch_add(4 + msg.encoded_len() as u64, Ordering::Relaxed);
        match msg.kind {
            Some(coord_to_daemon::Kind::Deliver(d)) => {
                let epoch = d.change.as_ref().map_or(0, |c| c.epoch);
                let delivered = DaemonToCoord {
                    kind: Some(daemon_to_coord::Kind::Delivered(Delivered {
                        request_id: d.request_id,
                        engine: d.engine,
                        epoch,
                        acks: d
                            .to_ranks
                            .iter()
                            .map(|&rank| Ack {
                                rank,
                                epoch,
                                ..Ack::default()
                            })
                            .collect(),
                        error: String::new(),
                    })),
                };
                if frame::send(link, &delivered).await.is_err() {
                    return;
                }
            }
            Some(coord_to_daemon::Kind::Committed(c)) if c.engine == held.engine => {
                if let Some(m) = c.membership.as_ref() {
                    held.epoch = held.epoch.max(m.epoch);
                }
                // SHRINK: the lost ranks leave the node's workers, in order
                for r in c.lost {
                    held.workers.retain(|w| *w != r);
                    if !held.lost.contains(&r) {
                        held.lost.push(r);
                    }
                }
            }
            _ => {}
        }
    }
}

/// The resident set and the CPU time this process used so far, from `ps`: the one reading
/// both hosts the harness runs on give.
fn usage() -> (u64, f64) {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=,time=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut fields = text.split_whitespace();
    let rss_kib: u64 = fields.next().and_then(|f| f.parse().ok()).unwrap_or(0);
    // [[dd-]hh:]mm:ss[.ss]
    let cpu = fields.next().map_or(0.0, |t| {
        let (days, rest) = t
            .split_once('-')
            .map_or((0.0, t), |(d, r)| (d.parse::<f64>().unwrap_or(0.0), r));
        rest.split(':').fold(0.0, |acc, part| {
            acc * 60.0 + part.parse::<f64>().unwrap_or(0.0)
        }) + days * 86_400.0
    });
    (rss_kib * 1024, cpu)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "an index into a short list of samples"
    )]
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

/// Wait until every daemon in `received` is linked to `coord` and each engine of `want` has as
/// many members as it names; fail the test at `deadline`, saying how far linking got.
async fn linked(
    coord: &Coordinator,
    received: &[Arc<Received>],
    want: &BTreeMap<String, usize>,
    deadline: Instant,
) {
    let nodes = received.len();
    loop {
        let linked = coord.daemons().await.len();
        let mut viewed = 0;
        for (engine, members) in want {
            if coord
                .engine(engine)
                .await
                .is_some_and(|v| v.members.len() == *members)
            {
                viewed += 1;
            }
        }
        if linked == nodes && viewed == want.len() {
            break;
        }
        let refused: u64 = received
            .iter()
            .map(|r| r.refused.load(Ordering::Relaxed))
            .sum();
        assert!(
            Instant::now() < deadline,
            "not linked in time: {linked} of {nodes} daemons linked, {viewed} of {} engines \
             whole, {refused} connections refused",
            want.len()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// The cell's coordinator stops, and a successor takes its store under the next term and listens
/// at its address: how long until every daemon is linked to the successor and every engine's
/// view is as the predecessor last held it, and then how long the successor's first change takes.
/// The successor's serving ends with `cancel`.
async fn failover(
    coord: &Arc<Coordinator>,
    stopped: &CancellationToken,
    policy: Policy,
    store: Arc<Delayed>,
    address: &Address,
    received: &[Arc<Received>],
    cancel: &CancellationToken,
) -> serde_json::Value {
    let mut want = BTreeMap::new();
    for e in &policy.engines {
        let members = coord.engine(&e.name).await.map_or(0, |v| v.members.len());
        want.insert(e.name.clone(), members);
    }
    let started = Instant::now();
    stopped.cancel();
    let next = coordinator(policy, store, 1).await;
    let listener = Listener::bind(address, None).await.expect("rebind");
    tokio::spawn(Arc::clone(&next).serve(listener, cancel.clone()));
    linked(&next, received, &want, started + Duration::from_mins(10)).await;
    let relinked_s = started.elapsed().as_secs_f64();
    // the last engine still has rank 2: no change before took it
    let last = want.keys().next_back().expect("an engine").clone();
    let t = Instant::now();
    next.rank_lost(&last, 2, Cause::Fault)
        .await
        .expect("the successor changes an engine");
    let first_change_s = t.elapsed().as_secs_f64();
    // a change once the successor has had two seconds to finish what the hellos set off: as fast
    // as one before the failover when the first waited on that work, as slow when it did not
    tokio::time::sleep(Duration::from_secs(2)).await;
    let first = want.keys().next().expect("an engine").clone();
    let t = Instant::now();
    next.rank_lost(&first, 2, Cause::Fault)
        .await
        .expect("the successor changes another engine");
    serde_json::json!({
        "relinked_s": relinked_s,
        "first_change_s": first_change_s,
        "settled_change_s": t.elapsed().as_secs_f64(),
    })
}

/// `changes` SHRINK changes, one at a time, each on another engine: each one's time, the
/// daemons it sent to, and the bytes they received.
async fn one_at_a_time(
    coord: &Arc<Coordinator>,
    received: &[Arc<Received>],
    engines: usize,
    changes: usize,
) -> (Vec<f64>, Vec<usize>, Vec<u64>) {
    let mut seconds = Vec::new();
    let mut touched = Vec::new();
    let mut bytes = Vec::new();
    for k in 0..changes {
        let engine = format!("e{}", (k * engines / changes.max(1)) % engines);
        let before: Vec<(u64, u64)> = received.iter().map(|r| r.take()).collect();
        let t = Instant::now();
        coord
            .rank_lost(&engine, 0, Cause::Fault)
            .await
            .expect("committed");
        seconds.push(t.elapsed().as_secs_f64());
        // the commit's own sends land after the report returns: count them once settled
        tokio::time::sleep(Duration::from_millis(500)).await;
        let after: Vec<(u64, u64)> = received.iter().map(|r| r.take()).collect();
        touched.push(before.iter().zip(&after).filter(|(b, a)| a.0 > b.0).count());
        bytes.push(
            before
                .iter()
                .zip(&after)
                .map(|(b, a)| a.1 - b.1)
                .sum::<u64>(),
        );
    }
    (seconds, touched, bytes)
}

/// `burst` SHRINK changes reported at once, each on another engine (rank 1, so none is an
/// engine the one-at-a-time changes took rank 0 of twice): how long the burst took, and each
/// change's report-to-commit time. Engines that change independently finish a burst in about
/// one change's time; a lock every change holds throughout finishes it in `burst` changes' time.
async fn burst_of(coord: &Arc<Coordinator>, engines: usize, burst: usize) -> serde_json::Value {
    let started = Instant::now();
    let tasks: Vec<_> = (0..burst)
        .map(|i| {
            let coord = Arc::clone(coord);
            let engine = format!("e{}", i * engines / burst.max(1));
            tokio::spawn(async move {
                let t = Instant::now();
                coord
                    .rank_lost(&engine, 1, Cause::Fault)
                    .await
                    .expect("committed");
                t.elapsed().as_secs_f64()
            })
        })
        .collect();
    let mut seconds = Vec::new();
    for t in tasks {
        seconds.push(t.await.expect("join"));
    }
    let wall = started.elapsed().as_secs_f64();
    seconds.sort_by(f64::total_cmp);
    serde_json::json!({
        "n": burst,
        "wall_s": wall,
        "p50": percentile(&seconds, 0.5),
        "max": seconds.last().copied().unwrap_or(0.0),
    })
}

/// What the operator API's reads cost (decision 0042): one refresh of the feed, which every
/// read and every quarter second runs, with nothing changed since the last; and one list of the
/// cell's nodes, refresh included.
async fn feed_reads(coord: &Arc<Coordinator>, nodes: usize) -> (f64, f64) {
    let cancel = CancellationToken::new();
    // its own refresh only at start: the reads timed here are the only others
    let feed = Feed::start(
        Arc::clone(coord),
        Pacing {
            refresh: Duration::from_hours(1),
            ..Pacing::SERVED
        },
        &cancel,
    );
    // the first refresh shows every object: timed after it, a refresh finds nothing new
    feed.refresh().await;
    let t = Instant::now();
    feed.refresh().await;
    let refresh_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (_, listed) = feed.list(Kind::Nodes).await;
    let list_s = t.elapsed().as_secs_f64();
    assert_eq!(listed.len(), nodes);
    cancel.cancel();
    (refresh_s, list_s)
}

/// Run a cell of `nodes` nodes in engines of `nodes_per_engine`, then `changes` SHRINK changes
/// one at a time, each on another engine, then `burst` at once; print what it measured as one
/// JSON line, and return it.
async fn cell(
    nodes: usize,
    nodes_per_engine: usize,
    changes: usize,
    burst: usize,
    link_within: Duration,
    store_write: Duration,
) -> serde_json::Value {
    let engines = nodes / nodes_per_engine;
    let dir = tempfile::tempdir().expect("tmp");
    let cancel = CancellationToken::new();
    let store = Arc::new(Delayed {
        inner: MemoryStore::default(),
        delay: store_write,
    });
    let coord = coordinator(
        cell_policy(engines, nodes_per_engine),
        Arc::clone(&store),
        0,
    )
    .await;
    let stop = CancellationToken::new();
    let address = Address::Unix(dir.path().join("coord.sock"));
    let listener = Listener::bind(&address, None).await.expect("bind");
    tokio::spawn(Arc::clone(&coord).serve(listener, cancel.clone()));
    let (rss_idle, cpu_idle) = usage();

    // link every node
    let started = Instant::now();
    let received: Vec<Arc<Received>> = (0..nodes).map(|_| Arc::default()).collect();
    for (node, r) in received.iter().enumerate() {
        tokio::spawn(daemon(
            address.clone(),
            node,
            node / nodes_per_engine,
            nodes_per_engine,
            Arc::clone(r),
            stop.clone(),
        ));
    }
    let whole: BTreeMap<String, usize> = (0..engines)
        .map(|e| (format!("e{e}"), nodes_per_engine * DEVICES as usize))
        .collect();
    linked(&coord, &received, &whole, started + link_within).await;
    let link_s = started.elapsed().as_secs_f64();
    // what linking sent settles before a change is counted
    tokio::time::sleep(Duration::from_secs(2)).await;
    let (rss_linked, cpu_linked) = usage();

    let (mut seconds, touched, bytes) = one_at_a_time(&coord, &received, engines, changes).await;
    let burst = burst_of(&coord, engines, burst).await;
    let (rss_changed, cpu_changed) = usage();

    // the read side, as the API's feed and lists read it
    let t = Instant::now();
    let ready = coord.readiness().await;
    let readiness_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let listed = coord.node_views().await;
    let node_views_s = t.elapsed().as_secs_f64();
    assert_eq!(ready.engines.len(), engines);
    assert_eq!(listed.len(), nodes);
    let (feed_refresh_s, node_list_s) = feed_reads(&coord, nodes).await;
    let succeeded = CancellationToken::new();
    let failover = failover(
        &coord,
        &cancel,
        cell_policy(engines, nodes_per_engine),
        store,
        &address,
        &received,
        &succeeded,
    )
    .await;

    seconds.sort_by(f64::total_cmp);
    let report = serde_json::json!({
        "nodes": nodes,
        "store_write_s": store_write.as_secs_f64(),
        "gpus": nodes * DEVICES as usize,
        "engines": engines,
        "nodes_per_engine": nodes_per_engine,
        "link_s": link_s,
        "connections_refused": received
            .iter()
            .map(|r| r.refused.load(Ordering::Relaxed))
            .sum::<u64>(),
        "rss_bytes": { "idle": rss_idle, "linked": rss_linked, "after_changes": rss_changed },
        "cpu_s": { "link": cpu_linked - cpu_idle, "changes": cpu_changed - cpu_linked },
        "burst": burst,
        "change_s": {
            "n": seconds.len(),
            "p50": percentile(&seconds, 0.5),
            "p99": percentile(&seconds, 0.99),
            "max": seconds.last().copied().unwrap_or(0.0),
        },
        "daemons_sent_to_per_change": touched,
        "bytes_received_per_change": bytes,
        "readiness_s": readiness_s,
        "node_views_s": node_views_s,
        "feed_refresh_s": feed_refresh_s,
        "api_node_list_s": node_list_s,
        "failover": failover,
    });
    println!("{report}");
    stop.cancel();
    succeeded.cancel();
    report
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement: needs ulimit -n 16384, see the file's doc"]
async fn cell_1250() {
    cell(1250, 2, 20, 100, Duration::from_mins(10), Duration::ZERO).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement: needs ulimit -n 16384, see the file's doc"]
async fn cell_1250_store_10ms() {
    cell(
        1250,
        2,
        20,
        100,
        Duration::from_mins(10),
        Duration::from_millis(10),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement: needs ulimit -n 16384, see the file's doc"]
async fn cell_2500() {
    cell(2500, 2, 20, 100, Duration::from_mins(10), Duration::ZERO).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement: needs ulimit -n 16384, see the file's doc"]
async fn cell_5000() {
    cell(5000, 2, 20, 100, Duration::from_mins(10), Duration::ZERO).await;
}

/// The harness itself, at a size any host runs: 40 engines, more than a link's channel once
/// held, so a link coming up that waits on its own sends never links (at 1,250 nodes, every link
/// parked).
#[tokio::test(flavor = "multi_thread")]
async fn a_cell_of_more_engines_than_a_link_once_buffered_links_and_commits_its_changes() {
    let report = cell(80, 2, 4, 8, Duration::from_secs(30), Duration::ZERO).await;
    // a change goes to its engine's two nodes, not the cell's eighty (without an audience, a cell
    // of 1,250 nodes sends every change to all 1,250)
    assert_eq!(
        report["daemons_sent_to_per_change"],
        serde_json::json!([2, 2, 2, 2]),
        "{report}"
    );
}
