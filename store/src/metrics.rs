// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What an operator watches on the mirror, as Prometheus metrics. Names are stable.

use everserve_telemetry::Telemetry;
use everserve_telemetry::prometheus::{
    CounterVec, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts,
};

/// The mirror's metrics, registered once.
#[derive(Clone)]
pub struct StoreMetrics {
    /// Commits accepted.
    pub commits_total: IntCounter,
    /// Commits refused, by reason.
    pub refusals_total: IntCounterVec,
    /// Bytes allocated to row tables.
    pub bytes_in_use: IntGauge,
    /// Rows streamed out by restores.
    pub restore_rows_total: IntCounter,
    /// Steps a replica target is behind this store, by engine, rank and target.
    pub replica_lag_steps: IntGaugeVec,
    /// Steps acknowledged by replica targets, by engine and target.
    pub replicated_total: IntCounterVec,
    /// Steps dropped because a target was too far behind, by engine and target.
    pub replication_dropped_total: IntCounterVec,
    /// Links to replica targets that failed and are reconnecting, by engine and target.
    pub replication_link_failures_total: IntCounterVec,
    /// Steps applied on this store as a replica.
    pub replica_applied_total: IntCounter,
    /// Seconds commits waited for room in a target's queue, by engine and target.
    pub replication_wait_seconds_total: CounterVec,
    /// A restore's seconds, by phase: waiting for the store's lock (`lock`), and reading the
    /// tables' written windows under it (`work`).
    pub restore_seconds: HistogramVec,
    /// Snapshots a carrier began, by result: `sent` (the replica took it), `refused` (it did
    /// not), `held` (labelled below the replica's fence, so not sent until this store commits
    /// at it).
    pub snapshots_total: IntCounterVec,
}

impl StoreMetrics {
    /// Register the mirror's metrics with `telemetry`.
    ///
    /// # Panics
    ///
    /// If the names are already registered: a store registers once.
    #[must_use]
    pub fn register(telemetry: &Telemetry) -> Self {
        let r = telemetry.registry();
        let commits_total =
            IntCounter::new("everserve_store_commits_total", "commits accepted").expect("opts");
        let refusals_total = IntCounterVec::new(
            Opts::new(
                "everserve_store_refusals_total",
                "commits and replicated steps refused, by reason (checksum and unchecked among them)",
            ),
            &["reason"],
        )
        .expect("opts");
        let bytes_in_use =
            IntGauge::new("everserve_store_bytes_in_use", "bytes in row tables").expect("opts");
        let restore_rows_total = IntCounter::new(
            "everserve_store_restore_rows_total",
            "rows streamed by restores",
        )
        .expect("opts");
        r.register(Box::new(commits_total.clone()))
            .expect("register once");
        r.register(Box::new(refusals_total.clone()))
            .expect("register once");
        r.register(Box::new(bytes_in_use.clone()))
            .expect("register once");
        r.register(Box::new(restore_rows_total.clone()))
            .expect("register once");
        let replica_lag_steps = IntGaugeVec::new(
            Opts::new(
                "everserve_store_replica_lag_steps",
                "steps a replica target is behind",
            ),
            &["engine", "rank", "target"],
        )
        .expect("opts");
        let replicated_total = IntCounterVec::new(
            Opts::new(
                "everserve_store_replicated_total",
                "steps acknowledged by replica targets",
            ),
            &["engine", "target"],
        )
        .expect("opts");
        let replication_dropped_total = IntCounterVec::new(
            Opts::new(
                "everserve_store_replication_dropped_total",
                "steps a target cannot take (over the frame cap, undecodable)",
            ),
            &["engine", "target"],
        )
        .expect("opts");
        let replication_link_failures_total = IntCounterVec::new(
            Opts::new(
                "everserve_store_replication_link_failures_total",
                "links to replica targets that failed (no answer, refused, closed) and reconnect",
            ),
            &["engine", "target"],
        )
        .expect("opts");
        let replica_applied_total = IntCounter::new(
            "everserve_store_replica_applied_total",
            "steps applied as a replica",
        )
        .expect("opts");
        let replication_wait_seconds_total = CounterVec::new(
            Opts::new(
                "everserve_store_replication_wait_seconds_total",
                "seconds commits waited for room in a target's queue",
            ),
            &["engine", "target"],
        )
        .expect("opts");
        r.register(Box::new(replica_lag_steps.clone()))
            .expect("register once");
        r.register(Box::new(replicated_total.clone()))
            .expect("register once");
        r.register(Box::new(replication_dropped_total.clone()))
            .expect("register once");
        r.register(Box::new(replication_link_failures_total.clone()))
            .expect("register once");
        r.register(Box::new(replica_applied_total.clone()))
            .expect("register once");
        r.register(Box::new(replication_wait_seconds_total.clone()))
            .expect("register once");
        let (restore_seconds, snapshots_total) = resync_metrics(telemetry);
        Self {
            commits_total,
            refusals_total,
            bytes_in_use,
            restore_rows_total,
            replica_lag_steps,
            replicated_total,
            replication_dropped_total,
            replication_link_failures_total,
            replica_applied_total,
            replication_wait_seconds_total,
            restore_seconds,
            snapshots_total,
        }
    }
}

/// The restore and snapshot metrics, registered with `telemetry`.
fn resync_metrics(telemetry: &Telemetry) -> (HistogramVec, IntCounterVec) {
    let r = telemetry.registry();
    let restore_seconds = HistogramVec::new(
        HistogramOpts::new(
            "everserve_store_restore_seconds",
            "a restore's seconds by phase",
        )
        .buckets(vec![0.001, 0.005, 0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
        &["phase"],
    )
    .expect("opts");
    let snapshots_total = IntCounterVec::new(
        Opts::new(
            "everserve_store_snapshots_total",
            "snapshots a carrier began, by result",
        ),
        &["result"],
    )
    .expect("opts");
    r.register(Box::new(restore_seconds.clone()))
        .expect("register once");
    r.register(Box::new(snapshots_total.clone()))
        .expect("register once");
    (restore_seconds, snapshots_total)
}
