// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The control plane's Prometheus metrics: the coordinator's (changes) and the daemon's (workers).
//! Names are stable: add, never rename.

use everserve_telemetry::Telemetry;
use everserve_telemetry::prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts,
};

/// The coordinator's metrics, registered once.
#[derive(Clone)]
pub struct CoordMetrics {
    /// Membership changes driven, by engine, cause (`fault` or `notice`), outcome taken
    /// (`replace`, `shrink`, `rebuild`, or `none`/`unknown` when none was) and result
    /// (`committed` or an error kind).
    pub changes_total: IntCounterVec,
    /// Seconds from a lost rank to the committed membership, by cause and outcome.
    pub change_seconds: HistogramVec,
    /// Changes in flight right now.
    pub changes_in_flight: IntGauge,
    /// Seconds from a change's first folded report to its delivery: the controller's decision
    /// (decision 0036), held to a fault path that never shows in a gap.
    pub change_decision_seconds: Histogram,
    /// Node daemons connected right now.
    pub daemons_connected: IntGauge,
    /// Spares in the pool, by engine and state (`warm`, `on_demand`, `mismatched`).
    pub spares: IntGaugeVec,
    /// Whether an engine is down (1) or not (0), by engine (decision 0032): its head departed, or
    /// a change failed after its delivery. Readiness says why.
    pub engines_down: IntGaugeVec,
    /// Second heads refused, by engine: a head registered on another node while the engine's
    /// head was alive.
    pub head_conflicts_total: IntCounterVec,
    /// Releases of a communicator left running past their bound, by engine (decision 0025).
    pub releases_leaked_total: IntCounterVec,
    /// The controllers' own metrics (decision 0036).
    pub control: ControlMetrics,
}

impl CoordMetrics {
    /// Register the coordinator's metrics with `telemetry`.
    ///
    /// # Panics
    ///
    /// If the names are already registered: a coordinator registers once.
    #[must_use]
    pub fn register(telemetry: &Telemetry) -> Self {
        let r = telemetry.registry();
        let changes_total = counter(
            "everserve_changes_total",
            "membership changes driven by the coordinator",
            &["engine", "cause", "outcome", "result"],
        );
        let change_seconds = histogram(
            "everserve_change_seconds",
            "lost rank to committed membership",
            &[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 30.0, 60.0],
            &["cause", "outcome"],
        );
        let change_decision_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "everserve_change_decision_seconds",
                "first report to delivery, per change",
            )
            .buckets(vec![0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.25, 1.0]),
        )
        .expect("opts");
        let changes_in_flight = IntGauge::new(
            "everserve_changes_in_flight",
            "changes being driven right now",
        )
        .expect("opts");
        let daemons_connected = IntGauge::new(
            "everserve_daemons_connected",
            "node daemons connected to the coordinator",
        )
        .expect("opts");
        let spares = gauge(
            "everserve_spares",
            "spares in the pool",
            &["engine", "state"],
        );
        let head_conflicts_total = counter(
            "everserve_head_conflicts_total",
            "second heads refused while the engine's head was alive",
            &["engine"],
        );
        let releases_leaked_total = counter(
            "everserve_releases_leaked_total",
            "releases of a communicator left running past their bound",
            &["engine"],
        );
        let engines_down = gauge(
            "everserve_engine_down",
            "whether the engine is down",
            &["engine"],
        );
        for c in [
            Box::new(changes_total.clone())
                as Box<dyn everserve_telemetry::prometheus::core::Collector>,
            Box::new(change_seconds.clone()),
            Box::new(changes_in_flight.clone()),
            Box::new(change_decision_seconds.clone()),
            Box::new(daemons_connected.clone()),
            Box::new(spares.clone()),
            Box::new(engines_down.clone()),
            Box::new(head_conflicts_total.clone()),
            Box::new(releases_leaked_total.clone()),
        ] {
            r.register(c).expect("register once");
        }
        Self {
            changes_total,
            change_seconds,
            changes_in_flight,
            change_decision_seconds,
            daemons_connected,
            spares,
            engines_down,
            head_conflicts_total,
            releases_leaked_total,
            control: ControlMetrics::register(telemetry),
        }
    }
}

/// The controllers' own metrics (decision 0036): what each reconcile did and how long it took,
/// their queues, the resources' conditions, the cover's retries and the spec's writes.
#[derive(Clone)]
pub struct ControlMetrics {
    /// Reconciles run, by controller (`engine` or `node`) and what each did (decision 0036).
    pub reconciles_total: IntCounterVec,
    /// Each reconcile's wall time, by controller: held under a millisecond (decision 0036).
    pub reconcile_seconds: HistogramVec,
    /// Keys queued and not yet taken, by controller.
    pub queue_depth: IntGaugeVec,
    /// Seconds from a key's enqueue to its reconcile, by controller.
    pub queue_wait_seconds: HistogramVec,
    /// Each engine condition's current status: 1 on the series of the status it holds, 0 on the
    /// others (decision 0036). How long an engine spent short of its spec is a query over this.
    pub engine_condition: IntGaugeVec,
    /// Each node condition's current status, as `engine_condition`.
    pub node_condition: IntGaugeVec,
    /// Cover plans delivered again to retry a failed group, by engine (decision 0035).
    pub cover_retries_total: IntCounterVec,
    /// Spec writes, by resource kind, writer (`operator_api` or `node_notice`) and result
    /// (`written`, `conflict`, `refused`, `failed`).
    pub spec_writes_total: IntCounterVec,
    /// Each pool's deficit: the spares it wants less those standing (decision 0031).
    pub pool_deficit: IntGaugeVec,
    /// 1 once a daemon reported a newer lease term than this coordinator's, or its hold of the
    /// lease ended before it stopped (decision 0036).
    pub deposed: IntGauge,
    /// How a coordinator that starts or takes the lease heard the cell again.
    pub takeover: TakeoverMetrics,
}

/// How a coordinator that starts or takes the lease heard the cell again (decision 0036,
/// amended 2026-10-15).
#[derive(Clone)]
pub struct TakeoverMetrics {
    /// Each daemon's first hello to this coordinator, in seconds after the first daemon's: how
    /// long the daemons' path took to carry them all, against the allowance after which a rank
    /// still unheard is lost.
    pub relink_seconds: Histogram,
    /// 1 while engines served before this coordinator and no daemon has reached it within its
    /// bound: it is the one cut off, and it changes nothing.
    pub unreachable: IntGauge,
    /// Ranks lost because their daemon was not heard within the allowance after the first
    /// daemon relinked, by engine.
    pub unheard_lost_total: IntCounterVec,
}

impl TakeoverMetrics {
    fn register(r: &everserve_telemetry::prometheus::Registry) -> Self {
        let relink_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "everserve_coordinator_relink_seconds",
                "each daemon's first hello, in seconds after the first daemon's",
            )
            .buckets(vec![
                0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.25, 8.0, 15.0, 30.0,
            ]),
        )
        .expect("opts");
        let unreachable = IntGauge::new(
            "everserve_coordinator_unreachable",
            "1 while engines served before this coordinator and no daemon has reached it",
        )
        .expect("opts");
        let unheard_lost_total = IntCounterVec::new(
            Opts::new(
                "everserve_unheard_ranks_lost_total",
                "ranks lost because their daemon was not heard after a coordinator started",
            ),
            &["engine"],
        )
        .expect("opts");
        r.register(Box::new(relink_seconds.clone()))
            .expect("register once");
        r.register(Box::new(unreachable.clone()))
            .expect("register once");
        r.register(Box::new(unheard_lost_total.clone()))
            .expect("register once");
        Self {
            relink_seconds,
            unreachable,
            unheard_lost_total,
        }
    }
}

impl ControlMetrics {
    /// Register the controllers' metrics with `telemetry`.
    ///
    /// # Panics
    ///
    /// If the names are already registered: a coordinator registers once.
    #[must_use]
    pub fn register(telemetry: &Telemetry) -> Self {
        let r = telemetry.registry();
        let reconciles_total = counter(
            "everserve_reconciles_total",
            "reconciles run, by what each did",
            &["controller", "outcome"],
        );
        let reconcile_seconds = histogram(
            "everserve_reconcile_seconds",
            "one reconcile's wall time",
            &[0.000_05, 0.000_1, 0.000_25, 0.000_5, 0.001, 0.005, 0.025],
            &["controller"],
        );
        let queue_depth = gauge(
            "everserve_queue_depth",
            "keys queued and not yet taken",
            &["controller"],
        );
        let queue_wait_seconds = histogram(
            "everserve_queue_wait_seconds",
            "a key's enqueue to its reconcile",
            &[0.000_1, 0.001, 0.01, 0.1, 1.0, 10.0],
            &["controller"],
        );
        let engine_condition = gauge(
            "everserve_engine_condition",
            "1 on the status each engine condition holds",
            &["engine", "type", "status"],
        );
        let node_condition = gauge(
            "everserve_node_condition",
            "1 on the status each node condition holds",
            &["node", "type", "status"],
        );
        let cover_retries_total = counter(
            "everserve_cover_retries_total",
            "cover plans delivered again to retry a failed group",
            &["engine"],
        );
        let deposed = IntGauge::new(
            "everserve_coordinator_deposed",
            "1 once a daemon reported a newer lease term than this coordinator's",
        )
        .expect("opts");
        let pool_deficit = gauge(
            "everserve_pool_deficit",
            "the spares a pool wants less those standing",
            &["pool"],
        );
        let spec_writes_total = counter(
            "everserve_spec_writes_total",
            "spec writes, by writer and result",
            &["kind", "writer", "result"],
        );
        for c in [
            Box::new(reconciles_total.clone())
                as Box<dyn everserve_telemetry::prometheus::core::Collector>,
            Box::new(reconcile_seconds.clone()),
            Box::new(queue_depth.clone()),
            Box::new(queue_wait_seconds.clone()),
            Box::new(engine_condition.clone()),
            Box::new(node_condition.clone()),
            Box::new(cover_retries_total.clone()),
            Box::new(spec_writes_total.clone()),
            Box::new(pool_deficit.clone()),
            Box::new(deposed.clone()),
        ] {
            r.register(c).expect("register once");
        }
        Self {
            reconciles_total,
            reconcile_seconds,
            queue_depth,
            queue_wait_seconds,
            engine_condition,
            node_condition,
            cover_retries_total,
            spec_writes_total,
            pool_deficit,
            deposed,
            takeover: TakeoverMetrics::register(r),
        }
    }

    /// Set `conditions`' series on `gauge` for the resource `name`: 1 on each condition's status,
    /// 0 on the other two.
    pub fn set_conditions(
        gauge: &IntGaugeVec,
        name: &str,
        conditions: &[crate::status::Condition],
    ) {
        for c in conditions {
            for status in ["True", "False", "Unknown"] {
                let on = format!("{:?}", c.status) == status;
                gauge
                    .with_label_values(&[name, &c.kind, status])
                    .set(i64::from(on));
            }
        }
    }
}

fn counter(name: &str, help: &str, labels: &[&str]) -> IntCounterVec {
    IntCounterVec::new(Opts::new(name, help), labels).expect("opts")
}

fn gauge(name: &str, help: &str, labels: &[&str]) -> IntGaugeVec {
    IntGaugeVec::new(Opts::new(name, help), labels).expect("opts")
}

fn histogram(name: &str, help: &str, buckets: &[f64], labels: &[&str]) -> HistogramVec {
    HistogramVec::new(
        HistogramOpts::new(name, help).buckets(buckets.to_vec()),
        labels,
    )
    .expect("opts")
}

/// The node daemon's metrics, registered once.
#[derive(Clone)]
pub struct DaemonMetrics {
    /// Workers currently registered, by engine.
    pub workers_registered: IntGaugeVec,
    /// Whether the daemon is linked to a coordinator (1) or holding reports (0).
    pub coordinator_linked: IntGauge,
    /// Coordinator messages refused for an older term than the newest seen (decision 0036).
    pub fenced_total: IntCounter,
}

impl DaemonMetrics {
    /// Register the daemon's metrics with `telemetry`.
    ///
    /// # Panics
    ///
    /// If the names are already registered: a daemon registers once.
    #[must_use]
    pub fn register(telemetry: &Telemetry) -> Self {
        let r = telemetry.registry();
        let workers_registered = IntGaugeVec::new(
            Opts::new(
                "everserve_workers_registered",
                "workers registered with the daemon",
            ),
            &["engine"],
        )
        .expect("opts");
        let coordinator_linked = IntGauge::new(
            "everserve_coordinator_linked",
            "1 while the daemon is linked to a coordinator",
        )
        .expect("opts");
        let fenced_total = IntCounter::new(
            "everserve_fenced_total",
            "coordinator messages refused for an older term than the newest seen",
        )
        .expect("opts");
        for c in [
            Box::new(workers_registered.clone())
                as Box<dyn everserve_telemetry::prometheus::core::Collector>,
            Box::new(coordinator_linked.clone()),
            Box::new(fenced_total.clone()),
        ] {
            r.register(c).expect("register once");
        }
        Self {
            workers_registered,
            coordinator_linked,
            fenced_total,
        }
    }
}
