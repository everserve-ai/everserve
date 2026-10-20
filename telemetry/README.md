# everserve-telemetry

Metrics, health and logs for every Everserve process: one Prometheus registry per process,
metrics registered by the modules that own them, one HTTP endpoint serving `/metrics` (text
exposition) and `/healthz`, stopped by the same cancellation token as the process, and the
process's logs on standard error through `tracing`, as text for people or one JSON object per
line for a log pipeline (`init_logs`, `LogFormat`). OpenTelemetry export is not built; it would be
a feature flag on this crate, not a different design.

Metric names are stable and documented where they are registered (`controller/src/metrics`,
`store/src/metrics.rs`). Add names; never rename one.
