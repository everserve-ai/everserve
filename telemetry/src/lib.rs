// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Metrics, health and logs for every Everserve process.
//!
//! One Prometheus [`Registry`] per process, metrics registered by the modules that own them, and
//! one HTTP endpoint serving `/metrics` in the text exposition format and `/healthz`. The same
//! shape Dynamo's runtime uses (`docs/REVIEW-industry-shapes.md`); OpenTelemetry export is a later
//! feature behind a flag, not a different design.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use prometheus::{Encoder, Registry, TextEncoder};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

pub use axum;
pub use prometheus;

/// Why the endpoint could not be served.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// The address could not be bound.
    #[error("metrics endpoint: {0}")]
    Io(#[from] std::io::Error),
}

/// What a process reports about itself.
#[derive(Clone)]
pub struct Telemetry {
    registry: Arc<Registry>,
    healthy: Arc<AtomicBool>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    /// An empty registry, healthy.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: Arc::new(Registry::new()),
            healthy: Arc::new(AtomicBool::new(true)),
        }
    }

    /// The registry modules register their metrics with.
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Flip the health reported at `/healthz`.
    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Release);
    }

    /// The current exposition text, for tests and for logging.
    #[must_use]
    pub fn render(&self) -> String {
        let mut buf = Vec::new();
        let _ = TextEncoder::new().encode(&self.registry.gather(), &mut buf);
        String::from_utf8(buf).unwrap_or_default()
    }

    /// Bind `addr` and serve `/metrics` and `/healthz` there in the background until `cancel`
    /// fires; the bound address (port 0 picks a free port). A connection that sends no request
    /// headers within `header_read` is closed, so a caller cannot hold connections, and the
    /// process's file descriptors, by sending nothing. The bind happens before this returns, so a
    /// process whose port is taken stops at startup rather than running without its metrics and
    /// health.
    ///
    /// # Errors
    ///
    /// The bind error.
    pub async fn start(
        self,
        addr: SocketAddr,
        header_read: Duration,
        cancel: CancellationToken,
    ) -> Result<SocketAddr, TelemetryError> {
        let listener = TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        info!("metrics on http://{local}/metrics");
        let app = Router::new()
            .route("/metrics", get(metrics))
            .route("/healthz", get(healthz))
            .with_state(self);
        tokio::spawn(accept(listener, app, header_read, cancel));
        Ok(local)
    }
}

/// How long a served process's endpoint waits for a request's headers.
pub const HEADER_READ: Duration = Duration::from_secs(10);

/// How long the listener waits after an accept error (out of file descriptors, say) before it
/// accepts again, so a persistent error does not spin it.
const ACCEPT_BACKOFF: Duration = Duration::from_secs(1);

/// Take connections on `listener` until `cancel` fires, each served on its own task, which ends
/// its connection once `cancel` fires and the request in hand is answered.
async fn accept(
    listener: TcpListener,
    app: Router,
    header_read: Duration,
    cancel: CancellationToken,
) {
    loop {
        let (tcp, from) = tokio::select! {
            () = cancel.cancelled() => return,
            accepted = listener.accept() => match accepted {
                Ok(a) => a,
                Err(e) => {
                    warn!("metrics: accept: {e}");
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(ACCEPT_BACKOFF) => continue,
                    }
                }
            },
        };
        let app = app.clone();
        let cancel = cancel.clone();
        tokio::spawn(async move {
            let mut builder = hyper::server::conn::http1::Builder::new();
            builder
                .timer(TokioTimer::new())
                .header_read_timeout(header_read);
            let conn = builder.serve_connection(TokioIo::new(tcp), TowerToHyperService::new(app));
            tokio::pin!(conn);
            let served = tokio::select! {
                served = conn.as_mut() => served,
                () = cancel.cancelled() => {
                    conn.as_mut().graceful_shutdown();
                    conn.await
                }
            };
            if let Err(e) = served {
                debug!("metrics: {from}: {e}");
            }
        });
    }
}

async fn metrics(State(t): State<Telemetry>) -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        t.render(),
    )
}

async fn healthz(State(t): State<Telemetry>) -> impl IntoResponse {
    if t.healthy.load(Ordering::Acquire) {
        (StatusCode::OK, "ok")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "unhealthy")
    }
}

/// How a process writes its log lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFormat {
    /// For people: one line per event, its fields after its message.
    #[default]
    Text,
    /// For a log pipeline: one JSON object per line, its fields as keys.
    Json,
}

impl std::str::FromStr for LogFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(format!("{other}: a log format is text or json")),
        }
    }
}

/// Write this process's logs to standard error in `format`, the events `filter` lets through.
/// Called once, first thing, by each binary.
pub fn init_logs(format: LogFormat, filter: tracing_subscriber::EnvFilter) {
    let logs = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter);
    match format {
        LogFormat::Text => logs.init(),
        LogFormat::Json => logs.json().init(),
    }
}
