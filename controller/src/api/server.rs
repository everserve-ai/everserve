// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator API's listener (decision 0039): TLS with the coordinator's certificate, a client
//! certificate optional and verified when presented; each request taken to the [`Gate`] before
//! its route, with the caller's [`Identity`] handed to the route.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{MatchedPath, RawPathParams, Request, State};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use everserve_wire::transport::{ClientAuth, HANDSHAKE_TIMEOUT, Tls};
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use rustls::pki_types::CertificateDer;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::error::{ApiError, Reason};
use super::{Credential, Gate, Refusal, ask_of};

/// The client certificate a connection presented, if any.
#[derive(Clone)]
struct Presented(Option<CertificateDer<'static>>);

/// How long a caller may take over a request, so no caller holds a connection or a route's task
/// by sending slowly or not at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadlines {
    /// For a request's headers, from the moment the connection waits for them: a new connection's
    /// first request, or a kept-alive connection's next. Past it, the connection is closed.
    pub header_read: Duration,
    /// For a request once its headers are in, body and answer both. Past it, the caller is
    /// answered 408 and the route's work is dropped.
    pub request: Duration,
}

impl Deadlines {
    /// A coordinator's: headers in 10 s, an answer in 30 s, longer than any route's own wait (a
    /// store write's deadline is 10 s, a stacks dump's 5 s).
    pub const SERVED: Self = Self {
        header_read: Duration::from_secs(10),
        request: Duration::from_secs(30),
    };
}

/// How long the listener waits after an accept error (out of file descriptors, say) before it
/// accepts again, so a persistent error does not spin it.
const ACCEPT_BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);

/// Bind `addr` and serve `routes` there over TLS with `tls`, every request admitted by `gate`, and
/// `open` beside them, routes that check their own callers, each request within `deadlines`, in
/// the background until `cancel` fires; the bound address. The bind happens before this
/// returns, so a port in use stops the coordinator at startup rather than leaving it running
/// without its API.
///
/// # Errors
///
/// The listener could not be bound, or the TLS configuration built.
pub async fn start(
    routes: Router,
    open: Router,
    addr: SocketAddr,
    tls: &Tls,
    gate: Arc<dyn Gate>,
    deadlines: Deadlines,
    cancel: CancellationToken,
) -> std::io::Result<SocketAddr> {
    // a configuration that cannot be made fails the start, not the first caller
    tls.server_config(ClientAuth::Optional)
        .map_err(std::io::Error::other)?;
    let listener = TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    info!("operator API on https://{local}");
    // `open` routes check their own callers (a node's certificate request, decision 0037)
    let guarded = routes
        .route_layer(middleware::from_fn_with_state(gate, guard))
        .merge(open)
        .layer(middleware::from_fn_with_state(deadlines.request, within));
    tokio::spawn(accept(
        listener,
        tls.clone(),
        guarded,
        deadlines.header_read,
        cancel,
    ));
    Ok(local)
}

/// Take connections on `listener` until `cancel` fires, each served on its own task, its
/// handshake by the TLS material in use as it starts (a renewed certificate, the next caller's).
async fn accept(
    listener: TcpListener,
    tls: Tls,
    guarded: Router,
    header_read: Duration,
    cancel: CancellationToken,
) {
    loop {
        let (tcp, from) = tokio::select! {
            () = cancel.cancelled() => return,
            accepted = listener.accept() => match accepted {
                Ok(a) => a,
                Err(e) => {
                    warn!("operator API: accept: {e}");
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(ACCEPT_BACKOFF) => continue,
                    }
                }
            },
        };
        let acceptor = match tls.server_config(ClientAuth::Optional) {
            Ok(config) => TlsAcceptor::from(Arc::new(config)),
            Err(e) => {
                warn!("operator API: {from}: TLS configuration: {e}");
                continue;
            }
        };
        let guarded = guarded.clone();
        tokio::spawn(async move {
            let stream = match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(tcp)).await {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => return debug!("operator API: {from}: handshake: {e}"),
                Err(_) => return debug!("operator API: {from}: no handshake in time"),
            };
            let presented = stream
                .get_ref()
                .1
                .peer_certificates()
                .and_then(|c| c.first())
                .map(|c| c.clone().into_owned());
            let app = guarded.layer(axum::Extension(Presented(presented)));
            let served = hyper::server::conn::http1::Builder::new()
                .timer(TokioTimer::new())
                .header_read_timeout(header_read)
                .serve_connection(TokioIo::new(stream), TowerToHyperService::new(app))
                .await;
            if let Err(e) = served {
                debug!("operator API: {from}: {e}");
            }
        });
    }
}

/// Answer a request within `deadline`, its body read included, or 408.
async fn within(State(deadline): State<Duration>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_owned();
    let Ok(answer) = tokio::time::timeout(deadline, next.run(req)).await else {
        warn!("operator API: {path}: no answer within {deadline:?}");
        return ApiError::new(Reason::Timeout, format!("no answer within {deadline:?}"))
            .into_response();
    };
    answer
}

/// Admit a request before its route: what it asks, from the route and its path; who asks, from
/// its bearer token or else its certificate; the gate's answer, logged.
async fn guard(
    State(gate): State<Arc<dyn Gate>>,
    matched: MatchedPath,
    params: RawPathParams,
    mut req: Request,
    next: Next,
) -> Response {
    let name = params.iter().next().map(|(_, v)| v.to_owned());
    // as the list's query reads it: a watch is `watch=true`
    let watching = req
        .uri()
        .query()
        .is_some_and(|q| q.split('&').any(|pair| pair == "watch=true"));
    let Some(ask) = ask_of(req.method(), matched.as_str(), name, watching) else {
        return ApiError::new(Reason::NotFound, "no such route").into_response();
    };
    let bearer = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_owned);
    let presented = req
        .extensions()
        .get::<Presented>()
        .and_then(|p| p.0.clone());
    let credential = match (bearer, presented) {
        (Some(token), _) => Credential::Bearer(token),
        (None, Some(cert)) => Credential::Certificate(cert),
        (None, None) => Credential::None,
    };
    match gate.admit(&credential, &ask).await {
        Ok(who) => {
            if ask.reads() {
                debug!("operator API: {} may {ask}", who.name);
            } else {
                info!("operator API: {} may {ask}", who.name);
            }
            req.extensions_mut().insert(who);
            next.run(req).await
        }
        Err(refusal) => {
            warn!("operator API: {ask} refused: {refusal}");
            let reason = match refusal {
                Refusal::Unauthenticated(_) => Reason::Unauthorized,
                Refusal::Forbidden(_) => Reason::Forbidden,
                Refusal::Unavailable(_) => Reason::Unavailable,
            };
            ApiError::new(reason, refusal.to_string()).into_response()
        }
    }
}
