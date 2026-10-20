// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A node proves its name to be issued its certificate (decision 0037), on Kubernetes, as istiod
//! issues a pod's. A node daemon presents its pod's projected service account token, for the
//! audience [`AUDIENCE`], and a public key it made; the coordinator asks the API server's
//! `TokenReview` who the token is, and signs, with the deployment's CA, a certificate for the node
//! the API server says the token's pod runs on, valid for hours. Nothing the daemon says chooses
//! the name: a token from a pod on node-a is issued node-a's certificate or none. A token of any
//! other service account, one not bound to a pod, or one whose API server does not state the
//! pod's node is refused, never given a weaker identity.
//!
//! The endpoint is outside the operator API's gate, so it bounds what any caller can make it do:
//! a body of at most [`MOST_BODY_BYTES`], a bearer refused unread unless it is shaped as a token
//! the API server issues, and at most [`MOST_REVIEWS`] reviews in flight, a caller beyond them
//! told to come back (503, `Retry-After`) rather than queued against the API server.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::certs::{CertsError, NodeKey, sign_node};
use crate::http::Client;
use crate::kubernetes::{Access, ReviewError, review_token};

/// The audience a node daemon's token is projected for.
pub const AUDIENCE: &str = "everserve";

/// What the API server says of a token bound to a pod: the pod's node.
pub const NODE_NAME: &str = "authentication.kubernetes.io/node-name";

/// What the API server says of a token bound to a pod: the pod.
pub const POD_NAME: &str = "authentication.kubernetes.io/pod-name";

/// Where a node asks for its certificate.
pub const PATH: &str = "/v1/certificates/node";

/// The most a request's body may be: a public key, hex, in JSON.
pub const MOST_BODY_BYTES: usize = 4096;

/// The most token reviews in flight at once, the API server's load from this endpoint bounded.
pub const MOST_REVIEWS: usize = 16;

/// The longest bearer taken for review: a projected token is a few kilobytes.
const MOST_TOKEN_BYTES: usize = 16 * 1024;

/// Why no certificate was issued.
#[derive(Debug, thiserror::Error)]
pub enum IssueError {
    /// The token is not one the API server knows for [`AUDIENCE`].
    #[error("{0}")]
    Unauthenticated(String),
    /// The token is known, and not a node daemon's bound to a pod on a node.
    #[error("{0}")]
    Forbidden(String),
    /// The API server could not be asked, or the certificate could not be made.
    #[error("{0}")]
    Unavailable(String),
}

/// What a node is issued.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Issued {
    /// The node the certificate names.
    pub node: String,
    /// The certificate, PEM.
    pub certificate: String,
    /// The CA it chains to, PEM: what the node verifies its peers by.
    pub ca: String,
    /// When it expires, Unix milliseconds; the node renews at half its life.
    pub not_after_ms: i64,
}

/// What a node sends.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Asked {
    /// Its public key, an uncompressed ECDSA P-256 point, hex.
    pub public_key: String,
}

/// How the coordinator issues node certificates.
pub struct Issuer {
    ca: rcgen::CertifiedKey,
    ca_pem: String,
    store_name: String,
    lifetime: Duration,
    access: Access,
    client: Client,
    /// `system:serviceaccount:<namespace>:<name>`: the node daemons'.
    account: String,
    /// The reviews that may be in flight.
    reviews: Arc<tokio::sync::Semaphore>,
}

impl std::fmt::Debug for Issuer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Issuer")
            .field("account", &self.account)
            .field("lifetime", &self.lifetime)
            .finish_non_exhaustive()
    }
}

impl Issuer {
    /// An issuer signing with `ca` (whose certificate is `ca_pem`) node certificates valid for
    /// `lifetime` and for `store_name`, to the service account `account` in `namespace`, by the
    /// API server `access` reaches.
    ///
    /// # Errors
    ///
    /// The API access's CA cannot be read.
    pub fn new(
        ca: rcgen::CertifiedKey,
        ca_pem: String,
        store_name: impl Into<String>,
        lifetime: Duration,
        access: Access,
        namespace: &str,
        account: &str,
    ) -> Result<Self, crate::http::HttpError> {
        Ok(Self {
            ca,
            ca_pem,
            store_name: store_name.into(),
            lifetime,
            client: access.client()?,
            access,
            account: format!("system:serviceaccount:{namespace}:{account}"),
            reviews: Arc::new(tokio::sync::Semaphore::new(MOST_REVIEWS)),
        })
    }

    /// This issuer with at most `reviews` token reviews in flight, rather than [`MOST_REVIEWS`].
    #[must_use]
    pub fn reviews_at_once(self, reviews: usize) -> Self {
        Self {
            reviews: Arc::new(tokio::sync::Semaphore::new(reviews)),
            ..self
        }
    }

    /// The certificate for the node `token`'s pod runs on, for `key`.
    ///
    /// # Errors
    ///
    /// [`IssueError`].
    pub async fn issue(&self, token: &str, key: NodeKey) -> Result<Issued, IssueError> {
        let reviewed = review_token(&self.client, &self.access, token, &[AUDIENCE])
            .await
            .map_err(|e| match e {
                ReviewError::Unauthenticated(why) => IssueError::Unauthenticated(why),
                ReviewError::Unavailable(why) => IssueError::Unavailable(why),
            })?;
        // the API server refuses a token reviewed for an audience it was not issued for; this
        // holds it to having said so
        if !reviewed.audiences.iter().any(|a| a == AUDIENCE) {
            return Err(IssueError::Unauthenticated(format!(
                "the token is not for the audience {AUDIENCE}"
            )));
        }
        if reviewed.username != self.account {
            return Err(IssueError::Forbidden(format!(
                "{} is not the node daemons' service account",
                reviewed.username
            )));
        }
        let one = |claim: &str| match reviewed.extra.get(claim).map(Vec::as_slice) {
            Some([value]) if !value.is_empty() => Some(value.clone()),
            _ => None,
        };
        // the API server states a node for a token bound to a pod alone, so this is the pod's
        // binding too; a node's name is never taken from anything else
        let Some(node) = one(NODE_NAME) else {
            return Err(IssueError::Forbidden(
                "the API server does not say which node the token's pod runs on: the token is \
                 bound to no pod, or the API server states no node"
                    .to_owned(),
            ));
        };
        let pod = one(POD_NAME).unwrap_or_default();
        let (certificate, not_after) =
            sign_node(&self.ca, &node, &self.store_name, &key, self.lifetime).map_err(
                |e| match e {
                    CertsError::Material { reason, .. } => IssueError::Unavailable(reason),
                    other => IssueError::Unavailable(other.to_string()),
                },
            )?;
        info!("issued node {node} its certificate (pod {pod}), valid until {not_after}");
        Ok(Issued {
            node,
            certificate,
            ca: self.ca_pem.clone(),
            not_after_ms: i64::try_from(not_after.unix_timestamp_nanos() / 1_000_000)
                .unwrap_or(i64::MAX),
        })
    }
}

/// The issuing endpoint, [`PATH`]: a node's token as its bearer, its key in the body. Served on
/// the operator API's listener, beside the operator API and not behind its gate: a node is no
/// operator, and its token is checked here.
pub fn router(issuer: Arc<Issuer>) -> Router {
    Router::new()
        .route(PATH, post(ask))
        .layer(DefaultBodyLimit::max(MOST_BODY_BYTES))
        .with_state(issuer)
}

/// Whether `token` is shaped as a token the API server issues: three non-empty base64url parts,
/// within [`MOST_TOKEN_BYTES`]. One that is not is refused unread, never sent for review.
fn token_shaped(token: &str) -> bool {
    let parts: Vec<&str> = token.split('.').collect();
    token.len() <= MOST_TOKEN_BYTES
        && parts.len() == 3
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        })
}

async fn ask(
    State(issuer): State<Arc<Issuer>>,
    headers: HeaderMap,
    axum::Json(asked): axum::Json<Asked>,
) -> Response {
    let Some(token) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    else {
        return (StatusCode::UNAUTHORIZED, "a node's token is its bearer").into_response();
    };
    if !token_shaped(token) {
        return (StatusCode::UNAUTHORIZED, "the bearer is not a token").into_response();
    }
    let key = match unhex(&asked.public_key).map(NodeKey::new) {
        Some(Ok(key)) => key,
        Some(Err(e)) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        None => return (StatusCode::BAD_REQUEST, "the key is not hex").into_response(),
    };
    // checked cheaply first, then a review: only a well-formed request waits on the API server
    let Ok(_review) = issuer.reviews.try_acquire() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            "the issuer is reviewing as many tokens as it takes at once",
        )
            .into_response();
    };
    match issuer.issue(token, key).await {
        Ok(certificate) => axum::Json(certificate).into_response(),
        Err(e) => {
            warn!("node certificate refused: {e}");
            let status = match e {
                IssueError::Unauthenticated(_) => StatusCode::UNAUTHORIZED,
                IssueError::Forbidden(_) => StatusCode::FORBIDDEN,
                IssueError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            };
            (status, e.to_string()).into_response()
        }
    }
}

/// `text`'s bytes, two hex digits each; None when it is not that.
#[must_use]
pub fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

/// `bytes` as hex.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::{hex, token_shaped, unhex};

    #[test]
    fn hex_goes_both_ways() {
        let bytes = vec![0x04, 0xff, 0x00, 0x7a];
        assert_eq!(hex(&bytes), "04ff007a");
        assert_eq!(unhex("04ff007a"), Some(bytes));
        assert_eq!(unhex("04f"), None);
        assert_eq!(unhex("zz"), None);
    }

    #[test]
    fn only_a_bearer_shaped_as_a_token_is_reviewed() {
        assert!(token_shaped("eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0.c2ln-_"));
        for not in [
            "",
            "a.b",
            "a.b.c.d",
            "a..c",
            "a.b.c d",
            "a.b.c=",
            &"a".repeat(20_000),
        ] {
            assert!(!token_shaped(not), "{not:.20}");
        }
    }
}
