// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A node's side of decision 0037: it is issued its certificate by the coordinator, and renews it
//! at half its life. A key is made here and never leaves the node; the coordinator is sent its
//! public half and the pod's projected service account token, and answers with the certificate
//! for the node the token's pod runs on ([`crate::issuer`]). The key, the certificate and the CA
//! are written to one directory, each file replaced in one step, where the daemon and the store
//! read them as their TLS material and read them again as they change
//! ([`everserve_wire::transport::Tls::follow`]).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use everserve_wire::transport::jittered;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::http::{Bearer, Client, Method};
use crate::issuer::{Asked, Issued, PATH, hex};

/// The node's certificate, in the directory [`Enrollment`] writes.
pub const CERT_FILE: &str = "node.crt";
/// The node's key.
pub const KEY_FILE: &str = "node.key";
/// The CA its peers chain to.
pub const CA_FILE: &str = "ca.crt";
/// What was issued, last: the node and when the certificate expires (JSON, [`Held`]). Written
/// after the certificate, so it never says more than the files hold.
pub const ISSUED_FILE: &str = "issued.json";

/// What [`ISSUED_FILE`] records.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Held {
    /// The node the certificate names.
    pub node: String,
    /// When it expires, Unix milliseconds.
    pub not_after_ms: i64,
}

/// The first wait after a failed request; it doubles to [`RETRY_MOST`].
pub const RETRY_FIRST: Duration = Duration::from_secs(5);
/// The longest wait between two requests after failures.
pub const RETRY_MOST: Duration = Duration::from_mins(1);

/// Why no certificate was written.
#[derive(Debug, thiserror::Error)]
pub enum EnrollError {
    /// The coordinator could not be asked, or did not answer.
    #[error("the issuer: {0}")]
    Unreachable(String),
    /// The coordinator answered with no certificate.
    #[error("the issuer refused, {status}: {reason}")]
    Refused {
        /// Its status.
        status: u16,
        /// What it said.
        reason: String,
    },
    /// A key could not be made, or a file written.
    #[error("{0}")]
    Local(String),
}

/// How a node is issued its certificate, and where it is written.
#[derive(Clone)]
pub struct Enrollment {
    url: String,
    client: Client,
    out: PathBuf,
}

impl std::fmt::Debug for Enrollment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Enrollment")
            .field("url", &self.url)
            .field("out", &self.out)
            .finish_non_exhaustive()
    }
}

impl Enrollment {
    /// Ask the coordinator's operator API at `api` (`host:port` or `https://host:port`), whose
    /// certificate names `coordinator_name` and chains to the CA in `ca`, with the token in
    /// `token_file`, read again at every request as the kubelet rotates it; write into `out`.
    ///
    /// # Errors
    ///
    /// The CA cannot be read.
    pub fn new(
        api: &str,
        coordinator_name: &str,
        ca: &Path,
        token_file: &Path,
        out: &Path,
    ) -> Result<Self, EnrollError> {
        let roots = everserve_wire::transport::root_store(ca)
            .map_err(|e| EnrollError::Local(e.to_string()))?;
        let base = if api.starts_with("https://") {
            api.trim_end_matches('/').to_owned()
        } else {
            format!("https://{api}")
        };
        Ok(Self {
            url: format!("{base}{PATH}"),
            client: Client::with(Some(roots), Some(Bearer::File(token_file.to_path_buf())))
                .server_name(coordinator_name),
            out: out.to_path_buf(),
        })
    }

    /// Be issued a certificate for a new key, and write the three files: the key first, so the
    /// certificate is never on disk without it, each replacing its old file in one step.
    ///
    /// # Errors
    ///
    /// [`EnrollError`].
    pub async fn once(&self) -> Result<Issued, EnrollError> {
        let key = rcgen::KeyPair::generate().map_err(|e| EnrollError::Local(e.to_string()))?;
        let asked = Asked {
            public_key: hex(key.public_key_raw()),
        };
        let body = serde_json::to_vec(&asked).map_err(|e| EnrollError::Local(e.to_string()))?;
        let answer = self
            .client
            .send(
                Method::POST,
                &self.url,
                &[("content-type", "application/json")],
                Some(body),
            )
            .await
            .map_err(|e| EnrollError::Unreachable(e.to_string()))?;
        if answer.status != 200 {
            return Err(EnrollError::Refused {
                status: answer.status,
                reason: answer.body,
            });
        }
        let issued: Issued = serde_json::from_str(&answer.body)
            .map_err(|e| EnrollError::Unreachable(format!("not a certificate: {e}")))?;
        let write = |file: &str, pem: &str, private: bool| {
            let path = self.out.join(file);
            crate::certs::replace_pem(&path, pem, private)
                .map_err(|e| EnrollError::Local(format!("{}: {e}", path.display())))
        };
        write(KEY_FILE, &key.serialize_pem(), true)?;
        write(CERT_FILE, &issued.certificate, false)?;
        write(CA_FILE, &issued.ca, false)?;
        let held = serde_json::to_string(&Held {
            node: issued.node.clone(),
            not_after_ms: issued.not_after_ms,
        })
        .map_err(|e| EnrollError::Local(e.to_string()))?;
        write(ISSUED_FILE, &held, false)?;
        info!(
            "issued node {} its certificate, valid until {} ms",
            issued.node, issued.not_after_ms
        );
        Ok(issued)
    }

    /// Be issued a certificate, asking again after a failure, until one is written or `cancel`
    /// fires (None). A failed request is asked again after [`RETRY_FIRST`], doubling to
    /// [`RETRY_MOST`]; files already in place stay in use meanwhile, and are said to have
    /// expired once `expires` (Unix milliseconds) has passed.
    pub async fn issued(&self, expires: i64, cancel: &CancellationToken) -> Option<Issued> {
        let mut retry = RETRY_FIRST;
        loop {
            match self.once().await {
                Ok(issued) => return Some(issued),
                Err(e) if now_ms() >= expires => {
                    warn!("the node's certificate has expired, and being issued one failed: {e}");
                }
                Err(e) => warn!("being issued the node's certificate failed: {e}"),
            }
            let wait = jittered(retry);
            info!("asking for the node's certificate again in {wait:?}");
            tokio::select! {
                () = cancel.cancelled() => return None,
                () = tokio::time::sleep(wait) => {}
            }
            retry = (retry * 2).min(RETRY_MOST);
        }
    }

    /// What the directory holds, from [`ISSUED_FILE`]; None when it holds nothing readable.
    #[must_use]
    pub fn held(&self) -> Option<Held> {
        let text = std::fs::read_to_string(self.out.join(ISSUED_FILE)).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Renew the certificate at about half its remaining life, until `cancel` fires: the one
    /// the directory holds (written by an init container's `--once`), or else one issued now.
    pub async fn keep(&self, cancel: &CancellationToken) {
        let mut expires = match self.held() {
            Some(held) if held.not_after_ms > now_ms() => {
                info!(
                    "renewing the certificate node-cert was issued for {}",
                    held.node
                );
                held.not_after_ms
            }
            _ => match self.issued(i64::MAX, cancel).await {
                Some(issued) => issued.not_after_ms,
                None => return,
            },
        };
        loop {
            // at half the life left, give or take a quarter of that: nodes issued together (a
            // fleet at its start) do not all renew together
            let half = u64::try_from((expires - now_ms()) / 2).unwrap_or(0);
            tokio::select! {
                () = cancel.cancelled() => return,
                () = tokio::time::sleep(jittered(Duration::from_millis(half))) => {}
            }
            match self.issued(expires, cancel).await {
                Some(issued) => expires = issued.not_after_ms,
                None => return,
            }
        }
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}
