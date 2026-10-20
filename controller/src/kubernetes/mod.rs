// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The Kubernetes API as this process reaches it: an endpoint, the CA its server chains to, and
//! a service account's token. One client for every caller here (the node notice source, the
//! coordinator lease), so each reaches the API the same way: the token is read again at every
//! request, since the kubelet rotates a projected token and one read once expires within the
//! hour, as client-go reads it again for the same reason.

pub mod nodes;
pub mod objects;

use std::path::{Path, PathBuf};

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::http::{Bearer, Client, HttpError, Method};

/// The API server's in-cluster name, when no endpoint is given.
pub const IN_CLUSTER_API: &str = "https://kubernetes.default.svc:443";

/// Where a pod's service account token is mounted.
pub const SERVICE_ACCOUNT_TOKEN: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";

/// Where a pod's cluster CA is mounted.
pub const SERVICE_ACCOUNT_CA: &str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";

/// How this process reaches the Kubernetes API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Access {
    /// The API server, `https://host:port` (or `http://` for a test server).
    pub endpoint: String,
    /// The bearer token's file, read again at every request.
    pub token_file: PathBuf,
    /// The CA the API server's certificate chains to, for an `https` endpoint.
    pub ca_file: PathBuf,
}

impl Access {
    /// The client for this access.
    ///
    /// # Errors
    ///
    /// [`HttpError`] when the endpoint is `https` and the CA cannot be read.
    pub fn client(&self) -> Result<Client, HttpError> {
        let roots = if self.endpoint.starts_with("https://") {
            Some(roots(&self.ca_file).map_err(|reason| HttpError {
                url: self.endpoint.clone(),
                reason,
            })?)
        } else {
            None
        };
        Ok(Client::with(
            roots,
            Some(Bearer::File(self.token_file.clone())),
        ))
    }

    /// `path` on the API server.
    #[must_use]
    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.endpoint.trim_end_matches('/'))
    }
}

fn roots(ca_file: &Path) -> Result<std::sync::Arc<rustls::RootCertStore>, String> {
    everserve_wire::transport::root_store(ca_file).map_err(|e| e.to_string())
}

/// Who a bearer token is, as the API server's `TokenReview` says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reviewed {
    /// The user's name: a service account's is `system:serviceaccount:<namespace>:<name>`.
    pub username: String,
    /// Its groups.
    pub groups: Vec<String>,
    /// What the API server adds: for a token bound to a pod, the pod's name and its node's
    /// (`authentication.kubernetes.io/pod-name`, `authentication.kubernetes.io/node-name`).
    pub extra: BTreeMap<String, Vec<String>>,
    /// The audiences the token was found valid for, of those asked.
    pub audiences: Vec<String>,
    /// The review's `user` as the API server sent it.
    pub user: Value,
}

/// Why a token was not reviewed as someone.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum ReviewError {
    /// The API server does not know the token, or not for the audiences asked.
    #[error("{0}")]
    Unauthenticated(String),
    /// The API server could not be asked.
    #[error("the API server could not review the token: {0}")]
    Unavailable(String),
}

/// Who `token` is, by the API server `access` reaches, valid for `audiences` (none: the API
/// server's own).
///
/// # Errors
///
/// [`ReviewError`].
pub async fn review_token(
    client: &Client,
    access: &Access,
    token: &str,
    audiences: &[&str],
) -> Result<Reviewed, ReviewError> {
    let mut spec = json!({"token": token});
    if !audiences.is_empty() {
        spec["audiences"] = json!(audiences);
    }
    let body = json!({
        "apiVersion": "authentication.k8s.io/v1",
        "kind": "TokenReview",
        "spec": spec,
    });
    let path = "/apis/authentication.k8s.io/v1/tokenreviews";
    let answer = client
        .send(
            Method::POST,
            &access.url(path),
            &[
                ("Content-Type", "application/json"),
                ("Accept", "application/json"),
            ],
            Some(body.to_string().into_bytes()),
        )
        .await
        .map_err(|e| ReviewError::Unavailable(e.to_string()))?;
    if answer.status != 201 && answer.status != 200 {
        return Err(ReviewError::Unavailable(format!(
            "{path} answered {}: {}",
            answer.status, answer.body
        )));
    }
    let review: Value =
        serde_json::from_str(&answer.body).map_err(|e| ReviewError::Unavailable(e.to_string()))?;
    let status = &review["status"];
    if status["authenticated"].as_bool() != Some(true) {
        return Err(ReviewError::Unauthenticated(
            status["error"]
                .as_str()
                .unwrap_or("the cluster does not know the token")
                .to_owned(),
        ));
    }
    let user = status["user"].clone();
    let strings = |v: &Value| -> Vec<String> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    Ok(Reviewed {
        username: user["username"].as_str().unwrap_or_default().to_owned(),
        groups: strings(&user["groups"]),
        extra: user["extra"]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), strings(v))).collect())
            .unwrap_or_default(),
        audiences: strings(&status["audiences"]),
        user,
    })
}
