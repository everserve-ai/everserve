// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator API behind the question of who asks (decision 0039). Every request is
//! authenticated (a client certificate from the deployment's CA, or a bearer token the cluster
//! vouches for) and authorized as a verb on a resource of the `everserve.ai` group, before the
//! route sees it. The routes and their objects are decision 0042's ([`routes`], [`objects`]):
//! engines, nodes and pools, each a spec and a status, listed, read, written and watched one
//! way, their changes numbered by [`Feed`]; this module serves them over TLS behind a [`Gate`].

mod certificates;
pub mod error;
pub mod feed;
mod kubernetes;
pub mod objects;
pub mod routes;
mod server;

pub use certificates::{CertificateRoles, OPERATOR_DOMAIN, operator_name};
pub use feed::{Feed, Pacing};
pub use kubernetes::KubernetesGate;
pub use server::{Deadlines, start};

use rustls::pki_types::CertificateDer;

use crate::lease::BoxFuture;

/// What a request asks to do, as Kubernetes RBAC names an action: a verb on a resource of the
/// `everserve.ai` group, and its name when the request names one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ask {
    /// `get`, `list`, `watch`, `update` or `create`.
    pub verb: &'static str,
    /// `engines`, `nodes`, `pools` or `topology`.
    pub resource: &'static str,
    /// `events`, an object's event log; `stacks`, an engine's processes' stacks; `launcher`,
    /// a pool's spare launcher.
    pub subresource: Option<&'static str>,
    /// The resource's name: the node, the engine or the pool.
    pub name: Option<String>,
}

impl Ask {
    /// Whether the ask only reads.
    #[must_use]
    pub fn reads(&self) -> bool {
        matches!(self.verb, "get" | "list" | "watch")
    }
}

impl std::fmt::Display for Ask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.verb, self.resource)?;
        if let Some(s) = self.subresource {
            write!(f, "/{s}")?;
        }
        if let Some(n) = &self.name {
            write!(f, " {n}")?;
        }
        Ok(())
    }
}

/// What a caller presented.
#[derive(Debug, Clone)]
pub enum Credential {
    /// A client certificate the deployment's CA signed, verified at the handshake.
    Certificate(CertificateDer<'static>),
    /// A bearer token from the `Authorization` header.
    Bearer(String),
    /// Nothing.
    None,
}

/// Who a request was made by, once authenticated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// The caller's name: a certificate's operator or viewer name, or a Kubernetes user.
    pub name: String,
    /// The Kubernetes groups it belongs to; none for a certificate.
    pub groups: Vec<String>,
}

impl Identity {
    /// What a write by this caller is recorded as written by (decision 0036's `written_by`).
    #[must_use]
    pub fn writer(&self) -> String {
        format!("operator api: {}", self.name)
    }
}

/// Why a request was not let through.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    /// No credential, or one that proves no one: 401.
    #[error("unauthenticated: {0}")]
    Unauthenticated(String),
    /// A known caller without the right: 403.
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// What decides could not be asked (the cluster's API server): 503.
    #[error("unavailable: {0}")]
    Unavailable(String),
}

/// What decides who asks and whether they may.
pub trait Gate: Send + Sync {
    /// The caller behind `credential`, if it may do `ask`.
    ///
    /// # Errors
    ///
    /// [`Refusal`] when it is no one known, may not, or the answer cannot be had.
    fn admit<'a>(
        &'a self,
        credential: &'a Credential,
        ask: &'a Ask,
    ) -> BoxFuture<'a, Result<Identity, Refusal>>;
}

/// The gates a deployment configured: certificates by their names, bearer tokens by the cluster.
/// A credential of a kind no gate takes is refused.
#[derive(Default)]
pub struct Gates {
    /// Client certificates, by the operator and viewer names they are valid for.
    pub certificates: Option<CertificateRoles>,
    /// Bearer tokens, by `TokenReview` and `SubjectAccessReview`.
    pub kubernetes: Option<KubernetesGate>,
}

impl Gate for Gates {
    fn admit<'a>(
        &'a self,
        credential: &'a Credential,
        ask: &'a Ask,
    ) -> BoxFuture<'a, Result<Identity, Refusal>> {
        Box::pin(async move {
            match credential {
                Credential::Certificate(_) => match &self.certificates {
                    Some(gate) => gate.admit(credential, ask).await,
                    None => Err(Refusal::Unauthenticated(
                        "client certificates are not taken here".into(),
                    )),
                },
                Credential::Bearer(_) => match &self.kubernetes {
                    Some(gate) => gate.admit(credential, ask).await,
                    None => Err(Refusal::Unauthenticated(
                        "bearer tokens are not taken here".into(),
                    )),
                },
                Credential::None => Err(Refusal::Unauthenticated(
                    "present an operator certificate or a bearer token".into(),
                )),
            }
        })
    }
}

/// What the route `matched` (as axum names it) with `method` asks, with its path's `name`;
/// `watching` for a list asked with `?watch=true`, which RBAC grants as its own verb. None for a
/// route the API does not serve.
#[must_use]
pub fn ask_of(
    method: &axum::http::Method,
    matched: &str,
    name: Option<String>,
    watching: bool,
) -> Option<Ask> {
    let ask = |verb, resource, subresource, name| Ask {
        verb,
        resource,
        subresource,
        name,
    };
    let list = if watching { "watch" } else { "list" };
    Some(match (method.as_str(), matched) {
        ("GET", "/v1/engines") => ask(list, "engines", None, None),
        ("GET", "/v1/engines/{engine}") => ask("get", "engines", None, name),
        ("GET", "/v1/engines/{engine}/events") => ask("get", "engines", Some("events"), name),
        // signals a live process to dump its stacks: no viewer's
        ("POST", "/v1/engines/{engine}/stacks") => ask("create", "engines", Some("stacks"), name),
        ("GET", "/v1/nodes") => ask(list, "nodes", None, None),
        ("GET", "/v1/nodes/{node}") => ask("get", "nodes", None, name),
        ("PUT", "/v1/nodes/{node}") => ask("update", "nodes", None, name),
        ("GET", "/v1/nodes/{node}/events") => ask("get", "nodes", Some("events"), name),
        ("GET", "/v1/nodes/{node}/release") => ask("get", "nodes", Some("release"), name),
        ("GET", "/v1/pools") => ask(list, "pools", None, None),
        ("GET", "/v1/pools/{pool}") => ask("get", "pools", None, name),
        ("PUT", "/v1/pools/{pool}") => ask("update", "pools", None, name),
        // a command every node of the pool runs, and its environment: a grant of its own
        ("GET", "/v1/pools/{pool}/launcher") => ask("get", "pools", Some("launcher"), name),
        ("PUT", "/v1/pools/{pool}/launcher") => ask("update", "pools", Some("launcher"), name),
        ("GET", "/v1/pools/{pool}/events") => ask("get", "pools", Some("events"), name),
        ("GET", "/v1/topology") => ask("get", "topology", None, None),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::ask_of;
    use axum::http::Method;

    fn asked(method: &Method, path: &str, name: Option<&str>, watching: bool) -> Option<String> {
        ask_of(method, path, name.map(str::to_owned), watching).map(|a| a.to_string())
    }

    #[test]
    fn every_route_asks_a_verb_on_a_resource() {
        let get = |p, n| asked(&Method::GET, p, n, false);
        let put = |p, n| asked(&Method::PUT, p, n, false);
        assert_eq!(get("/v1/engines", None).as_deref(), Some("list engines"));
        assert_eq!(
            asked(&Method::GET, "/v1/engines", None, true).as_deref(),
            Some("watch engines")
        );
        assert_eq!(
            put("/v1/nodes/{node}", Some("node-a")).as_deref(),
            Some("update nodes node-a")
        );
        assert_eq!(
            asked(
                &Method::POST,
                "/v1/engines/{engine}/stacks",
                Some("k3-a"),
                false
            )
            .as_deref(),
            Some("create engines/stacks k3-a")
        );
        assert_eq!(
            put("/v1/pools/{pool}/launcher", Some("h100")).as_deref(),
            Some("update pools/launcher h100")
        );
        assert_eq!(
            get("/v1/pools/{pool}/launcher", Some("h100")).as_deref(),
            Some("get pools/launcher h100"),
            "a launcher's environment is read under its own grant too"
        );
        assert_eq!(
            asked(&Method::POST, "/v1/nodes", None, false),
            None,
            "an unserved route asks nothing"
        );
    }

    #[test]
    fn every_route_the_api_serves_asks_something() {
        // a route added without its ask would be refused to everyone, or, worse, mapped by hand
        // later to the wrong grant: every operation the document lists must have one
        let document = super::routes::document();
        let mut operations = 0;
        for (path, item) in &document.paths.paths {
            for (method, op) in [
                (Method::GET, &item.get),
                (Method::PUT, &item.put),
                (Method::POST, &item.post),
                (Method::DELETE, &item.delete),
                (Method::PATCH, &item.patch),
            ] {
                if op.is_some() {
                    operations += 1;
                    assert!(
                        ask_of(&method, path, Some("x".into()), false).is_some(),
                        "{method} {path} asks nothing"
                    );
                }
            }
        }
        assert!(
            operations >= 15,
            "the document lists {operations} operations"
        );
    }
}
