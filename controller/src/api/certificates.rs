// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Callers by client certificate, outside Kubernetes (decision 0039): a certificate the
//! deployment's CA signed, valid for `<name>.operators.everserve` where the settings call `name`
//! an operator (any verb) or a viewer (reads only). Whether it is valid for that name is
//! rustls-webpki's check, as for a daemon's node name (decision 0026). Operators have a domain of
//! their own so that no node's certificate, valid for its node's name and the stores' shared
//! name, can be valid for an operator's, whatever the settings name.

use everserve_wire::transport::Peer;

use super::{Ask, Credential, Gate, Identity, Refusal};
use crate::lease::BoxFuture;

/// The domain operator and viewer certificates are valid under: `<name>.operators.everserve`.
pub const OPERATOR_DOMAIN: &str = "operators.everserve";

/// The name a certificate for operator or viewer `name` is valid for.
#[must_use]
pub fn operator_name(name: &str) -> String {
    format!("{name}.{OPERATOR_DOMAIN}")
}

/// The operators and viewers, by the names their certificates are valid for.
#[derive(Debug, Clone, Default)]
pub struct CertificateRoles {
    /// May do anything the API serves.
    pub operators: Vec<String>,
    /// May read.
    pub viewers: Vec<String>,
}

impl Gate for CertificateRoles {
    fn admit<'a>(
        &'a self,
        credential: &'a Credential,
        ask: &'a Ask,
    ) -> BoxFuture<'a, Result<Identity, Refusal>> {
        Box::pin(async move {
            let Credential::Certificate(cert) = credential else {
                return Err(Refusal::Unauthenticated("no client certificate".into()));
            };
            let peer = Peer::Certified(cert.clone());
            let named = |names: &[String]| {
                names
                    .iter()
                    .find(|n| peer.is_named(&operator_name(n)))
                    .cloned()
            };
            let identity = |name: String| Identity {
                name,
                groups: Vec::new(),
            };
            if let Some(name) = named(&self.operators) {
                return Ok(identity(name));
            }
            match named(&self.viewers) {
                Some(name) if ask.reads() => Ok(identity(name)),
                Some(name) => Err(Refusal::Forbidden(format!(
                    "{name} is a viewer and may not {ask}"
                ))),
                None => Err(Refusal::Forbidden(
                    "the certificate names no operator or viewer".into(),
                )),
            }
        })
    }
}
