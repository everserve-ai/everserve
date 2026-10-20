// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Callers by bearer token, on Kubernetes (decision 0039), as kube-rbac-proxy and aggregated API
//! servers take them: the API server's `TokenReview` says who the token is, its
//! `SubjectAccessReview` whether that user may do the ask, as a verb on an `everserve.ai`
//! resource in the coordinator's namespace. Operators are granted by RBAC. Both answers are kept
//! for [`CACHE`], so a polling caller costs one review a period and a revoked grant ends within it;
//! an API server that cannot answer is never cached. At most [`MOST_REVIEWS`] reviews are in
//! flight: a caller beyond them is told to come back (503) rather than queued against the API
//! server, as the node certificate endpoint bounds its own reviews (`issuer`). The cache keys a
//! token by its digest, never the token itself.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::{Ask, Credential, Gate, Identity, Refusal};
use crate::http::{Client, Method};
use crate::kubernetes::{Access, ReviewError};
use crate::lease::BoxFuture;

/// How long an answer is kept.
pub const CACHE: Duration = Duration::from_secs(10);

/// The most answers kept: past it, the expired go, and if none have, all do.
const CACHE_ENTRIES: usize = 10_000;

/// The most token and access reviews in flight at once: a caller beyond them is answered 503.
pub const MOST_REVIEWS: usize = 16;

/// The API group every ask is a verb on.
pub const GROUP: &str = "everserve.ai";

/// A cached answer: when it was had, and what it was.
type Answer = (Instant, Result<Identity, Refusal>);

/// A token's SHA-256, the cache's key for it.
type TokenDigest = [u8; 32];

/// Bearer tokens checked by the cluster's API server.
pub struct KubernetesGate {
    client: Client,
    access: Access,
    namespace: String,
    answers: Mutex<HashMap<(TokenDigest, Ask), Answer>>,
    reviews: tokio::sync::Semaphore,
}

impl std::fmt::Debug for KubernetesGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KubernetesGate")
            .field("api", &self.access.endpoint)
            .field("namespace", &self.namespace)
            .finish_non_exhaustive()
    }
}

impl KubernetesGate {
    /// A gate asking the API server `access` reaches, under this process's own service account,
    /// for asks in `namespace`.
    ///
    /// # Errors
    ///
    /// The API access's CA cannot be read.
    pub fn new(
        access: Access,
        namespace: impl Into<String>,
    ) -> Result<Self, crate::http::HttpError> {
        Ok(Self {
            client: access.client()?,
            access,
            namespace: namespace.into(),
            answers: Mutex::new(HashMap::new()),
            reviews: tokio::sync::Semaphore::new(MOST_REVIEWS),
        })
    }

    /// This gate with at most `reviews` reviews in flight, rather than `MOST_REVIEWS`.
    #[must_use]
    pub fn reviews_at_once(self, reviews: usize) -> Self {
        Self {
            reviews: tokio::sync::Semaphore::new(reviews),
            ..self
        }
    }

    async fn review(&self, path: &str, body: Value) -> Result<Value, Refusal> {
        let answer = self
            .client
            .send(
                Method::POST,
                &self.access.url(path),
                &[
                    ("Content-Type", "application/json"),
                    ("Accept", "application/json"),
                ],
                Some(body.to_string().into_bytes()),
            )
            .await
            .map_err(|e| Refusal::Unavailable(e.to_string()))?;
        if answer.status != 201 && answer.status != 200 {
            return Err(Refusal::Unavailable(format!(
                "{path} answered {}: {}",
                answer.status, answer.body
            )));
        }
        serde_json::from_str(&answer.body).map_err(|e| Refusal::Unavailable(e.to_string()))
    }

    /// Who `token` is, by `TokenReview`.
    async fn who(&self, token: &str) -> Result<(Identity, Value), Refusal> {
        let reviewed = crate::kubernetes::review_token(&self.client, &self.access, token, &[])
            .await
            .map_err(|e| match e {
                ReviewError::Unauthenticated(why) => Refusal::Unauthenticated(why),
                ReviewError::Unavailable(why) => Refusal::Unavailable(why),
            })?;
        let identity = Identity {
            name: reviewed.username,
            groups: reviewed.groups,
        };
        Ok((identity, reviewed.user))
    }

    /// Whether `user` may do `ask`, by `SubjectAccessReview`.
    async fn may(&self, user: &Value, ask: &Ask) -> Result<(), Refusal> {
        let mut attributes = json!({
            "namespace": self.namespace,
            "verb": ask.verb,
            "group": GROUP,
            "resource": ask.resource,
        });
        if let Some(s) = ask.subresource {
            attributes["subresource"] = json!(s);
        }
        if let Some(n) = &ask.name {
            attributes["name"] = json!(n);
        }
        let review = self
            .review(
                "/apis/authorization.k8s.io/v1/subjectaccessreviews",
                json!({
                    "apiVersion": "authorization.k8s.io/v1",
                    "kind": "SubjectAccessReview",
                    "spec": {
                        "user": user["username"],
                        "uid": user["uid"],
                        "groups": user["groups"],
                        "extra": user["extra"],
                        "resourceAttributes": attributes,
                    },
                }),
            )
            .await?;
        if review["status"]["allowed"].as_bool() == Some(true) {
            Ok(())
        } else {
            Err(Refusal::Forbidden(format!(
                "{} may not {ask}{}",
                user["username"].as_str().unwrap_or("the caller"),
                review["status"]["reason"]
                    .as_str()
                    .filter(|r| !r.is_empty())
                    .map(|r| format!(": {r}"))
                    .unwrap_or_default()
            )))
        }
    }

    fn cached(&self, key: &(TokenDigest, Ask)) -> Option<Result<Identity, Refusal>> {
        let answers = self
            .answers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        answers
            .get(key)
            .filter(|(at, _)| at.elapsed() < CACHE)
            .map(|(_, answer)| answer.clone())
    }

    fn keep(&self, key: (TokenDigest, Ask), answer: &Result<Identity, Refusal>) {
        if matches!(answer, Err(Refusal::Unavailable(_))) {
            return;
        }
        let mut answers = self
            .answers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if answers.len() >= CACHE_ENTRIES {
            answers.retain(|_, (at, _)| at.elapsed() < CACHE);
            if answers.len() >= CACHE_ENTRIES {
                answers.clear();
            }
        }
        answers.insert(key, (Instant::now(), answer.clone()));
    }
}

/// `token`'s SHA-256.
fn digest_of(token: &str) -> TokenDigest {
    let digest = ring::digest::digest(&ring::digest::SHA256, token.as_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_ref());
    out
}

impl Gate for KubernetesGate {
    fn admit<'a>(
        &'a self,
        credential: &'a Credential,
        ask: &'a Ask,
    ) -> BoxFuture<'a, Result<Identity, Refusal>> {
        Box::pin(async move {
            let Credential::Bearer(token) = credential else {
                return Err(Refusal::Unauthenticated("no bearer token".into()));
            };
            let key = (digest_of(token), ask.clone());
            if let Some(answer) = self.cached(&key) {
                return answer;
            }
            let Ok(_review) = self.reviews.try_acquire() else {
                return Err(Refusal::Unavailable(
                    "the gate is reviewing as many tokens as it takes at once; come back".into(),
                ));
            };
            let answer = match self.who(token).await {
                Ok((identity, user)) => self.may(&user, ask).await.map(|()| identity),
                Err(e) => Err(e),
            };
            self.keep(key, &answer);
            answer
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(reviews: usize) -> KubernetesGate {
        let access = Access {
            endpoint: "http://127.0.0.1:9".into(),
            token_file: "/nonexistent/token".into(),
            ca_file: "/nonexistent/ca.crt".into(),
        };
        KubernetesGate::new(access, "everserve")
            .expect("an http endpoint needs no CA")
            .reviews_at_once(reviews)
    }

    #[tokio::test]
    async fn a_caller_beyond_the_reviews_in_flight_is_told_to_come_back_without_a_review() {
        let gate = gate(0);
        let ask = Ask {
            verb: "get",
            resource: "engines",
            subresource: None,
            name: None,
        };
        let refused = gate
            .admit(&Credential::Bearer("a-token".into()), &ask)
            .await
            .expect_err("no review can run");
        assert!(matches!(refused, Refusal::Unavailable(_)), "{refused}");
        assert!(
            gate.answers.lock().expect("lock").is_empty(),
            "nothing was cached"
        );
    }

    #[test]
    fn the_cache_keys_a_token_by_its_digest() {
        assert_ne!(digest_of("one"), digest_of("two"));
        assert_eq!(digest_of("one"), digest_of("one"));
    }
}
