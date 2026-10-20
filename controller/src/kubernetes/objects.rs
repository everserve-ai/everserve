// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! One kind of object in one namespace, as the API server keeps it: read, listed by label,
//! created, and replaced guarded by the `resourceVersion` it carries. What the coordinator's
//! `Lease` and its spec store's objects share; each caller says what a conflict means to it.

use std::time::Duration;

use serde_json::Value;

use crate::http::{Client, Method};

use super::Access;

/// What a write found.
#[derive(Debug)]
pub enum Put {
    /// The object as written.
    Written(Value),
    /// Another write landed since the version this one named, or, for a create, the object
    /// exists.
    Conflict,
    /// The object does not exist.
    Gone,
}

/// The objects of one kind in one namespace.
#[derive(Clone)]
pub struct Objects {
    client: Client,
    /// The collection's URL.
    collection: String,
    /// What they are, for errors: `leases in everserve`.
    what: String,
}

impl std::fmt::Debug for Objects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Objects")
            .field("what", &self.what)
            .finish_non_exhaustive()
    }
}

impl Objects {
    /// The objects of the resource `plural` in API group version `api` (`coordination.k8s.io/v1`),
    /// in `namespace`, reached by `access`, each request bounded by `timeout`.
    ///
    /// # Errors
    ///
    /// The access's CA cannot be read.
    pub fn new(
        access: &Access,
        api: &str,
        plural: &str,
        namespace: &str,
        timeout: Duration,
    ) -> Result<Self, String> {
        let client = access.client().map_err(|e| e.to_string())?.timeout(timeout);
        let collection = access.url(&format!("/apis/{api}/namespaces/{namespace}/{plural}"));
        Ok(Self {
            client,
            collection,
            what: format!("{plural} in {namespace}"),
        })
    }

    /// The objects of the core group's resource `plural` (`pods`, `configmaps`) in `namespace`,
    /// reached by `access`, each request bounded by `timeout`: the core group's path is
    /// `/api/v1`, not `/apis/<group>/<version>`.
    ///
    /// # Errors
    ///
    /// The access's CA cannot be read.
    pub fn core(
        access: &Access,
        plural: &str,
        namespace: &str,
        timeout: Duration,
    ) -> Result<Self, String> {
        let client = access.client().map_err(|e| e.to_string())?.timeout(timeout);
        let collection = access.url(&format!("/api/v1/namespaces/{namespace}/{plural}"));
        Ok(Self {
            client,
            collection,
            what: format!("{plural} in {namespace}"),
        })
    }

    /// What they are, for errors.
    #[must_use]
    pub fn what(&self) -> &str {
        &self.what
    }

    async fn request(
        &self,
        method: Method,
        url: &str,
        body: Option<&Value>,
    ) -> Result<(u16, Option<Value>), String> {
        self.request_as(method, url, "application/json", body).await
    }

    async fn request_as(
        &self,
        method: Method,
        url: &str,
        content_type: &str,
        body: Option<&Value>,
    ) -> Result<(u16, Option<Value>), String> {
        let body = body.map(|b| b.to_string().into_bytes());
        let answer = self
            .client
            .send(
                method,
                url,
                &[
                    ("Accept", "application/json"),
                    ("Content-Type", content_type),
                ],
                body,
            )
            .await
            .map_err(|e| e.to_string())?;
        Ok((answer.status, serde_json::from_str(&answer.body).ok()))
    }

    /// Merge `patch` into the object `name` (RFC 7386: a null removes a field), as the object
    /// is after; [`Put::Gone`] when there is no such object.
    ///
    /// # Errors
    ///
    /// The API server could not be asked, or answered otherwise.
    pub async fn merge_patch(&self, name: &str, patch: &Value) -> Result<Put, String> {
        match self
            .request_as(
                Method::PATCH,
                &format!("{}/{name}", self.collection),
                "application/merge-patch+json",
                Some(patch),
            )
            .await?
        {
            (200, Some(v)) => Ok(Put::Written(v)),
            (409, _) => Ok(Put::Conflict),
            (404, _) => Ok(Put::Gone),
            (status, v) => Err(format!("PATCH {name} answered {status}: {v:?}")),
        }
    }

    /// The object `name`; None while it does not exist.
    ///
    /// # Errors
    ///
    /// The API server could not be asked, or answered otherwise.
    pub async fn get(&self, name: &str) -> Result<Option<Value>, String> {
        match self
            .request(Method::GET, &format!("{}/{name}", self.collection), None)
            .await?
        {
            (200, Some(v)) => Ok(Some(v)),
            (404, _) => Ok(None),
            (status, v) => Err(format!("GET {name} answered {status}: {v:?}")),
        }
    }

    /// Every object labelled `selector` (`key=value`).
    ///
    /// # Errors
    ///
    /// The API server could not be asked, or answered otherwise.
    pub async fn list(&self, selector: &str) -> Result<Vec<Value>, String> {
        let url = format!("{}?labelSelector={selector}", self.collection);
        match self.request(Method::GET, &url, None).await? {
            (200, Some(v)) => Ok(v["items"].as_array().cloned().unwrap_or_default()),
            (status, v) => Err(format!("LIST {selector} answered {status}: {v:?}")),
        }
    }

    /// Create `object`; [`Put::Conflict`] when one of its name exists.
    ///
    /// # Errors
    ///
    /// The API server could not be asked, or answered otherwise.
    pub async fn create(&self, object: &Value) -> Result<Put, String> {
        match self
            .request(Method::POST, &self.collection, Some(object))
            .await?
        {
            (201 | 200, Some(v)) => Ok(Put::Written(v)),
            (409, _) => Ok(Put::Conflict),
            (status, v) => Err(format!("POST answered {status}: {v:?}")),
        }
    }

    /// Replace the object `object` names, guarded by the `resourceVersion` it carries.
    ///
    /// # Errors
    ///
    /// The API server could not be asked, or answered otherwise.
    pub async fn replace(&self, object: &Value) -> Result<Put, String> {
        let name = object["metadata"]["name"].as_str().unwrap_or_default();
        match self
            .request(
                Method::PUT,
                &format!("{}/{name}", self.collection),
                Some(object),
            )
            .await?
        {
            (200, Some(v)) => Ok(Put::Written(v)),
            (409, _) => Ok(Put::Conflict),
            (404, _) => Ok(Put::Gone),
            (status, v) => Err(format!("PUT {name} answered {status}: {v:?}")),
        }
    }
}
