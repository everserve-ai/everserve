// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The Kubernetes lease: a `coordination.k8s.io/v1` `Lease` object, for a coordinator with a
//! standby in a cluster (decision 0036), reached through the shared API access
//! (`crate::kubernetes`).
//!
//! The API server expires nothing: as client-go's leader election does, a contender judges a
//! lease expired when its record has not changed for its whole duration, by the contender's own
//! monotonic clock from when it first saw that version. The holder renews every quarter of the
//! duration and its hold ends three quarters of it after its last answered renewal was sent
//! (`lease::renewal`), so it has stopped before any contender counts the record as unchanged
//! for long enough.
//!
//! - **The term** is an annotation (`everserve.ai/term`), raised in the same compare-and-swap
//!   write that takes the lease: `leaseTransitions` does not rise when one holder takes the
//!   lease again and starts over when the object is recreated, so it cannot fence.
//! - **Every write** names the version it read (`metadata.resourceVersion`); one the server
//!   refuses as a conflict is read again, and goes ahead only while the object is still this
//!   hold's: the same holder, term and `uid`. A recreated object has another `uid`, so a hold
//!   never continues on it.
//! - **Deleting the `Lease`** must be denied to every role: a recreated object starts its term
//!   over, below what every daemon has seen, and daemons would refuse its holder until an
//!   operator restored the term.
//! - **Letting it go** clears the holder and keeps the term.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::renewal::{self, Renew, Renewed};
use super::{BoxFuture, Hold, Lease, LeaseError, LeaseWatch};
use crate::kubernetes::Access;
use crate::kubernetes::objects::{Objects, Put};
use crate::proto::everserve::coord::v1::LeaseRecord;

/// The annotation that holds the term.
pub const TERM_ANNOTATION: &str = "everserve.ai/term";
/// The annotation that holds the address daemons link to the holder at.
pub const ADDRESS_ANNOTATION: &str = "everserve.ai/address";

/// One `Lease` object on the API server.
struct Api {
    leases: Objects,
    namespace: String,
    name: String,
}

impl Api {
    fn at(&self) -> String {
        format!("kubernetes lease {}/{}", self.namespace, self.name)
    }

    fn failed(&self, reason: impl fmt::Display) -> LeaseError {
        LeaseError::Store {
            at: self.at(),
            reason: reason.to_string(),
        }
    }

    /// The object; None while it does not exist.
    async fn get(&self) -> Result<Option<Value>, LeaseError> {
        self.leases
            .get(&self.name)
            .await
            .map_err(|e| self.failed(e))
    }

    async fn create(&self, object: &Value) -> Result<Put, LeaseError> {
        self.leases.create(object).await.map_err(|e| self.failed(e))
    }

    /// Replace the object, guarded by the `resourceVersion` it carries.
    async fn replace(&self, object: &Value) -> Result<Put, LeaseError> {
        self.leases
            .replace(object)
            .await
            .map_err(|e| self.failed(e))
    }
}

/// The parts of a `Lease` the backend reads.
fn holder(lease: &Value) -> &str {
    lease["spec"]["holderIdentity"].as_str().unwrap_or_default()
}

fn term(lease: &Value) -> u64 {
    lease["metadata"]["annotations"][TERM_ANNOTATION]
        .as_str()
        .and_then(|t| t.parse().ok())
        .unwrap_or(0)
}

fn version(lease: &Value) -> &str {
    lease["metadata"]["resourceVersion"]
        .as_str()
        .unwrap_or_default()
}

fn uid(lease: &Value) -> &str {
    lease["metadata"]["uid"].as_str().unwrap_or_default()
}

fn duration(lease: &Value) -> Duration {
    Duration::from_secs(lease["spec"]["leaseDurationSeconds"].as_u64().unwrap_or(0))
}

/// Now as the API's `MicroTime`.
fn now() -> String {
    let format = time::macros::format_description!(
        "[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:6]Z"
    );
    time::OffsetDateTime::now_utc()
        .format(&format)
        .unwrap_or_default()
}

/// The Kubernetes lease, for a coordinator to contend for.
pub struct KubernetesLease {
    api: Arc<Api>,
    identity: String,
    ttl: Duration,
    /// The version of the record last seen and when, by this process's clock: a held lease is
    /// expired once its version has stood unchanged for its duration since then.
    seen: std::sync::Mutex<Option<(String, Instant)>>,
}

impl fmt::Debug for KubernetesLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KubernetesLease")
            .field("lease", &self.api.at())
            .field("identity", &self.identity)
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

impl KubernetesLease {
    /// The `Lease` `name` in `namespace`, held as `identity` (unique to this process), for
    /// `ttl`, at least two seconds, in whole seconds; requests bounded by `ttl`.
    ///
    /// # Errors
    ///
    /// The API access's CA cannot be read.
    pub fn new(
        access: &Access,
        namespace: &str,
        name: &str,
        identity: impl Into<String>,
        ttl: Duration,
    ) -> Result<Self, LeaseError> {
        let ttl = Duration::from_secs(ttl.as_secs().max(2));
        Ok(Self {
            api: Arc::new(api(access, namespace, name, ttl)?),
            identity: identity.into(),
            ttl,
            seen: std::sync::Mutex::new(None),
        })
    }

    /// Whether `lease` may be taken now: no one holds it, or its version has stood unchanged for
    /// its duration since this process first saw it.
    fn free(&self, lease: &Value) -> bool {
        if holder(lease).is_empty() {
            return true;
        }
        let mut seen = self
            .seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &*seen {
            Some((v, at)) if v == version(lease) => at.elapsed() >= duration(lease),
            _ => {
                *seen = Some((version(lease).to_owned(), Instant::now()));
                false
            }
        }
    }

    /// `lease` taken by this process at the next term, or a new one at term 1.
    fn taken(&self, lease: Option<&Value>) -> Value {
        let now = now();
        let ttl_s = self.ttl.as_secs();
        match lease {
            Some(lease) => {
                let mut next = lease.clone();
                next["spec"]["holderIdentity"] = json!(self.identity);
                next["spec"]["leaseDurationSeconds"] = json!(ttl_s);
                next["spec"]["acquireTime"] = json!(now);
                next["spec"]["renewTime"] = json!(now);
                next["spec"]["leaseTransitions"] =
                    json!(lease["spec"]["leaseTransitions"].as_u64().unwrap_or(0) + 1);
                let annotations = &mut next["metadata"]["annotations"];
                if !annotations.is_object() {
                    *annotations = json!({});
                }
                annotations[TERM_ANNOTATION] = json!((term(lease) + 1).to_string());
                if let Some(a) = annotations.as_object_mut() {
                    a.remove(ADDRESS_ANNOTATION);
                }
                next
            }
            None => json!({
                "apiVersion": "coordination.k8s.io/v1",
                "kind": "Lease",
                "metadata": {
                    "name": self.api.name,
                    "namespace": self.api.namespace,
                    "annotations": { TERM_ANNOTATION: "1" },
                },
                "spec": {
                    "holderIdentity": self.identity,
                    "leaseDurationSeconds": ttl_s,
                    "acquireTime": now,
                    "renewTime": now,
                    "leaseTransitions": 0,
                },
            }),
        }
    }

    async fn acquire(&self) -> Result<Option<Box<dyn Hold>>, LeaseError> {
        let sent = Instant::now();
        let current = self.api.get().await?;
        if current.as_ref().is_some_and(|l| !self.free(l)) {
            return Ok(None);
        }
        let next = self.taken(current.as_ref());
        let put = match &current {
            None => self.api.create(&next).await?,
            Some(_) => self.api.replace(&next).await?,
        };
        let Put::Written(lease) = put else {
            return Ok(None);
        };
        let hold = KubernetesHold {
            api: Arc::clone(&self.api),
            identity: self.identity.clone(),
            term: term(&lease),
            uid: uid(&lease).to_owned(),
            lease: Arc::new(tokio::sync::Mutex::new(lease)),
            ended: CancellationToken::new(),
            renewer: None,
        };
        if Instant::now() >= renewal::deadline(sent, self.ttl) {
            // won so slowly that the hold would already have ended by this process's clock
            let _ = Box::new(hold).release().await;
            return Ok(None);
        }
        Ok(Some(Box::new(hold.renewing(self.ttl, sent))))
    }
}

fn api(access: &Access, namespace: &str, name: &str, timeout: Duration) -> Result<Api, LeaseError> {
    let leases = Objects::new(
        access,
        "coordination.k8s.io/v1",
        "leases",
        namespace,
        timeout,
    )
    .map_err(|reason| LeaseError::Store {
        at: format!("kubernetes lease {namespace}/{name}"),
        reason,
    })?;
    Ok(Api {
        leases,
        namespace: namespace.to_owned(),
        name: name.to_owned(),
    })
}

impl Lease for KubernetesLease {
    fn try_acquire(&self) -> BoxFuture<'_, Result<Option<Box<dyn Hold>>, LeaseError>> {
        Box::pin(self.acquire())
    }

    fn describe(&self) -> String {
        self.api.at()
    }
}

/// A hold of the Kubernetes lease. Its renewals and its record's writes take turns on the object
/// it last wrote, so each names the version the other left.
struct KubernetesHold {
    api: Arc<Api>,
    identity: String,
    term: u64,
    uid: String,
    lease: Arc<tokio::sync::Mutex<Value>>,
    ended: CancellationToken,
    renewer: Option<tokio::task::JoinHandle<()>>,
}

impl KubernetesHold {
    fn renewing(mut self, ttl: Duration, sent: Instant) -> Self {
        self.renewer = Some(renewal::start(
            KubernetesRenewal {
                writer: self.writer(),
                ttl,
            },
            ttl,
            sent,
            ttl,
            self.ended.clone(),
        ));
        self
    }

    fn writer(&self) -> Writer {
        Writer {
            api: Arc::clone(&self.api),
            identity: self.identity.clone(),
            term: self.term,
            uid: self.uid.clone(),
            lease: Arc::clone(&self.lease),
        }
    }
}

impl fmt::Debug for KubernetesHold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KubernetesHold")
            .field("lease", &self.api.at())
            .field("term", &self.term)
            .field("ended", &self.ended.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Drop for KubernetesHold {
    fn drop(&mut self) {
        // a hold no one keeps is not renewed; contenders take it once it stands unchanged
        if let Some(r) = &self.renewer {
            r.abort();
        }
    }
}

/// The hold's writes to its object, each guarded by the version last written.
#[derive(Clone)]
struct Writer {
    api: Arc<Api>,
    identity: String,
    term: u64,
    uid: String,
    lease: Arc<tokio::sync::Mutex<Value>>,
}

impl Writer {
    /// Whether `lease` is still this hold's.
    fn ours(&self, lease: &Value) -> bool {
        holder(lease) == self.identity && term(lease) == self.term && uid(lease) == self.uid
    }

    /// Write `change` applied to the object. A conflict reads the object again and writes once
    /// more while it is still this hold's. Ok(false) when it is not, or no longer exists.
    async fn write(&self, change: impl Fn(&mut Value)) -> Result<bool, LeaseError> {
        let mut lease = self.lease.lock().await;
        for _ in 0..2 {
            let mut next = lease.clone();
            change(&mut next);
            match self.api.replace(&next).await? {
                Put::Written(written) => {
                    *lease = written;
                    return Ok(true);
                }
                Put::Gone => return Ok(false),
                Put::Conflict => match self.api.get().await? {
                    Some(current) if self.ours(&current) => *lease = current,
                    _ => return Ok(false),
                },
            }
        }
        Err(self.api.failed("the object kept changing under the hold"))
    }
}

/// The Kubernetes hold's renewal step: its `renewTime`, written over the version last written.
struct KubernetesRenewal {
    writer: Writer,
    ttl: Duration,
}

impl Renew for KubernetesRenewal {
    fn renew(&mut self) -> BoxFuture<'_, Renewed> {
        Box::pin(async move {
            let renewed = self
                .writer
                .write(|l| l["spec"]["renewTime"] = json!(now()))
                .await;
            match renewed {
                Ok(true) => Renewed::Until(self.ttl),
                Ok(false) => Renewed::Ended,
                Err(e) => Renewed::Failed(e.to_string()),
            }
        })
    }

    fn describe(&self) -> String {
        format!("{} at term {}", self.writer.api.at(), self.writer.term)
    }
}

impl Hold for KubernetesHold {
    fn term(&self) -> u64 {
        self.term
    }

    fn ended(&self) -> &CancellationToken {
        &self.ended
    }

    fn publish(&self, record: LeaseRecord) -> BoxFuture<'_, Result<(), LeaseError>> {
        Box::pin(async move {
            if self.ended.is_cancelled() {
                return Err(LeaseError::Lost(self.api.at()));
            }
            let address = record.address;
            let written = self
                .writer()
                .write(|l| {
                    if !l["metadata"]["annotations"].is_object() {
                        l["metadata"]["annotations"] = json!({});
                    }
                    l["metadata"]["annotations"][ADDRESS_ANNOTATION] = json!(address);
                })
                .await?;
            if written {
                Ok(())
            } else {
                // the store's word: the object is no longer this hold's
                self.ended.cancel();
                Err(LeaseError::Lost(self.api.at()))
            }
        })
    }

    fn release(mut self: Box<Self>) -> BoxFuture<'static, Result<(), LeaseError>> {
        if let Some(r) = self.renewer.take() {
            r.abort();
        }
        let writer = self.writer();
        Box::pin(async move {
            // no holder, the term kept
            writer
                .write(|l| {
                    l["spec"]["holderIdentity"] = json!("");
                    if let Some(a) = l["metadata"]["annotations"].as_object_mut() {
                        a.remove(ADDRESS_ANNOTATION);
                    }
                })
                .await
                .map(drop)
        })
    }

    fn describe(&self) -> String {
        format!("{} at term {}", self.api.at(), self.term)
    }
}

/// A daemon's read-only watch of the Kubernetes lease's record.
pub struct KubernetesLeaseWatch {
    api: Api,
}

impl fmt::Debug for KubernetesLeaseWatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KubernetesLeaseWatch")
            .field("lease", &self.api.at())
            .finish_non_exhaustive()
    }
}

impl KubernetesLeaseWatch {
    /// A watch of the `Lease` `name` in `namespace`; a read not answered within `timeout` fails.
    ///
    /// # Errors
    ///
    /// The API access's CA cannot be read.
    pub fn new(
        access: &Access,
        namespace: &str,
        name: &str,
        timeout: Duration,
    ) -> Result<Self, LeaseError> {
        Ok(Self {
            api: api(access, namespace, name, timeout)?,
        })
    }
}

impl LeaseWatch for KubernetesLeaseWatch {
    fn read(&self) -> BoxFuture<'_, Result<Option<LeaseRecord>, LeaseError>> {
        Box::pin(async move {
            Ok(self.api.get().await?.map(|lease| LeaseRecord {
                holder: holder(&lease).to_owned(),
                term: term(&lease),
                address: lease["metadata"]["annotations"][ADDRESS_ANNOTATION]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                ..LeaseRecord::default()
            }))
        })
    }

    fn describe(&self) -> String {
        format!("{} record", self.api.at())
    }
}
