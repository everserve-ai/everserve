// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The etcd lease: a key that exists only while its holder renews an etcd lease, for a
//! coordinator with a standby on another host (decision 0036).
//!
//! - **Taking it.** A contender grants itself an etcd lease, then puts the key in a transaction
//!   that succeeds only if the key does not exist. The term is that transaction's revision, which
//!   is the key's create revision: etcd raises it at every write, so every hold's term is higher
//!   than every earlier hold's, whoever held it. A contender that loses revokes its etcd lease.
//! - **Holding it.** Every write of the record is guarded by "the key's create revision is still
//!   this term", so a holder that lost the key cannot write over its successor's record.
//! - **Renewing it.** The hold renews itself on its own task: a keep-alive every quarter of the
//!   TTL, then a check that the key's create revision is still the term, since a key deleted
//!   under a live etcd lease is free for a contender. The hold ends, cancelling
//!   [`Hold::ended`], three quarters of the TTL after it sent its last answered renewal, by its
//!   own monotonic clock, whether or not a renewal is still in flight. etcd expires the key no
//!   earlier than a full TTL after it received that renewal, which is later, so the hold has
//!   ended before a contender can take the key.
//! - **Letting it go.** Revoking the etcd lease deletes the key. A daemon then reads no holder,
//!   and keeps the term it has (it never lowers its fence).
//!
//! The record's `term` field is not trusted from the value: a reader takes the key's create
//! revision, which only etcd writes.

use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use etcd_client::{
    Certificate, Client, Compare, CompareOp, ConnectOptions, Identity, LeaseKeepAliveStream,
    LeaseKeeper, PutOptions, TlsOptions, Txn, TxnOp,
};
use prost::Message as _;

use super::renewal::{self, Renew, Renewed};
use super::{BoxFuture, Hold, Lease, LeaseError, LeaseWatch};
use crate::proto::everserve::coord::v1::LeaseRecord;
use tokio_util::sync::CancellationToken;

/// The client certificates for an etcd cluster that serves TLS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EtcdTls {
    /// The CA that signed the cluster's server certificates, PEM.
    pub ca: PathBuf,
    /// This client's certificate and its key, PEM, for a cluster that requires client
    /// certificates.
    pub identity: Option<(PathBuf, PathBuf)>,
}

impl EtcdTls {
    fn options(&self) -> Result<TlsOptions, LeaseError> {
        let read = |path: &PathBuf| {
            std::fs::read(path).map_err(|source| LeaseError::Io {
                path: path.clone(),
                source,
            })
        };
        let mut tls = TlsOptions::new().ca_certificate(Certificate::from_pem(read(&self.ca)?));
        if let Some((cert, key)) = &self.identity {
            tls = tls.identity(Identity::from_pem(read(cert)?, read(key)?));
        }
        Ok(tls)
    }
}

/// Connect to the cluster at `endpoints`; a request not answered within `timeout` fails. Only
/// a member that sees a leader answers, so a member cut off from the quorum cannot answer from
/// stale state.
async fn connect(
    endpoints: &[String],
    tls: Option<&EtcdTls>,
    timeout: Duration,
) -> Result<Client, LeaseError> {
    let mut options = ConnectOptions::new()
        .with_connect_timeout(timeout)
        .with_timeout(timeout)
        .with_require_leader(true);
    if let Some(tls) = tls {
        options = options.with_tls(tls.options()?);
    }
    Client::connect(endpoints, Some(options))
        .await
        .map_err(|e| LeaseError::Store {
            at: format!("etcd {}", endpoints.join(",")),
            reason: e.to_string(),
        })
}

/// The etcd lease, for a coordinator to contend for.
#[derive(Clone)]
pub struct EtcdLease {
    client: Client,
    key: String,
    ttl: Duration,
}

impl fmt::Debug for EtcdLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EtcdLease")
            .field("key", &self.key)
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

impl EtcdLease {
    /// The lease at `key` on the cluster at `endpoints`. A hold lasts `ttl` past its last renewal
    /// on the cluster's side, at least two seconds; the holder's own deadline ends it sooner.
    ///
    /// # Errors
    ///
    /// The TLS files cannot be read, or the endpoints are not valid.
    pub async fn connect(
        endpoints: &[String],
        key: impl Into<String>,
        ttl: Duration,
        tls: Option<&EtcdTls>,
    ) -> Result<Self, LeaseError> {
        let ttl = ttl.max(Duration::from_secs(2));
        Ok(Self {
            // a request's timeout bounds how long one call may take, not how long a hold acts:
            // the hold's own deadline does that
            client: connect(endpoints, tls, ttl).await?,
            key: key.into(),
            ttl,
        })
    }

    async fn acquire(&self) -> Result<Option<Box<dyn Hold>>, LeaseError> {
        let mut client = self.client.clone();
        let store = |e: etcd_client::Error| store_error(&self.key, &e);
        let ttl_s = i64::try_from(self.ttl.as_secs()).unwrap_or(i64::MAX);
        let sent = Instant::now();
        let grant = client.lease_grant(ttl_s, None).await.map_err(store)?;
        let id = grant.id();
        let take = Txn::new()
            .when([Compare::create_revision(
                self.key.as_str(),
                CompareOp::Equal,
                0,
            )])
            .and_then([TxnOp::put(
                self.key.as_str(),
                LeaseRecord::default().encode_to_vec(),
                Some(PutOptions::new().with_lease(id)),
            )]);
        let taken = match client.txn(take).await {
            Ok(r) if r.succeeded() => r.header().map(etcd_client::ResponseHeader::revision),
            Ok(_) => None,
            Err(e) => {
                // it may have been taken all the same; revoking the etcd lease deletes the key
                let _ = client.lease_revoke(id).await;
                return Err(store(e));
            }
        };
        let granted = seconds(grant.ttl());
        let Some(revision) = taken.filter(|_| Instant::now() < renewal::deadline(sent, granted))
        else {
            // lost to another contender, or won so slowly that the hold would already have
            // ended by this process's clock: either way, not one to act under
            let _ = client.lease_revoke(id).await;
            return Ok(None);
        };
        let term = u64::try_from(revision).unwrap_or(0);
        let ended = CancellationToken::new();
        let renewer = renewal::start(
            EtcdRenewal {
                client: client.clone(),
                key: self.key.clone(),
                id,
                term,
                stream: None,
            },
            self.ttl,
            sent,
            granted,
            ended.clone(),
        );
        Ok(Some(Box::new(EtcdHold {
            client,
            key: self.key.clone(),
            id,
            term,
            ended,
            renewer,
        })))
    }
}

impl Lease for EtcdLease {
    fn try_acquire(&self) -> BoxFuture<'_, Result<Option<Box<dyn Hold>>, LeaseError>> {
        Box::pin(self.acquire())
    }

    fn describe(&self) -> String {
        format!("etcd lease {}", self.key)
    }
}

fn store_error(key: &str, e: &etcd_client::Error) -> LeaseError {
    LeaseError::Store {
        at: format!("etcd {key}"),
        reason: e.to_string(),
    }
}

/// Seconds as etcd gives a TTL, as a duration; none for a negative one.
fn seconds(ttl_s: i64) -> Duration {
    Duration::from_secs(u64::try_from(ttl_s).unwrap_or(0))
}

/// A hold of the etcd lease. Its renewal runs on its own task until the hold ends, is released
/// or is dropped.
struct EtcdHold {
    client: Client,
    key: String,
    id: i64,
    term: u64,
    ended: CancellationToken,
    renewer: tokio::task::JoinHandle<()>,
}

impl fmt::Debug for EtcdHold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EtcdHold")
            .field("key", &self.key)
            .field("term", &self.term)
            .field("ended", &self.ended.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Drop for EtcdHold {
    fn drop(&mut self) {
        // a hold no one keeps is not renewed; the store lets it lapse
        self.renewer.abort();
    }
}

impl Hold for EtcdHold {
    fn term(&self) -> u64 {
        self.term
    }

    fn ended(&self) -> &CancellationToken {
        &self.ended
    }

    fn publish(&self, mut record: LeaseRecord) -> BoxFuture<'_, Result<(), LeaseError>> {
        Box::pin(async move {
            if self.ended.is_cancelled() {
                return Err(LeaseError::Lost(format!("etcd {}", self.key)));
            }
            record.term = self.term;
            // the write lands only while the key is this hold's, so a hold that has lost the
            // key cannot write over its successor's record
            let guarded = Txn::new()
                .when([Compare::create_revision(
                    self.key.as_str(),
                    CompareOp::Equal,
                    i64::try_from(self.term).unwrap_or(i64::MAX),
                )])
                .and_then([TxnOp::put(
                    self.key.as_str(),
                    record.encode_to_vec(),
                    Some(PutOptions::new().with_lease(self.id)),
                )]);
            match self.client.clone().txn(guarded).await {
                Ok(r) if r.succeeded() => Ok(()),
                Ok(_) => {
                    // the store's word: the key is another hold's
                    self.ended.cancel();
                    Err(LeaseError::Lost(format!("etcd {}", self.key)))
                }
                Err(e) => Err(store_error(&self.key, &e)),
            }
        })
    }

    fn release(self: Box<Self>) -> BoxFuture<'static, Result<(), LeaseError>> {
        // the renewal stops when the hold drops, at the end of this call
        let (mut client, key, id) = (self.client.clone(), self.key.clone(), self.id);
        drop(self);
        Box::pin(async move {
            // revoking the etcd lease deletes the key: no one holds the lease, and the next
            // hold's create revision is higher than this one's
            client
                .lease_revoke(id)
                .await
                .map(drop)
                .map_err(|e| store_error(&key, &e))
        })
    }

    fn describe(&self) -> String {
        format!("etcd lease {} at term {}", self.key, self.term)
    }
}

/// The etcd hold's renewal step: a keep-alive, then a check that the key is still the hold's,
/// since a key deleted under a live etcd lease is free for a contender to take.
struct EtcdRenewal {
    client: Client,
    key: String,
    id: i64,
    term: u64,
    /// None after a renewal failed, so a late answer on that stream is never taken for the
    /// answer to the next.
    stream: Option<(LeaseKeeper, LeaseKeepAliveStream)>,
}

impl EtcdRenewal {
    async fn once(&mut self) -> Result<Renewed, etcd_client::Error> {
        if self.stream.is_none() {
            self.stream = Some(self.client.lease_keep_alive(self.id).await?);
        }
        let (keeper, answers) = self.stream.as_mut().expect("opened above");
        keeper.keep_alive().await?;
        let Some(alive) = answers.message().await? else {
            self.stream = None;
            return Ok(Renewed::Failed("the keep-alive stream closed".into()));
        };
        if alive.ttl() <= 0 {
            // etcd's word that the lease has expired
            return Ok(Renewed::Ended);
        }
        let check = Txn::new().when([Compare::create_revision(
            self.key.as_str(),
            CompareOp::Equal,
            i64::try_from(self.term).unwrap_or(i64::MAX),
        )]);
        Ok(if self.client.txn(check).await?.succeeded() {
            Renewed::Until(seconds(alive.ttl()))
        } else {
            Renewed::Ended
        })
    }
}

impl Renew for EtcdRenewal {
    fn renew(&mut self) -> BoxFuture<'_, Renewed> {
        Box::pin(async move {
            self.once().await.unwrap_or_else(|e| {
                self.stream = None;
                Renewed::Failed(e.to_string())
            })
        })
    }

    fn describe(&self) -> String {
        format!("etcd {} at term {}", self.key, self.term)
    }
}

/// A daemon's read-only watch of the etcd lease's record.
pub struct EtcdLeaseWatch {
    client: Client,
    key: String,
}

impl fmt::Debug for EtcdLeaseWatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EtcdLeaseWatch")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl EtcdLeaseWatch {
    /// A watch of `key` on the cluster at `endpoints`; a read not answered within `timeout`
    /// fails.
    ///
    /// # Errors
    ///
    /// The TLS files cannot be read, or the endpoints are not valid.
    pub async fn connect(
        endpoints: &[String],
        key: impl Into<String>,
        timeout: Duration,
        tls: Option<&EtcdTls>,
    ) -> Result<Self, LeaseError> {
        Ok(Self {
            client: connect(endpoints, tls, timeout).await?,
            key: key.into(),
        })
    }
}

impl LeaseWatch for EtcdLeaseWatch {
    fn read(&self) -> BoxFuture<'_, Result<Option<LeaseRecord>, LeaseError>> {
        Box::pin(async move {
            let at = || format!("etcd {}", self.key);
            let mut client = self.client.clone();
            let got = client
                .get(self.key.as_str(), None)
                .await
                .map_err(|e| LeaseError::Store {
                    at: at(),
                    reason: e.to_string(),
                })?;
            let Some(kv) = got.kvs().first() else {
                return Ok(None);
            };
            let mut record = LeaseRecord::decode(kv.value()).map_err(|e| LeaseError::Decode {
                at: at(),
                reason: e.to_string(),
            })?;
            record.term = u64::try_from(kv.create_revision()).unwrap_or(0);
            Ok(Some(record))
        })
    }

    fn describe(&self) -> String {
        format!("etcd lease record {}", self.key)
    }
}
