// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The coordinator's lease: whoever holds it is the one writer of every engine's epoch
//! (decision 0014), under a term that fences every other (decision 0036). Backends behind one
//! trait: a file lock for one host and the tests, and etcd for a coordinator with a standby; a
//! Kubernetes `Lease` object follows (decision 0017).
//!
//! A lease is held, not owned: the holder renews it, and a holder that cannot renew stops acting
//! by its own clock, before any contender may take the lease as expired. Daemons read its record,
//! read-only, through a [`LeaseWatch`]: the term raises their fence and the address says which
//! coordinator to link to.
//!
//! What every backend meets, checked against Kubernetes' client-go leader election, etcd's
//! election, Kafka's controller epoch and Kleppmann's fencing tokens, is written in 0036.

mod etcd;
mod file;
mod kubernetes;
mod renewal;

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;

use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

pub use etcd::{EtcdLease, EtcdLeaseWatch, EtcdTls};
pub use file::{FileLease, FileLeaseWatch, record_path};
pub use kubernetes::{ADDRESS_ANNOTATION, KubernetesLease, KubernetesLeaseWatch, TERM_ANNOTATION};

use crate::proto::everserve::coord::v1::LeaseRecord;

/// A boxed future, as the backends' trait methods return them.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One lease backend: where contenders try to take the lease.
pub trait Lease: Send + Sync {
    /// Take the lease now if no one holds it: the hold, under a term higher than every earlier
    /// hold's; None if another process holds it.
    ///
    /// # Errors
    ///
    /// The backend failed in a way that is not "someone else holds it".
    fn try_acquire(&self) -> BoxFuture<'_, Result<Option<Box<dyn Hold>>, LeaseError>>;

    /// The backend and what it holds, for logs.
    fn describe(&self) -> String;
}

/// One process's hold of the lease. The backend keeps it: a backend whose lease needs renewal
/// renews it on its own task, and ends the hold, by cancelling [`Hold::ended`], the moment it
/// can no longer be sure no contender may take the lease. A process acts under the hold only
/// while that token is not cancelled. Dropping a hold stops its renewal, and the store lets it
/// lapse; [`Hold::release`] lets it go at once.
pub trait Hold: Send + Sync {
    /// The term this hold runs under: raised at every acquisition, by any holder, and never
    /// reused (decision 0036). A coordinator stamps it on every message to a daemon, which
    /// refuses an older one.
    fn term(&self) -> u64;

    /// Cancelled when the hold ends without being released: its deadline passed by this
    /// process's own monotonic clock, or the store says the lease is no longer this hold's.
    /// Never cancelled for a hold that lasts as long as its process.
    fn ended(&self) -> &CancellationToken;

    /// Publish `record` as this hold's: who holds the lease and where daemons link to it, under
    /// this hold's term (decision 0036).
    ///
    /// # Errors
    ///
    /// [`LeaseError::Lost`] once the hold has ended; otherwise the backend failed to write.
    fn publish(&self, record: LeaseRecord) -> BoxFuture<'_, Result<(), LeaseError>>;

    /// Let the lease go: the store keeps the term, and no one holds the lease until a contender
    /// takes it.
    ///
    /// # Errors
    ///
    /// The backend failed to release it; the store then lets it lapse.
    fn release(self: Box<Self>) -> BoxFuture<'static, Result<(), LeaseError>>;

    /// The backend and the hold, for logs.
    fn describe(&self) -> String;
}

/// Stand by until `lease` is this process's, trying every one to two seconds, drawn afresh so
/// contenders started together do not try in step. A backend that cannot be reached is tried
/// again. None when `cancel` fires first.
pub async fn campaign(lease: &dyn Lease, cancel: &CancellationToken) -> Option<Box<dyn Hold>> {
    let mut standing_by = false;
    loop {
        match lease.try_acquire().await {
            Ok(Some(hold)) => return Some(hold),
            Ok(None) if !standing_by => {
                info!("{} held by another process; standing by", lease.describe());
                standing_by = true;
            }
            Ok(None) => {}
            Err(e) => warn!("{e}; trying again"),
        }
        tokio::select! {
            () = cancel.cancelled() => return None,
            () = tokio::time::sleep(contend_again()) => {}
        }
    }
}

/// One to two seconds, drawn afresh.
fn contend_again() -> Duration {
    use std::hash::{BuildHasher, RandomState};
    let draw = RandomState::new().hash_one(std::time::Instant::now());
    Duration::from_millis(1000 + draw % 1000)
}

/// What a daemon reads of the lease, read-only (decision 0036): its record, if one is published.
pub trait LeaseWatch: Send + Sync {
    /// The record as the backend holds it now; None while no record is published.
    ///
    /// # Errors
    ///
    /// The backend could not be read, or holds something that is not a record.
    fn read(&self) -> BoxFuture<'_, Result<Option<LeaseRecord>, LeaseError>>;

    /// The backend and what it watches, for logs.
    fn describe(&self) -> String;
}

/// Why a lease operation failed.
#[derive(Debug, thiserror::Error)]
pub enum LeaseError {
    /// The lock file could not be opened or locked.
    #[error("lease {path}: {source}")]
    Io {
        /// The lock file.
        path: PathBuf,
        /// The error.
        source: std::io::Error,
    },
    /// The hold has ended: its deadline passed, or another process took the lease.
    #[error("lease {0}: the hold has ended")]
    Lost(String),
    /// The record is not a lease record.
    #[error("lease record {at}: {reason}")]
    Decode {
        /// Where the record was read.
        at: String,
        /// Why.
        reason: String,
    },
    /// The store could not be reached, or refused a request.
    #[error("lease {at}: {reason}")]
    Store {
        /// The store and the key.
        at: String,
        /// Why.
        reason: String,
    },
    /// The lease's spec cannot be read.
    #[error(
        "lease {0:?}: expected a file's path, etcd://host:port[,host:port...]/key or \
         kubernetes://namespace/name"
    )]
    Spec(String),
    /// The lease's store is not one this process was given a way to reach.
    #[error("lease {0}: no {1} access given")]
    Unreachable(String, &'static str),
}

/// Where a lease is kept, as `--lease` and `--lease-watch` give it: a file's path,
/// `etcd://host:port[,host:port...]/key`, or `kubernetes://namespace/name` for a `Lease` object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseSpec {
    /// A lock file, its record beside it.
    File(PathBuf),
    /// A key in etcd, at these endpoints.
    Etcd {
        /// The endpoints, `host:port`.
        endpoints: Vec<String>,
        /// The key, with its leading `/`.
        key: String,
    },
    /// A `coordination.k8s.io/v1` `Lease` object.
    Kubernetes {
        /// Its namespace.
        namespace: String,
        /// Its name.
        name: String,
    },
}

/// How this process reaches a lease's store, beyond what its spec names.
#[derive(Debug, Clone, Default)]
pub struct Reach {
    /// TLS for an etcd cluster that serves it.
    pub etcd_tls: Option<EtcdTls>,
    /// The Kubernetes API.
    pub kubernetes: Option<crate::kubernetes::Access>,
}

impl LeaseSpec {
    /// Read `spec`.
    ///
    /// # Errors
    ///
    /// [`LeaseError::Spec`] for an etcd spec without an endpoint or a key, or a Kubernetes spec
    /// that is not a namespace and a name.
    pub fn parse(spec: &str) -> Result<Self, LeaseError> {
        let bad = || LeaseError::Spec(spec.to_owned());
        if let Some(rest) = spec.strip_prefix("kubernetes://") {
            let (namespace, name) = rest.split_once('/').ok_or_else(bad)?;
            if namespace.is_empty() || name.is_empty() || name.contains('/') {
                return Err(bad());
            }
            return Ok(Self::Kubernetes {
                namespace: namespace.to_owned(),
                name: name.to_owned(),
            });
        }
        let Some(rest) = spec.strip_prefix("etcd://") else {
            return Ok(Self::File(PathBuf::from(spec)));
        };
        let (hosts, key) = rest.split_once('/').ok_or_else(bad)?;
        let endpoints: Vec<String> = hosts
            .split(',')
            .filter(|h| !h.is_empty())
            .map(str::to_owned)
            .collect();
        if endpoints.is_empty() || key.is_empty() {
            return Err(bad());
        }
        Ok(Self::Etcd {
            endpoints,
            key: format!("/{key}"),
        })
    }

    /// The lease this spec names, for a coordinator to contend for as `holder` (unique to this
    /// process); a hold lasts `ttl` past its last renewal by the store's reckoning.
    ///
    /// # Errors
    ///
    /// The store cannot be reached as `reach` says: TLS or CA files that cannot be read,
    /// endpoints that are not valid, or no access for the store given.
    pub async fn lease(
        &self,
        ttl: Duration,
        holder: &str,
        reach: &Reach,
    ) -> Result<Box<dyn Lease>, LeaseError> {
        Ok(match self {
            Self::File(path) => Box::new(FileLease::new(path)),
            Self::Etcd { endpoints, key } => Box::new(
                EtcdLease::connect(endpoints, key.as_str(), ttl, reach.etcd_tls.as_ref()).await?,
            ),
            Self::Kubernetes { namespace, name } => Box::new(KubernetesLease::new(
                self.kubernetes(reach)?,
                namespace,
                name,
                holder,
                ttl,
            )?),
        })
    }

    /// A daemon's watch of the lease this spec names; a read not answered within `timeout`
    /// fails.
    ///
    /// # Errors
    ///
    /// As [`LeaseSpec::lease`].
    pub async fn watch(
        &self,
        timeout: Duration,
        reach: &Reach,
    ) -> Result<std::sync::Arc<dyn LeaseWatch>, LeaseError> {
        Ok(match self {
            Self::File(path) => std::sync::Arc::new(FileLeaseWatch::new(path)),
            Self::Etcd { endpoints, key } => std::sync::Arc::new(
                EtcdLeaseWatch::connect(endpoints, key.as_str(), timeout, reach.etcd_tls.as_ref())
                    .await?,
            ),
            Self::Kubernetes { namespace, name } => std::sync::Arc::new(KubernetesLeaseWatch::new(
                self.kubernetes(reach)?,
                namespace,
                name,
                timeout,
            )?),
        })
    }

    fn kubernetes<'a>(
        &self,
        reach: &'a Reach,
    ) -> Result<&'a crate::kubernetes::Access, LeaseError> {
        reach
            .kubernetes
            .as_ref()
            .ok_or_else(|| LeaseError::Unreachable(format!("{self:?}"), "Kubernetes API"))
    }
}

impl std::str::FromStr for LeaseSpec {
    type Err = LeaseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::LeaseSpec;

    #[test]
    fn a_spec_is_a_file_etcd_endpoints_and_a_key_or_a_kubernetes_lease() {
        assert_eq!(
            LeaseSpec::parse("/var/lib/everserve/coord.lease").unwrap(),
            LeaseSpec::File("/var/lib/everserve/coord.lease".into())
        );
        assert_eq!(
            LeaseSpec::parse("etcd://10.0.0.1:2379,10.0.0.2:2379/everserve/coordinator").unwrap(),
            LeaseSpec::Etcd {
                endpoints: vec!["10.0.0.1:2379".into(), "10.0.0.2:2379".into()],
                key: "/everserve/coordinator".into(),
            }
        );
        assert!(LeaseSpec::parse("etcd://10.0.0.1:2379").is_err(), "no key");
        assert!(
            LeaseSpec::parse("etcd:///everserve").is_err(),
            "no endpoint"
        );
        assert_eq!(
            LeaseSpec::parse("kubernetes://everserve/coordinator").unwrap(),
            LeaseSpec::Kubernetes {
                namespace: "everserve".into(),
                name: "coordinator".into(),
            }
        );
        assert!(
            LeaseSpec::parse("kubernetes://everserve").is_err(),
            "no name"
        );
        assert!(
            LeaseSpec::parse("kubernetes:///coordinator").is_err(),
            "no namespace"
        );
        assert!(
            LeaseSpec::parse("kubernetes://a/b/c").is_err(),
            "a name has no slash"
        );
    }
}
