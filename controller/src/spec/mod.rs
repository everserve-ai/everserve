// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Desired state and where it lives (decision 0036). A resource's spec is what it should be,
//! written only through the operator API; the controllers reconcile its observed status toward
//! it. The nodes are the source of observed state, and only what the control plane alone must
//! keep across a restart is kept here beside the specs (decision 0038): each engine's change in
//! flight, its assignments and its highest cover plan generation. The coordinator holding the
//! lease writes them, and no operator does.
//!
//! One contract, several stores: in memory (a coordinator that keeps no desired state across a
//! restart), a file (one coordinator, written atomically), and Kubernetes custom resources for a
//! coordinator with a standby. Everserve never implements consensus. A coordinator that starts
//! or takes the lease takes the store ([`SpecStore::take`]): it writes its term into everything
//! the store holds and reads it back in the same step, and that is what it acts on; a reconcile
//! reads the coordinator's copy in memory, never the store.
//!
//! Every write names the lease term its coordinator acts under, and a store refuses a write under
//! a term older than the newest that wrote to it (decision 0036), as Kafka fences a deposed
//! controller's metadata writes: the daemons refuse a deposed coordinator's messages, and the
//! store refuses its writes. A write that took effect before a successor took the store is in
//! what the successor read; one after it is refused.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use prost::Message as _;

use crate::lease::BoxFuture;

pub mod kubernetes;

use crate::proto::everserve::spec::v1::{ResourceSpec, ResourceStatus, SpecSnapshot};

/// Why a spec could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    /// Another coordinator wrote what this one's write was to replace, since this one took the
    /// store: a newer holder of the lease took it. Nothing this coordinator writes lands again.
    #[error("{key}: written since this coordinator took the store: another holds the lease")]
    Superseded {
        /// The resource.
        key: String,
    },
    /// The store answered with an error; the write did not land.
    #[error("{key}: the spec store refused: {reason}")]
    Refused {
        /// The resource.
        key: String,
        /// What the store said.
        reason: String,
    },
    /// Another write landed since the generation this one read: read again and decide.
    #[error("{key}: the write read generation {expected}, the store holds {actual}")]
    Conflict {
        /// The resource.
        key: String,
        /// The generation the writer read.
        expected: u64,
        /// The generation the store holds.
        actual: u64,
    },
    /// The write's lease term is older than the newest that wrote to the store: its coordinator
    /// was deposed, and another holds the lease.
    #[error("{key}: written under lease term {term}, the store was written under term {newer}")]
    Fenced {
        /// The resource.
        key: String,
        /// The writer's term.
        term: u64,
        /// The newest term that wrote to the store.
        newer: u64,
    },
    /// The store did not answer within the coordinator's deadline: the coordinator stops acting,
    /// whether the write landed or not.
    #[error("{key}: the spec store did not answer within {after:?}")]
    TimedOut {
        /// The resource.
        key: String,
        /// The deadline.
        after: std::time::Duration,
    },
    /// This coordinator no longer holds the lease (decision 0036): it writes nothing.
    #[error("{key}: this coordinator no longer holds the lease; write through its holder")]
    Deposed {
        /// The resource.
        key: String,
    },
    /// The store's file could not be read or written.
    #[error("spec store {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The error.
        source: std::io::Error,
    },
    /// The store's file is not a spec snapshot.
    #[error("spec store {path}: {reason}")]
    Decode {
        /// The file.
        path: PathBuf,
        /// Why.
        reason: String,
    },
}

impl SpecError {
    /// Whether trying the write again cannot succeed: a newer term wrote to the store or took
    /// it, this coordinator is deposed, or it stopped acting when the store did not answer in
    /// time.
    #[must_use]
    pub fn is_fenced(&self) -> bool {
        matches!(
            self,
            Self::Fenced { .. }
                | Self::Superseded { .. }
                | Self::Deposed { .. }
                | Self::TimedOut { .. }
        )
    }
}

/// Where a coordinator keeps its specs: a file, or a Kubernetes namespace's `EverserveState`
/// objects named for the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreAt {
    /// A file, written whole.
    File(PathBuf),
    /// `kubernetes://namespace/name`.
    Kubernetes {
        /// The namespace.
        namespace: String,
        /// The store's name: its root object's, and its objects' label.
        name: String,
    },
}

impl std::str::FromStr for StoreAt {
    type Err = String;

    fn from_str(spec: &str) -> Result<Self, String> {
        let Some(rest) = spec.strip_prefix("kubernetes://") else {
            return Ok(Self::File(PathBuf::from(spec)));
        };
        match rest.split_once('/') {
            Some((namespace, name))
                if !namespace.is_empty() && !name.is_empty() && !name.contains('/') =>
            {
                Ok(Self::Kubernetes {
                    namespace: namespace.to_owned(),
                    name: name.to_owned(),
                })
            }
            _ => Err(format!("{spec}: not kubernetes://namespace/name")),
        }
    }
}

/// One spec store.
pub trait SpecStore: Send + Sync {
    /// Take the store under lease term `term`: write the term into everything the store holds,
    /// and return what it holds, read in the same step. Every write before it is in what it
    /// returns; every write under an older term after it is refused. A coordinator takes the
    /// store once, before it acts.
    ///
    /// # Errors
    ///
    /// [`SpecError::Fenced`] when a newer term took or wrote to the store; otherwise the store
    /// could not be read or written.
    fn take(&self, term: u64) -> BoxFuture<'_, Result<SpecSnapshot, SpecError>>;

    /// Write `spec` as `key`'s under lease term `term`, if the store still holds generation
    /// `read` for it (0 for a resource it does not hold) and no newer term wrote to it. Returns
    /// the spec as stored, its generation raised by one.
    ///
    /// # Errors
    ///
    /// [`SpecError::Fenced`] when a newer term wrote to the store; [`SpecError::Conflict`] when
    /// another write landed since `read`; otherwise the store failed.
    fn put(
        &self,
        key: String,
        spec: ResourceSpec,
        read: u64,
        term: u64,
    ) -> BoxFuture<'_, Result<ResourceSpec, SpecError>>;

    /// Keep `status` as `key`'s under lease term `term`, if no newer term wrote to the store.
    ///
    /// # Errors
    ///
    /// [`SpecError::Fenced`] when a newer term wrote to the store; otherwise the store failed.
    fn put_status(
        &self,
        key: String,
        status: ResourceStatus,
        term: u64,
    ) -> BoxFuture<'_, Result<(), SpecError>>;

    /// The store and where it keeps specs, for logs.
    fn describe(&self) -> String;
}

/// The key a node's spec is kept under.
#[must_use]
pub fn node_key(node: &str) -> String {
    format!("node/{node}")
}

/// The key a pool's spec is kept under.
#[must_use]
pub fn pool_key(pool: &str) -> String {
    format!("pool/{pool}")
}

/// The key an engine's status is kept under.
#[must_use]
pub fn engine_key(engine: &str) -> String {
    format!("engine/{engine}")
}

/// Refuse a write under `term` to `snapshot` when a newer term wrote to it: the fence every
/// write passes, a spec's or a status's.
fn fence(snapshot: &SpecSnapshot, key: &str, term: u64) -> Result<(), SpecError> {
    if term < snapshot.term {
        return Err(SpecError::Fenced {
            key: key.to_owned(),
            term,
            newer: snapshot.term,
        });
    }
    Ok(())
}

/// Raise `spec`'s generation over what `snapshot` holds for `key`, if that is `read` and no term
/// newer than `term` wrote to it, and keep it there with `term`. The one rule every store applies.
fn apply(
    snapshot: &mut SpecSnapshot,
    key: &str,
    mut spec: ResourceSpec,
    read: u64,
    term: u64,
) -> Result<ResourceSpec, SpecError> {
    fence(snapshot, key, term)?;
    let actual = snapshot.resources.get(key).map_or(0, |s| s.generation);
    if actual != read {
        return Err(SpecError::Conflict {
            key: key.to_owned(),
            expected: read,
            actual,
        });
    }
    spec.generation = actual + 1;
    snapshot.term = term;
    snapshot.resources.insert(key.to_owned(), spec.clone());
    Ok(spec)
}

/// Take `snapshot` under `term`, if no newer term wrote to it: the term is its newest.
fn apply_take(snapshot: &mut SpecSnapshot, term: u64) -> Result<SpecSnapshot, SpecError> {
    fence(snapshot, "the store", term)?;
    snapshot.term = term;
    Ok(snapshot.clone())
}

/// Keep `status` as `key`'s in `snapshot` under `term`, if no newer term wrote to it.
fn apply_status(
    snapshot: &mut SpecSnapshot,
    key: &str,
    status: ResourceStatus,
    term: u64,
) -> Result<(), SpecError> {
    fence(snapshot, key, term)?;
    snapshot.term = term;
    snapshot.statuses.insert(key.to_owned(), status);
    Ok(())
}

/// Specs in memory: nothing survives the coordinator's restart.
#[derive(Debug, Default)]
pub struct MemoryStore {
    snapshot: Mutex<SpecSnapshot>,
}

impl MemoryStore {
    fn locked(&self) -> std::sync::MutexGuard<'_, SpecSnapshot> {
        self.snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl SpecStore for MemoryStore {
    fn take(&self, term: u64) -> BoxFuture<'_, Result<SpecSnapshot, SpecError>> {
        Box::pin(async move { apply_take(&mut self.locked(), term) })
    }

    fn put(
        &self,
        key: String,
        spec: ResourceSpec,
        read: u64,
        term: u64,
    ) -> BoxFuture<'_, Result<ResourceSpec, SpecError>> {
        Box::pin(async move { apply(&mut self.locked(), &key, spec, read, term) })
    }

    fn put_status(
        &self,
        key: String,
        status: ResourceStatus,
        term: u64,
    ) -> BoxFuture<'_, Result<(), SpecError>> {
        Box::pin(async move { apply_status(&mut self.locked(), &key, status, term) })
    }

    fn describe(&self) -> String {
        "in memory (desired state is not kept across a restart)".to_owned()
    }
}

/// Specs in one file, written whole and atomically: a temporary file beside it, synced, renamed
/// over it, and the directory synced, so a crash leaves the old snapshot or the new one, never a
/// part of either. Each write is a read, a check and a whole rewrite under an exclusive lock on
/// `<file>.lock`, so the checks hold across processes: a standby's coordinator on a file system
/// both share is fenced as one in this process is.
#[derive(Debug)]
pub struct FileStore {
    path: PathBuf,
}

impl FileStore {
    /// A store at `path`, created on its first write.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// What the file holds now, read without taking it: for a look at the store, never for a
    /// coordinator to act on.
    ///
    /// # Errors
    ///
    /// The file could not be read or decoded.
    pub fn read(&self) -> Result<SpecSnapshot, SpecError> {
        match std::fs::read(&self.path) {
            Ok(bytes) => SpecSnapshot::decode(bytes.as_slice()).map_err(|e| SpecError::Decode {
                path: self.path.clone(),
                reason: e.to_string(),
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(SpecSnapshot::default()),
            Err(source) => Err(SpecError::Io {
                path: self.path.clone(),
                source,
            }),
        }
    }

    /// Read the snapshot, change it with `change` and write it whole, all under the exclusive
    /// lock on `<file>.lock`; nothing is written when `change` refuses.
    fn update<T>(
        &self,
        change: impl FnOnce(&mut SpecSnapshot) -> Result<T, SpecError>,
    ) -> Result<T, SpecError> {
        let lock_path = self.path.with_extension("lock");
        let io = |source| SpecError::Io {
            path: lock_path.clone(),
            source,
        };
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(io)?;
        // held to the end of this call; closing the file lets it go
        lock.lock().map_err(io)?;
        let mut snapshot = self.read()?;
        let changed = change(&mut snapshot)?;
        self.write(&snapshot)?;
        Ok(changed)
    }

    fn write(&self, snapshot: &SpecSnapshot) -> Result<(), SpecError> {
        let io = |source| SpecError::Io {
            path: self.path.clone(),
            source,
        };
        let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
        let tmp = self.path.with_extension("tmp");
        let mut f = std::fs::File::create(&tmp).map_err(io)?;
        f.write_all(&snapshot.encode_to_vec()).map_err(io)?;
        f.sync_all().map_err(io)?;
        std::fs::rename(&tmp, &self.path).map_err(io)?;
        std::fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(io)
    }
}

impl FileStore {
    /// [`FileStore::update`] off the runtime's threads: the file is read, synced and renamed.
    fn updated<T: Send + 'static>(
        &self,
        change: impl FnOnce(&mut SpecSnapshot) -> Result<T, SpecError> + Send + 'static,
    ) -> BoxFuture<'_, Result<T, SpecError>> {
        let store = Self {
            path: self.path.clone(),
        };
        Box::pin(async move {
            tokio::task::spawn_blocking(move || store.update(change))
                .await
                .map_err(|e| SpecError::Io {
                    path: self.path.clone(),
                    source: std::io::Error::other(e),
                })?
        })
    }
}

impl SpecStore for FileStore {
    fn take(&self, term: u64) -> BoxFuture<'_, Result<SpecSnapshot, SpecError>> {
        self.updated(move |snapshot| apply_take(snapshot, term))
    }

    fn put(
        &self,
        key: String,
        spec: ResourceSpec,
        read: u64,
        term: u64,
    ) -> BoxFuture<'_, Result<ResourceSpec, SpecError>> {
        self.updated(move |snapshot| apply(snapshot, &key, spec, read, term))
    }

    fn put_status(
        &self,
        key: String,
        status: ResourceStatus,
        term: u64,
    ) -> BoxFuture<'_, Result<(), SpecError>> {
        self.updated(move |snapshot| apply_status(snapshot, &key, status, term))
    }

    fn describe(&self) -> String {
        format!("file {}", self.path.display())
    }
}

#[cfg(test)]
mod tests {
    use super::{FileStore, MemoryStore, SpecError, SpecStore, engine_key, node_key};
    use crate::proto::everserve::spec::v1::{
        EngineStatus, NodeSpec, ResourceSpec, ResourceStatus, resource_spec, resource_status,
    };

    fn cordoned(node: &str) -> ResourceSpec {
        ResourceSpec {
            kind: Some(resource_spec::Kind::Node(NodeSpec {
                node: node.into(),
                unschedulable: true,
                ..NodeSpec::default()
            })),
            ..ResourceSpec::default()
        }
    }

    async fn each_store_keeps_generations_and_refuses_a_stale_write(store: &dyn SpecStore) {
        let key = node_key("node-a");
        let first = store
            .put(key.clone(), cordoned("node-a"), 0, 1)
            .await
            .expect("create");
        assert_eq!(first.generation, 1);
        assert!(matches!(
            store.put(key.clone(), cordoned("node-a"), 0, 1).await,
            Err(SpecError::Conflict {
                expected: 0,
                actual: 1,
                ..
            })
        ));
        let second = store
            .put(key.clone(), cordoned("node-a"), 1, 1)
            .await
            .expect("update");
        assert_eq!(second.generation, 2);
        assert_eq!(
            store.take(1).await.expect("take").resources[&key].generation,
            2,
            "what was written is what is read"
        );
    }

    fn engine_status(generation: u64) -> ResourceStatus {
        ResourceStatus {
            kind: Some(resource_status::Kind::Engine(EngineStatus {
                engine: "llama".into(),
                plan_generation: generation,
                ..EngineStatus::default()
            })),
        }
    }

    async fn each_store_keeps_a_status_and_fences_it_with_the_specs(store: &dyn SpecStore) {
        // decision 0038: an engine's status is the coordinator's, written under its term
        let key = engine_key("llama");
        store
            .put_status(key.clone(), engine_status(1), 2)
            .await
            .expect("term 2");
        store
            .put_status(key.clone(), engine_status(2), 2)
            .await
            .expect("again");
        assert_eq!(
            store.take(2).await.expect("take").statuses[&key],
            engine_status(2)
        );
        assert!(
            matches!(
                store.put_status(key.clone(), engine_status(3), 1).await,
                Err(SpecError::Fenced {
                    term: 1,
                    newer: 2,
                    ..
                })
            ),
            "a deposed coordinator's status is refused"
        );
        store
            .put(node_key("node-a"), cordoned("node-a"), 0, 5)
            .await
            .expect("term 5");
        assert!(
            matches!(
                store.put_status(key.clone(), engine_status(3), 2).await,
                Err(SpecError::Fenced { newer: 5, .. })
            ),
            "a spec's term fences a status: one store, one term"
        );
        assert_eq!(
            store.take(5).await.expect("take").statuses[&key],
            engine_status(2)
        );
    }

    async fn each_store_refuses_a_write_under_an_older_term(store: &dyn SpecStore) {
        // decision 0036: a coordinator deposed without noticing writes nothing
        let key = node_key("node-b");
        store
            .put(key.clone(), cordoned("node-b"), 0, 3)
            .await
            .expect("term 3");
        assert!(
            matches!(
                store.put(key.clone(), cordoned("node-b"), 1, 2).await,
                Err(SpecError::Fenced {
                    term: 2,
                    newer: 3,
                    ..
                })
            ),
            "a write under an older term is refused, whatever generation it read"
        );
        assert!(
            matches!(
                store
                    .put(node_key("node-c"), cordoned("node-c"), 0, 2)
                    .await,
                Err(SpecError::Fenced { .. })
            ),
            "the term is the store's, not the resource's"
        );
        let next = store
            .put(key.clone(), cordoned("node-b"), 1, 4)
            .await
            .expect("term 4");
        assert_eq!(next.generation, 2);
        assert_eq!(store.take(4).await.expect("take").term, 4);
    }

    async fn each_store_is_taken_by_a_newer_term_alone(store: &dyn SpecStore) {
        // decision 0036: taking the store writes the term into it, so the deposed holder's next
        // write is refused, and what it wrote before is what the taker reads
        let key = node_key("node-d");
        store
            .put(key.clone(), cordoned("node-d"), 0, 6)
            .await
            .expect("term 6");
        let taken = store.take(7).await.expect("term 7 takes");
        assert_eq!(taken.term, 7);
        assert_eq!(
            taken.resources[&key].generation, 1,
            "the earlier write is read"
        );
        assert!(
            matches!(
                store.put(key.clone(), cordoned("node-d"), 1, 6).await,
                Err(SpecError::Fenced {
                    term: 6,
                    newer: 7,
                    ..
                })
            ),
            "the deposed holder writes nothing once the store is taken"
        );
        assert!(
            matches!(store.take(6).await, Err(SpecError::Fenced { .. })),
            "nor takes it back"
        );
    }

    #[tokio::test]
    async fn the_memory_store_keeps_generations_and_refuses_a_stale_write() {
        each_store_keeps_generations_and_refuses_a_stale_write(&MemoryStore::default()).await;
        each_store_refuses_a_write_under_an_older_term(&MemoryStore::default()).await;
        each_store_keeps_a_status_and_fences_it_with_the_specs(&MemoryStore::default()).await;
        each_store_is_taken_by_a_newer_term_alone(&MemoryStore::default()).await;
    }

    #[tokio::test]
    async fn the_file_store_keeps_a_status_across_a_reopen_and_fences_it() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("s.pb");
        each_store_keeps_a_status_and_fences_it_with_the_specs(&FileStore::new(&path)).await;
        assert_eq!(
            FileStore::new(&path)
                .take(5)
                .await
                .expect("reopen")
                .statuses[&engine_key("llama")],
            engine_status(2),
            "what a restarted coordinator reads"
        );
    }

    #[tokio::test]
    async fn the_file_store_refuses_a_write_under_an_older_term_and_is_taken_by_a_newer() {
        let dir = tempfile::tempdir().expect("tmp");
        each_store_refuses_a_write_under_an_older_term(&FileStore::new(dir.path().join("s.pb")))
            .await;
        each_store_is_taken_by_a_newer_term_alone(&FileStore::new(dir.path().join("t.pb"))).await;
    }

    #[test]
    fn two_file_stores_on_one_file_lose_no_write() {
        // two coordinators on a file system they share: every write is a read, a check and a
        // rewrite, and only a lock across processes keeps a second writer from reading the same
        // generation and writing over the first
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("specs.pb");
        let key = node_key("node-a");
        let written = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|s| {
            for i in 0..16 {
                let (path, key, written) = (&path, &key, &written);
                s.spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .expect("runtime");
                    // a store per thread, as a store per process
                    let store = FileStore::new(path);
                    for _ in 0..40 {
                        let read = store
                            .read()
                            .expect("read")
                            .resources
                            .get(key)
                            .map_or(0, |r| r.generation);
                        let put = store.put(key.clone(), cordoned(&format!("w{i}")), read, 1);
                        if rt.block_on(put).is_ok() {
                            written.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                });
            }
        });
        let generation = FileStore::new(&path).read().expect("read").resources[&key].generation;
        assert_eq!(
            generation,
            written.load(std::sync::atomic::Ordering::SeqCst),
            "every write that succeeded raised the generation once"
        );
    }

    #[tokio::test]
    async fn the_file_store_keeps_generations_across_a_reopen_and_refuses_a_stale_write() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("specs.pb");
        each_store_keeps_generations_and_refuses_a_stale_write(&FileStore::new(&path)).await;
        // a coordinator that restarts reads what the last one wrote
        let reopened = FileStore::new(&path);
        assert_eq!(
            reopened.take(1).await.expect("take").resources[&node_key("node-a")].generation,
            2
        );
        assert!(
            !path.with_extension("tmp").exists(),
            "no partial write left"
        );
    }
}
