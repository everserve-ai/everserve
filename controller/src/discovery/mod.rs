// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! How processes find each other (`proto/everserve/proto/discovery.proto`).
//!
//! Two backends, the same split Dynamo uses: a directory of records for development and single
//! hosts, and Kubernetes `EndpointSlices` in production (phase 2, implemented when there is a
//! cluster to test against). A node daemon publishes its record on start and withdraws it on
//! stop; readers never cache across a restart because the record carries the pid and start time.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use everserve_wire::proto::everserve::discovery::v1::DaemonRecord;

/// Why a record could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// The directory or file could not be accessed.
    #[error("discovery: {0}")]
    Io(#[from] std::io::Error),
    /// A file in the directory is not a `DaemonRecord`.
    #[error("discovery: {path} is not a daemon record: {source}")]
    Parse {
        /// The offending file.
        path: PathBuf,
        /// The JSON error.
        source: serde_json::Error,
    },
}

/// Read side: what the coordinator and the drill driver need.
pub trait Discovery {
    /// Every daemon currently published.
    ///
    /// # Errors
    ///
    /// [`DiscoveryError`] when the backend cannot be read or holds a malformed record.
    fn daemons(&self) -> Result<Vec<DaemonRecord>, DiscoveryError>;

    /// The daemon for `node`, if published.
    ///
    /// # Errors
    ///
    /// As [`Discovery::daemons`].
    fn daemon(&self, node: &str) -> Result<Option<DaemonRecord>, DiscoveryError> {
        Ok(self.daemons()?.into_iter().find(|r| r.node == node))
    }
}

/// A directory of `<node>.json` records, one per daemon.
#[derive(Debug, Clone)]
pub struct FileDiscovery {
    dir: PathBuf,
}

impl FileDiscovery {
    /// Records live in `dir`, which is created if missing.
    ///
    /// # Errors
    ///
    /// The directory could not be created.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, DiscoveryError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn path_for(&self, node: &str) -> PathBuf {
        self.dir.join(format!("{node}.json"))
    }

    /// Publish `record`, replacing any record for the same node. The write is atomic: a reader
    /// sees the old file or the new one, never a partial one.
    ///
    /// # Errors
    ///
    /// The file could not be written.
    pub fn publish(&self, record: &DaemonRecord) -> Result<(), DiscoveryError> {
        let path = self.path_for(&record.node);
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(record).map_err(|e| DiscoveryError::Parse {
            path: path.clone(),
            source: e,
        })?;
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Withdraw `node`'s record. Missing is not an error: the daemon may have crashed before
    /// publishing.
    ///
    /// # Errors
    ///
    /// A failure other than the file being absent.
    pub fn withdraw(&self, node: &str) -> Result<(), DiscoveryError> {
        match std::fs::remove_file(self.path_for(node)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

impl Discovery for FileDiscovery {
    fn daemons(&self) -> Result<Vec<DaemonRecord>, DiscoveryError> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let text = std::fs::read_to_string(&path)?;
            let record: DaemonRecord =
                serde_json::from_str(&text).map_err(|source| DiscoveryError::Parse {
                    path: path.clone(),
                    source,
                })?;
            out.push(record);
        }
        out.sort_by(|a, b| a.node.cmp(&b.node));
        Ok(out)
    }
}

/// A record for this process.
#[must_use]
pub fn record(
    node: &str,
    address: &Path,
    mirror_address: Option<&Path>,
    host: Option<&str>,
) -> DaemonRecord {
    let started_at_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
    DaemonRecord {
        node: node.to_owned(),
        address: address.display().to_string(),
        engines: Vec::new(),
        started_at_ms,
        pid: std::process::id(),
        mirror_address: mirror_address
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        host: host.unwrap_or_default().to_owned(),
    }
}
