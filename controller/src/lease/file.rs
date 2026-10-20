// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The file lease: an advisory lock on a file, for one host or a file system the coordinators and
//! daemons share. The lock lives as long as the open file that took it, so a hold needs no
//! renewal and never ends on its own: a holder that dies lets it go with its process. The lock
//! file keeps the last term; the record lies beside it, written whole and atomically.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use prost::Message as _;
use tokio_util::sync::CancellationToken;

use super::{BoxFuture, Hold, Lease, LeaseError, LeaseWatch};
use crate::proto::everserve::coord::v1::LeaseRecord;

/// Where a file lease's record is kept: beside its lock file, written whole and atomically so a
/// reader never sees part of one.
#[must_use]
pub fn record_path(lease: &Path) -> PathBuf {
    let mut name = lease.as_os_str().to_owned();
    name.push(".record");
    PathBuf::from(name)
}

/// An advisory lock on a file.
#[derive(Debug, Clone)]
pub struct FileLease {
    path: PathBuf,
}

impl FileLease {
    /// The lease on `path`, created at the first acquisition if missing.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The lock file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Lease for FileLease {
    fn try_acquire(&self) -> BoxFuture<'_, Result<Option<Box<dyn Hold>>, LeaseError>> {
        let path = self.path.clone();
        Box::pin(async move {
            let taken = tokio::task::spawn_blocking({
                let path = path.clone();
                move || lock(&path)
            })
            .await
            .map_err(|e| LeaseError::Io {
                path: path.clone(),
                source: std::io::Error::other(e),
            })??;
            Ok(taken.map(|(file, term)| {
                Box::new(FileHold {
                    path,
                    file,
                    term,
                    ended: CancellationToken::new(),
                }) as Box<dyn Hold>
            }))
        })
    }

    fn describe(&self) -> String {
        format!("file lease {}", self.path.display())
    }
}

/// Lock the file at `path` and take the next term under the lock; None if another open file
/// holds the lock.
fn lock(path: &Path) -> Result<Option<(File, u64)>, LeaseError> {
    let io = |source| LeaseError::Io {
        path: path.to_owned(),
        source,
    };
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(io)?;
    match file.try_lock() {
        Ok(()) => {
            let term = next_term(&file).map_err(io)?;
            Ok(Some((file, term)))
        }
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(source)) => Err(io(source)),
    }
}

/// A hold of a file lease: the open, locked file. Closing it, by release or drop, unlocks it.
#[derive(Debug)]
struct FileHold {
    path: PathBuf,
    file: File,
    term: u64,
    ended: CancellationToken,
}

impl Hold for FileHold {
    fn term(&self) -> u64 {
        self.term
    }

    fn ended(&self) -> &CancellationToken {
        &self.ended
    }

    fn publish(&self, mut record: LeaseRecord) -> BoxFuture<'_, Result<(), LeaseError>> {
        record.term = self.term;
        let path = record_path(&self.path);
        Box::pin(async move {
            let io = |source| LeaseError::Io {
                path: path.clone(),
                source,
            };
            let at = path.clone();
            tokio::task::spawn_blocking(move || {
                let tmp = at.with_extension("record.tmp");
                std::fs::write(&tmp, record.encode_to_vec())?;
                File::open(&tmp)?.sync_all()?;
                std::fs::rename(&tmp, &at)
            })
            .await
            .map_err(|e| io(std::io::Error::other(e)))?
            .map_err(io)
        })
    }

    fn release(self: Box<Self>) -> BoxFuture<'static, Result<(), LeaseError>> {
        Box::pin(async move {
            let path = self.path.clone();
            self.file
                .unlock()
                .map_err(|source| LeaseError::Io { path, source })
        })
    }

    fn describe(&self) -> String {
        format!("file lease {} at term {}", self.path.display(), self.term)
    }
}

/// A daemon's read-only watch of a file lease's record: for one host, or a file system the
/// coordinators and daemons share.
#[derive(Debug)]
pub struct FileLeaseWatch {
    path: PathBuf,
}

impl FileLeaseWatch {
    /// A watch of the record beside the lease file at `lease`.
    #[must_use]
    pub fn new(lease: &Path) -> Self {
        Self {
            path: record_path(lease),
        }
    }
}

impl LeaseWatch for FileLeaseWatch {
    fn read(&self) -> BoxFuture<'_, Result<Option<LeaseRecord>, LeaseError>> {
        Box::pin(async move {
            match tokio::fs::read(&self.path).await {
                Ok(bytes) => LeaseRecord::decode(bytes.as_slice())
                    .map(Some)
                    .map_err(|e| LeaseError::Decode {
                        at: self.path.display().to_string(),
                        reason: e.to_string(),
                    }),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(source) => Err(LeaseError::Io {
                    path: self.path.clone(),
                    source,
                }),
            }
        })
    }

    fn describe(&self) -> String {
        format!("file lease record {}", self.path.display())
    }
}

/// Read the term `file` holds (0 when empty), write the next one in its place and sync it: the
/// term this holder runs under.
fn next_term(mut file: &File) -> std::io::Result<u64> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut text = String::new();
    file.seek(SeekFrom::Start(0))?;
    file.read_to_string(&mut text)?;
    let term = text.trim().parse::<u64>().unwrap_or(0) + 1;
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    writeln!(file, "{term}")?;
    file.sync_all()?;
    Ok(term)
}
