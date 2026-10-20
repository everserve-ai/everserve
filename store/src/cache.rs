// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A rank's compile cache, moved between nodes by their stores (decision 0043). A serving
//! worker writes, beside its artifact, the manifest of the files its digest covers; a holder's
//! store serves exactly those files, each checked against its sha256 as it is read, and only for
//! the pair (key, digest) asked; a spare's node's store pulls them one chunk per request, checks
//! each again, renames each into place, and writes the manifest last, so a manifest present with
//! the pair's digest is the set in place. The file set has one implementation, the digest's: this
//! module moves what a manifest lists and never derives it.

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use everserve_wire::frame;
use everserve_wire::proto::everserve::mirror::v1::{
    CacheChunk, CacheFetch, CacheFile, CacheManifest, CacheManifestQuery, CacheRead, MirrorReply,
    MirrorRequest, mirror_reply, mirror_request,
};
use everserve_wire::transport::{Address, Tls, connect};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// The most of a file one `CacheChunk` carries, well under the link's frame cap.
pub const CACHE_CHUNK_BYTES: usize = 4 << 20;

/// What a fetched file is: the node group's to read and write, as the engines' vLLM writes beside
/// it and reads it (decision 0043; the pod contract's directories, 0037).
pub const FILE_MODE: u32 = 0o660;

/// What a directory the fetch makes is: the node group's, which what is made in it inherits.
pub const DIR_MODE: u32 = 0o2770;

/// How long a holder may take to answer one request before the fetch moves to the next holder.
pub const HOLDER_REPLY_TIMEOUT: Duration = Duration::from_secs(30);

/// Why a compile cache could not be served or fetched.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// The node keeps no compile caches: the store was started without a cache directory.
    #[error("this store keeps no compile caches (no --compile-cache-dir)")]
    NoDirectory,
    /// No manifest for the rank under the key.
    #[error("no compile cache for rank {rank} (dp {dp}) under key {key}")]
    Missing {
        /// The engine's key.
        key: String,
        /// The rank.
        rank: u32,
        /// Its data-parallel index.
        dp: u32,
    },
    /// The manifest held is of another digest than the one asked.
    #[error("rank {rank}'s compile cache under key {key} is {held}, not {asked}")]
    OtherDigest {
        /// The engine's key.
        key: String,
        /// The rank.
        rank: u32,
        /// The digest held.
        held: String,
        /// The digest asked.
        asked: String,
    },
    /// A key, or a path in a manifest, that would leave the key's directory.
    #[error("{0:?} is not a path inside a compile cache")]
    Path(String),
    /// A path the manifest does not list.
    #[error("{0} is not in the manifest")]
    Unlisted(String),
    /// A file whose bytes are not what the manifest says.
    #[error("{path}: sha256 {got}, the manifest says {listed}")]
    Altered {
        /// The file.
        path: String,
        /// Its sha256 as read.
        got: String,
        /// Its sha256 as listed.
        listed: String,
    },
    /// The fetch would keep more than the node's compile caches may hold.
    #[error("fetching {needed} bytes would keep {after} of compile caches, over {budget}")]
    OverBudget {
        /// The bytes the fetch adds.
        needed: u64,
        /// What the caches would hold after it.
        after: u64,
        /// What they may hold.
        budget: u64,
    },
    /// A malformed manifest.
    #[error("manifest {path}: {reason}")]
    Manifest {
        /// The manifest.
        path: PathBuf,
        /// Why.
        reason: String,
    },
    /// No holder served the set.
    #[error("no holder served rank {rank}'s compile cache: {why}")]
    NoHolder {
        /// The rank.
        rank: u32,
        /// Each holder's failure.
        why: String,
    },
    /// The file system failed.
    #[error("{path}: {source}")]
    Io {
        /// The path.
        path: PathBuf,
        /// The failure.
        source: std::io::Error,
    },
}

/// The manifest as a serving worker writes it (decision 0043).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Manifest {
    key: String,
    rank: u32,
    dp: u32,
    digest: String,
    files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ManifestFile {
    path: String,
    size: u64,
    sha256: String,
}

/// A node's compile caches: `<dir>/torch_compile_cache/torch_aot_compile/<key>`, vLLM's layout
/// under its cache root, holding at most `budget` bytes.
pub struct CompileCaches {
    root: PathBuf,
    budget: u64,
    tls: Option<Tls>,
    peer_name: String,
    /// One fetch at a time: a second for the same pair finds the first's manifest in place.
    fetching: Mutex<()>,
}

impl CompileCaches {
    /// The caches under the cache root `dir`, at most `budget` bytes, fetched from peer stores
    /// with `tls`, whose certificates name `peer_name`.
    #[must_use]
    pub fn new(dir: &Path, budget: u64, tls: Option<Tls>, peer_name: &str) -> Self {
        let root = dir.join("torch_compile_cache").join("torch_aot_compile");
        // a fetch this process did not finish (a crash, a full disk) left its partial file: it
        // would count against the budget and never be read
        let swept = sweep_fetching(&root);
        if swept > 0 {
            tracing::info!(
                "{swept} unfinished compile cache fetches removed under {}",
                root.display()
            );
        }
        Self {
            root,
            budget,
            tls,
            peer_name: peer_name.to_owned(),
            fetching: Mutex::new(()),
        }
    }

    fn key_dir(&self, key: &str) -> Result<PathBuf, CacheError> {
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(CacheError::Path(key.to_owned()));
        }
        Ok(self.root.join(key))
    }

    fn manifest_path(&self, key: &str, rank: u32, dp: u32) -> Result<PathBuf, CacheError> {
        Ok(self
            .key_dir(key)?
            .join(format!("rank_{rank}_{dp}.manifest.json")))
    }

    /// The manifest of rank `rank`'s set under `key`, if one is held, its every path checked.
    fn held(&self, key: &str, rank: u32, dp: u32) -> Result<Option<Manifest>, CacheError> {
        let path = self.manifest_path(key, rank, dp)?;
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(CacheError::Io { path, source }),
        };
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| CacheError::Manifest {
            path: path.clone(),
            reason: e.to_string(),
        })?;
        if (manifest.key.as_str(), manifest.rank, manifest.dp) != (key, rank, dp) {
            return Err(CacheError::Manifest {
                path,
                reason: format!(
                    "names key {} rank {} dp {}",
                    manifest.key, manifest.rank, manifest.dp
                ),
            });
        }
        for f in &manifest.files {
            inside(&f.path)?;
        }
        Ok(Some(manifest))
    }

    /// The manifest of the pair `q` asks for (a holder's answer).
    ///
    /// # Errors
    ///
    /// No cache directory, no manifest, another digest, or a malformed manifest.
    pub fn manifest(&self, q: &CacheManifestQuery) -> Result<CacheManifest, CacheError> {
        let m = self.pair(&q.key, q.rank, q.dp, &q.digest)?;
        Ok(CacheManifest {
            key: m.key,
            rank: m.rank,
            dp: m.dp,
            digest: m.digest,
            files: m
                .files
                .into_iter()
                .map(|f| CacheFile {
                    path: f.path,
                    size: f.size,
                    sha256: f.sha256,
                })
                .collect(),
        })
    }

    fn pair(&self, key: &str, rank: u32, dp: u32, digest: &str) -> Result<Manifest, CacheError> {
        let m = self
            .held(key, rank, dp)?
            .ok_or_else(|| CacheError::Missing {
                key: key.to_owned(),
                rank,
                dp,
            })?;
        if m.digest != digest {
            return Err(CacheError::OtherDigest {
                key: key.to_owned(),
                rank,
                held: m.digest,
                asked: digest.to_owned(),
            });
        }
        Ok(m)
    }

    /// Up to [`CACHE_CHUNK_BYTES`] of a file the pair's manifest lists, from `r.offset` (a
    /// holder's answer). A file's last chunk is checked against its listed sha256 before it is
    /// served, so a holder whose file was altered serves nothing of it.
    ///
    /// # Errors
    ///
    /// As [`CompileCaches::manifest`]; a path the manifest does not list; a file altered.
    pub async fn read(&self, r: &CacheRead) -> Result<CacheChunk, CacheError> {
        let m = self.pair(&r.key, r.rank, r.dp, &r.digest)?;
        let listed = m
            .files
            .iter()
            .find(|f| f.path == r.path)
            .ok_or_else(|| CacheError::Unlisted(r.path.clone()))?;
        let path = self.key_dir(&r.key)?.join(&listed.path);
        if r.offset.saturating_add(CACHE_CHUNK_BYTES as u64) >= listed.size {
            // the last chunk: the whole file is checked before its end is served
            let got = sha256_of(&path).await?;
            if got != listed.sha256 {
                return Err(CacheError::Altered {
                    path: listed.path.clone(),
                    got,
                    listed: listed.sha256.clone(),
                });
            }
        }
        let io = |source| CacheError::Io {
            path: path.clone(),
            source,
        };
        let mut file = tokio::fs::File::open(&path).await.map_err(io)?;
        file.seek(std::io::SeekFrom::Start(r.offset))
            .await
            .map_err(io)?;
        let want = usize::try_from(listed.size.saturating_sub(r.offset))
            .unwrap_or(usize::MAX)
            .min(CACHE_CHUNK_BYTES);
        let mut data = vec![0; want];
        file.read_exact(&mut data).await.map_err(io)?;
        Ok(CacheChunk { data })
    }

    /// Put the pair `f` names in place on this node from the first of its holders that serves
    /// it; nothing to do when its manifest is already here with the pair's digest.
    ///
    /// # Errors
    ///
    /// No cache directory; the fetch would exceed the budget; no holder served the set.
    pub async fn fetch(&self, f: &CacheFetch) -> Result<(), CacheError> {
        let _one = self.fetching.lock().await;
        if self
            .held(&f.key, f.rank, f.dp)?
            .is_some_and(|m| m.digest == f.digest)
        {
            return Ok(());
        }
        let mut why = Vec::new();
        for holder in &f.holders {
            match self.fetch_from(holder, f).await {
                Ok(()) => return Ok(()),
                Err(e @ CacheError::OverBudget { .. }) => return Err(e),
                Err(e) => {
                    warn!("rank {}'s compile cache from {holder}: {e}", f.rank);
                    why.push(format!("{holder}: {e}"));
                }
            }
        }
        Err(CacheError::NoHolder {
            rank: f.rank,
            why: if why.is_empty() {
                "none named".to_owned()
            } else {
                why.join("; ")
            },
        })
    }

    async fn fetch_from(&self, holder: &str, f: &CacheFetch) -> Result<(), CacheError> {
        let mut link = Holder::open(holder, self.tls.as_ref(), &self.peer_name, f.rank).await?;
        let m = link.manifest(f).await?;
        let dir = self.key_dir(&f.key)?;
        let mut needed = 0;
        let mut wanted = Vec::new();
        for file in &m.files {
            inside(&file.path)?;
            let dest = dir.join(&file.path);
            if dest.is_file() && sha256_of(&dest).await? == file.sha256 {
                continue;
            }
            needed += file.size;
            wanted.push(file.clone());
        }
        let held = bytes_under(&self.root)?;
        if held + needed > self.budget {
            return Err(CacheError::OverBudget {
                needed,
                after: held + needed,
                budget: self.budget,
            });
        }
        for file in &wanted {
            link.pull(f, file, &dir.join(&file.path)).await?;
        }
        self.write_manifest(m)?;
        info!(
            "rank {}'s compile cache {}/{} in place from {holder}: {} files, {needed} bytes fetched",
            f.rank,
            f.key,
            f.digest,
            wanted.len()
        );
        Ok(())
    }

    /// Write `m` as the rank's manifest, under a temporary name renamed into place: present, the
    /// set it lists is in place.
    fn write_manifest(&self, m: CacheManifest) -> Result<(), CacheError> {
        let path = self.manifest_path(&m.key, m.rank, m.dp)?;
        let manifest = Manifest {
            key: m.key,
            rank: m.rank,
            dp: m.dp,
            digest: m.digest,
            files: m
                .files
                .into_iter()
                .map(|f| ManifestFile {
                    path: f.path,
                    size: f.size,
                    sha256: f.sha256,
                })
                .collect(),
        };
        let temp = path.with_extension("json.fetching");
        let unfinished = Unfinished(temp.clone());
        let text = serde_json::to_vec(&manifest).map_err(|e| CacheError::Manifest {
            path: path.clone(),
            reason: e.to_string(),
        })?;
        std::fs::write(&temp, text).map_err(|source| CacheError::Io {
            path: temp.clone(),
            source,
        })?;
        set_mode(&temp, FILE_MODE)?;
        std::fs::rename(&temp, &path).map_err(|source| CacheError::Io { path, source })?;
        unfinished.finished();
        Ok(())
    }
}

/// A connection to a holder's store, one reply per request.
struct Holder {
    rd: tokio::io::ReadHalf<everserve_wire::transport::Stream>,
    wr: tokio::io::WriteHalf<everserve_wire::transport::Stream>,
    rank: u32,
}

impl Holder {
    async fn open(
        holder: &str,
        tls: Option<&Tls>,
        peer_name: &str,
        rank: u32,
    ) -> Result<Self, CacheError> {
        let failed = |why: String| CacheError::NoHolder { rank, why };
        let address = Address::parse(holder).map_err(|e| failed(e.to_string()))?;
        let stream = connect(&address, tls, peer_name)
            .await
            .map_err(|e| failed(e.to_string()))?;
        let (rd, wr) = tokio::io::split(stream);
        Ok(Self { rd, wr, rank })
    }

    fn failed(&self, why: String) -> CacheError {
        CacheError::NoHolder {
            rank: self.rank,
            why,
        }
    }

    /// Ask `kind` and wait at most [`HOLDER_REPLY_TIMEOUT`] for its answer. A wait cut short
    /// leaves the connection mid-frame; the fetch then drops it and asks the next holder.
    async fn ask(&mut self, kind: mirror_request::Kind) -> Result<mirror_reply::Kind, CacheError> {
        frame::send(&mut self.wr, &MirrorRequest { kind: Some(kind) })
            .await
            .map_err(|e| self.failed(e.to_string()))?;
        let reply: MirrorReply =
            match tokio::time::timeout(HOLDER_REPLY_TIMEOUT, frame::recv(&mut self.rd)).await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => return Err(self.failed(e.to_string())),
                Err(_) => {
                    return Err(self.failed(format!("no answer within {HOLDER_REPLY_TIMEOUT:?}")));
                }
            };
        match reply.kind {
            Some(k) if reply.error.is_empty() => Ok(k),
            _ => Err(self.failed(reply.error)),
        }
    }

    /// The holder's manifest for the pair `f` names, refused unless it is that pair.
    async fn manifest(&mut self, f: &CacheFetch) -> Result<CacheManifest, CacheError> {
        let asked = mirror_request::Kind::CacheManifest(CacheManifestQuery {
            key: f.key.clone(),
            rank: f.rank,
            dp: f.dp,
            digest: f.digest.clone(),
        });
        let mirror_reply::Kind::CacheManifest(m) = self.ask(asked).await? else {
            return Err(self.failed("answered the manifest query with something else".into()));
        };
        if (m.key.as_str(), m.rank, m.dp, m.digest.as_str())
            != (f.key.as_str(), f.rank, f.dp, f.digest.as_str())
        {
            return Err(self.failed(format!(
                "served key {} rank {} dp {} digest {}",
                m.key, m.rank, m.dp, m.digest
            )));
        }
        Ok(m)
    }

    /// Pull `file` of the pair `f` names into `dest`: chunk by chunk into a temporary name beside
    /// it, its size and sha256 checked, then renamed into place.
    async fn pull(
        &mut self,
        f: &CacheFetch,
        file: &CacheFile,
        dest: &Path,
    ) -> Result<(), CacheError> {
        let parent = dest.parent().unwrap_or(dest).to_owned();
        create_dirs(&parent)?;
        let temp = parent.join(format!(
            ".{}.fetching",
            dest.file_name().and_then(|n| n.to_str()).unwrap_or("file")
        ));
        let io = |source| CacheError::Io {
            path: temp.clone(),
            source,
        };
        // whatever ends this fetch short of its rename (the holder gone, a full disk, a
        // mismatch), the partial file goes with it
        let unfinished = Unfinished(temp.clone());
        let mut out = tokio::fs::File::create(&temp).await.map_err(io)?;
        let mut hash = ring::digest::Context::new(&ring::digest::SHA256);
        let mut offset = 0;
        while offset < file.size {
            let read = mirror_request::Kind::CacheRead(CacheRead {
                key: f.key.clone(),
                rank: f.rank,
                dp: f.dp,
                digest: f.digest.clone(),
                path: file.path.clone(),
                offset,
            });
            let mirror_reply::Kind::CacheChunk(c) = self.ask(read).await? else {
                return Err(self.failed("answered a read with something else".into()));
            };
            if c.data.is_empty() {
                return Err(self.failed(format!("{} ended at {offset}", file.path)));
            }
            hash.update(&c.data);
            out.write_all(&c.data).await.map_err(io)?;
            offset += c.data.len() as u64;
        }
        out.sync_all().await.map_err(io)?;
        drop(out);
        let got = hex(hash.finish().as_ref());
        if offset != file.size || got != file.sha256 {
            return Err(CacheError::Altered {
                path: file.path.clone(),
                got,
                listed: file.sha256.clone(),
            });
        }
        set_mode(&temp, FILE_MODE)?;
        std::fs::rename(&temp, dest).map_err(|source| CacheError::Io {
            path: dest.to_owned(),
            source,
        })?;
        unfinished.finished();
        Ok(())
    }
}

/// A file being written under a `.fetching` name: removed when dropped unless
/// [`Unfinished::finished`] said it was renamed into place.
struct Unfinished(PathBuf);

impl Unfinished {
    /// The file was renamed into place: nothing to remove.
    fn finished(self) {
        std::mem::forget(self);
    }
}

impl Drop for Unfinished {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Remove every `.fetching` file under `dir`; how many went. A directory that cannot be read
/// is left as it is.
fn sweep_fetching(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut swept = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            swept += sweep_fetching(&path);
        } else if path.extension().is_some_and(|e| e == "fetching")
            && std::fs::remove_file(&path).is_ok()
        {
            swept += 1;
        }
    }
    swept
}

/// `path` as a manifest names it: relative, `/`-separated, never leaving its key's directory.
fn inside(path: &str) -> Result<(), CacheError> {
    let p = Path::new(path);
    let ok = !path.is_empty()
        && !path.contains('\\')
        && p.components().all(|c| matches!(c, Component::Normal(_)));
    if ok {
        Ok(())
    } else {
        Err(CacheError::Path(path.to_owned()))
    }
}

/// Make `dir` and those of its parents missing, each [`DIR_MODE`]: stated, not left to the
/// process's umask or a parent's setgid bit. A directory already there is left as it is.
fn create_dirs(dir: &Path) -> Result<(), CacheError> {
    let missing: Vec<&Path> = dir.ancestors().take_while(|d| !d.exists()).collect();
    std::fs::create_dir_all(dir).map_err(|source| CacheError::Io {
        path: dir.to_owned(),
        source,
    })?;
    for made in missing.into_iter().rev() {
        set_mode(made, DIR_MODE)?;
    }
    Ok(())
}

fn set_mode(path: &Path, mode: u32) -> Result<(), CacheError> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).map_err(|source| {
        CacheError::Io {
            path: path.to_owned(),
            source,
        }
    })
}

async fn sha256_of(path: &Path) -> Result<String, CacheError> {
    let io = |source| CacheError::Io {
        path: path.to_owned(),
        source,
    };
    let mut file = tokio::fs::File::open(path).await.map_err(io)?;
    let mut hash = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buf).await.map_err(io)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(hex(hash.finish().as_ref()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Every byte of every file under `dir`; 0 for a directory not yet made.
fn bytes_under(dir: &Path) -> Result<u64, CacheError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(source) => {
            return Err(CacheError::Io {
                path: dir.to_owned(),
                source,
            });
        }
    };
    let mut total = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let meta = entry.metadata().map_err(|source| CacheError::Io {
            path: path.clone(),
            source,
        })?;
        total += if meta.is_dir() {
            bytes_under(&path)?
        } else {
            meta.len()
        };
    }
    Ok(total)
}
