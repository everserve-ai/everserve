// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Storage for one engine's committed rows, keyed by rank and layer, indexed by slot, in shared
//! memory (decision 0011).
//!
//! A rank's rows for one layer are a dense table with one row per slot id, backed by a file under
//! `<data dir>/<engine>/rank<r>/` that a restoring rank maps directly, plus a written mask held
//! here, so a put overwrites in place (idempotent re-runs) and a restore names exactly the written
//! slots. A commit names, in one descriptor, the rows of every table in a staging file the worker
//! wrote under the same directory (decision 0019); the store maps the file once and copies the
//! rows into the tables. Row shape and dtype are fixed by
//! the first commit for a layer; a later commit with a different shape is a protocol error, not a
//! resize. Each slot remembers the step of the commit that last wrote it, so a restore can name only
//! the slots written since a step the restoring rank already holds.

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use everserve_wire::proto::everserve::mirror::v1::{
    Replicated, ReplicatedTable, SlotRange, StagedStep, TableRef,
};
use memmap2::{Mmap, MmapMut};

use crate::shm;

/// Why a commit was refused.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// Replicated rows arrived without a checksum each (decision 0030).
    #[error("layer {layer}: {rows} rows came with {checksums} checksums")]
    Unchecked {
        /// The layer.
        layer: u32,
        /// Rows in the table.
        rows: usize,
        /// Checksums with them.
        checksums: usize,
    },
    /// A replicated row does not match the checksum its source recorded when it was written: the
    /// whole step is refused and nothing of it is written (decision 0030).
    #[error(
        "layer {layer}: slot {slot} does not match its checksum ({got:#010x}, recorded {recorded:#010x})"
    )]
    Checksum {
        /// The layer.
        layer: u32,
        /// The slot.
        slot: i64,
        /// The CRC32C of the bytes that arrived.
        got: u32,
        /// The CRC32C the source recorded.
        recorded: u32,
    },
    /// The rows' shape or dtype differs from the layer's.
    #[error(
        "layer {layer}: row shape {got:?} {got_dtype} differs from {expected:?} {expected_dtype}"
    )]
    Shape {
        /// The layer.
        layer: u32,
        /// Shape of the offending rows.
        got: Vec<u32>,
        /// Dtype of the offending rows.
        got_dtype: String,
        /// Shape the layer was created with.
        expected: Vec<u32>,
        /// Dtype the layer was created with.
        expected_dtype: String,
    },
    /// The staging file does not hold `slots.len()` rows at `offset`.
    #[error("layer {layer}: stage {stage} holds {bytes} bytes, {needed} needed from {offset}")]
    Size {
        /// The layer.
        layer: u32,
        /// The staging file.
        stage: String,
        /// Bytes the file has.
        bytes: usize,
        /// Bytes the rows need.
        needed: usize,
        /// Where the rows start.
        offset: u64,
    },
    /// The engine writes more slot ids than this store's tables hold, or states none.
    #[error(
        "engine {engine} rank {rank} writes {wants} slot ids, this store holds {holds} (its \
         --slots, the chart's node.store.slots): size it to the engine's KV blocks times its \
         block size"
    )]
    Undersized {
        /// The engine.
        engine: String,
        /// Its rank.
        rank: u32,
        /// The slot ids it writes; 0 when it stated none.
        wants: u64,
        /// The slot ids each of this store's tables holds.
        holds: usize,
    },
    /// An engine name a store would make a directory from, that is not one plain path component.
    #[error("engine name {engine:?}: not one plain path component")]
    EngineName {
        /// The name as it came.
        engine: String,
    },
    /// A row shape with no size, or one whose size overflows.
    #[error("layer {layer}: a row of shape {shape:?} has no size or overflows")]
    RowSize {
        /// The table.
        layer: u32,
        /// The shape as it came.
        shape: Vec<u32>,
    },
    /// A slot id is beyond the layer's table.
    #[error("layer {layer}: slot {slot} outside 0..{slots}")]
    Slot {
        /// The layer.
        layer: u32,
        /// The offending slot id.
        slot: i64,
        /// Slots in the table.
        slots: usize,
    },
    /// Accepting the commit would exceed the memory budget.
    #[error("memory budget of {budget} bytes exceeded: {needed} needed")]
    Budget {
        /// The budget.
        budget: usize,
        /// Bytes that would be in use.
        needed: usize,
    },
    /// A dtype this store does not know the size of.
    #[error("layer {layer}: unknown dtype {dtype:?}")]
    Dtype {
        /// The layer.
        layer: u32,
        /// The dtype name received.
        dtype: String,
    },
    /// The commit's epoch is older than the rank's current epoch.
    #[error("stale epoch {got} < {current}")]
    StaleEpoch {
        /// The commit's epoch.
        got: u64,
        /// The rank's current epoch.
        current: u64,
    },
    /// A staging file name that is not a plain file name in the rank directory.
    #[error("stage {stage:?} is not a file name")]
    StageName {
        /// The offending name.
        stage: String,
    },
    /// The rank directory, a table file or a staging file could not be opened or mapped.
    #[error("{what}: {source}")]
    Io {
        /// What was being opened.
        what: String,
        /// The error.
        source: std::io::Error,
    },
}

impl StorageError {
    /// A stable, low-cardinality name for metrics.
    #[must_use]
    pub fn reason(&self) -> &'static str {
        match self {
            Self::Shape { .. } => "shape",
            Self::Size { .. } => "size",
            Self::Slot { .. } => "slot",
            Self::Undersized { .. } => "undersized",
            Self::Budget { .. } => "budget",
            Self::Dtype { .. } => "dtype",
            Self::StaleEpoch { .. } => "stale_epoch",
            Self::StageName { .. } => "stage_name",
            Self::Io { .. } => "io",
            Self::Unchecked { .. } => "unchecked",
            Self::Checksum { .. } => "checksum",
            Self::EngineName { .. } => "engine_name",
            Self::RowSize { .. } => "row_size",
        }
    }
}

/// An engine name as a store may use it in a path: one plain component of letters, digits,
/// `-`, `_` and `.`, neither `.` nor `..`, at most 255 bytes. Checked before any path is made
/// from a name the wire supplied.
fn engine_name(engine: &str) -> Result<(), StorageError> {
    let plain = !engine.is_empty()
        && engine.len() <= 255
        && engine != "."
        && engine != ".."
        && engine
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.');
    if plain {
        Ok(())
    } else {
        Err(StorageError::EngineName {
            engine: engine.to_owned(),
        })
    }
}

/// The bytes of one row of `shape` with `elem_bytes` an element: every dimension at least one,
/// the product within `usize`.
fn row_bytes_of(layer: u32, shape: &[u32], elem_bytes: usize) -> Result<usize, StorageError> {
    let overflow = || StorageError::RowSize {
        layer,
        shape: shape.to_vec(),
    };
    if shape.is_empty() || shape.contains(&0) {
        return Err(overflow());
    }
    shape
        .iter()
        .try_fold(elem_bytes, |acc, &d| acc.checked_mul(d as usize))
        .filter(|&bytes| bytes > 0)
        .ok_or_else(overflow)
}

/// A table to make for a rank: all of `specs` are made, or none, the files of those made before
/// a failure removed so a tmpfs data directory keeps nothing uncounted.
struct LayerSpec {
    table: u32,
    shape: Vec<u32>,
    dtype: String,
    row_bytes: usize,
}

fn create_layers(
    rank_dir: &Path,
    specs: &[LayerSpec],
    num_slots: usize,
) -> Result<Vec<(u32, Layer)>, StorageError> {
    let mut made: Vec<(u32, Layer)> = Vec::with_capacity(specs.len());
    for spec in specs {
        let path = rank_dir.join(format!("table{}.rows", spec.table));
        match Layer::create(
            path.clone(),
            &spec.shape,
            &spec.dtype,
            spec.row_bytes,
            num_slots,
        ) {
            Ok(layer) => made.push((spec.table, layer)),
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                for (_, layer) in made {
                    let _ = std::fs::remove_file(&layer.path);
                }
                return Err(e);
            }
        }
    }
    Ok(made)
}

/// Check every table of a replicated step against the rank's tables before anything is written:
/// the dtype, the row size, the data's length, the slot ids, every row's checksum, and the shape
/// of each table the rank already has. Returns the tables to create and the memory they take.
fn validate_replicated(
    r: &Rank,
    msg: &Replicated,
    num_slots: usize,
) -> Result<(Vec<LayerSpec>, usize), StorageError> {
    let mut new_tables: Vec<LayerSpec> = Vec::new();
    let mut added = 0usize;
    for t in &msg.tables {
        let Some(elem) = dtype_bytes(&t.dtype) else {
            return Err(StorageError::Dtype {
                layer: t.table,
                dtype: t.dtype.clone(),
            });
        };
        let row_bytes = row_bytes_of(t.table, &t.shape, elem)?;
        let needed = t
            .slots
            .len()
            .checked_mul(row_bytes)
            .ok_or_else(|| StorageError::RowSize {
                layer: t.table,
                shape: t.shape.clone(),
            })?;
        if t.data.len() != needed {
            return Err(StorageError::Size {
                layer: t.table,
                stage: "replicated".into(),
                bytes: t.data.len(),
                needed,
                offset: 0,
            });
        }
        for &s in &t.slots {
            if s < 0 || usize::try_from(s).is_ok_and(|s| s >= num_slots) {
                return Err(StorageError::Slot {
                    layer: t.table,
                    slot: s,
                    slots: num_slots,
                });
            }
        }
        verify_rows(t, row_bytes)?;
        match r.layers.get(&t.table) {
            Some(layer) if layer.shape != t.shape || layer.dtype != t.dtype => {
                return Err(StorageError::Shape {
                    layer: t.table,
                    got: t.shape.clone(),
                    got_dtype: t.dtype.clone(),
                    expected: layer.shape.clone(),
                    expected_dtype: layer.dtype.clone(),
                });
            }
            Some(_) => {}
            None => {
                new_tables.push(LayerSpec {
                    table: t.table,
                    shape: t.shape.clone(),
                    dtype: t.dtype.clone(),
                    row_bytes,
                });
                added = added.saturating_add(row_bytes.saturating_mul(num_slots));
            }
        }
    }
    Ok((new_tables, added))
}

/// Every row of a replicated table matches the checksum its source recorded (decision 0030).
fn verify_rows(t: &ReplicatedTable, row_bytes: usize) -> Result<(), StorageError> {
    if t.crc32c.len() != t.slots.len() {
        return Err(StorageError::Unchecked {
            layer: t.table,
            rows: t.slots.len(),
            checksums: t.crc32c.len(),
        });
    }
    for (i, (&slot, &recorded)) in t.slots.iter().zip(&t.crc32c).enumerate() {
        let got = crc32c::crc32c(&t.data[i * row_bytes..(i + 1) * row_bytes]);
        if got != recorded {
            return Err(StorageError::Checksum {
                layer: t.table,
                slot,
                got,
                recorded,
            });
        }
    }
    Ok(())
}

fn io(what: impl Into<String>, source: std::io::Error) -> StorageError {
    StorageError::Io {
        what: what.into(),
        source,
    }
}

/// One layer's rows for one rank: a table file mapped writable, and per slot the step of the
/// commit that last wrote it (0: never written; steps are numbered from 1 on the wire).
#[derive(Debug)]
pub struct Layer {
    shape: Vec<u32>,
    dtype: String,
    row_bytes: usize,
    path: PathBuf,
    rows: MmapMut,
    written_step: Vec<u64>,
    /// Per slot, the CRC32C of the row as it was written (decision 0030); 0 for a slot never
    /// written.
    crc: Vec<u32>,
    /// How many slots have been written, so whether the layer holds rows is one comparison.
    written: usize,
}

impl Layer {
    fn create(
        path: PathBuf,
        shape: &[u32],
        dtype: &str,
        row_bytes: usize,
        num_slots: usize,
    ) -> Result<Self, StorageError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| io(format!("create table {}", path.display()), e))?;
        let len = u64::try_from(row_bytes * num_slots).unwrap_or(u64::MAX);
        file.set_len(len)
            .map_err(|e| io(format!("size table {}", path.display()), e))?;
        let mut rows =
            shm::map_table(&file).map_err(|e| io(format!("map table {}", path.display()), e))?;
        // touch every page now: a commit into a fresh table otherwise pays the page faults of the
        // whole file at its first write, seconds for a gigabyte on a busy host (decision 0020)
        for page in rows.chunks_mut(4096) {
            page[0] = 0;
        }
        Ok(Self {
            shape: shape.to_vec(),
            dtype: dtype.to_owned(),
            row_bytes,
            path,
            rows,
            written_step: vec![0; num_slots],
            crc: vec![0; num_slots],
            written: 0,
        })
    }

    /// Whether any slot has been written.
    #[must_use]
    pub fn holds_rows(&self) -> bool {
        self.written > 0
    }

    /// Bytes this layer's table occupies.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows.len()
    }

    /// Slot ids written by a commit after `since_step`, as ascending ranges; `0` means every
    /// written slot.
    #[must_use]
    pub fn written_ranges(&self, since_step: u64) -> Vec<SlotRange> {
        let hit = |i: usize| self.written_step[i] > since_step;
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.written_step.len() {
            if hit(i) {
                let start = i;
                while i < self.written_step.len() && hit(i) {
                    i += 1;
                }
                out.push(SlotRange {
                    start: i64::try_from(start).unwrap_or(i64::MAX),
                    count: i64::try_from(i - start).unwrap_or(i64::MAX),
                });
            } else {
                i += 1;
            }
        }
        out
    }

    /// The table as a restore reply names it, with the slots written after `since_step`.
    #[must_use]
    pub fn table_ref(&self, layer: u32, since_step: u64) -> TableRef {
        TableRef {
            layer,
            path: self.path.to_string_lossy().into_owned(),
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            written: self.written_ranges(since_step),
        }
    }

    /// Written slot ids in ascending order.
    #[must_use]
    pub fn written_slots(&self) -> Vec<usize> {
        (0..self.written_step.len())
            .filter(|&i| self.written_step[i] > 0)
            .collect()
    }

    /// The bytes of one row.
    #[must_use]
    pub fn row(&self, slot: usize) -> &[u8] {
        &self.rows[slot * self.row_bytes..(slot + 1) * self.row_bytes]
    }

    /// The CRC32C recorded for one row when it was written; 0 for a slot never written.
    #[must_use]
    pub fn crc(&self, slot: usize) -> u32 {
        self.crc[slot]
    }

    /// Write `data` as row `slot` at `step`, recording its CRC32C.
    fn put(&mut self, slot: usize, data: &[u8], step: u64, crc: u32) {
        let row_bytes = self.row_bytes;
        self.rows[slot * row_bytes..(slot + 1) * row_bytes].copy_from_slice(data);
        self.note_written(slot, step, crc);
    }

    /// Record that row `slot` was written at `step`, its checksum taken from the row in place,
    /// while its bytes are in cache: what replication carries and a replica verifies
    /// (decision 0030).
    fn mark_written(&mut self, slot: usize, step: u64) {
        let crc = crc32c::crc32c(self.row(slot));
        self.note_written(slot, step, crc);
    }

    fn note_written(&mut self, slot: usize, step: u64, crc: u32) {
        if self.written_step[slot] == 0 {
            self.written += 1;
        }
        self.written_step[slot] = step.max(1); // step 0 still counts as written
        self.crc[slot] = crc;
    }
}

/// One table a descriptor names: its id and where its rows start in the staging file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagedTable {
    /// The table id (`layer * table_stride + head slot`).
    pub table: u32,
    /// Byte offset of the table's first row in the staging file.
    pub offset: u64,
}

/// The tables a step descriptor names, in layer-major order.
///
/// # Errors
///
/// [`StorageError::StageName`] for a descriptor whose table ids would overflow.
pub fn derive(b: &StagedStep) -> Result<Vec<StagedTable>, StorageError> {
    let mut out = Vec::with_capacity(b.layers.len() * b.heads.len());
    for &layer in &b.layers {
        for (h, &head) in b.heads.iter().enumerate() {
            let table = layer
                .checked_mul(b.table_stride)
                .and_then(|t| t.checked_add(head))
                .ok_or_else(|| StorageError::StageName {
                    stage: format!("{}: table id overflow", b.stage),
                })?;
            let offset = b.base_offset
                + u64::from(layer) * b.layer_stride
                + u64::try_from(h).unwrap_or(u64::MAX) * b.head_stride;
            out.push(StagedTable { table, offset });
        }
    }
    Ok(out)
}

/// Bytes per element for the dtypes a KV cache uses; `None` for a name this store does not know.
#[must_use]
pub fn dtype_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "float8_e4m3fn" | "float8_e5m2" | "int8" | "uint8" => Some(1),
        "float16" | "bfloat16" | "int16" => Some(2),
        "float32" | "int32" => Some(4),
        "float64" | "int64" => Some(8),
        _ => None,
    }
}

/// What a store holds for some slots of one table of a rank, read together (`Storage::rows`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldRows {
    /// The rank's epoch and step at the read.
    pub epoch: u64,
    /// See `epoch`.
    pub step: u64,
    /// Each slot, the step that wrote it (0 for none), its row's bytes, and the CRC32C recorded
    /// when it was written.
    pub rows: Vec<(i64, u64, Vec<u8>, u32)>,
}

/// Everything held for one rank of one engine.
#[derive(Debug, Default)]
pub struct Rank {
    /// The membership epoch of the last accepted commit.
    pub epoch: u64,
    /// The step of the last accepted commit.
    pub step: u64,
    /// The source's carrier said it has nothing more queued through (`epoch`, `step`); cleared by
    /// every replicated step applied (decision 0030).
    caught_up: bool,
    layers: BTreeMap<u32, Layer>,
    /// The committed token count per request as of this rank's own last commit: this rank's
    /// rows are complete through these counts, whatever other ranks committed since.
    committed: BTreeMap<String, u64>,
    /// Staging files this rank's worker named, mapped read-only on first use.
    stages: HashMap<String, Mmap>,
}

/// One engine: its ranks and the committed token count per request.
#[derive(Debug, Default)]
pub struct Engine {
    ranks: BTreeMap<u32, Rank>,
    committed: BTreeMap<String, u64>,
}

/// A streamed snapshot's start ([`Storage::snapshot_plan`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotPlan {
    /// The epoch the snapshot is labelled with.
    pub epoch: u64,
    /// The step it is labelled with: the rank's last commit when it began.
    pub step: u64,
    /// Each table of the rank and its row size in bytes, in table order.
    pub tables: Vec<(u32, usize)>,
    /// The committed counts as of `step`, carried on the snapshot's last part.
    pub request_ids: Vec<String>,
    /// The count of each of `request_ids`.
    pub committed_tokens: Vec<u64>,
}

/// The whole store: its data directory and memory budget.
#[derive(Debug)]
pub struct Storage {
    data_dir: PathBuf,
    engines: BTreeMap<String, Engine>,
    num_slots: usize,
    budget: usize,
    in_use: usize,
    /// Staging prefixes handed out, one per worker connection that opened a rank.
    links: u64,
}

/// Whether `name` is a staging file's: `stage-*.rows`.
fn is_stage_file(name: &str) -> bool {
    name.starts_with("stage-") && Path::new(name).extension().is_some_and(|e| e == "rows")
}

impl Storage {
    /// The memory budget for row tables, in bytes.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// A store keeping tables under `data_dir`, with `num_slots` slot ids per layer table and
    /// tables totalling at most `budget` bytes.
    #[must_use]
    pub fn new(data_dir: PathBuf, num_slots: usize, budget: usize) -> Self {
        Self {
            data_dir,
            engines: BTreeMap::new(),
            num_slots,
            budget,
            in_use: 0,
            links: 0,
        }
    }

    /// A staging-file prefix no other connection of this store has: `stage-<n>-`. A worker names
    /// every staging file it creates from the prefix its connection was given, so the store can
    /// remove them when that connection ends however the worker ended ([`Storage::end_link`]).
    pub fn stage_prefix(&mut self) -> String {
        self.links += 1;
        format!("stage-{}-", self.links)
    }

    /// A worker connection that opened `rank` of `engine` with `prefix` ended: drop the maps of
    /// its staging files and remove them. Only files named from `prefix` go; another
    /// connection's staging files and the rank's tables stay. Returns how many were removed.
    pub fn end_link(&mut self, engine: &str, rank: u32, prefix: &str) -> usize {
        if let Some(r) = self
            .engines
            .get_mut(engine)
            .and_then(|e| e.ranks.get_mut(&rank))
        {
            r.stages.retain(|name, _| !name.starts_with(prefix));
        }
        let Ok(dir) = self.rank_dir(engine, rank) else {
            return 0;
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return 0;
        };
        let mut removed = 0;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(prefix)
                && is_stage_file(&name)
                && std::fs::remove_file(entry.path()).is_ok()
            {
                removed += 1;
            }
        }
        removed
    }

    /// Remove every staging file under the data directory. Run once at start, before the store
    /// listens: no connection exists yet, and a worker never outlives its store's connection, so
    /// every staging file there is a dead worker's, left when its store was killed.
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] if a directory cannot be read or a file removed.
    pub fn sweep_stages(&self) -> Result<usize, StorageError> {
        let mut removed = 0;
        let engines = match std::fs::read_dir(&self.data_dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(io(format!("read {}", self.data_dir.display()), e)),
        };
        for engine in engines.flatten() {
            let Ok(ranks) = std::fs::read_dir(engine.path()) else {
                continue;
            };
            for rank in ranks.flatten() {
                let Ok(files) = std::fs::read_dir(rank.path()) else {
                    continue;
                };
                for file in files.flatten() {
                    if is_stage_file(&file.file_name().to_string_lossy()) {
                        let path = file.path();
                        std::fs::remove_file(&path)
                            .map_err(|e| io(format!("remove {}", path.display()), e))?;
                        removed += 1;
                    }
                }
            }
        }
        Ok(removed)
    }

    /// Bytes the files under the data directory hold: an earlier run's tables, which this one
    /// makes again at their size in the same place. Run after [`Storage::sweep_stages`].
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] if a directory or a file cannot be read.
    pub fn held_bytes(&self) -> Result<u64, StorageError> {
        fn held(dir: &Path, depth: usize) -> std::io::Result<u64> {
            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
                Err(e) => return Err(e),
            };
            let mut bytes = 0;
            for entry in entries {
                let entry = entry?;
                let kind = entry.file_type()?;
                if kind.is_dir() && depth > 0 {
                    bytes += held(&entry.path(), depth - 1)?;
                } else if kind.is_file() {
                    bytes += entry.metadata()?.len();
                }
            }
            Ok(bytes)
        }
        // engine, rank, file
        held(&self.data_dir, 2).map_err(|e| io(format!("read {}", self.data_dir.display()), e))
    }

    /// Bytes currently allocated to row tables.
    #[must_use]
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// The directory for `rank` of `engine`.
    ///
    /// # Errors
    ///
    /// [`StorageError::EngineName`] for a name that is not one plain path component.
    pub fn rank_dir(&self, engine: &str, rank: u32) -> Result<PathBuf, StorageError> {
        engine_name(engine)?;
        Ok(self.data_dir.join(engine).join(format!("rank{rank}")))
    }

    /// Open `rank` of `engine`: create its directory and forget the staging files an earlier
    /// worker of that rank named. Returns the directory.
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] if the directory cannot be created.
    pub fn open(&mut self, engine: &str, rank: u32) -> Result<PathBuf, StorageError> {
        let dir = self.rank_dir(engine, rank)?;
        std::fs::create_dir_all(&dir).map_err(|e| io(format!("create {}", dir.display()), e))?;
        let r = self
            .engines
            .entry(engine.to_owned())
            .or_default()
            .ranks
            .entry(rank)
            .or_default();
        r.stages.clear();
        Ok(dir)
    }

    /// Create and touch `tables` for `rank` of `engine` with the given row shape, within the
    /// budget, so the first commit into them is as cheap as every later one. Tables that exist
    /// are left alone; a shape that differs from an existing table's is an error.
    ///
    /// # Errors
    ///
    /// See [`StorageError`].
    pub fn prepare(
        &mut self,
        engine: &str,
        rank: u32,
        slots: u64,
        tables: &[u32],
        dtype: &str,
        shape: &[u32],
    ) -> Result<(), StorageError> {
        let num_slots = self.num_slots;
        // what the engine writes must fit before anything is made: an engine that states no
        // count is refused too, since its first busy step could write any slot
        if slots == 0 || slots > u64::try_from(num_slots).unwrap_or(u64::MAX) {
            return Err(StorageError::Undersized {
                engine: engine.to_owned(),
                rank,
                wants: slots,
                holds: num_slots,
            });
        }
        let budget = self.budget;
        let rank_dir = self.rank_dir(engine, rank)?;
        std::fs::create_dir_all(&rank_dir)
            .map_err(|e| io(format!("create {}", rank_dir.display()), e))?;
        let Some(elem_bytes) = dtype_bytes(dtype) else {
            return Err(StorageError::Dtype {
                layer: tables.first().copied().unwrap_or(0),
                dtype: dtype.to_owned(),
            });
        };
        let row_bytes = row_bytes_of(tables.first().copied().unwrap_or(0), shape, elem_bytes)?;
        let r = self
            .engines
            .entry(engine.to_owned())
            .or_default()
            .ranks
            .entry(rank)
            .or_default();
        let mut missing = Vec::new();
        for &t in tables {
            match r.layers.get(&t) {
                Some(layer) if layer.shape != shape || layer.dtype != dtype => {
                    return Err(StorageError::Shape {
                        layer: t,
                        got: shape.to_vec(),
                        got_dtype: dtype.to_owned(),
                        expected: layer.shape.clone(),
                        expected_dtype: layer.dtype.clone(),
                    });
                }
                Some(_) => {}
                None => missing.push(t),
            }
        }
        missing.sort_unstable();
        missing.dedup();
        let needed = self.in_use + missing.len() * row_bytes * num_slots;
        if needed > budget {
            return Err(StorageError::Budget { budget, needed });
        }
        for t in missing {
            let path = rank_dir.join(format!("table{t}.rows"));
            r.layers
                .insert(t, Layer::create(path, shape, dtype, row_bytes, num_slots)?);
            self.in_use += row_bytes * num_slots;
        }
        Ok(())
    }

    /// Apply one commit: the descriptor is validated for every table it names before any row is
    /// written, so a refused commit changes nothing. `staged` is `None` for a step that wrote no
    /// rows (counts only).
    ///
    /// # Errors
    ///
    /// See [`StorageError`].
    pub fn commit(
        &mut self,
        engine: &str,
        rank: u32,
        epoch: u64,
        step: u64,
        staged: Option<&StagedStep>,
        committed: &[(String, u64)],
    ) -> Result<(), StorageError> {
        let num_slots = self.num_slots;
        let budget = self.budget;
        let rank_dir = self.rank_dir(engine, rank)?;
        let mut in_use = self.in_use;
        let eng = self.engines.entry(engine.to_owned()).or_default();
        let r = eng.ranks.entry(rank).or_default();
        if epoch < r.epoch {
            return Err(StorageError::StaleEpoch {
                got: epoch,
                current: r.epoch,
            });
        }
        let tables = staged.map(derive).transpose()?.unwrap_or_default();
        let (new_layers, added) = Self::validate(r, &rank_dir, num_slots, staged, &tables)?;
        in_use += added;
        if in_use > budget {
            return Err(StorageError::Budget {
                budget,
                needed: in_use,
            });
        }
        // create the new tables, all or none, then write
        if let Some(b) = staged {
            let wanted: Vec<LayerSpec> = tables
                .iter()
                .filter(|t| !r.layers.contains_key(&t.table))
                .filter_map(|t| {
                    new_layers.get(&t.table).map(|&row_bytes| LayerSpec {
                        table: t.table,
                        shape: b.shape.clone(),
                        dtype: b.dtype.clone(),
                        row_bytes,
                    })
                })
                .collect();
            for (table, layer) in create_layers(&rank_dir, &wanted, num_slots)? {
                r.layers.insert(table, layer);
            }
            let Some(stage) = r.stages.get(&b.stage) else {
                return Err(StorageError::StageName {
                    stage: b.stage.clone(),
                });
            };
            for t in &tables {
                let Some(layer) = r.layers.get_mut(&t.table) else {
                    continue;
                };
                let row_bytes = layer.row_bytes;
                let base = usize::try_from(t.offset).unwrap_or(usize::MAX);
                // consecutive slots at consecutive rows copy as one run: a prefill is mostly
                // whole blocks, so this is a memcpy per block, not per row
                let mut i = 0;
                while i < b.slots.len() {
                    let Ok(s) = usize::try_from(b.slots[i]) else {
                        i += 1; // padding
                        continue;
                    };
                    let mut n = 1;
                    while i + n < b.slots.len()
                        && usize::try_from(b.slots[i + n]).is_ok_and(|next| next == s + n)
                    {
                        n += 1;
                    }
                    let src = base + i * row_bytes;
                    layer.rows[s * row_bytes..(s + n) * row_bytes]
                        .copy_from_slice(&stage[src..src + n * row_bytes]);
                    for k in 0..n {
                        layer.mark_written(s + k, step);
                    }
                    i += n;
                }
            }
        }
        r.epoch = epoch;
        r.step = step;
        for (id, n) in committed {
            eng.committed.insert(id.clone(), *n);
            r.committed.insert(id.clone(), *n);
        }
        self.in_use = in_use;
        Ok(())
    }

    /// Check a descriptor against the rank's tables and staging file before anything is written:
    /// the dtype, the slot ids, the shape of every table it names and the file's length. Returns
    /// the tables not yet created and the memory they would take.
    fn validate(
        r: &mut Rank,
        rank_dir: &Path,
        num_slots: usize,
        staged: Option<&StagedStep>,
        tables: &[StagedTable],
    ) -> Result<(BTreeMap<u32, usize>, usize), StorageError> {
        let mut added = 0;
        let mut new_layers: BTreeMap<u32, usize> = BTreeMap::new();
        if let Some(b) = staged {
            let Some(elem_bytes) = dtype_bytes(&b.dtype) else {
                return Err(StorageError::Dtype {
                    layer: tables.first().map_or(0, |t| t.table),
                    dtype: b.dtype.clone(),
                });
            };
            let layer_named = tables.first().map_or(0, |t| t.table);
            let row_bytes = row_bytes_of(layer_named, &b.shape, elem_bytes)?;
            for &s in &b.slots {
                if s >= 0 && usize::try_from(s).is_ok_and(|s| s >= num_slots) {
                    return Err(StorageError::Slot {
                        layer: tables.first().map_or(0, |t| t.table),
                        slot: s,
                        slots: num_slots,
                    });
                }
            }
            let stage_len = Self::stage(rank_dir, &mut r.stages, &b.stage)?.len();
            let needed =
                b.slots
                    .len()
                    .checked_mul(row_bytes)
                    .ok_or_else(|| StorageError::RowSize {
                        layer: layer_named,
                        shape: b.shape.clone(),
                    })?;
            for t in tables {
                if let Some(layer) = r.layers.get(&t.table) {
                    if layer.shape != b.shape || layer.dtype != b.dtype {
                        return Err(StorageError::Shape {
                            layer: t.table,
                            got: b.shape.clone(),
                            got_dtype: b.dtype.clone(),
                            expected: layer.shape.clone(),
                            expected_dtype: layer.dtype.clone(),
                        });
                    }
                } else if new_layers.insert(t.table, row_bytes).is_none() {
                    added += row_bytes * num_slots;
                }
                let offset = usize::try_from(t.offset).unwrap_or(usize::MAX);
                if offset.saturating_add(needed) > stage_len {
                    return Err(StorageError::Size {
                        layer: t.table,
                        stage: b.stage.clone(),
                        bytes: stage_len,
                        needed,
                        offset: t.offset,
                    });
                }
            }
        }
        Ok((new_layers, added))
    }

    /// The staging file `name` in `rank_dir`, mapped on first use.
    fn stage<'a>(
        rank_dir: &Path,
        stages: &'a mut HashMap<String, Mmap>,
        name: &str,
    ) -> Result<&'a Mmap, StorageError> {
        if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
            return Err(StorageError::StageName {
                stage: name.to_owned(),
            });
        }
        if !stages.contains_key(name) {
            let path = rank_dir.join(name);
            let file =
                File::open(&path).map_err(|e| io(format!("open stage {}", path.display()), e))?;
            let map = shm::map_stage(&file)
                .map_err(|e| io(format!("map stage {}", path.display()), e))?;
            stages.insert(name.to_owned(), map);
        }
        Ok(&stages[name])
    }

    /// What a commit of `staged` at `step` wrote for `rank`, read back from the tables as a
    /// replica receives it; `None` if the rank is unknown.
    #[must_use]
    pub fn replicated(
        &self,
        engine: &str,
        rank: u32,
        epoch: u64,
        step: u64,
        staged: Option<&StagedStep>,
        committed: &[(String, u64)],
    ) -> Option<Replicated> {
        let r = self.engines.get(engine)?.ranks.get(&rank)?;
        let mut tables = Vec::new();
        if let Some(b) = staged {
            let slots: Vec<i64> = b.slots.iter().copied().filter(|s| *s >= 0).collect();
            for t in derive(b).ok()? {
                let Some(layer) = r.layers.get(&t.table) else {
                    continue;
                };
                let mut data = Vec::with_capacity(slots.len() * layer.row_bytes);
                let mut crc32c = Vec::with_capacity(slots.len());
                for &s in &slots {
                    let s = usize::try_from(s).unwrap_or(0);
                    data.extend_from_slice(layer.row(s));
                    crc32c.push(layer.crc(s));
                }
                tables.push(ReplicatedTable {
                    table: t.table,
                    dtype: layer.dtype.clone(),
                    shape: layer.shape.clone(),
                    slots: slots.clone(),
                    data,
                    crc32c,
                });
            }
        }
        Some(Replicated {
            engine: engine.to_owned(),
            rank,
            epoch,
            step,
            tables,
            request_ids: committed.iter().map(|(id, _)| id.clone()).collect(),
            committed_tokens: committed.iter().map(|(_, n)| *n).collect(),
        })
    }

    /// Apply a step replicated from another store: create the tables it names within the budget,
    /// write the rows with `step` as their write step, and record the counts.
    ///
    /// # Errors
    ///
    /// See [`StorageError`]. A step older than the rank's last is refused as stale.
    pub fn apply_replicated(&mut self, msg: &Replicated) -> Result<(), StorageError> {
        let num_slots = self.num_slots;
        let budget = self.budget;
        let rank_dir = self.rank_dir(&msg.engine, msg.rank)?;
        std::fs::create_dir_all(&rank_dir)
            .map_err(|e| io(format!("create {}", rank_dir.display()), e))?;
        let in_use = self.in_use;
        let eng = self.engines.entry(msg.engine.clone()).or_default();
        let r = eng.ranks.entry(msg.rank).or_default();
        if msg.epoch < r.epoch || (msg.epoch == r.epoch && msg.step < r.step) {
            return Err(StorageError::StaleEpoch {
                got: msg.epoch,
                current: r.epoch,
            });
        }
        r.caught_up = false; // rows are arriving: whatever the carrier said before, it has more
        // validate every table first
        let (new_tables, added) = validate_replicated(r, msg, num_slots)?;
        let in_use = in_use.saturating_add(added);
        if in_use > budget {
            return Err(StorageError::Budget {
                budget,
                needed: in_use,
            });
        }
        for (table, layer) in create_layers(&rank_dir, &new_tables, num_slots)? {
            r.layers.insert(table, layer);
        }
        for t in &msg.tables {
            let Some(layer) = r.layers.get_mut(&t.table) else {
                continue;
            };
            let row_bytes = layer.row_bytes;
            for (i, &s) in t.slots.iter().enumerate() {
                let s = usize::try_from(s).unwrap_or(0);
                layer.put(
                    s,
                    &t.data[i * row_bytes..(i + 1) * row_bytes],
                    msg.step,
                    t.crc32c[i],
                );
            }
        }
        r.epoch = msg.epoch;
        r.step = msg.step;
        for (id, n) in msg.request_ids.iter().zip(msg.committed_tokens.iter()) {
            eng.committed.insert(id.clone(), *n);
            r.committed.insert(id.clone(), *n);
        }
        self.in_use = in_use;
        Ok(())
    }

    /// What this store holds for `slots` of `table` of `rank`: the rank's epoch and step, and
    /// each slot's written step (0 when nothing wrote it) and row bytes, read together. None
    /// when the rank or the table is unknown here. A diagnostic; the serving path never calls it.
    #[must_use]
    pub fn rows(&self, engine: &str, rank: u32, table: u32, slots: &[i64]) -> Option<HeldRows> {
        let r = self.engines.get(engine)?.ranks.get(&rank)?;
        let layer = r.layers.get(&table)?;
        let rows = slots
            .iter()
            .filter_map(|&s| {
                let i = usize::try_from(s).ok()?;
                (i < layer.written_step.len()).then(|| {
                    (
                        s,
                        layer.written_step[i],
                        layer.row(i).to_vec(),
                        layer.crc(i),
                    )
                })
            })
            .collect();
        Some(HeldRows {
            epoch: r.epoch,
            step: r.step,
            rows,
        })
    }

    /// Where this store stands for `rank`: whether it holds any row of it, and the epoch and
    /// step of its last accepted commit or replicated step; `(false, 0, 0)` for a rank it has
    /// never held. A rank whose tables are prepared but unwritten holds nothing.
    /// The fourth element says whether the source's carrier has told this store it is caught up
    /// at that step (decision 0030).
    #[must_use]
    pub fn state(&self, engine: &str, rank: u32) -> (bool, u64, u64, bool) {
        self.engines
            .get(engine)
            .and_then(|e| e.ranks.get(&rank))
            .map_or((false, 0, 0, false), |r| {
                let held = r.layers.values().any(Layer::holds_rows);
                (held, r.epoch, r.step, r.caught_up)
            })
    }

    /// The source's carrier has nothing more queued for `rank` through (`epoch`, `step`): mark
    /// the rank caught up when that is where this store stands, and say whether it was. A marker
    /// for another step is stale (rows moved on, or have yet to arrive) and leaves the mark clear.
    pub fn mark_caught_up(&mut self, engine: &str, rank: u32, epoch: u64, step: u64) -> bool {
        let Some(r) = self
            .engines
            .get_mut(engine)
            .and_then(|e| e.ranks.get_mut(&rank))
        else {
            return false;
        };
        r.caught_up = (r.epoch, r.step) == (epoch, step);
        r.caught_up
    }

    /// The start of a streamed snapshot of `rank` (decision 0030's resync): the (epoch, step) it
    /// is labelled with, the committed counts as of that step, and each table with its row size,
    /// all taken at once under the caller's lock. None when nothing is written yet: a snapshot of
    /// no rows must not count as covering the step it is taken at, since the first commit is
    /// step 0 too.
    ///
    /// The rows are read afterwards, a run at a time ([`Storage::snapshot_run`]), each under the
    /// lock only while it is copied, so the store's commits go on between runs. A run can then
    /// hold a row a later commit wrote: the snapshot is fuzzy, and exact once every step after its
    /// label has been applied over it, which the carrier's ordered queue does before it tells the
    /// replica it is caught up.
    #[must_use]
    pub fn snapshot_plan(&self, engine: &str, rank: u32) -> Option<SnapshotPlan> {
        let r = self.engines.get(engine)?.ranks.get(&rank)?;
        if !r.layers.values().any(Layer::holds_rows) {
            return None;
        }
        let committed = self.committed_of(engine, rank);
        Some(SnapshotPlan {
            epoch: r.epoch,
            step: r.step,
            tables: r.layers.iter().map(|(t, l)| (*t, l.row_bytes)).collect(),
            request_ids: committed.iter().map(|(id, _)| id.clone()).collect(),
            committed_tokens: committed.iter().map(|(_, n)| *n).collect(),
        })
    }

    /// Up to `max_rows` written rows of `table` of `rank`, from slot `from` on, each with the
    /// checksum recorded at its write, and the slot to go on from (None once the table is done);
    /// None when the rank or the table is unknown.
    #[must_use]
    pub fn snapshot_run(
        &self,
        engine: &str,
        rank: u32,
        table: u32,
        from: usize,
        max_rows: usize,
    ) -> Option<(ReplicatedTable, Option<usize>)> {
        let layer = self
            .engines
            .get(engine)?
            .ranks
            .get(&rank)?
            .layers
            .get(&table)?;
        let mut slots = Vec::new();
        let mut next = None;
        for s in from..layer.written_step.len() {
            if layer.written_step[s] == 0 {
                continue;
            }
            if slots.len() == max_rows.max(1) {
                next = Some(s);
                break;
            }
            slots.push(s);
        }
        let mut data = Vec::with_capacity(slots.len() * layer.row_bytes);
        for &s in &slots {
            data.extend_from_slice(layer.row(s));
        }
        let run = ReplicatedTable {
            table,
            dtype: layer.dtype.clone(),
            shape: layer.shape.clone(),
            // the checksums recorded at the write, not taken now: a row damaged in this store's
            // memory since is caught at the replica rather than blessed here
            crc32c: slots.iter().map(|&s| layer.crc(s)).collect(),
            slots: slots
                .into_iter()
                .map(|s| i64::try_from(s).unwrap_or(i64::MAX))
                .collect(),
            data,
        };
        Some((run, next))
    }

    /// Every table of `rank` with the slots written after `since_step` (0: all written slots),
    /// in layer order, and the rank's last committed step; empty and 0 if unknown.
    #[must_use]
    pub fn restore(&self, engine: &str, rank: u32, since_step: u64) -> (Vec<TableRef>, u64) {
        self.engines
            .get(engine)
            .and_then(|e| e.ranks.get(&rank))
            .map(|r| {
                (
                    r.layers
                        .iter()
                        .map(|(l, layer)| layer.table_ref(*l, since_step))
                        .collect(),
                    r.step,
                )
            })
            .unwrap_or_default()
    }

    /// The layer tables of `rank`, for a reader in this process (tests, replication).
    #[must_use]
    pub fn layers(&self, engine: &str, rank: u32) -> Vec<(u32, &Layer)> {
        self.engines
            .get(engine)
            .and_then(|e| e.ranks.get(&rank))
            .map(|r| r.layers.iter().map(|(l, layer)| (*l, layer)).collect())
            .unwrap_or_default()
    }

    /// The committed token count of each request; zero for one never seen.
    #[must_use]
    pub fn committed(&self, engine: &str, request_ids: &[String]) -> Vec<u64> {
        let table = self.engines.get(engine).map(|e| &e.committed);
        request_ids
            .iter()
            .map(|id| table.and_then(|t| t.get(id)).copied().unwrap_or(0))
            .collect()
    }

    /// Every request's committed count as of `rank`'s own last commit, in request-id order.
    #[must_use]
    pub fn committed_of(&self, engine: &str, rank: u32) -> Vec<(String, u64)> {
        self.engines
            .get(engine)
            .and_then(|e| e.ranks.get(&rank))
            .map(|r| r.committed.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .unwrap_or_default()
    }

    /// The scheduler evicted `request_id`'s blocks beyond `kept_tokens`: lower, never raise.
    pub fn evicted(&mut self, engine: &str, request_id: &str, kept_tokens: u64) {
        if let Some(e) = self.engines.get_mut(engine) {
            let ranks = e.ranks.values_mut().map(|r| &mut r.committed);
            for table in std::iter::once(&mut e.committed).chain(ranks) {
                if let Some(n) = table.get_mut(request_id) {
                    *n = (*n).min(kept_tokens);
                }
            }
        }
    }

    /// Forget finished requests.
    pub fn finished(&mut self, engine: &str, request_ids: &[String]) {
        if let Some(e) = self.engines.get_mut(engine) {
            for id in request_ids {
                e.committed.remove(id);
                for r in e.ranks.values_mut() {
                    r.committed.remove(id);
                }
            }
        }
    }
}
