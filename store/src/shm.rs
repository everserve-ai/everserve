// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The two memory maps the store makes, and the only `unsafe` in the workspace.
//!
//! Mapping a file is unsafe in Rust because another process can change the file under the map.
//! The protocol makes that impossible for the ranges the store touches (decision 0011):
//!
//! - A **table** is created, sized and written only by the store. A restoring worker maps it
//!   read-only after a restore reply names it, and reads only slots the reply lists as written.
//!   The store writes a slot again only on a later commit by the table's own rank, and a restore
//!   is taken when that rank is dead, so a reader and the writer never overlap on a slot.
//! - A **staging file** is created and sized once by the worker and never shrunk; the worker
//!   writes a region, then names it in a commit, and writes it again only after the commit is
//!   acknowledged. The store reads only the named region, bounded by the size check in
//!   [`crate::storage::Storage::commit`].
//!
//! Everything else in the store is safe code; `unsafe_code` stays denied workspace-wide.
#![allow(unsafe_code)]

use std::fs::File;

use memmap2::{Mmap, MmapMut};

/// Map `file` writable, for a table the store owns.
///
/// # Errors
///
/// The map error.
pub fn map_table(file: &File) -> std::io::Result<MmapMut> {
    // SAFETY: see the module documentation; the store is the file's only writer.
    unsafe { MmapMut::map_mut(file) }
}

/// Map `file` read-only, for a staging file a worker owns.
///
/// # Errors
///
/// The map error.
pub fn map_stage(file: &File) -> std::io::Result<Mmap> {
    // Populated at map time: the pages are faulted in as one kernel operation rather than one
    // fault per page during the first commit's copy (decision 0020).
    // SAFETY: see the module documentation; the worker never shrinks or rewrites a named region
    // before its acknowledgement.
    unsafe { memmap2::MmapOptions::new().populate().map(file) }
}
