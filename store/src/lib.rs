// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The KV mirror service: committed rows in with their epoch, rows out on restore, the last
//! committed token per request. The production implementation of `everserve.state.CommittedKV`
//! (`docs/RUNTIME.md` section 3, `docs/CONTRACT.md` item 3).
//!
//! This crate holds the service and its storage. The control plane is the shared frame protocol
//! over a Unix socket; the data plane is shared memory under the store's data directory, where
//! workers stage rows and the store keeps its tables (decision 0011). Cross-node replication of
//! the tables to a restoring node lands with the two-node run and leaves the worker side as it is.
//! Memory is bounded: a commit that would exceed the budget is refused with an error the worker
//! can act on, never silently dropped.

pub mod cache;
pub mod metrics;
pub mod replication;
pub mod service;
pub mod shm;
pub mod storage;
