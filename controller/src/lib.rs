// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The Everserve control plane as a library: the wire schema, the policy document and the
//! outcome decision, the node daemon and the coordinator, and the lease the coordinator holds; the
//! transports they speak over are the wire crate's. The binary in `main.rs` is a thin CLI
//! over this crate, so every behaviour is testable without a process.

pub mod api;
pub mod certs;
pub mod check;
pub mod cli;
pub mod clock;
pub mod config;
pub mod coord;
pub mod discovery;
pub mod enroll;
pub mod http;
pub mod issuer;
pub mod kubernetes;
pub mod lease;
pub mod metrics;
pub mod node;
pub mod notice;
pub mod operator;
pub mod policy;
pub mod readiness;
pub mod spec;
pub mod status;
pub mod topology;

pub use everserve_wire::proto;
