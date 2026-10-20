// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The wire schema, generated from `proto/` at build time by this crate's `build.rs`. Do not edit;
//! edit the `.proto` files.
//!
//! The module tree mirrors the protobuf packages (`everserve.signals.v1` is
//! `proto::everserve::signals::v1`) because the generated code refers across packages by
//! relative path.

#[allow(missing_docs, clippy::all, clippy::pedantic)]
pub mod everserve {
    pub mod signals {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/everserve.signals.v1.rs"));
            include!(concat!(env!("OUT_DIR"), "/everserve.signals.v1.serde.rs"));
        }
    }
    pub mod mirror {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/everserve.mirror.v1.rs"));
            include!(concat!(env!("OUT_DIR"), "/everserve.mirror.v1.serde.rs"));
        }
    }
    pub mod policy {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/everserve.policy.v1.rs"));
            include!(concat!(env!("OUT_DIR"), "/everserve.policy.v1.serde.rs"));
        }
    }
    pub mod discovery {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/everserve.discovery.v1.rs"));
            include!(concat!(env!("OUT_DIR"), "/everserve.discovery.v1.serde.rs"));
        }
    }
    pub mod spec {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/everserve.spec.v1.rs"));
            include!(concat!(env!("OUT_DIR"), "/everserve.spec.v1.serde.rs"));
        }
    }
    pub mod coord {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/everserve.coord.v1.rs"));
            include!(concat!(env!("OUT_DIR"), "/everserve.coord.v1.serde.rs"));
        }
    }
}
