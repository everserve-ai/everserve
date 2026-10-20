// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The Everserve control plane binary. See the crate's `lib.rs` and README.

fn main() -> std::process::ExitCode {
    everserve_controller::cli::run()
}
