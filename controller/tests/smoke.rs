// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The binary builds and runs.

use std::process::Command;

#[test]
fn binary_prints_its_version() {
    let out = Command::new(env!("CARGO_BIN_EXE_everserve-controller"))
        .arg("--version")
        .output()
        .expect("run binary");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains(env!("CARGO_PKG_VERSION")), "got: {text}");
    // no subcommand is a usage error, not a crash
    let out = Command::new(env!("CARGO_BIN_EXE_everserve-controller"))
        .output()
        .expect("run binary");
    assert_eq!(out.status.code(), Some(2));
}
