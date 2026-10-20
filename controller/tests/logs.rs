// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A process's log lines go to standard error, one JSON object each when asked for, and a
//! command's output alone to standard output.

use std::process::Command;

fn failing_get(format: &str) -> std::process::Output {
    // a credential that cannot be read: the command logs why and exits 2
    Command::new(env!("CARGO_BIN_EXE_everserve-controller"))
        .args([
            "--log-format",
            format,
            "get",
            "engines",
            "--coordinator-api",
            "127.0.0.1:1",
            "--ca",
            "/nonexistent/ca.crt",
            "--cert",
            "/nonexistent/operator.crt",
            "--key",
            "/nonexistent/operator.key",
        ])
        .env_remove("EVERSERVE_LOG")
        .output()
        .expect("run")
}

#[test]
fn log_lines_are_one_json_object_each_on_standard_error() {
    let out = failing_get("json");
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(
        out.stdout.is_empty(),
        "logs stay off standard output: {out:?}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let lines: Vec<&str> = stderr.lines().filter(|l| !l.is_empty()).collect();
    assert!(!lines.is_empty(), "the failure is logged");
    for line in lines {
        let event: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
        assert_eq!(event["level"], "ERROR", "{line}");
        assert!(
            event["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("credentials")),
            "{line}"
        );
    }
}

#[test]
fn text_is_for_people_and_an_unknown_format_is_refused() {
    let out = failing_get("text");
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ERROR") && stderr.contains("credentials") && !stderr.starts_with('{'),
        "{stderr}"
    );
    let out = failing_get("yaml");
    assert_eq!(out.status.code(), Some(2), "clap refuses it: {out:?}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("text or json"));
}
