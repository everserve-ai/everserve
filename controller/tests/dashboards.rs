// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The dashboard and the alerting rules (`deploy/grafana`) name only series the control plane
//! exports, and match only the label values it writes: a metric renamed in the code, or a rule
//! written against one that never existed, fails here rather than staying silent on a cluster;
//! and every alert points at its section of the runbook.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Where an alert's runbook section is read: docs/RUNBOOK.md on the default branch.
const RUNBOOK_URL: &str = "https://github.com/everserve-ai/everserve/blob/main/docs/RUNBOOK.md";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Every Rust source under `dir`, its text.
fn sources(dir: &Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("dir").flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(std::fs::read_to_string(&path).expect("source"));
        }
    }
}

/// The words of `text` that start with `prefix` and go on in `[a-z0-9_]`.
fn words(text: &str, prefix: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = text;
    while let Some(at) = rest.find(prefix) {
        let tail = &rest[at..];
        let end = prefix.len()
            + tail[prefix.len()..]
                .find(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'))
                .unwrap_or(tail.len() - prefix.len());
        out.insert(tail[..end].to_owned());
        rest = &tail[end..];
    }
    out
}

/// The metric names the code registers: every `"everserve_..."` literal of the crates that export.
fn exported() -> BTreeSet<String> {
    let mut texts = Vec::new();
    for krate in ["controller/src", "store/src", "telemetry/src"] {
        sources(&repo().join(krate), &mut texts);
    }
    texts
        .iter()
        .flat_map(|t| words(t, "\"everserve_"))
        .map(|w| w.trim_start_matches('"').to_owned())
        .collect()
}

/// `name` without the suffixes Prometheus gives a histogram's or a summary's series.
fn family(name: &str) -> &str {
    ["_bucket", "_sum", "_count"]
        .iter()
        .find_map(|s| name.strip_suffix(s))
        .unwrap_or(name)
}

#[test]
fn the_dashboard_and_the_alerts_name_only_what_is_exported() {
    let exported = exported();
    assert!(exported.contains("everserve_engine_down"), "{exported:?}");
    for file in ["deploy/grafana/everserve.json", "deploy/grafana/alerts.yml"] {
        let text = std::fs::read_to_string(repo().join(file)).expect(file);
        let named = words(&text, "everserve_");
        assert!(!named.is_empty(), "{file} names no series");
        for name in &named {
            assert!(
                exported.contains(family(name)),
                "{file} names {name}, which nothing exports"
            );
        }
    }
}

#[test]
fn the_alerts_match_only_the_label_values_the_code_writes() {
    let rules = std::fs::read_to_string(repo().join("deploy/grafana/alerts.yml")).expect("rules");
    // a condition's type and a change's or a spec write's result, as the controller spells them,
    // and a store refusal's reason, as the store does
    for (label, krate) in [
        ("type=\"", "controller/src"),
        ("result=\"", "controller/src"),
        ("result!~\"", "controller/src"),
        ("reason=~\"", "store/src"),
        ("reason!~\"", "store/src"),
    ] {
        let mut texts = Vec::new();
        sources(&repo().join(krate), &mut texts);
        let code = texts.concat();
        let mut rest = rules.as_str();
        while let Some(at) = rest.find(label) {
            let tail = &rest[at + label.len()..];
            let end = tail.find('"').expect("a closing quote");
            for value in tail[..end].split('|') {
                assert!(
                    code.contains(&format!("\"{value}\"")),
                    "the alerts match {label}{value}\", which the code never writes"
                );
            }
            rest = &tail[end..];
        }
    }
}

#[test]
fn every_alert_points_at_its_runbook_section() {
    // an alert that pages someone says where its runbook is, and that section exists
    let rules = std::fs::read_to_string(repo().join("deploy/grafana/alerts.yml")).expect("rules");
    let runbook = std::fs::read_to_string(repo().join("docs/RUNBOOK.md")).expect("runbook");
    let rules_of: Vec<&str> = rules.split("- alert: ").skip(1).collect();
    assert!(!rules_of.is_empty(), "no alerts read");
    for rule in rules_of {
        let name = rule.split_whitespace().next().expect("a name");
        let url = format!("runbook_url: \"{RUNBOOK_URL}#{}\"", name.to_lowercase());
        assert!(rule.contains(&url), "{name} has no {url}");
        assert!(
            runbook.lines().any(|l| l == format!("### {name}")),
            "docs/RUNBOOK.md has no section for {name}"
        );
    }
}
