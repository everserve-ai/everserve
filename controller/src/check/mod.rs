// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! `everserve-controller check`: is the control plane ready for the engines the policy names?
//!
//! The policy says which nodes hold members and which hold spares. The coordinator's `/v1/nodes`
//! says which nodes' daemons are linked, their devices and where their stores take replicas.
//! The check compares the two and waits, up to a deadline, for every policy node to be connected,
//! every member's device to exist where the node reported devices, and every spare node's store
//! to advertise a replica address. It is what an operator runs after the install and what a
//! bare-metal install's hosts run before an engine starts (`deploy/baremetal/README.md`).

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use crate::api::objects::{List, Node};
use crate::config::CheckArgs;
use crate::policy;
use crate::proto::everserve::policy::v1::Policy;

/// What the policy needs of one node.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Need {
    /// Device indices members run on.
    pub devices: BTreeSet<u32>,
    /// The node takes a replica of some engine's rows (decision 0030): its store must take
    /// replicas.
    pub takes_replicas: bool,
}

/// What the policy needs of every node.
#[must_use]
pub fn needs(policy: &Policy) -> BTreeMap<String, Need> {
    let mut out: BTreeMap<String, Need> = BTreeMap::new();
    for engine in &policy.engines {
        for m in &engine.members {
            out.entry(m.node.clone())
                .or_default()
                .devices
                .insert(m.device);
        }
        for n in policy::replica_nodes(engine) {
            out.entry(n).or_default().takes_replicas = true;
        }
    }
    out
}

/// The comparison of the policy's needs with the connected nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// Every need is met.
    pub ok: bool,
    /// One line per policy node, in name order.
    pub lines: Vec<String>,
}

/// Compare `needs` with `nodes`.
#[must_use]
pub fn verdict(needs: &BTreeMap<String, Need>, nodes: &[Node]) -> Report {
    let mut ok = true;
    let mut lines = Vec::new();
    for (name, need) in needs {
        let Some(node) = nodes
            .iter()
            .find(|n| &n.metadata.name == name && n.status.linked)
            .map(|n| &n.status)
        else {
            ok = false;
            lines.push(format!("{name}: daemon not connected"));
            continue;
        };
        let mut faults = Vec::new();
        if !node.devices.is_empty() {
            let have: BTreeSet<u32> = node.devices.iter().map(|d| d.index).collect();
            let missing: Vec<String> = need
                .devices
                .difference(&have)
                .map(ToString::to_string)
                .collect();
            if !missing.is_empty() {
                faults.push(format!("devices {} not reported", missing.join(", ")));
            }
        }
        if need.takes_replicas && node.replica_address.is_none() {
            faults.push("spare node, but its store advertises no replica address".to_owned());
        }
        if faults.is_empty() {
            let devices = if node.devices.is_empty() {
                "no devices reported".to_owned()
            } else {
                format!("{} devices", node.devices.len())
            };
            let replicas = node
                .replica_address
                .as_deref()
                .map_or(String::new(), |a| format!(", replicas at {a}"));
            lines.push(format!("{name}: connected, {devices}{replicas}"));
        } else {
            ok = false;
            lines.push(format!("{name}: {}", faults.join("; ")));
        }
    }
    Report { ok, lines }
}

/// Why the check could not run.
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    /// The policy could not be loaded.
    #[error(transparent)]
    Policy(#[from] crate::policy::PolicyError),
    /// The coordinator could not be reached or answered badly.
    #[error("coordinator: {0}")]
    Coordinator(String),
}

/// Ask the coordinator until every need is met or the deadline passes; the last report either
/// way.
///
/// # Errors
///
/// [`CheckError`] when the policy cannot be loaded or the coordinator never answered.
pub async fn run(args: &CheckArgs) -> Result<Report, CheckError> {
    let policy = crate::policy::load(&args.policy)?;
    let needs = needs(&policy);
    let operator = crate::operator::Operator::connect(&args.api)
        .map_err(|e| CheckError::Coordinator(e.to_string()))?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(args.timeout_secs);
    let mut last_error = None;
    loop {
        match operator.get::<List<Node>>("/v1/nodes").await {
            Ok(nodes) => {
                let report = verdict(&needs, &nodes.items);
                if report.ok || tokio::time::Instant::now() >= deadline {
                    return Ok(report);
                }
            }
            Err(e) => last_error = Some(format!("/v1/nodes: {e}")),
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(CheckError::Coordinator(
                last_error.unwrap_or_else(|| "no answer".into()),
            ));
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::{Need, needs, verdict};
    use crate::api::objects::{Metadata, Node, NodeSpec, NodeStatus};
    use crate::topology::Device;

    fn device(node: &str, index: u32) -> Device {
        Device {
            node: node.into(),
            index,
            uuid: format!("GPU-{node}-{index}"),
            class: "H100".into(),
            memory_bytes: 80 << 30,
            domain: node.into(),
        }
    }

    fn connected(node: &str, devices: u32, replicas: Option<&str>) -> Node {
        Node {
            metadata: Metadata {
                name: node.into(),
                ..Metadata::default()
            },
            spec: NodeSpec::default(),
            status: NodeStatus {
                linked: true,
                devices: (0..devices).map(|i| device(node, i)).collect(),
                replica_address: replicas.map(str::to_owned),
                engines: Vec::new(),
                conditions: Vec::new(),
            },
        }
    }

    #[test]
    fn the_two_node_policy_needs_two_serving_nodes_and_a_spare_node() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/examples/policy-two-node.json");
        let n = needs(&crate::policy::load(&path).unwrap());
        assert_eq!(n.len(), 3);
        assert_eq!(n["node-a"].devices.len(), 4);
        assert!(!n["node-a"].takes_replicas);
        assert_eq!(
            n["node-c"],
            Need {
                devices: std::collections::BTreeSet::default(),
                takes_replicas: true
            }
        );
    }

    #[test]
    fn every_need_met_is_ok_and_each_fault_is_named() {
        let mut n = std::collections::BTreeMap::new();
        n.insert(
            "node-a".to_owned(),
            Need {
                devices: (0..4).collect(),
                takes_replicas: false,
            },
        );
        n.insert(
            "node-c".to_owned(),
            Need {
                devices: std::collections::BTreeSet::default(),
                takes_replicas: true,
            },
        );
        let ok = verdict(
            &n,
            &[
                connected("node-a", 4, None),
                connected("node-c", 4, Some("node-c:7001")),
            ],
        );
        assert!(ok.ok, "{ok:?}");
        assert!(ok.lines[1].contains("replicas at node-c:7001"), "{ok:?}");
        // a node without a driver reports no devices and is taken at the policy's word
        assert!(
            verdict(
                &n,
                &[
                    connected("node-a", 0, None),
                    connected("node-c", 0, Some("x"))
                ]
            )
            .ok
        );
        let bad = verdict(
            &n,
            &[connected("node-a", 2, None), connected("node-c", 4, None)],
        );
        assert!(!bad.ok);
        assert!(
            bad.lines[0].contains("devices 2, 3 not reported"),
            "{bad:?}"
        );
        assert!(bad.lines[1].contains("no replica address"), "{bad:?}");
        let missing = verdict(&n, &[connected("node-c", 4, Some("x"))]);
        assert!(!missing.ok);
        assert_eq!(missing.lines[0], "node-a: daemon not connected");
    }
}
