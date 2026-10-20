// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What the cluster's Nodes tell the coordinator (decision 0041), on Kubernetes: which nodes are
//! spot capacity, by a label the deployment names, so readiness says when an engine's head sits
//! on one; and which nodes the provider is about to take back, by another label, each a notice
//! the coordinator drains the node for, as it does a node's own (decision 0023). A node's daemon is
//! named as its Node is.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::clock::now_ms;
use crate::kubernetes::nodes::Nodes;

use super::Coordinator;
use super::nodes::NodeSpecError;

/// A label as `key=value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    /// The label's key.
    pub key: String,
    /// The value that marks the node.
    pub value: String,
}

impl std::str::FromStr for Label {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once('=') {
            Some((key, value)) if !key.is_empty() => Ok(Self {
                key: key.to_owned(),
                value: value.to_owned(),
            }),
            _ => Err(format!("{s}: a label is key=value")),
        }
    }
}

/// What the coordinator reads from the Nodes' labels.
#[derive(Debug, Clone, Default)]
pub struct NodeLabels {
    /// A node carrying this label is spot capacity.
    pub spot: Option<Label>,
    /// A node carrying this label is about to be taken back by its provider.
    pub preemption: Option<Label>,
    /// How long after the preemption label the provider takes the node: the drain's deadline.
    pub preemption_lead: Duration,
}

/// The nodes' capacity, as the cluster's Nodes say it.
#[derive(Debug, Clone, Default)]
pub struct Capacity {
    /// Every node the cluster lists.
    pub listed: BTreeSet<String>,
    /// Those that are spot.
    pub spot: BTreeSet<String>,
}

/// What `node` is, when the cluster's Nodes are followed: Some(true) for spot, Some(false) for
/// on-demand, None for a node the cluster does not list or when no Nodes are followed.
pub(super) fn is_spot(capacity: Option<&Capacity>, node: &str) -> Option<bool> {
    let c = capacity?;
    c.listed.contains(node).then(|| c.spot.contains(node))
}

impl Coordinator {
    /// Follow the cluster's Nodes that `access` reaches, read by `labels`, until `cancel` fires:
    /// a reflector keeps every Node, and this coordinator reads each change.
    pub fn follow_cluster(
        self: &Arc<Self>,
        access: crate::kubernetes::Access,
        labels: NodeLabels,
        cancel: &CancellationToken,
    ) {
        let (tx, rx) = watch::channel(Nodes::default());
        tokio::spawn(crate::kubernetes::nodes::reflect(
            access,
            tx,
            cancel.clone(),
        ));
        tokio::spawn(Arc::clone(self).follow_nodes(rx, labels, cancel.clone()));
    }

    /// Follow the cluster's Nodes as `nodes` delivers them, read by `labels`, until `cancel`
    /// fires.
    pub async fn follow_nodes(
        self: Arc<Self>,
        mut nodes: watch::Receiver<Nodes>,
        labels: NodeLabels,
        cancel: CancellationToken,
    ) {
        loop {
            let now = nodes.borrow_and_update().clone();
            self.observe_nodes(&now, &labels).await;
            tokio::select! {
                () = cancel.cancelled() => return,
                changed = nodes.changed() => if changed.is_err() { return },
            }
        }
    }

    /// The cluster's Nodes are `nodes` now: their capacity, and a drain for each node newly
    /// marked for preemption.
    async fn observe_nodes(self: &Arc<Self>, nodes: &Nodes, labels: &NodeLabels) {
        let marked = |label: &Option<Label>| -> BTreeSet<String> {
            label.as_ref().map_or_else(BTreeSet::new, |l| {
                nodes
                    .iter()
                    .filter(|(_, n)| n.labels.get(&l.key) == Some(&l.value))
                    .map(|(name, _)| name.clone())
                    .collect()
            })
        };
        *self.capacity.lock().await = Some(Capacity {
            listed: nodes.keys().cloned().collect(),
            spot: marked(&labels.spot),
        });
        let preempted = marked(&labels.preemption);
        let newly: Vec<String> = {
            let mut noticed = self.preempted.lock().await;
            let newly = preempted.difference(&noticed).cloned().collect();
            *noticed = preempted;
            newly
        };
        for node in newly {
            self.preemption_notice(&node, labels).await;
        }
        let engines: Vec<String> = self.engines.lock().await.keys().cloned().collect();
        for engine in engines {
            self.observed(&engine).await;
        }
    }

    /// `node`'s provider is taking it back: drain it by the provider's lead.
    async fn preemption_notice(self: &Arc<Self>, node: &str, labels: &NodeLabels) {
        let Some(label) = labels.preemption.as_ref() else {
            return;
        };
        let lead_ms = i64::try_from(labels.preemption_lead.as_millis()).unwrap_or(i64::MAX);
        let deadline_ms = now_ms().saturating_add(lead_ms);
        let detail = format!(
            "its Node carries {}={}: the provider takes it within {:?}",
            label.key, label.value, labels.preemption_lead
        );
        info!("node {node}: {detail}; draining");
        let writer = format!("node {node} (kubernetes label {})", label.key);
        match self.drain(node, deadline_ms, &writer, &detail).await {
            Ok(_) => info!("node {node}: drain set"),
            Err(NodeSpecError::NoMembers(_)) => {
                info!("node {node}: holds no member; nothing to drain");
            }
            Err(e) => warn!("node {node}: drain not set: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Capacity, Label, is_spot};

    #[test]
    fn a_label_is_key_equals_value() {
        let l: Label = "qos.coreweave.cloud/pending-preemption=true"
            .parse()
            .expect("label");
        assert_eq!(
            (l.key.as_str(), l.value.as_str()),
            ("qos.coreweave.cloud/pending-preemption", "true")
        );
        assert!("no-equals".parse::<Label>().is_err());
        assert!("=v".parse::<Label>().is_err());
    }

    #[test]
    fn a_node_is_spot_on_demand_or_unknown() {
        let c = Capacity {
            listed: ["a".into(), "b".into()].into(),
            spot: ["b".into()].into(),
        };
        assert_eq!(is_spot(Some(&c), "a"), Some(false));
        assert_eq!(is_spot(Some(&c), "b"), Some(true));
        assert_eq!(
            is_spot(Some(&c), "z"),
            None,
            "a node the cluster does not list"
        );
        assert_eq!(is_spot(None, "a"), None, "no Nodes followed");
    }
}
