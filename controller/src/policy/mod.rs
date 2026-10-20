// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The deployment's policy document and the one decision the controller makes per event.
//!
//! The document is `everserve.policy.v1.Policy` written as protobuf JSON. The same binary serves
//! every customer; what differs is this file. See `docs/RUNTIME.md` section 2.

pub mod cover;

use std::collections::BTreeSet;
use std::path::Path;

use crate::proto::everserve::policy::v1::{Engine, Policy, spare_pool};
use crate::proto::everserve::signals::v1::Outcome;

/// Why a policy document was rejected.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    /// The file could not be read.
    #[error("cannot read policy document: {0}")]
    Io(#[from] std::io::Error),
    /// The file is not a valid `Policy` in protobuf JSON.
    #[error("policy document is not valid: {0}")]
    Parse(#[from] serde_json::Error),
    /// The document parsed but makes no sense.
    #[error("policy {version:?} is invalid: {reason}")]
    Invalid {
        /// The document's version string.
        version: String,
        /// What is wrong.
        reason: String,
    },
}

/// Read and validate a policy document.
///
/// # Errors
///
/// `Io` if the file cannot be read, `Parse` if it is not a `Policy` in protobuf JSON, `Invalid`
/// if it parsed but would leave an engine with nothing to do (see [`validate`]).
pub fn load(path: &Path) -> Result<Policy, PolicyError> {
    let text = std::fs::read_to_string(path)?;
    let policy: Policy = serde_json::from_str(&text)?;
    validate(&policy)?;
    Ok(policy)
}

/// Reject a document that would leave an engine with nothing to do.
///
/// # Errors
///
/// `Invalid` when the document allows no outcome, names an unspecified outcome, or has no spare
/// loading rule.
pub fn validate(policy: &Policy) -> Result<(), PolicyError> {
    let invalid = |reason: &str| PolicyError::Invalid {
        version: policy.version.clone(),
        reason: reason.to_owned(),
    };
    if policy.outcomes_in_order.is_empty() {
        return Err(invalid("allows no outcome"));
    }
    if policy
        .outcomes_in_order
        .iter()
        .any(|o| Outcome::try_from(*o) == Ok(Outcome::Unspecified))
    {
        return Err(invalid("has an unspecified outcome"));
    }
    match policy.spares.as_ref().map(|s| s.loading) {
        Some(l) if l != spare_pool::Loading::Unspecified as i32 => {}
        _ => return Err(invalid("has no spare loading rule")),
    }
    let cover = policy.cover.as_ref().is_some_and(|c| c.enabled);
    for e in &policy.engines {
        // a spare that stands after the engine's processes started is pre-formed only through
        // the sets cover plans; without it, such an engine would serve unprotected, silently
        if e.members.is_empty() && !cover {
            return Err(PolicyError::Invalid {
                version: policy.version.clone(),
                reason: format!(
                    "engine {} pins no members, so the processes that register place it, and it \
                     is replaced only through the sets cover plans as its spares stand: enable \
                     cover",
                    e.name
                ),
            });
        }
        let world = world_size(e);
        if !e.members.is_empty() && usize::try_from(world).ok() != Some(e.members.len()) {
            return Err(PolicyError::Invalid {
                version: policy.version.clone(),
                reason: format!(
                    "pins {} members on engine {}, whose layout makes {world} ranks",
                    e.members.len(),
                    e.name
                ),
            });
        }
    }
    Ok(())
}

/// How many ranks `engine` has: tensor x pipeline x data parallel, an unset width counting as
/// one (decision 0041: the shape, not the placement, numbers the world).
#[must_use]
pub fn world_size(engine: &Engine) -> u32 {
    engine.tensor_parallel.max(1) * engine.pipeline_parallel.max(1) * engine.data_parallel.max(1)
}

/// What happens when a rank leaves: the first allowed outcome whose preconditions hold.
///
/// REPLACE needs a ready spare, SHRINK needs a placement-independent model, REBUILD needs nothing.
/// A policy whose listed outcomes all fail their preconditions falls back to REBUILD: an engine
/// that can rebuild must not die for lack of a preference.
#[must_use]
pub fn choose_outcome(
    policy: &Policy,
    spare_ready: bool,
    placement_independent_model: bool,
) -> Outcome {
    for o in &policy.outcomes_in_order {
        match Outcome::try_from(*o) {
            Ok(Outcome::Replace) if spare_ready => return Outcome::Replace,
            Ok(Outcome::Shrink) if placement_independent_model => return Outcome::Shrink,
            Ok(Outcome::Rebuild) => return Outcome::Rebuild,
            _ => {}
        }
    }
    Outcome::Rebuild
}

/// The engine called `name`, if the policy covers it.
#[must_use]
pub fn engine<'a>(policy: &'a Policy, name: &str) -> Option<&'a Engine> {
    policy.engines.iter().find(|e| e.name == name)
}

/// The nodes whose stores take the rows of a rank held on `holder` (decision 0030): the policy's
/// `replicaNodes` when it names any, else the nodes holding the engine's members now, in both
/// cases other than `holder`. Siblings follow the live membership, so after a move a rank's rows
/// go to the node that took its sibling, not to the node that left.
#[must_use]
pub fn replica_targets<'a>(
    engine: &'a Engine,
    holder: &str,
    member_nodes: impl IntoIterator<Item = &'a String>,
) -> BTreeSet<String> {
    let named: Box<dyn Iterator<Item = &String>> = if engine.replica_nodes.is_empty() {
        Box::new(member_nodes.into_iter())
    } else {
        Box::new(engine.replica_nodes.iter())
    };
    named.filter(|n| *n != holder).cloned().collect()
}

/// Whether `engine` is held to the exact class (decision 0030): exact unless the policy says
/// continuous. The compile-cache hold (0029) and deterministic kernels serve this class only.
#[must_use]
pub fn exact(engine: &Engine) -> bool {
    engine.guarantee() != crate::proto::everserve::policy::v1::Guarantee::Continuous
}

/// [`replica_targets`] for rank `rank` as the policy places the engine: the preflight's view,
/// before anything has moved. A single-node engine without `replicaNodes` replicates nowhere.
#[must_use]
pub fn replica_nodes_for(engine: &Engine, rank: u32) -> BTreeSet<String> {
    let own = engine
        .members
        .get(usize::try_from(rank).unwrap_or(usize::MAX))
        .map_or("", |m| m.node.as_str());
    replica_targets(engine, own, engine.members.iter().map(|m| &m.node))
}

/// Every node that takes a replica of some rank of `engine`: its store must take replicas.
#[must_use]
pub fn replica_nodes(engine: &Engine) -> BTreeSet<String> {
    (0..u32::try_from(engine.members.len()).unwrap_or(0))
        .flat_map(|r| replica_nodes_for(engine, r))
        .collect()
}
