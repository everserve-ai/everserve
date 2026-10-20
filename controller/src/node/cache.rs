// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A spare's compile cache (decision 0043): a spare told to stand asks for its rank's, and the
//! daemon has its node's store fetch the set from the stores the engine's placement names as
//! holding it, then answers. A spare never compiles for itself: one whose rank's cache cannot be
//! put on the node is refused, and says so as it leaves.

use std::time::Duration;

use tokio::time::Instant;
use tracing::{info, warn};

use crate::proto::everserve::signals::v1::{
    CacheReady, CacheRefused, CacheWanted, CompileCache, Signal, signal,
};
use everserve_wire::proto::everserve::mirror::v1::{CacheFetch, MirrorRequest, mirror_request};

use super::Daemon;

/// How long a spare's ask may take, its rank's pair waited for and its set fetched: the scale of
/// a launched spare's warming (decision 0031).
pub const CACHE_WAIT: Duration = Duration::from_mins(10);

impl Daemon {
    /// Answer `want`: the rank's set on this node, from its holders when it is not here yet, or
    /// why not, within `within`.
    pub(super) async fn cache_for(&self, want: &CacheWanted, within: Duration) -> Signal {
        let deadline = Instant::now() + within;
        let refused = |reason: String| {
            warn!(
                "{}: rank {}'s compile cache: {reason}",
                want.engine, want.rank
            );
            Signal {
                kind: Some(signal::Kind::CacheRefused(CacheRefused {
                    engine: want.engine.clone(),
                    rank: want.rank,
                    reason,
                })),
            }
        };
        let Some((socket, _)) = self.mirror.lock().await.clone() else {
            return refused("this node has no store to fetch it into".into());
        };
        let Some((pair, holders)) = self.rank_cache(want, deadline).await else {
            return refused(format!(
                "no pair for rank {} within {within:?}: its serving worker has not registered one",
                want.rank
            ));
        };
        let fetch = MirrorRequest {
            kind: Some(mirror_request::Kind::CacheFetch(CacheFetch {
                key: pair.key.clone(),
                rank: want.rank,
                dp: want.dp,
                digest: pair.digest.clone(),
                holders: holders.clone(),
            })),
        };
        let left = deadline.saturating_duration_since(Instant::now());
        match super::store::request(&socket, fetch, left).await {
            Err(super::store::StoreError::Timeout(_)) => {
                refused(format!("not fetched from {holders:?} within {within:?}"))
            }
            Err(e) => refused(e.to_string()),
            Ok(_) => {
                info!(
                    "{}: rank {}'s compile cache {}/{} is on the node",
                    want.engine, want.rank, pair.key, pair.digest
                );
                Signal {
                    kind: Some(signal::Kind::CacheReady(CacheReady {
                        engine: want.engine.clone(),
                        rank: want.rank,
                        key: pair.key,
                        digest: pair.digest,
                    })),
                }
            }
        }
    }

    /// Rank `want.rank`'s pair and the stores holding it, from the engine's placement as the
    /// coordinator says it, waited for until `deadline`.
    async fn rank_cache(
        &self,
        want: &CacheWanted,
        deadline: Instant,
    ) -> Option<(CompileCache, Vec<String>)> {
        let mut facts = self.facts(&want.engine).await.subscribe();
        loop {
            let held = facts.borrow_and_update().placement.as_ref().and_then(|p| {
                p.members
                    .iter()
                    .find(|m| m.rank == want.rank)
                    .and_then(|m| Some((m.compile_cache.clone()?, m.cache_holders.clone())))
            });
            if let Some((pair, holders)) = held
                && !holders.is_empty()
            {
                return Some((pair, holders));
            }
            if tokio::time::timeout_at(deadline, facts.changed())
                .await
                .map_or(true, |r| r.is_err())
            {
                return None;
            }
        }
    }
}
