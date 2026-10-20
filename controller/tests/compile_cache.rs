// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The compile cache (decision 0029): a spare is held to the cache of the rank it stands in
//! for, a re-formed engine sets its own compile key, a continuous engine reports a difference
//! and holds no spare to it, and an engine reads consistent only once every process reported.

mod common;

use everserve_controller::proto::everserve::signals::v1::Role;

use common::cluster::{cache, cache_under, cluster, engine_readiness, register_with};

/// (rank, digest or "none") per process, as the readiness view lists them.
type Pairs = Vec<(u32, String)>;

fn pairs(e: &everserve_controller::readiness::EngineReadiness) -> (Pairs, Pairs) {
    let text = |c: &Option<everserve_controller::readiness::CompileCacheReadiness>| {
        c.as_ref().map_or("none".to_owned(), |c| c.digest.clone())
    };
    (
        e.members
            .iter()
            .map(|m| (m.rank, text(&m.compile_cache)))
            .collect(),
        e.spares
            .iter()
            .map(|s| (s.rank, text(&s.compile_cache)))
            .collect(),
    )
}

#[tokio::test]
async fn a_spare_is_held_to_the_compile_cache_of_the_rank_it_stands_in_for() {
    // Decision 0029, per position: one key for the engine; a rank's digest is its own (each
    // pipeline stage compiles its own graph); a spare must report the pair of its rank.
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let _w0 = register_with(&sock_a, 0, 0, Role::Worker, Some(cache("r0"))).await;
    let _w1 = register_with(&sock_a, 1, 1, Role::Worker, Some(cache("r1"))).await;
    let _s1 = register_with(&sock_a, 1, 2, Role::Spare, Some(cache("r1"))).await;
    let _s0 = register_with(&sock_a, 0, 3, Role::Spare, Some(cache("r1"))).await;
    let e = engine_readiness(&cl).await;
    assert_eq!(e.compile_key.as_deref(), Some("760834052d"));
    let (members, spares) = pairs(&e);
    assert_eq!(members, [(0, "r0".into()), (1, "r1".into())]);
    assert_eq!(
        spares,
        [(1, "r1".into())],
        "the spare for rank 0 reported rank 1's digest and is not admitted: {e:?}"
    );
    assert!(e.compile_cache_consistent, "{e:?}");

    // a spare for a rank nobody serves yet is admitted and held once the rank reports
    let _s2 = register_with(&sock_a, 2, 4, Role::Spare, Some(cache("zz"))).await;
    let e = engine_readiness(&cl).await;
    assert_eq!(pairs(&e).1, [(1, "r1".into()), (2, "zz".into())]);
    assert!(
        !e.compile_cache_consistent,
        "rank 2 has no serving pair yet: {e:?}"
    );
    let _w2 = register_with(&sock_a, 2, 5, Role::Worker, Some(cache("r2"))).await;
    let e = engine_readiness(&cl).await;
    assert_eq!(
        pairs(&e).1,
        [(1, "r1".into())],
        "the spare on another pair left the pool: {e:?}"
    );
    assert!(e.compile_cache_consistent, "{e:?}");

    // another key is another compile: never consistent while it serves
    let w3 = register_with(
        &sock_a,
        3,
        6,
        Role::Worker,
        Some(cache_under("other", "r3")),
    )
    .await;
    let e = engine_readiness(&cl).await;
    assert!(!e.compile_cache_consistent, "{e:?}");
    drop(w3);
    let e = engine_readiness(&cl).await;
    assert!(e.compile_cache_consistent, "{e:?}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_re_formed_engine_sets_its_own_compile_key() {
    // an engine's rank sets the key and goes down; the engine that follows under the same name
    // on the same control plane is not held to it
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let w0 = register_with(&sock_a, 0, 0, Role::Worker, Some(cache_under("first", "a"))).await;
    let s1 = register_with(&sock_a, 1, 1, Role::Spare, Some(cache_under("first", "b"))).await;
    let e = engine_readiness(&cl).await;
    assert_eq!(e.compile_key.as_deref(), Some("first"));
    drop(w0);
    let e = engine_readiness(&cl).await;
    assert_eq!(
        e.compile_key.as_deref(),
        Some("first"),
        "a spare still holds the key: {e:?}"
    );
    drop(s1);
    let e = engine_readiness(&cl).await;
    assert!(
        e.compile_key.is_none(),
        "every process gone, the key with them: {e:?}"
    );
    let _w0 = register_with(
        &sock_a,
        0,
        0,
        Role::Worker,
        Some(cache_under("second", "c")),
    )
    .await;
    let _w1 = register_with(
        &sock_a,
        1,
        1,
        Role::Worker,
        Some(cache_under("second", "d")),
    )
    .await;
    let e = engine_readiness(&cl).await;
    assert_eq!(e.compile_key.as_deref(), Some("second"));
    assert!(e.compile_cache_consistent, "{e:?}");
    cl.cancel.cancel();
}

#[tokio::test]
async fn a_continuous_engine_reports_a_compile_difference_and_holds_no_spare_to_it() {
    // decision 0030: the compile hold serves the exact class; under the continuous class a spare
    // on another pair joins the pool, and readiness still says the engine is inconsistent
    let cl = cluster("policy-two-node-resident-continuous.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let _w0 = register_with(&sock_a, 0, 0, Role::Worker, Some(cache("r0"))).await;
    let _w1 = register_with(&sock_a, 1, 1, Role::Worker, Some(cache("r1"))).await;
    let _s1 = register_with(&sock_a, 1, 2, Role::Spare, Some(cache("other"))).await;
    let e = engine_readiness(&cl).await;
    assert_eq!(e.guarantee, "continuous");
    assert_eq!(
        pairs(&e).1,
        [(1, "other".into())],
        "the spare joins the pool: {e:?}"
    );
    assert!(
        !e.compile_cache_consistent,
        "the difference is still reported: {e:?}"
    );
    cl.cancel.cancel();
}

#[tokio::test]
async fn an_engine_reads_consistent_only_when_every_process_reported_its_cache() {
    // every process reports none: a flag that reads "consistent" over no pairs is no check at all
    let cl = cluster("policy-rig-4gpu.json").await;
    let (_da, sock_a) =
        common::daemon(cl.dir.path(), "node-a", &cl.address, Vec::new(), &cl.cancel).await;
    let w0 = register_with(&sock_a, 0, 0, Role::Worker, None).await;
    let w1 = register_with(&sock_a, 1, 1, Role::Worker, None).await;
    let s4 = register_with(&sock_a, 4, 4, Role::Spare, None).await;
    let e = engine_readiness(&cl).await;
    assert!(e.compile_key.is_none());
    assert!(
        !e.compile_cache_consistent,
        "no pair reported is not consistency: {e:?}"
    );
    assert_eq!(
        pairs(&e).1,
        [(4, "none".into())],
        "a spare with none is kept while no key: {e:?}"
    );
    let _w2 = register_with(&sock_a, 2, 2, Role::Worker, Some(cache("r2"))).await;
    let _w3 = register_with(&sock_a, 3, 3, Role::Worker, Some(cache("r3"))).await;
    let _s5 = register_with(&sock_a, 5, 5, Role::Spare, None).await;
    let e = engine_readiness(&cl).await;
    assert!(
        !e.compile_cache_consistent,
        "two ranks reported none: {e:?}"
    );
    // a spare with none is refused once a key exists, whether it registered before the key
    // (4, admitted then, held to the key now: decision 0036) or after it (5)
    assert!(pairs(&e).1.is_empty(), "{e:?}");
    let refused: Vec<u32> = e.refused_spares.iter().filter_map(|r| r.rank).collect();
    assert_eq!(refused, [4, 5], "{e:?}");
    drop(w0);
    drop(w1);
    drop(s4);
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let e = engine_readiness(&cl).await;
    assert!(
        e.compile_cache_consistent,
        "every remaining process reported its pair: {e:?}"
    );
    cl.cancel.cancel();
}
