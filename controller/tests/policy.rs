// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The pilot policy document in `docs/examples/` loads, validates and decides as documented.

use std::path::PathBuf;

use everserve_controller::policy;
use everserve_controller::proto::everserve::signals::v1::Outcome;

fn example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-pilot.json")
}

#[test]
fn pilot_policy_loads_and_describes_the_engine() {
    let p = policy::load(&example()).expect("load");
    assert_eq!(p.version, "pilot-1");
    assert_eq!(
        p.outcomes_in_order,
        vec![Outcome::Replace as i32, Outcome::Rebuild as i32]
    );
    let e = policy::engine(&p, "k3-a").expect("engine");
    assert_eq!(e.members.len(), 16);
    assert_eq!(e.members[8].node, "n2");
    assert!(policy::engine(&p, "nope").is_none());
}

#[test]
fn outcome_is_the_first_whose_preconditions_hold() {
    let p = policy::load(&example()).expect("load");
    assert_eq!(policy::choose_outcome(&p, true, false), Outcome::Replace);
    assert_eq!(policy::choose_outcome(&p, false, false), Outcome::Rebuild);
    let mut shrink_first = p.clone();
    shrink_first.outcomes_in_order = vec![Outcome::Shrink as i32, Outcome::Replace as i32];
    assert_eq!(
        policy::choose_outcome(&shrink_first, true, true),
        Outcome::Shrink
    );
    assert_eq!(
        policy::choose_outcome(&shrink_first, true, false),
        Outcome::Replace
    );
    assert_eq!(
        policy::choose_outcome(&shrink_first, false, false),
        Outcome::Rebuild
    );
}

#[test]
fn invalid_documents_are_rejected() {
    let dir = std::env::temp_dir().join(format!("everserve-policy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    let no_outcome = dir.join("a.json");
    std::fs::write(
        &no_outcome,
        r#"{"version":"x","spares":{"loading":"LOADING_PRELOADED"}}"#,
    )
    .expect("write");
    assert!(matches!(
        policy::load(&no_outcome),
        Err(policy::PolicyError::Invalid { reason, .. }) if reason == "allows no outcome"
    ));
    let no_loading = dir.join("b.json");
    std::fs::write(
        &no_loading,
        r#"{"version":"x","outcomesInOrder":["OUTCOME_REBUILD"]}"#,
    )
    .expect("write");
    assert!(matches!(
        policy::load(&no_loading),
        Err(policy::PolicyError::Invalid { reason, .. }) if reason == "has no spare loading rule"
    ));
    let garbage = dir.join("c.json");
    std::fs::write(&garbage, "not json").expect("write");
    assert!(matches!(
        policy::load(&garbage),
        Err(policy::PolicyError::Parse(_))
    ));
}

#[test]
fn an_engines_shape_numbers_its_world_and_a_pin_must_agree_with_it() {
    // decision 0041: the layout, not the placement, numbers the world; members only pin it
    let dir = tempfile::tempdir().expect("tmp");
    let doc = |engine: &str| {
        format!(
            r#"{{"version":"x","outcomesInOrder":["OUTCOME_REPLACE"],
               "spares":{{"loading":"LOADING_PRELOADED"}},"engines":[{engine}]}}"#
        )
    };
    let shaped = dir.path().join("shaped.json");
    let shape = r#"{"name":"e","tensorParallel":4,"pipelineParallel":2,"dataParallel":2}"#;
    std::fs::write(&shaped, doc(shape)).expect("write");
    // placed by what registers, it is replaced only through cover's sets: cover must be on
    assert!(
        matches!(
            policy::load(&shaped),
            Err(policy::PolicyError::Invalid { reason, .. }) if reason.contains("enable cover")
        ),
        "a shape-only engine without cover"
    );
    std::fs::write(
        &shaped,
        doc(shape).replace(r#""engines":"#, r#""cover":{"enabled":true},"engines":"#),
    )
    .expect("write");
    let p = policy::load(&shaped).expect("a shape with no placement, under cover");
    assert_eq!(policy::world_size(&p.engines[0]), 16);
    let pinned_wrong = dir.path().join("pinned.json");
    std::fs::write(
        &pinned_wrong,
        doc(r#"{"name":"e","tensorParallel":2,
                "members":[{"node":"a","device":0},{"node":"a","device":1},{"node":"b","device":0}]}"#),
    )
    .expect("write");
    assert!(
        matches!(
            policy::load(&pinned_wrong),
            Err(policy::PolicyError::Invalid { reason, .. }) if reason.contains("pins 3 members")
        ),
        "three members pinned on a two-rank layout"
    );
    // every example agrees with itself
    for entry in
        std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/examples"))
            .expect("examples")
    {
        let path = entry.expect("entry").path();
        if path.extension().is_some_and(|e| e == "json")
            && path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("policy-"))
        {
            policy::load(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        }
    }
}

#[test]
fn the_two_node_policy_names_the_spare_nodes_and_the_mismatch_bound() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-two-node.json");
    let p = policy::load(&path).expect("load");
    assert_eq!(p.version, "two-node-1");
    let e = policy::engine(&p, "k3-a").expect("engine");
    assert_eq!(e.members.len(), 8);
    assert_eq!((e.tensor_parallel, e.pipeline_parallel), (4, 2));
    assert_eq!(e.spare_nodes, vec!["node-c".to_owned()]);
    let spares = p.spares.as_ref().expect("spares");
    assert_eq!(spares.per_engine, 4);
    assert!(!spares.allow_mismatched);
    assert_eq!(spares.max_degrade_percent, 0);
}

#[test]
fn a_ranks_rows_replicate_to_its_siblings_unless_the_policy_names_replica_nodes() {
    use std::collections::BTreeSet;

    use everserve_controller::proto::everserve::policy::v1::Engine;
    use everserve_controller::proto::everserve::signals::v1::Member;

    // decision 0030: one stage per node, no replicaNodes: each rank's rows go to the other
    // node's store, so a shared spare restores from the surviving sibling
    let two = Engine {
        name: "rig".into(),
        members: vec![
            Member {
                node: "node-a".into(),
                device: 0,
            },
            Member {
                node: "node-b".into(),
                device: 0,
            },
        ],
        ..Default::default()
    };
    let a: BTreeSet<String> = ["node-a".to_owned()].into();
    let b: BTreeSet<String> = ["node-b".to_owned()].into();
    assert_eq!(policy::replica_nodes_for(&two, 0), b);
    assert_eq!(policy::replica_nodes_for(&two, 1), a);
    assert_eq!(policy::replica_nodes(&two), &a | &b);

    // the per-position shape names its spare node, which then takes every rank's rows
    let placed = Engine {
        replica_nodes: vec!["node-c".into()],
        ..two.clone()
    };
    let c: BTreeSet<String> = ["node-c".to_owned()].into();
    assert_eq!(policy::replica_nodes_for(&placed, 0), c);
    assert_eq!(policy::replica_nodes_for(&placed, 1), c);
    assert_eq!(policy::replica_nodes(&placed), c);

    // a named replica node that is the rank's own node is skipped, as a sibling would be
    let own = Engine {
        replica_nodes: vec!["node-a".into()],
        ..two.clone()
    };
    assert!(policy::replica_nodes_for(&own, 0).is_empty());
    assert_eq!(policy::replica_nodes_for(&own, 1), a);

    // a single-node engine without replicaNodes replicates nowhere
    let one = Engine {
        members: vec![Member {
            node: "rig".into(),
            device: 0,
        }],
        ..Default::default()
    };
    assert!(policy::replica_nodes_for(&one, 0).is_empty());
    assert!(policy::replica_nodes(&one).is_empty());

    // the example policies of the per-position shape name their spare node
    let managed = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/examples/policy-two-node-managed.json");
    let e = policy::engine(&policy::load(&managed).expect("load"), "k3-a")
        .expect("engine")
        .clone();
    assert!((0..8).all(|r| policy::replica_nodes_for(&e, r) == c));
}

#[test]
fn the_release_bound_is_the_engines_or_absent() {
    let managed = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/examples/policy-two-node-managed.json");
    let p = policy::load(&managed).expect("load");
    assert_eq!(
        policy::engine(&p, "k3-a").expect("engine").release_bound_ms,
        Some(10_000)
    );
    let plain =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-two-node.json");
    let p = policy::load(&plain).expect("load");
    assert_eq!(
        policy::engine(&p, "k3-a").expect("engine").release_bound_ms,
        None,
        "absent is the adapter's default, not zero"
    );
}
