// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A device fits a position by memory, is ranked by class, and reports its domain; the policy's
//! degrade bound reads the fit; the topology holds what the daemons reported.

use everserve_controller::proto::everserve::coord::v1::Device as WireDevice;
use everserve_controller::topology::{Device, Position, Topology, fit, within_degrade};

const GIB: u64 = 1 << 30;

fn device(node: &str, index: u32, class: &str, memory_gib: u64) -> Device {
    Device {
        node: node.into(),
        index,
        uuid: format!("GPU-{node}-{index}"),
        class: class.into(),
        memory_bytes: memory_gib * GIB,
        domain: node.into(),
    }
}

fn position(class: &str, memory_gib: u64) -> Position {
    Position {
        engine: "k3-a".into(),
        rank: 3,
        memory_bytes: memory_gib * GIB,
        class: class.into(),
        domain: "node-a".into(),
    }
}

#[test]
fn the_same_class_fits_at_full_speed_in_or_out_of_the_domain() {
    let f = fit(
        &position("NVIDIA H100 80GB HBM3", 70),
        &device("node-a", 0, "NVIDIA H100 80GB HBM3", 80),
    );
    assert!(f.ok && f.same_class && f.same_domain);
    assert!((f.speed - 1.0).abs() < f64::EPSILON);
    let f = fit(
        &position("NVIDIA H100 80GB HBM3", 70),
        &device("node-c", 0, "NVIDIA H100 80GB HBM3", 80),
    );
    assert!(f.ok && f.same_class && !f.same_domain, "{f:?}");
}

#[test]
fn too_little_memory_does_not_fit() {
    let f = fit(
        &position("NVIDIA H100 80GB HBM3", 70),
        &device("node-c", 0, "NVIDIA A100-SXM4-40GB", 40),
    );
    assert!(!f.ok, "{f:?}");
    assert!(f.why.contains("needs"), "{f:?}");
    assert!(!within_degrade(&f, 0));
}

#[test]
fn another_class_is_ranked_by_the_speed_table() {
    let f = fit(
        &position("NVIDIA H100 80GB HBM3", 70),
        &device("node-c", 0, "NVIDIA A100-SXM4-80GB", 80),
    );
    assert!(f.ok && !f.same_class, "{f:?}");
    assert!((f.speed - 0.5).abs() < 1e-9, "{f:?}");
    assert!(within_degrade(&f, 0), "0 means any known degrade");
    assert!(within_degrade(&f, 50));
    assert!(!within_degrade(&f, 49));
    // a faster class is a degrade of zero
    let up = fit(
        &position("NVIDIA A100-SXM4-80GB", 70),
        &device("node-c", 0, "NVIDIA H200", 141),
    );
    assert!(up.speed > 1.0 && within_degrade(&up, 1), "{up:?}");
}

#[test]
fn an_unknown_class_has_no_speed_and_passes_only_an_open_bound() {
    let f = fit(
        &position("NVIDIA H100 80GB HBM3", 70),
        &device("node-c", 0, "MI300X", 192),
    );
    assert!(f.ok && f.speed.is_nan(), "{f:?}");
    assert!(within_degrade(&f, 0));
    assert!(!within_degrade(&f, 90));
}

#[test]
fn an_unknown_position_class_fits_anything_that_holds_it() {
    let f = fit(&position("", 0), &device("node-c", 0, "NVIDIA L4", 24));
    assert!(f.ok && f.same_class, "{f:?}");
}

#[test]
fn the_topology_holds_what_each_daemon_reported_and_forgets_a_gone_node() {
    let mut t = Topology::default();
    t.set_node(
        "node-a",
        vec![
            device("node-a", 0, "H100", 80),
            device("node-a", 1, "H100", 80),
        ],
    );
    t.set_node("node-b", vec![device("node-b", 0, "H100", 80)]);
    assert_eq!(t.device("node-a", 1).map(|d| d.index), Some(1));
    assert!(t.device("node-a", 2).is_none());
    assert_eq!(t.devices().count(), 3);
    assert_eq!(t.nodes_in_domain("node-b"), vec!["node-b".to_owned()]);
    // a re-hello replaces, not appends
    t.set_node("node-a", vec![device("node-a", 0, "H100", 80)]);
    assert_eq!(t.devices().count(), 2);
    t.remove_node("node-b");
    assert!(t.device("node-b", 0).is_none());
    assert!(t.nodes_in_domain("node-b").is_empty());
}

#[test]
fn a_wire_device_without_a_domain_is_in_its_node() {
    let d = Device::from_wire(
        "node-a",
        &WireDevice {
            index: 2,
            uuid: "GPU-x".into(),
            class: "NVIDIA H100 80GB HBM3".into(),
            memory_bytes: 80 * GIB,
            domain: String::new(),
        },
    );
    assert_eq!(d.domain, "node-a");
    assert_eq!(d.index, 2);
    let d = Device::from_wire(
        "node-a",
        &WireDevice {
            index: 0,
            uuid: "GPU-y".into(),
            class: "H100".into(),
            memory_bytes: 80 * GIB,
            domain: "nvl72-rack-3".into(),
        },
    );
    assert_eq!(d.domain, "nvl72-rack-3");
}
