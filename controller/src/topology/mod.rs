// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The topology model (`docs/MULTINODE.md` section 1): devices, their domains, the positions an
//! engine needs, and `Fit`, the one place hardware classes are compared.
//!
//! A device is one GPU with an identity, a class (the product name the driver reports), its memory
//! and the domain it reaches at full speed. A position is one rank of one engine: what it needs
//! and the class it runs on now. [`fit`] says whether a device can hold a position and at what
//! expected speed relative to the position's current device. Every later story about mixed
//! hardware (a mismatched spare serving degraded until a planned swap-back; capacity-weighted
//! placement) is a change to what these carry, not to who uses them.

use std::collections::BTreeMap;

use crate::proto::everserve::coord::v1::Device as WireDevice;

/// One GPU.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Device {
    /// The node it sits in.
    pub node: String,
    /// The index the engine's `CUDA_VISIBLE_DEVICES` and `Member.device` use.
    pub index: u32,
    /// The driver's identity for it.
    pub uuid: String,
    /// The product name the driver reports; the class a fit compares.
    pub class: String,
    /// Device memory in bytes.
    pub memory_bytes: u64,
    /// The domain it reaches at full speed (the node, until a wider domain is found).
    pub domain: String,
}

impl Device {
    /// A device as a daemon's hello describes it.
    #[must_use]
    pub fn from_wire(node: &str, d: &WireDevice) -> Self {
        Self {
            node: node.to_owned(),
            index: d.index,
            uuid: d.uuid.clone(),
            class: d.class.clone(),
            memory_bytes: d.memory_bytes,
            domain: if d.domain.is_empty() {
                node.to_owned()
            } else {
                d.domain.clone()
            },
        }
    }
}

/// One rank of one engine: what it needs and what it runs on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    /// The engine.
    pub engine: String,
    /// The rank.
    pub rank: u32,
    /// Bytes the position's shard and KV need on a device; 0 when unknown.
    pub memory_bytes: u64,
    /// The class of the device the position runs on now, or ran on last; empty when unknown.
    pub class: String,
    /// The domain the position's peers are in; a spare outside it reaches them over the fabric.
    pub domain: String,
}

/// Whether a device can hold a position and how well.
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    /// The device can hold the position.
    pub ok: bool,
    /// The device is the position's class (or the class is unknown): no degrade to weigh.
    pub same_class: bool,
    /// Expected speed of the position on this device relative to its current class, 1.0 for the
    /// same class; a lower number is slower.
    pub speed: f64,
    /// Same domain as the position's peers.
    pub same_domain: bool,
    /// Why not, or what degrades.
    pub why: String,
}

/// Relative decode speed of the classes the fleet is likely to hold, against H100. A class not
/// listed compares as unknown: a mismatch is allowed only by a policy that accepts any degrade.
/// The numbers are coarse and for ranking only; a run measures the real figure.
fn speed_factor(class: &str) -> Option<f64> {
    let c = class.to_ascii_uppercase();
    let table: &[(&str, f64)] = &[
        ("B200", 1.8),
        ("GB200", 1.8),
        ("H200", 1.15),
        ("H100", 1.0),
        ("A100", 0.5),
        ("L40S", 0.3),
        ("L40", 0.25),
        ("A40", 0.25),
        ("A10", 0.15),
        ("L4", 0.12),
        ("T4", 0.06),
    ];
    table
        .iter()
        .find(|(name, _)| c.contains(name))
        .map(|(_, f)| *f)
}

/// Whether `device` can hold `position`, and how fast it would run it.
///
/// Memory decides `ok`; class decides `speed`; domain is reported, not decided here, because
/// whether a fabric-reached spare is acceptable is the policy's call.
#[must_use]
pub fn fit(position: &Position, device: &Device) -> Fit {
    let same_domain = position.domain.is_empty() || position.domain == device.domain;
    if position.memory_bytes > 0 && device.memory_bytes < position.memory_bytes {
        return Fit {
            ok: false,
            same_class: position.class == device.class,
            speed: 0.0,
            same_domain,
            why: format!(
                "{} has {} bytes, the position needs {}",
                device.class, device.memory_bytes, position.memory_bytes
            ),
        };
    }
    if position.class.is_empty() || position.class == device.class {
        return Fit {
            ok: true,
            same_class: true,
            speed: 1.0,
            same_domain,
            why: String::new(),
        };
    }
    match (speed_factor(&position.class), speed_factor(&device.class)) {
        (Some(from), Some(to)) => Fit {
            ok: true,
            same_class: false,
            speed: to / from,
            same_domain,
            why: format!("{} in place of {}", device.class, position.class),
        },
        _ => Fit {
            ok: true,
            same_class: false,
            speed: f64::NAN,
            same_domain,
            why: format!(
                "{} in place of {}: relative speed unknown",
                device.class, position.class
            ),
        },
    }
}

/// Whether a fit's degrade is within the policy's `max_degrade_percent` (0: any degrade, as long
/// as the speed is known or the policy allows unknown).
#[must_use]
pub fn within_degrade(f: &Fit, max_degrade_percent: u32) -> bool {
    if !f.ok {
        return false;
    }
    if f.speed.is_nan() {
        return max_degrade_percent == 0;
    }
    if max_degrade_percent == 0 {
        return true;
    }
    let slowdown = (1.0 - f.speed.min(1.0)) * 100.0;
    slowdown <= f64::from(max_degrade_percent)
}

/// Every device the daemons have reported, by node.
#[derive(Debug, Default, Clone)]
pub struct Topology {
    devices: BTreeMap<String, Vec<Device>>,
}

impl Topology {
    /// Replace `node`'s devices with what its daemon reported.
    pub fn set_node(&mut self, node: &str, devices: Vec<Device>) {
        self.devices.insert(node.to_owned(), devices);
    }

    /// Forget `node`.
    pub fn remove_node(&mut self, node: &str) {
        self.devices.remove(node);
    }

    /// The device at `index` on `node`, if reported.
    #[must_use]
    pub fn device(&self, node: &str, index: u32) -> Option<&Device> {
        self.devices.get(node)?.iter().find(|d| d.index == index)
    }

    /// Every device, in node order.
    pub fn devices(&self) -> impl Iterator<Item = &Device> {
        self.devices.values().flatten()
    }

    /// The domain `node`'s devices reach at full speed: their reported domain, or the node itself
    /// when it reported none.
    #[must_use]
    pub fn domain_of(&self, node: &str) -> String {
        self.devices
            .get(node)
            .and_then(|ds| ds.first())
            .map_or_else(|| node.to_owned(), |d| d.domain.clone())
    }

    /// The nodes whose devices share `domain`.
    #[must_use]
    pub fn nodes_in_domain(&self, domain: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .devices
            .iter()
            .filter(|(_, ds)| ds.iter().any(|d| d.domain == domain))
            .map(|(n, _)| n.clone())
            .collect();
        out.dedup();
        out
    }
}
