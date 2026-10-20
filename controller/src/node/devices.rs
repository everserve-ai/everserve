// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What GPUs this node has, for the daemon's hello: the driver's listing through `nvidia-smi`,
//! which every node with a driver carries, so the daemon needs no CUDA or NVML binding. A node
//! without it reports no devices and the coordinator fits spares by policy alone.

use everserve_wire::proto::everserve::coord::v1::Device;

/// The devices `nvidia-smi` lists, in index order; empty when it is absent or fails.
#[must_use]
pub fn list(node: &str) -> Vec<Device> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match out {
        Ok(o) if o.status.success() => parse(node, &String::from_utf8_lossy(&o.stdout)),
        _ => Vec::new(),
    }
}

/// Parse `nvidia-smi`'s CSV: `index, uuid, name, memory_mib`. Lines that do not parse are
/// skipped, so one odd line never hides the rest.
#[must_use]
pub fn parse(node: &str, csv: &str) -> Vec<Device> {
    csv.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 4 {
                return None;
            }
            Some(Device {
                index: f[0].parse().ok()?,
                uuid: f[1].to_owned(),
                class: f[2].to_owned(),
                memory_bytes: f[3].parse::<u64>().ok()? * 1024 * 1024,
                domain: node.to_owned(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn the_listing_parses_and_an_odd_line_skips_only_itself() {
        let csv = "0, GPU-aaaa, NVIDIA H100 80GB HBM3, 81559\n\
                   garbage\n\
                   1, GPU-bbbb, NVIDIA H100 80GB HBM3, 81559\n";
        let d = parse("node-a", csv);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].index, 0);
        assert_eq!(d[0].uuid, "GPU-aaaa");
        assert_eq!(d[0].class, "NVIDIA H100 80GB HBM3");
        assert_eq!(d[0].memory_bytes, 81559 * 1024 * 1024);
        assert_eq!(d[0].domain, "node-a");
        assert_eq!(d[1].index, 1);
        assert!(parse("node-a", "").is_empty());
    }
}
