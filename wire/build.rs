// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Generate the Rust types for `proto/` with `protox` (no system protoc) and `pbjson` for the
//! protobuf JSON mapping, so a policy document written as JSON parses into the same types the
//! wire uses.

use std::path::PathBuf;

use prost::Message;

fn main() {
    let include = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../proto");
    let dir = include.join("everserve/proto");
    let files = [
        "signals.proto",
        "mirror.proto",
        "policy.proto",
        "discovery.proto",
        "coord.proto",
        "spec.proto",
    ]
    .map(|f| dir.join(f));
    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    let descriptors = protox::compile(&files, [&include]).expect("compile proto/");
    let bytes = descriptors.encode_to_vec();
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let descriptor_path = out.join("descriptors.bin");
    std::fs::write(&descriptor_path, &bytes).expect("write descriptors");
    // a step config's fields in key order: compared and shown the same way every time
    let ordered = [".everserve.signals.v1.StepConfig.fields"];
    prost_build::Config::new()
        .btree_map(ordered)
        .file_descriptor_set_path(&descriptor_path)
        .skip_protoc_run()
        .compile_protos(&files, &[include.as_path()])
        .expect("prost-build");
    pbjson_build::Builder::new()
        .register_descriptors(&bytes)
        .expect("register descriptors")
        // pbjson orders every map of a message, named by the message's path
        .btree_map([".everserve.signals.v1.StepConfig"])
        .build(&[".everserve"])
        .expect("pbjson-build");
}
