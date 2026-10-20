// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A node's processes share what they make through the node's group: a shared directory is the
//! group's, mode 2770, and under the shared umask a directory, a file and a socket made there are
//! the group's to read and write and no one else's. One test: the umask is the process's.

#![cfg(unix)]

use std::os::unix::fs::{MetadataExt, PermissionsExt};

use everserve_wire::host::{SHARED_DIR_MODE, free_bytes, share, share_what_is_made};

fn mode(path: &std::path::Path) -> u32 {
    std::fs::metadata(path)
        .expect("metadata")
        .permissions()
        .mode()
        & 0o7777
}

#[test]
fn what_a_nodes_processes_make_in_a_shared_directory_is_their_groups() {
    let tmp = tempfile::tempdir().expect("tmp");
    let dir = tmp.path().join("run").join("everserve");
    // this process's own group: the one a test may give a directory it owns to
    let gid = std::fs::metadata(tmp.path()).expect("tmp").gid();
    share(&dir, gid).expect("share");
    assert_eq!(mode(&dir), SHARED_DIR_MODE, "made, and the group's, setgid");
    assert_eq!(std::fs::metadata(&dir).expect("dir").gid(), gid);

    share_what_is_made();
    let sub = dir.join("store");
    std::fs::create_dir(&sub).expect("dir");
    let file = dir.join("table");
    std::fs::write(&file, b"x").expect("file");
    let socket = dir.join("node.sock");
    let _listener = std::os::unix::net::UnixListener::bind(&socket).expect("bind");
    // everything made inside is the group's; on Linux, where nodes run, a directory made inside
    // is setgid too (BSD gives the group without copying the bit)
    for made in [&sub, &file, &socket] {
        assert_eq!(std::fs::metadata(made).expect("made").gid(), gid);
    }
    let setgid = if cfg!(target_os = "linux") { 0o2000 } else { 0 };
    assert_eq!(mode(&sub), 0o770 | setgid, "a directory made inside");
    assert_eq!(mode(&file), 0o660, "a file made inside");
    assert_eq!(mode(&socket) & 0o007, 0, "a socket: none of it the world's");
    assert_eq!(mode(&socket) & 0o060, 0o060, "the group may connect");

    assert!(free_bytes(&dir).expect("free") > 0);
}

#[test]
fn the_memory_limit_is_read_as_cgroup_v2_writes_it() {
    use everserve_wire::host::memory_limit_at;
    let tmp = tempfile::tempdir().expect("tmp");
    let path = tmp.path().join("memory.max");
    std::fs::write(&path, "1610612736\n").expect("limit");
    assert_eq!(memory_limit_at(&path).expect("read"), Some(1_610_612_736));
    std::fs::write(&path, "max\n").expect("none");
    assert_eq!(memory_limit_at(&path).expect("read"), None);
    assert_eq!(
        memory_limit_at(&tmp.path().join("absent")).expect("read"),
        None,
        "no cgroup v2"
    );
    std::fs::write(&path, "lots\n").expect("garbage");
    assert!(memory_limit_at(&path).is_err());
}
