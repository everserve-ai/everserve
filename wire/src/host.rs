// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What a node's processes share on the host: the pod contract's two directories (the daemon's
//! and the store's sockets, the store's tables and the engines' staging files and slot regions)
//! belong to the node's group, which the daemon, the store and every engine process on the node
//! are members of, and to nobody else. A shared directory is the group's, mode 2770: the setgid
//! bit makes everything made inside the group's too. A process that makes files there with
//! [`share_what_is_made`] in force makes them the group's to read and write.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// A shared directory's mode: its owner's and its group's, setgid.
pub const SHARED_DIR_MODE: u32 = 0o2770;

/// Make `dir` the group `gid`'s, mode [`SHARED_DIR_MODE`], creating it if it is missing. Run as
/// the directory's owner and a member of `gid`: an owner may give a file to a group it belongs to
/// without a capability.
///
/// # Errors
///
/// The directory cannot be made, given to the group, or have its mode set.
pub fn share(dir: &Path, gid: u32) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::os::unix::fs::chown(dir, None, Some(gid))?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(SHARED_DIR_MODE))
}

/// From now on, what this process makes (a directory, a file, a socket) is its owner's and its
/// group's to read and write, and no one else's: the umask 007. Called first thing by the daemon
/// and the store, before they make anything.
pub fn share_what_is_made() {
    rustix::process::umask(rustix::fs::Mode::from_raw_mode(0o007));
}

/// The bytes the file system holding `dir` has free for an unprivileged writer.
///
/// # Errors
///
/// The file system cannot be asked.
pub fn free_bytes(dir: &Path) -> std::io::Result<u64> {
    let stat = rustix::fs::statvfs(dir)?;
    Ok(stat.f_bavail.saturating_mul(stat.f_frsize))
}

/// Where a process under cgroup v2 reads the memory it may use.
pub const CGROUP_MEMORY_MAX: &str = "/sys/fs/cgroup/memory.max";

/// The bytes the cgroup memory limit at `path` allows; None when there is no limit (`max`) or
/// no such file (no cgroup v2, as on macOS). Shared memory a process writes counts against it.
///
/// # Errors
///
/// The file exists and cannot be read, or holds neither a number nor `max`.
pub fn memory_limit_at(path: &Path) -> std::io::Result<Option<u64>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    match text.trim() {
        "max" => Ok(None),
        n => n.parse().map(Some).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: {n:?}: {e}", path.display()),
            )
        }),
    }
}
