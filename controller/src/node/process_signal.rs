// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The signals a daemon sends to the processes it serves or supervises, through the system call.

use std::io;

use rustix::process::{Pid, Signal, kill_process};

/// What a signal asks of a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ask {
    /// Stop (`SIGTERM`): a spare the daemon stops, before it is killed after its grace.
    Stop,
    /// Write every thread's stack to the process's log and go on (`SIGUSR1`, which the vLLM
    /// plugin and the core runtime hand to `faulthandler`).
    Stacks,
}

/// Send `ask` to process `pid`.
///
/// # Errors
///
/// The system call's error: no such process, or not permitted; or a pid no process can have.
pub fn send(pid: u32, ask: Ask) -> io::Result<()> {
    let pid = i32::try_from(pid)
        .ok()
        .and_then(Pid::from_raw)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("no process {pid}")))?;
    let signal = match ask {
        Ask::Stop => Signal::TERM,
        Ask::Stacks => Signal::USR1,
    };
    kill_process(pid, signal).map_err(io::Error::from)
}

#[cfg(test)]
mod tests {
    use std::os::unix::process::ExitStatusExt;
    use std::process::Command;

    use super::{Ask, send};

    #[test]
    fn each_ask_is_its_signal() {
        // a child that does nothing but wait: `Stop` ends it with SIGTERM; `Stacks` with SIGUSR1,
        // whose default action ends a process that does not catch it
        for (ask, signal) in [(Ask::Stop, 15), (Ask::Stacks, libc_usr1())] {
            let mut child = Command::new("sleep").arg("30").spawn().expect("sleep");
            send(child.id(), ask).expect("signal");
            let status = child.wait().expect("wait");
            assert_eq!(status.signal(), Some(signal), "{ask:?}");
        }
        assert!(send(0, Ask::Stop).is_err(), "pid 0 names no process");
    }

    /// SIGUSR1's number on this system (10 on Linux, 30 on macOS).
    fn libc_usr1() -> i32 {
        rustix::process::Signal::USR1.as_raw()
    }
}
