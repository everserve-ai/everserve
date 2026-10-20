// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The spare processes this node's daemon owns (decision 0024): launched from the policy's
//! command with the placeholders filled in, watched, restarted with a backoff when they exit, and
//! stopped with the daemon. The daemon reads nothing of what the process is; it runs the command.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::proto::everserve::coord::v1::LaunchSpare;

/// The first wait before a spare that exited is launched again; doubles up to [`RESTART_MAX`].
pub const RESTART_MIN: Duration = Duration::from_secs(1);
/// The longest wait between launches of a spare that keeps exiting.
pub const RESTART_MAX: Duration = Duration::from_secs(30);
/// How long a spare gets after `SIGTERM` at stop before it is asked for its stacks and killed.
pub const STOP_GRACE: Duration = Duration::from_secs(10);
/// How long a spare that ignored `SIGTERM` gets to write its stacks before it is killed.
pub const STACKS_WAIT: Duration = Duration::from_secs(2);

/// Why a spare could not be launched.
#[derive(Debug, thiserror::Error)]
pub enum LaunchError {
    /// The command was empty.
    #[error("{engine} device {device}: an empty command")]
    Empty {
        /// The engine.
        engine: String,
        /// The device.
        device: u32,
    },
}

/// One launched spare as the daemon sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launched {
    /// The engine; empty for a pool's spare.
    pub engine: String,
    /// The pool; empty for an engine's own spare (decision 0031).
    pub pool: String,
    /// The rank it stands in for; None for a spare standing for any position (decision 0030).
    pub rank: Option<u32>,
    /// Its device.
    pub device: u32,
    /// Its process id now, or 0 between launches.
    pub pid: u32,
    /// Times it was launched again after exiting.
    pub restarts: u32,
}

#[derive(Debug)]
struct Slot {
    /// The rank it stands in for; None for any position (decision 0030).
    rank: Option<u32>,
    /// The pool it stands for; empty for an engine's own spare.
    pool: String,
    pid: Arc<Mutex<u32>>,
    restarts: Arc<Mutex<u32>>,
    cancel: CancellationToken,
}

/// The spares of one node.
pub struct Supervisor {
    node: String,
    socket: PathBuf,
    slots: Mutex<BTreeMap<(String, u32), Slot>>,
    cancel: CancellationToken,
}

impl Supervisor {
    /// A supervisor for `node`, whose daemon serves workers at `socket`; every spare it launches
    /// is stopped when `cancel` fires.
    #[must_use]
    pub fn new(node: &str, socket: &Path, cancel: CancellationToken) -> Self {
        Self {
            node: node.to_owned(),
            socket: socket.to_path_buf(),
            slots: Mutex::new(BTreeMap::new()),
            cancel,
        }
    }

    /// Launch the spare `spec` names and keep it running. A spare already launched for the same
    /// engine, or the same pool, on the same device is left as it is.
    ///
    /// # Errors
    ///
    /// [`LaunchError`] when the command is empty.
    pub async fn launch(&self, spec: LaunchSpare) -> Result<(), LaunchError> {
        let who = if spec.pool.is_empty() {
            spec.engine.clone()
        } else {
            format!("pool/{}", spec.pool)
        };
        let key = (who.clone(), spec.device);
        let mut slots = self.slots.lock().await;
        if slots.contains_key(&key) {
            debug!(
                "spare {who} on device {} already runs: nothing launched",
                spec.device
            );
            return Ok(());
        }
        if spec.command.is_empty() {
            return Err(LaunchError::Empty {
                engine: who,
                device: spec.device,
            });
        }
        let vars = [
            ("{engine}", spec.engine.clone()),
            ("{pool}", spec.pool.clone()),
            ("{engines}", spec.engines.join(",")),
            // empty for a spare standing for any position (decision 0030)
            ("{rank}", spec.rank.map_or(String::new(), |r| r.to_string())),
            ("{device}", spec.device.to_string()),
            ("{node}", self.node.clone()),
            ("{socket}", self.socket.display().to_string()),
        ];
        let command: Vec<String> = spec.command.iter().map(|a| render(a, &vars)).collect();
        let env: Vec<(String, String)> = spec
            .env
            .iter()
            .map(|(k, v)| (k.clone(), render(v, &vars)))
            .collect();
        let slot = Slot {
            rank: spec.rank,
            pool: spec.pool.clone(),
            pid: Arc::new(Mutex::new(0)),
            restarts: Arc::new(Mutex::new(0)),
            cancel: self.cancel.child_token(),
        };
        let (pid, restarts, cancel) = (
            Arc::clone(&slot.pid),
            Arc::clone(&slot.restarts),
            slot.cancel.clone(),
        );
        let label = format!("{who} {} on {}", standing(spec.rank), self.node);
        info!("launching spare {label}: {}", command.join(" "));
        tokio::spawn(keep_running(label, command, env, pid, restarts, cancel));
        slots.insert(key, slot);
        Ok(())
    }

    /// Every spare this supervisor launched.
    pub async fn launched(&self) -> Vec<Launched> {
        let slots = self.slots.lock().await;
        let mut out = Vec::new();
        for ((who, device), s) in slots.iter() {
            out.push(Launched {
                engine: if s.pool.is_empty() {
                    who.clone()
                } else {
                    String::new()
                },
                pool: s.pool.clone(),
                rank: s.rank,
                device: *device,
                pid: *s.pid.lock().await,
                restarts: *s.restarts.lock().await,
            });
        }
        out
    }

    /// Stop every launched spare and forget it.
    pub async fn stop_all(&self) {
        let slots = std::mem::take(&mut *self.slots.lock().await);
        for ((engine, device), s) in slots {
            info!(
                "stopping spare {engine} {} (device {device})",
                standing(s.rank)
            );
            s.cancel.cancel();
        }
    }
}

/// What a spare stands for, for a log line: one rank, or any position (decision 0030).
fn standing(rank: Option<u32>) -> String {
    rank.map_or_else(|| "for any rank".to_owned(), |r| format!("rank {r}"))
}

/// `template` with each `(placeholder, value)` of `vars` substituted.
#[must_use]
pub fn render(template: &str, vars: &[(&str, String)]) -> String {
    let mut out = template.to_owned();
    for (k, v) in vars {
        out = out.replace(k, v);
    }
    out
}

/// Run `command` until `cancel` fires, launching it again with a backoff each time it exits.
async fn keep_running(
    label: String,
    command: Vec<String>,
    env: Vec<(String, String)>,
    pid: Arc<Mutex<u32>>,
    restarts: Arc<Mutex<u32>>,
    cancel: CancellationToken,
) {
    let mut wait = RESTART_MIN;
    loop {
        let mut cmd = tokio::process::Command::new(&command[0]);
        cmd.args(&command[1..]).envs(env.iter().cloned());
        cmd.kill_on_drop(true);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                warn!(
                    "spare {label}: cannot start {}: {e}; retrying in {wait:?}",
                    command[0]
                );
                tokio::select! {
                    () = cancel.cancelled() => return,
                    () = tokio::time::sleep(wait) => {}
                }
                wait = (wait * 2).min(RESTART_MAX);
                continue;
            }
        };
        *pid.lock().await = child.id().unwrap_or(0);
        let status = tokio::select! {
            () = cancel.cancelled() => {
                stop(&mut child, &label, STOP_GRACE, STACKS_WAIT).await;
                return;
            }
            s = child.wait() => s,
        };
        *pid.lock().await = 0;
        match status {
            Ok(s) => warn!("spare {label} exited ({s}); launching again in {wait:?}"),
            Err(e) => warn!("spare {label}: wait failed ({e}); launching again in {wait:?}"),
        }
        tokio::select! {
            () = cancel.cancelled() => return,
            () = tokio::time::sleep(wait) => {}
        }
        *restarts.lock().await += 1;
        wait = (wait * 2).min(RESTART_MAX);
    }
}

/// `SIGTERM`; a spare still running after `grace` is asked for its stacks (`SIGUSR1`), so its own
/// log says where its stop was held, and killed after `stacks_wait`.
async fn stop(
    child: &mut tokio::process::Child,
    label: &str,
    grace: Duration,
    stacks_wait: Duration,
) {
    use super::process_signal::{Ask, send};
    let id = child.id();
    if let Some(id) = id
        && let Err(e) = send(id, Ask::Stop)
    {
        warn!("spare {label}: SIGTERM to {id} failed: {e}");
    }
    if tokio::time::timeout(grace, child.wait()).await.is_ok() {
        return;
    }
    warn!("spare {label} ignored SIGTERM for {grace:?}; asking for its stacks, then killing");
    if let Some(id) = id
        && let Err(e) = send(id, Ask::Stacks)
    {
        warn!("spare {label}: SIGUSR1 to {id} failed: {e}");
    }
    if tokio::time::timeout(stacks_wait, child.wait())
        .await
        .is_err()
    {
        let _ = child.kill().await;
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::process::ExitStatusExt;
    use std::process::Stdio;
    use std::time::Duration;

    use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

    use super::{STACKS_WAIT, render, stop};

    /// Run `script` under `sh`, wait for its `ready` line, stop it with the supervisor's own
    /// stacks wait, and return how it ended and what it printed after `ready`.
    async fn stopped(script: &str) -> (Option<i32>, String) {
        let mut child = tokio::process::Command::new("sh")
            .args(["-c", script])
            .stdout(Stdio::piped())
            .spawn()
            .expect("sh");
        let mut out = BufReader::new(child.stdout.take().expect("stdout"));
        let mut ready = String::new();
        out.read_line(&mut ready).await.expect("ready");
        assert_eq!(ready.trim(), "ready");
        stop(&mut child, "test", Duration::from_millis(300), STACKS_WAIT).await;
        let status = child.wait().await.expect("wait");
        let mut rest = String::new();
        out.read_to_string(&mut rest).await.expect("output");
        (status.signal(), rest)
    }

    #[tokio::test]
    async fn a_spare_that_ignores_sigterm_is_asked_for_its_stacks_before_the_kill() {
        // the shell runs a trap only between commands, and `wait` is interrupted by a trapped
        // signal where a foreground `sleep` is not: under load the trap waited out the sleep and
        // the stacks wait with it, and the kill came first (CI 38272352672, first attempt)
        let (signal, out) = stopped(
            "trap '' TERM; trap 'echo stacks' USR1; echo ready; \
             while :; do sleep 0.05 & wait $!; done",
        )
        .await;
        assert_eq!(signal, Some(9), "killed after the stacks wait");
        assert!(
            out.contains("stacks"),
            "the stacks request came first: {out:?}"
        );
    }

    #[tokio::test]
    async fn a_spare_that_stops_on_sigterm_is_not_asked_for_its_stacks() {
        let (signal, out) = stopped("trap 'echo stacks' USR1; echo ready; sleep 30").await;
        assert_eq!(signal, Some(15), "ended by SIGTERM");
        assert!(!out.contains("stacks"), "no stacks request: {out:?}");
    }

    #[test]
    fn placeholders_are_filled_in() {
        let vars = [
            ("{rank}", "3".to_owned()),
            ("{socket}", "/run/x.sock".to_owned()),
        ];
        assert_eq!(
            render("--rank {rank} --socket {socket} {other}", &vars),
            "--rank 3 --socket /run/x.sock {other}"
        );
    }
}
