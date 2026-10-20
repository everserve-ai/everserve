// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A throwaway etcd for the tests that need a real one: the etcd lease's, and the Kubernetes API
//! server's store. Started from the `etcd` binary on `PATH` (or at `EVERSERVE_TEST_ETCD`); a test
//! that needs one fails without it, since a lease test that skips proves nothing.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A throwaway etcd on loopback, killed when dropped.
pub struct Etcd {
    /// The etcd process.
    pub child: Child,
    /// Its client endpoint, `127.0.0.1:port`.
    pub endpoint: String,
    _dir: tempfile::TempDir,
}

impl Etcd {
    /// Start etcd on two free loopback ports. A port is chosen by binding port 0 and letting it
    /// go, so another process may take it before etcd binds it: an etcd that exits before it
    /// answers is started again on fresh ports. Its log is kept for the failure message.
    pub async fn start() -> Self {
        let bin =
            std::env::var_os("EVERSERVE_TEST_ETCD").map_or_else(|| "etcd".into(), PathBuf::from);
        let dir = tempfile::tempdir().expect("tmp");
        let mut attempt = 0;
        loop {
            attempt += 1;
            let log = dir.path().join(format!("etcd-{attempt}.log"));
            let data = dir.path().join(format!("data-{attempt}"));
            let (mut child, endpoint) = spawn_etcd(&bin, &data, &log);
            let why = match ready(&mut child, &endpoint).await {
                Ok(()) => {
                    return Self {
                        child,
                        endpoint,
                        _dir: dir,
                    };
                }
                Err(Startup::Exited(_)) if attempt < 3 => continue,
                Err(Startup::Exited(status)) => format!("exited {status}, three times"),
                Err(Startup::Silent) => "no answer in 15 s".to_owned(),
            };
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "etcd did not come up at {endpoint}: {why}; its log:\n{}",
                std::fs::read_to_string(&log).unwrap_or_default()
            );
        }
    }

    /// Its endpoints, as a client takes them.
    pub fn endpoints(&self) -> Vec<String> {
        vec![self.endpoint.clone()]
    }

    /// A plain client of it.
    pub async fn client(&self) -> etcd_client::Client {
        etcd_client::Client::connect([self.endpoint.as_str()], None)
            .await
            .expect("connect")
    }
}

impl Drop for Etcd {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Whether the etcd `child` answers at `endpoint`, within 15 s.
async fn ready(child: &mut Child, endpoint: &str) -> Result<(), Startup> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(status) = child.try_wait().expect("wait") {
            return Err(Startup::Exited(status));
        }
        if let Ok(Ok(mut c)) = tokio::time::timeout(
            Duration::from_secs(1),
            etcd_client::Client::connect([endpoint], None),
        )
        .await
            && let Ok(Ok(_)) =
                tokio::time::timeout(Duration::from_secs(1), c.get("/ready", None)).await
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(Startup::Silent);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Why an etcd did not come up.
#[derive(Debug)]
enum Startup {
    /// It exited first, most often because another process took a port it was given.
    Exited(std::process::ExitStatus),
    /// It ran but did not answer within 15 s.
    Silent,
}

/// An etcd on two free loopback ports, its data in `data` and its log in `log`; its endpoint.
fn spawn_etcd(
    bin: &std::path::Path,
    data: &std::path::Path,
    log: &std::path::Path,
) -> (Child, String) {
    let (client, peer) = (free_port(), free_port());
    let client_url = format!("http://127.0.0.1:{client}");
    let peer_url = format!("http://127.0.0.1:{peer}");
    let log = std::fs::File::create(log).expect("log");
    let child = Command::new(bin)
        .args(["--name", "test", "--data-dir"])
        .arg(data)
        .args(["--listen-client-urls", &client_url])
        .args(["--advertise-client-urls", &client_url])
        .args(["--listen-peer-urls", &peer_url])
        .args(["--initial-advertise-peer-urls", &peer_url])
        .args(["--initial-cluster", &format!("test={peer_url}")])
        .args(["--log-level", "warn"])
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .unwrap_or_else(|e| {
            panic!(
                "the etcd lease tests need etcd ({}: {e}); install it, or set \
                 EVERSERVE_TEST_ETCD to its path",
                bin.display()
            )
        });
    (child, format!("127.0.0.1:{client}"))
}

/// A loopback port that was free when chosen.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .port()
}
