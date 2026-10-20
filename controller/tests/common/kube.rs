// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A throwaway Kubernetes API server for the tests that need a real one: envtest's
//! `kube-apiserver` (on `PATH`, or at `EVERSERVE_TEST_KUBE_APISERVER`) over a test etcd, serving
//! TLS under a CA the controller's own `certs` writes, with a static bearer token in
//! `system:masters`. A test that needs one fails without it.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use everserve_controller::config::CertsArgs;
use everserve_controller::kubernetes::Access;

use super::Spawned;
use super::etcd::{Etcd, free_port};

/// The bearer token the server takes.
const TOKEN: &str = "everserve-test-token";

/// A running API server, stopped with its etcd when dropped.
pub struct ApiServer {
    /// How to reach it.
    pub access: Access,
    /// The `kube-apiserver` binary it runs, beside which its release's `kubectl` lies.
    bin: PathBuf,
    _server: Spawned,
    _etcd: Etcd,
    _dir: tempfile::TempDir,
}

impl ApiServer {
    /// Start one, ready once `/readyz` says every post-start hook has run (the bootstrap RBAC
    /// roles, `system:auth-delegator` among them, included) and its `default` namespace exists.
    pub async fn start() -> Self {
        let bin = std::env::var_os("EVERSERVE_TEST_KUBE_APISERVER")
            .map_or_else(|| "kube-apiserver".into(), PathBuf::from);
        let etcd = Etcd::start().await;
        let dir = tempfile::tempdir().expect("tmp");
        let files = Files::write(dir.path());
        let mut attempt = 0;
        loop {
            attempt += 1;
            let port = free_port();
            let log = dir.path().join(format!("apiserver-{attempt}.log"));
            let server = spawn(&bin, &etcd, &files, port, &log);
            let access = Access {
                endpoint: format!("https://localhost:{port}"),
                token_file: files.token.clone(),
                ca_file: files.ca.clone(),
            };
            match ready(server, &access).await {
                Ok(server) => {
                    return Self {
                        access,
                        bin: bin.clone(),
                        _server: server,
                        _etcd: etcd,
                        _dir: dir,
                    };
                }
                // another process took the port between its choice and the bind
                Err(Exited) if attempt < 3 => {}
                Err(why) => panic!(
                    "kube-apiserver did not come up ({why:?}); its log:\n{}",
                    std::fs::read_to_string(&log).unwrap_or_default()
                ),
            }
        }
    }
}

impl ApiServer {
    /// `kubectl` of the server's own release, pointed at it as the administrator: found beside
    /// the `kube-apiserver` binary (envtest ships them together), or at `EVERSERVE_TEST_KUBECTL`.
    pub fn kubectl(&self) -> Command {
        let bin = std::env::var_os("EVERSERVE_TEST_KUBECTL").map_or_else(
            || {
                let server = which(&self.bin).unwrap_or_else(|| self.bin.clone());
                let server = std::fs::canonicalize(&server).unwrap_or(server);
                server.with_file_name("kubectl")
            },
            PathBuf::from,
        );
        let mut c = Command::new(bin);
        c.arg(format!("--server={}", self.access.endpoint))
            .arg(format!("--token={TOKEN}"))
            .arg(format!(
                "--certificate-authority={}",
                self.access.ca_file.display()
            ));
        c
    }

    /// Run `kubectl` with `args` and `stdin`; its output, failing the test if it fails.
    pub fn kubectl_ok(&self, args: &[&str], stdin: Option<&str>) -> String {
        use std::io::Write as _;
        let mut child = self
            .kubectl()
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("kubectl");
        if let Some(input) = stdin {
            child
                .stdin
                .take()
                .expect("stdin")
                .write_all(input.as_bytes())
                .expect("write");
        }
        let out = child.wait_with_output().expect("kubectl");
        assert!(
            out.status.success(),
            "kubectl {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }
}

/// `bin` as found on `PATH`, when it names no directory.
fn which(bin: &Path) -> Option<PathBuf> {
    if bin.components().count() > 1 {
        return Some(bin.to_owned());
    }
    std::env::var_os("PATH")?
        .to_str()?
        .split(':')
        .map(|d| Path::new(d).join(bin))
        .find(|p| p.is_file())
}

/// The files the server and its clients read.
struct Files {
    ca: PathBuf,
    cert: PathBuf,
    key: PathBuf,
    service_account_key: PathBuf,
    service_account_public: PathBuf,
    tokens: PathBuf,
    token: PathBuf,
}

impl Files {
    fn write(dir: &Path) -> Self {
        let certs = dir.join("certs");
        everserve_controller::certs::write(&CertsArgs {
            out: certs.clone(),
            nodes: vec!["localhost".into()],
            operators: vec![],
            coordinator_name: "coordinator".into(),
            store_name: "store".into(),
            days: 1,
            overwrite: false,
        })
        .expect("certs");
        let service_account_key = dir.join("service-account.key");
        let key = rcgen::KeyPair::generate().expect("key");
        std::fs::write(&service_account_key, key.serialize_pem()).expect("write");
        let service_account_public = dir.join("service-account.pub");
        std::fs::write(&service_account_public, key.public_key_pem()).expect("write");
        let tokens = dir.join("tokens.csv");
        std::fs::write(&tokens, format!("{TOKEN},admin,admin,\"system:masters\"\n"))
            .expect("write");
        let token = dir.join("token");
        std::fs::write(&token, TOKEN).expect("write");
        Self {
            ca: certs.join("ca.crt"),
            cert: certs.join("localhost.crt"),
            key: certs.join("localhost.key"),
            service_account_key,
            service_account_public,
            tokens,
            token,
        }
    }
}

fn spawn(bin: &Path, etcd: &Etcd, files: &Files, port: u16, log: &Path) -> Spawned {
    let log = std::fs::File::create(log).expect("log");
    let arg = |name: &str, path: &Path| format!("--{name}={}", path.display());
    Command::new(bin)
        .arg(format!("--etcd-servers=http://{}", etcd.endpoint))
        .arg("--bind-address=127.0.0.1")
        .arg(format!("--secure-port={port}"))
        .arg(arg("tls-cert-file", &files.cert))
        .arg(arg("tls-private-key-file", &files.key))
        .arg(arg("token-auth-file", &files.tokens))
        .arg("--authorization-mode=RBAC")
        .arg("--service-account-issuer=https://localhost")
        .arg(arg(
            "service-account-key-file",
            &files.service_account_public,
        ))
        .arg(arg(
            "service-account-signing-key-file",
            &files.service_account_key,
        ))
        .arg("--service-cluster-ip-range=10.0.0.0/24")
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .map_or_else(
            |e| {
                panic!(
                    "the Kubernetes lease tests need envtest's kube-apiserver ({}: {e}); install \
                     it, or set EVERSERVE_TEST_KUBE_APISERVER to its path",
                    bin.display()
                )
            },
            Spawned,
        )
}

/// Why a server did not come up.
#[derive(Debug)]
enum Startup {
    /// It exited first.
    Exited,
    /// It ran but was not ready within a minute.
    Silent,
}
use Startup::Exited;

/// `server`, once it answers for its `default` namespace.
async fn ready(mut server: Spawned, access: &Access) -> Result<Spawned, Startup> {
    let client = access.client().expect("client");
    let deadline = Instant::now() + Duration::from_mins(1);
    loop {
        if server.0.try_wait().expect("wait").is_some() {
            return Err(Startup::Exited);
        }
        let ok = |path: &str| {
            let (client, url) = (&client, access.url(path));
            async move { client.get(&url, &[]).await.is_ok_and(|a| a.status == 200) }
        };
        if ok("/readyz").await && ok("/api/v1/namespaces/default").await {
            return Ok(server);
        }
        if Instant::now() >= deadline {
            return Err(Startup::Silent);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
