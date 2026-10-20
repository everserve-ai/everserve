// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The Everserve KV mirror binary:
//! `everserve-store --socket <path> --data-dir <dir> --slots <n> --budget-bytes <b>`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use everserve_store::service::Service;
use everserve_store::storage::Storage;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

/// Environment variable names.
mod environment_names {
    /// The mirror's Unix socket.
    pub const SOCKET: &str = "EVERSERVE_STORE_SOCKET";
    /// The shared-memory directory for staging files and tables.
    pub const DATA_DIR: &str = "EVERSERVE_STORE_DATA_DIR";
    /// Slot ids per layer table (the engine's blocks times its block size).
    pub const SLOTS: &str = "EVERSERVE_STORE_SLOTS";
    /// Memory budget for row tables, in bytes.
    pub const BUDGET_BYTES: &str = "EVERSERVE_STORE_BUDGET_BYTES";
    /// `tracing` filter.
    pub const LOG: &str = "EVERSERVE_LOG";
    /// How log lines are written: `text` or `json`.
    pub const LOG_FORMAT: &str = "EVERSERVE_LOG_FORMAT";
    /// Address for `/metrics` and `/healthz`; unset means no endpoint.
    pub const METRICS_ADDR: &str = "EVERSERVE_STORE_METRICS_ADDR";
    /// Where other stores connect to replicate into this one: `host:port` (TLS) or a Unix path.
    pub const REPLICA_LISTEN: &str = "EVERSERVE_STORE_REPLICA_LISTEN";
    pub const REPLICATION_QUEUE_BYTES: &str = "EVERSERVE_STORE_REPLICATION_QUEUE_BYTES";
    pub const REPLICATION_REPLY_TIMEOUT_S: &str = "EVERSERVE_STORE_REPLICATION_REPLY_TIMEOUT_S";
    /// The name on peer stores' certificates.
    pub const PEER_NAME: &str = "EVERSERVE_STORE_PEER_NAME";
    /// The node's compile cache root, the engines' `VLLM_CACHE_ROOT` (decision 0043).
    pub const COMPILE_CACHE_DIR: &str = "EVERSERVE_STORE_COMPILE_CACHE_DIR";
    /// The most bytes of compile caches the node keeps.
    pub const COMPILE_CACHE_BYTES: &str = "EVERSERVE_STORE_COMPILE_CACHE_BYTES";
    /// This store's TLS certificate chain (PEM).
    pub const TLS_CERT: &str = "EVERSERVE_TLS_CERT";
    /// This store's TLS private key (PEM).
    pub const TLS_KEY: &str = "EVERSERVE_TLS_KEY";
    /// The deployment's CA certificates (PEM).
    pub const TLS_CA: &str = "EVERSERVE_TLS_CA";
}

#[derive(Debug, Parser)]
#[command(name = "everserve-store", version, about = "Everserve KV mirror")]
struct Cli {
    /// How log lines are written to standard error: `text` for people, `json` for a log
    /// pipeline.
    #[arg(long, env = environment_names::LOG_FORMAT, default_value = "text")]
    log_format: everserve_telemetry::LogFormat,
    /// The Unix socket workers connect to.
    #[arg(long, env = environment_names::SOCKET)]
    socket: PathBuf,
    /// Directory for staging files and tables, on shared memory (`/dev/shm/everserve` on a node);
    /// workers map files under it, so it must be on the same host and readable by them.
    #[arg(long, env = environment_names::DATA_DIR)]
    data_dir: PathBuf,
    /// Slot ids per layer table: the engine's KV blocks times its block size.
    #[arg(long, env = environment_names::SLOTS)]
    slots: usize,
    /// Memory budget for row tables, in bytes.
    #[arg(long, env = environment_names::BUDGET_BYTES)]
    budget_bytes: usize,
    /// Address for `/metrics` and `/healthz`, e.g. `127.0.0.1:9109`. Unset: no endpoint.
    #[arg(long, env = environment_names::METRICS_ADDR)]
    metrics_addr: Option<std::net::SocketAddr>,
    /// Where other stores replicate into this one (`host:port` under TLS, or a Unix path).
    /// Unset: this store receives no replication.
    #[arg(long, env = environment_names::REPLICA_LISTEN)]
    replica_listen: Option<String>,
    /// Bytes each replication target's queue may hold before a commit waits for room; the
    /// table budget unless given.
    #[arg(long, env = environment_names::REPLICATION_QUEUE_BYTES)]
    replication_queue_bytes: Option<usize>,
    /// Seconds a replication carrier waits for a replica's reply before taking the link as
    /// failed and reconnecting; 30 unless given.
    #[arg(long, env = environment_names::REPLICATION_REPLY_TIMEOUT_S)]
    replication_reply_timeout_s: Option<u64>,
    /// The name on peer stores' certificates, for TCP targets.
    #[arg(long, env = environment_names::PEER_NAME, default_value = "store")]
    peer_name: String,
    /// The node's compile cache root, the engines' `VLLM_CACHE_ROOT` (decision 0043): the store
    /// serves a rank's cache from it to peer stores, and fetches one into it for a spare. Unset:
    /// the store keeps no compile caches.
    #[arg(long, env = environment_names::COMPILE_CACHE_DIR, requires = "compile_cache_bytes")]
    compile_cache_dir: Option<PathBuf>,
    /// The most bytes of compile caches the node keeps; a fetch past it is refused.
    #[arg(long, env = environment_names::COMPILE_CACHE_BYTES)]
    compile_cache_bytes: Option<u64>,
    /// This store's certificate chain (PEM), for TCP replication.
    #[arg(long, env = environment_names::TLS_CERT)]
    tls_cert: Option<PathBuf>,
    /// This store's private key (PEM).
    #[arg(long, env = environment_names::TLS_KEY)]
    tls_key: Option<PathBuf>,
    /// The CA certificates peer stores must chain to (PEM).
    #[arg(long, env = environment_names::TLS_CA)]
    tls_ca: Option<PathBuf>,
}

/// The node's compile caches the settings name (decision 0043), fetched from peer stores with
/// `tls`; none without a cache directory. `--compile-cache-dir` requires `--compile-cache-bytes`.
fn compile_caches(
    cli: &Cli,
    tls: Option<&everserve_wire::transport::Tls>,
) -> Option<everserve_store::cache::CompileCaches> {
    let dir = cli.compile_cache_dir.as_ref()?;
    Some(everserve_store::cache::CompileCaches::new(
        dir,
        cli.compile_cache_bytes.unwrap_or(0),
        tls.cloned(),
        &cli.peer_name,
    ))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    everserve_telemetry::init_logs(
        cli.log_format,
        tracing_subscriber::EnvFilter::try_from_env(environment_names::LOG)
            .or_else(|_| tracing_subscriber::EnvFilter::try_from_default_env())
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    );
    // what this store makes (its socket, its directories, its tables) is its node group's too:
    // the engines' processes on the node open the socket and write into the rank directories
    everserve_wire::host::share_what_is_made();
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            error!("tokio runtime: {e}");
            return ExitCode::from(3);
        }
    };
    rt.block_on(async move {
        let cancel = CancellationToken::new();
        let telemetry = everserve_telemetry::Telemetry::new();
        if let Some(addr) = cli.metrics_addr
            && let Err(e) = telemetry
                .clone()
                .start(addr, everserve_telemetry::HEADER_READ, cancel.clone())
                .await
        {
            error!("metrics on {addr}: {e}");
            return ExitCode::from(2);
        }
        if let Err(e) = std::fs::create_dir_all(&cli.data_dir) {
            error!("data dir {}: {e}", cli.data_dir.display());
            return ExitCode::from(3);
        }
        let tls = match tls_of(&cli) {
            Ok(t) => t,
            Err(e) => {
                error!("{e}");
                return ExitCode::from(2);
            }
        };
        if let Some(tls) = &tls {
            follow_tls(tls.clone(), cancel.clone());
        }
        let storage = Storage::new(cli.data_dir.clone(), cli.slots, cli.budget_bytes);
        match storage.sweep_stages() {
            Ok(0) => {}
            Ok(n) => info!("removed {n} staging files dead workers left"),
            Err(e) => {
                error!("{e}");
                return ExitCode::from(2);
            }
        }
        if let Err(e) = room_for(&storage, &cli.data_dir) {
            error!("{e}");
            return ExitCode::from(2);
        }
        let caches = compile_caches(&cli, tls.as_ref());
        let service = Service::with_tls(storage, &telemetry, tls.clone(), &cli.peer_name, caches);
        if let Some(bytes) = cli.replication_queue_bytes {
            service.set_replication_queue_bytes(bytes);
        }
        if let Some(secs) = cli.replication_reply_timeout_s {
            service.set_replication_reply_timeout(std::time::Duration::from_secs(secs));
        }
        if let Some(listen) = &cli.replica_listen {
            let address = match everserve_wire::transport::Address::parse(listen) {
                Ok(a) => a,
                Err(e) => {
                    error!("{e}");
                    return ExitCode::from(2);
                }
            };
            let listener =
                match everserve_wire::transport::Listener::bind(&address, tls.as_ref()).await {
                    Ok(l) => l,
                    Err(e) => {
                        error!("{e}");
                        return ExitCode::from(2);
                    }
                };
            tokio::spawn(std::sync::Arc::clone(&service).serve_peers(listener, cancel.clone()));
        }
        let server = tokio::spawn(service.serve(cli.socket, cancel.clone()));
        let signal_cancel = cancel.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            info!("shutdown signal received; stopping the mirror");
            signal_cancel.cancel();
        });
        match server.await {
            Ok(Ok(())) => ExitCode::SUCCESS,
            Ok(Err(e)) => {
                error!("mirror failed: {e}");
                ExitCode::from(3)
            }
            Err(e) => {
                error!("mirror task: {e}");
                ExitCode::from(3)
            }
        }
    })
}

/// Read `tls`'s files again every [`TLS_RELOAD_EVERY`](everserve_wire::transport::TLS_RELOAD_EVERY)
/// until `cancel` fires, so a certificate renewed on disk is the one the next handshake uses
/// (decision 0037).
fn follow_tls(tls: everserve_wire::transport::Tls, cancel: CancellationToken) {
    tokio::spawn(async move {
        tokio::select! {
            () = cancel.cancelled() => {}
            () = tls.follow(everserve_wire::transport::TLS_RELOAD_EVERY) => {}
        }
    });
}

/// Whether the file system holding `data_dir` has room for `storage`'s budget: what it has free,
/// and what an earlier run's tables hold there, which this run makes again in place. A table is
/// mapped and touched page by page; a page the file system has no room for is a SIGBUS in the
/// store, so a budget that cannot fit is refused at start, not found at a commit.
fn room_for(storage: &Storage, data_dir: &std::path::Path) -> Result<(), String> {
    let free = everserve_wire::host::free_bytes(data_dir)
        .map_err(|e| format!("data dir {}: {e}", data_dir.display()))?;
    let held = storage.held_bytes().map_err(|e| e.to_string())?;
    let budget = u64::try_from(storage.budget()).unwrap_or(u64::MAX);
    // tables in shared memory count against the container's memory limit: a budget above it is
    // the kernel's OOM kill at a commit, refused here instead
    let limit = everserve_wire::host::memory_limit_at(std::path::Path::new(
        everserve_wire::host::CGROUP_MEMORY_MAX,
    ))
    .map_err(|e| format!("the memory limit: {e}"))?;
    within_memory(budget, limit)?;
    if free.saturating_add(held) < budget {
        return Err(format!(
            "data dir {}: its file system has room for {} bytes of tables ({free} free, {held} \
             held by an earlier run's), below the budget of {budget}: give the node's shared \
             memory more room, or lower --budget-bytes",
            data_dir.display(),
            free.saturating_add(held),
        ));
    }
    Ok(())
}

/// The TLS material `cli` names: all three files, or none.
fn tls_of(cli: &Cli) -> Result<Option<everserve_wire::transport::Tls>, String> {
    match (&cli.tls_cert, &cli.tls_key, &cli.tls_ca) {
        (None, None, None) => Ok(None),
        (Some(c), Some(k), Some(a)) => everserve_wire::transport::Tls::load(c, k, a)
            .map(Some)
            .map_err(|e| e.to_string()),
        _ => Err("give all of --tls-cert, --tls-key and --tls-ca, or none".to_owned()),
    }
}

/// Whether a `budget` of table bytes fits a memory `limit` (None: no limit), the tables in
/// shared memory counting against it, with room left for the store itself.
fn within_memory(budget: u64, limit: Option<u64>) -> Result<(), String> {
    match limit {
        Some(limit) if budget >= limit => Err(format!(
            "a budget of {budget} bytes of tables, which count against this store's memory \
             limit of {limit}: give the store more memory than its budget (the chart's \
             node.store.memoryOverheadBytes above node.store.budgetBytes), or lower \
             --budget-bytes"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::within_memory;

    #[test]
    fn a_budget_must_fit_under_the_memory_limit_its_tables_count_against() {
        // a budget of 1 GiB under a 512 MiB limit: the kernel kills the store at its first
        // commits
        let err = within_memory(1 << 30, Some(512 << 20)).unwrap_err();
        assert!(err.contains("memory limit of 536870912"), "{err}");
        assert!(within_memory(1 << 30, Some((1 << 30) + (512 << 20))).is_ok());
        assert!(within_memory(1 << 30, None).is_ok(), "no limit");
    }
}
