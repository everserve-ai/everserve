// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator's commands: parse the command line, run the selected role, shut down cleanly.
//!
//! A `SIGINT` or `SIGTERM` cancels the role; in-flight changes get the graceful period, then the
//! process exits with [`EXIT_GRACEFUL_TIMEOUT`] if they are still running. A coordinator whose
//! hold of the lease ends before it stops exits with [`EXIT_LEASE_LOST`].

mod operator;

use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::config::{
    CertsArgs, CheckArgs, Cli, Command, CoordArgs, NodeArgs, NodeCertArgs, NodeDirsArgs,
    environment_names,
};
use crate::coord::Coordinator;
use crate::lease::{Hold, LeaseError};
use crate::node::Daemon;
use crate::policy;
use crate::proto::everserve::coord::v1::LeaseRecord;
use crate::proto::everserve::policy::v1::Policy;
use crate::spec::kubernetes::KubernetesStore;
use crate::spec::{FileStore, SpecError, SpecStore, StoreAt};
use everserve_wire::transport::{Address, Listener, TLS_RELOAD_EVERY, Tls};

/// Exit code when the graceful period ran out: 124, the convention `timeout(1)` uses, and a value
/// that survives the 8-bit exit status (Dynamo's 911 arrives at the shell as 143).
pub const EXIT_GRACEFUL_TIMEOUT: u8 = 124;

/// Exit code of a coordinator whose hold of the lease ended before it stopped: 75, `EX_TEMPFAIL`.
/// It claimed nothing from the moment the hold ended; its supervisor restarts it, and it stands by
/// until the lease is free (decision 0036).
pub const EXIT_LEASE_LOST: u8 = 75;

/// Parse the command line and run.
#[must_use]
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    everserve_telemetry::init_logs(
        cli.log_format,
        tracing_subscriber::EnvFilter::try_from_env(environment_names::LOG)
            .or_else(|_| tracing_subscriber::EnvFilter::try_from_default_env())
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    );
    match cli.command {
        Command::Node(args) => run_node(&args),
        Command::Coord(args) => run_coord(&args),
        Command::Certs(args) => run_certs(&args),
        Command::Check(args) => run_check(&args),
        Command::NodeCert(args) => run_node_cert(&args),
        Command::NodeDirs(args) => run_node_dirs(&args),
        Command::Openapi => run_openapi(),
        Command::Get(args) => operator::run_get(&args),
        Command::Watch(args) => operator::run_watch(&args),
        Command::Cordon(args) => operator::run_cordon(&args, true),
        Command::Uncordon(args) => operator::run_cordon(&args, false),
        Command::Drain(args) => operator::run_drain(&args),
        Command::Undrain(args) => operator::run_undrain(&args),
    }
}

fn run_check(args: &CheckArgs) -> ExitCode {
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(code) => return code,
    };
    match rt.block_on(crate::check::run(args)) {
        Ok(report) => {
            for line in &report.lines {
                println!("{line}");
            }
            if report.ok {
                println!("ready");
                ExitCode::SUCCESS
            } else {
                println!("not ready after {}s", args.timeout_secs);
                ExitCode::from(1)
            }
        }
        Err(e) => {
            error!("{e}");
            ExitCode::from(2)
        }
    }
}

fn run_certs(args: &CertsArgs) -> ExitCode {
    match crate::certs::write(args) {
        Ok(written) => {
            for path in written.files {
                info!("wrote {}", path.display());
            }
            for path in written.kept {
                info!("kept {} (--overwrite replaces it)", path.display());
            }
            if written.ca_reused {
                info!("reused the CA in {}", args.out.display());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            error!("{e}");
            ExitCode::from(2)
        }
    }
}

fn run_node_dirs(args: &NodeDirsArgs) -> ExitCode {
    for dir in &args.dirs {
        if let Err(e) = everserve_wire::host::share(dir, args.group) {
            error!("{}: {e}", dir.display());
            return ExitCode::from(2);
        }
        info!("{} is group {}'s, mode 2770", dir.display(), args.group);
    }
    ExitCode::SUCCESS
}

fn run_node_cert(args: &NodeCertArgs) -> ExitCode {
    let enrollment = match crate::enroll::Enrollment::new(
        &args.coordinator_api,
        &args.coordinator_name,
        &args.ca,
        &args.token_file,
        &args.out,
    ) {
        Ok(e) => e,
        Err(e) => {
            error!("{e}");
            return ExitCode::from(2);
        }
    };
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(code) => return code,
    };
    rt.block_on(async move {
        let cancel = CancellationToken::new();
        tokio::spawn(shutdown_on_signal(cancel.clone(), "node certificate"));
        if args.once {
            // before the daemon and the store: wait out a coordinator not yet serving
            return match enrollment.issued(i64::MAX, &cancel).await {
                Some(_) => ExitCode::SUCCESS,
                None => ExitCode::from(1),
            };
        }
        enrollment.keep(&cancel).await;
        ExitCode::SUCCESS
    })
}

fn runtime() -> Result<tokio::runtime::Runtime, ExitCode> {
    tokio::runtime::Runtime::new().map_err(|e| {
        error!("tokio runtime: {e}");
        ExitCode::from(3)
    })
}

fn run_node(args: &NodeArgs) -> ExitCode {
    // what the daemon makes (its socket above all) is its node group's too: the engines'
    // processes on the node connect to it
    everserve_wire::host::share_what_is_made();
    let tls = match args.tls.load() {
        Ok(t) => t,
        Err(e) => {
            error!("{e}");
            return ExitCode::from(2);
        }
    };
    let coordinator = match args.coordinator.as_deref().map(Address::parse).transpose() {
        Ok(a) => a,
        Err(e) => {
            error!("{e}");
            return ExitCode::from(2);
        }
    };
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(code) => return code,
    };
    rt.block_on(async move {
        let cancel = CancellationToken::new();
        follow_tls(tls.as_ref(), &cancel);
        let telemetry = everserve_telemetry::Telemetry::new();
        if let Err(code) = serve_metrics(&telemetry, args.metrics_addr, &cancel).await {
            return code;
        }
        let (daemon, _committed) = Daemon::new(&args.node_name(), &telemetry);
        daemon
            .set_heartbeat(std::time::Duration::from_millis(args.heartbeat_ms))
            .await;
        if let Some(socket) = &args.mirror_socket {
            daemon
                .set_mirror(socket.clone(), args.mirror_replica.clone())
                .await;
        }
        if let Some(dir) = &args.discovery_dir
            && let Err(e) = daemon
                .publish_to(
                    dir,
                    &args.socket,
                    args.mirror_socket.as_deref(),
                    args.advertise_host.as_deref(),
                )
                .await
        {
            error!("{e}");
            return ExitCode::from(2);
        }
        if let Err(e) = start_link(&daemon, args, coordinator, tls, &cancel).await {
            error!("{e}");
            return ExitCode::from(2);
        }
        tokio::spawn(crate::notice::run(
            args.notice(),
            Arc::clone(&daemon),
            cancel.clone(),
        ));
        let server = tokio::spawn(Arc::clone(&daemon).serve(args.socket.clone(), cancel.clone()));
        tokio::spawn(shutdown_on_signal(cancel.clone(), "daemon"));
        match server.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                error!("daemon failed: {e}");
                return ExitCode::from(3);
            }
            Err(e) => {
                error!("daemon task: {e}");
                return ExitCode::from(3);
            }
        }
        if tokio::time::timeout(args.graceful_shutdown(), daemon.drained())
            .await
            .is_err()
        {
            error!(
                "deliveries still in flight after {}s; exiting (set {})",
                args.graceful_shutdown_secs,
                environment_names::GRACEFUL_SHUTDOWN_SECS
            );
            return ExitCode::from(EXIT_GRACEFUL_TIMEOUT);
        }
        info!("daemon stopped");
        ExitCode::SUCCESS
    })
}

fn run_coord(args: &CoordArgs) -> ExitCode {
    let policy = match policy::load(&args.policy) {
        Ok(p) => p,
        Err(e) => {
            error!("{e}");
            return ExitCode::from(2);
        }
    };
    let tls = match args.tls.load() {
        Ok(t) => t,
        Err(e) => {
            error!("{e}");
            return ExitCode::from(2);
        }
    };
    let listen = match Address::parse(&args.listen) {
        Ok(a) => a,
        Err(e) => {
            error!("{e}");
            return ExitCode::from(2);
        }
    };
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(code) => return code,
    };
    rt.block_on(async move {
        let cancel = CancellationToken::new();
        tokio::spawn(shutdown_on_signal(cancel.clone(), "coordinator"));
        follow_tls(tls.as_ref(), &cancel);
        // health and metrics from the start: a standby is a pod that stands, and says so
        let telemetry = everserve_telemetry::Telemetry::new();
        if let Err(code) = serve_metrics(&telemetry, args.metrics_addr, &cancel).await {
            return code;
        }
        // a restarted pod may still carry the holder's label from its last life
        mark_holder(args, false).await;
        let hold = match hold_lease(args, &cancel).await {
            Ok(Some(hold)) => hold,
            Ok(None) => return ExitCode::SUCCESS,
            Err(code) => return code,
        };
        mark_holder(args, true).await;
        let Ok(coord) = coordinator(policy, &telemetry, args, hold.term())
            .await
            .map_err(|e| error!("{e}"))
        else {
            return ExitCode::from(2);
        };
        if let Err(e) = coord.act_under(hold.term(), hold.ended().clone()) {
            error!("{e}");
            return ExitCode::from(2);
        }
        if let Err(e) = publish(hold.as_ref(), &record_of(args)).await {
            error!("{e}");
            return ExitCode::from(2);
        }
        // a hold that ends stops the coordinator; its supervisor restarts it to contend again
        tokio::spawn(stop_when_ended(hold.ended().clone(), cancel.clone()));
        coord.set_node_loss_misses(args.node_loss_misses);
        // a write may take as long as the hold lasts past a renewal, never longer
        coord.set_store_write_deadline(crate::coord::store_write_deadline_for(
            std::time::Duration::from_secs(args.lease_ttl_secs),
        ));
        if args.nodes.kubernetes_nodes {
            coord.follow_cluster(args.kubernetes.access(), args.nodes.labels(), &cancel);
        }
        if let Err(e) = start_api(args, &coord, tls.as_ref(), &cancel).await {
            error!("{e}");
            return ExitCode::from(2);
        }
        let listener = match Listener::bind(&listen, tls.as_ref()).await {
            Ok(l) => l,
            Err(e) => {
                error!("{e}");
                return ExitCode::from(2);
            }
        };
        let outcome = serve_links(&coord, listener, &cancel).await;
        // whatever ended the serving (a stop, a hold that ended), the Service must not select
        // this pod any longer
        mark_holder(args, false).await;
        if let Err(code) = outcome {
            return code;
        }
        let code = if tokio::time::timeout(args.graceful_shutdown(), coord.drained())
            .await
            .is_err()
        {
            error!(
                "changes still in flight after {}s; exiting (set {})",
                args.graceful_shutdown_secs,
                environment_names::GRACEFUL_SHUTDOWN_SECS
            );
            ExitCode::from(EXIT_GRACEFUL_TIMEOUT)
        } else {
            info!("coordinator stopped");
            ExitCode::SUCCESS
        };
        let_go(hold).await.unwrap_or(code)
    })
}

/// Link `daemon` to its coordinator: the one the lease it follows names, its term fencing every
/// other (decision 0036), or the one `coordinator` gives; with neither, reports are held.
///
/// # Errors
///
/// The lease's etcd TLS files cannot be read, or its endpoints are not valid.
async fn start_link(
    daemon: &Arc<Daemon>,
    args: &NodeArgs,
    coordinator: Option<Address>,
    tls: Option<Tls>,
    cancel: &CancellationToken,
) -> Result<(), LeaseError> {
    if let Some(lease) = &args.lease_watch {
        let watch = lease
            .watch(
                LEASE_READ_TIMEOUT,
                &crate::config::reach(&args.etcd, &args.kubernetes)?,
            )
            .await?;
        let (target, followed) = tokio::sync::watch::channel(None);
        tokio::spawn(Arc::clone(daemon).watch_lease(
            watch,
            target,
            LEASE_WATCH_EVERY,
            cancel.clone(),
        ));
        tokio::spawn(Arc::clone(daemon).link_to(
            followed,
            tls,
            args.coordinator_name.clone(),
            cancel.clone(),
        ));
    } else if let Some(address) = coordinator {
        tokio::spawn(Arc::clone(daemon).link(
            address,
            tls,
            args.coordinator_name.clone(),
            cancel.clone(),
        ));
    } else {
        warn!(
            "no coordinator given ({} or {}): reports will be held until one is",
            environment_names::COORDINATOR,
            environment_names::LEASE_WATCH
        );
    }
    Ok(())
}

/// This coordinator's lease record: its host and process as the holder, and the address daemons
/// link to it at.
fn record_of(args: &CoordArgs) -> LeaseRecord {
    LeaseRecord {
        holder: format!("{}:{}", crate::config::hostname(), std::process::id()),
        address: args
            .advertise
            .clone()
            .unwrap_or_else(|| args.listen.clone()),
        ..LeaseRecord::default()
    }
}

/// How often a daemon reads the lease it follows.
const LEASE_WATCH_EVERY: std::time::Duration = std::time::Duration::from_millis(500);

/// How long a daemon's read of the lease may take before it fails and the next read is tried.
const LEASE_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Publish `record` as `hold`'s, stamped now: who holds the lease and where daemons link to it
/// (decision 0036).
async fn publish(hold: &dyn Hold, record: &LeaseRecord) -> Result<(), LeaseError> {
    let record = LeaseRecord {
        renewed_at_ms: crate::clock::now_ms(),
        ..record.clone()
    };
    hold.publish(record).await?;
    info!("published the lease record: {}", hold.describe());
    Ok(())
}

/// Serve the daemons' links on `listener` until `cancel` fires; the exit code when the serving
/// itself failed.
async fn serve_links(
    coord: &Arc<Coordinator>,
    listener: Listener,
    cancel: &CancellationToken,
) -> Result<(), ExitCode> {
    match tokio::spawn(Arc::clone(coord).serve(listener, cancel.clone())).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => {
            error!("coordinator failed: {e}");
            Err(ExitCode::from(3))
        }
        Err(e) => {
            error!("coordinator task: {e}");
            Err(ExitCode::from(3))
        }
    }
}

/// The label the lease holder sets on its own pod, which the chart's Service selects by.
pub const HOLDER_LABEL: &str = "everserve.ai/lease-holder";

/// Label this pod as the holder, or clear the label, where the coordinator runs under a
/// `kubernetes://` lease and knows its pod's name (`--pod-name`); nothing elsewhere. A label
/// that cannot be set is logged: the Service then selects no pod until it can, which an operator
/// sees, rather than a standby.
async fn mark_holder(args: &CoordArgs, holding: bool) {
    let Some(pod) = &args.pod_name else { return };
    let crate::lease::LeaseSpec::Kubernetes { namespace, .. } = &args.lease else {
        return;
    };
    let pods = match crate::kubernetes::objects::Objects::core(
        &args.kubernetes.access(),
        "pods",
        namespace,
        HOLDER_LABEL_TIMEOUT,
    ) {
        Ok(pods) => pods,
        Err(e) => {
            warn!("the holder label on pod {pod} cannot be set: {e}");
            return;
        }
    };
    let patch = serde_json::json!({
        "metadata": {"labels": {HOLDER_LABEL: holding.then_some("true")}}
    });
    match pods.merge_patch(pod, &patch).await {
        Ok(_) if holding => info!("pod {pod} labelled {HOLDER_LABEL}=true: the Service selects it"),
        Ok(_) => info!("pod {pod}: the {HOLDER_LABEL} label cleared"),
        Err(e) => warn!("the holder label on pod {pod} could not be written: {e}"),
    }
}

/// How long a holder label's write may take.
const HOLDER_LABEL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Stand by until the lease `args` names is this coordinator's. `None` when `cancel` fires
/// first.
async fn hold_lease(
    args: &CoordArgs,
    cancel: &CancellationToken,
) -> Result<Option<Box<dyn Hold>>, ExitCode> {
    let fail = |e: LeaseError| {
        error!("{e}");
        ExitCode::from(2)
    };
    let reach = crate::config::reach(&args.etcd, &args.kubernetes).map_err(fail)?;
    let ttl = std::time::Duration::from_secs(args.lease_ttl_secs);
    let holder = record_of(args).holder;
    let lease = args.lease.lease(ttl, &holder, &reach).await.map_err(fail)?;
    let hold = crate::lease::campaign(lease.as_ref(), cancel).await;
    if let Some(hold) = &hold {
        info!("holding the {}", hold.describe());
    }
    Ok(hold)
}

/// Stop the coordinator through `cancel` when its hold of the lease `ended`.
async fn stop_when_ended(ended: CancellationToken, cancel: CancellationToken) {
    tokio::select! {
        () = ended.cancelled() => {
            error!("the lease is no longer this coordinator's: stopping, exit {EXIT_LEASE_LOST}");
            cancel.cancel();
        }
        () = cancel.cancelled() => {}
    }
}

/// The coordinator's guarded work has finished: unless the hold ended first, say no one holds
/// the lease, keeping the term, and let it go. [`EXIT_LEASE_LOST`] if the hold had ended.
async fn let_go(hold: Box<dyn Hold>) -> Option<ExitCode> {
    if hold.ended().is_cancelled() {
        return Some(ExitCode::from(EXIT_LEASE_LOST));
    }
    if let Err(e) = publish(hold.as_ref(), &LeaseRecord::default()).await {
        warn!("{e}");
    }
    if let Err(e) = hold.release().await {
        warn!("{e}");
    }
    None
}

/// Serve the operator API `args` asks for (decision 0039): under `tls`, behind the gates its
/// settings name. None asked: nothing served.
///
/// # Errors
///
/// An API address without TLS material or without any gate, or a Kubernetes gate without a
/// namespace or with an unreadable CA.
async fn start_api(
    args: &CoordArgs,
    coord: &Arc<Coordinator>,
    tls: Option<&Tls>,
    cancel: &CancellationToken,
) -> Result<(), String> {
    let Some(addr) = args.api.api_addr else {
        return Ok(());
    };
    let tls = tls.ok_or("the operator API needs TLS material (--tls-cert, --tls-key, --tls-ca)")?;
    let api = &args.api;
    let certificates = (!api.api_operators.is_empty() || !api.api_viewers.is_empty()).then(|| {
        crate::api::CertificateRoles {
            operators: api.api_operators.clone(),
            viewers: api.api_viewers.clone(),
        }
    });
    let kubernetes = if api.api_kubernetes {
        let namespace = api
            .namespace
            .clone()
            .ok_or("--api-kubernetes reviews rights in a namespace: set --namespace")?;
        Some(
            crate::api::KubernetesGate::new(args.kubernetes.access(), namespace)
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    if certificates.is_none() && kubernetes.is_none() {
        return Err(
            "the operator API admits no one: name --api-operators or --api-viewers, or set \
             --api-kubernetes"
                .into(),
        );
    }
    let gate: Arc<dyn crate::api::Gate> = Arc::new(crate::api::Gates {
        certificates,
        kubernetes,
    });
    let open = if args.issue.issue_node_certs {
        crate::issuer::router(Arc::new(issuer(args)?))
    } else {
        axum::Router::new()
    };
    crate::api::start(
        crate::api::routes::router(crate::api::Feed::start(
            Arc::clone(coord),
            crate::api::Pacing::SERVED,
            cancel,
        )),
        open,
        addr,
        tls,
        gate,
        crate::api::Deadlines::SERVED,
        cancel.clone(),
    )
    .await
    .map(drop)
    .map_err(|e| format!("operator API on {addr}: {e}"))
}

/// Print the operator API's description.
fn run_openapi() -> ExitCode {
    match crate::api::routes::document().to_pretty_json() {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            error!("the API's description: {e}");
            ExitCode::from(2)
        }
    }
}

/// Serve `telemetry`'s metrics and health on `addr`, if one is set, until `cancel` fires; a port
/// that cannot be bound stops the process.
async fn serve_metrics(
    telemetry: &everserve_telemetry::Telemetry,
    addr: Option<std::net::SocketAddr>,
    cancel: &CancellationToken,
) -> Result<(), ExitCode> {
    let Some(addr) = addr else { return Ok(()) };
    telemetry
        .clone()
        .start(addr, everserve_telemetry::HEADER_READ, cancel.clone())
        .await
        .map(drop)
        .map_err(|e| {
            error!("metrics on {addr}: {e}");
            ExitCode::from(2)
        })
}

/// The node certificate issuer `args` set (decision 0037): the deployment's CA from its key and
/// name, the CA certificate the TLS material names, and the node daemons' service account in the
/// coordinator's namespace.
fn issuer(args: &CoordArgs) -> Result<crate::issuer::Issuer, String> {
    let issue = &args.issue;
    let key_path = issue
        .issue_ca_key
        .as_ref()
        .ok_or("--issue-node-certs signs with the deployment CA's key: set --issue-ca-key")?;
    let ca_path = args
        .tls
        .tls_ca
        .as_ref()
        .ok_or("--issue-node-certs hands nodes the CA in --tls-ca: set it")?;
    let read = |p: &std::path::Path| {
        std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))
    };
    let ca = crate::certs::authority_from(&read(key_path)?, &issue.issue_ca_name)
        .map_err(|e| e.to_string())?;
    let namespace = args.api.namespace.clone().ok_or(
        "--issue-node-certs reviews tokens of the node daemons' namespace: set --namespace",
    )?;
    crate::issuer::Issuer::new(
        ca,
        read(ca_path)?,
        issue.issue_store_name.clone(),
        std::time::Duration::from_secs(issue.issue_hours.max(1) * 3600),
        args.kubernetes.access(),
        &namespace,
        &issue.issue_service_account,
    )
    .map_err(|e| e.to_string())
}

/// The coordinator, keeping desired state where `args`' spec store says (a `kubernetes://` store
/// reached by its Kubernetes access), or in memory when it names none, which it takes under lease
/// term `term`.
async fn coordinator(
    policy: Policy,
    telemetry: &everserve_telemetry::Telemetry,
    args: &CoordArgs,
    term: u64,
) -> Result<Arc<Coordinator>, SpecError> {
    let store: Arc<dyn SpecStore> = match &args.spec_store {
        None => return Ok(Coordinator::new(policy, telemetry).0),
        Some(StoreAt::File(path)) => Arc::new(FileStore::new(path)),
        Some(StoreAt::Kubernetes { namespace, name }) => Arc::new(KubernetesStore::new(
            &args.kubernetes.access(),
            namespace,
            name,
            crate::http::REQUEST_TIMEOUT,
        )?),
    };
    Ok(Coordinator::with_store(policy, telemetry, store, term)
        .await?
        .0)
}

/// Read `tls`'s files again every [`TLS_RELOAD_EVERY`] until `cancel` fires, so a certificate
/// renewed on disk is the one the next handshake uses (decision 0037).
fn follow_tls(tls: Option<&Tls>, cancel: &CancellationToken) {
    let Some(tls) = tls.cloned() else { return };
    let cancel = cancel.clone();
    tokio::spawn(async move {
        tokio::select! {
            () = cancel.cancelled() => {}
            () = tls.follow(TLS_RELOAD_EVERY) => {}
        }
    });
}

async fn shutdown_on_signal(cancel: CancellationToken, what: &'static str) {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(t) => t,
                Err(e) => {
                    error!("cannot listen for SIGTERM: {e}");
                    let _ = ctrl_c.await;
                    cancel.cancel();
                    return;
                }
            };
        tokio::select! {
            _ = ctrl_c => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
    }
    info!("shutdown signal received; stopping the {what}");
    cancel.cancel();
}
