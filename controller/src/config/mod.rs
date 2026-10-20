// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Configuration of the controller binary: every setting has a flag and an environment variable,
//! and every variable name is listed here so operators can grep for it.

use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};

use crate::lease::LeaseSpec;

/// Environment variable names, in one place.
pub mod environment_names {
    /// The policy document (protobuf JSON); the coordinator reads it.
    pub const POLICY: &str = "EVERSERVE_POLICY";
    /// The node daemon's Unix socket.
    pub const SOCKET: &str = "EVERSERVE_SOCKET";
    /// Seconds a process waits for in-flight changes after a shutdown signal.
    pub const GRACEFUL_SHUTDOWN_SECS: &str = "EVERSERVE_GRACEFUL_SHUTDOWN_SECS";
    /// This node's name, as the policy document names it.
    pub const NODE: &str = "EVERSERVE_NODE";
    /// Directory of daemon records for the file discovery backend.
    pub const DISCOVERY_DIR: &str = "EVERSERVE_DISCOVERY_DIR";
    /// The mirror service's socket on this node, published in the daemon's record.
    pub const MIRROR_SOCKET: &str = "EVERSERVE_MIRROR_SOCKET";
    /// The address other stores replicate into this node's store through, advertised to the
    /// coordinator (`host:port` under TLS, or a Unix path on one node).
    pub const MIRROR_REPLICA: &str = "EVERSERVE_MIRROR_REPLICA";
    /// `tracing` filter, as `RUST_LOG`.
    pub const LOG: &str = "EVERSERVE_LOG";
    /// Address for `/metrics` and `/healthz`; unset means no endpoint.
    pub const METRICS_ADDR: &str = "EVERSERVE_METRICS_ADDR";
    /// The coordinator's address: `unix:/path` on a node, `host:port` across nodes.
    pub const COORDINATOR: &str = "EVERSERVE_COORDINATOR";
    /// The name on the coordinator's certificate, verified by daemons over TLS.
    pub const COORDINATOR_NAME: &str = "EVERSERVE_COORDINATOR_NAME";
    /// How log lines are written: `text` or `json`.
    pub const LOG_FORMAT: &str = "EVERSERVE_LOG_FORMAT";
    /// The coordinator's operator API, for the commands that call it.
    pub const API: &str = "EVERSERVE_API";
    /// The CA the operator API's certificate chains to.
    pub const API_CA: &str = "EVERSERVE_API_CA";
    /// An operator's or a viewer's certificate for the operator API.
    pub const API_CERT: &str = "EVERSERVE_API_CERT";
    /// Its key.
    pub const API_KEY: &str = "EVERSERVE_API_KEY";
    /// A bearer token's file for the operator API, on Kubernetes.
    pub const API_TOKEN_FILE: &str = "EVERSERVE_API_TOKEN_FILE";
    /// The address the coordinator listens on.
    pub const LISTEN: &str = "EVERSERVE_LISTEN";
    /// The coordinator's lease: a lock file's path, `etcd://host:port[,host:port...]/key`, or
    /// `kubernetes://namespace/name`.
    pub const LEASE: &str = "EVERSERVE_LEASE";
    /// Seconds an etcd lease outlives its holder's last renewal.
    pub const LEASE_TTL_SECS: &str = "EVERSERVE_LEASE_TTL_SECS";
    /// The CA that signed the etcd cluster's server certificates.
    pub const ETCD_CA: &str = "EVERSERVE_ETCD_CA";
    /// This process's client certificate for etcd.
    pub const ETCD_CERT: &str = "EVERSERVE_ETCD_CERT";
    /// This process's client key for etcd.
    pub const ETCD_KEY: &str = "EVERSERVE_ETCD_KEY";
    /// The address daemons link to the coordinator at, published in the lease's record.
    pub const ADVERTISE: &str = "EVERSERVE_ADVERTISE";
    /// The coordinator lease a daemon follows: its record names the coordinator to link to and
    /// the term that fences the others (decision 0036).
    pub const LEASE_WATCH: &str = "EVERSERVE_LEASE_WATCH";
    /// The coordinator's spec store file: what the operator asked of each node.
    pub const SPEC_STORE: &str = "EVERSERVE_SPEC_STORE";
    /// This process's TLS certificate chain (PEM).
    pub const TLS_CERT: &str = "EVERSERVE_TLS_CERT";
    /// This process's TLS private key (PEM).
    pub const TLS_KEY: &str = "EVERSERVE_TLS_KEY";
    /// The deployment's CA certificates (PEM), which every peer's certificate must chain to.
    pub const TLS_CA: &str = "EVERSERVE_TLS_CA";
    /// The daemon's notice source: `none`, `aws`, `gcp`, `azure` or `kubernetes`.
    pub const NOTICE_SOURCE: &str = "EVERSERVE_NOTICE_SOURCE";
    /// The source's endpoint, replacing the cloud's default (a fake in a test).
    pub const NOTICE_ENDPOINT: &str = "EVERSERVE_NOTICE_ENDPOINT";
    /// Milliseconds between polls of a source that is polled.
    pub const NOTICE_POLL_MS: &str = "EVERSERVE_NOTICE_POLL_MS";
    /// The lead a source without a deadline of its own gets.
    pub const NOTICE_LEAD_MS: &str = "EVERSERVE_NOTICE_LEAD_MS";
    /// The Kubernetes API server; unset, the in-cluster service.
    pub const KUBERNETES_ENDPOINT: &str = "EVERSERVE_KUBERNETES_ENDPOINT";
    /// `--kubernetes-nodes`.
    pub const KUBERNETES_NODES: &str = "EVERSERVE_KUBERNETES_NODES";
    /// `--issue-node-certs`.
    pub const ISSUE_NODE_CERTS: &str = "EVERSERVE_ISSUE_NODE_CERTS";
    /// `--issue-ca-key`.
    pub const ISSUE_CA_KEY: &str = "EVERSERVE_ISSUE_CA_KEY";
    /// `--issue-ca-name`.
    pub const ISSUE_CA_NAME: &str = "EVERSERVE_ISSUE_CA_NAME";
    /// `--issue-service-account`.
    pub const ISSUE_SERVICE_ACCOUNT: &str = "EVERSERVE_ISSUE_SERVICE_ACCOUNT";
    /// `--issue-hours`.
    pub const ISSUE_HOURS: &str = "EVERSERVE_ISSUE_HOURS";
    /// `--issue-store-name`.
    pub const ISSUE_STORE_NAME: &str = "EVERSERVE_ISSUE_STORE_NAME";
    /// `--spot-label`.
    pub const SPOT_LABEL: &str = "EVERSERVE_SPOT_LABEL";
    /// `--preemption-label`.
    pub const PREEMPTION_LABEL: &str = "EVERSERVE_PREEMPTION_LABEL";
    /// `--preemption-lead-secs`.
    pub const PREEMPTION_LEAD_SECS: &str = "EVERSERVE_PREEMPTION_LEAD_SECS";
    /// Where the operator API listens, under TLS (decision 0039).
    pub const API_ADDR: &str = "EVERSERVE_API_ADDR";
    /// The operators' certificate names, comma-separated: they may do anything the API serves.
    pub const API_OPERATORS: &str = "EVERSERVE_API_OPERATORS";
    /// The viewers' certificate names, comma-separated: they may read.
    pub const API_VIEWERS: &str = "EVERSERVE_API_VIEWERS";
    /// Take bearer tokens, checked by the Kubernetes API server.
    pub const API_KUBERNETES: &str = "EVERSERVE_API_KUBERNETES";
    /// The namespace a caller's rights are reviewed in, the coordinator's own.
    pub const NAMESPACE: &str = "EVERSERVE_NAMESPACE";
    /// This pod's name, on Kubernetes: the lease holder labels it for the chart's Service.
    pub const POD_NAME: &str = "EVERSERVE_POD_NAME";
    /// The bearer token file for the Kubernetes API (the service account's), read again at every
    /// request.
    pub const KUBERNETES_TOKEN_FILE: &str = "EVERSERVE_KUBERNETES_TOKEN_FILE";
    /// The CA file the Kubernetes API server's certificate chains to.
    pub const KUBERNETES_CA_FILE: &str = "EVERSERVE_KUBERNETES_CA_FILE";
    /// Milliseconds between the daemon's heartbeats to the coordinator; 0 for none.
    pub const HEARTBEAT_MS: &str = "EVERSERVE_HEARTBEAT_MS";
    /// The address other nodes reach this node at, published in the daemon's record.
    pub const ADVERTISE_HOST: &str = "EVERSERVE_ADVERTISE_HOST";
    /// Missed heartbeats after which the coordinator takes a daemon's node as lost.
    pub const NODE_LOSS_MISSES: &str = "EVERSERVE_NODE_LOSS_MISSES";
}

/// The controller command line.
#[derive(Debug, Parser)]
#[command(
    name = "everserve-controller",
    version,
    about = "Everserve control plane"
)]
pub struct Cli {
    /// How log lines are written to standard error: `text` for people, `json` for a log
    /// pipeline.
    #[arg(long, global = true, env = environment_names::LOG_FORMAT, default_value = "text")]
    pub log_format: everserve_telemetry::LogFormat,
    /// What to run.
    #[command(subcommand)]
    pub command: Command,
}

/// The controller's roles.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the per-node daemon: workers register over a Unix socket; the daemon links to the
    /// coordinator and delivers its changes.
    Node(Box<NodeArgs>),
    /// Run the coordinator: the one writer of every engine's epoch, holding a lease.
    Coord(Box<CoordArgs>),
    /// Write a deployment CA and a certificate per node and for the coordinator, for a
    /// deployment without a CA of its own. Run again with more nodes to add them under the same
    /// CA.
    Certs(CertsArgs),
    /// Is the control plane ready for the engines the policy names? Waits for every policy
    /// node's daemon to connect, its devices to be reported, and every spare node's store to
    /// take replicas. Exit 0 when ready, 1 when not by the deadline.
    Check(CheckArgs),
    /// Be issued this node's certificate by the coordinator, from the pod's service account
    /// token (decision 0037), and renew it at half its life: the key, the certificate and the CA
    /// written to one directory, where the daemon and the store read them as they change.
    NodeCert(NodeCertArgs),
    /// Make the node's shared directories (the daemon's sockets, the shared memory) its node
    /// group's, mode 2770, so the daemon, the store and the engines' processes, all members, share
    /// them and no one else does. Run as the directories' owner, a member of the group.
    NodeDirs(NodeDirsArgs),
    /// Print the operator API's published description (decision 0042): the JSON document
    /// `docs/api/openapi.json` holds.
    Openapi,
    /// Read engines, nodes or pools from the operator API.
    Get(GetArgs),
    /// Follow the changes of engines, nodes or pools, one line each, until stopped.
    Watch(WatchArgs),
    /// Cordon a node: no spare takes a position on it; what it holds stays.
    Cordon(NodeWriteArgs),
    /// Uncordon a node.
    Uncordon(NodeWriteArgs),
    /// Drain a node: every rank on it moves to a spare by a deadline, then it is released. A
    /// drained node is cordoned.
    Drain(DrainArgs),
    /// Withdraw a node's drain that waits its turn or was blocked; one underway cannot be.
    Undrain(NodeWriteArgs),
}

/// Settings of the node's shared directories.
#[derive(Debug, Args, Clone)]
pub struct NodeDirsArgs {
    /// The node group's id.
    #[arg(long)]
    pub group: u32,
    /// A directory to share; made if it is missing.
    #[arg(required = true)]
    pub dirs: Vec<PathBuf>,
}

/// Settings of a node's certificate enrollment.
#[derive(Debug, Args, Clone)]
pub struct NodeCertArgs {
    /// The coordinator's operator API, `host:port` or `https://host:port`, which issues node
    /// certificates (`--issue-node-certs`).
    #[arg(long)]
    pub coordinator_api: String,
    /// The name on the coordinator's certificate (its `--coordinator-name`).
    #[arg(long, env = environment_names::COORDINATOR_NAME, default_value = "coordinator")]
    pub coordinator_name: String,
    /// The CA the coordinator's certificate chains to.
    #[arg(long)]
    pub ca: PathBuf,
    /// The pod's service account token, projected for the audience `everserve`; read at every
    /// request, as the kubelet rotates it.
    #[arg(long)]
    pub token_file: PathBuf,
    /// The directory `node.key`, `node.crt` and `ca.crt` are written to.
    #[arg(long)]
    pub out: PathBuf,
    /// Be issued one certificate and exit: before the daemon and the store start. Without it,
    /// keep renewing until stopped.
    #[arg(long)]
    pub once: bool,
}

/// How a command reaches the coordinator's operator API (decision 0039): its address, the name on
/// its certificate and the CA that certificate chains to, and the caller's credential, an
/// operator's or a viewer's certificate or a bearer token. Each may be set once in the
/// environment.
#[derive(Debug, Args, Clone)]
pub struct ApiAccessArgs {
    /// The coordinator's operator API, `host:port` or `https://host:port`.
    #[arg(long, env = environment_names::API)]
    pub coordinator_api: String,
    /// The name on the coordinator's certificate (its `--coordinator-name`).
    #[arg(long, env = environment_names::COORDINATOR_NAME, default_value = "coordinator")]
    pub coordinator_name: String,
    /// The CA the coordinator's certificate chains to.
    #[arg(long, env = environment_names::API_CA)]
    pub ca: PathBuf,
    /// An operator's or a viewer's certificate (`certs --operator`), with `--key`.
    #[arg(long, env = environment_names::API_CERT, requires = "key", conflicts_with = "token_file")]
    pub cert: Option<PathBuf>,
    /// Its key.
    #[arg(long, env = environment_names::API_KEY, requires = "cert")]
    pub key: Option<PathBuf>,
    /// A bearer token's file, on Kubernetes, in place of a certificate.
    #[arg(long, env = environment_names::API_TOKEN_FILE)]
    pub token_file: Option<PathBuf>,
}

/// Settings of the readiness check.
#[derive(Debug, Args, Clone)]
pub struct CheckArgs {
    /// The policy document (protobuf JSON).
    #[arg(long, env = environment_names::POLICY)]
    pub policy: PathBuf,
    /// The operator API.
    #[command(flatten)]
    pub api: ApiAccessArgs,
    /// Seconds to wait for the control plane to become ready.
    #[arg(long, default_value_t = 60)]
    pub timeout_secs: u64,
}

/// A kind of the operator API's objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ObjectKind {
    /// Engines.
    Engines,
    /// Nodes.
    Nodes,
    /// Pools.
    Pools,
}

impl ObjectKind {
    /// The kind's path under `/v1`.
    #[must_use]
    pub fn path(self) -> &'static str {
        match self {
            Self::Engines => "engines",
            Self::Nodes => "nodes",
            Self::Pools => "pools",
        }
    }
}

/// How a read is printed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Output {
    /// A table, one line per object.
    #[default]
    Table,
    /// The API's JSON, as it answered.
    Json,
}

/// Settings of `get`.
#[derive(Debug, Args, Clone)]
pub struct GetArgs {
    /// The objects' kind.
    pub kind: ObjectKind,
    /// One object's name; every object of the kind without it.
    pub name: Option<String>,
    /// How it is printed.
    #[arg(long, short, value_enum, default_value_t = Output::Table)]
    pub output: Output,
    /// The operator API.
    #[command(flatten)]
    pub api: ApiAccessArgs,
}

/// Settings of `watch`.
#[derive(Debug, Args, Clone)]
pub struct WatchArgs {
    /// The objects' kind.
    pub kind: ObjectKind,
    /// How each change is printed.
    #[arg(long, short, value_enum, default_value_t = Output::Table)]
    pub output: Output,
    /// The operator API.
    #[command(flatten)]
    pub api: ApiAccessArgs,
}

/// Settings of a write to a node's spec: `cordon`, `uncordon`, `undrain`.
#[derive(Debug, Args, Clone)]
pub struct NodeWriteArgs {
    /// The node.
    pub node: String,
    /// Why, for people; kept with the node's spec.
    #[arg(long, default_value = "")]
    pub reason: String,
    /// The operator API.
    #[command(flatten)]
    pub api: ApiAccessArgs,
}

/// Settings of `drain`.
#[derive(Debug, Args, Clone)]
pub struct DrainArgs {
    /// The node.
    pub node: String,
    /// Seconds from now the node goes: every rank on it moves to a spare by then.
    #[arg(long = "in", value_name = "SECONDS")]
    pub lead_secs: u64,
    /// Why, for people; kept with the node's spec.
    #[arg(long, default_value = "")]
    pub reason: String,
    /// The operator API.
    #[command(flatten)]
    pub api: ApiAccessArgs,
}

/// Settings of the certificate writer.
#[derive(Debug, Args, Clone)]
pub struct CertsArgs {
    /// Directory the files go in: `ca.crt`, `ca.key`, `coordinator.crt`/`.key`, and
    /// `<node>.crt`/`.key` per node. An existing CA there is reused.
    #[arg(long)]
    pub out: PathBuf,
    /// A node's name, as the policy names it; repeat per node.
    #[arg(long = "node")]
    pub nodes: Vec<String>,
    /// An operator's or a viewer's name for the operator API (decision 0039); repeat per name.
    /// Written as `operator-<name>.crt`/`.key`, valid for `<name>.operators.everserve`.
    #[arg(long = "operator")]
    pub operators: Vec<String>,
    /// The name daemons verify on the coordinator's certificate (`--coordinator-name`).
    #[arg(long, default_value = "coordinator")]
    pub coordinator_name: String,
    /// The name stores verify on a peer store's certificate (the store's `--peer-name`); every
    /// node's certificate carries it beside the node's name.
    #[arg(long, default_value = "store")]
    pub store_name: String,
    /// Days the certificates are valid for.
    #[arg(long, default_value_t = 3650)]
    pub days: u32,
    /// Replace a certificate or key that already exists. Without it an existing file is kept and
    /// reported, so a rerun that adds a node touches nothing a running process holds.
    #[arg(long)]
    pub overwrite: bool,
}

/// TLS material for TCP between nodes; all three or none.
#[derive(Debug, Args, Clone)]
pub struct TlsArgs {
    /// This process's certificate chain (PEM).
    #[arg(long, env = environment_names::TLS_CERT)]
    pub tls_cert: Option<PathBuf>,
    /// This process's private key (PEM).
    #[arg(long, env = environment_names::TLS_KEY)]
    pub tls_key: Option<PathBuf>,
    /// The CA certificates every peer must chain to (PEM).
    #[arg(long, env = environment_names::TLS_CA)]
    pub tls_ca: Option<PathBuf>,
}

impl TlsArgs {
    /// Load the material if all three paths are given.
    ///
    /// # Errors
    ///
    /// A partial set of paths, or material that does not load.
    pub fn load(
        &self,
    ) -> Result<Option<everserve_wire::transport::Tls>, everserve_wire::transport::TransportError>
    {
        match (&self.tls_cert, &self.tls_key, &self.tls_ca) {
            (None, None, None) => Ok(None),
            (Some(c), Some(k), Some(a)) => everserve_wire::transport::Tls::load(c, k, a).map(Some),
            _ => Err(everserve_wire::transport::TransportError::Tls {
                path: PathBuf::from("tls"),
                reason: "give all of --tls-cert, --tls-key and --tls-ca, or none".into(),
            }),
        }
    }
}

/// TLS for an etcd cluster that keeps the lease: the CA alone, or the CA and a client
/// certificate with its key.
#[derive(Debug, Args, Clone, Default)]
pub struct EtcdTlsArgs {
    /// The CA that signed the etcd cluster's server certificates (PEM).
    #[arg(long, env = environment_names::ETCD_CA)]
    pub etcd_ca: Option<PathBuf>,
    /// This process's client certificate for etcd (PEM).
    #[arg(long, env = environment_names::ETCD_CERT, requires = "etcd_key")]
    pub etcd_cert: Option<PathBuf>,
    /// This process's client key for etcd (PEM).
    #[arg(long, env = environment_names::ETCD_KEY, requires = "etcd_cert")]
    pub etcd_key: Option<PathBuf>,
}

/// How a process reaches its lease's store: the etcd TLS and the Kubernetes API it was given.
///
/// # Errors
///
/// A client certificate for etcd without a CA.
pub fn reach(
    etcd: &EtcdTlsArgs,
    kubernetes: &KubernetesArgs,
) -> Result<crate::lease::Reach, crate::lease::LeaseError> {
    Ok(crate::lease::Reach {
        etcd_tls: etcd.tls()?,
        kubernetes: Some(kubernetes.access()),
    })
}

impl EtcdTlsArgs {
    /// The TLS settings, if a CA is given.
    ///
    /// # Errors
    ///
    /// A client certificate without a CA.
    pub fn tls(&self) -> Result<Option<crate::lease::EtcdTls>, crate::lease::LeaseError> {
        let identity = self.etcd_cert.clone().zip(self.etcd_key.clone());
        match (&self.etcd_ca, identity) {
            (None, None) => Ok(None),
            (None, Some(_)) => Err(crate::lease::LeaseError::Store {
                at: "etcd".into(),
                reason: "a client certificate needs --etcd-ca".into(),
            }),
            (Some(ca), identity) => Ok(Some(crate::lease::EtcdTls {
                ca: ca.clone(),
                identity,
            })),
        }
    }
}

/// The operator API's settings (decision 0039).
#[derive(Debug, Args, Clone, Default)]
pub struct ApiArgs {
    /// Where the operator API listens, e.g. `0.0.0.0:7443`. Unset: no operator API.
    #[arg(long, env = environment_names::API_ADDR)]
    pub api_addr: Option<std::net::SocketAddr>,
    /// Operators by the names their client certificates are valid for (`certs --operator`).
    #[arg(long, env = environment_names::API_OPERATORS, value_delimiter = ',')]
    pub api_operators: Vec<String>,
    /// Viewers by the names their client certificates are valid for: they may read.
    #[arg(long, env = environment_names::API_VIEWERS, value_delimiter = ',')]
    pub api_viewers: Vec<String>,
    /// Take bearer tokens: `TokenReview` says who, `SubjectAccessReview` whether they may, as
    /// `everserve.ai` verbs in `--namespace`.
    #[arg(long, env = environment_names::API_KUBERNETES)]
    pub api_kubernetes: bool,
    /// The namespace rights are reviewed in: the coordinator's own.
    #[arg(long, env = environment_names::NAMESPACE)]
    pub namespace: Option<String>,
}

/// How a process reaches the Kubernetes API: one set of settings for every caller (the node's
/// notice source, a `kubernetes://` lease and its watch).
#[derive(Debug, Args, Clone)]
pub struct KubernetesArgs {
    /// The API server, `https://host:port`. Unset: the in-cluster service the kubelet names in
    /// `KUBERNETES_SERVICE_HOST` and `KUBERNETES_SERVICE_PORT`.
    #[arg(long, env = environment_names::KUBERNETES_ENDPOINT)]
    pub kubernetes_endpoint: Option<String>,
    /// The bearer token file (the service account's), read again at every request.
    #[arg(
        long,
        env = environment_names::KUBERNETES_TOKEN_FILE,
        default_value = crate::kubernetes::SERVICE_ACCOUNT_TOKEN
    )]
    pub kubernetes_token_file: PathBuf,
    /// The CA the API server's certificate chains to.
    #[arg(
        long,
        env = environment_names::KUBERNETES_CA_FILE,
        default_value = crate::kubernetes::SERVICE_ACCOUNT_CA
    )]
    pub kubernetes_ca_file: PathBuf,
}

impl KubernetesArgs {
    /// The access these settings give. Without an endpoint, the in-cluster service from the
    /// kubelet's environment, or its well-known name; the library itself reads no environment.
    #[must_use]
    pub fn access(&self) -> crate::kubernetes::Access {
        let endpoint = self.kubernetes_endpoint.clone().unwrap_or_else(|| {
            match std::env::var("KUBERNETES_SERVICE_HOST") {
                Ok(host) => {
                    let port =
                        std::env::var("KUBERNETES_SERVICE_PORT").unwrap_or_else(|_| "443".into());
                    format!("https://{host}:{port}")
                }
                Err(_) => crate::kubernetes::IN_CLUSTER_API.to_owned(),
            }
        });
        crate::kubernetes::Access {
            endpoint,
            token_file: self.kubernetes_token_file.clone(),
            ca_file: self.kubernetes_ca_file.clone(),
        }
    }
}

/// Settings of the node daemon.
#[derive(Debug, Args, Clone)]
pub struct NodeArgs {
    /// The Unix socket workers connect to.
    #[arg(long, env = environment_names::SOCKET)]
    pub socket: PathBuf,
    /// The coordinator's address (`unix:/path` or `host:port`). Without it the daemon serves its
    /// links and holds every report until a coordinator is given on a restart.
    #[arg(long, env = environment_names::COORDINATOR)]
    pub coordinator: Option<String>,
    /// The coordinator lease to follow (decision 0036): its record names the coordinator to link
    /// to, in place of `--coordinator`, and its term refuses any other. A lease file's path, or
    /// `etcd://host:port[,host:port...]/key`.
    #[arg(long, env = environment_names::LEASE_WATCH, conflicts_with = "coordinator")]
    pub lease_watch: Option<LeaseSpec>,
    /// TLS for an etcd lease.
    #[command(flatten)]
    pub etcd: EtcdTlsArgs,
    /// The name on the coordinator's certificate, for TCP.
    #[arg(long, env = environment_names::COORDINATOR_NAME, default_value = "coordinator")]
    pub coordinator_name: String,
    /// Seconds to wait for in-flight deliveries after SIGINT or SIGTERM before exiting anyway.
    #[arg(long, env = environment_names::GRACEFUL_SHUTDOWN_SECS, default_value_t = 30)]
    pub graceful_shutdown_secs: u64,
    /// This node's name, as the policy document names it. Defaults to the hostname.
    #[arg(long, env = environment_names::NODE)]
    pub node: Option<String>,
    /// Directory the daemon publishes its record in (file discovery). None: do not publish.
    #[arg(long, env = environment_names::DISCOVERY_DIR)]
    pub discovery_dir: Option<PathBuf>,
    /// The mirror service's socket on this node, published in the record.
    #[arg(long, env = environment_names::MIRROR_SOCKET)]
    pub mirror_socket: Option<PathBuf>,
    /// The address other stores replicate into this node's store through (the store's
    /// `--replica-listen`), advertised to the coordinator. Unset: this node takes no replicas.
    #[arg(long, env = environment_names::MIRROR_REPLICA)]
    pub mirror_replica: Option<String>,
    /// Address for `/metrics` and `/healthz`, e.g. `127.0.0.1:9108`. Unset: no endpoint.
    #[arg(long, env = environment_names::METRICS_ADDR)]
    pub metrics_addr: Option<std::net::SocketAddr>,
    /// Where this node learns it is leaving: `none` (the operator's drain call only), `aws`,
    /// `gcp`, `azure` (the cloud's metadata endpoint) or `kubernetes` (this node's taints).
    #[arg(long, env = environment_names::NOTICE_SOURCE, default_value = "none")]
    pub notice_source: crate::notice::Source,
    /// The source's endpoint instead of the cloud's default (`http://host:port`), for a fake.
    #[arg(long, env = environment_names::NOTICE_ENDPOINT)]
    pub notice_endpoint: Option<String>,
    /// Milliseconds between polls of a source that is polled. Azure recommends once a second and
    /// AWS's own handler polls every two; the metadata endpoints are link-local.
    #[arg(long, env = environment_names::NOTICE_POLL_MS, default_value_t = 1000)]
    pub notice_poll_ms: u64,
    /// The lead a source without a deadline of its own gets: GCP's preemption flag (set it to the
    /// VM's preemption notice, 120000 for a VM created with the 120 s notice; GCP's default
    /// notice is 0 s) and a Kubernetes cordon.
    #[arg(long, env = environment_names::NOTICE_LEAD_MS, default_value_t = 30_000)]
    pub notice_lead_ms: u64,
    /// The Kubernetes API, for the `kubernetes` notice source and a `kubernetes://` lease.
    #[command(flatten)]
    pub kubernetes: KubernetesArgs,
    /// The address other nodes reach this node at (the fabric's, or the tunnel's), published in
    /// the daemon's record for the workers' rendezvous across nodes. Unset on a single node.
    #[arg(long, env = environment_names::ADVERTISE_HOST)]
    pub advertise_host: Option<String>,
    /// Milliseconds between heartbeats to the coordinator, which takes this node as lost after
    /// its count of misses (decision 0023); 0 sends none, and the node is lost only when the link
    /// closes.
    #[arg(long, env = environment_names::HEARTBEAT_MS, default_value_t = 500)]
    pub heartbeat_ms: u64,
    /// TLS material for a TCP coordinator address.
    #[command(flatten)]
    pub tls: TlsArgs,
}

impl NodeArgs {
    /// The notice source's settings.
    #[must_use]
    pub fn notice(&self) -> crate::notice::Settings {
        crate::notice::Settings {
            source: self.notice_source,
            endpoint: self.notice_endpoint.clone(),
            poll: Duration::from_millis(self.notice_poll_ms),
            lead: Duration::from_millis(self.notice_lead_ms),
            node: self.node_name(),
            kubernetes: Some(self.kubernetes.access()),
        }
    }

    /// The graceful period as a duration.
    #[must_use]
    pub fn graceful_shutdown(&self) -> Duration {
        Duration::from_secs(self.graceful_shutdown_secs)
    }

    /// The node name: the flag, or the hostname.
    #[must_use]
    pub fn node_name(&self) -> String {
        self.node.clone().unwrap_or_else(hostname)
    }
}

/// Settings of the coordinator.
#[derive(Debug, Args, Clone)]
pub struct CoordArgs {
    /// The policy document (protobuf JSON).
    #[arg(long, env = environment_names::POLICY)]
    pub policy: PathBuf,
    /// Where daemons connect: `unix:/path` on one node, `host:port` across nodes (TLS required).
    #[arg(long, env = environment_names::LISTEN)]
    pub listen: String,
    /// The coordinator's lease; it acts only while it holds it (decision 0036). A lock file's path
    /// for one host, `etcd://host:port[,host:port...]/key` for a coordinator with a standby, or
    /// `kubernetes://namespace/name` for a Lease the cluster's API server keeps.
    #[arg(long, env = environment_names::LEASE)]
    pub lease: LeaseSpec,
    /// Seconds an etcd lease outlives its holder's last renewal: the longest a standby waits after
    /// a holder dies. The holder renews every quarter of it and stops acting three quarters of it
    /// after its last answered renewal. At least 2.
    #[arg(long, env = environment_names::LEASE_TTL_SECS, default_value_t = 10)]
    pub lease_ttl_secs: u64,
    /// TLS for an etcd lease.
    #[command(flatten)]
    pub etcd: EtcdTlsArgs,
    /// The Kubernetes API, for a `kubernetes://` lease.
    #[command(flatten)]
    pub kubernetes: KubernetesArgs,
    /// This pod's name, under a `kubernetes://` lease: the holder labels its own pod
    /// `everserve.ai/lease-holder=true` for the chart's Service to select, clears the label at
    /// its start and when its hold ends, and needs `patch` on pods in the lease's namespace.
    /// Unset: no pod is labelled.
    #[arg(long, env = environment_names::POD_NAME)]
    pub pod_name: Option<String>,
    /// The address daemons link to this coordinator at, published in the lease's record for the
    /// daemons that follow it (decision 0036). Unset: the listen address, when it is one a daemon
    /// can reach.
    #[arg(long, env = environment_names::ADVERTISE)]
    pub advertise: Option<String>,
    /// The spec store: what the operator asked of each node and pool, and each engine's change
    /// in flight, kept across a restart and a takeover (decisions 0036, 0038). A file, or
    /// `kubernetes://namespace/name` for the cluster's `EverserveState` objects. Unset: kept in
    /// memory, lost with the process.
    #[arg(long, env = environment_names::SPEC_STORE)]
    pub spec_store: Option<crate::spec::StoreAt>,
    /// Seconds to wait for in-flight changes after SIGINT or SIGTERM before exiting anyway.
    #[arg(long, env = environment_names::GRACEFUL_SHUTDOWN_SECS, default_value_t = 30)]
    pub graceful_shutdown_secs: u64,
    /// Address for `/metrics` and `/healthz`, e.g. `127.0.0.1:9107`. Unset: no endpoint.
    #[arg(long, env = environment_names::METRICS_ADDR)]
    pub metrics_addr: Option<std::net::SocketAddr>,
    /// The operator API (decision 0039): where it listens, under TLS with this coordinator's
    /// certificate, every request authenticated and authorized. Unset: no operator API.
    #[command(flatten)]
    pub api: ApiArgs,
    /// Missed heartbeats after which a daemon's node is lost and its ranks changed (decision
    /// 0023). With the daemons' 500 ms heartbeat, 3 is 1.5 s. A false loss costs a whole node's
    /// handover: raise it on a fabric that drops packets.
    #[arg(long, env = environment_names::NODE_LOSS_MISSES, default_value_t = 3)]
    pub node_loss_misses: u64,
    /// The cluster's Nodes (decision 0041).
    #[command(flatten)]
    pub nodes: NodesArgs,
    /// Node certificates on Kubernetes (decision 0037).
    #[command(flatten)]
    pub issue: IssueArgs,
    /// TLS material for a TCP listen address.
    #[command(flatten)]
    pub tls: TlsArgs,
}

/// How the coordinator issues node certificates on Kubernetes (decision 0037), on its operator
/// API's listener: to a node daemon's pod-bound token, by `TokenReview` through the Kubernetes
/// access its other arguments give.
#[derive(Debug, Clone, Args)]
pub struct IssueArgs {
    /// Issue node certificates at `POST /v1/certificates/node`.
    #[arg(long, env = environment_names::ISSUE_NODE_CERTS, requires = "issue_ca_key")]
    pub issue_node_certs: bool,
    /// The deployment CA's key (PEM), which signs them; its certificate is `--tls-ca`.
    #[arg(long, env = environment_names::ISSUE_CA_KEY)]
    pub issue_ca_key: Option<PathBuf>,
    /// The deployment CA's subject, its common name, which every certificate it signs names as
    /// its issuer: `certs`' unless the CA is the operator's own.
    #[arg(long, env = environment_names::ISSUE_CA_NAME, default_value = crate::certs::CA_NAME)]
    pub issue_ca_name: String,
    /// The node daemons' service account, in the coordinator's `--namespace`: only its tokens are
    /// issued a certificate.
    #[arg(long, env = environment_names::ISSUE_SERVICE_ACCOUNT, default_value = "everserve-node")]
    pub issue_service_account: String,
    /// Hours a node certificate is valid; the node renews it at half.
    #[arg(long, env = environment_names::ISSUE_HOURS, default_value_t = 12)]
    pub issue_hours: u64,
    /// The name stores verify on a peer store, which every node certificate also carries.
    #[arg(long, env = environment_names::ISSUE_STORE_NAME, default_value = "store")]
    pub issue_store_name: String,
}

/// How the coordinator reads the cluster's Nodes (decision 0041), through the Kubernetes access
/// its other arguments give. A node's daemon is named as its Node is.
#[derive(Debug, Clone, Args)]
pub struct NodesArgs {
    /// Follow the cluster's Nodes: which nodes are spot, and which their provider is taking back.
    #[arg(long, env = environment_names::KUBERNETES_NODES)]
    pub kubernetes_nodes: bool,
    /// `key=value`: a Node with this label is spot capacity, which an engine whose policy wants
    /// its head on on-demand capacity must not have its head on (readiness's `HeadOnSpot`).
    #[arg(long, env = environment_names::SPOT_LABEL, requires = "kubernetes_nodes")]
    pub spot_label: Option<crate::coord::Label>,
    /// `key=value`: a Node with this label is being taken back by its provider, and is drained
    /// by `--preemption-lead-secs` from when the label appears. On `CoreWeave`'s spot pools,
    /// `qos.coreweave.cloud/pending-preemption=true`.
    #[arg(long, env = environment_names::PREEMPTION_LABEL, requires = "kubernetes_nodes")]
    pub preemption_label: Option<crate::coord::Label>,
    /// Seconds from the preemption label to the provider's own drain: `CoreWeave` drains two
    /// minutes after it labels the node.
    #[arg(long, env = environment_names::PREEMPTION_LEAD_SECS, default_value_t = 120)]
    pub preemption_lead_secs: u64,
}

impl NodesArgs {
    /// What the coordinator reads from the Nodes' labels.
    #[must_use]
    pub fn labels(&self) -> crate::coord::NodeLabels {
        crate::coord::NodeLabels {
            spot: self.spot_label.clone(),
            preemption: self.preemption_label.clone(),
            preemption_lead: std::time::Duration::from_secs(self.preemption_lead_secs),
        }
    }
}

impl CoordArgs {
    /// The graceful period as a duration.
    #[must_use]
    pub fn graceful_shutdown(&self) -> Duration {
        Duration::from_secs(self.graceful_shutdown_secs)
    }
}

pub(crate) fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_owned())
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "localhost".to_owned())
}
