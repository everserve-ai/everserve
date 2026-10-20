// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The two transports every control-plane connection uses: a Unix socket on a node, and TCP with
//! mutual TLS across nodes. Both carry the frame protocol (`everserve_wire::frame`) unchanged,
//! so the daemon, the coordinator and the workers speak the same frames whichever way they are
//! connected (`docs/RUNTIME.md` section 0, rule 2).
//!
//! Mutual TLS: both ends present a certificate signed by the deployment's CA and verify the
//! other's. The certificates come from the cluster (Kubernetes secrets) or the operator's CA; this
//! module only loads PEM files. An address without TLS material is refused for TCP: there is no
//! plaintext TCP in a deployment. Shared by the controller and the store, which both dial peers.
//!
//! The link is gated by the CA alone, and that sets its limits: there is no cap on accepted
//! connections, no idle timeout on an accepted link and no keepalive, since a peer that can
//! connect is one of the deployment's own, and a half-open link costs one task until the kernel
//! notices; a handshake has a bound so an unanswered one does not. TLS 1.2 is allowed beside 1.3
//! with rustls's default suites, for CA material an operator already has.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, PoisonError, RwLock};
use std::time::Duration;

use rustls::pki_types::pem::{self, PemObject};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio_rustls::{TlsAcceptor, TlsConnector};

/// One end of a connection: a Unix or a TLS stream behind one type.
pub type Stream = Pin<Box<dyn AsyncStream>>;

/// Who is at the other end of an accepted connection (decision 0026).
///
/// A Unix socket is node-local by construction: whatever connects is one of the node's own
/// processes, and it is taken as whoever it says it is. A TLS connection carries the peer's
/// certificate, verified against the deployment's CA at the handshake; what name it is valid
/// for is checked by rustls-webpki, the same code that verifies a server's name, not parsed here.
#[derive(Debug, Clone)]
pub enum Peer {
    /// A process on this node, over a Unix socket.
    Local,
    /// A process across the fabric, with the leaf certificate it presented.
    Certified(rustls::pki_types::CertificateDer<'static>),
}

impl Peer {
    /// Whether the peer may call itself `name`: always for a local peer; for a certified one,
    /// only when its certificate is valid for `name` as a DNS name.
    #[must_use]
    pub fn is_named(&self, name: &str) -> bool {
        match self {
            Self::Local => true,
            Self::Certified(cert) => {
                let Ok(name) = rustls::pki_types::ServerName::try_from(name.to_owned()) else {
                    return false;
                };
                webpki::EndEntityCert::try_from(cert)
                    .is_ok_and(|c| c.verify_is_valid_for_subject_name(&name).is_ok())
            }
        }
    }
}

/// An accepted connection: the stream, and who is at the other end.
pub struct Accepted {
    /// The connection.
    pub stream: Stream,
    /// Who connected.
    pub peer: Peer,
}

/// What a stream is: readable, writable, movable between tasks.
pub trait AsyncStream: AsyncRead + AsyncWrite + Send {}
impl<T: AsyncRead + AsyncWrite + Send> AsyncStream for T {}

/// Where a control-plane process listens or connects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Address {
    /// A Unix socket path on this node.
    Unix(PathBuf),
    /// A TCP address across nodes; always with mutual TLS.
    Tcp(SocketAddr),
}

impl Address {
    /// Parse an operator's spelling: `unix:/path`, a bare path, or `host:port`.
    ///
    /// # Errors
    ///
    /// A string that is neither a path nor a socket address.
    pub fn parse(text: &str) -> Result<Self, TransportError> {
        if let Some(path) = text.strip_prefix("unix:") {
            return Ok(Self::Unix(PathBuf::from(path)));
        }
        if text.starts_with('/') || text.starts_with('.') {
            return Ok(Self::Unix(PathBuf::from(text)));
        }
        text.parse::<SocketAddr>()
            .map(Self::Tcp)
            .map_err(|_| TransportError::Address(text.to_owned()))
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unix(p) => write!(f, "unix:{}", p.display()),
            Self::Tcp(a) => write!(f, "{a}"),
        }
    }
}

/// Why a transport could not be set up or used.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// A connect or a TLS handshake did not finish within [`HANDSHAKE_TIMEOUT`].
    #[error("{what} did not finish within {HANDSHAKE_TIMEOUT:?}")]
    Timeout {
        /// What was waited for.
        what: &'static str,
    },
    /// Not a path and not a socket address.
    #[error("not an address: {0:?} (expected unix:/path, /path, or host:port)")]
    Address(String),
    /// A TCP address was given without TLS material.
    #[error("TCP address {0} needs TLS: certificate, key and CA")]
    TlsRequired(SocketAddr),
    /// The PEM files could not be read or parsed.
    #[error("TLS material {path}: {reason}")]
    Tls {
        /// The file.
        path: PathBuf,
        /// What was wrong with it.
        reason: String,
    },
    /// The TLS configuration was rejected by rustls.
    #[error("TLS configuration: {0}")]
    TlsConfig(#[from] rustls::Error),
    /// The socket failed.
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

/// Whether a TLS server requires its clients' certificates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientAuth {
    /// Every client presents a certificate the CA signed: the control plane's links.
    Required,
    /// A client may present one, verified if it does: a server that also takes bearer tokens.
    Optional,
}

/// The deployment's TLS material: this process's certificate chain and key, and the CA that
/// signed every peer's, as PEM files. Clones share it, and [`Tls::reload`] replaces it in place:
/// a certificate renewed on disk (a node's, decision 0037; a Secret cert-manager rotates) is the
/// one every later handshake presents and verifies by, on a listener already bound and a
/// connector already made, without a restart. A connection already open keeps the certificate it
/// was made with.
#[derive(Clone)]
pub struct Tls {
    files: Arc<Files>,
    current: Arc<RwLock<Arc<Material>>>,
}

/// Where the material is read from.
#[derive(Debug)]
struct Files {
    cert: PathBuf,
    key: PathBuf,
    ca: PathBuf,
}

/// The material as last read, with the bytes it was read from.
struct Material {
    certs: Vec<rustls::pki_types::CertificateDer<'static>>,
    key: rustls::pki_types::PrivateKeyDer<'static>,
    roots: Arc<rustls::RootCertStore>,
    read: [Vec<u8>; 3],
}

impl std::fmt::Debug for Tls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let current = self.material();
        f.debug_struct("Tls")
            .field("files", &self.files)
            .field("certs", &current.certs.len())
            .field("roots", &current.roots.len())
            .finish_non_exhaustive()
    }
}

impl Tls {
    /// Load PEM files: the certificate chain, the private key, the CA certificates.
    ///
    /// # Errors
    ///
    /// [`TransportError::Tls`] for a file that cannot be read or holds no usable item, or a key
    /// that is not the certificate's.
    pub fn load(cert: &Path, key: &Path, ca: &Path) -> Result<Self, TransportError> {
        let files = Files {
            cert: cert.to_path_buf(),
            key: key.to_path_buf(),
            ca: ca.to_path_buf(),
        };
        let material = files.read()?;
        Ok(Self {
            files: Arc::new(files),
            current: Arc::new(RwLock::new(Arc::new(material))),
        })
    }

    /// Read the files again and, where they changed, use what they now hold: true when the
    /// material was replaced. Material that does not load (a file half written, a key that is
    /// not the certificate's) replaces nothing, and the material in use stays.
    ///
    /// # Errors
    ///
    /// What [`Tls::load`] refuses.
    pub fn reload(&self) -> Result<bool, TransportError> {
        let read = self.files.bytes()?;
        if read == self.material().read {
            return Ok(false);
        }
        let material = self.files.parse(read)?;
        *self.current.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(material);
        Ok(true)
    }

    /// [`Tls::reload`] every `every`, forever, saying when the material is replaced and when its
    /// files do not load: what a process runs beside its listeners, ended by dropping the
    /// future.
    pub async fn follow(&self, every: Duration) {
        loop {
            tokio::time::sleep(every).await;
            match self.reload() {
                Ok(true) => tracing::info!("TLS material replaced by its renewed files"),
                Ok(false) => {}
                Err(e) => tracing::warn!("TLS material kept, its files do not load: {e}"),
            }
        }
    }

    fn material(&self) -> Arc<Material> {
        Arc::clone(&self.current.read().unwrap_or_else(PoisonError::into_inner))
    }

    fn acceptor(&self) -> Result<TlsAcceptor, TransportError> {
        Ok(TlsAcceptor::from(Arc::new(
            self.server_config(ClientAuth::Required)?,
        )))
    }

    /// A server configuration with the material in use: this process's certificate, and
    /// clients' certificates verified against the CA, required or, for a server that also
    /// authenticates another way (a bearer token, decision 0039), optional. A certificate
    /// presented is always verified; an optional one may be absent. Made for each connection,
    /// so a reload reaches the next handshake.
    ///
    /// # Errors
    ///
    /// [`TransportError::Tls`] when the verifier or the configuration cannot be built.
    pub fn server_config(
        &self,
        client_auth: ClientAuth,
    ) -> Result<rustls::ServerConfig, TransportError> {
        let material = self.material();
        let builder = rustls::server::WebPkiClientVerifier::builder(Arc::clone(&material.roots));
        let builder = match client_auth {
            ClientAuth::Required => builder,
            ClientAuth::Optional => builder.allow_unauthenticated(),
        };
        let verifier = builder.build().map_err(|e| TransportError::Tls {
            path: self.files.ca.clone(),
            reason: e.to_string(),
        })?;
        Ok(rustls::ServerConfig::builder()
            .with_client_cert_verifier(verifier)
            .with_single_cert(material.certs.clone(), material.key.clone_key())?)
    }

    fn connector(&self) -> Result<TlsConnector, TransportError> {
        Ok(TlsConnector::from(Arc::new(self.client_config()?)))
    }

    /// A client configuration with the material in use: servers verified against the CA, this
    /// process's certificate presented as its own.
    ///
    /// # Errors
    ///
    /// [`TransportError::Tls`] when the certificate and key do not make a configuration.
    pub fn client_config(&self) -> Result<rustls::ClientConfig, TransportError> {
        let material = self.material();
        Ok(rustls::ClientConfig::builder()
            .with_root_certificates(Arc::clone(&material.roots))
            .with_client_auth_cert(material.certs.clone(), material.key.clone_key())?)
    }
}

impl Files {
    fn bytes(&self) -> Result<[Vec<u8>; 3], TransportError> {
        let read = |path: &Path| {
            std::fs::read(path).map_err(|e| TransportError::Tls {
                path: path.to_path_buf(),
                reason: e.to_string(),
            })
        };
        Ok([read(&self.cert)?, read(&self.key)?, read(&self.ca)?])
    }

    fn read(&self) -> Result<Material, TransportError> {
        self.parse(self.bytes()?)
    }

    fn parse(&self, read: [Vec<u8>; 3]) -> Result<Material, TransportError> {
        let [cert, key, ca] = &read;
        let certs = pem_certs(&self.cert, cert)?;
        if certs.is_empty() {
            return Err(TransportError::Tls {
                path: self.cert.clone(),
                reason: "no certificate".into(),
            });
        }
        let refused = |reason: String| TransportError::Tls {
            path: self.key.clone(),
            reason,
        };
        let key = PrivateKeyDer::from_pem_slice(key).map_err(|e| match e {
            pem::Error::NoItemsFound => refused("no private key".into()),
            other => refused(other.to_string()),
        })?;
        // a certificate and a key from two renewals, read between their writes, make a
        // configuration that fails every handshake: refused here, so the pair in use stays
        let signer = rustls::crypto::ring::sign::any_supported_type(&key)
            .map_err(|e| refused(e.to_string()))?;
        rustls::sign::CertifiedKey::new(certs.clone(), signer)
            .keys_match()
            .map_err(|e| refused(format!("not the certificate's key: {e}")))?;
        Ok(Material {
            certs,
            key,
            roots: roots_of(&self.ca, ca)?,
            read,
        })
    }
}

/// The CA certificates in the PEM file at `ca`, as a root store: what a client that
/// authenticates another way (a bearer token to a Kubernetes API server) verifies the server by.
///
/// # Errors
///
/// [`TransportError::Tls`] when the file cannot be read or holds no certificate.
pub fn root_store(ca: &Path) -> Result<Arc<rustls::RootCertStore>, TransportError> {
    let bytes = std::fs::read(ca).map_err(|e| TransportError::Tls {
        path: ca.to_path_buf(),
        reason: e.to_string(),
    })?;
    roots_of(ca, &bytes)
}

/// The CA certificates in `pem`, read from `path`, as a root store.
fn roots_of(path: &Path, pem: &[u8]) -> Result<Arc<rustls::RootCertStore>, TransportError> {
    let mut roots = rustls::RootCertStore::empty();
    for c in pem_certs(path, pem)? {
        roots.add(c).map_err(|e| TransportError::Tls {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?;
    }
    if roots.is_empty() {
        return Err(TransportError::Tls {
            path: path.to_path_buf(),
            reason: "no CA certificate".into(),
        });
    }
    Ok(Arc::new(roots))
}

/// The certificates in `pem`, read from `path`.
fn pem_certs(
    path: &Path,
    pem: &[u8],
) -> Result<Vec<rustls::pki_types::CertificateDer<'static>>, TransportError> {
    CertificateDer::pem_slice_iter(pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| TransportError::Tls {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })
}

/// `wait` with a quarter of itself taken or added at random, so peers that lost the same link
/// do not reconnect in step (every daemon after a coordinator restart, every carrier after a
/// replica's). The randomness is the clock's nanoseconds: enough to spread, no dependency.
#[must_use]
pub fn jittered(wait: Duration) -> Duration {
    let nanos = u64::from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos()),
    );
    let quarter = wait / 4;
    let span = u64::try_from(quarter.as_nanos() * 2)
        .unwrap_or(u64::MAX)
        .max(1);
    wait.saturating_sub(quarter) + Duration::from_nanos(nanos % span)
}

/// The longest a TCP connect or a TLS handshake may take, either side. A peer that connects and
/// never finishes its handshake must not hold a listener or a connector for longer.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// How often a process reads its TLS files again ([`Tls::follow`]): a renewal is in use within
/// this of being written, far inside the half life left when a node renews (decision 0037).
pub const TLS_RELOAD_EVERY: Duration = Duration::from_secs(30);

/// A bound listener of either kind.
pub enum Listener {
    /// A Unix socket at its path. The path stays when the listener is dropped: the next bind
    /// replaces it, and while none listens a connect to it is refused as to no path at all. A
    /// dropped listener that removed it could remove a successor's socket bound at the same path
    /// since, and no inode check tells them apart where the system reuses the freed inode (Linux
    /// does), and no one could connect again.
    Unix(UnixListener, PathBuf),
    /// A TCP listener with the TLS material every connection's handshake uses, as it is at that
    /// handshake.
    Tcp(TcpListener, Tls),
}

impl Listener {
    /// Bind `address`. A TCP address needs `tls`; a Unix path is replaced if it exists.
    ///
    /// # Errors
    ///
    /// `TlsRequired` for TCP without TLS; the bind error otherwise.
    pub async fn bind(address: &Address, tls: Option<&Tls>) -> Result<Self, TransportError> {
        match address {
            Address::Unix(path) => {
                if path.exists() {
                    std::fs::remove_file(path)?;
                }
                Ok(Self::Unix(UnixListener::bind(path)?, path.clone()))
            }
            Address::Tcp(addr) => {
                let tls = tls.ok_or(TransportError::TlsRequired(*addr))?;
                // a configuration that cannot be made fails the bind, not the first peer
                tls.acceptor()?;
                Ok(Self::Tcp(TcpListener::bind(addr).await?, tls.clone()))
            }
        }
    }

    /// The address actually bound (a TCP listener on port 0 learns its port here).
    ///
    /// # Errors
    ///
    /// The socket's own error.
    pub fn local_address(&self) -> Result<Address, TransportError> {
        Ok(match self {
            Self::Unix(_, p) => Address::Unix(p.clone()),
            Self::Tcp(l, _) => Address::Tcp(l.local_addr()?),
        })
    }

    /// Accept one connection, handshake included, with who connected.
    ///
    /// # Errors
    ///
    /// The accept or handshake error; a TLS peer that presented no certificate is refused.
    pub async fn accept(&self) -> Result<Accepted, TransportError> {
        match self {
            Self::Unix(l, _) => {
                let (s, _) = l.accept().await?;
                Ok(Accepted {
                    stream: Box::pin(s),
                    peer: Peer::Local,
                })
            }
            Self::Tcp(l, tls) => {
                let (s, _) = l.accept().await?;
                let acceptor = tls.acceptor()?;
                // every frame is a message and most are answered: no Nagle, or a request and
                // its reply wait on the peer's delayed acknowledgement
                s.set_nodelay(true)?;
                let s = tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(s))
                    .await
                    .map_err(|_| TransportError::Timeout {
                        what: "the TLS handshake",
                    })??;
                let leaf = s
                    .get_ref()
                    .1
                    .peer_certificates()
                    .and_then(|certs| certs.first())
                    .cloned()
                    .ok_or_else(|| TransportError::Tls {
                        path: PathBuf::from("peer"),
                        reason: "no certificate presented".into(),
                    })?;
                Ok(Accepted {
                    stream: Box::pin(s),
                    peer: Peer::Certified(leaf.into_owned()),
                })
            }
        }
    }
}

/// Connect to `address`. A TCP address needs `tls` and `server_name`, the name on the peer's
/// certificate.
///
/// # Errors
///
/// `TlsRequired` for TCP without TLS; the connect or handshake error otherwise.
pub async fn connect(
    address: &Address,
    tls: Option<&Tls>,
    server_name: &str,
) -> Result<Stream, TransportError> {
    match address {
        Address::Unix(path) => Ok(Box::pin(UnixStream::connect(path).await?)),
        Address::Tcp(addr) => {
            let tls = tls.ok_or(TransportError::TlsRequired(*addr))?;
            let name =
                rustls::pki_types::ServerName::try_from(server_name.to_owned()).map_err(|e| {
                    TransportError::Tls {
                        path: PathBuf::from(server_name),
                        reason: format!("not a server name: {e}"),
                    }
                })?;
            let connector = tls.connector()?;
            let s = tokio::time::timeout(HANDSHAKE_TIMEOUT, async {
                let s = TcpStream::connect(addr).await?;
                s.set_nodelay(true)?; // as on accept: a frame is a message, sent at once
                connector.connect(name, s).await
            })
            .await
            .map_err(|_| TransportError::Timeout {
                what: "the connect and TLS handshake",
            })??;
            Ok(Box::pin(s))
        }
    }
}
