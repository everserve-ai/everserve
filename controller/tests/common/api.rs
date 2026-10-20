// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator API as the tests reach it (decision 0039): served over TLS behind the certificate
//! gate, with an operator's, a viewer's and an anonymous client.

use std::path::Path;
use std::sync::Arc;

use everserve_controller::api::{CertificateRoles, Deadlines, Gates, Pacing};
use everserve_controller::config::CertsArgs;
use everserve_controller::coord::Coordinator;
use everserve_controller::http::{Client, Method};
use everserve_wire::transport::{Tls, root_store};
use tokio_util::sync::CancellationToken;

/// The operator API of one coordinator, and its callers.
pub struct OperatorApi {
    /// `https://localhost:<port>`.
    pub base: String,
    /// An operator's client: may do anything.
    pub operator: Client,
    /// A viewer's client: may read.
    pub viewer: Client,
    /// A client with no certificate.
    pub anonymous: Client,
    /// The material the API was served with, in `dir/api`: `ca.crt`, `coordinator.crt`, and
    /// `operator-operator.crt` and `operator-viewer.crt` with their keys.
    pub certs: std::path::PathBuf,
}

impl OperatorApi {
    /// `method path` with a JSON `body` as `client`: the status and the body.
    pub async fn call(
        &self,
        client: &Client,
        method: Method,
        path: &str,
        body: &str,
    ) -> (u16, String) {
        let answer = client
            .send(
                method,
                &format!("{}{path}", self.base),
                &[("Content-Type", "application/json")],
                (!body.is_empty()).then(|| body.as_bytes().to_vec()),
            )
            .await
            .expect("request");
        (answer.status, answer.body)
    }
}

/// The client presenting the certificate `name` in `certs`.
pub fn client_as(certs: &Path, name: &str) -> Client {
    let tls = Tls::load(
        &certs.join(format!("{name}.crt")),
        &certs.join(format!("{name}.key")),
        &certs.join("ca.crt"),
    )
    .expect("tls");
    Client::with_tls(Arc::new(tls.client_config().expect("config")), None)
}

/// Serve `coord`'s operator API on loopback until `cancel` fires, with an operator named
/// `operator` and a viewer named `viewer`.
pub async fn serve(
    coord: &Arc<Coordinator>,
    dir: &Path,
    cancel: &CancellationToken,
) -> OperatorApi {
    let roles = CertificateRoles {
        operators: vec!["operator".into()],
        viewers: vec!["viewer".into()],
    };
    serve_behind(coord, dir, cancel, |_| Gates {
        certificates: Some(roles),
        kubernetes: None,
    })
    .await
}

/// As [`serve`], behind the gates `gates` makes from the material's directory.
pub async fn serve_behind(
    coord: &Arc<Coordinator>,
    dir: &Path,
    cancel: &CancellationToken,
    gates: impl FnOnce(&Path) -> Gates,
) -> OperatorApi {
    serve_within(coord, dir, cancel, gates, Deadlines::SERVED).await
}

/// As [`serve_behind`], each request within `deadlines`.
pub async fn serve_within(
    coord: &Arc<Coordinator>,
    dir: &Path,
    cancel: &CancellationToken,
    gates: impl FnOnce(&Path) -> Gates,
    deadlines: Deadlines,
) -> OperatorApi {
    serve_with(coord, dir, cancel, gates, deadlines, Pacing::SERVED).await
}

/// As [`serve`], its objects' feed kept at `pacing`.
pub async fn serve_paced(
    coord: &Arc<Coordinator>,
    dir: &Path,
    cancel: &CancellationToken,
    pacing: Pacing,
) -> OperatorApi {
    let roles = CertificateRoles {
        operators: vec!["operator".into()],
        viewers: vec!["viewer".into()],
    };
    let gates = |_: &Path| Gates {
        certificates: Some(roles),
        kubernetes: None,
    };
    serve_with(coord, dir, cancel, gates, Deadlines::SERVED, pacing).await
}

async fn serve_with(
    coord: &Arc<Coordinator>,
    dir: &Path,
    cancel: &CancellationToken,
    gates: impl FnOnce(&Path) -> Gates,
    deadlines: Deadlines,
    pacing: Pacing,
) -> OperatorApi {
    let certs = dir.join("api");
    everserve_controller::certs::write(&CertsArgs {
        out: certs.clone(),
        nodes: Vec::new(),
        operators: vec!["operator".into(), "viewer".into()],
        // the server's certificate, valid for the name the clients reach it by
        coordinator_name: "localhost".into(),
        store_name: "store".into(),
        days: 1,
        overwrite: false,
    })
    .expect("certs");
    let tls = Tls::load(
        &certs.join("coordinator.crt"),
        &certs.join("coordinator.key"),
        &certs.join("ca.crt"),
    )
    .expect("tls");
    let gate = Arc::new(gates(&certs));
    let port = everserve_controller::api::start(
        everserve_controller::api::routes::router(everserve_controller::api::Feed::start(
            Arc::clone(coord),
            pacing,
            cancel,
        )),
        axum::Router::new(),
        "127.0.0.1:0".parse().expect("addr"),
        &tls,
        gate,
        deadlines,
        cancel.clone(),
    )
    .await
    .expect("bound")
    .port();
    OperatorApi {
        base: format!("https://localhost:{port}"),
        operator: client_as(&certs, "operator-operator"),
        viewer: client_as(&certs, "operator-viewer"),
        anonymous: Client::with(Some(root_store(&certs.join("ca.crt")).expect("ca")), None),
        certs,
    }
}

/// A TLS connection to `api` presenting the operator's certificate, nothing sent on it yet.
pub async fn operator_connection(
    api: &OperatorApi,
) -> tokio_rustls::client::TlsStream<tokio::net::TcpStream> {
    let tls = everserve_wire::transport::Tls::load(
        &api.certs.join("operator-operator.crt"),
        &api.certs.join("operator-operator.key"),
        &api.certs.join("ca.crt"),
    )
    .expect("tls");
    let connector =
        tokio_rustls::TlsConnector::from(Arc::new(tls.client_config().expect("config")));
    let addr = api.base.trim_start_matches("https://");
    let tcp = tokio::net::TcpStream::connect(addr.replace("localhost", "127.0.0.1"))
        .await
        .expect("connect");
    connector
        .connect("localhost".try_into().expect("name"), tcp)
        .await
        .expect("handshake")
}

/// What `conn` reads until the server closes it; within `bound`, or the test fails.
pub async fn read_to_close(
    conn: &mut tokio_rustls::client::TlsStream<tokio::net::TcpStream>,
    bound: std::time::Duration,
) -> String {
    use tokio::io::AsyncReadExt;
    let mut got = Vec::new();
    let _ = tokio::time::timeout(bound, conn.read_to_end(&mut got))
        .await
        .expect("the server closed the connection in time");
    String::from_utf8_lossy(&got).into_owned()
}
