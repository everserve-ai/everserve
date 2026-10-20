// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! One request, one answer: the HTTP/1.1 client the controller's own callers share (the notice
//! sources, the `check` command); and one request, an answer read as it comes, for a Kubernetes
//! watch.
//!
//! Plain HTTP for link-local metadata endpoints and the control plane's own ports; HTTPS verified
//! by a given root store, with an optional bearer token, for a Kubernetes API server (see
//! `crate::kubernetes`). A token kept in a file is read again at every request: the kubelet
//! rotates a projected service account token, and a token read once expires within the hour. Built on
//! `hyper`'s connection API over the crates already in the tree, with no connection pool: every
//! caller here makes one request every second or so and a fresh connection per request is what
//! the metadata services expect.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use http_body_util::{BodyExt, Full};
pub use hyper::Method;
use hyper::body::{Bytes, Incoming};
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::net::TcpStream;

/// How long one request may take, connect to last byte.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Why a request failed.
#[derive(Debug, thiserror::Error)]
#[error("{url}: {reason}")]
pub struct HttpError {
    /// The URL asked for.
    pub url: String,
    /// What went wrong.
    pub reason: String,
}

/// The bearer token a client sends.
#[derive(Clone, Debug)]
pub enum Bearer {
    /// This token.
    Token(String),
    /// The token in this file, read again at every request.
    File(PathBuf),
}

/// A client: plain, or verifying servers by its TLS configuration and sending `bearer`.
#[derive(Clone)]
pub struct Client {
    tls: Option<Arc<rustls::ClientConfig>>,
    bearer: Option<Bearer>,
    timeout: Duration,
    /// The name a server's certificate is verified for, where it is not the URL's host: a
    /// coordinator's certificate carries its role's name (`--coordinator-name`), not an address.
    server_name: Option<String>,
}

impl Default for Client {
    fn default() -> Self {
        Self {
            tls: None,
            bearer: None,
            timeout: REQUEST_TIMEOUT,
            server_name: None,
        }
    }
}

/// An answer: the status and the body as text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    /// The status code.
    pub status: u16,
    /// The body.
    pub body: String,
}

impl Client {
    /// A plain client.
    #[must_use]
    pub fn plain() -> Self {
        Self::default()
    }

    /// A client that verifies `https` servers by `roots` (none: `https` is refused) and sends
    /// `bearer` as its authorization.
    #[must_use]
    pub fn with(roots: Option<Arc<rustls::RootCertStore>>, bearer: Option<Bearer>) -> Self {
        let tls = roots.map(|roots| {
            Arc::new(
                rustls::ClientConfig::builder()
                    .with_root_certificates(roots)
                    .with_no_client_auth(),
            )
        });
        Self {
            tls,
            bearer,
            timeout: REQUEST_TIMEOUT,
            server_name: None,
        }
    }

    /// A client that speaks `https` with `tls` (a client certificate in it is presented, as an
    /// operator's is to the operator API, decision 0039) and sends `bearer` as its authorization.
    #[must_use]
    pub fn with_tls(tls: Arc<rustls::ClientConfig>, bearer: Option<Bearer>) -> Self {
        Self {
            tls: Some(tls),
            bearer,
            timeout: REQUEST_TIMEOUT,
            server_name: None,
        }
    }

    /// This client verifying servers' certificates for `name` rather than the URL's host.
    #[must_use]
    pub fn server_name(self, name: impl Into<String>) -> Self {
        Self {
            server_name: Some(name.into()),
            ..self
        }
    }

    /// This client with requests bounded by `timeout`, connect to last byte, rather than
    /// [`REQUEST_TIMEOUT`].
    #[must_use]
    pub fn timeout(self, timeout: Duration) -> Self {
        Self { timeout, ..self }
    }

    /// `GET url` with `headers`.
    ///
    /// # Errors
    ///
    /// [`HttpError`] when the request fails or times out.
    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<Answer, HttpError> {
        self.send(Method::GET, url, headers, None).await
    }

    /// `PUT url` with `headers` and no body.
    ///
    /// # Errors
    ///
    /// As [`Client::get`].
    pub async fn put(&self, url: &str, headers: &[(&str, &str)]) -> Result<Answer, HttpError> {
        self.send(Method::PUT, url, headers, None).await
    }

    /// `method url` with `headers` and `body`.
    ///
    /// # Errors
    ///
    /// [`HttpError`] when the token cannot be read, or the request fails or times out.
    pub async fn send(
        &self,
        method: Method,
        url: &str,
        headers: &[(&str, &str)],
        body: Option<Vec<u8>>,
    ) -> Result<Answer, HttpError> {
        let fail = |reason: String| HttpError {
            url: url.to_owned(),
            reason,
        };
        let target = Target::parse(url).ok_or_else(|| fail("not an http or https url".into()))?;
        let token = self.token().await.map_err(fail)?;
        let exchange = async {
            let res = self
                .exchange(method, &target, headers, token, body.unwrap_or_default())
                .await?;
            let status = res.status().as_u16();
            let body = res.into_body().collect().await?.to_bytes();
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(Answer {
                status,
                body: String::from_utf8_lossy(&body).into_owned(),
            })
        };
        tokio::time::timeout(self.timeout, exchange)
            .await
            .map_err(|_| fail(format!("no answer within {:?}", self.timeout)))?
            .map_err(|e| fail(e.to_string()))
    }

    /// `GET url` with `headers`, its body read as it comes: the status within the client's
    /// timeout, then the body for as long as the server sends it. A Kubernetes watch.
    ///
    /// # Errors
    ///
    /// [`HttpError`] when the token cannot be read, or the request fails or has no answer in time.
    pub async fn stream(&self, url: &str, headers: &[(&str, &str)]) -> Result<Streamed, HttpError> {
        let fail = |reason: String| HttpError {
            url: url.to_owned(),
            reason,
        };
        let target = Target::parse(url).ok_or_else(|| fail("not an http or https url".into()))?;
        let token = self.token().await.map_err(fail)?;
        let exchange = self.exchange(Method::GET, &target, headers, token, Vec::new());
        let res = tokio::time::timeout(self.timeout, exchange)
            .await
            .map_err(|_| fail(format!("no answer within {:?}", self.timeout)))?
            .map_err(|e| fail(e.to_string()))?;
        Ok(Streamed {
            status: res.status().as_u16(),
            body: res.into_body(),
            pending: Vec::new(),
        })
    }

    /// The bearer token to send, read again from its file every time.
    async fn token(&self) -> Result<Option<String>, String> {
        Ok(match &self.bearer {
            None => None,
            Some(Bearer::Token(t)) => Some(t.clone()),
            Some(Bearer::File(path)) => Some(
                tokio::fs::read_to_string(path)
                    .await
                    .map_err(|e| format!("{}: {e}", path.display()))?
                    .trim()
                    .to_owned(),
            ),
        })
    }

    async fn exchange(
        &self,
        method: Method,
        target: &Target,
        headers: &[(&str, &str)],
        token: Option<String>,
        body: Vec<u8>,
    ) -> Result<Response<Incoming>, Box<dyn std::error::Error + Send + Sync>> {
        let mut req = Request::builder()
            .method(method)
            .uri(&target.path)
            .header(hyper::header::HOST, &target.host_header)
            .header(hyper::header::CONNECTION, "close");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        if let Some(token) = token {
            req = req.header(hyper::header::AUTHORIZATION, format!("Bearer {token}"));
        }
        let req = req.body(Full::new(Bytes::from(body)))?;
        let tcp = TcpStream::connect((target.host.as_str(), target.port)).await?;
        let response = if target.tls {
            let config = self
                .tls
                .clone()
                .ok_or("an https endpoint needs a CA to verify it")?;
            let name = rustls::pki_types::ServerName::try_from(
                self.server_name
                    .clone()
                    .unwrap_or_else(|| target.host.clone()),
            )?;
            let stream = tokio_rustls::TlsConnector::from(config)
                .connect(name, tcp)
                .await?;
            send(TokioIo::new(stream), req).await?
        } else {
            send(TokioIo::new(tcp), req).await?
        };
        Ok(response)
    }
}

async fn send<S>(
    io: TokioIo<S>,
    req: Request<Full<Bytes>>,
) -> Result<Response<Incoming>, Box<dyn std::error::Error + Send + Sync>>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await?;
    tokio::spawn(conn);
    Ok(sender.send_request(req).await?)
}

/// An answer read as it comes: its status, and its body line by line.
pub struct Streamed {
    /// The status code.
    pub status: u16,
    body: Incoming,
    /// Bytes read past the last whole line.
    pending: Vec<u8>,
}

impl Streamed {
    /// The body's next line, without its newline; None once the body ends. A last line with no
    /// newline is returned as it is.
    ///
    /// # Errors
    ///
    /// The connection failed while the body was read.
    pub async fn line(&mut self) -> Result<Option<String>, HttpError> {
        loop {
            if let Some(at) = self.pending.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = self.pending.drain(..=at).collect();
                return Ok(Some(
                    String::from_utf8_lossy(&line[..line.len() - 1]).into_owned(),
                ));
            }
            match self.body.frame().await {
                Some(Ok(frame)) => {
                    if let Ok(data) = frame.into_data() {
                        self.pending.extend_from_slice(&data);
                    }
                }
                Some(Err(e)) => {
                    return Err(HttpError {
                        url: String::new(),
                        reason: e.to_string(),
                    });
                }
                None if self.pending.is_empty() => return Ok(None),
                None => {
                    let line = std::mem::take(&mut self.pending);
                    return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
                }
            }
        }
    }
}

/// The pieces of an `http://host[:port]/path` or `https://...` URL.
#[derive(Debug, PartialEq, Eq)]
struct Target {
    tls: bool,
    host: String,
    port: u16,
    host_header: String,
    path: String,
}

impl Target {
    fn parse(url: &str) -> Option<Self> {
        let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
            (true, r)
        } else {
            (false, url.strip_prefix("http://")?)
        };
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) if !h.contains(']') || h.ends_with(']') => (h, p.parse().ok()?),
            _ => (authority, if tls { 443 } else { 80 }),
        };
        if host.is_empty() {
            return None;
        }
        Some(Self {
            tls,
            host: host.trim_matches(['[', ']']).to_owned(),
            port,
            host_header: authority.to_owned(),
            path: path.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Target;

    #[test]
    fn urls_split_into_their_pieces() {
        let t =
            Target::parse("http://169.254.169.254/latest/meta-data/spot/instance-action").unwrap();
        assert_eq!(
            (t.tls, t.host.as_str(), t.port, t.path.as_str()),
            (
                false,
                "169.254.169.254",
                80,
                "/latest/meta-data/spot/instance-action"
            )
        );
        let t = Target::parse("https://10.0.0.1:6443/api/v1/nodes/n").unwrap();
        assert_eq!(
            (t.tls, t.host.as_str(), t.port, t.host_header.as_str()),
            (true, "10.0.0.1", 6443, "10.0.0.1:6443")
        );
        let t = Target::parse("http://127.0.0.1:8080").unwrap();
        assert_eq!((t.port, t.path.as_str()), (8080, "/"));
        assert!(Target::parse("ftp://x/").is_none());
        assert!(Target::parse("http:///x").is_none());
    }
}
