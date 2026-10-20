// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator API's client (decisions 0039, 0042): what `everserve-controller check`, `get`,
//! `watch`, `cordon`, `drain` and the rest call. A write to a node reads the node, changes its
//! spec, and writes it back at the version it read: one that changed meanwhile is refused, and
//! said so, never written over.

use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::api::error::ApiError;
use crate::api::error::Reason;
use crate::api::objects::{Drain, List, Node, NodeSpec, Write, WriteMetadata};
use crate::config::{ApiAccessArgs, ObjectKind};
use crate::http::{Client, Method, Streamed};

/// How long a request may take before it is given up.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Why a call to the operator API did not do what it asked.
#[derive(Debug, thiserror::Error)]
pub enum OperatorError {
    /// The credentials could not be used.
    #[error("credentials: {0}")]
    Credentials(String),
    /// The coordinator could not be reached.
    #[error("{0}")]
    Unreachable(String),
    /// The API refused, as its error says.
    #[error("{} {:?}: {}", .0.code, .0.reason, .0.message)]
    Refused(ApiError),
    /// The answer was not what the API answers.
    #[error("unexpected answer {status}: {body}")]
    Unexpected {
        /// Its status.
        status: u16,
        /// Its body.
        body: String,
    },
}

/// How many objects one page of a list holds.
pub const PAGE: usize = 500;

/// How many times a list whose first version expired while it was read starts again.
pub const LIST_TRIES: u32 = 3;

/// A caller of the operator API, with its credential.
pub struct Operator {
    client: Client,
    base: String,
}

impl Operator {
    /// The caller `args` names: an operator's or a viewer's certificate, or a bearer token, the
    /// coordinator verified against the CA by the name on its certificate.
    ///
    /// # Errors
    ///
    /// [`OperatorError::Credentials`] when neither credential is given or one cannot be read.
    pub fn connect(args: &ApiAccessArgs) -> Result<Self, OperatorError> {
        let unusable = |e: everserve_wire::transport::TransportError| {
            OperatorError::Credentials(e.to_string())
        };
        let client = match (&args.cert, &args.key, &args.token_file) {
            (Some(cert), Some(key), _) => {
                let tls =
                    everserve_wire::transport::Tls::load(cert, key, &args.ca).map_err(unusable)?;
                Client::with_tls(
                    std::sync::Arc::new(tls.client_config().map_err(unusable)?),
                    None,
                )
            }
            (_, _, Some(token)) => Client::with(
                Some(everserve_wire::transport::root_store(&args.ca).map_err(unusable)?),
                Some(crate::http::Bearer::File(token.clone())),
            ),
            _ => {
                return Err(OperatorError::Credentials(
                    "present --cert and --key, or --token-file".into(),
                ));
            }
        };
        let base = if args.coordinator_api.starts_with("https://") {
            args.coordinator_api.trim_end_matches('/').to_owned()
        } else {
            format!("https://{}", args.coordinator_api)
        };
        Ok(Self {
            client: client
                .server_name(args.coordinator_name.clone())
                .timeout(REQUEST_TIMEOUT),
            base,
        })
    }

    async fn call(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> Result<(u16, String), OperatorError> {
        let answer = self
            .client
            .send(
                method,
                &format!("{}{path}", self.base),
                &[("Content-Type", "application/json")],
                body,
            )
            .await
            .map_err(|e| OperatorError::Unreachable(e.to_string()))?;
        Ok((answer.status, answer.body))
    }

    /// `path`'s object, or the API's refusal.
    ///
    /// # Errors
    ///
    /// [`OperatorError`] when the coordinator cannot be reached or refuses.
    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, OperatorError> {
        let (status, body) = self.call(Method::GET, path, None).await?;
        answered(status, &body)
    }

    /// Every object of `kind`, read [`PAGE`] at a time: one list, at the first page's version,
    /// so a watch from it misses nothing. A list whose first page's version expires while it is
    /// read starts again, at most [`LIST_TRIES`] times.
    ///
    /// # Errors
    ///
    /// [`OperatorError`] when the coordinator cannot be reached or refuses.
    pub async fn list(&self, kind: ObjectKind) -> Result<List<Value>, OperatorError> {
        self.list_by(kind, PAGE).await
    }

    /// [`Operator::list`], `page` objects at a time.
    ///
    /// # Errors
    ///
    /// [`OperatorError`] when the coordinator cannot be reached or refuses.
    pub async fn list_by(
        &self,
        kind: ObjectKind,
        page: usize,
    ) -> Result<List<Value>, OperatorError> {
        let mut tries = 0;
        'list: loop {
            tries += 1;
            let mut list: List<Value> = self
                .get(&format!("/v1/{}?limit={page}", kind.path()))
                .await?;
            while let Some(token) = list.metadata.continue_token.take() {
                let path = format!("/v1/{}?limit={page}&continue={token}", kind.path());
                match self.get::<List<Value>>(&path).await {
                    Ok(next) => {
                        list.items.extend(next.items);
                        list.metadata.continue_token = next.metadata.continue_token;
                    }
                    Err(OperatorError::Refused(e))
                        if e.reason == Reason::Gone && tries < LIST_TRIES =>
                    {
                        continue 'list;
                    }
                    Err(e) => return Err(e),
                }
            }
            return Ok(list);
        }
    }

    /// Write `body` to `path` with `method`; the API's answer, or its refusal.
    ///
    /// # Errors
    ///
    /// [`OperatorError`] when the coordinator cannot be reached or refuses.
    pub async fn send<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, OperatorError> {
        let bytes = serde_json::to_vec(body).map_err(|e| OperatorError::Unexpected {
            status: 0,
            body: e.to_string(),
        })?;
        let (status, answer) = self.call(method, path, Some(bytes)).await?;
        answered(status, &answer)
    }

    /// The changes of `kind`'s objects as they come, starting with every object.
    ///
    /// # Errors
    ///
    /// [`OperatorError`] when the coordinator cannot be reached or refuses.
    pub async fn watch(&self, kind: ObjectKind) -> Result<Streamed, OperatorError> {
        let url = format!("{}/v1/{}?watch=true", self.base, kind.path());
        let streamed = self
            .client
            .stream(&url, &[])
            .await
            .map_err(|e| OperatorError::Unreachable(e.to_string()))?;
        if streamed.status != 200 {
            return Err(OperatorError::Unexpected {
                status: streamed.status,
                body: String::new(),
            });
        }
        Ok(streamed)
    }

    /// Make `node`'s spec what `change` makes of it, written at the version it was read at; a
    /// node never written has a spec of nothing set.
    ///
    /// # Errors
    ///
    /// [`OperatorError::Refused`] when the API refuses it: the node changed since it was read
    /// (`Conflict`), its drain is underway (`DrainUnderway`), the spec asks what cannot be.
    pub async fn write_node(
        &self,
        node: &str,
        change: impl FnOnce(&mut NodeSpec),
    ) -> Result<Node, OperatorError> {
        let path = format!("/v1/nodes/{node}");
        let (status, body) = self.call(Method::GET, &path, None).await?;
        let (mut spec, version) = if status == 404 {
            (NodeSpec::default(), None)
        } else {
            let read: Node = answered(status, &body)?;
            (read.spec, Some(read.metadata.resource_version))
        };
        change(&mut spec);
        let write = Write {
            metadata: WriteMetadata {
                resource_version: version,
            },
            spec,
        };
        self.send(Method::PUT, &path, &write).await
    }

    /// Cordon `node`, or uncordon it, for `reason`.
    ///
    /// # Errors
    ///
    /// As [`Operator::write_node`].
    pub async fn cordon(&self, node: &str, on: bool, reason: &str) -> Result<Node, OperatorError> {
        self.write_node(node, |s| {
            s.unschedulable = on;
            reason.clone_into(&mut s.reason);
        })
        .await
    }

    /// Drain `node` by `deadline_ms`, Unix milliseconds, for `reason`: cordoned and drained.
    ///
    /// # Errors
    ///
    /// As [`Operator::write_node`].
    pub async fn drain(
        &self,
        node: &str,
        deadline_ms: i64,
        reason: &str,
    ) -> Result<Node, OperatorError> {
        self.write_node(node, |s| {
            s.unschedulable = true;
            s.drain = Some(Drain { deadline_ms });
            reason.clone_into(&mut s.reason);
        })
        .await
    }

    /// Withdraw `node`'s drain, for `reason`; the node stays cordoned.
    ///
    /// # Errors
    ///
    /// As [`Operator::write_node`].
    pub async fn undrain(&self, node: &str, reason: &str) -> Result<Node, OperatorError> {
        self.write_node(node, |s| {
            s.drain = None;
            reason.clone_into(&mut s.reason);
        })
        .await
    }
}

/// `body` as the 200 it should be, or the API's error.
fn answered<T: DeserializeOwned>(status: u16, body: &str) -> Result<T, OperatorError> {
    if status == 200 || status == 202 {
        return serde_json::from_str(body).map_err(|e| OperatorError::Unexpected {
            status,
            body: format!("{e}: {body}"),
        });
    }
    Err(match serde_json::from_str::<ApiError>(body) {
        Ok(e) => OperatorError::Refused(e),
        Err(_) => OperatorError::Unexpected {
            status,
            body: body.to_owned(),
        },
    })
}

/// `objects` of `kind`, as `get` prints them: one line each under a header, columns aligned.
#[must_use]
pub fn table(kind: ObjectKind, objects: &[Value]) -> String {
    let header: &[&str] = match kind {
        ObjectKind::Engines => &["NAME", "STATE", "EPOCH", "MEMBERS", "SPARES", "COVERED"],
        ObjectKind::Nodes => &["NAME", "LINKED", "CORDONED", "DRAINING", "ENGINES"],
        ObjectKind::Pools => &["NAME", "WANTS", "STANDING", "WARMING", "DEFICIT"],
    };
    let mut rows: Vec<Vec<String>> = vec![header.iter().map(|h| (*h).to_owned()).collect()];
    for o in objects {
        let s = &o["status"];
        let condition = |kind: &str| {
            s["conditions"]
                .as_array()
                .and_then(|cs| cs.iter().find(|c| c["type"] == kind))
                .map_or("-".to_owned(), |c| {
                    if c["status"] == "True" {
                        text(&c["reason"])
                    } else {
                        "-".to_owned()
                    }
                })
        };
        let len = |v: &Value| v.as_array().map_or(0, Vec::len).to_string();
        rows.push(match kind {
            ObjectKind::Engines => vec![
                text(&o["metadata"]["name"]),
                text(&s["state"]),
                text(&s["epoch"]),
                format!("{}/{}", len(&s["members"]), text(&s["worldSize"])),
                (s["spares"].as_array().map_or(0, Vec::len)
                    + s["anySpares"].as_array().map_or(0, Vec::len))
                .to_string(),
                s["conditions"]
                    .as_array()
                    .and_then(|cs| cs.iter().find(|c| c["type"] == "Covered"))
                    .map_or("-".to_owned(), |c| text(&c["status"]).to_lowercase()),
            ],
            ObjectKind::Nodes => vec![
                text(&o["metadata"]["name"]),
                text(&s["linked"]),
                text(&o["spec"]["unschedulable"]),
                condition("Draining"),
                s["engines"].as_array().map_or("-".to_owned(), |es| {
                    let names: Vec<String> = es.iter().map(|e| text(&e["engine"])).collect();
                    if names.is_empty() {
                        "-".to_owned()
                    } else {
                        names.join(",")
                    }
                }),
            ],
            ObjectKind::Pools => vec![
                text(&o["metadata"]["name"]),
                text(&o["spec"]["wants"]),
                text(&s["standing"]),
                text(&s["warming"]),
                text(&s["deficit"]),
            ],
        });
    }
    let widths: Vec<usize> = (0..header.len())
        .map(|i| rows.iter().map(|r| r[i].len()).max().unwrap_or(0))
        .collect();
    rows.iter()
        .map(|r| {
            r.iter()
                .zip(&widths)
                .map(|(c, w)| format!("{c:w$}"))
                .collect::<Vec<_>>()
                .join("  ")
                .trim_end()
                .to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A JSON value as a table cell: a string without its quotes, anything else as it prints.
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "-".to_owned(),
        other => other.to_string(),
    }
}
