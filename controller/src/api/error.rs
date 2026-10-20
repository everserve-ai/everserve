// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The one shape every error of the operator API takes (decision 0042): its HTTP status, a reason
//! a client can act on without reading the message, and the message for people.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

/// Why a request was not done, from a closed set a client matches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub enum Reason {
    /// No credential, or one that is not the deployment's or the cluster's (401).
    Unauthorized,
    /// A credential whose role may not do this (403).
    Forbidden,
    /// No such resource (404).
    NotFound,
    /// The object changed since the `resourceVersion` the write carried (409): read it again.
    Conflict,
    /// The write moves or clears the deadline of a drain whose notices went out (409).
    DrainUnderway,
    /// The request is malformed, or asks what cannot be (422).
    Invalid,
    /// The watch's `resourceVersion` is older than the changes kept, or of another term (410):
    /// list again.
    Gone,
    /// The request took longer than the coordinator allows (408).
    Timeout,
    /// The coordinator cannot answer now: it is not the holder, or the cluster's API server
    /// did not answer (503).
    Unavailable,
    /// The coordinator failed (500), or a node it asked did (502).
    Internal,
}

/// An error as the operator API answers it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ApiError {
    /// The HTTP status.
    pub code: u16,
    /// Why, for a client.
    pub reason: Reason,
    /// Why, for people.
    pub message: String,
}

impl ApiError {
    /// An error of `reason` with `message`, answered with the status `reason` maps to.
    pub fn new(reason: Reason, message: impl Into<String>) -> Self {
        let code = match reason {
            Reason::Unauthorized => StatusCode::UNAUTHORIZED,
            Reason::Forbidden => StatusCode::FORBIDDEN,
            Reason::NotFound => StatusCode::NOT_FOUND,
            Reason::Conflict | Reason::DrainUnderway => StatusCode::CONFLICT,
            Reason::Invalid => StatusCode::UNPROCESSABLE_ENTITY,
            Reason::Gone => StatusCode::GONE,
            Reason::Timeout => StatusCode::REQUEST_TIMEOUT,
            Reason::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Reason::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            code: code.as_u16(),
            reason,
            message: message.into(),
        }
    }

    /// A node the coordinator asked failed: 502, reason `Internal`.
    pub fn upstream(message: impl Into<String>) -> Self {
        Self {
            code: StatusCode::BAD_GATEWAY.as_u16(),
            reason: Reason::Internal,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}
