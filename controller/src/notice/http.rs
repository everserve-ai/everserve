// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The notice sources' view of the controller's HTTP client (`crate::http`): the same requests,
//! with a failure named for the source, and the client a source's settings call for.

use super::{NoticeError, Settings, Source};
pub use crate::http::{Answer, REQUEST_TIMEOUT};

/// A client for one source.
#[derive(Clone)]
pub struct Client {
    inner: crate::http::Client,
    source: Source,
}

impl Client {
    /// A plain client for `source`.
    #[must_use]
    pub fn plain(source: Source) -> Self {
        Self {
            inner: crate::http::Client::plain(),
            source,
        }
    }

    /// The client `settings` call for: plain for the metadata endpoints; for Kubernetes, the
    /// shared access (`crate::kubernetes`): the service account's token, read again at every
    /// request, and for an `https` endpoint the cluster's CA.
    ///
    /// # Errors
    ///
    /// [`NoticeError::Endpoint`] when the CA file cannot be read.
    pub fn for_source(settings: &Settings) -> Result<Self, NoticeError> {
        if settings.source != Source::Kubernetes {
            return Ok(Self::plain(settings.source));
        }
        let access = settings.kubernetes.as_ref().ok_or(NoticeError::Endpoint {
            from: settings.source,
            reason: "no Kubernetes API access given".into(),
        })?;
        let inner = access.client().map_err(|e| NoticeError::Endpoint {
            from: settings.source,
            reason: e.to_string(),
        })?;
        Ok(Self {
            inner,
            source: settings.source,
        })
    }

    /// `GET url` with `headers`.
    ///
    /// # Errors
    ///
    /// [`NoticeError::Endpoint`] when the request fails or times out.
    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<Answer, NoticeError> {
        self.inner
            .get(url, headers)
            .await
            .map_err(|e| self.failed(&e))
    }

    /// `PUT url` with `headers` and no body.
    ///
    /// # Errors
    ///
    /// As [`Client::get`].
    pub async fn put(&self, url: &str, headers: &[(&str, &str)]) -> Result<Answer, NoticeError> {
        self.inner
            .put(url, headers)
            .await
            .map_err(|e| self.failed(&e))
    }

    fn failed(&self, e: &crate::http::HttpError) -> NoticeError {
        NoticeError::Endpoint {
            from: self.source,
            reason: e.to_string(),
        }
    }
}
