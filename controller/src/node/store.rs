// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The daemon's client to its node's store: one request, one reply, over the mirror's Unix
//! socket, for the control requests the coordinator routes through the daemon (`Replicate`).
//! Each is answered within a deadline: the daemon awaits it in its coordinator link's loop, so a
//! store that took the request and never answered would stop the link with it.

use std::path::Path;
use std::time::Duration;

use everserve_wire::frame as wire;
use everserve_wire::proto::everserve::mirror::v1::{MirrorReply, MirrorRequest, mirror_request};
use tokio::net::UnixStream;

/// Why a store request failed.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The socket could not be reached or the frame failed.
    #[error("store: {0}")]
    Wire(#[from] wire::WireError),
    /// The store refused the request.
    #[error("store refused: {0}")]
    Refused(String),
    /// The store did not answer in time.
    #[error("store: no answer within {0:?}")]
    Timeout(Duration),
}

/// How long a control request (`Replicate`, `Expect`) may wait for the store's answer: the store
/// answers one at once, so longer is a store that is stuck, not busy.
pub const CONTROL_REPLY: Duration = Duration::from_secs(10);

/// Send one request to the store at `socket` and return its reply, answered within `within`.
///
/// # Errors
///
/// [`StoreError`] when the store is unreachable, refuses, or does not answer in time.
pub async fn request(
    socket: &Path,
    req: MirrorRequest,
    within: Duration,
) -> Result<MirrorReply, StoreError> {
    let asked = async {
        let mut stream = UnixStream::connect(socket)
            .await
            .map_err(|e| StoreError::Wire(wire::WireError::Io(e)))?;
        wire::send(&mut stream, &req).await?;
        let reply: MirrorReply = wire::recv(&mut stream).await?;
        Ok::<_, StoreError>(reply)
    };
    // the connection is the request's alone: one cut short is dropped, not read again
    let reply: MirrorReply = tokio::time::timeout(within, asked)
        .await
        .map_err(|_| StoreError::Timeout(within))??;
    if reply.error.is_empty() {
        Ok(reply)
    } else {
        Err(StoreError::Refused(reply.error))
    }
}

/// Tell the store to replicate `rank` of `engine` to `targets`, naming itself `node` to them
/// (decision 0026), answered within [`CONTROL_REPLY`].
///
/// # Errors
///
/// As [`request`].
pub async fn replicate(
    socket: &Path,
    node: &str,
    engine: &str,
    rank: u32,
    targets: &[String],
) -> Result<(), StoreError> {
    request(
        socket,
        MirrorRequest {
            kind: Some(mirror_request::Kind::Replicate(
                everserve_wire::proto::everserve::mirror::v1::Replicate {
                    engine: engine.to_owned(),
                    rank,
                    targets: targets.to_vec(),
                    node: node.to_owned(),
                },
            )),
        },
        CONTROL_REPLY,
    )
    .await
    .map(|_| ())
}

/// Tell the store that `rank` of `engine` replicates into it from `from_node` and nobody else;
/// an empty `from_node` expects nobody (decision 0026).
///
/// # Errors
///
/// As [`request`].
pub async fn expect(
    socket: &Path,
    engine: &str,
    rank: u32,
    from_node: &str,
    epoch: u64,
) -> Result<(), StoreError> {
    request(
        socket,
        MirrorRequest {
            kind: Some(mirror_request::Kind::Expect(
                everserve_wire::proto::everserve::mirror::v1::Expect {
                    engine: engine.to_owned(),
                    rank,
                    from_node: from_node.to_owned(),
                    epoch,
                },
            )),
        },
        CONTROL_REPLY,
    )
    .await
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use everserve_wire::proto::everserve::mirror::v1::{Expect, MirrorRequest, mirror_request};

    use super::{StoreError, request};

    #[tokio::test]
    async fn a_store_that_takes_a_request_and_never_answers_is_given_up_on() {
        // the daemon awaits its store in its coordinator link's loop: a stuck store must not stop
        // the link with it
        let dir = tempfile::tempdir().expect("tmp");
        let socket = dir.path().join("stuck.sock");
        let listener = tokio::net::UnixListener::bind(&socket).expect("bind");
        let held = tokio::spawn(async move {
            let (conn, _) = listener.accept().await.expect("accept");
            // reads nothing, answers nothing, and keeps the connection open
            tokio::time::sleep(Duration::from_mins(1)).await;
            drop(conn);
        });
        let ask = MirrorRequest {
            kind: Some(mirror_request::Kind::Expect(Expect::default())),
        };
        let asked = tokio::time::timeout(
            Duration::from_secs(5),
            request(&socket, ask, Duration::from_millis(300)),
        )
        .await
        .expect("given up within its deadline");
        assert!(matches!(asked, Err(StoreError::Timeout(_))), "{asked:?}");
        held.abort();
    }
}
