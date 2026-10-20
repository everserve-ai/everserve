// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A frame read that is dropped mid-frame: `read_frame` loses the bytes it had, `FrameReader`
//! keeps them. The first is what a losing `select!` branch did to a daemon's coordinator link.

use std::time::Duration;

use everserve_wire::frame::{self, FrameReader, WireError};
use tokio::io::AsyncWriteExt;

/// A payload whose first four bytes read as a length far over `MAX_FRAME`: the protobuf bytes
/// a daemon once read as a length (`0a 13 08 01`).
const FIRST: &[u8] = &[
    0x0a, 0x13, 0x08, 0x01, 0x10, 0x02, 0x1a, 0x05, b'h', b'e', b'l', b'l', b'o',
];
const SECOND: &[u8] = b"second frame";

/// The length prefix of `payload`, on its own.
fn prefix(payload: &[u8]) -> [u8; 4] {
    u32::try_from(payload.len()).unwrap().to_be_bytes()
}

#[tokio::test]
async fn a_frame_reader_carries_on_after_a_dropped_read() {
    let (mut peer, ours) = tokio::io::duplex(1024);
    let mut rd = FrameReader::new(ours);
    // the prefix arrives, the payload does not yet; the read is dropped by a timeout, as a
    // losing select branch drops it
    peer.write_all(&prefix(FIRST)).await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(50), rd.next())
            .await
            .is_err(),
        "a read of a frame with no payload yet completed"
    );
    peer.write_all(FIRST).await.unwrap();
    frame::write_frame(&mut peer, SECOND).await.unwrap();
    assert_eq!(rd.next().await.unwrap(), FIRST);
    assert_eq!(rd.next().await.unwrap(), SECOND);
}

#[tokio::test]
async fn a_dropped_read_frame_takes_payload_for_a_length() {
    let (mut peer, mut ours) = tokio::io::duplex(1024);
    peer.write_all(&prefix(FIRST)).await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(50), frame::read_frame(&mut ours))
            .await
            .is_err()
    );
    peer.write_all(FIRST).await.unwrap();
    frame::write_frame(&mut peer, SECOND).await.unwrap();
    // the prefix went with the dropped future: the payload's first bytes are read as a length
    match frame::read_frame(&mut ours).await {
        Err(WireError::TooLarge(n)) => assert_eq!(n, 0x0a13_0801),
        other => panic!("expected the payload read as a length, got {other:?}"),
    }
}

#[tokio::test]
async fn a_frame_reader_splits_frames_that_arrive_together() {
    let (mut peer, ours) = tokio::io::duplex(1024);
    let mut rd = FrameReader::new(ours);
    let mut both = Vec::new();
    both.extend_from_slice(&prefix(FIRST));
    both.extend_from_slice(FIRST);
    both.extend_from_slice(&prefix(SECOND));
    both.extend_from_slice(SECOND);
    peer.write_all(&both).await.unwrap();
    assert_eq!(rd.next().await.unwrap(), FIRST);
    assert_eq!(rd.next().await.unwrap(), SECOND);
}

#[tokio::test]
async fn a_frame_reader_reports_a_close_inside_a_frame_as_eof() {
    let (mut peer, ours) = tokio::io::duplex(1024);
    let mut rd = FrameReader::new(ours);
    peer.write_all(&prefix(FIRST)).await.unwrap();
    peer.write_all(&FIRST[..3]).await.unwrap();
    drop(peer);
    match rd.next().await {
        Err(WireError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof),
        other => panic!("expected eof, got {other:?}"),
    }
}

#[tokio::test]
async fn a_frame_reader_refuses_a_length_over_the_limit() {
    let (mut peer, ours) = tokio::io::duplex(1024);
    let mut rd = FrameReader::new(ours);
    peer.write_all(&u32::MAX.to_be_bytes()).await.unwrap();
    assert!(matches!(rd.next().await, Err(WireError::TooLarge(_))));
}

/// A writer that counts its writes and keeps what was written.
struct Counting {
    writes: usize,
    bytes: Vec<u8>,
}

impl tokio::io::AsyncWrite for Counting {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        self.writes += 1;
        self.bytes.extend_from_slice(buf);
        std::task::Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn a_frame_is_one_write() {
    let mut w = Counting {
        writes: 0,
        bytes: Vec::new(),
    };
    frame::write_frame(&mut w, SECOND).await.unwrap();
    assert_eq!(w.writes, 1, "the length and the payload went out apart");
    let mut want = prefix(SECOND).to_vec();
    want.extend_from_slice(SECOND);
    assert_eq!(w.bytes, want);
}
