// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Framing for every Everserve connection: a 4-byte big-endian length, then one protobuf message.
//! Identical to `core/everserve/wire.py`.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A frame larger than this is a protocol error, not a message.
pub const MAX_FRAME: usize = 16 * 1024 * 1024;

/// Why a frame could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    /// The socket failed or the peer closed it.
    #[error("connection: {0}")]
    Io(#[from] std::io::Error),
    /// The length prefix is not a frame length.
    #[error("frame of {0} bytes exceeds {MAX_FRAME}")]
    TooLarge(usize),
    /// The payload is not the expected message.
    #[error("decode: {0}")]
    Decode(#[from] prost::DecodeError),
}

/// Send one frame.
///
/// # Errors
///
/// `TooLarge` for a payload over `MAX_FRAME`; `Io` when the socket fails.
pub async fn write_frame<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    payload: &[u8],
) -> Result<(), WireError> {
    if payload.len() > MAX_FRAME {
        return Err(WireError::TooLarge(payload.len()));
    }
    let len = u32::try_from(payload.len()).map_err(|_| WireError::TooLarge(payload.len()))?;
    // one buffer, one write: a frame is one segment on TCP, and a request and its reply never
    // sit behind the peer's delayed acknowledgement (Nagle holds a length written apart from its
    // payload until the delayed-ACK timer fires, one timer per replicated step)
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&len.to_be_bytes());
    frame.extend_from_slice(payload);
    w.write_all(&frame).await?;
    w.flush().await?;
    Ok(())
}

/// Receive one frame's payload.
///
/// Not cancel-safe: the future holds the bytes it has read, so dropping it mid-frame (as a
/// losing `tokio::select!` branch or an expired `timeout` does) leaves the stream in the middle
/// of a frame, and the next read takes payload bytes as a length. Use it for a request and its
/// reply on a stream nothing else reads or waits on; a stream read in a `select!` is read through
/// a [`FrameReader`].
///
/// # Errors
///
/// `TooLarge` for a length prefix over `MAX_FRAME`; `Io` when the socket fails, with
/// `UnexpectedEof` when the peer closed.
pub async fn read_frame<R: AsyncReadExt + Unpin>(r: &mut R) -> Result<Vec<u8>, WireError> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let n = u32::from_be_bytes(len) as usize;
    if n > MAX_FRAME {
        return Err(WireError::TooLarge(n));
    }
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Send one protobuf message as a frame.
///
/// # Errors
///
/// As [`write_frame`].
pub async fn send<W: AsyncWriteExt + Unpin, M: prost::Message>(
    w: &mut W,
    msg: &M,
) -> Result<(), WireError> {
    write_frame(w, &msg.encode_to_vec()).await
}

/// Receive one protobuf message from a frame. Not cancel-safe, as [`read_frame`].
///
/// # Errors
///
/// As [`read_frame`], plus `Decode` when the payload is not the expected message.
pub async fn recv<R: AsyncReadExt + Unpin, M: prost::Message + Default>(
    r: &mut R,
) -> Result<M, WireError> {
    let buf = read_frame(r).await?;
    Ok(M::decode(buf.as_slice())?)
}

/// How many bytes one read takes from the stream.
const CHUNK: usize = 64 * 1024;

/// A frame reader whose reads can be dropped between polls without losing a byte.
///
/// Every byte read from the stream lands in the reader, not in the future, and each read is one
/// `AsyncReadExt::read`, which is cancel-safe. So [`FrameReader::next`] is cancel-safe: a losing
/// `tokio::select!` branch or an expired `timeout` drops the future and the next call carries on
/// from the same byte. A stream read in a `select!` is read through this and nothing else: a
/// frame it has buffered is not on the stream any more.
///
/// A daemon's link to the coordinator was read with [`read_frame`] in a `select!` against its
/// heartbeat tick and its outbound queue; a frame that arrived across either dropped the read
/// mid-frame, the next read took payload bytes as a length, and the daemon closed the link as
/// too large. The coordinator took that for a lost node.
pub struct FrameReader<R> {
    inner: R,
    buf: Vec<u8>,
    scratch: Vec<u8>,
}

impl<R: AsyncReadExt + Unpin> FrameReader<R> {
    /// Read frames from `inner`, which has no frame or part of one read from it yet.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            buf: Vec::new(),
            scratch: vec![0u8; CHUNK],
        }
    }

    /// The next frame's payload. Cancel-safe.
    ///
    /// # Errors
    ///
    /// As [`read_frame`]: `TooLarge` for a length prefix over `MAX_FRAME`; `Io` when the socket
    /// fails, with `UnexpectedEof` when the peer closed, between frames or inside one.
    pub async fn next(&mut self) -> Result<Vec<u8>, WireError> {
        loop {
            if let Some(frame) = self.take_frame()? {
                return Ok(frame);
            }
            let n = self.inner.read(&mut self.scratch).await?;
            if n == 0 {
                return Err(WireError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "peer closed",
                )));
            }
            self.buf.extend_from_slice(&self.scratch[..n]);
        }
    }

    /// The next protobuf message. Cancel-safe.
    ///
    /// # Errors
    ///
    /// As [`FrameReader::next`], plus `Decode` when the payload is not the expected message.
    pub async fn recv<M: prost::Message + Default>(&mut self) -> Result<M, WireError> {
        let buf = self.next().await?;
        Ok(M::decode(buf.as_slice())?)
    }

    /// One whole frame out of the buffer, or none yet.
    fn take_frame(&mut self) -> Result<Option<Vec<u8>>, WireError> {
        if self.buf.len() < 4 {
            return Ok(None);
        }
        let mut len = [0u8; 4];
        len.copy_from_slice(&self.buf[..4]);
        let n = u32::from_be_bytes(len) as usize;
        if n > MAX_FRAME {
            return Err(WireError::TooLarge(n));
        }
        if self.buf.len() < 4 + n {
            return Ok(None);
        }
        let frame = self.buf[4..4 + n].to_vec();
        self.buf.drain(..4 + n);
        Ok(Some(frame))
    }
}
