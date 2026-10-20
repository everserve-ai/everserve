// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The one wire schema, generated from `proto/` at build time, and the frame protocol every
//! Everserve process speaks: a 4-byte big-endian length, then one protobuf message. Identical to
//! `core/everserve/wire.py`.

pub mod frame;
#[cfg(unix)]
pub mod host;
pub mod proto;
pub mod transport;

/// The protocol this build speaks, from the one definition in `signals.proto`; every first
/// message carries it (decision 0027).
pub const PROTOCOL_VERSION: u32 = proto::everserve::signals::v1::Protocol::Version as u32;

/// The oldest protocol this build serves correctly (decision 0040); every first message carries
/// it beside [`PROTOCOL_VERSION`].
pub const PROTOCOL_OLDEST: u32 =
    proto::everserve::signals::v1::ProtocolOldest::ProtocolOldest as u32;

/// A first message from a build outside the range this one serves, or that does not serve this
/// one (decision 0040): both builds' ranges, for the one refusal every receiver gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "protocol {sent} (serving from {sent_oldest}) refused: this build speaks protocol \
     {PROTOCOL_VERSION} and serves from {PROTOCOL_OLDEST}"
)]
pub struct ProtocolMismatch {
    /// What the sender's first message carried.
    pub sent: u32,
    /// The oldest protocol the sender serves.
    pub sent_oldest: u32,
}

/// Accept a first message's protocol and the oldest its sender serves, or say why not: the
/// version the link speaks, the lower of the two builds'. Each build must serve the other's: the
/// sender's version is at least this build's oldest, and this build's version at least the
/// sender's oldest. A sender that names no oldest (0, before protocol 10) serves its own version
/// alone. The one comparison in the codebase.
///
/// # Errors
///
/// [`ProtocolMismatch`] when either build does not serve the other.
pub fn check_protocol(sent: u32, sent_oldest: u32) -> Result<u32, ProtocolMismatch> {
    let sent_oldest = if sent_oldest == 0 { sent } else { sent_oldest };
    if sent >= PROTOCOL_OLDEST && PROTOCOL_VERSION >= sent_oldest && sent_oldest <= sent {
        Ok(sent.min(PROTOCOL_VERSION))
    } else {
        Err(ProtocolMismatch { sent, sent_oldest })
    }
}

#[cfg(test)]
mod tests {
    use super::{PROTOCOL_OLDEST, PROTOCOL_VERSION, check_protocol};

    #[test]
    fn a_link_opens_only_when_each_build_serves_the_other() {
        let (v, o) = (PROTOCOL_VERSION, PROTOCOL_OLDEST);
        assert_eq!(check_protocol(v, o), Ok(v), "the same build");
        assert_eq!(
            check_protocol(v, 0),
            Ok(v),
            "naming no oldest: its own version alone"
        );
        // a newer build that serves this one: the link speaks this build's version
        assert_eq!(check_protocol(v + 1, v), Ok(v));
        assert!(
            check_protocol(v + 1, v + 1).is_err(),
            "a newer build that does not serve this one"
        );
        assert!(
            check_protocol(o - 1, o - 1).is_err(),
            "older than this build serves"
        );
        assert!(
            check_protocol(v, v + 1).is_err(),
            "an oldest above its own version is nonsense"
        );
    }
}
