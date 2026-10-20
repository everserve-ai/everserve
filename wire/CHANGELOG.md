# Changelog: everserve-wire

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versions are the
workspace's. The landing notes from before the first release are in the private archive the repository
was published from (`README.md`, History), not in this history.

## [Unreleased]

### Added
- The schema types generated from `proto/` (`everserve.signals.v1`, `coord.v1`, `mirror.v1`,
  `policy.v1`, `spec.v1`, `discovery.v1`), with the protocol version and the oldest served
  (decisions 0027, 0040).
- The frame protocol: a length prefix, a 16 MiB cap checked before allocation, one write per
  frame, a reader safe to cancel mid-frame.
- The transports: a Unix socket on a node, mutual TLS across nodes with the peer's name verified
  by webpki, certificates and keys reloaded as a pair.
- Host facts: the cgroup's memory limit.

### Changed
- The crate is Apache-2.0 (decision 0046).
