# Changelog: everserve-telemetry

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versions are the
workspace's. The landing notes from before the first release are in the private archive the repository
was published from (`README.md`, History), not in this history.

## [Unreleased]

### Added
- `Telemetry`: a registry, a health flag, `/metrics` and `/healthz` over axum, bound before
  `start` returns and served in the background until cancelled; a process's own routes served
  beside them.

### Changed
- The crate is Apache-2.0 (decision 0046).
