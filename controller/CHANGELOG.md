# Changelog: everserve-controller

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versions are the
workspace's. The landing notes from before the first release are in the private archive the repository
was published from (`README.md`, History), not in this history.

## [Unreleased]

### Added
- The coordinator: the one writer of every engine's epoch, under a lease (a file, etcd, or a
  Kubernetes Lease) with its term stamped on every message; a change claimed, logged in the
  spec store, delivered, acknowledged and committed, and finished by a successor after a
  takeover (decisions 0014, 0036, 0038's change log).
- The node daemon: the links to every process on its node, deliveries and commits, notices from
  the cloud's own signal (AWS, GCP, Azure, Kubernetes), the node's certificate issued by the
  coordinator from the pod's token (0037), standing notices carried in its hello.
- Spare pools, cover plans, the compile cache held per position, readiness, the engine's state
  in a framework's terms (0029, 0031, 0035, 0041, 0043).
- The operator API: every caller authenticated and authorized, by certificate or by the
  cluster's reviews, its document generated from the routes and tested against them
  (0039, 0042); token reviews bounded, tokens cached by digest.
- Under a Kubernetes lease the holder labels its own pod for the chart's Service, and every
  pod serves its health while it stands, so a rollout completes.

### Changed
- A daemon that relinks before its old link's reader times out is not lost; the daemon applies
  the coordinator's messages in the order they came; a notice's deadline is armed once and
  counts as in flight only when its change runs; the store-write deadline follows the lease.
- The crate is Apache-2.0 (decision 0046).
