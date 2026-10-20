# Changelog: everserve (core)

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versions are the
workspace's. The landing notes from before the first release are in the private archive the repository
was published from (`README.md`, History), not in this history.

## [Unreleased]

### Added
- The collective a model's traffic goes through, with replacement sets pre-formed beside the
  live group and the fabric family a pipeline spans; cover plans (decision 0035).
- The slot collective: a shared region of one row per member, summed in rank order, a dead
  member detected in the kernel (decisions 0006, 0012, 0021); its layout declared once in
  `csrc/kernels.h`, exported by the extension and checked before a region is mapped.
- The exchange between pipeline stages (`Exchange`): an object the adapter owns, its depth in
  flight the stage count, a wait on a dead peer's transfer ended at the refusal.
- Committed KV through the node's store over shared memory, the commit asynchronous behind a
  one-step fence, the step thread never reading the device (decisions 0009, 0010, 0011, 0019).
- The contract a process speaks to its node's daemon: registration with its identity (0034),
  changes, commits, notices, cover plans, placement; every first message carries the protocol
  (0027, 0040); `Participant` names every hook the link calls.
- The kernels' loader: a prebuilt stamped with what it was built against, loaded only by a
  process it matches; a named build that cannot be loaded is refused, never built or served
  around; the torch paths by choice alone (`EVERSERVE_KERNELS=torch`).

### Changed
- The reference tier (decision 0004: SHRINK on reference cores, the loopback store and
  transport, the placement plans) is `everserve.reference`, which no adapter imports; the
  product path is what `everserve.comm` and `everserve.state` hold.
- The package is Apache-2.0 (decision 0046); the manifest is the one version source.
