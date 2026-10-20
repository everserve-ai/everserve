# Changelog: everserve-store

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versions are the
workspace's. The landing notes from before the first release are in the private archive the repository
was published from (`README.md`, History), not in this history.

## [Unreleased]

### Added
- The mirror on every node: each rank's committed rows in shared memory, fenced by epoch, the
  socket carrying control alone (decision 0011); tables prepared and touched at registration
  (0020); every row's CRC32C recorded at write.
- Replication to the nodes that may restore a rank, over mutual TLS, with resync on connect,
  rows verified at apply, and a snapshot streamed a frame at a time (0016 as amended by 0030);
  a link's failures counted.
- The compile cache a spare's node fetches for its position (0043), verified against its
  manifest, a fetch's partial file removed whatever ends it.
- Start-up refusals: a budget the node's memory or shared memory cannot hold, a taken port.

### Changed
- An engine name is checked before any path is made from it; a row shape with no size or an
  overflowing one is refused; a step's new tables are made all or none; an accept that fails is
  tried again rather than ending the mirror.
- The crate is Apache-2.0 (decision 0046).
