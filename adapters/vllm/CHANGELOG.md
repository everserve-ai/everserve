# Changelog: everserve-vllm

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versions are the
workspace's. The landing notes from before the first release are in the private archive the repository
was published from (`README.md`, History), not in this history.

## [Unreleased]

### Added
- The vLLM plugin against vLLM 0.31.0 (`PIN`): an executor that treats a worker's death as an
  event and completes a membership change in the engine core's own step loop; the KV connector
  that commits each step's rows; the graph-safe all-reduce through the slot collective; the
  pipeline's transfers through the exchange.
- The spare, built ahead of a loss from the start-up calls the engine core sent its rank
  (decision 0044), standing for one rank or for any position, resident with every shard in
  host memory (0028, 0030); its join through the engine's own queues, declared JSON (0024).
- The node process: asks what its node's GPUs are for, starts the workers or the spare, leaves
  on notice.
- Every private vLLM line the adapter reaches, listed in `PRIVATE-INTERNALS.md` by file and
  function for the pin, with the upstream hook that would replace it; a test that fails when a
  reach is not listed.

### Changed
- Every permanent patch the adapter puts on vLLM or torch is one entry of `patches.py`, with its
  ledger row, its condition, an install, an uninstall and a listing; the pipeline methods are
  wrapped once with both behaviours, and the warm-up wraps compose by a marker, not a name.
- The plugin changes nothing in a vLLM process no Everserve launcher made an engine's
  (`EVERSERVE_ENGINE` unset): the stacks signal and NCCL's own error handling off are set only
  there.
- The identity believes a checkout's commit only when the checkout is this tree's own.
- The package is Apache-2.0 (decision 0046); the manifest is the one version source.
