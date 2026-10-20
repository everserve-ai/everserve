# Changelog: everserve-dynamo

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versions are the
workspace's. The landing notes from before the first release are in the private archive the repository
was published from (`README.md`, History), not in this history.

## [Unreleased]

### Added
- The renderer (decision 0041): `python -m everserve_dynamo instance.json --out DIR` writes an
  instance's `DynamoGraphDeployment`, validated against the vendored schema of the pinned Dynamo,
  with the NetworkPolicies for the engine's unauthenticated transports, every engine pod as
  Dynamo's runtime user in the node's group, and the images pulled with the instance's secrets.
- The engine's state reported in Dynamo's terms; the head's loss stated as the known gap
  (decision 0045) with the single-instance case.

### Changed
- The package is Apache-2.0 (decision 0046); the manifest is the one version source.
