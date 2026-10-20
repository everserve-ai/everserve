# Review: the control plane's surface, against a due-diligence reading

*A review note from 2026-10-14: a snapshot, kept for the record and not maintained. The current design is docs/ARCHITECTURE.md and docs/decisions/.*

2026-10-14. Asked for: a production-grade MVP surface, the Dynamo integration and a
standalone control-plane API. This review covers the control plane's half: the operator API, the
alerts, the store and the release. The Dynamo integration and the Python side are reviewed beside
it by their owner.

"Standalone" is read as decision 0041's "One interface" made true: one versioned API over
engines, nodes and pools that an operator, an autoscaler or a framework uses the same way, with or
without Dynamo. It is not a second framework integration; 0041 keeps standalone serving
possible and builds nothing for it.

References: the Kubernetes API conventions (spec and status, `metadata.generation`,
`resourceVersion`, list+watch, the `Status` error object, optimistic concurrency on writes);
kube-rbac-proxy (authorization of a delegated API); the Prometheus Operator's runbooks
(`runbook_url` on every alert); Sigstore's keyless signing (images signed by the workflow that
built them).

## The operator API

| Item | Reference | Ours, 2026-10-14 | Verdict |
|---|---|---|---|
| Authentication and authorization | TokenReview + SubjectAccessReview, or client certificates | Both (0039) | Right |
| Resources | Every kind listed and read by name, spec and status | Readiness, topology, nodes and pools are reads of different shapes; engines have no resource, readiness stands in | F1, built |
| Watch | One list+watch over a resource version | None: a caller polls | F2, built |
| Writes name their resource in the path | Yes, so RBAC scopes by `resourceNames` | Cordon, uncordon, undrain and pools do; `/v1/drain` names its node in the body | F3, built |
| Optimistic concurrency | A write carries the version it read; a stale write is 409 | Writes carry none: two operators overwrite each other | F4, built |
| Errors | One structured error object (code, reason, message) | Plain text bodies | F5, built |
| Schema | Published and checked against the server | None | F6, built |
| Connection limits | Header-read and idle timeouts | None on any HTTP listener (the operator API, the issuer on it, the metrics port): a client could hold a connection open indefinitely | F7, built |
| Operator tooling | A command line over the API, as `kubectl` over Kubernetes' | curl and JSON by hand | Built 2026-10-14: `everserve-controller get`, `watch`, `cordon`, `uncordon`, `drain`, `undrain` |
| Commands run on nodes | A distinct, separately granted privilege | A pool's `command` and `env` run on every node's daemon, granted with `update pools` | F8, built |

## Operations

| Item | Reference | Ours | Verdict |
|---|---|---|---|
| Alerts load and fire as written | `promtool check rules`, `promtool test rules` | Added 2026-10-14 (59e1cbd, on the held branch) | Right once landed |
| A runbook per alert | `runbook_url` on every alert, its page says what to do | No alert had one; RUNBOOK.md was a placeholder | F9, built |
| Store refusals alerted | Yes | Added 2026-10-14 (59e1cbd) | Right once landed |

## Release

| Item | Reference | Ours | Verdict |
|---|---|---|---|
| Images signed | Keyless, by the workflow that built them, verifiable by digest | Built and scanned in CI (SBOM); not published or signed | F10, built, not yet run on a tag |
| Chart published at a version | Packaged, versioned with the image | Used from the tree | F10, built, not yet run on a tag |

## Findings

F1 to F6 and F8 are built as decision 0042 (2026-10-14), its route table as designed below.

- **F1. The API's resources.** Engines, nodes and pools are each listed at `/v1/<kind>` and read
  at `/v1/<kind>/{name}`, as one object: `metadata` (name, generation, resourceVersion), `spec`
  (desired, written through the API) and `status` (observed, with conditions). Readiness and
  topology stay as the views they are.
- **F2. Watch.** `?watch=true&resourceVersion=N` streams the changes after N, one JSON object a
  line, as Kubernetes does; a version older than the server keeps answers 410, and the caller
  lists again.
- **F3. Drain as a node's spec.** A drain is written to `/v1/nodes/{node}` as `spec.drain`.
  `/v1/drain` goes.
- **F4. Optimistic concurrency.** A write may carry the `resourceVersion` it read; a stale one is
  409 Conflict with the current object's version.
- **F5. Errors.** Every error is `{"code", "reason", "message"}`, its reason one of a closed set.
- **F6. Schema.** An OpenAPI document generated from the types, committed, and checked against
  the server's routes by a test.
- **F7. Connection limits.** Built 2026-10-14: every HTTP listener closes a connection that sends
  no request headers within 10 s, a new connection's or a kept-alive one's next; the operator API
  answers 408 to a request not answered within 30 s, its body read included.
- **F8. Commands on nodes.** Writing a pool's `command` or `env` needs its own grant
  (`update pools/command`), so a pool's size can be delegated without delegating code execution
  on every GPU node.
- **F9. Runbooks.** Built 2026-10-14: `docs/RUNBOOK.md` says how to reach the API, move work off
  a node, take a hung process's stacks, and answer each alert; every alert carries a
  `runbook_url` to its section, and `controller/tests/dashboards.rs` fails an alert without one
  or a section that does not exist.
- **F10. Release.** Built 2026-10-14 (`.github/workflows/release.yml`), not yet run on a tag: a
  tag builds the image, pushes it, signs it by digest with the workflow's identity and attests
  its SBOM, and pushes the chart at the same version, signed; one version is checked across the
  tag, the workspace and the chart. The engine image (`deploy/images/engine`) has no release workflow
  yet: it is built by the harness on a rented VM, and its release is open, its owner's.
