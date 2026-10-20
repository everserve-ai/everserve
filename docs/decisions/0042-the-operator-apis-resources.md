# 0042: the operator API's resources

2026-10-14. Status: current; built.

**Context.** Decision 0041 rules one interface for every caller: engines, nodes and pools, each a
resource with a spec and a status, listed, read and watched one way; a write names its resource in
its path; what an action asks for is spec, what came of it is status. Decision 0039 rules who may
ask. The API as built on 2026-10-14 predates both: reads of four different shapes (readiness,
topology, nodes, pools), drains and cordons as actions, `/v1/drain` naming its node in the body,
no watch, writes that overwrite each other, plain-text errors and no schema
(`docs/reviews/2026-10-14-control-plane-mvp-surface.md`, F1–F6). A pool's spare launcher, a
command every node's daemon runs, is granted with the pool's size (F8). A customer who drives
Everserve from their own scheduler, with or without Dynamo, needs the first form; the second is
what a due-diligence reader would find.

**How others do it.** The Kubernetes API: every object is `metadata` (name, `generation`, raised
by a spec write, and `resourceVersion`, raised by any write), `spec` and `status`; a list carries
the list's `resourceVersion`, and `?watch=true&resourceVersion=N` streams the changes after it as
`{"type": "ADDED" | "MODIFIED" | "DELETED", "object": ...}`, one per line, answering 410 Gone to a
version older than the server keeps, so the client lists again; a write carrying a stale
`resourceVersion` is 409 Conflict; an error is a `Status` object with a machine-readable `reason`;
a dangerous action is a subresource RBAC grants apart (`pods/exec`, `pods/eviction`). The schema is
published (OpenAPI) and generated from the types.

**Decision.**

Every route is under `/v1`, every object `{"metadata", "spec", "status"}`:

| route | verb (0039) | what |
|---|---|---|
| `GET /v1/engines`, `GET /v1/engines/{engine}` | `list`, `get` `engines` | spec: the engine's policy, read-only here (it is the policy's); status: what readiness says today, its conditions among it |
| `GET /v1/nodes`, `GET /v1/nodes/{node}` | `list`, `get` `nodes` | spec: `unschedulable`, `drain` (`{deadlineMs}` or absent), `reason`; status: linked, devices, each engine's ranks on it with the outcome expected and whether it can be reclaimed, conditions Draining and Degraded |
| `PUT /v1/nodes/{node}` | `update` `nodes` | write the spec: cordon, uncordon, drain (`spec.drain`), undrain (no `spec.drain`) |
| `GET /v1/pools`, `GET /v1/pools/{pool}` | `list`, `get` `pools` | spec: engines, wants, device class, domain, reason; status: standing, taking, warming, deficit, conditions |
| `PUT /v1/pools/{pool}` | `update` `pools` | write the spec; the launcher is kept as it is |
| `PUT /v1/pools/{pool}/launcher` | `update` `pools/launcher` | the command and environment a node's daemon runs for a spare; a grant of its own |
| `GET /v1/{kind}/{name}/events` | `get` `{kind}/events` | as today |
| `POST /v1/engines/{engine}/stacks` | `create` `engines/stacks` | a process's stacks, the engine named in the path so a grant can be scoped to it |
| `GET /v1/topology` | `get` `topology` | as today |

- **A drain is spec.** `spec.drain.deadlineMs` is a time, Unix milliseconds, never a lead: a
  spec says what is wanted, and the same spec written twice means the same thing. A write
  answers 200 with the node; what came of the drain is its status. The Draining condition's
  reason is `Scheduled` (its notices out, the change at the deadline), `Waiting` (other drains
  already move as many positions as the policy allows at once; the message says how many) or
  `Blocked` (its change was refused), and the engine's status keeps each node's outcome and
  whether it can be reclaimed. A write that
  moves the deadline of a drain whose notices went out is refused, 409 `DrainUnderway`, the held
  deadline unmoved; a write of the spec as it is changes nothing and answers 200.
- **Versions.** `metadata.generation` is the spec's, raised by each spec write as today.
  `metadata.resourceVersion` is one counter per coordinator term, raised by any change of any
  object's spec or status the API shows, and stamped on the object it changed. A list carries
  the version it was read at.
- **Pages** (amended 2026-10-14, after the synthetic cell's 5,000-node list). `?limit=N` answers
  at most N objects in name order and, when more remain, `metadata.continue`, a token for the next
  page (`?limit=N&continue=T`). Every page carries the first page's version, as Kubernetes'
  pages do. A later page reads the objects as they are then, so a watch from that version after
  the last page replays whatever changed while the pages were read. A page whose first version
  can no longer be watched from is 410 Gone, and the client lists again. A token the API did
  not give, or a watch with `limit` or `continue`, is 422. `everserve-controller get` reads a list
  500 objects at a time.
- **Watch.** `GET /v1/{kind}?watch=true&resourceVersion=N` streams each change after N as one JSON
  line, `{"type", "object"}`, `ADDED`, `MODIFIED` or `DELETED`, and a `BOOKMARK` with the current
  version every 30 s so an idle watch knows it is current. The coordinator keeps the last 4096
  changes; a version older than those, or from another term, is 410 Gone, and the client lists
  again. A watch ends with the coordinator's term.
  Changes are found by comparing each object as the API shows it with its last shown form, after
  every state change the coordinator makes and at least once a second, so a field added to an
  object is watched without a call site to remember.
- **Concurrency.** A write may carry `metadata.resourceVersion`; if the object changed since, the
  write is refused, 409, with the current object. A write without one applies as today.
- **Errors.** Every error is `{"code", "reason", "message"}`, `reason` one of `Unauthorized`,
  `Forbidden`, `NotFound`, `Conflict` (a stale `resourceVersion`), `DrainUnderway`, `Invalid`,
  `Gone`, `Timeout`, `Unavailable`, `Internal`.
- **Schema.** `docs/api/openapi.json` is generated from the routes and the request and object
  types by `utoipa` (MIT or Apache-2.0; the derive the axum ecosystem uses, a new dependency taken
  deliberately) and committed; a test fails when the generated document differs from it, so a
  route or a field changed in the code and not in the document fails the gate.
- **Leaves.** `/v1/readiness`, `/v1/nodes` as a list of linked daemons, `/v1/drain`,
  `/v1/nodes/{node}/cordon|uncordon|undrain`, `POST /v1/pools/{pool}` and `POST /v1/stacks` go in the
  same landing. No shim keeps them: the callers are this repository's (the harness, the smoke,
  `everserve-controller check`) and move with it.

**Consequences.**

- The chart's `everserve-operator` role grants `update` on `nodes` and `pools` and `create` on
  `engines/stacks`; `pools/launcher` is granted by an `everserve-launcher` role of its own.
- A drain is a node's spec: what it was asked for survives a coordinator's failover with the rest
  of the spec, and the node's Draining condition says what came of it.
- An integration (0041) and a customer's scheduler read the same objects and watch them the same
  way; Dynamo's integration reads the engine's object, never a route of its own.

**Built** (2026-10-14). `crate::api`: the objects (`objects`), the feed that computes them, numbers
their changes per term and keeps the last 4096 (`feed`), the routes declared once with their
schema (`routes`), and the one error shape (`error`); `everserve-controller openapi` and
`docs/api/openapi.json`. The coordinator's `put_node_spec` makes a node's spec what a writer
asks, at the generation it read when it names one, and a notice's drain is the same write; a
pool's launcher is written apart. The chart's `everserve-viewer`, `everserve-operator` and
`everserve-launcher` roles. Tested (`controller/tests/resources.rs`, `readiness.rs`, `api.rs`,
`chart.rs`): a watch from a list's version sees each change after it, at the version a read
shows, and a bookmark while idle; a watch from no version starts with every object and sees a
node that goes deleted; a version older than the changes kept, or of another term, is 410, and a
watch ends with its term; a write against a stale version is 409 `Conflict`, against the current
one 200; a moved deadline under way is 409 `DrainUnderway` and the same spec again is no write;
every error is one shape; the published document is the one the routes declare; every route the
document lists asks a verb; on a real API server, a watch is its own verb and update on pools
grants no launcher, and each chart role is granted exactly its routes.

Not built: the Kubernetes native forms 0041 names (a custom resource per kind mirroring these
objects, a disruption budget held to a node's release).
