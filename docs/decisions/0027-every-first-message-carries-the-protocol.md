# 0027: every first message carries the protocol, and another protocol is refused by name

2026-10-08. Status: amended by 0040 (2026-10-13); current as amended: a build serves a range.

**Context.** Protobuf reads a field the receiver does not know as absent, and absent is zero.
Zero means something in several of our first messages: a heartbeat interval of zero is "never",
and the coordinator then takes a node as lost only when its link closes. A daemon from one build
and a coordinator from the next would link, exchange hellos, and run with node-loss detection
silently off. Nothing in the protocol said which build was talking. The review of the wire crate
against production transports (`docs/reviews/2026-10-08-wire-and-replication-against-references.md`)
listed it as the one handshake gap with a correctness consequence.

**Decision.** The protocol has a version, defined once, in `signals.proto`, as
`Protocol.PROTOCOL_VERSION`, and every language reads it from the generated code
(`everserve_wire::PROTOCOL_VERSION`, `everserve.signals.PROTOCOL_VERSION`). Every first message
carries it: a daemon's hello to the coordinator, a worker's or client's registration with its
daemon, a worker's open of its rank at the store, a source store's first request to a replica.
A receiver refuses a first message from another version at once, closing or answering with an
error that names both versions. The match is exact: one version exists. A compatible range is a
later change to the one comparison, when rolling upgrades need it.

**Consequences.**

- Two builds that differ in protocol do not talk at all, which is the failure that is visible and
  cheap. A refused daemon reconnects with backoff and logs the two versions each time; a refused
  worker is absent from readiness and its daemon's log says why.
- Changing the protocol is a one-line bump in `signals.proto` with the fields that changed, and
  every process built from the tree refuses every process that was not.
- The Python client and the Rust processes cannot drift apart, since both read the number from
  the generated code of the same file, and `scripts/gen-proto.sh --check` in the gate keeps the
  generated Python current.

**Amended** 2026-10-13 by 0040: the exact match became a range each build declares (the oldest
protocol it serves), and a link speaks the lower of the two versions, so upgrades roll.
