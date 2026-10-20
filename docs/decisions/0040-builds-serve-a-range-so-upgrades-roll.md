# 0040: builds serve a range of protocols, so upgrades roll

2026-10-13. Status: current; built.

**Context.** Decision 0027 made every first message carry the protocol and refused any other
version by exact match. That is safe, and it makes every upgrade a stop of the whole control
plane and every engine on it, since a coordinator, its daemons, their stores and the engines'
processes must all move at once. A neocloud upgrades a fleet node by node, without stopping it;
a due-diligence review asks how. 0027 left the place for the answer: a compatible range is a
change to its one comparison.

**How others do it.** Kubernetes states a version skew policy: the API server first, kubelets up
to three minors older, each component serving the range the policy names. CockroachDB runs
mixed-version clusters through an upgrade and gates a new feature on the cluster's version.
Both rest on the same discipline: a newer build declares what older one it still serves, and
keeps any behaviour the older one cannot follow behind the version both speak.

**Decision.**

- Each build declares two numbers in `signals.proto`: the protocol it speaks (`PROTOCOL_VERSION`)
  and the oldest it serves correctly (`ProtocolOldest.PROTOCOL_OLDEST`). Every first message
  carries both (`protocol`, `protocol_oldest`): a daemon's hello, a worker's registration and its
  open at the store, a source store's first request to a replica.
- A link opens when each build serves the other: the sender's version is at least the
  receiver's oldest, and the receiver's version at least the sender's oldest. It speaks the
  lower version. Otherwise the receiver refuses, naming both ranges, as before. A sender that
  names no oldest (zero: a build before protocol 10) serves its own version alone, so a
  protocol 9 build and a protocol 10 build do not talk; protocol 10 is the last lockstep upgrade.
- Raising `PROTOCOL_VERSION` without raising `PROTOCOL_OLDEST` is a promise the landing makes and
  its review checks: every field added since the oldest reads, absent, as what that build did,
  and anything the older build cannot follow is gated on the link's negotiated version. A change
  that cannot keep the promise raises the oldest with the version, and says so.
- The upgrade order: the coordinator and its standby first (a coordinator serves the daemons it
  had), then each node's daemon and store, node by node, then each engine's processes, which move
  onto the new build by planned drains onto spares, a rank at a time, as any planned change does.

**Consequences.**

- From protocol 10 on, a release serves the one before it, unless it says otherwise; an upgrade
  rolls.
- A link from a newer build that does not serve an older one is refused at once, with both
  ranges in the log, as 0027's refusal was.
- From the first release on, CI runs the bare-metal smoke with the previous release's daemons and
  stores against the new coordinator, so the promise is tested, not only reviewed.
