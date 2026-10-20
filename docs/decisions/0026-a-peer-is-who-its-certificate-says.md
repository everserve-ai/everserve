# 0026: a peer is who its certificate says

2026-10-08. Status: current, as amended below (2026-10-15); how a node on Kubernetes comes to hold
its certificate is 0037.

**Context.** Every control-plane link across nodes is mutual TLS against the deployment's CA,
and `everserve-controller certs` issues one certificate per node with the node's name in it.
But the name a daemon claimed in its hello was a string of its choosing: any process holding any
certificate from the CA could say "I am node-b", take node-b's ranks, receive its changes, launch
its spares and publish its record. Inside one operator's VPC that is a misconfiguration waiting
to happen more than an attack; it is also the first question a security review asks of a control
plane, and the review of the wire crate against production transports listed it.

**Decision.** An accepted connection carries who connected (`transport::Accepted`, with a
`Peer`): a local peer over a Unix socket, which is one of the node's own processes and is taken
as whoever it says, or a certified peer with the leaf certificate it presented. A certified peer
may call itself a name only when its certificate is valid for that name, checked by rustls-webpki
as it checks a server's name, not by parsing of our own. The coordinator refuses a daemon's hello
for a node the certificate is not valid for, closing with both names in its log, after the
protocol check of decision 0027 and before anything the hello says is believed.

The same rule binds the store-to-store link, the second half of this decision. The coordinator
decides who replicates what: when it tells a rank's holder where the rank replicates to, it tells
each target's daemon who the rank comes from (`ExpectReplication`, into the store as `Expect`),
and a store that stops being a target is told to expect nobody. A source store names its node in
its first request to a replica, the node its daemon gave it with the targets. The replica binds
that link to the one rank only when the source's certificate is valid for the node it names and
the coordinator said the rank comes from that node; rows for any other rank, or on a link that
bound nothing, are refused. A rank that moves takes its expectation with it in the same refresh
that moves its targets. The expectation carries the engine's epoch, and a store fences the rank
at it, never lowering it: a replicated step of an older epoch is refused from then on. A source
whose daemon died with its worker is told nothing on the move (the clear goes to the dead
daemon) and keeps draining its backlog; without the fence it wrote the replica the spare was
restoring from (dry run 1, 2026-10-08: one request diverging a few tokens after the resume, only
under a node fault).

*Amended 2026-10-15.* A peer store's link carries the replication protocol and nothing else: a
source's state query, its rows, its caught-up mark, and a compile cache's manifest and reads
(0043). The node's own requests (its workers' commits and restores, and its daemon's targets and
expectations) are refused on it. Until then the peer listener answered every request the
worker socket did, so a peer holding any certificate from the CA could send a store an
expectation naming a node of its choosing, and the store took that node's rows: the binding
above, undone by the request that sets it. The listener decides this, not the certificate,
because a peer over a Unix socket is a local peer too.

**Consequences.**

- A node's certificate is its identity on the fabric; the certs command issues it with the
  node's name, and an operator's own CA must do the same.
- A daemon started with another node's certificate never links, and the coordinator says why.
- The daemon's and the store's worker sockets are Unix and node-local; they take no identity, by
  construction rather than by check. A replica reached over a Unix socket (one node's tests)
  still needs the coordinator's word on who sends what; only the certificate check is moot.
- Any later check of who sent what (an operator's drain, a record's publisher) starts from the
  same `Peer`, one place.
