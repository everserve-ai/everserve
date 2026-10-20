# Review: the wire crate and the store's replication, against references

*A review note from 2026-10-08: a snapshot, kept for the record and not maintained. The current design is docs/ARCHITECTURE.md and docs/decisions/.*

2026-10-08. Asked for after the three-node runs of 2026-10-07/08 found, one run at a time,
three defects in the hand-rolled transport (a cancel-unsafe frame read, a step sent as one frame
over the cap, a Nagle stall on length-then-payload writes) and three in replication (targets read
from one policy field, a drop-oldest queue that left holes, a source that kept sending after its
rank moved). Each was fixed the day it was found. This review asks what else those layers would
have learned from the code that already works, and closes it in one landing rather than per run.

References: tokio-util's `LengthDelimitedCodec` (framing, cancel safety, size limits); tonic's
transport defaults (TCP_NODELAY, connect timeout, HTTP/2 keepalive with timeout, backoff with
jitter, max message size); rustls (orderly close, client identity); the Raft follower contract
(leader tracks each follower's match index; snapshot install when behind) and Kafka's replica
fetch (the follower fetches from its own log end; the leader never guesses it).

## Transport

| Item | Reference | Ours, 2026-10-07 morning | Verdict |
|---|---|---|---|
| Framing | 4-byte length, cap on read and write | Same | Right |
| Cancel safety of a read | Bytes read live in the codec, not the future | Lost on a dropped read (daemon link, Nebius run 8) | Fixed 9584ce1 (`FrameReader`) |
| One segment per message | One write per frame | Length and payload as two writes | Fixed ded0bfc |
| TCP_NODELAY | Set | Not set: delayed-ACK stall, 280 ms per replicated step (run 12) | Fixed ded0bfc |
| Connect and handshake timeout | Yes | 10 s both sides | Right |
| Dead peer on an idle link | Keepalive ping with a timeout | Heartbeats on the daemon links (0023); nothing on the replication link: a half-open peer held the carrier forever | This landing: a reply timeout on the carrier. Socket keepalive wants `socket2`, a dependency decision |
| Reconnect backoff | Exponential with jitter | Exponential, in step | This landing: `transport::jittered` |
| Orderly close | TLS close_notify | Sockets dropped; the peer logs a lost connection | This landing: shutdown after the daemon's `Bye`, at the store's clean end, when a target is removed |
| Peer identity | Certificate bound to a name | Mutual TLS, but any certificate from the CA may say any node name in its hello | Follow-up, its own decision: the coordinator checks the hello's node against the certificate's SAN |
| Protocol version | Checked at the handshake | None: mixed builds decode each other's unknown fields as absent | Follow-up: a version in every hello, refused on mismatch, the Python client included |
| Max message size, write backpressure | Enforced, bounded | 16 MiB, queues bounded in bytes | Right |
| Unix socket permissions | Restricted | The process umask | Note: the node's sockets are shared by the node's own processes; revisit with multi-tenant nodes |

## Replication

| Item | Reference | Ours, 2026-10-07 morning | Verdict |
|---|---|---|---|
| Ordered, acknowledged | Every entry in order, acked | Same, parts pipelined | Right |
| Bounded, never dropped | A slow follower slows the leader, or refetches | Dropped the oldest step past 64, silently (run 11) | Fixed d228022: bytes, backpressure, no drop |
| Term fencing | A term in every entry, stale refused | An epoch in every step, stale refused | Right |
| Leader change | The old leader stops | The old holder kept draining (run 12) | Fixed fa49bdb |
| The leader knows the follower's position | Match index per follower; the follower says where it is | The source never asked. A target named after the engine ran, a replica or a source restarted, acknowledgements lost with a link: the replica was behind with counts a restore would trust | **This landing**: `ReplicaStateQuery` on every connection; unless the answer is where the carrier left it, the rank as held goes first |
| Snapshot install | When the follower is too far behind | None | This landing: `Storage::snapshot`, through the bounded queue, queued steps it covers skipped |
| Follower lag visible | Yes | Steps behind; the wait since d228022 | Right |
| Durability of the follower | Log on disk | Tables in memory; a restarted replica is empty | Covered by the resync; a replica on disk is a later choice |

## What this landing closes

- A replica is never silently behind: the carrier asks where it stands on every connection and
  fills from the source's tables when it is not where the carrier left it. Tests: a target named
  after the commits restores every row; a replica restarted empty restores every row.
- A reply that never comes is a failed link (`REPLY_TIMEOUT`), not a carrier held forever.
- Reconnects are jittered on both the daemon's and the carrier's links.
- Links end with a TLS close where the end is orderly.

## Follow-ups, each needing a decision line

1. Certificate-bound identity: the coordinator refuses a hello whose node name is not in the
   client certificate's SAN. The certs command already issues per-node certificates.
2. A protocol version in every hello (daemon, worker, store peer), refused on mismatch.
3. Settings now held as constants, into the policy with these as defaults: the release bound
   (10 s), the replication queue bound (the table budget), the reply timeout (30 s).
4. Socket keepalive on the TCP transport, which needs a dependency allowed through `deny`.

## What the review did not find

Nothing in either layer's architecture. The frame protocol is the right shape for both ends,
the two transports behind one stream type are right, and replication as one ordered, acknowledged
queue per target feeding the same apply path a commit takes is right. The gaps were the
operational contracts around those shapes, which the references hold and the first implementation
did not: what a read does when cancelled, what a segment is, who knows where the follower stands.
