# The operator API's reads at 5,000 nodes: S2's first step

No pre-registration: these are diagnostic runs, taken while fixing what the S1 record showed (the
node list growing about 3x per doubling of the cell). Numbers: `results.json`. Same host and
harness as `../2026-10-14-cell-s1`. The harness now also times one refresh of the API feed with
nothing changed since the last refresh, and one API list of the cell's nodes.

| at 5,000 nodes | S1 (31789ed) | linear reads | + typed comparison |
|---|---|---|---|
| one feed refresh, nothing changed | 323 ms | 127 ms | 56 ms |
| one API list of 5,000 nodes | 153 ms | 58 ms | 27 ms |
| coordinator node list | 89 ms | 7 ms | 8 ms |
| coordinator readiness | 30 ms | 10 ms | 14 ms |

- **The feed could not keep its own pace.** It refreshes every 250 ms, and every read refreshes
  it first. At 323 ms a refresh took longer than its interval, so the feed ran continuously.
- **Linear reads.** The node list scanned every device in the cell for each node, and each
  lookup of an engine's policy scanned every engine. Devices are now grouped by node once, and
  the policy's engines are indexed by name.
- **Typed comparison.** The feed serialized every object to JSON on every refresh, to compare it
  with what it last showed. It now compares the objects as they are, and serializes only one
  that changed. Each engine's spec, which is the static policy, is serialized once.

## What is left of S2

At 56 ms per 250 ms, an idle 5,000-node cell spends about a fifth of a core on the feed. The rest
of S2 is to recompute only the objects the coordinator marked as changed, with a full sweep as a
safety net. A missed mark would show a stale object until the next sweep, so it needs a test that
compares the marked feed with a full recompute.
