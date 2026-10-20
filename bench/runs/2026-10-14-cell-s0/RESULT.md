# S0 result: today's coordinator does not bring up a 2,500-node cell

Pre-registration: `PREREGISTRATION.md`. Numbers: `results.json`. Host: one Mac (14 cores,
36 GiB). The absolute times are this host's; the shape across sizes is the result.

## What happened

1. **The first run measured nothing.** The fake daemons gave up on their first refused connect.
   The macOS listen backlog is 128, and 1,250 nodes connecting at once overflow it. The harness
   now retries with the daemon's own backoff (`RECONNECT_MIN` 200 ms, doubling to
   `RECONNECT_MAX` 5 s, no jitter) and counts the refusals.
2. **At 1,250 nodes the coordinator deadlocked while linking.** Each link had a 32-message
   channel, and only that link's own task drained it. A link coming up sent every engine's
   policy, placement and state into that channel before it drained anything. Any policy naming
   33 or more engines blocked every link at its hello, at any node count. Fixed in 1f19f2e: an
   outbox per link, written by its own task. The suite's 40-engine cell (80 nodes) fails without
   the fix and passes with it.
3. **At 1,250 nodes (10,000 GPUs) on 1f19f2e**, the cell linked and committed all 20 changes:

   | | |
   |---|---|
   | daemons sent to, per change | 1,250 every time (the engine has 2) |
   | bytes received, per change | 110–115 KB |
   | report-to-commit, p50 / max | 16 ms / 19 ms |
   | link time | 123 s, 240 CPU-seconds, 2,744 refused connects |
   | resident memory once linked | 2.46 GB |
   | one readiness / one node list | 7 ms / 10 ms |

4. **At 2,500 nodes (20,000 GPUs) the cell did not come up in 10 minutes.** All 2,500 daemons
   linked, but only 72 of 1,250 engines had a whole view, after 15,182 refused connects.
5. **5,000 nodes was not run.** 2,500 did not link, and linking memory grows past this host.

## Against the predictions

1. **Held:** every change went to every daemon in the cell (S1).
2. **Not tested:** bytes per change across sizes needs a second size that links. At 1,250 nodes
   they are about 92 bytes per daemon in the cell.
3. **Not tested** for the same reason. At 1,250 nodes the change itself is cheap: 16 ms.
4. **Not tested** across sizes. At 1,250 nodes both reads are fast.
5. **Failed:** 2,500 did not link. That is the main finding.

## What it means

Linking is where the coordinator stops scaling, not the change. Each hello costs work in
proportion to the whole cell:

- it pushes that engine's placement and state to every linked daemon;
- `refresh_replication` walks every rank of every engine under the one engines lock;
- the new link receives every engine's policy, placement and state.

So bring-up costs about nodes squared. Memory follows it: each link's outbox holds every
engine's facts.

The link burst itself needs work too. The daemon's reconnect has no jitter, so after a
coordinator restart every daemon retries in step.

## What changes in the plan

S1 comes first, and it widens. An engine's facts go only to its audience: the nodes holding a
member, a spare, an assignment or a policy slot of it, and the nodes where a process of it
asked for its role. A hello refreshes only the engines it names. A new link gets only its
engines' facts. Reconnects get jitter. The harness then runs again at 1,250, 2,500 and 5,000
nodes, with these figures as the baseline.
