# Attempt 7, read while the VMs were up

The budget's cap released the five VMs at 23:45:20 to 23:47:01 UTC, before the driver collected,
so the nodes' logs and the drill's result were not kept. What follows was read from the nodes over
ssh during the run, quoted as read; the driver's own reads are in `reads/` and `actions.jsonl`.

## The coordinator's log on node-a (`/workspace/out/coord.log`, colour codes removed)

```
23:26:12.236648 coord::transition rig: notice for rank 1 at 1791847592233 ms
23:26:12.236653 coord::nodes      rig: drain of node-b releases ranks [1] at 1791847592233 ms
23:26:13.856634 coord::nodes      node-c: its drain waits: 2 positions would move at once, the policy allows 1
23:26:32.435292 coord::transition rig: membership committed at epoch 1 (Replace, lost [1])
23:26:32.435404 coord::cover      rig: cover plan 6 for epoch 1: 2 groups
23:26:32.435512 coord::transition rig: notice for rank 2 at 1791847693853 ms
23:26:32.435518 coord::nodes      rig: drain of node-c releases ranks [2] at 1791847693853 ms
23:28:13.859670 coord::link       rig: ranks [0, 2] reported lost from node node-d: pipeline peers of rank 1: isend_tensor_dict: MemberSetChanged: step aborted: the member set changed: the pip...
23:28:14.425214 coord::transition rig: membership committed at epoch 2 (Replace, lost [2])
23:28:14.425302 coord::cover      rig: cover plan 7 for epoch 2: 1 groups
23:28:14.425449 coord::link       WARN rig: change for ranks [0, 2] failed: policy chose REBUILD for rank 0; the executor must rebuild
```

## The drill's result on node-a (`/workspace/out/drains.json`, 23:29)

- 16,000 steps over 188.7 s; the four sessions each 16,000 of 16,000 tokens.
- The longest gaps between steps: 0.58 s (after step 11,942), 0.21 s (after step 2,998), 0.15 s
  (after step 0), 0.03 s.
- Node-b's change: ranks [1], committed at epoch 1, applied by ranks 0 and 2 and by the spare
  for rank 1; rank 0's boundary moved to members `0-R1@node-d:0-2` in 0.0021 s, its cover sets
  released.

## Node-b's daemon log

39 lines, the last the relink to the restarted coordinator at 23:32:02. No notice line: the
daemon does not log the notices it forwards, so the driver's wait for one there (step 4) ran its
120 s out and the restart came at 23:32:20, after both commits, not inside node-b's lead.

## What this attempt shows

- Node-c's change, at epoch 2, committed: the spare outside the epoch-1 membership moved to the
  change's own epoch (`5f973a4`), which attempt 6 failed on.
- The sessions ran to their end through both changes, the longest gap 0.58 s.
- Rank 1's spare (node-d) reported its live pipeline peers 0 and 2 lost when node-c's change
  refused its send (MemberSetChanged). The coordinator refused the report's change; the engine
  kept serving. Fixed after the attempt: a refused transfer reports no peers.
- N4 was not exercised (the restart came after both commits) and C2's read came after node-c's
  commit, when epoch 2 had no set for rank 2 to read. Both are the driver's: fixed after the attempt.
- After the budget's release, the driver, edited on disk while it ran, re-entered the drains
  step at 23:48:53 against released VMs and failed there; nothing ran on a node.
