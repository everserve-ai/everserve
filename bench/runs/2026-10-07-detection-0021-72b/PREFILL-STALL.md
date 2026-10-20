# The prefill stall is in the store round trip, not the gather

**Source.** The owner's brief asked for the span to be named before anything changes; nothing has
changed.
- The record: the first commit stamps of `adapter_1` in the 72B verification run (5× H100 SXM, US,
  tree `2182eb2`), recorded per rank as `first_ms_from_commit`.
- Raw outputs: `data/h100/out/adapter_1.json`, not in git.
- Times are in ms from each step's own commit, except the "commit at" column, which is from step 0's
  commit.

| Rank | Step | Commit at | Gather done | Sender starts | Store acks | Store round trip |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 34 | 34 | 822 | **788** |
| 0 | 1 | 11 | 811 | 811 | 1119 | **308** |
| 0 | 2 | 181 | 949 | 949 | 950 | 1 |
| 1 | 0 | 0 | 34 | 34 | 1128 | **1094** |
| 1 | 1 | 11 | 1117 | 1117 | 1119 | 2 |
| 1 | 2 | 180 | 950 | 950 | 956 | 6 |
| 2 | 0 | 0 | 34 | 34 | 534 | **500** |
| 2 | 1 | 11 | 524 | 524 | 1118 | **595** |
| 2 | 2 | 181 | 949 | 949 | 951 | 2 |
| 3 | 0 | 0 | 26 | 26 | 255 | **229** |
| 3 | 1 | 7 | 248 | 248 | 528 | **279** |
| 3 | 2 | 180 | 355 | 355 | 949 | **594** |

"Sender starts" is when the sender thread reached the step: it is later than the gather when the
sender was still busy with the step before.

**What it shows.**
- **The gather is not the stall.** Every rank's prefill rows were in the staging file 26–34 ms after
  the commit.
- **The store's round trip is.** The prefill's commits (steps 0 and 1, the two prefill chunks)
  took 229–1,094 ms from the sender to the store's acknowledgement. Decode commits take 1–6 ms.
- **The store answers the four ranks one at a time.** Step 0's acknowledgements arrive in a
  staircase about 290 ms apart: rank 3 at 255 ms, rank 2 at 534, rank 0 at 822, rank 1 at 1,128.
  - Rank 3's decode commit at step 2 waited 594 ms in the round trip, behind the other ranks'
    prefill commits.
  - Each prefill chunk is about 0.65 GB per rank: 8,192 tokens × 80 layers × 2 KV heads × 128 × 2
    (K and V) × 2 bytes.
- **So the fence stalls once per rank.** Fence waits peak at 271–948 ms, two steps behind, waiting
  for those acknowledgements.

**Why the 601-token runs seemed not to stall.** The transport kept only the last 512 commits'
stamps, so a 601-step run had already dropped its prefill. The stall was there and not recorded.
The transport now keeps the first 16 commits as well, however long the run.

**What decides the fix.** The cost is the store applying one rank's large commit at a time. The
fix belongs in the store: apply ranks' commits concurrently, or copy a commit with more than one
thread. The gather and the step thread are not involved. This note changes nothing; the fix needs
its own decision.
