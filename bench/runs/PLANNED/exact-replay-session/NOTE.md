# Exact replay session, 2026-10-06: what landed and what it cost

The build and drills for decision 0013, then the battery on the same pod
(`bench/runs/2026-10-06-exact-replay-4gpu/`). The owner capped the whole session at $8.

## What landed

| Commit | What |
|---|---|
| `d740223` | the exact replay: running requests stay running with their blocks, computed length set to the committed count, every rank re-adds them from the same state, the tail one position per step (`OneTokenReplay`); private vLLM lines A11, A11a, A11b, A20a in `PRIVATE-INTERNALS.md` |
| `eb67ad3` | the REPLACE and notice drills' three exactness checks, logprobs in every arm |

## The drills (development runs on the pod, before the battery)

One REPLACE (between) and one notice, each with its no-fault reference: all three checks passed
in both. 8/8 sessions identical in tokens and every logprob; first step back one token per
request from exactly 2,149 (REPLACE) and 2,449 (notice), FULL graph on every rank, nothing
re-added; committed K and V rows unchanged on all four ranks. Raw outputs in `data/` (not in git).

## Spend

Pod `heippwhi15marw`, 5× A100 SXM, $7.95/h, 23:35:15Z–00:16:45Z (41.5 min): **$5.50** in all, of
which the battery $2.92 and setup and drills $2.58. Cap $8.
