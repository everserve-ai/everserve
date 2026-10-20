# Review: the adapter port (commit 836416b)

*A review note from 2026-10-04: a snapshot, kept for the record and not maintained. The current design is docs/ARCHITECTURE.md and docs/decisions/.*

Reviewed 2026-10-04 against `docs/REVIEW-CHECKLIST.md`, before its GPU run. Verdict: accepted,
with the follow-ups below. None of them changes what the run measures.

## What the landing does well

- The daemon acts on a reported loss, forwards a notice to every worker and the spare and drives
  the same change at the deadline while the rank is alive, checks the acknowledged epoch, hands
  the spare the dead rank's place on commit, and tells every client. Three roadmap items, done in
  the daemon's own shape.
- The store keeps committed counts per rank and returns the restored rank's own counts with its
  rows: a restored rank is complete through exactly those counts, whatever other ranks committed
  since. The mirror client makes one device copy per step and splits commits under the frame
  limit with the counts on the last frame, so a count is never ahead of its rows.
- `Runtime.at_boundary` states plainly that vLLM's step loop cannot be re-run from outside and
  applies the change between steps; the scheduler replays the tail from the committed counts.
- `PRIVATE-INTERNALS.md` lists every private line with its pin location and names the one
  upstream hook the port surfaced (the engine core calling the executor between steps,
  20–40 lines in `v1/engine/core.py`). The drill-only restore check and the kill live in
  `bench/harness/`, not the product tree.

## Follow-ups

| # | finding | checklist rule | action |
|---|---|---|---|
| 1 | the commit message carries a `Co-Authored-By` trailer | plain messages, no trailers | noted for later commits; history is not rewritten |
| 2 | `executor.py` and `collective_op.py` read `EVERSERVE_*` variables | library code reads no environment | vLLM constructs these objects; move the settings to the engine's `additional_config`, read the environment only in the launcher under `bench/harness/` |
| 3 | process singletons: `worker.STATE`, `collective_op._COLLECTIVE`, `connector.WORKER` | no module-level runtime state | sanctioned exception, now written into the checklist: a torch custom op and a vLLM-instantiated connector can only find our objects through a per-process singleton set once at bootstrap |
| 4 | `sys._getframe(1).f_locals["self"]` in `worker.boundary` and the spare | no frame inspection | listed as A19 with its removing hook (an RPC target that receives the `WorkerProc`, or a ring-attach API); sanctioned while listed |
| 5 | `hostslot.py` is the harness kernel carried unchanged, compiled through CuPy at run time | nothing copied verbatim from the harness | decision 0006 already moves it to `core/csrc` with the peer-to-peer variant; that is the next kernel work and is no longer optional |

## Not changed by this review

The pre-registration and its predictions. The run proceeds on 836416b.
