# 0013: the replay is exact: from the committed position, one step at a time, in the same shape

2026-10-06. Status: current.

**Context.** After every REPLACE in the kernels-order re-run one session in eight diverged from
the no-fault run at the same token (`bench/runs/2026-10-05-kernels-order-4gpu/`). The diagnostic
session (`bench/runs/2026-10-06-replace-divergence/NOTE.md`) found the cause in the replay, not in
the restore, the sum or the spare. The executor resumes the running requests by preempting and
re-adding them (`replay_running`), so the scheduler resumes each through the prefix cache, which
reuses whole blocks: with the committed position at 2,149 and a block size of 16, every request
restarts at 2,144. The first step back therefore recomputes six positions per request as a prefill
chunk, a shape no captured graph covers, so it runs eagerly; eager arithmetic differs from the
graph's in the last bits (a no-fault eager run diverges from a no-fault graph run in three sessions
with no fault at all); and the recomputed K and V for positions 2,144–2,148, which were computed,
committed and correct, are overwritten with the different values on every rank, the three that
never failed included. Every session's logits differ from the first token back; one session's top
choice flips. The same replay runs after a notice swap, where the tokens happened to match over the
window measured and the logits were never compared. The slow first step after any change (90–126
ms of GPU) is this eager step.

**Decision.**

1. *A committed position is never recomputed.* The replay starts at each request's committed
   count exactly, not at a block boundary. The requests stay running with their blocks; nothing is
   preempted, freed or re-added, so the prefix cache is not consulted.
2. *The tail is replayed one position per step, in decode shape.* The positions beyond the
   committed count, one or two with the one-step fence (decision 0009), are recomputed as one-token
   decode steps in their original order, the same shape as the steps they replace, so they run on
   the same captured graphs and reproduce the same arithmetic. A replay is never a prefill chunk.
3. *Exactness is tested where it was lost.* The REPLACE and notice drills assert, on every rank,
   that the K and V rows at positions below the committed count are byte-identical before and after
   the change; that the first step back schedules exactly the uncommitted positions and runs on a
   graph; and that logits, not only tokens, match the no-fault run from the first step back.

**Consequences.** The adapter's resume path changes from preempt-and-re-add to setting the running
requests' computed length and bounding their next chunks to one token; whatever private vLLM lines
that touches join `PRIVATE-INTERNALS.md`. `docs/CONTRACT.md` item 3 says "from the committed
length, exactly". The first step back costs what a decode step costs. The notice path gets the
same replay and the same assertion. An eager step anywhere in serving is a divergence from a graph
run, so the runtime's recapture window after a change (`placement/recapture.py`) is reviewed for
any step that could run eagerly on the survivors; today none should, since captured graphs survive
a change by design (decision 0005).
