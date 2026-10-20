# 0009: the mirror commit is asynchronous with a one-step fence

2026-10-05. Status: amended by 0010 (2026-10-05) and 0019 (2026-10-06); current as amended: the
ordering stands, and the fence runs two steps behind.

**Context.** The adapter port committed synchronously inside the step guard: the step's rows left
the device after the forward and the agreement waited for the store. On the four-GPU run that was
16.7–17.9 ms per decode step, about one whole step, and 27.8 % overhead against a prediction of
4–6 % (`bench/runs/2026-10-05-adapter-port-4gpu/RESULT.md`). The week-five harness ran the same
mirror asynchronously at a few percent.

**Decision.** `CommittedKV.commit` queues the step (side-stream gather into pinned memory, a sender
thread, counts on the last frame) and returns. Before the agreement for step N the runtime calls
`fence(N-1)`, which waits for the store's acknowledgement of step N-1. A step the survivors accept
therefore has every rank's rows durable through the previous step; the current step's rows are in
flight and are replayed if lost. The store already returns a restored rank's own counts with its
rows, so a restore is exact to the fenced step.

**Consequences.** Overhead drops to the gather cost; on a fault the scheduler replays at most one
extra step. The pipeline holds two steps; a third commit waits for the oldest and the wait is
counted. A failed send poisons the transport and surfaces at the next fence, so a step whose rows
did not land is never agreed. Built in `core/everserve/state/mirror.py` (`MirrorCommittedKV`) and
called from the adapter's `Lifecycle.commit`. `docs/CONTRACT.md` item 3 states the fence. The pre-registration in
`bench/runs/PLANNED/` for the asynchronous commit predicts under 2 % overhead and is the test.

**Outcome 2026-10-05.** Built and run (`bench/runs/2026-10-05-async-commit-4gpu/`). The fence and
the pipeline behaved as designed: zero back-pressure, restores exact, commit within 36 ms of
resuming. The overhead did not move, because the cost was never the store's round trip: it was a
device read on the step thread inside `commit()`. The context paragraph above misattributed it.
Decision 0010 removes the read; this decision's fence and ordering stand unchanged on top of it.

**Outcome 2026-10-06, second.** The frontier run showed the tail of a commit (gather, event, round
trip) exceeding one step's lead on a dense 72B at TP4, so the fence moved to two steps behind and
the commit became one descriptor per step (decision 0019). The ordering this decision set, commit
first, fence before the agreement, counts with the rows, stands.
