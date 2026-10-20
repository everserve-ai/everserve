# Pre-registration: the frontier shape on the one-descriptor commit (decision 0019)

Status: PLANNED 2026-10-06, arms and predictions set by the owner. One 5× H100 SXM pod (secure
cloud, NVLink, `/dev/shm` large), Qwen2.5 72B Instruct at TP4, cap $30 for this battery. The run
log names the pod, its price and every arm's spend.

**Second attempt.** The first, on `a0b3378`, stopped when `adapter_1` failed forming the shadow
groups (`NOTE.md`); nothing was scored. The arms and predictions below are unchanged from it. Only
the tree moved, to take the formation fix and the slot-collective probe.

**Tree.** `5fd376d`, or a later commit whose diff from it touches only `bench/`, named in
`run.log` with `scripts/preflight.sh` and CI green. On top of the frontier run's tree
(`bench/runs/2026-10-06-frontier-1node/`, `37e4147`) it has:
- decision 0019: a step's commit is one descriptor; the fence runs two steps behind (`d1b5d1b`);
- decision 0020: a rank's tables are created and touched at registration, and a commit copies
  runs (`c63daa2`), with each registration's time in the store's log (`7f6b773`);
- the spare's catch-up on a notice: a full restore at the change, an incremental restore at the
  swap of what the leaving rank committed after it, run to completion before the spare answers
  the boundary, the leaving rank drained first (`96fa2ca`); a request resumes from its committed
  count capped at one below its tokens (`9460510`, the stall's fix,
  `bench/runs/PLANNED/catchup-stall-hunt/NOTE.md`);
- membership changes as sets of ranks (`56c5d41`);
- the shadow sets' device memory measured where they form (`474f2a4`);
- since the first attempt: the engine pre-forms a replacement set per launched spare (`5ed4a19`,
  the first attempt's failure; checked on 3× L40S in `bench/runs/PLANNED/shadow-formation-check/`);
  the slot-collective variant chosen by a probe after NVML, its verdict cached and reported
  (`4162939`, `798e2d0`, `c9aca25`, `c23c6eb`); store replication (`d704d0f`) and the notice
  source (`2550119`), neither used by these arms.

## Why this run exists

The frontier run kept every session exact but ran 42–48 % behind stock. The decode step was
stock's. The step thread waited on the mirror's one-step fence:
- a step's commit was 160 messages, one per table, and took 27–29 ms;
- the store span was 2–3 ms;
- the fence waits were 8 s per rank per run.

Decision 0019 made the commit one message and the fence two steps deep. In development that
brought the overhead to about 4 %, but fence waits stayed at 4 s per rank per run. Decision
0020 traces those to the prefill's first commit paying the page faults of fresh tables, and
moves that cost to registration. This battery measures both on the same shape.

## What runs

Qwen2.5 72B Instruct, TP4, 8 sessions × 2,048-token prompts, 200 decode steps. Every arm runs with
the determinism settings and records top-20 logprobs. In this order, nothing repeated:
1. stock ×1, custom all-reduce on;
2. adapter ×2;
3. adapter with the mirror off ×1;
4. REPLACE mid-step ×1 (rank 2 killed during the step after step 100);
5. the notice reference ×1 (no fault, 601 tokens): a reference for the notice's tokens, not scored;
6. notice ×1 (`--notice-steps 300`, 601 tokens).

**Recorded,** as the frontier run:
- `results.json`;
- `/metrics` from the daemon and the store after every arm;
- change seconds per cause and outcome;
- `commit.transport` and the timeline spans per rank;
- the gather and slot paths;
- every rank's event log in `flake/`;
- topology, versions, `/dev/shm`, spend.

Plus:
- the commit messages per step, as the transport counts them;
- the shadow sets' device memory per rank, measured before and after their formation;
- for the fault runs, decision 0013's checks and the spare's catch-up (rows copied against
  positions committed after the watermark; positions replayed).

## Predictions (kill conditions in bold)

Per decode step where a figure is per step. The decode window is steps 10–200.

1. **Adapter overhead** against stock, wall-clock tok/s over the decode window, each adapter run:
   under 3 %. **Kill: over 5 %.**
2. **Adapter against mirror-off:** within 2 points. **Kill: more than 4 points apart.**
3. **Store span:** `sender` p50 under 2 ms on every rank of every mirror-on run. **Kill: any over
   2 ms.** The band was raised from 1 ms by the owner before the run: with the fence two steps
   behind, the round trip is off the step thread's path (main measured 480.6 against stock's
   483.1 tok/s with `sender` at 1.7–2.0 ms).
4. **Fence wait:** `fence_wait` p50 under 1 ms on every rank of every mirror-on run. **Kill: any
   over 3 ms.** The p50 is over every step's record (about 200 per run); the prefill's commit is
   one of them and is reported on its own line as a one-time figure (the span's maximum and
   `fence_waits_s`), not scored.
5. **Notice pause**, the largest inter-token gap, with the swap: under 0.4 s. **Kill: no swap, or
   over 1.0 s.** Beside it, the catch-up check: the second restore copies rows only for positions
   committed after the watermark, it ends (synchronize included) before any rank's first step
   back, and the replay is at most two positions.
6. **REPLACE pause** as the frontier run held it: 1.5–2.5 s. **Kill: over 4 s.**
7. **Sessions continue:** 8/8 kept, tokens identical to the no-fault run's after the fault, resumed
   from the committed count with no re-prefill, in the REPLACE and the notice run. **Kill: any
   session lost, re-prefilled or mismatched.**
8. **Restored rows equal** the replaced rank's committed rows, every table. **Kill: any mismatch.**
9. **Commit before resume**, epochs matched. **Kill: a commit after resume.**

**Reported, not scored:**
- the shadow sets' device memory per rank, measured at formation;
- the commit messages per step;
- the timeline spans per rank;
- each rank's registration time from the store's log;
- logits: logprobs are not bit-reproducible between runs on H100 (decision 0019);
- the slot-collective variant and how it was chosen, per adapter-engine arm (NVLink: the probe is
  expected to confirm peer-to-peer on the first start and the cache to answer after it).

## What this run does not claim

Long contexts, multi-node, or the 16K restore (the incremental restore here applies to notices; a
fault's restore is unchanged).
