# Commit-path session 2, 2026-10-06: REPLACE on main is clean; the catch-up still stalls a notice

One 5× H100 SXM pod (`q1u8bne5jeacvk`, secure, US, $17.45/h), Qwen2.5 72B Instruct at TP4,
`/dev/shm` 586 GB. Commands and timestamps in `run.log`; raw outputs, including the stalled run's
log with every worker's stack, in `data/` (not in git). Stopped on the owner's word before the
battery.

## Step 1: REPLACE mid on main as it is (`c63daa2`, no catch-up)

Completed. 8/8 sessions identical to the no-fault run, committed K/V rows unchanged on every rank,
first step back one token per request on the FULL graph. **The hang is in the catch-up wiring, not
decision 0019.** Development figures beside it, not verdicts:

| Run | tok/s | `sender` p50 per rank | `fence_waits_s` per rank | shadow sets at formation |
|---|---|---|---|---|
| `adapter_1` (no fault) | 480.6 | 1.66–2.04 ms | 0.73–1.12 s | 1.525 GB per rank |
| `replace_mid_1` | 470.1 | 1.22–1.79 ms | 0.00–0.70 s | (as above) |

The frontier pod's stock ran 483.1 tok/s. `sender` is above the drafted 1 ms band.

## Step 2: the catch-up, with the ordering rule stated (`catchup-wip2`, not pushed)

The ordering rule already held in the code; it is now stated where the spare joins and checked
by the drill:
- the incremental restore runs to completion, device synchronize included, before the spare
  answers READY;
- the executor sends no step before READY.

The notice drill stalled after the swap again:
- **Rank 0's step thread** was in vLLM's `copy_to_uva` (the sampler's staged writes, in
  `add_requests` inside `execute_model`), as rank 3's was last session.
- **Rank 0's mirror sender thread** was blocked reading the store's reply to a commit.
- **Ranks 1 and 3 and the spare** were waiting for their next RPC.
- **Every GPU was idle.**
- **The store was idle:** every thread asleep, its counters not moving, its last log line the
  registrations. So the request rank 0 waits on is not being worked on.
- The "spare still copying while survivors enter the collective" hypothesis does not fit: the
  catch-up had finished before READY. The mechanism is open.

What differs from step 1's clean path:
- the leaving rank drains its commits at the boundary while the executor waits for it;
- the spare's second restore runs since the watermark;
- the replay after the swap is about one position.

## Decision 0020's registration, from the store's new log line

`prepared engine=rig rank=N tables=160 ms=708–816` per serving rank, once, before serving; the
spare's preparation of rank 2's tables, already present, took 0.11 ms.

## Spend

Pod `q1u8bne5jeacvk`, 04:28:05Z–04:55:39Z (27.6 min) at $17.45/h: **$8.02** (cap $30).
