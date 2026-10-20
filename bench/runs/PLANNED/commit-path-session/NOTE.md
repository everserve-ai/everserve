# Commit-path session, 2026-10-06: the catch-up hangs a notice; the battery was not run

One 5× H100 SXM pod (`44wipvnh0u72xc`, secure, CA, $17.45/h), Qwen2.5 72B Instruct at TP4,
`/dev/shm` 293 GB. Commands and timestamps in `run.log`; raw outputs in `data/` (not in git).

## What happened

1. **Determinism:** two stock runs gave identical tokens in all 8 sessions.
2. **Landing 1, the spare's catch-up (`fff6d85`, kept on the local branch `catchup-wip`, not
   pushed).** A notice run hung twice after the swap.
   - Rank 3 was asleep on its first step back, in vLLM's `copy_to_uva` (the sampler's staged
     writes, from `model_runner.add_requests` inside `execute_model`).
   - Ranks 0 and 1 and the spare waited for their next RPC; every GPU was idle.
   - `copy_to_uva` is a CPU copy into pinned memory, so the thread was probably waiting for the
     GIL; two of rank 3's threads showed no Python frames. Not proved.
3. **The discriminating run.** The same notice drill on `d1b5d1b`'s product code, without the
   catch-up, completed. So the hang needs landing 1's path.
   - One difference it makes: with the catch-up the replay after the swap is about one position,
     so the one-token cap never engages. The first step back then has the same shape as a
     REPLACE's.
   - No REPLACE has run on `d1b5d1b` yet. A REPLACE drill on that tree is the next test: it
     separates "the catch-up" from "a one-position replay under fence depth 3".
4. **Landing 2, the shadow-set memory measured at formation:** works. It is on main with the
   catch-up checks (`bfba0b1`, `474f2a4`).
5. **The battery (`bench/runs/2026-10-06-commit-path-72b/`) was not run.**
   - Its notice prediction needs landing 1.
   - The development figures below put decision 0019 alone past the fence-waits kill on this
     shape.

## What decision 0019 does on this shape (development runs, `d1b5d1b`, not verdicts)

| Figure | No-fault reference (601 tokens) | Band / kill in the drafted pre-registration |
|---|---|---|
| tok/s | 463.0 (the frontier pod's stock: 483.1) | overhead under 3 % / over 5 % |
| `fence_waits_s` per rank | 4.1–4.2 s | under 0.5 / over 2 s |
| `sender` p50 per rank | 0.94–1.13 ms | under 1 / over 2 ms |
| commit messages per step | 1.0 | one |
| shadow sets, measured at formation | 1.53 GB per rank | under 1 / over 2 GB |

The frontier run's 42–48 % is now about 4 %, and a step's commit is one message. But the step
thread still waits about 4 s per run with the fence two steps deep. The commit's tail still exceeds
two steps somewhere, and these figures do not locate it.

## Spend

Pod `44wipvnh0u72xc`, 03:19:35Z–04:00:06Z (40.5 min) at $17.45/h: **$11.78** (cap $30).
