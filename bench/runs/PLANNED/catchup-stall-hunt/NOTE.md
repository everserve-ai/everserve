# The catch-up stall, 2026-10-06: found, fixed, proven

## Local, free

The store traffic of a notice with the catch-up, in the drill's order, each rank on its own
connection and thread, against the real store binary
(`core/tests/test_mirror_client.py::test_the_notice_catch_up_sequence_answers_every_request`).
Every request was answered, and the catch-up copied exactly the rows committed after the
watermark. So the store does not lose requests in this sequence.

## One cheap run (5× A100 SXM, community, $6.95/h)

5× A40 had no supply for 10 minutes. Instead: 5× A100 SXM, the NVLink and peer-to-peer slot path
the H100 stall ran on, with the 30B MoE.
- The store ran at debug level (`EVERSERVE_LOG=debug`).
- The drill ran under `bench/harness/stallwatch.sh`. It is the drill's launcher, so py-spy can
  read the ranks: the container's ptrace scope is 1.

**What it showed.** The run did not hang; it livelocked. The store's debug log has every rank,
the spare included, committing empty steps at epoch 1 from the swap on, about 1,800 a second
(step 405 to 72,027 in 38 s, `rows 0`). The engine was stepping with nothing scheduled.
- That explains the H100 picture: GPUs idle, ranks running, the step thread always found in the
  sampler's per-step bookkeeping, the store's commit counter in the hundreds of thousands.

**The cause.**
- The catch-up's counts are the leaving rank's after its drain.
- Under async scheduling the leaving rank had already run the step for the last sampled token
  before the engine received that token. So its committed count (2,484) equalled the engine's
  token count.
- The resume set computed = committed, which left nothing to compute. No step could produce the
  next token.
- A dead rank's count lags the tokens, so REPLACE never hit it. The old notice path restored many
  steps before the swap, so it never hit it either.

**The fix (`9460510`).** A request resumes from `min(committed, tokens - 1)`, so its last
position is always recomputed, a one-token decode step on the captured graph. The resume moved
to `everserve_vllm/resume.py`, testable without vLLM, and is pinned by `tests/test_resume.py`.

**The proof, on the same pod.** One notice run with the catch-up completed:
- the catch-up copied 13,824 rows for 288 positions committed after the watermark: 48 per
  position, as the full restore's 940,416 for 19,592;
- it ended before any rank's first step back;
- the replay was one position;
- K/V and first-step checks pass;
- pause 0.304 s;
- all 8 sessions are identical to `notice_ref`'s, tokens and logprobs.

**On the side.**
- The watchdog's first start fired during engine start-up: `grep -c` printed 0 and exited
  non-zero, so `|| echo 0` made the count "0\n0". Fixed before the real run.
- The worker-side mirror debug lines did not reach the drill log. The store's log was enough.

## Spend

Pod `jsrppu4glnkl3k`, 10:42:29Z–11:03:24Z (20.9 min) at $6.95/h: **$2.42** (cap $5).
