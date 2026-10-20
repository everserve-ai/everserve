# Result: detection at the collective (decision 0021) on the 72B shape

**What ran.** Two H100 attempts and five cheap checks; `run.log` has every command and time.

- **First H100 attempt** (`yq3jxlut1ojcgh`, 5× H100 SXM, US, driver 580; tree `2182eb2`):
  - `stock_1` and `adapter_1` completed.
  - `replace_mid_1` was detected by the slot collective on all three survivors, then failed. A
    survivor's report reached the daemon after it had sent a `Change`, and was read as that
    worker's `Ack`. The link broke and the change failed.
- **The fix:** every worker frame is a `Signal`, the `Ack` included (`fbc6efc`). It was proved on
  5× A6000 with three survivors.
- **The rerun** (`1xbucv0yh1ifv7`, 5× H100 SXM, IN; tree `c61af38` = `fbc6efc` plus test and
  comment changes): `adapter_1` and `replace_mid_1`. The owner approved going past the $10 cap
  for it.
- No file under `vllm/` was edited.

## Verdict per prediction

| # | Prediction (kill) | Measured | Verdict |
|---|---|---|---|
| 1 | REPLACE pause 0.6–1.2 s (> 2 s) | **1.250 s** | **weakened**: above the band, below the kill |
| 2 | detected by the slot collective, under 0.15 s (by the sentinel, or > 0.5 s) | **by the sentinel, 0.364 s** | **dead** |
| 3 | no report in the adapter run | none in either H100 adapter run; live ranks late 0.16–0.72 s at the prefill recorded as slow, not reported | **alive** |
| 4 | overhead < 3 % (> 5 %) | 0.97 % (486.4 against 481.7 tok/s, first attempt's pod; the rerun has no stock) | **alive** |
| 5 | sessions exact, rows equal, commit before resume | 8/8 identical, no re-prefill, 160/160 tables, epochs matched, commit 41.6 ms before resume | **alive** |

## Prediction 2: why the slot collective did not see this death

| Run | Pause | Detect | Exit | Change | Restore | Resume | First token |
|---|---|---|---|---|---|---|---|
| rerun replace_mid_1 | 1.250 | 0.364 (sentinel) | 0.365 | 0.144 | 0.410 | 0.042 | 0.283 |

**What the record shows.**
- **The dead process exited in 0.365 s on this host,** and the sentinel fired then.
- **No survivor's slot kernel was late at any time after the kill.** The only late episodes in
  the run are the prefill stall, 1.93 s before it. So when rank 2 died, the survivors were not
  waiting in a slot all-reduce.
- **The detection bound was measured as 52 ms** (3 step periods of 17.3 ms).

**What is not established.** Where the survivors waited instead.
- **Inference:** each decode step also runs one NCCL all-gather for the logits, outside the slot
  collective. The killed process's GPU work runs on until its context is torn down. If that work
  finished the step's slot all-reduces, the survivors would wait in NCCL, where nothing names a
  late member.
- **What fits the inference:** the first H100 attempt, with the same kill, was detected by the
  slot collective. The four cheap REPLACE runs were all detected by the slot collective, in
  44–45 ms. Detection by the slot then depends on where in the step the kill lands.
- **What would establish it:** the survivors' stacks at the kill.

**So detection by the collective covers deaths that land in a slot all-reduce,** not every death
in a decode step. A step whose collectives are all in the slot collective would close the gap: the
logits gather as a slot all-gather, or a lateness check on the NCCL wait.

## The cheap checks

| Check | Pod | Result |
|---|---|---|
| 3× A6000, TP2, `e1644e8` (first) | `ysekielbivv4bz` | no test ran: the script lacked `CUDA_HOME` |
| 3× A6000, TP2, `5634c15` | `cu0m31k34fo9o4` | slot detection 44 ms, pause 0.853 s, sessions exact |
| 5× A6000, TP4, `12d3d2d` (first) | `mbvgrx2jn38n3g` | no test ran: the script's filter lost its quotes |
| 5× A6000, TP4, `5488ad3` | `elk3b1toavwztq` | REPLACE failed: a report taken for the `Ack`, the race the H100 hit |
| 5× A6000, TP4, `fbc6efc` | `bhkgdpjumwck19` | slot detection 45 ms, three reports folded, pause 1.119 s, sessions exact |
| 5× A6000, TP4, `4b973af` (the review's mask) | `2mo24hgs0raga7` | slot detection 44 ms, a cut-short push named, pause 1.154 s, sessions exact |

## Dollars

| Pod | Time | Cost |
|---|---|---|
| H100 first attempt `yq3jxlut1ojcgh` | 00:11:00–00:34:52 at $17.45/h | $6.94 |
| H100 rerun `1xbucv0yh1ifv7` | 01:22:49–01:41:54 at $17.45/h | $5.55 |
| Six cheap checks | 3× A6000 at $1.59/h, 5× A6000 at $2.65/h | $2.85 |
| **Total** | | **$15.34** (cap $10, raised by the owner for the rerun) |

The balance is $10.13. No pods remain.

## Addendum, 2026-10-07: where the survivors were

Written after the result; the verdicts above are unchanged. The inference under prediction 2
named the logits all-gather. The stacks taken later on 5× A6000 (`bench/runs/PLANNED/
stall-stacks/`) show otherwise:
- A kill 0.9 into a step leaves every survivor's step thread idle in vLLM's worker loop, waiting
  for the executor's next RPC. The step's collectives finished, and the executor waits for the
  dead rank's answer.
- Decision 0021's step-stall rule covers that case. Verified with kills at 0.85, 0.9 and 0.95 of
  a step, found in 41–46 ms (`bench/runs/PLANNED/stall-verify/`).
