# Pre-registration: the planned head handoff measured on GPUs (headOnRankNodes)

Written before the run. This is a measurement, with no pass bar, as the NCCL probes were. Its
numbers are inputs to proof-24's rewritten pre-registration, which tests the handoff under
`headOnRankNodes`. Each number has the decision it makes written down here, before it is seen,
so that the result can't be read to suit.

**The run.** `bench/harness/k8s/dynamo_proof.sh --head-on-rank-nodes --loss-target head
--loss-how notice --losses 2 --notice-s 120 --spares 1 --second-engine --platform gpu-h100-sxm
--preset 1gpu-16vcpu-200gb --hours 1.5`.
- The images are built from main at the commit that lands this file with the harness's
  measurement (`dynamo_proof.sh --head-on-rank-nodes`, `handoff_read.py`). The record names the
  built hash.
- **Engine A** (`proof`) runs Qwen2.5-0.5B at TP 1 × PP 2, on 2 GPU nodes plus 1 spare node. Its
  core runs as a process in one of its own node pods, started by the coordinator.
- **Engine B** (`proof-b`) runs on 1 GPU node. Both engines share Dynamo's global namespace, so
  streams that A's retired core breaks can move to B.
- Each notice is the operator's `drain` of the core's node, over the coordinator's API, with its
  deadline 120 s away. That is the synthetic notice Cort chose: "fake notices with a lets say 2
  min … time window".
- The policy's defaults: `headHandoffMs` 30 s, `startHeadMs` 300 s, cover on.

**Cap.** Cort approved it on 2026-10-15: 4 × H100 for about 30 minutes, at about $4.45 per
GPU-hour, comes to about $9. Building the images costs under $1. The 1.5 h watchdog makes the
worst case about $27.

## The sequence

1. **Bring-up.** No core runs. The coordinator starts one on the node of the lowest rank told to
   serve. This is start 1.
2. **Notice 1**, on the core's node.
   - A successor starts at once on another of A's nodes. This is start 2.
   - At about T+90 s, the node's ranks move to their spares.
   - Then the core is retired, while 32 streams run (measurement 3 says why 32).
   - The successor takes the head, and the change commits at epoch 2.
   - The drained node is uncordoned, so its pod asks again and stands as a spare.
3. **Notice 2**, on the new core's node, once its ranks are covered. The successor starts on
   the node now holding rank 0. This is start 3. The head change commits at epoch 4.

If the drained node's pod never stands as a spare for the second notice's rank, notice 2 is not
sent. The record states this (`never covered within 900 s`), and measurement 1 then has two
starts, not three.

## Measurements, and what each decides

| # | Measured | Read from | Decides |
|---|---|---|---|
| 1 | **Cold start.** For each core the coordinator starts, on the coordinator's clock: the time from StartHead sent to the core's ask answered (`is told ROLE_HEAD` or `ROLE_HEAD_SPARE`), and, for a successor, on to its registration as the standby (`a standby head stands`), which is what the handoff waits for. A head registering logs no line, so the bring-up start has the answer alone. At least 3 starts: bring-up and two successors. Also the time of the rank move, from `headHandoffMs` before the drain's deadline to its commit, and of the head change, from the retire to its commit, with the retired core's graceful shutdown inside it. | `starts.json` (`handoff_read.py starts`), `changes.json` (`handoff_read.py changes`), both from the coordinator's log | `startHeadMs` is set to 2 × the slowest successor's time to registration. (The start's bound clears on the answer, so 2 × the slowest answer is its floor.) Which clouds' notice windows fit a handoff is decided by the handoff's own two constraints, from these numbers alone: (a) the successor's time to registration ≤ window − `headHandoffMs`/2, since the start runs from T+0 alongside the wait; and (b) the rank move + the head change ≤ `headHandoffMs`/2, since the ranks move at deadline − `headHandoffMs`. The windows are AWS 120 s, Azure about 30 s, and GCP 0 or 30 s. The serial sum, slowest start + rank move + head change ≤ window, is also reported, labelled as conservative and stricter than the product needs. The README states for each cloud whether (a) and (b) hold. |
| 2 | **Footprint.** The GPU memory used on the successor's GPU before notice 1 and once the successor stands: `nvidia-smi` per process, or the GPU's memory used where the container lists no process. Also whether the worker's KV cache allocation changed (its log). | `gpu-before-1-*.csv`, `gpu-after-1-*.csv` (`handoff_read.py footprint --utilization 0.8`); the worker's KV cache lines | If the core takes more than the headroom vLLM's `--gpu-memory-utilization` leaves (1 − 0.8 of the GPU), the core must be counted in the worker's budget, a change to the renderer. If it takes less, nothing changes. |
| 3 | **Retire under load.** The frontend logs at debug level. 32 streams are sent through the shared frontend, unpinned, and the router splits them between A and B. The streams first served by A's pods are the ones in flight on A when its core gets SIGTERM, with B up, and the record counts them. The streams are not pinned to A's core: a stream sent to one worker would be sent to it again when it migrated, which would test the pin, not the product. For each stream first on A: how it ended (finished whole, or its error's class, `backend.engine_shutdown` or another), whether it migrated, and to which instance. | `ends-N.json` (`handoff_read.py streams --retired proof-0-`: `on_retired`, `on_retired_migrated`, `on_retired_errors`), `served-N.json`, `frontend-migration-N.txt`, `load-N.json` | If every stream that was on A migrates and finishes, the planned handoff is seamless with two or more replicas, as the README says. If any ends in a class that can't be migrated, that blocks the GPU proof until it is understood (the lesson of proof-24's attempt 2). If fewer than 8 streams were on A in a notice, the reading for that notice is too thin to decide on, and the record says so. |

## The record holds no credential

The certificates and their keys, the cluster's kubeconfig and the operator's token are kept in a
directory outside the record, which is removed once the cluster is down. `.gitignore` also keeps
any run's `certs/`, `kubeconfig`, `*.key` and `*.token` out of git.

## What this run does not do

- **The page cache.** A cold page cache is not forced: dropping caches needs a privileged
  container, which the engine's pods are not. Start 1 is the first run of the core's command in
  its pod, but the node process has already imported vLLM from the same files. The record states
  this as the condition the starts were measured under.
- **Unannounced loss.** A node lost with no notice still restarts the engine. That is decision
  0038's stated behaviour and is not measured here.

The record is `bench/runs/2026-10-15-dynamo-handoff-measure/` with a `RESULT.md` that gives each
number and the decision it made. The peer reviews it.
