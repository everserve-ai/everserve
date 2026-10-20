# Result: run 3, the drains smoke, on five L40S VMs

**Verdict: pass.** Attempt 11 (2026-10-13, tree `4df4aac`) met every pre-registered prediction:
H1 to H3, R1 to R5, N1 to N5, C1 and C2. Its record is `run/attempt11/`, scored by
`bench/harness/twonode/drains_verdict.py` into `run/attempt11/verdict.json`.

**What ran.**
- **Shape:** Qwen3-1.7B at TP1 x PP3 over node-a, node-b and node-c. Spares for rank 1 on node-d and rank 2 on node-e. Five `gpu-l40s-a 1gpu-8vcpu-32gb` VMs on Nebius eu-north1.
- **Load:** four sessions of a 512-token prompt and 16,000 generated tokens each.
- **The test:** two operator drains under load, node-b with a 20 s lead and node-c with a 120 s lead. The coordinator was killed (SIGKILL) and restarted 10 s after node-b's first notice, inside its lead.
- **Before the drains:** the warm-up staged node-b's and node-c's compile caches onto the spares' nodes, and a cordon round took both spare nodes out and back.
- **Steps:** the pre-registration's.
- **vLLM:** no file under `vllm/` was edited.

## Verdict per prediction

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| H1 | the node guards pass, none skipped | all passed | **alive** |
| H2 | only node-a registers as the engine's head | head registrations a 2, b 0, c 0 | **alive** |
| H3 | no head conflict | none read, `everserve_head_conflicts_total` 0 | **alive** |
| R1 | each drained node releases its worker at its boundary | node-b at epoch 1, node-c at epoch 2, neither overdue | **alive** |
| R2 | the release kills at once, the process gone | killed, 0.16 s; no engine process left on b or c | **alive** |
| R3 | the release is not a loss; positions on the spares' nodes | no exit reported lost; rank 1 on node-d, rank 2 on node-e | **alive** |
| R4 | the drained nodes hold nothing and exit | both "holds nothing", `node_main` exited 0 | **alive** |
| R5 | every session reaches 16,000 tokens across both changes | 4 x 16,000, 2 changes, drill exit 0 | **alive** |
| N1 | cordoned spares take nothing | none failed | **alive** |
| N2 | node-c's notices wait for node-b's commit | none failed | **alive** |
| N3 | a drain underway cannot be moved | 409, deadline held | **alive** |
| N4 | stored drains survive the restart, in order | adopted, re-sent with node-b's deadline, committed in order | **alive** |
| N5 | the engine is never down | down in no read, nor in the coordinator's log | **alive** |
| C1 | both losses covered before the drains | covered at registration and after the uncordon | **alive** |
| C2 | rank 2's set covered again before node-c's deadline | covered at epoch 1, 79 s before it | **alive** |

## The changes

| Change | Claim to commit | Acks | Spare staged wait | Restore | Boundary | Resume | Sessions' pause |
|---|---|---|---|---|---|---|---|
| node-b, rank 1, epoch 1 | 1.673 s | 1.673 s | 1.481 s | 0.187 s | 12 ms | 5.5 ms | 1.68 s |
| node-c, rank 2, epoch 2 | 0.563 s | 0.563 s | 0.1 ms | 0.558 s (3.56 M rows) | 29 ms | 9.9 ms | 0.59 s |

- **Node-b's 1.48 s wait is explained by and fixed in `7a1587d`; not yet confirmed on hardware.** The coordinator staged rank 1 onto node-d from node-e's store, although node-d already held rank 1's replica. That re-pointed node-d's store mid-stream, and the spare waited for node-e to catch up; the coordinator's log shows the staging. With the fix, a spare whose node holds its rank's replica restores there, with nothing staged. No run has read a change on that tree.
- **The steady-state figure is node-c's 0.56 s.**
- **The sessions' pause is the longest gap between decode steps at each change.** Those are the two longest of the run's 16,000 steps.
- **Throughput:** 674 tok/s over steps 10 to 200.

**At the restart.** The restarted coordinator adopted cover plan 6 from node-a's daemon. All five daemons relinked within 5 ms, and it issued no plan until the epoch-1 commit. It read both drains from its spec store and re-sent node-b's notice with the original deadline.

## How it got here

Attempts 1 to 10 each stopped on a fault this smoke was built to find. Each is noted in the pre-registration with its record under `run/`. The faults and their fixes:

- **Rental and harness:** the public IPv4 quota, compile caches not staged, sessions too short, logs not collected before release, a driver edited while it ran, waits counted in polls, the restart timed from the wrong log, and a watchdog spent by reruns.
- **Process side:**
  - Released workers ignored their death pipe; they are now killed at once.
  - A process outside the membership counted its own epoch; it now takes the change's.
  - A pipeline transfer refused by a change was reported as a loss.
  - A planned departure was decided from a notice each process remembered, which a boundary cleared. It is now decided by the change itself (protocol 9, `Change.departing`).
  - Drained nodes could not tell their rank had left. The commit now carries the whole membership.
  - A standing spare read its rank's departure as its own.
  - The cover took a newer rendezvous for one already forming.
- **Control plane (the peer's):**
  - The drained node never got the commit.
  - Ports were reused while still bound.
  - A restart re-planned the cover with new ports; it now adopts the daemons' plan.
  - A drain refused by the cover gate stayed blocked; it now retries once covered.
  - The staging above.

**Spend.** About $60 over twelve rentals, attempt 11 about $4 (`results.json`). Attempt 12, a repeat of the pass on `7a1587d` to read node-b's change without the staging defect, was released before it ran: Nebius had no capacity for the VM size (`NotEnoughResources`).

**Not established.** The pass is one run. A repeat would read the claim to commit for node-b's change without the staging defect, and shows whether the pass repeats. The model is small; nothing here measures a frontier shape.
