# Shadow-set formation and the slot-collective probe on a PCIe pod, 2026-10-06

The $1 startup check on main, tree `c23c6eb`. It asks whether the shadow sets form, and whether
the adapter starts on a PCIe pod now that a probe chooses the slot-collective variant. Qwen3-1.7B,
TP2 on GPUs 0 and 1 with GPU 2 as the spare, policy `bench/harness/policies/rig-2gpu.json`. The
arms ran in this order: stock, adapter, REPLACE mid-step (rank 1 killed).

## Result

| Arm | rc | Slot variant | How chosen | Shadow sets formed | Decode tok/s (steps 10–200) |
|---|---|---|---|---|---|
| stock_1 | 0 | n/a | n/a | n/a | n/a |
| adapter_1 | 0 | host | probe: no answer from device 0 in 15.0 s (15.19 s) | rank 0 and spare: `0-R1` in 0.21 s | 1,442.1 |
| replace_mid_1 | 0 | host | cache (the probe's verdict above) | rank 0 and spare: `0-R1` in 0.21 s | 1,438.4 |

- **The probe did its job.** NVML reported peer reads among the three L40S GPUs (topology SYS/NODE,
  no NVLink). The peer-to-peer collective did not answer within its deadline, so the executor took
  the host-memory variant and recorded the reason. Before the probe, this case hung the engine at
  its first collective (A6000, A40 and L40S attempts below). The second start reused the cached
  verdict and paid nothing.
- **The shadow sets form.** Rank 0 and the spare formed `0-R1` at start in both adapter-path arms.
  Rank 1 has nothing to form under this policy.
- **REPLACE completed on the host path.** There was one change, with epochs `{0: 1}` and spare
  epoch 1, and a restore of 0.23 s. All 8 sessions' tokens are identical to adapter_1's.
- The log line "shadow groups formed" counted 0 because the drill does not print INFO lines. The
  formation is read from the `started` record in each arm's JSON.

This is a startup check on a small model, not a performance figure. The 72B battery
(`bench/runs/2026-10-06-commit-path-72b/`) ran next on 5× H100.

## Pods and spend

Each figure is the price times the time up. All pods were secure cloud.

| Pod | What happened | Cost |
|---|---|---|
| `15ooah9ti2nlhx`, 5× A6000, US, $2.65/h, 13:12–13:50 | engine hung at its first collective (PCIe p2p); watchdog | $1.66 |
| `a27kystrveyjlw`, 5× A40, SE, $2.45/h, 14:57–15:29 | same hang, which led to the investigation | $1.30 |
| `kh0ckkgp9kdlfq`, 5× L40S, ID, $5.45/h, 16:42–16:57 | deployed by a stale script, unused | $1.38 |
| `jl7jsp5i70xwap`, 3× L40S, SE, $3.27/h, 16:48–17:08 | `NCCL_P2P_DISABLE` retry; the slot collective still hung | $1.10 |
| `mvg3f4cuzkwq2e`, 3× A40, CA, $1.47/h, 18:44–19:14 | never got SSH; the check waited until the watchdog fired | $0.74 |
| `on7ndlzelod38x`, 3× A40, CA, $1.47/h, 19:16–19:22 | no SSH in 5 min, so terminated (the fix for the row above) | $0.13 |
| `cmcjhlfwfdbou7`, 3× L40S, SE, $3.27/h, 19:22–19:33 | this result | $0.63 |
| **Total** | | **$6.94** |

The account balance fell from $14.86 to $8.72, which is $6.14. That is below the estimate
probably because RunPod did not bill the CA pods that never came up. No pods or watchdogs remain.
`run.log` has every command and timestamp. Raw outputs are in `data/` and are not in git.
