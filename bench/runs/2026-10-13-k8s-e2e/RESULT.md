# Result: the Kubernetes test without GPUs, on a Nebius managed cluster

**Verdict: pass.** On 2026-10-13, `bench/harness/k8s/e2e.sh` met all four of its assertions in one
run, from the cluster's creation to its release. Its output is `run/run.log`, and each probe's
record is `run/probe-*.log`.

**What ran.**
- **Cluster:** Nebius managed Kubernetes 1.33 in eu-north1. Three `cpu-d3 8vcpu-32gb` nodes, no
  GPUs. Up at 21:53:20Z; the take-down was asked at 21:54:33Z and finished at 22:05:06Z. The
  watchdog, set at 0.75 h, found nothing left at 22:38:19Z.
- **Images:**
  - control plane: `everserve:dev-0c766a8` (the coordinator and the node DaemonSet);
  - probes: `everserve-probe:dev-27e7961`, built from the tree committed as `27e7961`, the
    harness and 1bf03d6's link fix included.
- **Chart:** Everserve's, with one coordinator replica, the node DaemonSet on all three nodes,
  and each node's certificate issued in the cluster.
- **Probes:** probes stand in for an instance's node pods (`bench/harness/k8s/stage_probe.py`).
  Each registers undecided over protocol 11, as `everserve_vllm.node_main --slot` does, with
  the pod contract's mounts, user and environment:
  - node 0's probe (`probe-head`);
  - a StatefulSet of three asking by ordinal for stage [2, 3] (`probe-nodes`).
- **Policy:** `run/policy.json`. One engine at TP 2 × PP 2 (world 4, two ranks per stage), two
  spares per engine with cover on, and the members placed by what registers.

## The assertions

| # | Assertion | Read | Verdict |
|---|---|---|---|
| 1 | every stage is served by exactly one probe, and the rest stand | stage [0, 1]: `probe-head-0` worker; stage [2, 3]: `probe-nodes-1` worker, `probe-nodes-0` and `probe-nodes-2` spare | **pass** |
| 2 | every probe hears every rank placed, at its serving probe's host | ranks 0 and 1 at 10.79.18.240, ranks 2 and 3 at 10.79.18.127, on every probe | **pass** |
| 3 | a served stage that never loaded is given again to the pod that replaces it | `probe-nodes-1` deleted; its replacement was given stage [2, 3] | **pass** |
| 4 | across a coordinator restart, decisions are kept: a new probe asking for a served stage stands | coordinator pod deleted and rolled out; `probe-nodes-3` told spare for stage [2, 3] | **pass** |

The two `"failed": "a probe asked for stage [0, 1]"` entries under assertion 1 are the check's
polls before every probe had registered. The driver re-collects the logs and re-runs the check
until its deadline (`wait_for` in `e2e.sh`), and it passed on the third read.

## What the earlier runs found

Two earlier runs on the same harness failed. Each finding was fixed before this run, and the
fixes are on main:
- the coordinator's named non-root user and its unreadable TLS secret (0c766a8: numbered users,
  `fsGroup`, mode 0440);
- the node certificate's 226 s back-off, now capped at a minute;
- the placement never reaching a node that held only spares (every daemon now hears each
  placement);
- the instance state (protocol 11's InstanceState, Signal 19) ending every Python link
  (1bf03d6).

## What this does not show

GPUs, vLLM and NVIDIA Dynamo. Probes register and report and serve nothing. The integration's
proof under Dynamo on GPU nodes is a separate run.
