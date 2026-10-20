# Result: the Kubernetes test without GPUs, with the step guard's refusal

**Verdict: pass, on the fourth attempt.** On 2026-10-14, `bench/harness/k8s/e2e.sh` met all
five of its assertions in one run. The fifth is new: a pod that steps otherwise than its engine
is refused at its ask, before it is told anything, and exits 2. All four attempts and what
stopped the first three are below. Three of them were the harness, none the product. The
passing run's output is `run/run.log`, each probe's record is `run/logs/*.log`, and the refused
pod's own record, as Kubernetes kept it, is `run/logs/probe-odd-0.pod.json`.

## What ran

As the [first run](../2026-10-13-k8s-e2e/RESULT.md), with these differences:
- **Images:** built from main at `0c8f3fa` by `bench/harness/build_images.sh`:
  - control plane: `everserve:dev-0c8f3fa`;
  - probes: `everserve-probe:dev-0c8f3fa`.
- **Harness:** run from the tree carrying the fixes below (`7ef6536`).
- **Cluster:** Nebius managed Kubernetes in eu-west2, three `cpu-d3 8vcpu-32gb` nodes.
  - Asked for at 19:11:39Z; taken down at 19:31:30Z.
  - A 0.75 h watchdog held it throughout.
- **Assertion 5:** a third probe group, `probe-odd`, at no replicas until the test scales it to
  one. It says it steps with `"other"` where every other probe says `"probe"`
  (`stage_probe.py --step other`).

## The assertions

| # | Assertion | Read | Verdict |
|---|---|---|---|
| 1 | every stage is served by exactly one probe, and the rest stand | stage [0, 1]: `probe-head-0` worker; stage [2, 3]: `probe-nodes-2` worker, `probe-nodes-0` and `probe-nodes-1` spare | **pass** |
| 2 | every probe hears every rank placed, at its serving probe's host | ranks 0 and 1 at 10.78.7.114, ranks 2 and 3 at 10.78.7.137 | **pass** |
| 3 | a served stage that never loaded is given again to the pod that replaces it | `probe-nodes-2` deleted; its replacement was given stage [2, 3] | **pass** |
| 4 | across a coordinator restart, decisions are kept: a new probe asking for a served stage stands | `probe-nodes-3` told spare for stage [2, 3] | **pass** |
| 5 | a pod that steps otherwise than its engine is refused before it is told anything, and exits 2 | `probe-odd-0` refused, `probe is "other" here, "probe" on the engine`, never told a role; its container's last state: terminated, exit code 2, restarted once | **pass** |

## The four attempts

| Attempt | Where | What stopped it | Whose |
|---|---|---|---|
| 1 | eu-north1 | The CPU node group never came up. The region allows no non-GPU vCPUs, and the cloud does not refuse such a node group: its create waited until the CLI's 15-minute timeout, with no reason given. | harness: the run named a project in the wrong region. Fixed by `8f4a10c`: `mk8s up` reads the region's quota and refuses at once, naming the need and the limit. |
| 2 | eu-west2 | Assertions 1–3 passed. Assertion 4 checked the new probe once, 9 s after the coordinator's restart, before it had said anything. | harness: a race. Fixed by `85c7883`: assertion 4 waits for the probe's answer, as every other assertion does. |
| 3 | eu-west2 | Assertions 1–4 passed. `probe-odd-0` was refused rightly, but the check asked for the words "step config", which the step guard uses only for an empty config or a digest alone. Otherwise it names the field that differs. | harness: the check. Fixed by `d0d9059`: it reads the guard's own words. |
| 4 | eu-west2 | — | **pass** |

Before attempt 4, every check was replayed offline against attempt 3's captured logs. That
found one more harness defect: "decided" and "placed" read every probe, the refused one
included, and failed on it. In the run they passed only because they ran before assertion 5.
Fixed by `7ef6536`, which also captures the refused pod's record so its exit is read from what
Kubernetes kept. On attempt 3's logs, after that commit, every check that its data could
decide came out as it should. The exit could not be decided, because no pod record was
captured then.

## What this does not show

GPUs, vLLM and NVIDIA Dynamo. Those are the Dynamo proofs: `../2026-10-14-dynamo-proof*/`.
