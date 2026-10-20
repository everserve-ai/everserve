# harness

The drivers behind the runs under `bench/runs`. Drill-only: nothing here is product code, and a
shortcut a run needs lives here, labelled, never in `adapters/` or `core/`. `record.py` is the
rule for what of a run's output is committed: the evidence a reader needs to check the result,
with the dumps scrubbed.

Three harnesses and the helpers they share.

## `k8s/`: the Kubernetes test and the Dynamo proofs

- `e2e.sh`: the Kubernetes test without GPUs on a rented Nebius managed cluster of CPU nodes: the
  chart, the pod contract, undecided registrations, placement and release, with probes standing
  for an instance's node pods (`bench/runs/2026-10-13-k8s-e2e`).
- `dynamo_proof.sh`: an instance under NVIDIA Dynamo on a rented cluster of GPU nodes loses a
  node of the engine that is not its head while streams run through Dynamo's frontend; the
  verdict is the coordinator's record (`bench/runs/2026-10-14-dynamo-proof` and after).
- `stream_load.py`: long streaming completions through an OpenAI-compatible front end, every
  token timed.
- `killer.py`: a crash from inside a pod, every process serving a rank killed with SIGKILL.
- `proof_verdict.py`: what the coordinator's Engine object says: covered before the loss, a
  REPLACE committed after it.
- `replay_checks.sh`: every check the proof's harness acts on, run against saved Engine records
  before a rental.
- `stage_probe.py`, `probes.py`, `check.py`: a node pod of an instance without its engine, the
  probe pods as a List, and what the test asserts of their records.
- `stock_pp.py`: stock vLLM, pipeline parallel across two pods, the proof's control.
- `nccl_probe.py`: what NCCL's nonblocking initialisation does on the engine's image.
- `capture.sh`: every process of a pod from outside it, Python and native frames.
- `Dockerfile`: the probe image (the core package, CPU torch, `stage_probe.py`).

## `twonode/`: the VM drills

The control plane on two or more hosts and the engine across them (`twonode/README.md`):
`twonode.sh` (the stock two-node smoke), `nodes.sh` (the adapter across hosts: stages, losses,
the operator's drain, collection), `battery.sh` (the two-node pre-registration's arms),
`drains.sh` (the drains smoke, `bench/runs/2026-10-13-run3-drains-smoke`), `pp_head.py` (the
head of a stock TP1 x PP2 engine), `bw.py` (NCCL bandwidth between two pods), `policy.py` (a
policy for a drill shape), `accept.py` (a loss arm's verdict), `summary.py` (one file per trial),
`drains_verdict.py` (the drains smoke's verdict), `stock_restart.py` (the stock-restart
baseline), `clockoffset.py` (the clock offset between nodes), `backup_rows.py`, `store_rows.py`
and `kv_rows.py` (rows of a store or a kill-time KV file, for a row-level diff), `cache_digest.py`
(every compile cache on a node), `triton_helper.py` (Triton's driver helper in a staged cache),
`shard_copy.py` (the time to copy a shard from host memory to a GPU), `written_since.py` (every
file written since a mark).

## Images and clouds

- `build_images.sh`: builds and pushes the project's images on a rented Nebius CPU VM and
  releases the VM: the engine's (`deploy/images/engine`), the control plane's (the repository's
  `Dockerfile`) and the probe's (`k8s/`).
- `mk8s.py`: rent and release a Nebius managed Kubernetes cluster, with a watchdog that takes it
  down at a time cap.
- `nebius.py`: rent and release Nebius AI Cloud VMs for the drills.
- `runpod.py`: RunPod pods for a drill: deploy, describe, terminate, balance, and a watchdog.

## One node

- `adapter_drill.py`: one engine, eight sessions, one arm per invocation (stock, adapter, REPLACE
  with a between-step or mid-step kill, notice); a client of the node daemon.
- `adapter_port.sh`: the pod side of the adapter-port run (setup, build, fresh daemons per run,
  arms, cleanup).
- `summarize_adapter_port.py`: every pre-registered figure per run into `results.json`.
- `restore_check.py`: digests of the dead rank's rows and the spare's, for the restore check.
- `divergence_probe.py`: what each step ran and K and V rows by position, for the REPLACE
  divergence run.
- `null_connector.py`: the adapter's connector with the mirror switched off, for attribution.
- `slot_bench.py`: microseconds per decode-size all-reduce, the slot collective against NCCL.
- `stallwatch.sh`, `catch_flake.sh`: a drill with the mirror's request logging and stacks on a
  stall; a slow test run until it first fails, keeping the evidence.
- `policies/`: the engines the drills run; `sitehooks/`: a `sitecustomize` that measures the
  shadow sets' device memory; `tests/`: the harness's own tests.
