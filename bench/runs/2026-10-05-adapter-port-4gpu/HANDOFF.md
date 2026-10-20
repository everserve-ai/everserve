# Handoff: the adapter port on the four-GPU rig

Written 2026-10-04 for an agent that will run it. Repository: `everserve-ai/everserve` (private),
cloned at `~/workspace/everserve`. Reference harness: `infertap/invent` at `~/workspace/invent`,
`docs/invent5/week5-tests/A/plugin/` (week five, pinned vLLM `6e517b15`). Budget: **about $5 of
GPU time, hard cap $10.** Everything you write lands in this repository and must pass its gate:
this is production code a customer will read, not a harness.

## Why this run exists

`PREREGISTRATION.md` in this directory says why. In one line: every core module was ported and
proven on CPU; the vLLM adapter was not, because it cannot be validated without an engine on a GPU.
This run ports it onto the repository's core and proves REPLACE end to end on real GPUs.

Read, in this order: `PREREGISTRATION.md` (here); `docs/RUNTIME.md` sections 0, 2, 3 and 7 (the
single runtime, the three outcomes, the four layers, which harness module becomes what);
`docs/CONTRACT.md` (item 3's table is the connector mapping you implement); `adapters/vllm/README.md`
and `adapters/vllm/PRIVATE-INTERNALS.md`; `docs/REVIEW-industry-shapes.md` (the vLLM rows);
`core/everserve/comm/runtime.py`, `collective.py`, `daemon.py`, `core/everserve/state/mirror.py`
(the APIs the adapter wires); then the harness: `~/workspace/invent/docs/invent5/week5-tests/A/`
`HANDOFF.md`, `PART2-results.md`, `private-internals.md`, and `plugin/live_mem/` (`executor.py`,
`graphsafe.py`, `shadow.py`, `spare.py`, `mirror.py`, `connector.py`, `relaxed.py`).

## Part 1: confirm the pre-registration (before any code or GPU)

`PREREGISTRATION.md` is committed. Fill in only what it leaves open (the checkpoint path and the
pod type you will use), commit, and do not change a prediction after that.

## Part 2a: the port (laptop first, no GPU)

Create the adapter under `adapters/vllm/everserve_vllm/`, one class per concern, every public
symbol documented, lint clean (`scripts/preflight.sh python`), tests that skip cleanly where vLLM
is absent. Nothing under `vllm/` is edited. The mapping, from `docs/RUNTIME.md` section 7:

1. `executor.py`: one `Executor` subclass (no strict/relaxed split) from the harness's
   `LiveExecutor` plus `RelaxedExecutor.sweep_dead_reader` and `hard_stop`: a worker death is an
   event. On death it reports `RankLost` to the node daemon through the worker's
   `DaemonConnection` and keeps the engine core alive; it does **not** decide the outcome. Keep the
   broadcast-ring sweep and the RPC-without-the-dead. Every private line goes in
   `PRIVATE-INTERNALS.md` with file:line at the pin.
2. `collective_op.py`: the graph-safe custom all-reduce op from `graphsafe.py`, now calling a
   process-global `Collective` (`everserve.comm.collective`) instead of its own table. The slot
   collective stays the harness's host-memory kernel for this run, wrapped behind
   `everserve.comm.slot.SlotCollective`; it moves to `core/csrc` later (decision 0006). Decode-size
   tensors go to the slot, larger ones to the current group. Keep `compilation_config`.
3. `groups.py`: a `ShrinkableGroup` subclass over vLLM's `PyNcclCommunicator` (the harness's
   shadow groups were pynccl over `StatelessProcessGroup`), so `Collective` can hold the live
   group and every pre-formed replacement set (`everserve.policy.sets_to_preform`). `shrink()`
   raises: REPLACE never shrinks.
4. `worker.py`: per-worker bootstrap: build the `Collective` (live group, shadows, slot), the
   `Runtime`, the `MirrorCommittedKV` over `MirrorClient` to the node's `everserve-store`, and
   the `DaemonConnection` to `everserve-controller`; wire `Runtime.commit_callbacks` to the
   connector's commit and `attempt_callbacks` to `clear`. The runtime's `on_fault(dead,
   outcome=REPLACE)` arrives from the daemon as a `Change`.
5. `connector.py`: the `KVConnectorBase_V1` implementation per `docs/CONTRACT.md` item 3, with
   `handle_preemptions` -> `evicted` and `request_finished` -> `finished`.
6. `spare.py`: the replacement process from the harness's `spare.py` and `shadow.py`: pre-loaded
   on the spare GPU, its own `Collective` with a `LocalStandIn` live group and the replacement
   sets pre-formed, graphs captured against the stand-in; on `Change(REPLACE)` it restores the
   dead rank's rows with `MirrorCommittedKV.restore(dead, identity moves)`, swaps to the
   replacement set, attaches to the executor's ring and answers as that rank. The scheduler's
   replay of the uncommitted tail uses the committed count per request.
7. Drop `standby.py` and `relaxed.py`'s rebuild for this run (REBUILD is not under test).

Unit tests without vLLM: the op's size dispatch against a fake `Collective`; the connector's
lifecycle mapping against a fake `MirrorCommittedKV`; the spare's move table. Commit after the
port passes `scripts/preflight.sh`.

## Part 2b: confirm on GPUs (the paid part)

**Hardware.** The cheapest 5-GPU A40-class pod on the RunPod account configured on this machine,
same host flags as the harness weeks (`NCCL_P2P_DISABLE=1`, `disable_custom_all_reduce=True`).
**Stop at $10; never exceed $10.** Terminate the pod at the end; delete weights first.

**Setup on the pod.** Install the repository with `uv sync --all-packages` and vLLM at the pin;
build the Rust binaries with `cargo build --release -p everserve-store -p everserve-controller`;
start `everserve-store` (slots = blocks x block size of the engine, budget 8 GB) and
`everserve-controller node --policy docs/examples/policy-rig-4gpu.json --socket /tmp/es-node.sock
--discovery-dir /tmp/es-discovery --mirror-socket /tmp/es-mirror.sock --node rig`. The engine's
workers find both through the daemon record (`everserve.discovery.FileDiscovery`).

**Runs.** Stock no-fault baseline x3; adapter no-fault x3 (overhead); REPLACE between-step kill x3;
REPLACE mid-step kill x3; notice path x3 (`Notice` 10 s before a kill the driver performs at the
deadline). The driver is a script in `bench/harness/` that talks to the daemon over its socket
(`everserve.comm.wire`) to deliver the fault and the notice; commit it.

**Outputs** in this directory, then move the directory to `bench/runs/2026-MM-DD-adapter-port-4gpu/`:
`results.json` with every prediction's figure per run, `RESULT.md` with the table, the verdict per
prediction (alive, dead, weakened), the pause breakdown (detect, change, restore, first token), the
mirror commit time per step, the daemon's commit latency, the private-internals delta, surprises,
dollars; `run.log` with every command, price and timestamp. Commit after Part 1, after Part 2a,
and after Part 2b. Do not change the predictions after the run.

## If something does not work

- vLLM at the pin no longer installs on the pod image: record the error, try the nearest
  installable nightly, write the new pin into `adapters/vllm/PIN` with the reason, and re-check
  every line of `PRIVATE-INTERNALS.md` against it before running.
- The pynccl shadow groups cannot coexist with the live group on this host: record the error and
  run with a single replacement set formed at startup; if that fails, run the baselines and the
  overhead arm and report the blocker with the estimated hook as the main finding.
- The Rust binaries do not build on the pod (toolchain): install `rustup` with the pinned version
  from `rust-toolchain.toml`; if that is impossible, record it and run with the Python loopback
  store (`LoopbackCommittedKV`) and no daemon, driving `Runtime.on_fault` directly, and say so.
- Budget reached: stop, terminate, report what ran.

## The paragraph to start the agent

> Read `bench/runs/PLANNED/p1-adapter-port-4gpu/HANDOFF.md` in the repository at
> `~/workspace/everserve` and do exactly what it says, in order: confirm and commit the
> pre-registration first; port the week-five harness adapter from `~/workspace/invent` onto this
> repository's core as `adapters/vllm/everserve_vllm/` (executor, collective op, pynccl groups,
> worker bootstrap, connector, spare), with every private vLLM line listed in
> `PRIVATE-INTERNALS.md`, unit tests that skip without vLLM, and `scripts/preflight.sh` green, and
> commit; then rent the cheapest suitable 5-GPU pod on the RunPod account configured on this
> machine, build the Rust store and controller, run the baselines and the REPLACE and notice arms
> as specified with a hard cap of $10, terminate the pod, and write and commit the results with
> the verdict per prediction and the pause breakdown. Never edit a file under `vllm/`; if you
> cannot proceed without doing so, record where and why and continue with the remaining arms. Do
> not change the predictions after the run. Keep every command, price and timestamp in `run.log`.
> Reply when Part 1 is committed, when the port passes the gate, and when the results are
> committed, each time in under 150 words with the numbers and the verdict.
