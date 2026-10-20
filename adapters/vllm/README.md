# everserve-vllm

The vLLM adapter. A plugin: nothing under `vllm/` is edited.

## The three doors

1. **The plugin entry point**, `vllm.general_plugins`: `everserve_vllm.register`, which vLLM
   calls in every process. It registers the adapter's model classes (`models/__init__.py`,
   `register_models`: no model is ported yet, so it registers nothing), installs the step hook
   (A47) and the executor resolution (A48), and, in a process whose launcher set
   `EVERSERVE_GRAPH_SAFE=1`, the graph-safe all-reduce and the log-prob warm-up.
2. **The executor**, `EverserveExecutor`, a subclass of the stock multiprocess executor: a
   worker death is an event, not a shutdown. An engine's process is given
   `--distributed-executor-backend mp`, which the plugin resolves to it (`backend.py`): vLLM 0.31
   refuses a pipeline-parallel engine whose executor is named by its class's path. A process no
   Everserve launcher made an engine's resolves `mp` as stock.
3. **The KV connector**, `EverserveConnector`, through the documented connector interface
   (`--kv-transfer-config`): committed blocks to the mirror each step (`docs/CONTRACT.md`
   item 3).

Beside the doors, the adapter patches vLLM where it offers no surface: the step hook, the TP
all-reduce and the pipeline methods on `GroupCoordinator`, `torch.distributed.broadcast`, the
warm-up entry points, and the spare's stand-in groups. Every process-wide patch is a named entry
of `patches.py`, which `register` installs in order and under each one's condition and which can
uninstall them and list what is on; the spare's and the per-object patches stay with their owners,
named in that module's docstring. Every one is listed with its scope in the Patches section of
`PRIVATE-INTERNALS.md`.

## Process scope of the plugin

`vllm.general_plugins` loads in every vLLM process: the API server, the engine core and each
worker. Two hooks go in whatever the process: `EngineCore.step_fn` becomes a data descriptor on the
class (`serving.py`, A47), so an engine core on Everserve's executor completes membership changes
in its own step loop and one on a stock executor steps as it did; and `Executor.get_class` is
wrapped (`backend.py`, A48), resolving `mp` to Everserve's executor only in a process whose
environment names an engine. Everything else the plugin does to a process (the stacks signal,
NCCL's own error handling off, the graph-safe all-reduce, the log-prob warm-up) happens only in a
process an Everserve launcher made an engine's, `EVERSERVE_ENGINE` set with
`EVERSERVE_GRAPH_SAFE`; a vLLM process without them stays stock vLLM, whatever is installed
beside it.

## Private lines

Every line that imports, subclasses, calls or patches a vLLM name outside a documented surface
is listed in `PRIVATE-INTERNALS.md`, anchored by function on the adapter side and by file, symbol
and line in vLLM at the commit in `PIN`. `tests/test_private_internals_ledger.py` parses the
ledger and the adapter's source without vLLM: every adapter anchor must name a function, class or
method that exists; every vLLM module the adapter imports must have a row naming its path; every
private attribute reach the test greps for must be named; and every function that assigns onto a
vLLM or torch object must be in the ledger's Patches section. The list is re-read at every pin
move. The nightly job (`.github/workflows/nightly-vllm-main.yml`) installs vLLM's nightly wheel
and runs `pytest -m "not slow" adapters/vllm/tests` against it, the ledger test among them, so
these break here before they break in a deployment.

## What runs where (REPLACE on stock model code)

| module | process | does |
|---|---|---|
| `serving.py` | engine core | the step loop's hook: a step a loss ended, or a commit between steps, completes the change on the engine core's thread |
| `executor.py` | engine core | worker death is an event: report `RankLost` to the node daemon, keep the engine core alive; on the daemon's `MembershipCommitted`, the engine's half of the change (new ring, spare in place, replay of the tail) |
| `collective_op.py` | every worker, the spare | the graph-safe TP all-reduce through one `Collective`; decode sizes to the slot collective, larger to the current group; a change aborts at once and applies at the step boundary |
| `groups.py` | every worker, the spare | the live group and every pre-formed replacement set over vLLM's pynccl |
| `worker.py` | every worker, the spare | bootstrap: collective, runtime, mirror transport, daemon link; the step-boundary RPC |
| `connector.py`, `lifecycle.py` | scheduler and workers | each step's committed rows and counts to the mirror (`docs/CONTRACT.md` item 3) |
| `spare.py` | the spare | pre-loaded replacement process: restores the dead rank's rows on `Change(REPLACE)`, joins as that rank |

Install: after the wheels, build Everserve's CUDA kernels against the torch and toolkit beside
the engine, once per node, as part of the install:

```
python -m everserve.kernels
```

A process never compiles them on first use: the slot collective and the mirror's gather load
the built extension, and a spare standing for any position loads it when it stands, so a node
installed without this step shows it at standing rather than on a change (pass 14 of the
battery's dry run, 2026-10-09: the build at a spare's take cost 48 s of the pause).

The environment's bytecode is compiled at install for the same reason (`uv sync
--compile-bytecode`, `uv pip install --compile-bytecode`, or `python -m compileall` over the
environment): a spare first imports much of vLLM's stack at the take, and on pass 17 its first
take on a node compiled 1,578 modules' bytecode, the take's module term 4.4 s against 1.8 s on
its second.

Install the adapter from a `git archive` of one commit or from a checkout, as a workspace
member (`uv sync`); never from a wheel built off a checkout. Every process registers the commit
it runs (decision 0034): an archive carries it stamped into `everserve_vllm/_source.py`
(`export-subst`), and a checkout is asked for its `HEAD` and whether `core` or `adapters`
differ from it. A wheel built from a checkout has neither, so its processes register no
adapter commit and the coordinator refuses each of them as a destination, naming the field
(`refused_spares` in readiness). A wheel built from an archive keeps the stamp.

Launch: stock `vllm serve`, with `EVERSERVE_GRAPH_SAFE=1` and `EVERSERVE_ENGINE / NODE /
DISCOVERY_DIR / POLICY` in the environment and the flags `python -m everserve_vllm.flags --head`
prints: `--distributed-executor-backend mp`, which the plugin resolves to Everserve's executor in a
process whose environment names an engine (`backend.py`; vLLM 0.31 refuses a pipeline-parallel
engine whose executor is named by its class's path), the connector's `--kv-transfer-config`, and
the compile settings. The plugin's hook completes changes in the serving
engine core's step loop, so the OpenAI-compatible server keeps answering through a change;
`bench/harness/adapter_drill.py` drives the same executor in-process for the rig's runs.

## Known gaps

- The engine core's step-loop hook is a data descriptor on a private class attribute
  (`EngineCore.step_fn`, A47) until vLLM offers a supported one.
- Losing the engine's head ends the engine: a known gap (decision 0045). Under Dynamo the frontend
  migrates its streams to another instance.
- REBUILD (the old relaxed mode) and SHRINK model registrations are not in this adapter yet.
