# Landing review checklist

The gate (format, lint, docstrings, tests, schema freshness, SPDX) is mechanical. This list is what
a reviewer checks by reading, before a landing is treated as accepted. CI green is the
precondition for this review, not a substitute. The bar: code we would show a neocloud's platform
team without blinking.

## Structure

- One class or module per concern; nothing does two jobs because they happened to be adjacent.
- No module-level mutable state carrying runtime data (`_M = {}`, `SHADOW = {}` and the like).
  State lives on an object with a documented lifetime. One sanctioned exception: a per-process
  singleton set once at bootstrap where the host framework instantiates the object (a torch
  custom op, a vLLM-instantiated connector) and there is no other way for it to find ours. The
  docstring says so.
- No `sys._getframe`, no monkeypatching of a third-party class at import, no `print`. Logging goes
  through `logging`; facts for the operator go through metrics or the event log. A frame
  inspection or a patch of an engine class is tolerated only as a listed private line whose
  entry names the upstream hook that removes it.
- Library code reads no environment variables. Configuration arrives through constructors, the
  policy document or the CLI.
- Nothing copied verbatim from a measurement harness. A harness proves the physics; the
  repository holds the design.

## Placement (decisions 0038 and 0045)

- Does this put the only copy of something, or a decision for more than one node, on a GPU node?
  A GPU node holds only its own GPUs' work and copies of state that exist elsewhere (its store's
  rows are replicated to its siblings). The coordinator and its spec store are the only copies of
  the control plane's state, and they live off the GPU nodes. The one exception is the engine's
  head, the scheduler's process, which runs on rank 0's node: its loss restarts the engine and is
  the known gap of decision 0045, not a placement a landing may add to.

## Engine boundary

- Every line that touches a private vLLM or SGLang name is in `adapters/vllm/PRIVATE-INTERNALS.md` with file
  and line at the pin, and the entry says why it is needed and what upstream hook would remove it.
- A shortcut a GPU run needed lives in `bench/harness/`, labelled, never in `adapters/` or `core/`.
- The adapter exposes only what `docs/CONTRACT.md` names.

## Correctness

- Nothing on the step thread reads device memory while rows are saved or committed: no `.cpu()`,
  `.tolist()`, `.item()`, `.numpy()`, no `bool(tensor)`, no boolean-mask index. A read there
  waits for the whole forward (decision 0010, the 2026-10-05 run). The test covers the method
  forms; the mask form is yours to catch.

- A test exists that would fail if the claim were false (not a test of something next to it).
- Timeouts on every wait; a failure surfaces as an error, never as silence or a hang.
- The commit message says what is now true.

## Documents

- The module docstring says what the module is for and what it is not for.
- The changelog of every touched package has a line.
- A changed contract, flag or field has its doc updated in the same commit.
