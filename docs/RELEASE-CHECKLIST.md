# Release checklist

What the open-source release of this repository waits on, from the pre-release audit of
2026-10-19 (origin/main at `80aa21e3`). Each item is done when its box is ticked and the commit
that did it is named. The file goes with the release.

## 1. Licence

- [x] Apache-2.0 text in `LICENSE`, a `NOTICE`, a copy in each Python package.
- [x] Every SPDX header `Apache-2.0`; `scripts/spdx-check.sh` pins the identifier.
- [x] `Cargo.toml`, `deny.toml`, the three `pyproject.toml`, the OpenAPI document's licence.
- [x] A decision record for the licence; `docs/ARCHITECTURE.md` and decision 0001 no longer say private.

## 2. Run records

- [x] A curation rule written in `bench/runs/README.md` and enforced by `bench/harness/record.py`:
      what a committed record keeps, what is scrubbed.
- [x] Raw output (logs, `out/` trees, `.bak`, `.kill`, `.armed`, `.err`) out of the tree; the
      records the harness tests replay moved to `bench/harness/tests/fixtures/`.
- [x] Owner paths, temp paths, registry and instance ids, public IPs scrubbed from what stays.
- [x] `.gitignore` covers logs at any depth.
- [x] The pre-registration rule restated honestly; an index of proofs by number.

## 3. Private context

- [x] No reference to the prior private harness, "the analyzer", "the rig", or a named owner's
      word as a reason; no quoted chat; no agent residue in `docs/reviews/`.
- [x] Personal credential and key names out of the harness.
- [x] Docstrings state the rule; run provenance moves to the CHANGELOG or the decision.

## 4. Code defects

Controller:
- [x] A reconnecting daemon's new link survives its old handler (`coord/link.rs`).
- [x] Coordinator messages handled in order on the daemon (`node/mod.rs`).
- [x] The store-write deadline derived from the lease TTL.
- [x] A pending notice deadline does not hold a graceful shutdown.
- [x] Bounded token reviews on the operator API; saturating deadline arithmetic; the
      coordinator's delivery timeout above the daemon's acknowledgement timeout.

Store:
- [x] Engine names validated before any path is made from them.
- [x] `held` answered in O(layers); accept errors backed off, not fatal; budget accounted per
      table; checked shape arithmetic; `.fetching` temps swept and dropped on failure.
- [x] `EngineState` field 5 and `CommitAck.error` reserved; the proto check covers `.pyi`.
- [x] The dead import and the two-stage dispatch in `service.rs` gone.

Adapter:
- [x] Nothing at plugin load changes a vLLM process that is not Everserve's.
- [x] The identity never reports a foreign checkout's commit.
- [x] One version source; `queue.Empty` caught by name.

Core:
- [x] A missing or stale kernel build is refused, not degraded to; the chosen path is in the identity.
- [x] Slot-kernel constants exported from the extension and checked at init.
- [x] Orphan docstrings; `Participant` complete; one link loop; the exchange's state on an object;
      the mirror's sender catches `Exception` and locks its counters; the trace appends.

Harness and chart:
- [x] The proof passes only when every stream completed and the loss fell inside the streams.
- [x] `e2e.sh` keeps its secrets out of the record.
- [x] Chart: readiness and the holder Service by label; release-prefixed names; the image tag
      defaults to the chart's appVersion.

## 5. Ledger and CI

- [x] `PRIVATE-INTERNALS.md` regenerated against the pin with function anchors, the torch patch
      listed, the harness history removed; the ledger test enforces the anchors.
- [x] The nightly job runs the ledger and vLLM-gated tests against vLLM main and can fail.

## 6. Structure

- [x] The SHRINK reference tier out of the shipped `everserve` package.
- [x] One `everserve_vllm/patches.py` with every vLLM patch, its ledger row, install and uninstall.
- [x] Controller: one work queue; `view.rs`/`views.rs` renamed; `node/mod.rs` and
      `tests/membership.rs` split by concern.

## 7. Documentation

- [x] One current `docs/ARCHITECTURE.md`; RUNTIME, MULTINODE, ROADMAP, PRODUCT-ARCHITECTURE
      rewritten or retired.
- [x] A status line on every decision; an index; 0045 rewritten as a scoping decision; 0038 marked.
- [x] README status from the RESULT files, with the head gap stated.
- [x] Every package README true to its tree; CHANGELOGs in release form.
- [x] A glossary.
- [x] `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR templates, a DCO statement; CONTRIBUTING's
      developer setup.

## Open decisions recorded, not done here

None yet.
