# Contributing

## Rules that do not bend

1. **Nothing under `vllm/` or `sglang/` is edited.** An adapter reaches an engine only through
   documented surfaces or through lines it lists in its `PRIVATE-INTERNALS.md`, by file and line,
   per pin.
2. **Pre-register every measurement.** A GPU run has a pre-registration committed before it
   starts and a `results.json` committed before any number appears in prose.
3. **GPU spend is stated before a run and approved**, never discovered after.
4. **Docs move with code.** A change to a contract, field or flag updates its doc in the same
   commit.
5. **A check must fail if its claim is false.** No grep over compiler output; exit codes only.
   Ask "if this claim were false, would this check go red?"
6. **Small commits, plain messages, no trailers but the sign-off.** The subject line says what is
   now true; the only trailer is the Developer Certificate of Origin's `Signed-off-by`.
7. **Every source file carries the SPDX header** (`scripts/spdx-check.sh`). Generated protobuf code
   is the one exemption.
8. **Read the neighbours before inventing.** `docs/REVIEW-industry-shapes.md` records how Dynamo,
   vLLM and NIXL do what we do and what we adopt or deliberately differ on; extend it when a new
   piece of structure lands.

## Licence

Everything here is Apache-2.0 (decision 0046). A contribution is accepted under the same licence
with a Developer Certificate of Origin sign-off on each commit (`git commit -s`, which adds
`Signed-off-by`); there is no contributor licence agreement.

## Review

A landing is accepted when a reviewer has read it against `docs/REVIEW-CHECKLIST.md`. The gate
being green is where the review starts.

## Setting up

What the gates need, and where CI gets each (`.github/workflows/ci.yml` pins the versions):

| tool | for | version |
|---|---|---|
| `uv` | the Python workspace (`uv sync --all-packages --index https://download.pytorch.org/whl/cpu` on a machine without a GPU: CPU torch, the tests' only need) | any recent |
| Rust | the crates; the toolchain is pinned in `rust-toolchain.toml` and `rustup` installs it | 1.97.1 |
| `protoc` | the etcd client generates its gRPC types with it | any recent |
| `etcd` | the lease tests start a real one | v3.7.2 |
| `kube-apiserver` and `kubectl` | the lease, spec-store and chart tests start a real API server (envtest, from the `envtest-v1.37.0` release of kubernetes-sigs/controller-tools, checked against its `.sha512`) | v1.37.0 |
| `helm` | the chart tests render the chart | v4 |
| `buf` | the schema's lint and its breaking check, in CI only | v1.74.0 |

`brew install protobuf etcd helm`, or `apt install protobuf-compiler` and etcd from its release.
Preflight checks the four binaries first and names the one missing. The Python tests drive the
real store and controller binaries, so preflight builds those two crates before the Python
tier: there is no cargo-free Python loop, by design (a Python fake of the control plane would
pass what the control plane refuses). vLLM is not needed; the adapter's vLLM-gated tests skip
without it and CI runs them against the pin.

## Gates

- `scripts/preflight.sh [package]`: format, lint, engine-free tests. Exit code is the verdict.
- `scripts/prepush.sh`: preflight for the workspace, plus `cargo doc` with warnings denied.
- CI runs clippy and ruff with warnings as errors. A warning locally is an error there.
- The Rust toolchain is pinned in `rust-toolchain.toml` and CI installs that exact version, so the
  local gate and CI run the same lints. Move the pin on purpose, in its own commit.

## Package boundaries

`core` imports no engine. `adapters/*` import `core` and one engine. `controller` and `store`
speak to `core` only over the signal protocol and the mirror API. A split is
`git subtree split` plus a CI change; if a change would make that harder, it is wrong.

## Unsafe code

`unsafe_code` is denied across the workspace. The one exception is `store/src/shm.rs`, two memory
map calls whose soundness rests on the mirror protocol, written above them (decision 0011). A new
`unsafe` block anywhere else needs its own decision record and the same treatment: one module, the
invariants in its documentation, nothing else in it.
