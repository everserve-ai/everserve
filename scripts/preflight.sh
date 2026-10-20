#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Inner loop. Format, lint, the fast tests. Judged by exit code only.
# The slow tier (multi-process torch suites, marked `slow`) runs in prepush.sh and in CI.
# Usage: scripts/preflight.sh [python|rust] [--slow]
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:-all}"
slow=""
for arg in "$@"; do
  [[ "$arg" == "--slow" ]] && slow="yes"
done
[[ "$what" == "--slow" ]] && what="all"

if [[ "$what" == "all" || "$what" == "python" ]]; then
  uv sync --all-packages --quiet
  scripts/spdx-check.sh
  python3 -m json.tool deploy/grafana/everserve.json > /dev/null   # the dashboard is valid JSON
  scripts/gen-proto.sh --check
  uv run ruff format --check .
  uv run ruff check .
  # the mirror client's tests start the store binary, and the worker-loss end to end the
  # controller's, each preferring the debug build: build both from this tree first, or a binary
  # left from an older build answers them (a protocol bump read as nine failures, 2026-10-10; the
  # end to end failing on a fix its stale controller lacked, 2026-10-19)
  cargo build --quiet -p everserve-store -p everserve-controller
  if [[ -n "$slow" ]]; then
    uv run pytest
  else
    uv run pytest -m "not slow"
  fi
fi

if [[ "$what" == "all" || "$what" == "rust" ]]; then
  # the controller's build needs protoc (etcd-client); its lease tests start a real etcd and
  # envtest's kube-apiserver (kubernetes-sigs/controller-tools releases, envtest-v1.37.0)
  for tool in protoc etcd kube-apiserver helm; do
    command -v "$tool" > /dev/null || { echo "preflight: $tool is not on PATH (see CONTRIBUTING.md, Gates)" >&2; exit 1; }
  done
  cargo fmt --all --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace --quiet
  # the deployed control plane on one machine: three nodes over TLS, from the built binaries
  cargo build --quiet -p everserve-controller -p everserve-store
  deploy/baremetal/smoke.sh target/debug
fi

echo "preflight: green"
