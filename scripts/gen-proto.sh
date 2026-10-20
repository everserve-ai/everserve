#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Generate the Python protobuf modules from proto/ into core/everserve/proto/.
# With --check, regenerate into a temp dir and fail if the committed code differs.
# Files live at proto/everserve/proto/*.proto so protoc emits package-qualified imports
# (`from everserve.proto import signals_pb2`), with the package root as the output directory.
set -euo pipefail
cd "$(dirname "$0")/.."
files=(proto/everserve/proto/signals.proto proto/everserve/proto/mirror.proto proto/everserve/proto/policy.proto proto/everserve/proto/discovery.proto proto/everserve/proto/coord.proto proto/everserve/proto/spec.proto)
out="core"
tmp=""
if [[ "${1:-}" == "--check" ]]; then
  tmp="$(mktemp -d)"
  out="$tmp"
fi
uv run python -m grpc_tools.protoc -I proto --python_out="$out" --pyi_out="$out" "${files[@]}"
if [[ -n "$tmp" ]]; then
  for f in signals mirror policy discovery coord spec; do
    for ext in py pyi; do
      if ! diff -q "$tmp/everserve/proto/${f}_pb2.$ext" "core/everserve/proto/${f}_pb2.$ext" >/dev/null; then
        echo "generated protobuf code is stale (${f}_pb2.$ext): run scripts/gen-proto.sh" >&2
        rm -rf "$tmp"
        exit 1
      fi
    done
  done
  rm -rf "$tmp"
fi
