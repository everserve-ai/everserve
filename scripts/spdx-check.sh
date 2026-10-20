#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Every source file carries the SPDX header naming this repository's licence. Generated protobuf
# code is exempt (protoc owns it).
set -euo pipefail
cd "$(dirname "$0")/.."
missing=0
while IFS= read -r f; do
  case "$f" in
    core/everserve/proto/*_pb2.py|core/everserve/proto/*_pb2.pyi) continue ;;
  esac
  if ! head -3 "$f" | grep -q "SPDX-License-Identifier: Apache-2.0$"; then
    echo "missing or wrong SPDX header (expected Apache-2.0): $f" >&2
    missing=1
  fi
done < <(git ls-files '*.py' '*.pyi' '*.rs' '*.sh' '*.proto' '*.toml' '*.yml' '*.yaml' '*.cu' '*.cpp' '*.h')
exit $missing
