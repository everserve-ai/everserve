#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Run a slow test file until it first fails, then keep the evidence: the pytest output, the
# runtime event logs (rank*.json) of the failing run, and the commit it ran on. It does not retry
# a failure into green: it stops at the first one. Each iteration gets its own TMPDIR, so the
# newest everserve-* directory is the failing run() call's (the tests mkdtemp and never delete).
# Usage: bench/harness/catch_flake.sh OUT_DIR [TEST_PATH] [MAX_ITERATIONS]
set -uo pipefail
cd "$(dirname "$0")/../.."
out=$1; test=${2:-core/tests/test_runtime_fault.py}; max=${3:-40}
mkdir -p "$out"
commit=$(git rev-parse HEAD)
dirty=$(git status --porcelain -- core | wc -l | tr -d ' ')
for i in $(seq "$max"); do
  tmp=$(mktemp -d)
  log="$out/pytest-iter$i.txt"
  TMPDIR="$tmp" uv run pytest -x -m slow -v "$test" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    run=$(ls -td "$tmp"/everserve-* 2>/dev/null | head -1)
    mkdir -p "$out/events"
    [ -n "$run" ] && cp "$run"/rank*.json "$out/events/" 2>/dev/null
    failed=$(grep -m1 "^FAILED" "$log")
    printf 'commit: %s (uncommitted changes under core/: %s)\nhost: %s\nfailed at iteration: %s of %s\ntest: %s\nevent logs from: %s\ndate: %s\n' \
      "$commit" "$dirty" "$(uname -mrs)" "$i" "$max" "$failed" "$run" "$(date -u +%FT%TZ)" > "$out/FAILURE.txt"
    ls "$out/events" >> "$out/FAILURE.txt" 2>/dev/null
    find "$out" -name 'pytest-iter*.txt' ! -name "pytest-iter$i.txt" -delete
    echo "failed at iteration $i: $failed"
    exit 1
  fi
  rm -rf "$tmp"
done
echo "no failure in $max iterations" | tee "$out/NO-FAILURE.txt"
find "$out" -name 'pytest-iter*.txt' -delete
