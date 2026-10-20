#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Every check of the record the Dynamo proof's harness acts on (proof_verdict.py), run against saved
# Engine records, one line per check and record: what the harness would have read there. Run it
# before a rental, and compare each line with what the run that saved the record did: a harness
# that reads a record early stops a run that should have gone on.
#
#   bench/harness/k8s/replay_checks.sh <engines-proof*.json>...
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="$(cd "$here/../../.." && pwd)/.venv/bin/python"
verdict="$here/proof_verdict.py"
check() {
  local f="$1" name="$2" out code
  shift 2
  out="$("$py" "$verdict" "$name" "$f" "$@" 2>&1)"
  code=$?
  printf '%-4s %-16s %-4s %s\n' "$([ "$code" = 0 ] && echo ok || echo no)" "$name" "$*" "$out"
}
for f in "$@"; do
  echo "== $f"
  check "$f" serving
  for r in 0 1; do
    check "$f" holder "$r"
    check "$f" covered "$r"
  done
  check "$f" engine-covered 1
  for e in 1 2; do
    check "$f" replaced 1 "$e"
  done
done
