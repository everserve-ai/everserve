#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Drill-only: run one drill with the mirror's request logging on both ends, and if it stalls, dump
# every thread of every rank and of the store with native frames, then stop it.
#
#   bash bench/harness/stallwatch.sh NAME DRILL_ARGS...
#
# This script launches the drill, so every rank and the store are its descendants: a container
# that refuses py-spy on a process it did not start (ptrace scope) still lets an ancestor read it.
# A stall is 30 s with no new "mirror reply" line in the drill log once the first one appeared.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
W=${W:-/workspace}; OUT=${OUT:-$W/out}; name=$1; shift
export EVERSERVE_LOG=debug EVERSERVE_MIRROR_DEBUG=1
log="$OUT/$name.log"
bash "$R/bench/harness/adapter_port.sh" drill "$name" "$@" &
drill=$!
last=""; idle=0
while kill -0 "$drill" 2>/dev/null; do
  sleep 5
  n=$(grep -a -c "mirror reply" "$log" 2>/dev/null); n=${n:-0}
  if [ "$n" = "0" ]; then continue; fi
  if [ "$n" = "$last" ]; then idle=$((idle + 5)); else idle=0; last=$n; fi
  if [ "$idle" -ge 30 ]; then
    echo "$(date -u +%FT%TZ) STALL: no mirror reply for 30 s; dumping every process"
    dump="$OUT/$name.stall.txt"
    for p in $(pgrep -f "VLLM::|adapter_drill.py|everserve-store|EverserveSpare"); do
      {
        echo "===== pid $p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-100)"
        "$R/.venv/bin/py-spy" dump --native --pid "$p" 2>&1 || true
        if grep -q everserve-store /proc/$p/cmdline 2>/dev/null; then
          for t in /proc/$p/task/*; do echo "task $(basename $t) $(cat $t/comm) $(cat $t/wchan 2>/dev/null)"; done
        fi
      } >> "$dump"
    done
    grep -a "mirror request\|mirror reply" "$log" | tail -40 > "$OUT/$name.last_requests.txt"
    grep -a "request " "$OUT/$name.store.log" | tail -40 > "$OUT/$name.store_last_requests.txt"
    pkill -P "$drill"; kill "$drill"; pkill -f adapter_drill.py; sleep 2
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    echo "$(date -u +%FT%TZ) dump in $dump"
    exit 3
  fi
done
wait "$drill"
