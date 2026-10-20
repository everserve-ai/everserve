#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# What every process of a pod is doing, from outside it: each Python process's every thread, its
# Python and native frames (py-spy, from an ephemeral debug container sharing the pod's process
# namespace), and the GPU's utilization and processes (nvidia-smi). For a pod that stops making
# progress, where a process's own stack dump cannot reach the log (a vLLM worker's faulthandler
# output does not).
#
#   capture.sh NAMESPACE POD OUTDIR
set -euo pipefail
ns="$1"
pod="$2"
out="$3"
mkdir -p "$out"
name="capture-$(date -u +%H%M%S)"
# root and ptrace in the debug container alone; the pod's own containers keep their user
cat >"$out/$name.profile.json" <<'EOF'
{"securityContext": {"runAsUser": 0, "runAsNonRoot": false, "capabilities": {"add": ["SYS_PTRACE"]}}}
EOF
kubectl -n "$ns" debug "$pod" --image=python:3.12-slim --target=main \
  --custom="$out/$name.profile.json" --attach=false -c "$name" -- sh -c '
pip install -q py-spy >/dev/null 2>&1
for d in /proc/[0-9]*; do  # the slim image has no ps: the process table is /proc
  pid="${d#/proc/}"
  case "$(cat "$d/comm" 2>/dev/null)" in python*|VLLM*|pt_main*) ;; *) continue ;; esac
  echo "=== pid $pid: $(tr "\0" " " <"$d/cmdline" | cut -c1-160)"
  py-spy dump --native --pid "$pid" 2>&1
done
[ -x "$(command -v py-spy)" ] || echo "py-spy did not install: the stacks above are missing"' >/dev/null
until kubectl -n "$ns" get pod "$pod" -o jsonpath="{.status.ephemeralContainerStatuses[?(@.name=='$name')].state}" \
    | grep terminated >/dev/null; do
  sleep 5
done
kubectl -n "$ns" logs "$pod" -c "$name" >"$out/$pod.stacks.txt" 2>&1
kubectl -n "$ns" exec "$pod" -c main -- nvidia-smi >"$out/$pod.nvidia-smi.txt" 2>&1 || true
echo "$out/$pod.stacks.txt"
