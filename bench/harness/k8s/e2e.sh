#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The Kubernetes test without GPUs, on a rented Nebius managed cluster of CPU nodes: the chart (the
# coordinator, the node DaemonSet and each node's certificate), the pod contract, and protocol 11's
# undecided registrations, placement and release, with probes standing for an instance's node pods
# (stage_probe.py). The cluster is taken down when the run ends, whatever its outcome, and by a
# watchdog at --hours in any case.
#
#   EVERSERVE_NEBIUS_PROJECT=<project> bench/harness/k8s/e2e.sh \
#       --control-tag cr.eu-north1.nebius.cloud/<registry>/everserve:<tag> \
#       --probe-tag cr.eu-north1.nebius.cloud/<registry>/everserve-probe:<tag> [--hours 1.5] [--keep]
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
py="$root/.venv/bin/python"
control=""
probe=""
hours=1.5
keep=0
out="${TMPDIR:-/tmp}/everserve-e2e-$(date -u +%Y%m%dT%H%M%SZ)"
while [ $# -gt 0 ]; do
  case "$1" in
    --control-tag) control="$2"; shift 2 ;;
    --probe-tag) probe="$2"; shift 2 ;;
    --hours) hours="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --keep) keep=1; shift ;;  # leave the cluster up after a failure, for a look; the watchdog stays
    *) echo "e2e.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$control" ] && [ -n "$probe" ] || { echo "e2e.sh: --control-tag and --probe-tag" >&2; exit 2; }
mkdir -p "$out/logs"
logf="$out/run.log"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$logf"; }
# a cluster of this run alone: a finished run's watchdog can reach no other run's cluster
name="everserve-e2e-$(date -u +%m%d%H%M%S)"
mk8s="$py $root/bench/harness/mk8s.py"
# The run's credentials, never in its record ($out is committed as the run's record): the
# certificates and their keys and the cluster's kubeconfig, in a directory of this run's alone,
# removed once the cluster is down, and kept while it is up (--keep)
secrets="$(mktemp -d "${TMPDIR:-/tmp}/everserve-e2e-secrets.XXXXXX")"
export KUBECONFIG="$secrets/kubeconfig"
ns=everserve
pns=everserve-e2e
registry="${control%%/*}"
ok=0

finish() {
  if [ "$ok" = 1 ] || [ "$keep" = 0 ]; then
    say "taking the cluster down"
    if $mk8s down --name "$name" --log "$logf" >/dev/null; then
      rm -rf "$secrets"
    else
      say "the take-down failed: the watchdog holds; credentials in $secrets"
    fi
  else
    say "kept up after a failure, until the watchdog: KUBECONFIG=$KUBECONFIG"
  fi
}
trap finish EXIT
trap 'exit 130' INT TERM

# before anything is rented: a build here could fail, or contend with another build, on the clock
say "the certificates, made here; each node's is issued in the cluster"
(cd "$root" && cargo run -q -p everserve-controller -- certs --out "$secrets/certs" --operator e2e) >>"$logf" 2>&1

say "the cluster: CPU nodes, released at the latest after $hours h"
# the watchdog detaches into its own session: nothing that stops this script stops it
$mk8s watchdog --name "$name" --at "$("$py" -c "import time; print(time.time() + $hours * 3600)")" \
  --log "$logf" >/dev/null 2>&1
$mk8s up --name "$name" --platform cpu-d3 --preset 8vcpu-32gb --nodes 3 --kubeconfig "$KUBECONFIG" \
  --log "$logf" >>"$logf"

token="$("${NEBIUS_BIN:-nebius}" --profile "${NEBIUS_PROFILE:-everserve}" iam get-access-token)"
for n in "$ns" "$pns"; do
  kubectl create namespace "$n" >>"$logf"
  # the DaemonSet mounts host paths; the probes mount them too
  kubectl label namespace "$n" pod-security.kubernetes.io/enforce=privileged >>"$logf"
  kubectl -n "$n" create secret docker-registry nebius-cr --docker-server="$registry" \
    --docker-username=iam --docker-password="$token" >>"$logf"
done
kubectl -n "$ns" create secret generic everserve-tls --from-file="$secrets/certs/coordinator.crt" \
  --from-file="$secrets/certs/coordinator.key" --from-file="$secrets/certs/ca.crt" \
  --from-file="$secrets/certs/ca.key" >>"$logf"

say "the chart"
cat >"$out/policy.json" <<'EOF'
{"version": "e2e-1", "outcomesInOrder": ["OUTCOME_REPLACE"],
 "spares": {"perEngine": 2, "loading": "LOADING_PRELOADED"}, "maxConcurrentLosses": 1,
 "noticeMinMs": 2000, "cover": {"enabled": true},
 "engines": [{"name": "k", "tensorParallel": 2, "pipelineParallel": 2, "headOnDemand": false}]}
EOF
helm install everserve "$root/deploy/kubernetes/everserve" -n "$ns" \
  --set image.repository="${control%:*}" --set image.tag="${control##*:}" \
  --set coordinator.replicas=1 --set "imagePullSecrets[0].name=nebius-cr" \
  --set-json "policy=$("$py" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$out/policy.json")" \
  --set-file tls.caCertificate="$secrets/certs/ca.crt" \
  --set node.store.slots=1024 --set node.store.budgetBytes=67108864 --wait --timeout 10m >>"$logf"
kubectl -n "$ns" get pods -o wide >>"$logf"

say "the probes: node 0's, and a group of three asking by ordinal"
"$py" "$here/probes.py" --image "$probe" --namespace "$pns" --world 4 --nnodes 2 --nodes 3 \
  | "$py" -c "
import json, sys
d = json.load(sys.stdin)
for i in d['items']:
    i['spec']['template']['spec']['imagePullSecrets'] = [{'name': 'nebius-cr'}]
print(json.dumps(d))" | kubectl apply -f - >>"$logf"

collect() {
  for p in $(kubectl -n "$pns" get pods -o name); do
    kubectl -n "$pns" logs "$p" >"$out/logs/${p#pod/}.log" 2>/dev/null || true
  done
}
wait_for() {  # until the check passes on fresh logs, or the deadline
  local deadline=$(( $(date +%s) + $1 )); shift
  until collect; "$py" "$here/check.py" "$@" >>"$logf" 2>&1; do
    [ "$(date +%s)" -lt "$deadline" ] || { "$py" "$here/check.py" "$@" | tee -a "$logf"; return 1; }
    sleep 5
  done
}

say "1. every stage is served by exactly one probe, and the rest stand"
wait_for 300 decided "$out/logs" --world 4 --nnodes 2
say "2. every probe hears every rank placed, by its serving probe's host"
wait_for 120 placed "$out/logs" --world 4
worker="$("$py" -c "
import json, sys
sys.path.insert(0, '$here')
import check
recs = check.records(__import__('pathlib').Path('$out/logs'))
print(next(p for p, r in recs.items() if p.startswith('probe-nodes') and r.get('told') == 'worker'))")"
say "3. the stage of $worker, which never loads, is given again to the pod that replaces it"
kubectl -n "$pns" delete pod "$worker" --wait=true >>"$logf"
rm -f "$out/logs"/*.log
wait_for 300 reserved "$out/logs" --pod "$worker" --stage 2,3
say "4. the coordinator restarts and keeps its decisions: a fifth probe asking for a served stage stands"
kubectl -n "$ns" delete pod -l app.kubernetes.io/component=coordinator --wait=true >>"$logf"
kubectl -n "$ns" rollout status deployment -l app.kubernetes.io/component=coordinator --timeout=5m \
  >>"$logf" 2>&1 || kubectl -n "$ns" wait --for=condition=Ready pod \
  -l app.kubernetes.io/component=coordinator --timeout=5m >>"$logf"
kubectl -n "$pns" scale statefulset probe-nodes --replicas=4 >>"$logf"
# the fifth probe's own answer, waited for: the four already decided pass "decided" before it has
# even started (checked once 9 s after the restart, it has said nothing yet)
wait_for 300 stood "$out/logs" --pod probe-nodes-3 --stage 2,3
say "5. a pod that steps otherwise than its engine is refused before it is told anything, and exits 2"
kubectl -n "$pns" scale statefulset probe-odd --replicas=1 >>"$logf"
wait_for 120 refused "$out/logs" --pod probe-odd-0
# its exit, as its pod reports it: the pod's record is kept beside the logs, whatever it shows
exited=1
for _ in $(seq 1 24); do
  kubectl -n "$pns" get pod probe-odd-0 -o json >"$out/logs/probe-odd-0.pod.json" 2>>"$logf" || true
  if "$py" "$here/check.py" exited "$out/logs" --pod probe-odd-0 --code 2 >>"$logf" 2>&1; then
    exited=0; break
  fi
  sleep 5
done
"$py" "$here/check.py" exited "$out/logs" --pod probe-odd-0 --code 2 | tee -a "$logf"
[ "$exited" = 0 ] || exit 1
say "passed"
ok=1
