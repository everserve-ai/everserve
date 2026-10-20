#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The deployed control plane on one machine: the certificates from `certs`, one coordinator over
# TLS, and a store and a daemon for each of the two-node policy's three nodes, every TCP link
# under mutual TLS, replication listeners up. Passes when the coordinator reports three daemons,
# every daemon is linked, every store is healthy, and the operator API answers. The same
# commands, with real hosts in the env files, are the install.
# Usage: deploy/baremetal/smoke.sh [binaries-dir]   (default: target/debug); SMOKE_PORT_BASE
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${1:-$R/target/debug}"
# Below every common ephemeral range (Linux 32768+, macOS 49152+), so no outgoing connection
# holds one of these ports when a process binds it.
P="${SMOKE_PORT_BASE:-27100}"
T="$(mktemp -d)"
# the stores' tables in shared memory, as on a node (/dev/shm/everserve), not on the disk the
# build fills; a machine without /dev/shm (macOS) keeps them beside the rest
if [[ -d /dev/shm ]]; then S="$(mktemp -d /dev/shm/everserve-smoke.XXXXXX)"; else S="$T/shm"; fi
pids=()
cleanup() {
  for p in "${pids[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  rm -rf "$T" "$S"
}
trap cleanup EXIT
C="$T/certs"
"$BIN/everserve-controller" certs --out "$C" --node node-a --node node-b --node node-c --operator smoke > "$T/certs.log" 2>&1
tls_coord=(--tls-cert "$C/coordinator.crt" --tls-key "$C/coordinator.key" --tls-ca "$C/ca.crt")
EVERSERVE_LOG=info "$BIN/everserve-controller" coord --policy "$R/docs/examples/policy-two-node.json" \
  --listen "127.0.0.1:$P" --lease "$T/coord.lease" --metrics-addr "127.0.0.1:$((P + 1))" \
  --api-addr "127.0.0.1:$((P + 3))" --api-operators smoke \
  "${tls_coord[@]}" > "$T/coord.log" 2>&1 &
pids+=($!)
i=0
for n in node-a node-b node-c; do
  i=$((i + 1))
  mkdir -p "$S/$n" "$T/$n/disc"
  tls_node=(--tls-cert "$C/$n.crt" --tls-key "$C/$n.key" --tls-ca "$C/ca.crt")
  rp=$((P + 10 * i))       # the store's replica listener
  sm=$((P + 10 * i + 1))   # the store's metrics
  dm=$((P + 10 * i + 2))   # the daemon's metrics
  EVERSERVE_LOG=info "$BIN/everserve-store" --socket "$T/$n/mirror.sock" --data-dir "$S/$n" \
    --slots 64 --budget-bytes 67108864 --metrics-addr "127.0.0.1:$sm" \
    --replica-listen "127.0.0.1:$rp" --peer-name store "${tls_node[@]}" > "$T/$n.store.log" 2>&1 &
  pids+=($!)
  EVERSERVE_LOG=info "$BIN/everserve-controller" node --coordinator "127.0.0.1:$P" --coordinator-name coordinator \
    --socket "$T/$n/node.sock" --node "$n" --discovery-dir "$T/$n/disc" \
    --mirror-socket "$T/$n/mirror.sock" --mirror-replica "127.0.0.1:$rp" \
    --metrics-addr "127.0.0.1:$dm" --notice-source none "${tls_node[@]}" > "$T/$n.node.log" 2>&1 &
  pids+=($!)
done
fail() { echo "smoke: $1" >&2; for f in "$T"/*.log; do echo "== $f"; tail -20 "$f"; done; exit 1; }
# the same check an operator runs after the install: every policy node connected, devices
# reported, every spare node's store taking replicas
"$BIN/everserve-controller" check --policy "$R/docs/examples/policy-two-node.json" \
  --coordinator-api "127.0.0.1:$((P + 3))" --ca "$C/ca.crt" --cert "$C/operator-smoke.crt" --key "$C/operator-smoke.key" \
  --timeout-secs 20 > "$T/check.log" 2>&1 || fail "check: $(cat "$T/check.log")"
grep -q "node-c: connected, .*replicas at 127.0.0.1:$((P + 30))" "$T/check.log" || fail "check did not see node-c's replica address: $(cat "$T/check.log")"
curl -sf "http://127.0.0.1:$((P + 1))/metrics" | grep -q '^everserve_daemons_connected 3$' || fail "the coordinator does not report three daemons"
for i in 1 2 3; do
  curl -sf "http://127.0.0.1:$((P + 10 * i + 2))/metrics" | grep -q '^everserve_coordinator_linked 1$' || fail "daemon $i is not linked"
  [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$((P + 10 * i + 1))/healthz")" == 200 ]] || fail "store $i is not healthy"
done
# the operator API (decision 0039): TLS to the coordinator's name, the operator's certificate
api=(--cacert "$C/ca.crt" --cert "$C/operator-smoke.crt" --key "$C/operator-smoke.key" --resolve "coordinator:$((P + 3)):127.0.0.1")
API="https://coordinator:$((P + 3))"
# the objects (decision 0042)
[[ "$(curl -s -o /dev/null -w '%{http_code}' --cacert "$C/ca.crt" --resolve "coordinator:$((P + 3)):127.0.0.1" "$API/v1/engines")" == 401 ]] || fail "the operator API answered without a credential"
body="$(curl -sf "${api[@]}" "$API/v1/engines")" || fail "no engines"
[[ "$body" == *'"items":[]'* ]] || fail "unexpected engines: $body"
body="$(curl -sf "${api[@]}" "$API/v1/nodes/node-c")" || fail "no node-c"
[[ "$body" == *'"linked":true'* ]] || fail "node-c is not linked: $body"
deadline=$(( $(date +%s) * 1000 + 1000 ))
code="$(curl -s -o /dev/null -w '%{http_code}' -X PUT -H 'Content-Type: application/json' \
  -d "{\"spec\":{\"unschedulable\":true,\"drain\":{\"deadlineMs\":$deadline}}}" "${api[@]}" "$API/v1/nodes/node-x")"
[[ "$code" == 422 ]] || fail "a drain of a node holding nothing answered $code"
# the operator's commands over the same API, set once in the environment
export EVERSERVE_API="127.0.0.1:$((P + 3))" EVERSERVE_API_CA="$C/ca.crt" \
  EVERSERVE_API_CERT="$C/operator-smoke.crt" EVERSERVE_API_KEY="$C/operator-smoke.key"
listed="$("$BIN/everserve-controller" get nodes)" || fail "get nodes failed"
grep -qE '^node-c +true' <<< "$listed" || fail "get nodes did not list node-c linked: $listed"
set +e
"$BIN/everserve-controller" drain node-x --in 1 > "$T/drain.out" 2>&1
drained=$?
set -e
[[ "$drained" == 1 ]] && grep -q "holds no member" "$T/drain.out" \
  || fail "a drain of a node holding nothing exited $drained: $(cat "$T/drain.out")"
grep -q "linked to the coordinator" "$T/node-c.node.log" || fail "node-c's daemon did not log its link"
echo "smoke: green (three nodes over TLS on 127.0.0.1:$P)"
