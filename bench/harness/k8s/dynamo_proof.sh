#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The integration's first proof (decision 0041): an instance under NVIDIA Dynamo on a rented Nebius
# managed cluster of GPU nodes loses a node of the engine that is not its head while streams run
# through Dynamo's frontend, and every stream finishes without an error, Dynamo's router unaware.
#
# The cluster: --nodes GPU nodes, one GPU each. Dynamo's platform (its operator and Grove) at
# DYNAMO_PLATFORM, Everserve's chart, and an instance rendered by integrations/dynamo: the head on
# one node, the engine's other node, and --spares spare nodes. Taken down when the run ends,
# whatever its outcome, and by a watchdog at --hours in any case.
#
#   EVERSERVE_NEBIUS_PROJECT=<eu-north1 project> bench/harness/k8s/dynamo_proof.sh \
#       --control-tag <registry>/everserve:<tag> --engine-tag <registry>/everserve-engine:<tag> \
#       [--model Qwen/Qwen2.5-0.5B-Instruct] [--spares 1] [--losses 2] [--loss-how delete,crash]
#       [--hours 1.5] [--keep] [--nccl-debug] [--nccl-probe] [--nccl-probe-arms 'vllm split']
#       [--platform gpu-l40s-a --preset 1gpu-8vcpu-32gb]
#
# --losses takes the node pod that holds rank 1 that many times, each under its own streams: the
# first time the spare takes the rank; each later time, the spare the planner covered rank 1 with
# again, the pod Grove made in place of the one taken before.
#
# --loss-how says how each loss comes, in order (the last repeats): `delete`, the pod deleted as a
# drain or a reclaim does, its node noticing its ranks first; `crash`, every process of the pod
# serving a rank killed (SIGKILL, killer.py), as a GPU fault or an OOM kill ends them, no notice
# given. Default: delete.
#
# --nccl-debug adds NCCL's and torch.distributed's own logging of communicator formation to every
# engine pod. A head that never becomes ready has each engine pod's every thread, Python and
# native, captured into the run's hang/ directory (capture.sh) either way.
#
# --platform and --preset name the GPU nodes' Nebius platform and preset, one GPU each (default:
# gpu-l40s-a, 1gpu-8vcpu-32gb). `nebius compute platform list` gives a project's choices.
#
# --nccl-probe asks, after the losses and on the same GPUs, what NCCL's nonblocking initialisation
# does on the engine's image (nccl_probe.py). Its answers go to the run's nccl-probe/ and do not
# change the verdict. --nccl-probe-arms names the ways of joining it tries, in order (default:
# every one, as nccl_probe.py's join --how lists them).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
py="$root/.venv/bin/python"
control=""
engine=""
model="Qwen/Qwen2.5-0.5B-Instruct"
hours=1.5
keep=0
spares=1
losses=2
loss_how="delete"
nccl_debug=0
nccl_probe=0
platform=gpu-l40s-a
preset=1gpu-8vcpu-32gb
nccl_probe_arms="vllm blocking nonblocking-polled split split-missing"
# the engine's KV cache, stated in its arguments: the renderer gives the chart the node stores'
# slots from it (<instance>.values.json), as an operator installs it
kv_blocks=4096
block_size=16
# the store's row tables, in /dev/shm (integrations/dynamo/README.md): Qwen2.5-0.5B at PP 2 holds
# 12 layers x 2 KV heads x 65,536 slots x 2 x 64 x 2 B = 403 MB per rank, its own and a sibling's
store_budget=1073741824
platform_version=1.5.1  # Dynamo's released platform; it serves the graph's v1beta1 (integrations/dynamo)
out="${TMPDIR:-/tmp}/everserve-dynamo-proof-$(date -u +%Y%m%dT%H%M%SZ)"
while [ $# -gt 0 ]; do
  case "$1" in
    --control-tag) control="$2"; shift 2 ;;
    --engine-tag) engine="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --hours) hours="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --keep) keep=1; shift ;;
    --spares) spares="$2"; shift 2 ;;
    --losses) losses="$2"; shift 2 ;;
    --loss-how) loss_how="$2"; shift 2 ;;
    --nccl-debug) nccl_debug=1; shift ;;
    --nccl-probe) nccl_probe=1; shift ;;
    --platform) platform="$2"; shift 2 ;;
    --preset) preset="$2"; shift 2 ;;
    --nccl-probe-arms) nccl_probe_arms="$2"; shift 2 ;;
    *) echo "dynamo_proof.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$control" ] && [ -n "$engine" ] || { echo "dynamo_proof.sh: --control-tag, --engine-tag" >&2; exit 2; }
# the engine's two nodes, one the head's, and its spares
nodes=$((2 + spares))
mkdir -p "$out/logs"
logf="$out/run.log"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$logf"; }
# a cluster of this run alone: a finished run's watchdog can reach no other run's cluster
name="everserve-proof-$(date -u +%m%d%H%M%S)"
mk8s="$py $root/bench/harness/mk8s.py"
# The run's credentials, never in its record ($out is committed as the run's record): the
# certificates and their keys and the cluster's kubeconfig, in a directory of this run's alone,
# removed once the cluster is down, and kept while it is up (--keep)
secrets="$(mktemp -d "${TMPDIR:-/tmp}/everserve-proof-secrets.XXXXXX")"
export KUBECONFIG="$secrets/kubeconfig"
ns=everserve
dyn=dynamo-system
app=serving
inst=proof
registry="${control%%/*}"
ok=0
pf=""
api=""

collect() {
  for n in "$ns" "$app" "$dyn"; do
    kubectl -n "$n" get pods -o wide >"$out/logs/$n-pods.txt" 2>&1 || true
    for p in $(kubectl -n "$n" get pods -o name 2>/dev/null); do
      kubectl -n "$n" logs "$p" --all-containers --prefix >"$out/logs/$n-${p#pod/}.log" 2>&1 || true
      # a container that restarted: what it said before, which is where it failed
      kubectl -n "$n" logs "$p" --all-containers --prefix --previous \
        >"$out/logs/$n-${p#pod/}.previous.log" 2>/dev/null || rm -f "$out/logs/$n-${p#pod/}.previous.log"
    done
  done
}
finish() {
  [ -z "$pf" ] || kill "$pf" 2>/dev/null || true
  [ -z "$api" ] || kill "$api" 2>/dev/null || true
  collect
  if [ "$ok" = 1 ] || [ "$keep" = 0 ]; then
    say "taking the cluster down"
    if $mk8s down --name "$name" --log "$logf" >/dev/null; then
      rm -rf "$secrets"
    else
      say "the take-down failed: the watchdog holds; the credentials in $secrets"
    fi
  else
    say "kept up after a failure, until the watchdog: KUBECONFIG=$KUBECONFIG"
  fi
}
trap finish EXIT
trap 'exit 130' INT TERM

# before anything is rented: a build here could fail, or contend with another build, on the clock
say "the certificates, made here"
(cd "$root" && cargo run -q -p everserve-controller -- certs --out "$secrets/certs" --operator proof) >>"$logf" 2>&1

say "the cluster: $nodes $platform $preset nodes, released at the latest after $hours h"
# the watchdog detaches into its own session: nothing that stops this script stops it
$mk8s watchdog --name "$name" --at "$("$py" -c "import time; print(time.time() + $hours * 3600)")" \
  --log "$logf" >/dev/null 2>&1
$mk8s up --name "$name" --version 1.33 --platform "$platform" --preset "$preset" \
  --nodes "$nodes" --drivers cuda13.0 --kubeconfig "$KUBECONFIG" --log "$logf" >"$out/mk8s-up.json"
cat "$out/mk8s-up.json" >>"$logf"
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu' | tee -a "$logf"
if ! kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' | grep 1 >/dev/null; then
  say "the nodes advertise no GPU: the GPU operator's device plugin, the drivers being the image's"
  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >>"$logf" 2>&1 || true
  helm repo update >>"$logf" 2>&1
  helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace \
    --set driver.enabled=false --wait --timeout 15m >>"$logf"
fi

token="$("${NEBIUS_BIN:-nebius}" --profile "${NEBIUS_PROFILE:-everserve}" iam get-access-token)"
for n in "$ns" "$app"; do
  kubectl create namespace "$n" >>"$logf"
  kubectl label namespace "$n" pod-security.kubernetes.io/enforce=privileged >>"$logf"
  kubectl -n "$n" create secret docker-registry nebius-cr --docker-server="$registry" \
    --docker-username=iam --docker-password="$token" >>"$logf"
done
kubectl -n "$ns" create secret generic everserve-tls --from-file="$secrets/certs/coordinator.crt" \
  --from-file="$secrets/certs/coordinator.key" --from-file="$secrets/certs/ca.crt" \
  --from-file="$secrets/certs/ca.key" >>"$logf"

say "Dynamo's platform $platform_version, with Grove"
(cd "$out" && helm fetch "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-$platform_version.tgz")
helm install dynamo-platform "$out/dynamo-platform-$platform_version.tgz" -n "$dyn" --create-namespace \
  --set global.grove.install=true --wait --timeout 15m >>"$logf"

say "the instance: the head, one node of the engine, $spares spare node(s)"
"$py" - "$out" "$inst" "$app" "$engine" "$model" "$spares" "$kv_blocks" "$block_size" <<'EOF'
import json, sys
out, name, ns, image, model, spares, blocks, block = sys.argv[1:]
json.dump({"name": name, "namespace": ns, "image": image, "frontendImage": image,
           "engineArgs": ["--model", model, "--tensor-parallel-size", "1",
                          "--pipeline-parallel-size", "2", "--max-model-len", "8192",
                          "--gpu-memory-utilization", "0.8",
                          # the KV cache the node's store was sized for (node.store.slots)
                          "--num-gpu-blocks-override", blocks, "--block-size", block],
           "gpusPerNode": 1, "spareNodes": int(spares), "guarantee": "continuous",
           "headOnDemand": False, "imagePullSecrets": ["nebius-cr"],
           "dynamoRuntimeVersion": "1.6.0"},  # DYNAMO_PIN's runtime
          open(f"{out}/instance.json", "w"))
EOF
"$py" -m everserve_dynamo "$out/instance.json" --out "$out" >>"$logf"
if [ "$nccl_debug" = 1 ]; then
  "$py" - "$out/$inst.k8s.json" <<'EOF'
import json, sys
path = sys.argv[1]
doc = json.load(open(path))
debug = [{"name": "NCCL_DEBUG", "value": "INFO"},
         {"name": "NCCL_DEBUG_SUBSYS", "value": "INIT,BOOTSTRAP,NET"},
         {"name": "TORCH_DISTRIBUTED_DEBUG", "value": "DETAIL"}]
for obj in doc.get("items", [doc]):
    for comp in obj.get("spec", {}).get("components", []):
        if comp["name"] in ("head", "nodes"):
            for c in comp["podTemplate"]["spec"]["containers"]:
                if c["name"] == "main":
                    c["env"] = [*c.get("env", []), *debug]
json.dump(doc, open(path, "w"))
EOF
fi
# the front end's migration decisions, each with its error's class, in the record: at info,
# Dynamo's DYN_LOG filter keeps no migration line (a 503 alone says nothing of its migration)
"$py" - "$out/$inst.k8s.json" <<'EOF'
import json, sys
path = sys.argv[1]
doc = json.load(open(path))
log = {"name": "DYN_LOG", "value": "info,dynamo_llm::migration=debug,dynamo_llm::http::service=debug"}
for obj in doc.get("items", [doc]):
    for comp in obj.get("spec", {}).get("components", []):
        if comp.get("type") == "frontend":
            for c in comp["podTemplate"]["spec"]["containers"]:
                c["env"] = [e for e in c.get("env", []) if e["name"] != "DYN_LOG"] + [log]
json.dump(doc, open(path, "w"))
EOF
say "Everserve's chart"
# the policy's engine is the renderer's entry for the instance, as an operator adds it to the
# chart's policy.engines (integrations/dynamo): its widths and guarantee are the instance's
"$py" - "$out/policy.json" "$out/$inst.engine.json" <<'EOF'
import json, sys
json.dump({"version": "proof-1", "outcomesInOrder": ["OUTCOME_REPLACE"],
           "spares": {"perEngine": 1, "loading": "LOADING_PRELOADED"}, "maxConcurrentLosses": 1,
           "noticeMinMs": 2000, "cover": {"enabled": True},
           "engines": [json.load(open(p)) for p in sys.argv[2:]]},
          open(sys.argv[1], "w"))
EOF
helm install everserve "$root/deploy/kubernetes/everserve" -n "$ns" \
  --set image.repository="${control%:*}" --set image.tag="${control##*:}" \
  --set coordinator.replicas=1 --set "imagePullSecrets[0].name=nebius-cr" \
  --set-json "policy=$(cat "$out/policy.json")" --set-file tls.caCertificate="$secrets/certs/ca.crt" \
  -f "$out/$inst.values.json" --set node.store.budgetBytes="$store_budget" \
  --wait --timeout 10m >>"$logf"

kubectl apply -f "$out/$inst.k8s.json" >>"$logf"
say "the coordinator's API: its record decides when the loss is taken and whether it passed"
kubectl -n "$ns" create serviceaccount proof-viewer >>"$logf"
kubectl -n "$ns" create rolebinding proof-viewer --clusterrole=everserve-viewer \
  --serviceaccount="$ns:proof-viewer" >>"$logf"
viewer="$(kubectl -n "$ns" create token proof-viewer)"
forward_api() {
  kubectl -n "$ns" port-forward svc/everserve-coordinator 17443:7443 >>"$logf" 2>&1 &
  api=$!
  sleep 3
}
forward_api
# The Engine object (decision 0042: its readiness under .status) or its events. A read that fails
# leaves no file, never the last one: a check on it reads "no record", not an older state. The
# forward is opened again if it ended: the run reads the record for as long as its losses take.
record() {
  local file="$out/${1//\//-}.json"
  rm -f "$file"
  kill -0 "$api" 2>/dev/null || forward_api
  curl -sS --fail --cacert "$secrets/certs/ca.crt" --resolve coordinator:17443:127.0.0.1 \
    -H "Authorization: Bearer $viewer" "https://coordinator:17443/v1/$1" \
    >"$file.part" 2>>"$logf" && mv "$file.part" "$file" || rm -f "$file.part"
}
# the graph's head pod
heads_of() { echo "nvidia.com/dynamo-component=head,nvidia.com/dynamo-graph-deployment-name=$1"; }
say "waiting for $inst to serve"
deadline=$(( $(date +%s) + 1800 ))
# the graph serves once its head is ready
head_ready() {
  kubectl -n "$app" get pods -l "$(heads_of "$inst")" \
    -o jsonpath='{.items[*].status.containerStatuses[*].ready}' 2>/dev/null | grep true >/dev/null
}
until head_ready; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "$inst's head never became ready: every engine pod's stacks to $out/hang"
    for p in $(kubectl -n "$app" get pods -o name | grep -E 'head|node' | cut -d/ -f2); do
      "$here/capture.sh" "$app" "$p" "$out/hang" >>"$logf" 2>&1 || true
      kubectl -n "$app" logs "$p" -c main >"$out/hang/$p.log" 2>&1 || true
    done
    exit 1
  fi
  sleep 15
done
# and the coordinator's record says it serves: every position held, a head registered, no members
# running other software (proof_verdict.py serving); a ready head alone is read before the record
line=""
until record "engines/$inst" && line="$("$py" "$here/proof_verdict.py" serving "$out/engines-$inst.json")"; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "$inst never served by its record: ${line:-no record read}"
    exit 1
  fi
  sleep 5
done
say "$inst serves: $line"
kubectl -n "$app" get pods -o wide | tee -a "$logf"
frontend="$(kubectl -n "$app" get svc -o name | grep -i "/$inst-frontend" | head -1)"
[ -n "$frontend" ] || { say "no frontend Service of $inst: $(kubectl -n "$app" get svc -o name | tr '\n' ' ')"; exit 1; }
kubectl -n "$app" port-forward "$frontend" 18000:8000 >>"$logf" 2>&1 &
pf=$!
sleep 5

# the head holds rank 0; the engine's one node pod holds rank 1, and that pod is the one taken
lost_rank=1

say "warm: one short completion"
"$py" "$here/stream_load.py" --url http://127.0.0.1:18000 --model "$model" --streams 1 --tokens 32 \
  --out "$out/warm.json" | tee -a "$logf"

# One loss under load, read from the coordinator's record: wait for a formed set to cover rank 1,
# take the node pod the record places rank 1 on while 8 streams run, and read the change. Loss n
# commits epoch n: the verdict holds the record to exactly that change.
lose() {
  local n="$1" covered=0 line node victim streams=0 replaced=0 how
  # this loss's way, from --loss-how in order, the last repeating
  how="$(echo "$loss_how" | tr ',' '\n' | sed -n "${n}p")"
  [ -n "$how" ] || how="$(echo "$loss_how" | tr ',' '\n' | tail -1)"
  case "$how" in delete|crash) ;; *) say "--loss-how: $how is neither delete nor crash"; exit 2 ;; esac
  say "loss $n of $losses: waiting for a formed set to cover the loss of rank $lost_rank"
  # rank 1's set formed, and the engine reading Covered True: the head's node's loss restarts
  # the engine and leaves Covered's list, so this engine shape can be covered
  local engine_line=""
  for _ in $(seq 1 60); do
    record "engines/$inst"
    if line="$("$py" "$here/proof_verdict.py" covered "$out/engines-$inst.json" "$lost_rank")" &&
       engine_line="$("$py" "$here/proof_verdict.py" engine-covered "$out/engines-$inst.json" "$lost_rank")"; then
      covered=1; break
    fi
    engine_line="$("$py" "$here/proof_verdict.py" engine-covered "$out/engines-$inst.json" "$lost_rank" || true)"
    sleep 10
  done
  say "cover: $line"
  say "the engine: $engine_line"
  if [ "$covered" = 0 ]; then
    record "engines/$inst/events"
    say "the loss of rank $lost_rank was never covered"
    exit 1
  fi
  node="$("$py" "$here/proof_verdict.py" holder "$out/engines-$inst.json" "$lost_rank")"
  victim="$(kubectl -n "$app" get pods -l "nvidia.com/dynamo-component=nodes" \
    -o jsonpath='{range .items[*]}{.metadata.name} {.spec.nodeName}{"\n"}{end}' |
    while read -r pod on; do if [ "$on" = "$node" ]; then echo "$pod"; fi; done)"
  [ -n "$victim" ] || { say "no node pod on $node, where the record places rank $lost_rank"; exit 1; }

  say "loss $n ($how): 8 streams of 6000 tokens (about 40 s each); $victim, which holds rank $lost_rank on $node, taken away while they run"
  "$py" "$here/stream_load.py" --url http://127.0.0.1:18000 --model "$model" --streams 8 \
    --tokens 6000 --out "$out/load-$n.json" >>"$logf" 2>&1 &
  load=$!
  sleep 8  # the streams under way, most of them still to run (1500 tokens take about 9.5 s)
  if ! kill -0 "$load" 2>/dev/null; then
    # the streams ended before any node was taken: the engine failed under load alone
    wait "$load" || true
    say "the streams ended before the loss: $(head -c 400 "$out/load-$n.json")"
    exit 1
  fi
  # its log first: a deleted pod's goes with it, and the serving node's is the record of its rank
  kubectl -n "$app" logs "$victim" --all-containers --prefix >"$out/logs/$app-$victim.taken.log" 2>&1 || true
  # when the node is taken, on this clock, which the load's records keep too: the verdict reads
  # that the loss fell inside every stream, not that the streams happened to finish
  taken_at="$("$py" -c "import time; print(time.time())")"
  echo "$taken_at" >"$out/taken-$n.at"
  if [ "$how" = crash ]; then
    # every process serving a rank in the pod killed at once, no signal it could answer: vLLM's
    # workers by the title vLLM gives them, and a spare that took its rank, which serves in its own
    # process. The node learns of it as their departure and gives no notice (a container's PID 1
    # ignores SIGKILL sent from inside, so the processes are named)
    kubectl -n "$app" exec -i "$victim" -c main -- python3 - <"$here/killer.py" \
      >"$out/kill-$n.txt" 2>&1 || say "loss $n: the kill in $victim reported an error (its pod may be ending)"
    say "loss $n: $(head -1 "$out/kill-$n.txt"), $(grep -c killed "$out/kill-$n.txt") processes killed"
  else
    kubectl -n "$app" delete pod "$victim" --wait=false >>"$logf"
  fi
  wait "$load" || streams=1
  "$py" -c "import json; d=json.load(open('$out/load-$n.json')); print(json.dumps({k: v for k, v in d.items() if k != 'per_stream'}))" | tee -a "$logf"
  if line="$("$py" "$here/proof_verdict.py" inside "$out/load-$n.json" "$taken_at")"; then
    say "the loss: $line"
  else
    say "the loss fell outside the streams, so they show nothing of it: $line"
    exit 1
  fi

  say "loss $n: the coordinator's record of it"
  record "engines/$inst"
  record "engines/$inst/events"
  cp "$out/engines-$inst.json" "$out/engines-$inst-loss-$n.json" 2>/dev/null || true
  cp "$out/engines-$inst-events.json" "$out/engines-$inst-events-loss-$n.json" 2>/dev/null || true
  line="$("$py" "$here/proof_verdict.py" replaced "$out/engines-$inst.json" "$lost_rank" "$n")" && replaced=1
  say "the change: $line"
  say "who serves what: every node pod's answer, the one Grove made again included"
  for p in $(kubectl -n "$app" get pods -l "nvidia.com/dynamo-component=nodes" -o name); do
    echo "${p#pod/}: $(kubectl -n "$app" logs "$p" -c main 2>/dev/null | grep -o 'told [a-z]*' | tail -1)" | tee -a "$logf"
  done
  if [ "$streams" = 0 ]; then
    say "loss $n: every stream finished its tokens without an error"
  else
    say "loss $n: a stream ended in an error, or short of its tokens"
    exit 1
  fi
  # streams that finish are not the proof alone: they finish while a change is refused and the
  # taken pod serves through its grace period
  [ "$replaced" = 1 ] || { say "loss $n: the spare never took rank $lost_rank at epoch $n"; exit 1; }
}

# --nccl-probe: what NCCL's nonblocking join does on the engine's image, on this cluster's GPUs
# beside the serving engine (nccl_probe.py). One process in the head's pod and one in a node pod,
# on another node. Its answers are the run's record for decision 0035's join, and do not change
# the proof's verdict.
probe() {
  local pod="$1"
  shift
  # bounded: an abort that never returns is one of the answers (exit 124); the rounds take ~5 min
  # NCCL's own account of the initialisation, to the process's stderr: where a join waits
  kubectl -n "$app" exec -i "$pod" -c main -- env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET \
    NCCL_DEBUG_FILE=/dev/stderr timeout 600 python3 - "$@" <"$here/nccl_probe.py"
}
# a group of two formed one way ($3), rank 0 in pod $1 and rank 1 in pod $2, then its all-reduces
join_pair() {
  local a="$1" b="$2" how="$3" dir="$4" uid="" first
  say "nccl probe: a group of two formed by $how, then eager and graphed all-reduces ($a, $b)"
  probe "$a" join --rank 0 --how "$how" >"$dir/join-$how-0.json" 2>"$dir/join-$how-0.err" &
  first=$!
  for _ in $(seq 1 120); do
    uid="$(sed -n 's/^id //p' "$dir/join-$how-0.json" | head -1)"
    if [ -n "$uid" ] || ! kill -0 "$first" 2>/dev/null; then break; fi
    sleep 1
  done
  if [ -n "$uid" ]; then
    probe "$b" join --rank 1 --id "$uid" --how "$how" \
      >"$dir/join-$how-1.json" 2>"$dir/join-$how-1.err" || true
  else
    say "nccl probe: $how: rank 0 printed no id"
  fi
  wait "$first" || true
  say "nccl probe: $how rank 0: $(tail -1 "$dir/join-$how-0.json")"
  say "nccl probe: $how rank 1: $(tail -1 "$dir/join-$how-1.json" 2>/dev/null)"
}
nccl_probe() {
  local a b dir="$out/nccl-probe" uid=""
  mkdir -p "$dir"
  a="$(kubectl -n "$app" get pods -l "$(heads_of "$inst")" -o name | head -1)"
  b="$(kubectl -n "$app" get pods -l "nvidia.com/dynamo-component=nodes,nvidia.com/dynamo-graph-deployment-name=$inst" -o name | head -1)"
  [ -n "$a" ] && [ -n "$b" ] || { say "nccl probe: no head pod or no node pod"; return 1; }
  say "nccl probe: a group of two whose second member never comes ($b)"
  probe "$b" missing --deadline-s 10 >"$dir/missing-server.json" 2>"$dir/missing-server.err" || true
  say "nccl probe: $(tail -1 "$dir/missing-server.json")"
  say "nccl probe: a member whose server made its id and is gone ($a)"
  probe "$b" mint >"$dir/mint.json" 2>"$dir/mint.err" || true
  uid="$("$py" -c 'import json, sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["id"])' \
    "$dir/mint.json")" || uid=""
  if [ -n "$uid" ]; then
    probe "$a" missing --as 1 --id "$uid" --deadline-s 10 \
      >"$dir/missing-member.json" 2>"$dir/missing-member.err" || true
    say "nccl probe: $(tail -1 "$dir/missing-member.json")"
  fi
  # each way of joining in turn, on the same two pods: vLLM's blocking init (the control),
  # ncclCommInitRankConfig blocking, nonblocking polled on each call, and a blocking child split
  # from a nonblocking parent, with every member and with one that leaves before the split
  for how in $nccl_probe_arms; do
    join_pair "$a" "$b" "$how" "$dir"
  done
}

# counted up from 1, and none for --losses 0 (BSD's `seq 1 0` counts down)
n=1
while [ "$n" -le "$losses" ]; do
  lose "$n"
  n=$((n + 1))
done
kill "$api" 2>/dev/null || true
say "passed: $losses losses of rank $lost_rank under load, each taken by a spare"
[ "$nccl_probe" = 0 ] || nccl_probe || say "nccl probe: did not run to the end"
ok=1
