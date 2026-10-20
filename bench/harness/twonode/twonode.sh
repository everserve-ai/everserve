#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The two-node smoke: two 1-GPU secure pods on RunPod's private network, stock vLLM at TP1 x PP2
# (Qwen3-1.7B), measuring (5) the inter-pod bandwidth, (3) what the head's executor does when
# node-b's worker dies and whether a restarted node-b rejoins, and (2) the launch topology.
set -u
cd ~/workspace/everserve
S=${TWONODE_SCRATCH:?a scratch directory for pod ids and ssh wrappers}
D=bench/runs/PLANNED/two-node-smoke; L=$D/run.log; mkdir -p $D/data2
R=$D/data2
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
RP="uv run python bench/harness/runpod.py"
TYPES="NVIDIA RTX A4000,NVIDIA RTX A4500,NVIDIA RTX A5000,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX 2000 Ada Generation,NVIDIA A40"
PODS=()
finish() {
  for p in "${PODS[@]}"; do $RP terminate $p --log $L; pkill -f "watchdog $p"; done
  log "pods terminated"; $RP balance --log $L; touch $S/done; exit 0
}
deploy() {  # name, data-center -> id in $ID, ssh wrapper at $S/ssh_$name
  local name=$1 dc=$2 out id o
  out=$($RP deploy --networked --gpus 1 --gpu-type "$TYPES" --disk 60 --name "everserve-$name" ${dc:+--data-center $dc} --log $L 2>&1) || { log "deploy $name failed: $out"; finish; }
  id=$(echo "$out" | tail -1 | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])") || { log "no id for $name: $out"; finish; }
  PODS+=($id); echo $id > $S/$name.id
  T0=$(date +%s); (nohup $RP watchdog $id --at $((T0+40*60)) --log $L > $S/wd_$name.out 2>&1 &)
  log "pod $name $id; watchdog at +40 min"
  for k in $(seq 1 90); do
    o=$($RP pod --networked $id)
    if echo "$o" | python3 -c "import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get('publicIp') and (d.get('portMappings') or {}).get('22') else 1)" 2>/dev/null; then break; fi
    sleep 5
  done
  echo "$o" > $S/$name.pod.json
  python3 - "$o" "$S/ssh_$name" <<'PY'
import json,sys
d=json.loads(sys.argv[1]); ip=d['publicIp']; port=d['portMappings']['22']
open(sys.argv[2],'w').write('#!/bin/sh\nexec ssh -i ${EVERSERVE_SSH_KEY:-~/.ssh/everserve} -p %s -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 root@%s "$@"\n' % (port, ip))
PY
  chmod +x $S/ssh_$name
  # the REST record's machine block is empty at creation; GraphQL has the data center at once
  DCGT=$(uv run python - $id <<'PY'
import sys; sys.path.insert(0, "bench/harness"); import runpod
d = runpod.gql('query($id: String!) { pod(input: {podId: $id}) { costPerHr machine { gpuDisplayName dataCenterId } } }', {"id": sys.argv[1]})["data"]["pod"]
print(d["machine"]["dataCenterId"], d["machine"]["gpuDisplayName"], d["costPerHr"])
PY
)
  DC=${DCGT%% *}; GT=${DCGT#* }
  log "pod $name in $DC, $GT"
  ID=$id
}
sshwait() { local n=0; until $1 true 2>/dev/null; do n=$((n+1)); [ $n -ge 60 ] && { log "no ssh via $1 after 5 min"; finish; }; sleep 5; done; }
log "third pass, pinned to EU-SE-1, aborting before setup if the tunnel is dead. The cheap two-node smoke (points 2, 3, 5) before landing 5. Tree $(git rev-parse --short HEAD). Stock vLLM, Qwen3-1.7B, TP1 x PP2, one GPU per pod, global networking."
deploy a "EU-SE-1"; A=$ID; sshwait $S/ssh_a  # the tunnel passed traffic in EU-SE-1 and not in CA-MTL-1
deploy b "$DC"; B=$ID; sshwait $S/ssh_b
# private addresses
AIP=$($S/ssh_a "ip -4 -o addr | awk '\$2!=\"lo\" && \$4 ~ /^10\\./ {print \$4}' | cut -d/ -f1 | head -1")
BIP=$($S/ssh_b "ip -4 -o addr | awk '\$2!=\"lo\" && \$4 ~ /^10\\./ {print \$4}' | cut -d/ -f1 | head -1")
IFA=$($S/ssh_a 'ip -4 -o addr | tr -s " " | cut -d" " -f2,4 | tr "\n" ";"')
log "private addresses: a=$AIP b=$BIP; interfaces a: $IFA"
[ -n "$AIP" ] && [ -n "$BIP" ] || { log "no private address: global networking not in effect"; finish; }
# the tunnel must pass traffic before any money goes into setup: retry for two minutes, else stop
PG="failed"; for k in $(seq 1 24); do
  if $S/ssh_a "timeout 5 bash -c 'cat < /dev/null > /dev/tcp/$BIP/22'" 2>/dev/null; then PG="ok after $((k*5)) s"; break; fi; sleep 5
done
log "reach b from a over the tunnel: $PG"
[ "$PG" != failed ] || { log "the tunnel passes no traffic between these pods; stopping before setup"; finish; }
# setup on both in parallel: uv, the repo, the pinned vLLM wheel, compat, the model; no rust, no kernels, no tests
SETUP='cd /workspace/everserve && export PATH=$HOME/.local/bin:$PATH && (command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) && export PATH=$HOME/.local/bin:$PATH && uv sync --all-packages > /workspace/sync.log 2>&1 && uv pip install --python .venv/bin/python "vllm==0.31.0" >> /workspace/sync.log 2>&1 && (apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq iperf3 cuda-compat-13-0 >/dev/null 2>&1; true) && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-} && .venv/bin/python -c "import torch; torch.ones(1, device=\"cuda\"); import vllm; print(vllm.__version__)" > /workspace/vllm_ok.txt 2>&1 && .venv/bin/hf download Qwen/Qwen3-1.7B --local-dir /workspace/models/Qwen3-1.7B > /workspace/model.log 2>&1 && touch /workspace/SETUP_OK'
for n in a b; do git archive HEAD | $S/ssh_$n "mkdir -p /workspace/everserve && tar -x -C /workspace/everserve"; $S/ssh_$n "mkdir -p /workspace/smoke"; for f in bw.py pp_head.py; do $S/ssh_$n "cat > /workspace/smoke/$f" < $S/$f; done; $S/ssh_$n "(setsid nohup sh -c '$SETUP' > /workspace/setup.log 2>&1 < /dev/null &)"; done
log "setup started on both"
until $S/ssh_a "test -f /workspace/SETUP_OK" 2>/dev/null && $S/ssh_b "test -f /workspace/SETUP_OK" 2>/dev/null; do sleep 15; done
SA=$($S/ssh_a 'tail -1 /workspace/vllm_ok.txt; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader' | tr '\n' ' ')
SB=$($S/ssh_b 'tail -1 /workspace/vllm_ok.txt')
log "setup done: a: $SA; b: $SB"
ENV='cd /workspace/everserve && export PATH=/workspace/everserve/.venv/bin:$HOME/.local/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-} NCCL_DEBUG=WARN VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_SOCKET_IFNAME=podnet1 GLOO_SOCKET_IFNAME=podnet1 PYTHONUNBUFFERED=1'
# (5) bandwidth: iperf3 and NCCL send/recv
$S/ssh_a "nohup iperf3 -s -1 > /workspace/iperf_s.log 2>&1 &"; sleep 2
IP3=$($S/ssh_b "iperf3 -c $AIP -t 8 -P 4 2>&1 | grep -E 'SUM.*receiver|receiver' | tail -1")
log "iperf3 b->a: $IP3"
$S/ssh_a "$ENV && (setsid nohup .venv/bin/python /workspace/smoke/bw.py 0 $AIP > /workspace/bw_a.log 2>&1 < /dev/null &)"; sleep 3
NC=$($S/ssh_b "$ENV && timeout -s KILL 240 .venv/bin/python /workspace/smoke/bw.py 1 $AIP 2>&1 | grep -E 'per send|Error|error' | tr '\n' ';'")
log "nccl b<-a: $NC"
# (3) the engine: node-b headless, the head in-process
HL="$ENV && export VLLM_HOST_IP=$BIP && (setsid nohup .venv/bin/vllm serve /workspace/models/Qwen3-1.7B --headless --nnodes 2 --node-rank 1 --master-addr $AIP --master-port 29501 --tensor-parallel-size 1 --pipeline-parallel-size 2 --dtype bfloat16 --max-model-len 1024 --max-num-seqs 8 --gpu-memory-utilization 0.6 --seed 0 > /workspace/headless.log 2>&1 < /dev/null &)"
$S/ssh_b "$HL"
log "node-b headless started"
$S/ssh_a "$ENV && (setsid nohup .venv/bin/python /workspace/smoke/pp_head.py /workspace/models/Qwen3-1.7B $AIP /workspace/head.json > /workspace/head.log 2>&1 < /dev/null &)"
n=0; until $S/ssh_a "test -f /workspace/HEAD_READY" 2>/dev/null; do n=$((n+1)); if [ $n -ge 60 ]; then log "the head did not get ready in 10 min"; $S/ssh_a 'tail -5 /workspace/head.log | cut -c1-300' >> $L; $S/ssh_b 'tail -5 /workspace/headless.log | cut -c1-300' >> $L; finish; fi; sleep 10; done
EU=$($S/ssh_a 'cat /workspace/head.json' | python3 -c "import json,sys;d=json.load(sys.stdin);print('ready',d['engine_ready_s'],'s; executor',d['executor']); print('baseline',d['phases'])")
log "engine up: $EU"
PROCS=$($S/ssh_b "ps -eo pid,etimes,comm,args | grep -E 'vllm|Vllm|python' | grep -v grep | cut -c1-120 | tr '\n' ';'")
log "node-b processes: $PROCS"
# kill node-b's worker process
WPID=$($S/ssh_b "pgrep -f 'VLLM::[W]orker|Vll[m]Worker' | head -1")
WCMD=$($S/ssh_b "ps -o pid,comm,args -p $WPID | tail -1 | cut -c1-120"); log "node-b worker to kill: $WCMD"
TK=$(date +%s.%N); $S/ssh_b "kill -9 $WPID"; log "killed node-b worker pid $WPID at $TK"
$S/ssh_a "touch /workspace/GO_KILLED"
n=0; until $S/ssh_a "test -f /workspace/HEAD_AFTER_KILL || ! pgrep -f pp_head.py >/dev/null" 2>/dev/null; do n=$((n+1)); [ $n -ge 20 ] && break; sleep 5; done
AK=$($S/ssh_a 'cat /workspace/head.json' | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['phases'].get('after_node_b_worker_killed'))")
HT=$($S/ssh_a 'grep -aiE "error|nccl|timeout|died|exit" /workspace/head.log | tail -4 | cut -c1-200 | tr "\n" ";"')
log "after the kill: $AK ; head log tail: $HT"
NB=$($S/ssh_b "ps -eo pid,comm | grep -E 'vllm|Vllm|python' | tr '\n' ';'; tail -3 /workspace/headless.log | cut -c1-200 | tr '\n' ';'")
log "node-b after the kill: $NB"
if $S/ssh_a "pgrep -f pp_head.py >/dev/null"; then
  # restart node-b headless and see whether the head recovers
  $S/ssh_b "kill -9 \$(pgrep -f 'vllm serv[e]|VLLM::[W]orker') 2>/dev/null; sleep 2; $HL"; log "node-b headless restarted"
  $S/ssh_a "touch /workspace/GO_REJOINED"
  n=0; until $S/ssh_a "test -f /workspace/HEAD_DONE || ! pgrep -f pp_head.py >/dev/null" 2>/dev/null; do n=$((n+1)); [ $n -ge 24 ] && break; sleep 5; done
  AR=$($S/ssh_a 'cat /workspace/head.json' | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['phases'].get('after_node_b_restarted'))")
  NR=$($S/ssh_b 'tail -3 /workspace/headless.log | cut -c1-200 | tr "\n" ";"')
  log "after the restart: $AR; node-b: $NR"
fi
for n in a b; do mkdir -p $R/$n; $S/ssh_$n "cd /workspace && tar -cf - *.log *.json *.txt smoke 2>/dev/null" | tar -C $R/$n -xf - 2>/dev/null; done
log "data copied"
finish
