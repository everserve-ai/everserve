#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The two-node harness: the control plane on two hosts and the engine across them.
#
# Run from the laptop with ssh wrappers for the nodes (`ssh_a`, `ssh_b`, ... in $TWONODE_SCRATCH,
# as twonode.sh writes them) and their private addresses in AIP, BIP, ... NODES lists the nodes
# ("a b" by default; "a b c" adds a third holding only a daemon and the spares it launches: the
# product's spare node). node-a is the head and the coordinator; SERVING ("a b" by default) lists
# the nodes the engine spans in node-rank order, node-a first, so "a b c" puts a stage on each.
# Stages:
#   setup     the repo, the venv, the pinned vLLM wheel, the Rust binaries and the model on every node
#   certs     a CA and a certificate per node, made on node-a and copied to the others
#   control   the store and the daemon on each node, the coordinator on node-a (TCP, TLS), and
#             every node's discovery record copied to every other
#   headless  every serving node's part of the engine but node-a's: our executor in its node role
#             (everserve_vllm.node_main), at its node rank
#   update    the working tree's tracked files to every node, over the setup's checkout: a fix
#             tried on the nodes before it is committed (the run's record says which)
#   drill     the drill on node-a as the head: `nodes.sh drill NAME ARGS...`
#   lose      node-b is lost: its daemon and workers killed together, the time written for the drill
#   kill      one of node-b's workers is killed: `nodes.sh kill NAME RANK`
#   stacks    a native stack (gdb) of every engine process on the nodes named (node-a and node-b
#             unless named), into the outputs
#   drain     the operator's drain of a node through the operator API (PUT /v1/nodes/NODE, a
#             deadline LEAD_MS from now on node-a's clock): `nodes.sh drain [NODE [LEAD_MS]]`
#             (node-b and LEAD_MS by default); the Node written and its HTTP status on stdout
#   cordon    `nodes.sh cordon NODE` and `nodes.sh uncordon NODE`: its spares take no position, or
#             may again (the spec's unschedulable); the Node written and its HTTP status on stdout
#   specs     GET /v1/nodes: every node's spec and status as the coordinator holds them, on stdout
#   coord_restart the coordinator killed (SIGKILL, as a crash leaves it) and started again on
#             the same lease and spec store; returns once the daemons have linked again
#   collect   every node's outputs and logs into $OUT_DIR/<node>/
#   down      stop everything on every node: the engine's processes interrupted (SIGINT, so they
#             run their exits), their stacks asked for through the coordinator (POST /v1/engines/rig/stacks)
#             at half of DOWN_GRACE seconds and killed at its end if any remain; each node's exit kind
#             (clean, killed, none) kept on the node for the start's written_since read
#   mark      before a start: a mark on every node, the node's start count advanced
#   written   after a start: every file written since the mark on every node (written_since.py),
#             into $OUT_DIR: `nodes.sh written NAME PHASE`
#   took      a take on node-c counted (the take index the next start's read records)
#   idle      what every node writes over SECONDS with no start (written_since_idle_baseline_<node>.json)
#   page_cache how much of the checkpoint node-a and node-b hold in page cache: `nodes.sh page_cache NAME`
# Settings: NODES, SERVING, MODEL (HF id), TP, PP (2), SERVE_GPUS_A, SERVE_GPUS_<N> (each serving
# node's devices, 0 by default), SPARE_GPU_A, SPARE_RANK, SPEC_STORE (the coordinator's spec store
# file on node-a; unset, it keeps the spec in memory),
# POLICY, OUT_DIR, MIRROR_DIR (the store's data directory, on tmpfs: /dev/shm/mirror on a VM),
# IFACE (the private network's interface: podnet1 on RunPod, eth0 on Nebius),
# SUDO ("sudo -n" when the ssh user is not root, as on Nebius; empty on RunPod),
# MODEL_REVISION (setup: the checkpoint's full commit hash; the download must resolve to it),
# DEPLOY_COMMIT (setup and update: the commit the nodes run, named once when the deploy starts;
# its tree is sent, never the working tree, so nothing changed in the clone after that reaches a
# node, and the record names the commit every node ran).
set -u
S=${TWONODE_SCRATCH:?a scratch directory holding ssh_a, ssh_b, ...}
NODES=${NODES:-"a b"}
SERVING=${SERVING:-"a b"}; NNODES=$(echo $SERVING | wc -w | tr -d ' ')
AIP=${AIP:?the private address of node-a}; BIP=${BIP:?the private address of node-b}
ip_of() { eval echo "\$$(echo "$1" | tr a-z A-Z)IP"; }  # node c's address is CIP, and so on
for n in $NODES; do [ -n "$(ip_of $n)" ] || { echo "no address for node-$n" >&2; exit 2; }; done
MODEL=${MODEL:-Qwen/Qwen3-1.7B}; TP=${TP:-1}; PP=${PP:-2}
SERVE_GPUS_A=${SERVE_GPUS_A:-0}; SPARE_GPU_A=${SPARE_GPU_A:-1}; SPARE_RANK=${SPARE_RANK:-$TP}
POLICY=${POLICY:-bench/harness/policies/two-node-tp1pp2.json}
OUT_DIR=${OUT_DIR:?where to collect outputs}
R=/workspace/everserve; W=/workspace
# the engine as the adapter pins it (adapters/vllm/PIN: the v0.31.0 release, decision 0041)
WHEEL="vllm==0.31.0"
COORD_PORT=7400; COORD_METRICS=7401; REPLICA_PORT=7402; COORD_API=7403
# the operator API (decision 0039): TLS to the coordinator's name on node-a, each caller with its
# operator certificate; `driver` acts, `scorer` only reads (a viewer: 403 on every write)
API=https://coordinator:$COORD_API
api_curl() { echo "curl -s --cacert $W/certs/ca.crt --cert $W/certs/operator-$1.crt --key $W/certs/operator-$1.key --resolve coordinator:$COORD_API:$AIP"; }
IFACE=${IFACE:-podnet1}; SUDO=${SUDO:-}
# the store's data directory holds the mirror's staging files, which the workers pin into the
# device: it must be on shared memory (tmpfs), which /workspace is on a pod and is not on a VM
MIRROR_DIR=${MIRROR_DIR:-$W/mirror}
log() { echo "$(date -u +%FT%TZ) $*"; }
on() { local n=$1; shift; "$S/ssh_$n" "$@"; }
# a detached remote process, so ssh returns and the process outlives the session
# appended, with a dated line between launches: a stage restarted mid-run keeps the earlier log
spawn() { local n=$1 logf=$2; shift 2; on "$n" "echo \"==== $(date -u +%FT%TZ) launch\" >> $logf; (setsid nohup sh -c '$*' >> $logf 2>&1 < /dev/null &)"; }
ENV="cd $R && export PATH=$R/.venv/bin:\$HOME/.local/bin:\$HOME/.cargo/bin:/usr/local/cuda-13.0/bin:\$PATH CUDA_HOME=/usr/local/cuda-13.0 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\${LD_LIBRARY_PATH:-} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 NCCL_NVLS_ENABLE=0 EVERSERVE_GRAPH_SAFE=1 EVERSERVE_LAYOUT=$TP,$PP"
MODEL_DIR="$W/models/$(basename "$MODEL")"

setup() {
  # the checkpoint's revision is pinned, a full commit hash, never a branch that can move
  [[ "${MODEL_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] || { log "MODEL_REVISION must be the checkpoint's full commit hash (got '${MODEL_REVISION:-}')"; return 1; }
  for n in $NODES; do
    # the work directory, owned by the ssh user (root on a pod; a sudoer on a VM)
    on "$n" "$SUDO mkdir -p $W && $SUDO chown \$(id -u):\$(id -g) $W"
    git archive "${DEPLOY_COMMIT:?the commit this deploy runs}" -- . ":!bench/runs" | on "$n" "mkdir -p $R && tar -x -C $R"
    on "$n" "cat > $W/setup.sh" <<EOF
set -e
cd $R
export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH
(rustup show active-toolchain || rustup toolchain install "\$(sed -n 's/^channel = "\\(.*\\)"/\\1/p' rust-toolchain.toml)")
# bytecode compiled at install, as the kernels are: a process never compiles a module on its first
# import (a spare's first take on a node compiles 1,578 modules' bytecode otherwise, 33.7 MB under
# site-packages, the modules term 4.4 s against 1.8 on its second take)
uv sync --all-packages --compile-bytecode
uv pip install --compile-bytecode --python .venv/bin/python "$WHEEL"
.venv/bin/python -m compileall -q -j 0 core/everserve adapters/vllm/everserve_vllm bench/harness
$SUDO apt-get update -qq >/dev/null 2>&1; $SUDO apt-get install -y -qq iperf3 build-essential python3-dev gdb >/dev/null 2>&1 || true  # gcc and Python.h: Triton's JIT; gdb: a native stack of a survivor
# fincore reads the checkpoint's page-cache residency for the stock restart's record (page_cache);
# Ubuntu 24.04 moved it from util-linux to util-linux-extra, and a stock restart's read is empty
# without it: a node without it fails setup
command -v fincore >/dev/null || $SUDO apt-get install -y -qq util-linux-extra >/dev/null 2>&1 || true
command -v fincore >/dev/null || { echo "fincore is not installed (util-linux-extra)"; exit 4; }
# the controller's build generates etcd's gRPC types with protoc (etcd-client, the lease's etcd
# backend): a node without it fails setup here, not inside cargo's build script
command -v protoc >/dev/null || $SUDO apt-get install -y -qq protobuf-compiler >/dev/null 2>&1 || true
command -v protoc >/dev/null || { echo "protoc is not installed (protobuf-compiler)"; exit 4; }
# the OS's own package refresh stopped for the node's life, after the install's: apt's daily timer
# writes 122 MB under /var/cache/apt and apt-esm, and inside a take's window that is read as the
# take's writes. A node without systemd (a pod) runs no such timer
if [ -d /run/systemd/system ]; then
  $SUDO systemctl disable --now apt-daily.timer apt-daily-upgrade.timer >/dev/null || { echo "apt's timers not disabled"; exit 5; }
  $SUDO systemctl stop apt-daily.service apt-daily-upgrade.service unattended-upgrades.service >/dev/null 2>&1 || true
  ! systemctl is-active -q apt-daily.timer apt-daily-upgrade.timer || { echo "apt's timers still active"; exit 5; }
fi
# the install builds the CUDA kernels ahead (everserve.kernels), never a process at first use; torch's
# build runs ninja, which the environment's own bin holds (not on the setup's PATH),
# and the headers the line above installs (Python.h)
PATH=$R/.venv/bin:\$PATH .venv/bin/python -m everserve.kernels
$SUDO sysctl -q -w kernel.core_pattern=$W/out/core.%e.%p >/dev/null 2>&1 || true  # a death leaves a core beside the outputs
cargo build --release -p everserve-store -p everserve-controller
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\${LD_LIBRARY_PATH:-}
.venv/bin/python -c "import torch; torch.ones(1, device='cuda'); import vllm; print(vllm.__version__)"
# the checkpoint at its pinned revision, and the revision the download resolved read back from
# what hf wrote beside each file: a mistyped or moved pin fails the setup rather than serving
# another revision
.venv/bin/hf download $MODEL --revision $MODEL_REVISION --local-dir $MODEL_DIR --include "*.json" --include "*.safetensors" --include "merges.txt" --include "vocab.json"
resolved=\$(head -qn1 $MODEL_DIR/.cache/huggingface/download/*.metadata | sort -u)
[ "\$resolved" = "$MODEL_REVISION" ] || { echo "the download resolved \$resolved, not the pinned $MODEL_REVISION"; exit 3; }
echo \$resolved > $W/MODEL_REVISION
echo $DEPLOY_COMMIT > $W/BUILT_AT
touch $W/SETUP_OK
EOF
    # the script's own exit status says how it ended: done, or failed with its code (a RuntimeError
    # the log match does not see would leave the loop waiting out its 28 minutes)
    on "$n" "rm -f $W/SETUP_OK $W/SETUP_FAILED"
    spawn "$n" "$W/setup.log" "sh $W/setup.sh || echo \$? > $W/SETUP_FAILED"
  done
  log "setup started on $NODES"
  while :; do
    local pending=""
    for n in $NODES; do
      on "$n" "test -f $W/SETUP_OK" 2>/dev/null && continue
      if on "$n" "test -f $W/SETUP_FAILED" 2>/dev/null; then log "setup failed on $n (exit $(on "$n" "cat $W/SETUP_FAILED"))"; on "$n" "tail -5 $W/setup.log"; return 1; fi
      pending="$pending $n"
    done
    [ -z "$pending" ] && break
    sleep 15
  done
  log "setup done on $NODES"
  # the adapter's guards that only an installed vLLM or a device can check, run once per deploy
  # before any arm: the pin (PRIVATE-INTERNALS.md A8a: a failed worker call crosses as its text)
  # and the resident copy page-locked in place at its own size; a guard that skips is a guard
  # that did not run, so a skip fails setup as a failure does
  mkdir -p "$OUT_DIR"
  on a "cd $R && .venv/bin/python -m pytest -q -rA -p no:cacheprovider adapters/vllm/tests/test_executor_failed.py adapters/vllm/tests/test_node.py adapters/vllm/tests/test_resident.py adapters/vllm/tests/test_join.py adapters/vllm/tests/test_serving.py" > "$OUT_DIR/pin_guards_a.txt" 2>&1 < /dev/null
  local rc=$?
  if [ $rc -ne 0 ] || grep -q "skipped" "$OUT_DIR/pin_guards_a.txt"; then log "the node guards failed or skipped on a (rc $rc): $(tail -3 "$OUT_DIR/pin_guards_a.txt" | tr '\n' ' ')"; return 1; fi
  log "the node guards on a: $(tail -1 "$OUT_DIR/pin_guards_a.txt")"
}

update() {
  # the deploy's commit, without the run records (bench/runs is 116 MB the nodes never read), and
  # the binaries and kernels built again on a node whose last build was of another commit: a
  # sync alone leaves the old protocol's binaries under the new protocol's Python
  local c=${DEPLOY_COMMIT:?the commit this deploy runs} n
  for n in $NODES; do
    git archive "$c" -- . ":!bench/runs" | on "$n" "tar -x -C $R && cd $R && .venv/bin/python -m compileall -q -j 0 core/everserve adapters/vllm/everserve_vllm bench/harness" || return 1
    on "$n" "$ENV && [ \"\$(cat $W/BUILT_AT 2>/dev/null)\" = $c ] || { cargo build --release -p everserve-store -p everserve-controller 2>&1 | tail -1 && .venv/bin/python -m everserve.kernels > /dev/null && echo $c > $W/BUILT_AT; }" || { log "the build at $c failed on $n"; return 1; }
  done
  log "nodes at $(git rev-parse --short "$c"), the binaries and kernels built at it"
}

certs() {
  local names=""; for n in $NODES; do names="$names --node node-$n"; done
  on a "mkdir -p $W/certs && $ENV && target/release/everserve-controller certs --out $W/certs $names --operator driver --operator scorer --overwrite"
  for n in $NODES; do
    [ "$n" = a ] && continue
    on a "cd $W/certs && tar -cf - ca.crt node-$n.crt node-$n.key" | on "$n" "mkdir -p $W/certs && tar -x -C $W/certs"
  done
  log "certs made on a, the other nodes' copied"
}

coord_start() {  # the coordinator on node-a, with the spec store when SPEC_STORE names one
  local store=""
  [ -n "${SPEC_STORE:-}" ] && store="--spec-store $SPEC_STORE"
  spawn a "$W/out/coord.log" "$ENV && EVERSERVE_LOG=info target/release/everserve-controller coord --policy $R/$POLICY --listen $AIP:$COORD_PORT --lease $W/run/coord.lease --metrics-addr $AIP:$COORD_METRICS --api-addr $AIP:$COORD_API --api-operators driver --api-viewers scorer --tls-cert $W/certs/coordinator.crt --tls-key $W/certs/coordinator.key --tls-ca $W/certs/ca.crt $store"
}

linked() {  # every daemon reports its coordinator link up
  local n
  for n in $NODES; do
    on "$n" "curl -s http://127.0.0.1:9108/metrics | grep -q 'everserve_coordinator_linked 1'" 2>/dev/null || return 1
  done
}

coord_restart() {  # the coordinator killed as a crash leaves it, then started again on the same lease and spec store
  on a "kill -9 \$(pgrep -f 'everserve-controlle[r] coord') 2>/dev/null; true"
  for _ in $(seq 50); do on a "pgrep -f 'everserve-controlle[r] coord'" > /dev/null || break; sleep 0.1; done
  log "coordinator killed"
  coord_start
  for _ in $(seq 120); do
    if linked; then log "coordinator restarted: daemons linked again"; return 0; fi
    sleep 0.5
  done
  log "the daemons did not link to the restarted coordinator"; return 1
}

control() {
  down_control
  for n in $NODES; do on "$n" "mkdir -p $W/run $W/disc $MIRROR_DIR $W/out; rm -f $W/run/*.sock $W/disc/*.json"; done
  # the store allocates every table at its full slots when an engine prepares: its data directory
  # must hold the budget, or the store refuses at the first prepare rather than here
  local free
  for n in $NODES; do
    free=$(on "$n" "df -B1 --output=avail $MIRROR_DIR | tail -1 | tr -d ' '")
    [ "${free:-0}" -ge "${STORE_BUDGET:-8000000000}" ] || { log "node-$n: $MIRROR_DIR holds ${free:-?} bytes, under the store's budget ${STORE_BUDGET:-8000000000}"; return 1; }
  done
  local common="--slots ${SLOTS:-20800} --budget-bytes ${STORE_BUDGET:-8000000000}"
  for n in $NODES; do
    spawn "$n" "$W/out/store.log" "$ENV && target/release/everserve-store --socket $W/run/mirror.sock --data-dir $MIRROR_DIR $common --replica-listen $(ip_of $n):$REPLICA_PORT --peer-name store --tls-cert $W/certs/node-$n.crt --tls-key $W/certs/node-$n.key --tls-ca $W/certs/ca.crt --metrics-addr 127.0.0.1:9109"
  done
  coord_start
  for n in $NODES; do
    local ip; ip=$(ip_of $n)
    # the daemon advertises its node's tunnel address: where groups led by its ranks rendezvous
    # and where a spare it launches binds its queues (decision 0024)
    spawn "$n" "$W/out/daemon.log" "$ENV && EVERSERVE_LOG=info target/release/everserve-controller node --coordinator $AIP:$COORD_PORT --coordinator-name coordinator --socket $W/run/node.sock --node node-$n --discovery-dir $W/disc --mirror-socket $W/run/mirror.sock --mirror-replica $ip:$REPLICA_PORT --advertise-host $ip --metrics-addr 127.0.0.1:9108 --tls-cert $W/certs/node-$n.crt --tls-key $W/certs/node-$n.key --tls-ca $W/certs/ca.crt"
  done
  for _ in $(seq 60); do
    local up=1
    for n in $NODES; do
      on "$n" "test -f $W/disc/node-$n.json && curl -s http://127.0.0.1:9108/metrics | grep -q 'everserve_coordinator_linked 1'" 2>/dev/null || { up=0; break; }
    done
    if [ $up = 1 ]; then
      # each node's record where every other node's processes read it (file discovery, N hosts)
      for n in $NODES; do for m in $NODES; do
        # written beside it and renamed into place, as the daemon writes its own: a spare reading
        # the directory must never see a record half-copied (a spare reading an empty one exits,
        # and its daemon launches it again a second later)
        [ "$n" = "$m" ] || on "$n" "cat $W/disc/node-$n.json" | on "$m" "cat > $W/disc/.node-$n.json.tmp && mv $W/disc/.node-$n.json.tmp $W/disc/node-$n.json"
      done; done
      log "control plane up: coordinator on a at $AIP:$COORD_PORT, daemons linked on $NODES, records exchanged"
      return 0
    fi
    sleep 1
  done
  log "control plane did not come up"; for n in $NODES; do on "$n" "tail -3 $W/out/coord.log $W/out/daemon.log 2>/dev/null"; done; return 1
}

headless() {
  local i=0 n gpus
  for n in $SERVING; do
    if [ "$n" = a ]; then i=$((i + 1)); continue; fi
    gpus=$(eval echo "\${SERVE_GPUS_$(echo "$n" | tr a-z A-Z):-0}")
    headless_on "$n" "$i" "$gpus"
    i=$((i + 1))
  done
}

engine_args() {  # the engine's arguments in vLLM's spelling: the stock arm's, the node role's and
  # the policy's spare launch (policy.py engine_argv) alike, so every process builds one config
  echo "--model $MODEL_DIR --tensor-parallel-size $TP --pipeline-parallel-size $PP --dtype bfloat16 --max-model-len ${MAX_MODEL_LEN:-2304} --max-num-seqs ${N:-8} --max-num-batched-tokens $(( ${N:-8} * ${PROMPT_LEN:-2048} )) --enable-prefix-caching --seed 0 --num-gpu-blocks-override ${BLOCKS:-1300} --gpu-memory-utilization ${GPU_UTIL:-0.85}"
}

headless_on() {  # NODE NODE_RANK DEVICES: that node's part of the engine
  local n=$1 rank=$2 gpus=$3
  on "$n" "kill -9 \$(pgrep -f 'everserve_vllm.node_mai[n]|vllm.entrypoints|VLLM::[W]orker') 2>/dev/null; true"
  if [ "${STOCK:-0}" = 1 ]; then  # the stock arm: vLLM's own headless part, the same rendezvous as node_main's
    local cmd=".venv/bin/vllm serve --headless $(engine_args) --nnodes $NNODES --node-rank $rank --master-addr $AIP --master-port 29501 --distributed-executor-backend mp"
    echo "$cmd" >> "$OUT_DIR/stock_launch_$n.txt"  # the stock launch, named in the record
    spawn "$n" "$W/out/headless.log" "$ENV && export CUDA_VISIBLE_DEVICES=$gpus && $cmd"
    log "node-$n headless started: stock vLLM ($MODEL, TP$TP x PP$PP, rank $rank of $NNODES nodes)"
    return 0
  fi
  # the node's engine arguments mirror the head's (adapter_drill.py), so every part agrees on
  # every buffer size; the node role builds the configuration from them (engine_args.py)
  # the drill's probes (divergence_probe, sitehooks) are imported by every worker of the engine,
  # every node's included, as the drill arranges for the head's
  spawn "$n" "$W/out/headless.log" "$ENV && ulimit -c unlimited; export CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=$R/bench/harness:$R/bench/harness/sitehooks:\${PYTHONPATH:-} && .venv/bin/python -m everserve_vllm.node_main --engine rig --node node-$n --policy $R/$POLICY --discovery-dir $W/disc $(engine_args) --nnodes $NNODES --node-rank $rank --master-addr $AIP --master-port 29501; echo \"==== node_main exited \$?\""
  log "node-$n headless started: the node role ($MODEL, TP$TP x PP$PP, rank $rank of $NNODES nodes)"
}

drill() {  # NAME, then the drill's arguments beyond the two-node ones
  local name=$1; shift
  on a "kill -9 \$(pgrep -f 'adapter_dril[l].py|VLLM::[W]orker|EverserveSpar[e]') 2>/dev/null; rm -f $W/out/$name.json $W/out/$name.json.kill $W/out/$name.json.armed; true"
  headless
  # under a managed policy (spare_launch) the daemons own the spares and the executor launches
  # none; the drill's spare arguments are then unused
  # the head's launch, named in the record (under the stock arms, the stock engine's)
  echo "adapter_drill.py $* (STOCK=${STOCK:-0})" >> "$OUT_DIR/$name.launch.txt"
  on a "$ENV && ulimit -c unlimited; export VLLM_HOST_IP=$AIP EVERSERVE_ENGINE=rig EVERSERVE_NODE=node-a EVERSERVE_DISCOVERY_DIR=$W/disc EVERSERVE_POLICY=$R/$POLICY EVERSERVE_SPARE_GPU=$SPARE_GPU_A EVERSERVE_SPARE_RANK=$SPARE_RANK EVERSERVE_EVENT_DIR=$W/out/events_$name && mkdir -p $W/out/events_$name && timeout -s KILL ${RUN_TIMEOUT:-900} .venv/bin/python bench/harness/adapter_drill.py --model $MODEL_DIR --out $W/out/$name.json --tp $TP --pp $PP --nnodes $NNODES --master-addr $AIP --serve-gpus $SERVE_GPUS_A --spare-gpu $SPARE_GPU_A --kill-rank $SPARE_RANK --engine rig --node node-a --discovery-dir $W/disc --policy $R/$POLICY --event-dir $W/out/events_$name --blocks ${BLOCKS:-1300} --prompt-len ${PROMPT_LEN:-2048} --n ${N:-8} --max-model-len ${MAX_MODEL_LEN:-2304} $*" > "$OUT_DIR/$name.drill.log" 2>&1
  local rc=$?
  log "end $name rc=$rc"
  # the drill's own status is the call's, never the log line's: a stock engine that dies with 1
  # must not read 0
  return $rc
}

lose() {  # node-b is lost: its daemon and its workers killed together, as a dead host leaves them
  local name=$1
  # under the stock arm node-b's half is vLLM's own headless server (vllm.entrypoints), lost the same way
  local t; t=$(on b "date +%s.%N; kill -9 \$(pgrep -f 'everserve-controlle[r] node|everserve_vllm.node_mai[n]|vllm.entrypoint[s]|VLLM::[W]orker') 2>/dev/null; true")
  on a "echo '{\"t_kill\": $t, \"mode\": \"node-b lost\"}' > $W/out/$name.json.kill"
  log "node-b lost at $t (daemon and workers killed)"
}

kill_worker() {  # one of node-b's workers dies: vLLM titles each VLLM::Worker[_PP<stage>][_TP<column>]
  local name=$1 rank=$2 title="VLLM::[W]orker"
  [ "$PP" -gt 1 ] && title="${title}_PP$(( rank / TP ))"
  [ "$TP" -gt 1 ] && title="${title}_TP$(( rank % TP ))"
  local pid; pid=$(on b "pgrep -f '$title' | head -1")
  if [ -z "$pid" ]; then log "no worker for rank $rank on node-b ($title)"; return 1; fi
  local t; t=$(on b "date +%s.%N; kill -9 $pid")  # the node's own clock, the instant before the kill
  on a "echo '{\"t_kill\": $t, \"mode\": \"node-b worker $rank killed\", \"pid\": $pid}' > $W/out/$name.json.kill"
  log "node-b's worker for rank $rank (pid $pid) killed at $t"
}

stacks() {  # a native stack of every engine process on the nodes: `nodes.sh stacks NAME [NODE...]`
  local name=$1 n; shift
  local nodes="${*:-a b}"
  for n in $nodes; do
    on "$n" "for p in \$(pgrep -f 'VLLM::[W]orker|adapter_dril[l].py|everserve_vllm.node_mai[n]'); do echo \"==== pid \$p \$(tr '\\0' ' ' < /proc/\$p/cmdline | cut -c1-80)\"; $SUDO gdb -batch -p \$p -ex 'thread apply all bt 8' 2>/dev/null | grep -E '^Thread|^#' | head -120; done > $W/out/${name}_native_stacks_node-$n.txt 2>&1 &"
  done
  log "native stacks of the engine processes on $nodes requested for $name"
}

put_node() {  # NODE SPEC_JSON: the node's spec written (PUT /v1/nodes/NODE); the Node and its HTTP status on stdout
  on a "$(api_curl driver) -w '\\nHTTP %{http_code}\\n' -X PUT $API/v1/nodes/$1 -H 'Content-Type: application/json' -d '{\"spec\": $2}'"
}

drain() {  # [NODE [LEAD_MS]]: the operator releases a node: a notice to its ranks, one change at the deadline
  # the spec states a deadline, never a lead (decision 0042): node-a's clock now plus the lead,
  # the coordinator's own clock, so the deadline it holds is the one this lead meant
  local node=node-${1:-b} lead=${2:-${LEAD_MS:-5000}} now
  now=$(on a "date +%s%3N")
  put_node "$node" "{\"unschedulable\": true, \"drain\": {\"deadlineMs\": $((now + lead))}, \"reason\": \"nodes.sh drain\"}"
}

cordon() {  # cordon|uncordon NODE: the node's spares take no position, or may again
  local verb=$1 node=node-${2:?a node} unschedulable=true
  [ "$verb" = uncordon ] && unschedulable=false
  put_node "$node" "{\"unschedulable\": $unschedulable, \"reason\": \"nodes.sh $verb\"}"
}

specs() { on a "$(api_curl driver) $API/v1/nodes"; }
api_read() {  # PATH: an operator API read on stdout, as the scorer (a viewer)
  on a "$(api_curl scorer) $API$1"
}

collect() {
  # a mirror: nothing stale from an earlier run. A dying rank's kill-time rows (NAME.rows_rankK.pt,
  # 2.8 GB of one rank's rows at 70B) stay on their node: the battery reads them there and records
  # the comparison (localise_rows); copied, each collect takes 3.2 min
  for n in $NODES; do rm -rf "$OUT_DIR/$n/out"; mkdir -p "$OUT_DIR/$n"; on "$n" "cd $W && tar --exclude='*.rows_rank*.pt' -cf - out setup.log 2>/dev/null" | tar -C "$OUT_DIR/$n" -xf - 2>/dev/null; done
  log "collected into $OUT_DIR"
}

clocks() {  # every other node's clock against node-a's (clockoffset.py): logged, and appended to $OUT_DIR/clocks.json for the record
  mkdir -p "$OUT_DIR"
  for n in $NODES; do
    [ "$n" = a ] && continue
    # restarted, not reused: a server from an older tree (the UDP probe's) answers nothing the client sends
    # the subcommand through a variable: this shell's own command line must not match the pattern
    # pkill looks for, or pkill kills the shell before it starts the server;
    # the server alone in the background with every stream redirected: a backgrounded `cd && nohup`
    # list runs in a subshell that keeps the session's streams and holds ssh open (40 min)
    on "$n" "m=serve; pkill -f 'clockoffset.py serv[e]'; sleep 0.2; cd $R && (nohup .venv/bin/python bench/harness/twonode/clockoffset.py \$m < /dev/null > /dev/null 2>&1 &)"
  done
  sleep 1
  for n in $NODES; do
    [ "$n" = a ] && continue
    local probe; probe=$(on a "cd $R && .venv/bin/python bench/harness/twonode/clockoffset.py probe $(ip_of $n)")
    log "clock of node-$n against node-a: $probe"
    echo "{\"node\": \"node-$n\", \"against\": \"node-a\", \"when\": \"${1:-}\", \"probe\": $probe}" >> "$OUT_DIR/clocks.json"
  done
}

fabric() {  # the fabric's bandwidth between node-a and every other node: iperf3 (the pre-registration's floor) and an NCCL send (bw.py); $OUT_DIR/fabric.json
  mkdir -p "$OUT_DIR"; : > "$OUT_DIR/fabric.json"
  for n in $NODES; do
    [ "$n" = a ] && continue
    on a "pkill -f 'iperf3 -[s]' 2>/dev/null; true"  # its own call: a shell that also starts the server would match its own line
    on a "mkdir -p $W/out; iperf3 -s -1 -p 5201 -D"; sleep 2
    local ip3; ip3=$(on "$n" "iperf3 -c $AIP -p 5201 -t 8 -P 4 -J 2>/dev/null" | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(round(d['end']['sum_received']['bits_per_second']/1e9,2))
except Exception as e: print('null')")
    log "iperf3 node-$n -> node-a: ${ip3} Gbit/s"
    local nccl="null"
    if [ "$n" = b ]; then  # the NCCL path at the decode, prefill and replication sizes (bw.py), head on a
      on b "mkdir -p $W/out"
      on a "cd $R && mkdir -p $W/out && export PATH=\$HOME/.local/bin:\$PATH && CUDA_VISIBLE_DEVICES=0 timeout 180 .venv/bin/python bench/harness/twonode/bw.py 0 $AIP > $W/out/bw_a.log 2>&1" &
      sleep 2; on b "cd $R && CUDA_VISIBLE_DEVICES=0 timeout 180 .venv/bin/python bench/harness/twonode/bw.py 1 $AIP > $W/out/bw_b.log 2>&1"; wait
      nccl=$(on b "grep -h 'GB/s\|Gbit' $W/out/bw_b.log | tail -3 | tr '\n' ';'" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))")  # rank 1 prints the numbers
      log "NCCL send node-a <-> node-b: $(echo $nccl | cut -c1-200)"
    fi
    echo "{\"node\": \"node-$n\", \"iperf3_gbit_s\": ${ip3:-null}, \"nccl\": ${nccl}}" >> "$OUT_DIR/fabric.json"
  done
}

cache_sync() {  # the serving node's compile cache staged onto the spare's node (decision 0029): CACHE_FROM (b) to CACHE_TO (c ...); digests of both for the record
  # vLLM's torch_compile_cache holds Inductor's autotune results chosen by timing on each
  # machine; a spare that compiled on its own node rounds differently from the rank it
  # replaces. The spare's node gets the serving node's cache, its own
  # kept aside, before the spare launches; every node's (key, digest) lands in $OUT_DIR.
  # Triton's own cache comes with it: the model runner's kernels (input batch, block table, the
  # sampler) compile there, keyed by source and shapes, not by rank, so the serving node's serve any
  # stage the spare takes; without them the take's warm-up compiles cold (15.9 s)
  local from=${CACHE_FROM:-b} to=${CACHE_TO:-c} n C=/home/everserve/.cache/vllm/torch_compile_cache T=/home/everserve/.triton/cache
  for n in $to; do
    on "$n" "[ -d $C ] && mv $C $C.own.\$(date +%s); mkdir -p $C $T" || return 1
    on "$from" "cd $C && tar -cf - ." | on "$n" "cd $C && tar -xf -" || return 1
    on "$from" "mkdir -p $T && cd $T && tar -cf - ." | on "$n" "cd $T && tar -xf -" || return 1
    # Triton's driver helper built in place in the staged cache, off the take's path: node-local,
    # it is staged only when the serving node's copy happened to hold it (triton_helper.py)
    on "$n" "$ENV && .venv/bin/python bench/harness/twonode/triton_helper.py $C" > "$OUT_DIR/triton_helper_$n.json" 2>> "$OUT_DIR/triton_helper_$n.err" || { log "node-$n: Triton's driver helper not built in the staged cache: $(tail -c 300 "$OUT_DIR/triton_helper_$n.json")"; return 1; }
    [ -s "$OUT_DIR/triton_helper_$n.err" ] || rm -f "$OUT_DIR/triton_helper_$n.err"
  done
  for n in $from $to; do
    on "$n" "cd $R && .venv/bin/python bench/harness/twonode/cache_digest.py" > "$OUT_DIR/cache_digest_$n.json" 2>/dev/null
  done
  log "compile cache staged from node-$from onto node-$to: $(for n in $from $to; do echo -n "$n=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(','.join(c['key']+':'+c['digest'][:12] for c in d['caches']))" "$OUT_DIR/cache_digest_$n.json" 2>/dev/null) "; done)"
}

down_control() { for n in $NODES; do on "$n" "kill \$(pgrep -f 'everserve-controlle[r]|everserve-stor[e]') 2>/dev/null; true"; done; }
# the engine's processes: the drill, the node role, the spare (vLLM retitles it EverserveSpare and
# its workers VLLM::Worker). The drill and the node role are interrupted; a spare is its daemon's
# child, stopped by its daemon (SIGTERM, then a kill after the daemon's grace), so the daemon is
# stopped and not the spare: a spare stopped under it is launched again a second later
ENGINE_MAIN='adapter_dril[l].py|everserve_vllm.node_mai[n]'
DAEMON='everserve-controlle[r] node'
ENGINE_ALL="$ENGINE_MAIN|everserve_vllm.spare_mai[n]|EverserveSpar[e]|VLLM::[W]orker"
down() {
  # the engine's main processes interrupted on every node (SIGINT: their exits run) and the daemons
  # stopped (SIGTERM: each stops its spares as it leaves); every rank's
  # process and the spare asked for their stacks through the coordinator (POST
  # /v1/engines/rig/stacks, the
  # product's way: the daemon holding each sends it SIGUSR1 and the stacks go to its own log) at
  # half the grace if any is still there; whatever remains at its end killed. Each node's exit
  # kind (clean, killed, none) is kept on the node for the start's written_since read.
  local n k half=$(( ${DOWN_GRACE:-20} / 2 )) left asked=""
  for n in $NODES; do
    on "$n" "mkdir -p $W/marks; if pgrep -f '$ENGINE_ALL' > /dev/null; then echo running > $W/marks/exit; kill -INT \$(pgrep -f '$ENGINE_MAIN') 2>/dev/null; kill -TERM \$(pgrep -f '$DAEMON') 2>/dev/null; else echo none > $W/marks/exit; fi; true" < /dev/null &
  done
  wait
  local t0; t0=$(date +%s)
  while :; do
    left=""; for n in $NODES; do on "$n" "pgrep -f '$ENGINE_ALL|$DAEMON' > /dev/null" < /dev/null && left="$left $n"; done
    [ -z "$left" ] && break
    [ $(( $(date +%s) - t0 )) -ge "${DOWN_GRACE:-20}" ] && break
    if [ -z "$asked" ] && [ $(( $(date +%s) - t0 )) -ge "$half" ]; then asked=1; stacks_asked "still running on$left"; fi
    sleep 1
  done
  for n in $NODES; do
    # killed: a process still there now, or a spare its daemon killed after its own grace (the
    # daemon's log since its last launch says so), which would otherwise read as stopped
    on "$n" "since=\$(grep -n '^==== ' $W/out/daemon.log 2>/dev/null | tail -1 | cut -d: -f1)
      if pgrep -f '$ENGINE_ALL' > /dev/null; then kill -9 \$(pgrep -f '$ENGINE_ALL') 2>/dev/null; echo killed > $W/marks/exit
      elif [ -n \"\$since\" ] && tail -n +\$since $W/out/daemon.log | grep -q 'ignored SIGTERM'; then echo killed > $W/marks/exit
      elif [ \"\$(cat $W/marks/exit)\" = running ]; then echo clean > $W/marks/exit; fi; true" < /dev/null &
  done
  wait
  down_control; log "nodes $NODES down: $(for n in $NODES; do echo -n "$n=$(on "$n" "cat $W/marks/exit 2>/dev/null") "; done)"
}

stacks_asked() {  # WHY: every rank's process and the spare standing for any position on node-c, through the coordinator; the answers into $OUT_DIR/stop_stacks.log
  local r q
  {
    echo "$(date -u +%FT%TZ) $1"
    for r in $(seq 0 $(( TP * PP - 1 ))); do
      q="{\"rank\": $r}"; echo "$q -> $(on a "$(api_curl driver) -X POST $API/v1/engines/rig/stacks -H 'Content-Type: application/json' -d '$q'" < /dev/null)"
    done
    q="{\"node\": \"node-c\", \"device\": 0}"; echo "$q -> $(on a "$(api_curl driver) -X POST $API/v1/engines/rig/stacks -H 'Content-Type: application/json' -d '$q'" < /dev/null)"
  } >> "$OUT_DIR/stop_stacks.log" 2>&1
  log "stacks asked of every rank's process and the spare ($1)"
}

page_cache() {  # NAME: how much of the checkpoint each serving node holds in its page cache (fincore), into $OUT_DIR/page_cache_NAME_<node>.json
  local name=$1 n
  for n in a b; do
    on "$n" "fincore --json --bytes $MODEL_DIR/*.safetensors" > "$OUT_DIR/page_cache_${name}_$n.json" 2>/dev/null
  done
  log "$name: the checkpoint in page cache: $(for n in a b; do echo -n "$n=$(python3 -c "import json,sys; f=json.load(open(sys.argv[1]))['fincore']; c=sum(int(x['res']) for x in f); s=sum(int(x['size']) for x in f); print(f'{c}/{s} bytes')" "$OUT_DIR/page_cache_${name}_$n.json" 2>/dev/null || echo unread) "; done)"
}

# a start's files (written_since.py): a mark on every node before it, the reads after; the
# excluded prefixes are the run's own logs, outputs and marks, not what the engine writes
mark() {
  local n
  for n in $NODES; do on "$n" "mkdir -p $W/marks && rm -f $W/marks/exit && echo \$(( \$(cat $W/marks/starts 2>/dev/null || echo 0) + 1 )) > $W/marks/starts && touch $W/marks/start"; done
}
written() {  # NAME PHASE [MARK]: every node's files written since the mark (the start's, or MARK) into $OUT_DIR/written_since_NAME_PHASE_<node>.json
  local name=$1 phase=$2 mark=${3:-start} n
  for n in $NODES; do
    on "$n" "cd $R && .venv/bin/python bench/harness/twonode/written_since.py $W/marks/$mark --exclude $W/out $W/run $W/marks $W/cores $W/disc $W/setup.log /var/log --start \$(cat $W/marks/starts 2>/dev/null || echo 0) --takes \$(cat $W/marks/takes 2>/dev/null || echo 0) \$([ -f $W/marks/exit ] && echo --exit \$(cat $W/marks/exit))" > "$OUT_DIR/written_since_${name}_${phase}_$n.json" 2> "$OUT_DIR/written_since_${name}_${phase}_$n.err" < /dev/null &
  done
  wait
  for n in $NODES; do [ -s "$OUT_DIR/written_since_${name}_${phase}_$n.err" ] || rm -f "$OUT_DIR/written_since_${name}_${phase}_$n.err"; done
  log "$name $phase: written since the start: $(for n in $NODES; do echo -n "$n=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d['files']} files {d['bytes']} bytes (start {d['start']}, takes {d['takes']}, exit {d['exit']})\")" "$OUT_DIR/written_since_${name}_${phase}_$n.json" 2>/dev/null) "; done)"
}
idle() {  # SECONDS: what every node writes over an interval with no start, the baseline a start's read is set against
  local s=${1:-60} n
  for n in $NODES; do on "$n" "mkdir -p $W/marks && touch $W/marks/idle"; done
  sleep "$s"
  written idle baseline idle
}
took() { on c "mkdir -p $W/marks && echo \$(( \$(cat $W/marks/takes 2>/dev/null || echo 0) + 1 )) > $W/marks/takes"; }

case "${1:-}" in
  setup|update|certs|control|headless|collect|down|fabric|cache_sync|mark|took|specs|coord_restart) "$1" ;;
  drain) shift; drain "$@" ;;
  cordon|uncordon) cordon "$@" ;;
  page_cache) shift; page_cache "$@" ;;
  idle) shift; idle "$@" ;;
  written) shift; written "$@" ;;
  clocks) shift; clocks "$@" ;;
  lose) shift; lose "$@" ;;
  kill) shift; kill_worker "$@" ;;
  stacks) shift; stacks "$@" ;;
  api_read) shift; api_read "$@" ;;
  drill) shift; drill "$@" ;;
  *) echo "usage: nodes.sh setup|update|certs|control|headless|drill NAME ARGS|lose NAME|kill NAME RANK|drain [NODE [LEAD_MS]]|cordon NODE|uncordon NODE|specs|coord_restart|collect|down|mark|written NAME PHASE|took|page_cache NAME|idle SECONDS" >&2; exit 2 ;;
esac
