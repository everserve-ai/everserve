#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The adapter-port run, ON the pod, from a checkout of this repository.
# Usage: bash bench/harness/adapter_port.sh setup|build|models|smoke|stock|adapter|replace|notice|notice_ref|all|summary|cleanup
# Every adapter run gets a fresh everserve-store and everserve-controller: both fence by epoch,
# and every new engine starts at epoch 0.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
W=${W:-/workspace}
M=$W/models
OUT=${OUT:-$W/out}
# the engine as the adapter pins it (adapters/vllm/PIN: the v0.31.0 release, decision 0041)
WHEEL="vllm==0.31.0"
HF_MODEL=${HF_MODEL:-Qwen/Qwen3-30B-A3B}   # a frontier-scale run sets a 70B-class checkpoint
MODEL=${MODEL:-$M/$(basename "$HF_MODEL")}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
# the coordinator's policy; a smaller rig (bench/harness/policies/rig-2gpu.json) sets its own
POLICY=${POLICY:-$R/docs/examples/policy-rig-4gpu.json}
STORE_BUDGET=${STORE_BUDGET:-8000000000}   # the 16K-context run needs more than the 8 GB default
BLOCKS=${BLOCKS:-1300}
BLOCK_SIZE=${BLOCK_SIZE:-16}
SOCK=/tmp/es-node.sock
MSOCK=/tmp/es-mirror.sock
MDATA=/dev/shm/everserve
DISC=/tmp/es-discovery
DAEMON_METRICS=127.0.0.1:9108
COORD_METRICS=127.0.0.1:9107
CSOCK=/tmp/es-coord.sock
CLEASE=/tmp/es-coord.lease
STORE_METRICS=127.0.0.1:9109
mkdir -p "$M" "$OUT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# peer-to-peer stays on; the topology goes in the run log
# diagnostics for the drill: NCCL warnings and Python stacks on a fatal signal
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} PYTHONFAULTHANDLER=1
# Fixed-order all-reduces in every arm, stock included. Token equality after a fault is checked
# against another run, so the engine must give the same tokens twice. On Hopper vLLM's dispatch
# puts FlashInfer's all-reduce first ("does not provide a fixed reduction order", vLLM's
# cuda_communicator.py), and torch symmetric-memory multimem and NCCL NVLS reduce in the NVSwitch;
# two identical stock runs on 5x H100 differ. These are the backends vLLM's own
# batch-invariant mode turns off. A100 has none of the three, and its runs were identical.
# ALLREDUCE_DEFAULTS=1 leaves vLLM's and NCCL's own choice, for measuring what the settings cost
if [ "${ALLREDUCE_DEFAULTS:-0}" != 1 ]; then
  export VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 NCCL_NVLS_ENABLE=0
fi
[ -f "$W/compat.env" ] && source "$W/compat.env"
export PATH="$R/.venv/bin:$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"  # the venv is active: ninja, for the kernel build
[ -d /usr/local/cuda-13.0 ] && export CUDA_HOME=/usr/local/cuda-13.0
PY="$R/.venv/bin/python"

setup() {
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  nvidia-smi topo -m | head -7; free -g | head -2; nproc; df -h /dev/shm "$W" | tail -2
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
  (cd "$R" && rustup show active-toolchain || rustup toolchain install "$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)")
  (cd "$R" && uv sync --all-packages)
  (cd "$R" && uv pip install --python "$PY" "$WHEEL")
  # a host driver older than the wheel's CUDA 13.0 needs forward compatibility (harness week four)
  if ! "$PY" -c "import torch; torch.ones(1, device='cuda'); import vllm._C" 2>/dev/null; then
    apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq cuda-compat-13-0 >/dev/null 2>&1
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}' > "$W/compat.env"
    # keep the compatibility libraries only if CUDA works with them: on a driver newer than the
    # toolkit they break it (driver 595)
    if (source "$W/compat.env"; "$PY" -c "import torch; torch.ones(1, device='cuda')" 2>/dev/null); then
      echo "driver older than CUDA 13: cuda-compat-13-0 installed"
    else
      echo "" > "$W/compat.env"; echo "cuda-compat-13-0 does not help on this driver: not used"
    fi
  else echo "" > "$W/compat.env"; fi
  source "$W/compat.env"
  # the kernels build against the engine's torch (cu130), so they need a CUDA 13 toolkit
  if ! command -v /usr/local/cuda-13.0/bin/nvcc >/dev/null; then
    apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq cuda-toolkit-13-0 >/dev/null 2>&1
  fi
  export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
  (cd "$R" && "$PY" -m everserve.kernels)                 # build once; workers only load it
  nvidia-smi topo -m > "$OUT/topology.txt"
  "$PY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"
  (cd "$R" && "$PY" -m pytest -q adapters/vllm/tests core/tests -m "not slow")
}

build() {
  # etcd-client's build script generates etcd's gRPC types with protoc (the lease's etcd backend)
  command -v protoc >/dev/null || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq protobuf-compiler >/dev/null 2>&1; }
  command -v protoc >/dev/null || { echo "protoc is not installed (protobuf-compiler)" >&2; return 1; }
  (cd "$R" && cargo build --release -p everserve-store -p everserve-controller)
  ls -la "$R/target/release/everserve-store" "$R/target/release/everserve-controller"
}

models() {
  "$R/.venv/bin/hf" download "$HF_MODEL" --local-dir "$MODEL" --include "*.json" --include "*.safetensors" \
    --include "merges.txt" --include "vocab.json"
  du -sh "$MODEL"; cat "$MODEL"/*.safetensors > /dev/null      # warm the page cache
}

daemons_up() {   # $1: log prefix
  daemons_down
  rm -rf "$DISC" "$SOCK" "$MSOCK" "$MDATA" "$CSOCK" "$CLEASE"
  # rows move through shared memory under $MDATA (decision 0011); /dev/shm must hold the tables
  # plus the staging files, so check `df -h /dev/shm` on a new pod and raise --shm-size if small
  "$R/target/release/everserve-store" --socket "$MSOCK" --data-dir "$MDATA" --slots $((BLOCKS * BLOCK_SIZE)) \
    --budget-bytes "${STORE_BUDGET:-8000000000}" --metrics-addr "$STORE_METRICS" > "$OUT/$1.store.log" 2>&1 &
  # the coordinator owns the policy and every epoch (decision 0014); the daemon is its agent
  EVERSERVE_LOG=info "$R/target/release/everserve-controller" coord --policy "${POLICY:-$R/docs/examples/policy-rig-4gpu.json}" \
    --listen "unix:$CSOCK" --lease "$CLEASE" --metrics-addr "$COORD_METRICS" > "$OUT/$1.coord.log" 2>&1 &
  EVERSERVE_LOG=info "$R/target/release/everserve-controller" node --coordinator "unix:$CSOCK" \
    --socket "$SOCK" --discovery-dir "$DISC" --mirror-socket "$MSOCK" --node rig \
    --metrics-addr "$DAEMON_METRICS" > "$OUT/$1.daemon.log" 2>&1 &
  for _ in $(seq 100); do
    [ -S "$CSOCK" ] && [ -S "$SOCK" ] && [ -S "$MSOCK" ] && [ -f "$DISC/rig.json" ] \
      && curl -s "http://$DAEMON_METRICS/metrics" | grep "everserve_coordinator_linked 1" >/dev/null && return 0
    sleep 0.1
  done
  echo "daemons did not come up"; return 1
}

gpu_free() {   # vLLM renames its workers, so a name match misses them: kill whatever holds a GPU
  pkill -f adapter_drill.py 2>/dev/null; sleep 2
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  for _ in $(seq 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 1; done
  echo "GPUs still held: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)"
}

daemons_down() {
  pkill -f "target/release/everserve-controller" 2>/dev/null; pkill -f "target/release/everserve-store" 2>/dev/null
  sleep 0.5
}

drill() {   # $1: run name, then driver arguments
  local name=$1; shift
  gpu_free
  echo "$(date -u +%FT%TZ) start $name: $*"
  if [ "$1" != "--arm" ] || [ "$2" != "stock" ]; then daemons_up "$name" || return 1; fi
  # a hung run must not burn the budget: ten minutes is five times a healthy run of the 30B model
# (RUN_TIMEOUT raises it for a 70B one). Every arm records
# its top-20 logprobs, so a fault run's logits are compared with its no-fault reference's
# (decision 0013) and every arm pays the same for it
  (cd "$R" && timeout -s KILL "$RUN_TIMEOUT" "$PY" bench/harness/adapter_drill.py --model "$MODEL" --blocks "$BLOCKS" \
     --event-dir "$OUT/events_$name" --out "$OUT/$name.json" --logprobs 20 "$@") > "$OUT/$name.log" 2>&1
  local rc=$?
  # the daemons are fresh per run, so their counters describe this run alone
  if [ "$1" != "--arm" ] || [ "$2" != "stock" ]; then
    # change metrics live on the coordinator, worker metrics on the daemon: one file, both texts
    { curl -s "http://$COORD_METRICS/metrics"; curl -s "http://$DAEMON_METRICS/metrics"; } > "$OUT/$name.daemon.metrics.txt"
    curl -s "http://$STORE_METRICS/metrics" > "$OUT/$name.store.metrics.txt"
  fi
  daemons_down
  gpu_free
  rm -f /dev/shm/everserve_* 2>/dev/null
  echo "$(date -u +%FT%TZ) end $name rc=$rc: $(tail -1 "$OUT/$name.log")"
}

smoke()   { drill smoke_replace --arm replace --kill between; }
stock()   { for i in 1 2 3; do drill "stock_$i" --arm stock; done; }
# stock with vLLM's own custom all-reduce where the host runs it; on a failure the run is repeated
# with it off, and the result file says which ran (custom_all_reduce)
# setup for stock runs only (the H100 determinism check): the engine and the model, no kernels,
# no toolchain, no tests
setup_stock() {
  nvidia-smi --query-gpu=index,name,driver_version --format=csv; nvidia-smi topo -m | head -6
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  (cd "$R" && uv sync --all-packages && uv pip install --python "$PY" "$WHEEL")
  if ! "$PY" -c "import torch; torch.ones(1, device='cuda'); import vllm._C" 2>/dev/null; then
    apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq cuda-compat-13-0 >/dev/null 2>&1
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}' > "$W/compat.env"
  else echo "" > "$W/compat.env"; fi
}

stock_default() {
  for i in 1 2 3; do
    drill "stock_$i" --arm stock --custom-all-reduce default
    if [ ! -s "$OUT/stock_$i.json" ]; then
      echo "$(date -u +%FT%TZ) stock_$i failed with the custom all-reduce on: repeating it off"
      mv "$OUT/stock_$i.log" "$OUT/stock_${i}_custom_ar_failed.log" 2>/dev/null
      drill "stock_$i" --arm stock
    fi
  done
}
adapter() { for i in 1 2 3; do drill "adapter_$i" --arm adapter; done; }
replace() {
  for i in 1 2 3; do drill "replace_between_$i" --arm replace --kill between; done
  for i in 1 2 3; do drill "replace_mid_$i" --arm replace --kill mid; done
}
notice()  { for i in 1 2 3; do drill "notice_$i" --arm notice --notice-steps 300; done; }
all()     { stock; adapter; replace; notice; summary; }
# Post-hoc, after the pre-registered runs (see run.log): the REPLACE arms again after the abort-order
# fix, and the notice arm with 600 generated tokens. At 200 tokens the sessions finished ~3.5 s after
# the notice, before its 10 s deadline, so the pre-registered notice runs never exercised the swap.
LONG=(--max-tokens 601 --max-model-len 2688)
posthoc_replace() {
  for i in 1 2 3; do drill "posthoc_replace_between_$i" --arm replace --kill between; done
  for i in 1 2 3; do drill "posthoc_replace_mid_$i" --arm replace --kill mid; done
}
posthoc_notice() { for i in 1 2 3; do BLOCKS=1400 drill "posthoc_notice_$i" --arm notice --notice-steps 300 "${LONG[@]}"; done; }
smoke_notice()   { BLOCKS=1400 drill smoke_notice --arm notice --notice-steps 300 "${LONG[@]}"; }

# bench/runs/PLANNED/p1-async-commit-4gpu, arms in the pre-registered order
nomirror()     { for i in 1 2 3; do drill "adapter_nomirror_$i" --arm adapter_nomirror; done; }
async_notice() { for i in 1 2 3; do BLOCKS=1400 drill "notice_$i" --arm notice --notice-steps 300 "${LONG[@]}"; done; }
# the notice arm's no-fault reference: same length and model length, so its sessions' tokens after
# the swap are compared as a REPLACE run's are with adapter_1 (summarize_adapter_port.reference_for)
notice_ref()   { BLOCKS=1400 drill notice_ref --arm adapter "${LONG[@]}"; }
summary() { (cd "$R" && "$PY" bench/harness/summarize_adapter_port.py "$OUT" "$OUT/results.json"); }
cleanup() { rm -rf "$M" ~/.cache/huggingface /dev/shm/everserve_*; df -h "$W" | tail -1; }

"$@"
