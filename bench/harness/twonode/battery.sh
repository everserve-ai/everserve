#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The two-node battery (bench/runs/PLANNED/p2-two-node/PREREGISTRATION.md, "What runs"), on hosts
# nodes.sh already holds (deployed, set up, the model downloaded): the fabric's bandwidth first,
# refused below the floor; then the arms in the pre-registration's order, each measured arm as
# trials with a fresh control per trial; one summary.json per trial (summary.py: the figures
# beside their predictions, the verdict, the clocks) and one arm.json per arm (median and worst
# case); the clocks probed at the start and the end. Nothing the site reads is computed from logs.
#
# Settings (environment): NODES ("a b c"), AIP/BIP/CIP, TWONODE_SCRATCH (the ssh wrappers),
# SPEND_SINCE, SPEND_RATE, SPEND_GPUS (the rental's start, its billed USD per GPU-hour and its
# GPUs, as nebius.py budget holds them: each trial's line then carries the spend so far),
# RUN_DIR (where the record goes), PREDICTIONS (the predictions.json beside the run's
# pre-registration, which every summary is scored against), MODEL, MODEL_DIR, TP, PP, MEMBERS ("a:4 b:4"), SPARES ("c:4"),
# N (sessions, 8), PROMPT_LEN (2048), MAX_TOKENS (3000), MAX_MODEL_LEN, BLOCKS, SLOTS,
# STORE_BUDGET, TRIALS_FAULT (5), TRIALS_SINGLE_DEATH (TRIALS_FAULT), TRIALS_OTHER (3),
# TRIALS_STOCK_RESTART (3), TRIALS_ADAPTER (2), FLOOR_GBIT (25), LONG_CTX (65536),
# LEAD_MS (30000, the drain's lead), ARMS (the arms to run, in order; default: all nine), RESIDENT
# (1: the spares hold the whole model in host memory and load the shard at the take, 0028/0030).
# The repeat arm (not in the default list): the same sessions twice with no fault, token for token;
# a battery whose ARMS name it stops there unless the two agree. Every start on every node is
# marked and its files read (written_since.py) while it runs and once it is stopped.
# A smoke of the script itself: the three-VM TP2 x PP2 shape with MEMBERS="a:2 b:2" SPARES="c:2".
set -u
cd "$(dirname "$0")/../../.."
S=${TWONODE_SCRATCH:?the scratch directory holding ssh_a, ssh_b, ...}
RUN_DIR=${RUN_DIR:?where the record goes}
PREDICTIONS=${PREDICTIONS:?the predictions.json beside the pre-registration of the run}
[ -s "$PREDICTIONS" ] || { echo "no predictions file at $PREDICTIONS" >&2; exit 2; }
MODEL=${MODEL:?the model HF id}; MODEL_DIR=${MODEL_DIR:-/workspace/models/$(basename "$MODEL")}
TP=${TP:?}; PP=${PP:?}; MEMBERS=${MEMBERS:?node:count ...}; SPARES=${SPARES:?node:count}
N=${N:-8}; PROMPT_LEN=${PROMPT_LEN:-2048}; MAX_TOKENS=${MAX_TOKENS:-3000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((PROMPT_LEN + MAX_TOKENS + 256))}
BLOCKS=${BLOCKS:-$(( (N * MAX_MODEL_LEN + 15) / 16 + 64 ))}
SLOTS=${SLOTS:-$((BLOCKS * 16))}
TRIALS_FAULT=${TRIALS_FAULT:-5}; TRIALS_OTHER=${TRIALS_OTHER:-3}; TRIALS_STOCK_RESTART=${TRIALS_STOCK_RESTART:-3}
# each pre-registration states its arms' counts (for instance single death x3, adapter x1)
TRIALS_SINGLE_DEATH=${TRIALS_SINGLE_DEATH:-$TRIALS_FAULT}; TRIALS_ADAPTER=${TRIALS_ADAPTER:-2}
FLOOR_GBIT=${FLOOR_GBIT:-25}; LONG_CTX=${LONG_CTX:-65536}; LEAD_MS=${LEAD_MS:-30000}
ARMS=${ARMS:-"stock adapter mirror_off drain node_fault single_death notice_ref long_context shard_copy"}
# the fault arms and their controls both carry: every rank's K/V rows digested per position at
# the end of the run (compared offline, position by position) and top-2 logprobs
FAULT_READS=${FAULT_READS-"--kv-digest --logprobs 2"}
export NODES=${NODES:-"a b c"} MODEL MODEL_DIR TP PP N PROMPT_LEN MAX_MODEL_LEN BLOCKS SLOTS LEAD_MS
export STORE_BUDGET=${STORE_BUDGET:-8000000000} MIRROR_DIR=${MIRROR_DIR:-/dev/shm/mirror}
# where node-b's rows replicate: the spare node on the preloaded shape, the sibling on the resident
[ -n "${RESIDENT:-}" ] && REPLICA_NODE=${REPLICA_NODE:-a} || REPLICA_NODE=${REPLICA_NODE:-c}
export POLICY=bench/harness/policies/battery.json RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
export SERVE_GPUS_A SERVE_GPUS_B SPARE_GPU_A SPARE_RANK
N_=bench/harness/twonode/nodes.sh
PY=.venv/bin/python
mkdir -p "$RUN_DIR"; L=$RUN_DIR/run.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
spend() {  # the rental's spend so far, reckoned as nebius.py budget reckons it; nothing unset
  [ -n "${SPEND_RATE:-}" ] || return 0
  $PY -c "import sys, time; sys.path.insert(0, 'bench/harness'); import nebius; print(f', \${nebius.spend_usd(float(sys.argv[1]), time.time(), int(sys.argv[2]), float(sys.argv[3])):.2f} spent')" "$SPEND_SINCE" "$SPEND_GPUS" "$SPEND_RATE"
}
COORD_METRICS=7401

# the shape: the policy for it, and the GPUs each node serves (nodes.sh's settings)
count_of() { echo "$1" | tr ' ' '\n' | grep "^$2:" | cut -d: -f2; }
gpus() { python3 -c "print(','.join(str(i) for i in range($1)))"; }  # not seq: BSD seq appends the separator
A_GPUS=$(count_of "$MEMBERS" a); B_GPUS=$(count_of "$MEMBERS" b); C_GPUS=$(count_of "$SPARES" c)
[ -n "$A_GPUS" ] && [ -n "$B_GPUS" ] && [ -n "$C_GPUS" ] || { log "MEMBERS and SPARES name node-a, node-b and node-c"; exit 2; }
SERVE_GPUS_A=$(gpus "$A_GPUS"); SERVE_GPUS_B=$(gpus "$B_GPUS"); SPARE_GPU_A=0; SPARE_RANK=$A_GPUS
NODE_B_FIRST_RANK=$A_GPUS
# the store's table numbering for node-b's first rank: (layer - its stage's first layer) * heads + head; the row
# is one head slot's K and V for one token (Qwen3-1.7B: 28 layers, 8 KV heads, 128 dims, bf16)
NODE_B_FIRST_LAYER=${NODE_B_FIRST_LAYER:-14}; KV_HEADS=${KV_HEADS:-8}; KV_ROW_BYTES=${KV_ROW_BYTES:-512}
# the long-context arm serves one session of LONG_CTX tokens: its own model length, blocks and
# policy (the spare's launch carries them), swapped in for its trials and back after
LONG_MAX_MODEL_LEN=$((LONG_CTX + 1024)); LONG_BLOCKS=$(( LONG_MAX_MODEL_LEN / 16 + 64 )); LONG_SLOTS=$((LONG_BLOCKS * 16))
write_policy() {  # standard | long: the policy for the shape, synced to the nodes
  if [ "$1" = long ]; then
    $PY bench/harness/twonode/policy.py --tp "$TP" --pp "$PP" --members $MEMBERS --spares $SPARES ${RESIDENT:+--resident} --model "$MODEL_DIR" --max-model-len "$LONG_MAX_MODEL_LEN" --blocks "$LONG_BLOCKS" --max-num-seqs 1 --prompt-len "$LONG_CTX" --policy-path "$POLICY" --out "$POLICY" || return 1
  else
    $PY bench/harness/twonode/policy.py --tp "$TP" --pp "$PP" --members $MEMBERS --spares $SPARES ${RESIDENT:+--resident} --model "$MODEL_DIR" --max-model-len "$MAX_MODEL_LEN" --blocks "$BLOCKS" --max-num-seqs "$N" --prompt-len "$PROMPT_LEN" --policy-path "$POLICY" --out "$POLICY" || return 1
  fi
  # the policy is generated, not tracked: nodes.sh update syncs tracked files, so it goes by hand,
  # and so do the row readers until they are tracked
  local n f; for n in $NODES; do
    $S/ssh_$n "cat > /workspace/everserve/$POLICY" < "$POLICY" || return 1
    for f in store_rows.py kv_rows.py cache_digest.py; do $S/ssh_$n "cat > /workspace/everserve/bench/harness/twonode/$f" < "bench/harness/twonode/$f" || return 1; done
  done
  return 0
}
write_policy standard || exit 2
log "battery on $MODEL, TP$TP x PP$PP, members $MEMBERS, spares $SPARES, $N sessions x $PROMPT_LEN prompt, $MAX_TOKENS tokens, blocks $BLOCKS, slots $SLOTS, long context $LONG_CTX, store budget ${STORE_BUDGET:-8000000000}, node-b's first layer $NODE_B_FIRST_LAYER, KV heads $KV_HEADS, row bytes $KV_ROW_BYTES; arms: $ARMS; trials $TRIALS_FAULT (node fault) / $TRIALS_SINGLE_DEATH (single death) / $TRIALS_OTHER (other) / $TRIALS_ADAPTER (adapter); commit $(git rev-parse --short "${DEPLOY_COMMIT:?the commit this deploy runs}"), checkpoint revision $($S/ssh_a "cat /workspace/MODEL_REVISION" 2>/dev/null)"
cp "$POLICY" "$RUN_DIR/policy.json"
export OUT_DIR=$RUN_DIR/setup; mkdir -p "$OUT_DIR"  # the fabric and clock stages write here (on a fresh RUN_DIR without it the fabric stage leaves no file and refuses)
$N_ update >> "$L" 2>&1 || { log "update failed"; exit 2; }
write_policy standard || { log "the policy did not reach the nodes"; exit 2; }

# the fabric first: below the floor the run proves nothing about the mechanism, and says so
$N_ fabric >> "$L" 2>&1
MIN_GBIT=$(python3 -c "
import json,sys
vals=[json.loads(l).get('iperf3_gbit_s') for l in open('$RUN_DIR/setup/fabric.json') if l.strip()]
vals=[v for v in vals if isinstance(v,(int,float))]; print(min(vals) if vals else 0)")
log "fabric: slowest link to node-a $MIN_GBIT Gbit/s (floor $FLOOR_GBIT)"
if python3 -c "import sys; sys.exit(0 if float('$MIN_GBIT') >= float('$FLOOR_GBIT') else 1)"; then :; else
  cat > "$RUN_DIR/REFUSED.md" <<EOF
# Refused: the fabric is below the pre-registration's floor

The slowest link to node-a measured $MIN_GBIT Gbit/s by iperf3 (\`setup/fabric.json\`); the floor is
$FLOOR_GBIT Gbit/s (PREREGISTRATION.md, Platform). Below it the run proves the platform, not the
mechanism, so no arm was started. $(date -u +%FT%TZ)
EOF
  log "REFUSED: fabric $MIN_GBIT Gbit/s under the floor $FLOOR_GBIT; no arm started (REFUSED.md)"
  exit 3
fi
$N_ clocks start >> "$L" 2>&1
cp "$RUN_DIR/setup/clocks.json" "$RUN_DIR/clocks.json" 2>/dev/null

ready_spare() {  # the spares register once pre-formed with the engine: wait, then the record of it
  local want=$1 k
  for k in $(seq 1 60); do
    $N_ api_read /v1/engines/rig 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin)['status'];sys.exit(0 if len(d.get('spares') or [])+len(d.get('anySpares') or [])>=$want else 1)" 2>/dev/null && break
    sleep 5
  done
  # the Engine object (decision 0042), its readiness under .status; the file keeps its name
  $N_ api_read /v1/engines/rig > "$OUT_DIR/readiness_before.json" 2>/dev/null
  log "readiness: $(cut -c1-300 "$OUT_DIR/readiness_before.json")"
  # the pool and the membership have to be whole before a kill: a spare the coordinator did not
  # admit, or a rank it does not list, is the finding, not a trial (a kill with no spare in the
  # pool measures nothing)
  python3 -c "
import json,sys;d=json.load(open('$OUT_DIR/readiness_before.json'))['status']
# a spare standing for any position (decision 0030) counts toward the pool as one for a rank does
spares=len(d.get('spares') or [])+len(d.get('anySpares') or []); members=len(d.get('members') or []); world=d.get('worldSize')
print('pool', spares, 'of', $want, 'wanted (any position:', len(d.get('anySpares') or []), '); members', members, 'of', world)
sys.exit(0 if spares >= $want and members == world else 1)" >> "$L" 2>&1 || { log "the spare pool or the membership is short (see readiness); the trial is not run"; return 1; }
  python3 -c "
import json,sys;d=json.load(open('$OUT_DIR/readiness_before.json'))['status'];m={x['rank']:x.get('replicatesTo') or [] for x in d.get('members',[])}
bad=[r for r in m if r>=$NODE_B_FIRST_RANK and not m[r]]
print('replication targets', m); sys.exit(1 if bad else 0)" >> "$L" 2>&1 || { log "node-b's ranks have no replica target; the trial is not run"; return 1; }
  # decision 0029: every member of the engine, the spare included, registered with the compile
  # cache the first member loaded (key and digest); a trial on an inconsistent engine is not run
  python3 -c "
import json,sys;d=json.load(open('$OUT_DIR/readiness_before.json'))['status']
pairs={x['rank']:((x.get('compileCache') or {}).get('key'),((x.get('compileCache') or {}).get('digest') or '')[:12]) for x in d.get('members',[])}
print('compile key', d.get('compileKey'), 'consistent', d.get('compileCacheConsistent'), 'per rank', pairs, 'spares', {s.get('rank'):(((s.get('compileCache') or {}).get('key') or '')[:12], ((s.get('compileCache') or {}).get('digest') or '')[:12]) for s in d.get('spares',[])})
sys.exit(0 if d.get('compileCacheConsistent', True) or '${CACHE_CHECK:-1}' == '0' else 1)" >> "$L" 2>&1 || { log "the engine's members registered with different compile caches; the trial is not run"; return 1; }
}

store_metrics() {  # every store's replica lag, drops and refusals over the trial (a lost node's store reads nothing)
  local n
  for n in $NODES; do
    $S/ssh_$n "curl -s http://127.0.0.1:9109/metrics" 2>/dev/null | sed "s/^/$n /"
  done | python3 -c "
import sys,re,json
lag=[];drop=0;refused={};read=set()
for line in sys.stdin:
    node,_,line=line.partition(' ')
    if line.startswith('#'): continue
    m=re.match(r'(\w+)(\{[^}]*\})?\s+([0-9.e+-]+)', line)
    if not m: continue
    read.add(node)
    name,labels,val=m.group(1),m.group(2) or '',float(m.group(3))
    if name=='everserve_store_replica_lag_steps': lag.append(val)
    if name=='everserve_store_replication_dropped_total': drop+=val
    if name=='everserve_store_refusals_total':
        r=re.search(r'reason=\"([^\"]*)\"', labels); k=r.group(1) if r else ''
        refused[k]=refused.get(k,0)+val
print(json.dumps({'replica_lag_max_steps': max(lag) if lag else None, 'replication_dropped_total': drop, 'refusals_total': refused, 'stores_read': sorted(read)}))" > "$OUT_DIR/store_metrics.json" 2>/dev/null
}

run_drill() {  # NAME, drill args...: the engine up on both nodes, the drill on node-a in the background; DPID is its pid
  local name=$1; shift
  DPID=""
  $N_ mark >> "$L" 2>&1  # every file a start writes is read against this (written_since.py)
  $N_ control >> "$L" 2>&1 || return 1
  $N_ drill "$name" --clock-offsets /workspace/out/clocks.json "$@" >> "$L" 2>&1 &
  DPID=$!  # set here, not echoed from a subshell: the parent must own the job to wait on it
}

fault_action() {  # ARM NAME: what the trial does to node-b once the drill is armed
  local arm=$1 name=$2 k
  for k in $(seq 1 90); do grep -q "engine ready in" "$OUT_DIR/$name.drill.log" 2>/dev/null && break; sleep 5; done
  case $arm in
    node_fault|single_death|long_context) ready_spare "$C_GPUS" || return 1 ;;
    stock_restart) ;;  # stock vLLM has no spare to wait for
    drain)
      # the drill's notice arm sends the notice itself at its kill step (client.notice, decision
      # 0019) and records the swap, with the spare pre-formed by then; waiting here for the
      # spare's registration races that swap (readiness read after it finds the spare consumed
      # and fails the trial). The operator's drain through the coordinator's
      # API is a second path to the same change, to be measured as its own arm later.
      return 0 ;;
  esac
  for k in $(seq 1 180); do $S/ssh_a "test -f /workspace/out/$name.json.armed" 2>/dev/null && break; sleep 1; done
  log "$name armed: $($S/ssh_a "cat /workspace/out/$name.json.armed 2>/dev/null")"
  # stock has no change to commit: node-b lost, the trial waits on the dead engine's exit
  if [ "$arm" = stock_restart ]; then $N_ lose "$name" >> "$L" 2>&1; return 0; fi
  local e0; e0=$(readiness_epoch)  # before the kill: a fast change can commit before a read after it
  case $arm in
    node_fault|long_context) $N_ lose "$name" >> "$L" 2>&1 ;;
    single_death) $N_ kill "$name" "$NODE_B_FIRST_RANK" >> "$L" 2>&1 ;;
    drain) t=$(date +%s.%N); $N_ drain >> "$L" 2>&1; $S/ssh_a "echo '{\"t_notice\": $t, \"mode\": \"drain node-b\", \"lead_ms\": $LEAD_MS}' > /workspace/out/$name.json.kill"; log "$name: node-b drained with lead ${LEAD_MS} ms at $t" ;;
  esac
  # native stacks only for a change that has not committed in 30 s: gdb stops every process it
  # attaches to, and a dump at 4 s freezes the executor mid-change for 3 s (the join answer's leg
  # then reads 3.6 s); a hang is what the stacks are for
  for k in $(seq 1 60); do [ "$(readiness_epoch)" -gt "${e0:-0}" ] 2>/dev/null && break; sleep 0.5; done
  if [ "$(readiness_epoch)" -gt "${e0:-0}" ] 2>/dev/null; then log "$name: committed at epoch $(readiness_epoch); no native stacks taken"
    case $arm in node_fault|single_death|long_context) $N_ took >> "$L" 2>&1 ;; esac
  else log "$name: no change committed in 30 s; native stacks"; $N_ stacks "$name" >> "$L" 2>&1; fi
  # the readiness view once the change committed or the wait gave up, the engine serving: down
  # here is the engine lost in the trial (decision 0032), where the read after the trial reads
  # the head's orderly goodbye at the drill's end
  $N_ api_read /v1/engines/rig > "$OUT_DIR/readiness_committed.json" 2>/dev/null
}

stop_read() {  # NAME: what the start wrote while it ran, its processes stopped, what it wrote by its exit
  # three causes a node's slow first starts allow, split by the two reads: a cache written only at
  # a clean exit (files in the stopped read alone), a key that changes between starts (files in
  # each start's read, a hit later), nothing on disk (no files, the cost held in the process)
  $N_ written "$1" running >> "$L" 2>&1
  $N_ down >> "$L" 2>&1
  $N_ written "$1" stopped >> "$L" 2>&1
  # the logs again after the stop: a process that would not stop wrote its stacks there
  $N_ collect >> "$L" 2>&1
}

readiness_epoch() { $N_ api_read /v1/engines/rig 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['status'].get('epoch', 0))" 2>/dev/null || echo 0; }

localise_rows() {  # NAME: for the rows the drill sampled on the spare, the same rows from the dying rank's kill-time file and both stores, before the control plane goes
  local name=$1 items store_items
  items=$($PY -c "import json,sys; r=json.load(open(sys.argv[1])); print(' '.join(((r.get('exact') or {}).get('spare_rows') or {}).get('rows') or {}))" "$OUT_DIR/result.json")
  # no sampled rows (an exact trial): the first prompt slots of requests 0 and 1 on the first table, the
  # written_step there being the replication signature (1 once the prefill step reached the replica)
  if [ -z "$items" ]; then store_items="0:16 0:17 0:528"; else store_items=$($PY -c "import sys; print(' '.join(f\"{(int(k.split(':')[0]) - $NODE_B_FIRST_LAYER) * $KV_HEADS + int(k.split(':')[1])}:{s}\" for k, s in (it.split('@') for it in sys.argv[1:])))" $items)
  $S/ssh_b "cd /workspace/everserve && .venv/bin/python bench/harness/twonode/kv_rows.py /workspace/out/$name.rows_rank$NODE_B_FIRST_RANK.pt $items" > "$OUT_DIR/rows_kill_time.json" 2>> "$L"
  fi
  # through each store's rows query (its socket), under the store's lock, with written_step
  $S/ssh_b "cd /workspace/everserve && .venv/bin/python bench/harness/twonode/store_rows.py /workspace/run/mirror.sock rig $NODE_B_FIRST_RANK $store_items" > "$OUT_DIR/rows_source_store.json" 2>> "$L"
  # the replica: the spare's node on the preloaded shape, node-b's sibling (node-a) on the resident
  # shape (decision 0030), where node-c holds only what was staged onto it at the change
  $S/ssh_${REPLICA_NODE:-c} "cd /workspace/everserve && .venv/bin/python bench/harness/twonode/store_rows.py /workspace/run/mirror.sock rig $NODE_B_FIRST_RANK $store_items" > "$OUT_DIR/rows_replica_store.json" 2>> "$L"
  [ -n "${RESIDENT:-}" ] && $S/ssh_c "cd /workspace/everserve && .venv/bin/python bench/harness/twonode/store_rows.py /workspace/run/mirror.sock rig $NODE_B_FIRST_RANK $store_items" > "$OUT_DIR/rows_staged_store.json" 2>> "$L"
  log "$name: rows for $(echo $items | wc -w | tr -d ' ') sampled slots pulled from the kill-time file and both stores"
  # an exact trial's kill-time rows are never read; an inexact one's stay on node-b, beside what
  # was pulled from them, for a reading by hand before the node is released
  if [ -z "$items" ]; then
    $S/ssh_b "rm -f /workspace/out/$name.rows_rank$NODE_B_FIRST_RANK.pt"
    log "$name: exact; its kill-time rows removed from node-b"
  fi
}

backup_check() {  # NAME: after a take, each rank's rows at its replica against its holder (decision 0030)
  # rank 1 is held by node-c's spare after the take and replicated to node-a; rank 0 is held by
  # node-a and replicated to node-c: what a second loss of either would restore from
  # the store keeps one table per KV head slot (every layer's row in it): KV_HEADS tables per rank
  local name=$1 r holder replica out="$OUT_DIR/backup_coverage.json"
  local args="--tables $KV_HEADS --table-stride 1 --slots $SLOTS --slot-stride 16"
  echo "{" > "$out"
  for r in 0 1; do
    if [ $r = 1 ]; then holder=c; replica=a; else holder=a; replica=c; fi
    $S/ssh_$holder "cd /workspace/everserve && .venv/bin/python bench/harness/twonode/backup_rows.py /workspace/run/mirror.sock rig $r $args" > "$OUT_DIR/backup_rank${r}_holder.json" 2>> "$L"
    $S/ssh_$replica "cd /workspace/everserve && .venv/bin/python bench/harness/twonode/backup_rows.py /workspace/run/mirror.sock rig $r $args" > "$OUT_DIR/backup_rank${r}_replica.json" 2>> "$L"
    echo "\"$r\": $($PY bench/harness/twonode/backup_rows.py --compare "$OUT_DIR/backup_rank${r}_holder.json" "$OUT_DIR/backup_rank${r}_replica.json" 2>> "$L" || echo null)$([ $r = 0 ] && echo ,)" >> "$out"
  done
  echo "}" >> "$out"
  log "$name: backup after the take: $(tr -d '\n' < "$out" | cut -c1-400)"
}

trial() {  # ARM K: one trial of an arm, its record under RUN_DIR/arms/ARM/trialK
  local arm=$1 k=$2 name=${1}_$2 dpid rc control="" stock=""
  export OUT_DIR=$RUN_DIR/arms/$arm/trial$k; mkdir -p "$OUT_DIR"
  $S/ssh_a "cp /workspace/out/clocks.json /workspace/out/clocks.json.bak 2>/dev/null; true"
  cp "$RUN_DIR/clocks.json" "$OUT_DIR/clocks.json" 2>/dev/null; $S/ssh_a "cat > /workspace/out/clocks.json" < "$RUN_DIR/clocks.json" 2>/dev/null
  log "== $arm trial $k$(spend)"
  case $arm in
    stock)        export STOCK=1; run_drill "$name" --arm stock --max-tokens "$MAX_TOKENS"; dpid=$DPID; unset STOCK ;;
    stock_restart)
      # the baseline the headline is against (decision 0030): node-b lost under stock vLLM, the
      # engine dead with it and relaunched by the harness as an operator's supervisor would; the
      # figure is the time from the loss to the relaunched engine's first token, every session
      # started over from its prompt (stock keeps none)
      export STOCK=1; run_drill "$name" --arm stock --kill external --max-tokens "$MAX_TOKENS"; dpid=$DPID
      [ -n "${dpid:-}" ] || { unset STOCK; log "$name: no drill"; return 1; }
      fault_action stock_restart "$name" || { unset STOCK; kill "$dpid" 2>/dev/null; $S/ssh_a "pkill -9 -f 'adapter_dril[l].py'"; return 1; }
      wait "$dpid"; rc=$?
      # node-a's clock when the dead engine's process was seen gone (one ssh round trip late):
      # the kill to here is stock's detection, here to the relaunch the supervisor's share
      local t_exit; t_exit=$($S/ssh_a 'date +%s.%N')
      log "$name: the stock engine exited $rc after node-b's loss; relaunching it"
      # stock's best case, as it stands: the checkpoint's pages as the relaunch finds them, recorded
      $N_ page_cache "$name" >> "$L" 2>&1
      local t_relaunch; t_relaunch=$($S/ssh_a 'date +%s.%N')  # node-a's clock, as the restart's record
      run_drill "${name}_restart" --arm stock --max-tokens "$MAX_TOKENS"; dpid=$DPID; unset STOCK
      [ -n "${dpid:-}" ] || { log "$name: no relaunch"; return 1; }
      wait "$dpid"
      $N_ collect >> "$L" 2>&1
      [ -s "$OUT_DIR/a/out/${name}_restart.json" ] || { log "$name: the relaunch left no results"; return 1; }
      cp "$OUT_DIR/a/out/${name}_restart.json" "$OUT_DIR/result.json"
      $PY bench/harness/twonode/stock_restart.py --trial "$k" --kill "$OUT_DIR/a/out/$name.json.kill" --exited "$t_exit" --relaunched "$t_relaunch" --result "$OUT_DIR/result.json" --clocks "$OUT_DIR/clocks.json" --predictions "$PREDICTIONS" --out "$OUT_DIR/summary.json" 2>&1 | tee -a "$L"
      stop_read "$name"
      return 0 ;;
    adapter)      run_drill "$name" --arm adapter --max-tokens "$MAX_TOKENS"; dpid=$DPID ;;
    repeat)       run_drill "$name" --arm adapter --max-tokens "$MAX_TOKENS" $FAULT_READS; dpid=$DPID ;;
    mirror_off)   run_drill "$name" --arm adapter_nomirror --max-tokens "$MAX_TOKENS"; dpid=$DPID ;;
    notice_ref)   run_drill "$name" --arm adapter --max-tokens "$MAX_TOKENS"; dpid=$DPID ;;
    drain)        $N_ cache_sync >> "$L" 2>&1; run_drill "$name" --arm notice --kill external --max-tokens "$MAX_TOKENS" --stacks-after 3 $FAULT_READS; dpid=$DPID ;;
    node_fault|single_death)
                  # the spare launches with the control plane: the serving node's compile cache is staged first (0029)
                  $N_ cache_sync >> "$L" 2>&1; run_drill "$name" --arm replace --kill external --max-tokens "$MAX_TOKENS" --stacks-after 3 $FAULT_READS; dpid=$DPID ;;
    long_context) $N_ cache_sync >> "$L" 2>&1; N=1 PROMPT_LEN=$LONG_CTX MAX_MODEL_LEN=$LONG_MAX_MODEL_LEN BLOCKS=$LONG_BLOCKS SLOTS=$LONG_SLOTS run_drill "$name" --arm replace --kill external --max-tokens 512 --stacks-after 3 $FAULT_READS; dpid=$DPID ;;
    shard_copy)   $S/ssh_c "cd /workspace/everserve && .venv/bin/python bench/harness/twonode/shard_copy.py $MODEL_DIR --tp $TP --n 5" > "$OUT_DIR/shard_copy.json" 2>> "$L"; log "shard copy: $(cut -c1-300 "$OUT_DIR/shard_copy.json")"; return 0 ;;
  esac
  [ -n "${dpid:-}" ] || { log "$name: no drill"; return 1; }
  case $arm in drain|node_fault|single_death|long_context) fault_action "$arm" "$name" || { kill "$dpid" 2>/dev/null; $S/ssh_a "pkill -9 -f 'adapter_dril[l].py'"; return 1; } ;; esac
  wait "$dpid"; rc=$?
  store_metrics
  $S/ssh_a "mkdir -p /workspace/cores; mv /workspace/out/core.* /workspace/cores/ 2>/dev/null; true"
  $N_ collect >> "$L" 2>&1
  $N_ api_read /v1/engines/rig > "$OUT_DIR/readiness_after.json" 2>/dev/null
  [ -s "$OUT_DIR/a/out/$name.json" ] || { log "$name left no results (rc $rc)"; return 1; }
  cp "$OUT_DIR/a/out/$name.json" "$OUT_DIR/result.json"
  localise_rows "$name"
  case $arm in node_fault|single_death) [ -n "${RESIDENT:-}" ] && backup_check "$name" ;; esac
  case $arm in
    drain|node_fault|single_death|long_context|repeat) control=$RUN_DIR/arms/$arm/trial$k/control.json
      [ -s "$control" ] || control="" ;;
  esac
  [ -s "$RUN_DIR/arms/stock/trial1/result.json" ] && stock=$RUN_DIR/arms/stock/trial1/result.json
  stop_read "$name"
  # the take's start on the spare's node, read while it ran, is the one scored
  local written="$OUT_DIR/written_since_${name}_running_c.json"; [ -s "$written" ] || written=""
  $PY bench/harness/twonode/summary.py --arm "$arm" --trial "$k" --result "$OUT_DIR/result.json" ${control:+--control "$control"} ${stock:+--stock "$stock"} --store-metrics "$OUT_DIR/store_metrics.json" ${RESIDENT:+--backup "$OUT_DIR/backup_coverage.json"} ${written:+--written "$written"} $([ -s "$BASELINE" ] && echo --baseline "$BASELINE") --stopped "$OUT_DIR"/written_since_${name}_stopped_*.json --engine-gpus $((TP * PP)) --readiness-before "$OUT_DIR/readiness_before.json" --readiness-committed "$OUT_DIR/readiness_committed.json" --readiness-after "$OUT_DIR/readiness_after.json" --cache-position "$OUT_DIR/cache_digest_${CACHE_FROM:-b}.json" --cache-spare "$OUT_DIR/cache_digest_${CACHE_TO:-c}.json" --predictions "$PREDICTIONS" --out "$OUT_DIR/summary.json" 2>&1 | tee -a "$L"
  return 0
}

control_for() {  # ARM K: a fresh control run for a fault trial, the same sessions with no fault
  local arm=$1 k=$2 dpid
  export OUT_DIR=$RUN_DIR/arms/$arm/trial$k/control; mkdir -p "$OUT_DIR"
  log "== $arm trial $k: control"
  if [ "$arm" = long_context ]; then
    N=1 PROMPT_LEN=$LONG_CTX MAX_MODEL_LEN=$LONG_MAX_MODEL_LEN BLOCKS=$LONG_BLOCKS SLOTS=$LONG_SLOTS run_drill control_${arm}_$k --arm adapter --max-tokens 512 $FAULT_READS; dpid=$DPID
  else
    run_drill control_${arm}_$k --arm adapter --max-tokens "$MAX_TOKENS" $FAULT_READS; dpid=$DPID
  fi
  [ -n "${dpid:-}" ] || return 1
  wait "$dpid"
  $N_ collect >> "$L" 2>&1
  [ -s "$OUT_DIR/a/out/control_${arm}_$k.json" ] || { log "control for $arm trial $k left no results"; return 1; }
  cp "$OUT_DIR/a/out/control_${arm}_$k.json" "$RUN_DIR/arms/$arm/trial$k/control.json"
  stop_read "control_${arm}_$k"
}

trial_failed() {  # ARM K: a trial that ended without its record still counts, red, in its arm
  local d=$RUN_DIR/arms/$1/trial$2
  mkdir -p "$d"
  [ -s "$d/summary.json" ] && return 0
  # the error the drill died of, its log's last exception line (say, the pipeline refusal vLLM's
  # worker RPC carries), so nothing is read by hand from the log
  local error; error=$(grep -E "^[A-Za-z_.]*(Error|Exception)[:(]" "$d/${1}_$2.drill.log" 2>/dev/null | tail -1)
  python3 - "$1" "$2" "$d/summary.json" "$(grep "$1 trial $2 failed\|${1}_$2" "$L" | tail -3 | tr '\n' ' ')" "$error" <<'EOF'
import json, sys
arm, trial, out, why, error = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
line = f"the trial failed before its record was written: {why.strip()}"
json.dump({"arm": arm, "trial": trial, "failed": True, "drill_error": error or None, "figures": {},
           "verdict": {"green": False, "line": line}}, open(out, "w"), indent=2)
EOF
}

arm_summary() {  # ARM: median and worst case over the arm's trials, a failed one included, arm.json
  local arm=$1
  python3 - "$RUN_DIR/arms/$arm" <<'EOF' | tee -a "$L"
import glob, json, statistics, sys, os
d = sys.argv[1]
trials = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/trial*/summary.json"))]
out = {"arm": os.path.basename(d), "trials": len(trials), "figures": {}}
names = sorted({n for t in trials for n in (t.get("figures") or {})})
for n in names:
    vals = [t["figures"][n]["value"] for t in trials if n in t.get("figures", {})]
    nums = [v for v in vals if isinstance(v, (int, float))]
    f = {"values": vals, "prediction": trials[0]["figures"][n].get("prediction") if trials else None}
    if nums:
        f.update(median=statistics.median(nums), worst=max(nums))
        f["worst_over_median"] = round(max(nums) / statistics.median(nums), 2) if statistics.median(nums) else None
    out["figures"][n] = f
out["failed"] = [t["trial"] for t in trials if t.get("failed")]
out["green"] = bool(trials) and all((t.get("verdict") or {}).get("green", True) for t in trials)
json.dump(out, open(f"{d}/arm.json", "w"), indent=2, default=str)
print(f"{out['arm']}: {out['trials']} trials ({len(out['failed'])} failed), green {out['green']}, " + ", ".join(f"{n}={f.get('median', f['values'])}" for n, f in out["figures"].items()))
EOF
}

# the baseline every start's read is set against: what the nodes write over a minute with no
# start (the observability agent's files on a Nebius VM), read once before the first arm
export OUT_DIR=$RUN_DIR/setup
$N_ idle "${IDLE_S:-60}" >> "$L" 2>&1
BASELINE=$RUN_DIR/setup/written_since_idle_baseline_c.json
for arm in $ARMS; do
  if [ "$arm" = long_context ]; then write_policy long || { log "the long-context policy failed; the arm is skipped"; continue; }; fi
  case $arm in
    node_fault) trials=$TRIALS_FAULT ;;
    single_death) trials=$TRIALS_SINGLE_DEATH ;;
    stock_restart) trials=$TRIALS_STOCK_RESTART ;;  # the comparison, not scored: three (p3, arm 4)
    drain|long_context) trials=$TRIALS_OTHER ;;
    adapter) trials=$TRIALS_ADAPTER ;;
    *) trials=1 ;;
  esac
  for k in $(seq 1 "$trials"); do
    case $arm in drain|node_fault|single_death|long_context|repeat) control_for "$arm" "$k" || { log "$arm trial $k: no control; the arm stops"; break; } ;; esac
    trial "$arm" "$k" || { log "$arm trial $k failed"; trial_failed "$arm" "$k"; }
  done
  arm_summary "$arm"
  # the repeat (the same sessions twice, no fault) gates every arm after it: a loss arm's exactness
  # means nothing on an engine that does not repeat itself
  if [ "$arm" = repeat ] && ! $PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('green') else 1)" "$RUN_DIR/arms/repeat/arm.json" 2>/dev/null; then
    log "the repeat differs from its first run (or did not run): the battery stops before any measured arm"; exit 1
  fi
  if [ "$arm" = long_context ]; then write_policy standard || log "the standard policy did not come back"; fi
done
export OUT_DIR=$RUN_DIR/setup
$N_ clocks end >> "$L" 2>&1
cp "$RUN_DIR/setup/clocks.json" "$RUN_DIR/clocks.json" 2>/dev/null
log "battery done: $(ls "$RUN_DIR/arms" | tr '\n' ' '); the record is $RUN_DIR"
