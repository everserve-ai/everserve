#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The drains smoke (bench/runs/2026-10-13-run3-drains-smoke/PREREGISTRATION.md, "Steps"), on hosts nodes.sh
# already holds (set up with OUT_DIR=$RUN_DIR/setup, so the node-a guards' record is there): an
# engine over three nodes (SERVING "a b c", TP1 x PP3, node-a the head), node-d's spare for rank 1
# and node-e's for rank 2, one concurrent loss, the coordinator on a file spec store. The sessions run the
# whole time (the drill's follow arm completes every change the coordinator commits). With the
# engine serving: every registration read; the spare nodes cordoned and uncordoned, read after each; node-b
# drained, node-c drained, node-b drained again with another lead; the coordinator killed and
# started again once node-b's notices are out; both changes awaited. Before it, the engine starts
# once without spares, so every serving node compiles under the run's configuration, and each
# serving node's compile cache is staged onto its spare's node (decision 0029: a spare that
# compiled on its own node is refused). Every node's logs are collected on every exit, the
# early ones included. Every API answer goes into
# $RUN_DIR/reads/ as it comes, every action with its time into $RUN_DIR/actions.jsonl, and
# drains_verdict.py scores the record into $RUN_DIR/verdict.json.
#
# Settings (environment): TWONODE_SCRATCH, AIP/BIP/CIP/DIP, RUN_DIR, DEPLOY_COMMIT, MODEL
# (Qwen/Qwen3-1.7B), N (4), PROMPT_LEN (512), MAX_TOKENS (16000), LEAD_B (20000), LEAD_C (120000),
# RESTART_AFTER_S (10: from node-b's first notice to the coordinator's kill), WAIT_S (900: the
# longest wait for both changes), SPEND_SINCE/SPEND_RATE/SPEND_GPUS (each step's line then carries
# the spend so far).
set -u
cd "$(dirname "$0")/../../.."
S=${TWONODE_SCRATCH:?the scratch directory holding ssh_a, ssh_b, ...}
RUN_DIR=${RUN_DIR:?where the record goes}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}; MODEL_DIR=${MODEL_DIR:-/workspace/models/$(basename "$MODEL")}
N=${N:-4}; PROMPT_LEN=${PROMPT_LEN:-512}; MAX_TOKENS=${MAX_TOKENS:-16000}
MAX_MODEL_LEN=$((PROMPT_LEN + MAX_TOKENS + 256)); BLOCKS=$(( (N * MAX_MODEL_LEN + 15) / 16 + 64 ))
SLOTS=$((BLOCKS * 16))
LEAD_B=${LEAD_B:-20000}; LEAD_C=${LEAD_C:-120000}; COVER_WAIT_S=${COVER_WAIT_S:-300}; RESTART_AFTER_S=${RESTART_AFTER_S:-10}
WAIT_S=${WAIT_S:-900}
export NODES="a b c d e" SERVING="a b c" TP=1 PP=3 MODEL MODEL_DIR N PROMPT_LEN MAX_MODEL_LEN BLOCKS SLOTS
export STORE_BUDGET=${STORE_BUDGET:-8000000000} MIRROR_DIR=${MIRROR_DIR:-/dev/shm/mirror}
export POLICY=bench/harness/policies/drains.json SPEC_STORE=/workspace/run/spec.json
export SERVE_GPUS_A=0 SERVE_GPUS_B=0 SERVE_GPUS_C=0 SPARE_GPU_A=0 SPARE_RANK=1
export RUN_TIMEOUT=$((WAIT_S + 600)) OUT_DIR=$RUN_DIR/collect
N_=bench/harness/twonode/nodes.sh
PY=.venv/bin/python
COORD_METRICS=7401
mkdir -p "$RUN_DIR/reads" "$OUT_DIR"; L=$RUN_DIR/run.log
log() { echo "$(date -u +%FT%TZ) $*$(spend)" | tee -a "$L"; }
spend() {  # the rental's spend so far, reckoned as nebius.py budget reckons it; nothing unset
  [ -n "${SPEND_RATE:-}" ] || return 0
  $PY -c "import sys, time; sys.path.insert(0, 'bench/harness'); import nebius; print(f', \${nebius.spend_usd(float(sys.argv[1]), time.time(), int(sys.argv[2]), float(sys.argv[3])):.2f} spent')" "$SPEND_SINCE" "$SPEND_GPUS" "$SPEND_RATE"
}
action() {  # NAME, then the command whose output is the action's record: its time and status kept
  local name=$1; shift
  local t; t=$(date +%s.%N)
  "$@" > "$RUN_DIR/reads/$name.txt" 2>&1
  local rc=$?
  echo "{\"action\": \"$name\", \"t\": $t, \"rc\": $rc}" >> "$RUN_DIR/actions.jsonl"
  return $rc
}
on_a() { "$S/ssh_a" "$@"; }
reads() {  # PHASE: the engine, the nodes, the event logs and the coordinator's metrics, as they stand
  local phase=$1 n
  # the Engine object (decision 0042), its readiness under .status; the file keeps its name
  $N_ api_read /v1/engines/rig > "$RUN_DIR/reads/${phase}_readiness.json" 2>/dev/null
  $N_ api_read /v1/nodes > "$RUN_DIR/reads/${phase}_nodes.json" 2>/dev/null
  on_a "curl -s http://$AIP:$COORD_METRICS/metrics" > "$RUN_DIR/reads/${phase}_metrics.txt" 2>/dev/null
  $N_ api_read /v1/engines/rig/events > "$RUN_DIR/reads/${phase}_events_rig.json" 2>/dev/null
  for n in b c d e; do
    $N_ api_read /v1/nodes/node-$n/events > "$RUN_DIR/reads/${phase}_events_node-$n.json" 2>/dev/null
  done
  echo "{\"action\": \"read_$phase\", \"t\": $(date +%s.%N), \"rc\": 0}" >> "$RUN_DIR/actions.jsonl"
}
grep_on() {  # NODE FILE PATTERN: whether the node's file holds the pattern
  "$S/ssh_$1" "grep -q '$3' $2" 2>/dev/null
}
until_true() {  # SECONDS, then a command: polled every second until it succeeds or the time runs out
  # by the clock, not by count: a poll over ssh takes a second or two of its own
  local end=$((SECONDS + $1)); shift
  until "$@"; do [ "$SECONDS" -ge "$end" ] && return 1; sleep 1; done
}
stop_here() {  # WHY: the run stops early; every node's logs are collected first
  log "$1"; [ -n "${DRILL:-}" ] && kill "$DRILL" 2>/dev/null
  $N_ collect >> "$L" 2>&1; $N_ down >> "$L" 2>&1; exit 2
}
refused() {  # the coordinator's refusals of this run's spares, as it logged them
  on_a "sed 's/\x1b\[[0-9;]*m//g' /workspace/out/coord.log | grep 'not admitted'" 2>/dev/null
}
spares_standing() {  # both spares registered with the coordinator
  $N_ api_read /v1/engines/rig 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin)['status']; sys.exit(0 if len(d.get('spares') or []) >= 2 else 1)" 2>/dev/null
}
covered() {  # RANK [EPOCH]: the plan's set for RANK's loss is covered, at EPOCH or later
  $N_ api_read /v1/engines/rig 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)['status'].get('cover') or {}
ok = d.get('epoch', -1) >= int(sys.argv[2]) and any(
    s.get('purpose') == 'REPLACE of ranks [' + sys.argv[1] + ']' and s.get('covered')
    for s in d.get('sets') or [])
sys.exit(0 if ok else 1)" "$1" "${2:-0}" 2>/dev/null
}
covered_both() { covered 1 && covered 2; }
committed_b() { grep_on a /workspace/out/coord.log 'membership committed at epoch .*lost \[1\]'; }
committed_both() {  # the coordinator has committed node-b's change and node-c's
  grep_on a /workspace/out/coord.log 'membership committed at epoch .*lost \[1\]' \
    && grep_on a /workspace/out/coord.log 'membership committed at epoch .*lost \[2\]'
}
drill_running() { kill -0 "$DRILL" 2>/dev/null; }

write_policy() {  # SPARES (empty for none): the shape's policy, one concurrent loss, synced to the nodes
  $PY bench/harness/twonode/policy.py --tp 1 --pp 3 --members a:1 b:1 c:1 ${1:+--spares $1} \
    --max-concurrent-losses 1 --cover --model "$MODEL_DIR" --max-model-len "$MAX_MODEL_LEN" --blocks "$BLOCKS" \
    --max-num-seqs "$N" --prompt-len "$PROMPT_LEN" --policy-path "$POLICY" --out "$POLICY" || return 1
  local n
  for n in $NODES; do "$S/ssh_$n" "cat > /workspace/everserve/$POLICY" < "$POLICY" || return 1; done
}
log "drains smoke on $MODEL, TP1 x PP3 over a b c, spares d:1 e:1, $N sessions x $PROMPT_LEN prompt, $MAX_TOKENS tokens, blocks $BLOCKS; leads node-b $LEAD_B ms, node-c $LEAD_C ms; commit $(git rev-parse --short "${DEPLOY_COMMIT:?the commit this deploy runs}")"
$N_ update >> "$L" 2>&1 || stop_here "update failed"

# the warm-up: the engine once without spares, so node-b and node-c compile under this run's
# configuration; their caches are then staged onto the spares' nodes before the spares launch
write_policy "" || stop_here "the warm-up policy did not reach the nodes"
$N_ control >> "$L" 2>&1 || stop_here "the control plane did not come up for the warm-up"
$N_ drill warmup --arm adapter --max-tokens 32 >> "$L" 2>&1 || stop_here "the warm-up start failed"
$N_ down >> "$L" 2>&1
CACHE_FROM=b CACHE_TO=d $N_ cache_sync >> "$L" 2>&1 || stop_here "node-b's compile cache not staged onto node-d"
CACHE_FROM=c CACHE_TO=e $N_ cache_sync >> "$L" 2>&1 || stop_here "node-c's compile cache not staged onto node-e"
log "warm-up done; compile caches staged b to d and c to e"

write_policy "d:1 e:1" || stop_here "the policy did not reach the nodes"
cp "$POLICY" "$RUN_DIR/policy.json"
on_a "rm -f $SPEC_STORE"  # the spec store starts empty: what it holds afterwards is this run's
$N_ control >> "$L" 2>&1 || stop_here "the control plane did not come up"
$N_ clocks start >> "$L" 2>&1

# the engine and its sessions; the reads and the drains happen once the drill says it is stepping
on_a "rm -f /workspace/out/drains.start"
$N_ drill drains --arm follow --kill external --kill-step 50 --max-tokens "$MAX_TOKENS" --start-after /workspace/out/drains.start >> "$L" 2>&1 &
DRILL=$!
for _ in $(seq 300); do
  spares_standing && break
  r=$(refused); [ -n "$r" ] && stop_here "a spare was refused: $(echo "$r" | cut -c1-400 | tr '\n' ';')"
  sleep 1
done
spares_standing || stop_here "the spares never registered"
until_true "$COVER_WAIT_S" covered_both || stop_here "the sets for ranks 1 and 2 were not covered in $COVER_WAIT_S s"
log "the engine up, both spares standing, both losses covered"
reads registered

action cordon_d $N_ cordon d; action cordon_e $N_ cordon e; reads cordoned
action uncordon_d $N_ uncordon d; action uncordon_e $N_ uncordon e; reads uncordoned
# the cordon released the spares' groups; the uncordon's plan forms them again
until_true "$COVER_WAIT_S" covered_both || stop_here "the sets were not covered again in $COVER_WAIT_S s"
reads recovered
log "node-d and node-e cordoned and uncordoned; both losses covered again"

# the sessions now, so every change runs under load
on_a "touch /workspace/out/drains.start"
until_true 900 grep_on a /workspace/out/drains.json.armed . || stop_here "the engine never reached step 50"
log "serving at step 50"

action drain_b $N_ drain b "$LEAD_B"
# the restart is timed from node-b's first notice, which the coordinator logs as it sends it
# (node-b's daemon does not log it), and runs beside the other drains' calls and their read so
# that they do not delay it: run in sequence, the restart lands after both changes
(
  until_true 120 grep_on a /workspace/out/coord.log 'rig: notice for rank 1' \
    || log "node-b's notice not seen in 120 s"
  sleep "$RESTART_AFTER_S"
  action coord_restart $N_ coord_restart || log "the coordinator did not come back"
  reads restarted
  log "coordinator restarted"
) &
RESTART=$!
action drain_c $N_ drain c "$LEAD_C"
action drain_b_again $N_ drain b "$((LEAD_B + 60000))"
reads draining
log "drains asked: node-b ($LEAD_B ms), node-c ($LEAD_C ms), node-b again ($((LEAD_B + 60000)) ms)"
wait "$RESTART"

until_true "$WAIT_S" committed_b || log "node-b's change not committed in $WAIT_S s"
# node-c's change goes only through a covered set (0035's REPLACE gate): rank 2's at epoch 1
if until_true "$COVER_WAIT_S" covered 2 1; then rc=0; else rc=1; fi
echo "{\"action\": \"covered_after_b\", \"t\": $(date +%s.%N), \"rc\": $rc}" >> "$RUN_DIR/actions.jsonl"
reads covered_after_b
log "rank 2's set at epoch 1: $([ $rc = 0 ] && echo covered || echo "not covered in $COVER_WAIT_S s")"
until_true "$WAIT_S" committed_both || log "both changes not committed in $WAIT_S s"
reads committed
log "both changes committed; the sessions run to their end"
# a drained node's worker still running once both changes committed has not released its
# position: its Python stacks (SIGUSR1, into its node's log) and native ones say where it waits
# (a worker that never releases is the finding, and only its stacks say why)
for n in b c; do
  if "$S/ssh_$n" "pgrep -f 'VLLM::[W]orker' > /dev/null"; then
    log "node-$n's worker still running after both commits: its stacks taken"
    "$S/ssh_$n" "pkill -USR1 -f 'VLLM::[W]orker'"
    $N_ stacks after_commits "$n" >> "$L" 2>&1
  fi
done
until_true 600 bash -c "! kill -0 $DRILL 2>/dev/null" || log "the drill still running 600 s after the commits"
wait "$DRILL"; echo "{\"action\": \"drill_end\", \"t\": $(date +%s.%N), \"rc\": $?}" >> "$RUN_DIR/actions.jsonl"
reads final
for n in b c; do "$S/ssh_$n" "pgrep -af 'everserve_vllm.node_mai[n]|VLLM::[W]orker'" > "$RUN_DIR/reads/final_procs_$n.txt" 2>&1; done
$N_ clocks end >> "$L" 2>&1
$N_ collect >> "$L" 2>&1
$N_ down >> "$L" 2>&1
$PY bench/harness/twonode/drains_verdict.py "$RUN_DIR" --max-tokens "$MAX_TOKENS" > "$RUN_DIR/verdict.json"
log "verdict: $(python3 -c "import json,sys; v=json.load(open(sys.argv[1])); print(v['verdict'], [k for k, p in v['predictions'].items() if not p['pass']])" "$RUN_DIR/verdict.json")"
