# The two-node smoke

Drill-only. `twonode.sh` rents two one-GPU pods on RunPod's private network (`runpod.py deploy
--networked`), measures the bandwidth between them (iperf3 and `bw.py`, an NCCL send at the
decode, prefill and replication sizes), starts stock vLLM at TP1 × PP2 with node-b headless and
the head in-process (`pp_head.py`), kills node-b's worker, restarts node-b, copies the outputs and
terminates both pods. What it found on 2026-10-07 is in `bench/runs/PLANNED/two-node-smoke/NOTE.md`.
`TWONODE_SCRATCH` names a directory for the pod ids and ssh wrappers it writes.

`nodes.sh` is the harness for the adapter across hosts (`NODES`, "a b" by default, "a b c" with
a third node holding only a daemon and the spares it launches, the product's spare node): the
control plane on every node over TLS (stores, daemons, the coordinator on node-a, every node's
discovery record copied to every other), node-b's half of the engine under our executor's node
role (`everserve_vllm.node_main`), the drill on node-a as the head, the losses (`lose` kills
node-b's daemon and workers together; `kill NAME RANK` kills one of node-b's workers), the
operator's drain, and the collection. Its header lists the stages and settings;
`policies/two-node-*.json` and `three-node-*.json` are the engines it runs. The hosts come from
`runpod.py` (pods on RunPod's tunnel network, `IFACE=podnet1`, root) or `nebius.py` (VMs on a
Nebius AI Cloud project's VPC, `IFACE=eth0`, `SUDO="sudo -n"`; GPU clusters on an InfiniBand
fabric for the rental).

## The battery

`battery.sh` runs the two-node pre-registration's arms (`bench/runs/PLANNED/p2-two-node/
PREREGISTRATION.md`, "What runs") on hosts `nodes.sh` already holds: the fabric's bandwidth
first (`nodes.sh fabric`: iperf3 to node-a from every other node, the NCCL send at the decode,
prefill and replication sizes), refused below the pre-registration's floor with `REFUSED.md`
saying why; then stock, adapter ×2, mirror-off, drain ×3, node fault ×5, single death ×5, the
notice reference, long context ×3 and the shard copy, each measured arm's trial on a fresh
control; `summary.py` writes one `summary.json` per trial (every scored figure beside its
prediction and kill condition, `accept.py`'s verdict, the clock offsets, the change's spans and
stamps) and `arm.json` per arm (median, worst, worst over median); the clocks are probed at the
start and the end (`clockoffset.py`). The policy for the shape is written by `policy.py`
(`MEMBERS="a:4 b:4" SPARES="c:4"` for the rental, `"a:2 b:2"` / `"c:2"` for a three-VM smoke of
the script). Nothing the site reads is computed from logs: the record is
`RUN_DIR/arms/<arm>/trial<k>/{result,control,summary}.json`, `arm.json`, `clocks.json`,
`setup/fabric.json`, `policy.json`, `run.log`.

## The drains smoke

`drains.sh` runs run 3's smoke (`bench/runs/2026-10-13-run3-drains-smoke/PREREGISTRATION.md`) on hosts
`nodes.sh` already holds: an engine over three nodes (`SERVING="a b c"`, TP1 x PP3), node-d's two
spares for ranks 1 and 2, one concurrent loss, and the coordinator on a file spec store. The drill
runs its sessions under the follow arm, which completes every change the coordinator commits and
sends nothing itself. With the engine serving, the driver cordons and uncordons node-d, drains
node-b and node-c, drains node-b again with another lead, kills the coordinator and starts it
again once node-b's notices are out, and waits for both changes. Every API answer is kept in
`RUN_DIR/reads/` and every action in `actions.jsonl`. `drains_verdict.py` scores each
pre-registered prediction from that record into `verdict.json`, one pass/fail line with the
reason per prediction.
