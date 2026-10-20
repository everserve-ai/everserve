# The two-node smoke: what the two-node run would need, 2026-10-07

The owner's brief, item 3 of the detection review: before landing 5, a cheap smoke on two pods to
answer three questions about the two-node run (`bench/runs/PLANNED/p2-two-node/`): who launches
and drives a remote node's processes (point 2), what the head's executor does when a remote
worker dies and whether a restarted node rejoins (point 3), and what the network between pods is
(point 5). Stock vLLM at the pin, Qwen3-1.7B, tensor parallel 1 × pipeline parallel 2, one A40 per
pod, `run.log` for every command; raw outputs in `data/` (first pass) and `data2/` (third pass), not in git.

## Point 5: the network between RunPod pods

- **Ordinary pods cannot reach each other at all.** A pod's `eth0` is a NAT'd `172.16.x` address;
  only ports declared at creation are reachable, mapped to other port numbers. vLLM's multi-node
  mode binds random ports (its broadcast queue, each remote worker's response queue, NCCL), so it
  cannot run over that.
- **RunPod's "global networking" gives pods a private network,** on the REST API only
  (`globalNetworking: true`; the GraphQL API the harness used has no such field). Each pod gets a
  `podnet1` interface on a `10.0.0.0/10` address. The harness's `runpod.py deploy --networked`
  uses it.
- **It is a tunnel shaped to 100 Mbit/s.** `podnet1` is a TUN device, MTU 1300, with a token-bucket
  qdisc at `rate 100Mbit`. iperf3 between two EU-SE-1 pods: 91 Mbit/s. That is about 1/1000 of an
  InfiniBand link.
- **It did not pass traffic between two CA-MTL-1 pods.** Created a minute apart in the same data
  center, neither could reach the other's port 22 over the tunnel in sixteen minutes; the second
  pass ended there.
- **What this means for the two-node run.** At 100 Mbit/s a prefill chunk's pipeline send (268 MB)
  takes about 24 s, and replicating one prefill burst from four ranks (about 2.6 GB) about four
  minutes. The run as pre-registered cannot be done on ordinary RunPod pods. It needs RunPod's
  Instant Clusters product (InfiniBand between nodes, with a minimum size), or a provider with a
  flat high-speed network between GPU hosts. Decode traffic alone (128 KB per step) did run here,
  at 360–430 tok/s for four short sequences.

## Point 2: who launches node-b, and what the head holds

- **Node-b's workers are launched by node-b's own `vllm serve --headless --node-rank 1`,** a stock
  `MultiprocExecutor` with no engine (`serve.py:run_headless`). The head launches nothing on
  node-b and holds no process handle for its workers.
- **The head's executor** (`data/a/head.json`): `world_size 2`, `local_world_size 1`, one local
  worker handle; `response_mqs` = one local queue and one *remote* queue; its broadcast queue
  publishes to remote readers at `tcp://10.1.84.61:32857`, a random port chosen at startup and
  carried to node-b's workers through torch.distributed's rendezvous.
- **Two settings the head needs that single-node runs never did:** `GLOO_SOCKET_IFNAME` as well
  as `NCCL_SOCKET_IFNAME` (vLLM's CPU groups bound `eth0`'s NAT address and could not connect), and
  `VLLM_HOST_IP` for the queue's address. The last pipeline stage JIT-compiles FlashInfer's
  sampler; without the CUDA 13 toolkit (which the harness's full setup installs) it needs
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Consequence for the adapter.** Our executor subclass could not have created node-b's slot
  region, set its workers' environment, or launched a spare there: the remote node's processes
  are the headless process's, and in the deployed shape must be the daemon's, as the design
  constraint in the brief says. The region and the environment on a remote node are that
  node's to create.

## Point 3: a remote worker dies; a restarted node

- **A restarted node-b cannot join a running engine.** torch.distributed's rendezvous is a
  startup-only exchange; the restarted rank 1 failed against the head's store
  (`DistNetworkError`, "TCPStore server has shut down"), while the head kept running on the old
  world. A spare's join must be ours, outside vLLM's world formation.
- **A remote worker killed** (third pass, `data2/`): node-b's `VLLM::Worker_PP1` was killed with
  SIGKILL at 12:11:55 while the head ran a generation.
  - **The head's local worker saw it within two seconds.** At 12:11:57 its gloo pair reported
    "Connection closed by peer [10.0.123.233]" and its pipeline send failed with
    `ncclRemoteError`: a killed process's TCP sockets close, and both transports notice. That is a
    cross-node death signal that costs nothing, and it reaches the survivor on the other node,
    not only its own.
  - **The head's executor never noticed.** Its step thread sat in `get_response` →
    `response_mqs[1].dequeue` → ZMQ `poll` on the remote reader's subscription
    (`stacks_after_node_b_worker_killed.txt`). A dead remote writer produces no error on a
    pub/sub socket. The step hung for the 60 s watchdog, and torch's own NCCL watchdog logged the
    failed collective at 12:12:57, 62 s after the kill. Stock vLLM has no path from a remote
    worker's death to the executor.
  - **Node-b's headless executor shut the whole node down at once:** "Worker proc VllmWorker-1
    died unexpectedly, shutting down executor" at 12:11:55, and every local worker exited. With
    four workers per node, one GPU's death on node-b takes node-b's other three with it. The
    remote node's executor must be ours, with death as an event, or a single death is always a
    node's loss, which is decision 0022's rule enforced by vLLM rather than chosen by us.
  - **For the adapter:** the executor's wait on a remote rank's queue needs a loss signal: the
    coordinator's commit naming the rank (its daemon sees the worker die), or the local worker's
    `ncclRemoteError` on the pipeline send reported as that peer lost. Decision 0021's stall rule
    does not apply across nodes: `/proc` is node-local, and the stalled step here was the head's,
    whose own worker was alive.
- **NCCL over the tunnel** (one send, GPU to GPU): 128 KB in 8.36 ms (0.02 GB/s), 268 MB in
  22.4 s (0.01 GB/s); the 650 MB send failed at the 120 s timeout. A decode step's pipeline send
  alone would be half of the 72B shape's 17 ms step.

## What this smoke cost

Three passes: about $0.67 (two A40s in EU-SE-1, 41 min: the engine ran, the kill missed its
target), $0.28 (two A40s in CA-MTL-1, 17 min: no tunnel), and $0.17 (an A40 and an A5000 in
EU-SE-1, 13 min: the kill landed). The balance fell $1.16, from $7.88 to $6.72. Several minutes
of each pass were my script's mistakes, listed in `run.log`; the scripts as they ended up are in
`bench/harness/twonode/`.
