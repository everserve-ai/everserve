# Installing the control plane on bare metal

The deployed shape for a GPU tier that is not on Kubernetes: one policy file, two binaries, three
systemd units, certificates from the operator's CA or from `everserve-controller certs`. One
coordinator per deployment; one store and one daemon per node. The engine's workers run beside
them with the vLLM plugin (`adapters/vllm`) and find the daemon through `/run/everserve`.

| unit | runs | on |
|---|---|---|
| `everserve-coord` | `everserve-controller coord`, the one writer of every engine's epoch, holding `/var/lib/everserve/coord.lease` | one host; a second instance stands by: on the same host with the lease file, or on another with an etcd lease (`EVERSERVE_LEASE=etcd://...`); one that loses the lease exits 75 and systemd restarts it to stand by |
| `everserve-store` | `everserve-store`, the committed-KV mirror in `/dev/shm/everserve`, replicating to the stores the policy's `replicaNodes` name (decision 0030) | every node |
| `everserve-node` | `everserve-controller node`, the coordinator's agent and the workers' registry | every node |

## Steps

1. Build: `cargo build --release` (or take the released binaries). The controller's build needs
   `protoc` (`apt install protobuf-compiler`), for the etcd client.
2. On every host: `sudo deploy/baremetal/install.sh`. It installs the binaries, the units, the
   `tmpfiles.d` entry for `/run/everserve`, `/dev/shm/everserve` and `/var/lib/everserve`, the
   `everserve` user, and the example settings into `/etc/everserve/` (existing settings are kept).
3. Certificates. With the operator's CA: one certificate for the coordinator carrying the name
   daemons verify (`EVERSERVE_COORDINATOR_NAME`, default `coordinator`), and one per node carrying
   the node's name and the name stores verify on each other (`EVERSERVE_STORE_PEER_NAME`, default
   `store`). Without one:
   ```sh
   everserve-controller certs --out /etc/everserve/certs --node node-a --node node-b --node node-c
   ```
   then copy each node its own `.crt` and `.key` and `ca.crt`; run it again with more `--node`
   flags to add nodes under the same CA. Keys are readable by their owner only.
4. The policy at `/etc/everserve/policy.json` on the coordinator's host
   (`docs/examples/policy-two-node.json` is the three-node shape: two serving nodes, one spare
   node).
5. Edit `/etc/everserve/coord.env`, `store.env` and `node.env`; every line is documented in the
   examples and every variable is a flag the binary's `--help` describes. On a spare node the
   store's `EVERSERVE_STORE_REPLICA_LISTEN` and the daemon's `EVERSERVE_MIRROR_REPLICA` name the
   address the other nodes reach it through; on a serving node that takes no replicas both stay
   unset. `EVERSERVE_NOTICE_SOURCE` names where the node learns it is leaving.
6. `systemctl enable --now everserve-store everserve-node` on every node, then
   `systemctl enable --now everserve-coord` on the coordinator's host. Order does not matter: a
   daemon holds its reports until a coordinator is reachable, and the coordinator takes daemons
   as they link.

## Checking it

- On any host that reaches the coordinator's operator API, with an operator's or a viewer's
  certificate (`everserve-controller certs --out /etc/everserve/certs --operator ops`):
  ```sh
  everserve-controller check --policy /etc/everserve/policy.json --coordinator-api coordinator:7443 \
    --ca /etc/everserve/certs/ca.crt --cert /etc/everserve/certs/operator-ops.crt \
    --key /etc/everserve/certs/operator-ops.key
  ```
  waits until every node the policy names has its daemon connected, its members' devices
  reported, and, for a spare node, its store taking replicas; prints one line per node; exits 0
  when ready and 1 when not by `--timeout-secs`. Run it before an engine starts.
- The coordinator's metrics port: `everserve_daemons_connected` equals the node count.
- The operator API (decision 0042): `GET /v1/engines` lists each engine once its workers
  registered, with per node whether it can be reclaimed; `PUT /v1/nodes/{node}` with a
  `spec.drain` is the operator's reclaim call (`docs/RUNBOOK.md`).
- Each daemon's metrics: `everserve_coordinator_linked 1`.
- Each store's metrics: `everserve_store_replica_lag_steps` per rank on a serving node once the
  engine commits, `everserve_store_replica_applied_total` on a spare node.

`deploy/baremetal/smoke.sh` runs exactly this shape on one machine, three nodes over TLS on the
loopback, and is part of the gate; it is the check to run on real hosts before an engine starts.

## Operating

- SIGTERM gives in-flight changes `EVERSERVE_GRACEFUL_SHUTDOWN_SECS`; the units allow that plus a
  margin.
- A coordinator restart resumes at the epoch the daemons hold; a daemon restart re-registers
  nothing by itself, the workers reconnect to it.
- Rotate certificates by writing new ones and restarting the units; a CA change is a full
  redeploy of `ca.crt`.

Kubernetes is the other deployed shape (`docs/MULTINODE.md` section 9): the same binaries as a
DaemonSet and a Deployment, the lease through the API, each node's certificate issued by the
coordinator. Its chart is `deploy/kubernetes/everserve` (`deploy/kubernetes/README.md`).

## Upgrading

A release serves the protocol of the one before it unless its notes say otherwise (decision
0040), so an upgrade rolls without stopping the fleet: the coordinator's host (and its standby's)
first, then each node's daemon and store, one node at a time, then each engine's processes, by
draining a node onto its spares and restarting its workers on the new build. A link between
builds that do not serve each other is refused at once, both versions in the log.
