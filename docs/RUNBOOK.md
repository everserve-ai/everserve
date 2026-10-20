# Runbook

What an operator does with a running control plane: reach its API, move work off a node, and
answer each alert in `deploy/grafana/alerts.yml`. Installing it is in `deploy/kubernetes/` and
`deploy/baremetal/`; what each piece is, in `docs/ARCHITECTURE.md`.

## Reaching the operator API

The coordinator serves its API over TLS (decision 0039). Every request is authenticated, reads
included. `everserve-controller`, in the control plane's image or built from this repository,
calls it; each setting may be given once in the environment.

On Kubernetes, with a service account bound to `everserve-viewer` (reads and watches),
`everserve-operator` (also writes) or `everserve-launcher` (a pool's spare launcher) in the
release's namespace:

```sh
kubectl -n everserve get secret everserve-tls -o jsonpath='{.data.ca\.crt}' | base64 -d > ca.crt
kubectl -n everserve port-forward svc/everserve-coordinator 7443 &
kubectl -n everserve create token <service-account> > token
export EVERSERVE_API=127.0.0.1:7443 EVERSERVE_API_CA=ca.crt EVERSERVE_API_TOKEN_FILE=token
```

On bare metal, with an operator's certificate from `everserve-controller certs --operator <name>`:

```sh
export EVERSERVE_API=<coordinator>:7443 EVERSERVE_API_CA=ca.crt \
  EVERSERVE_API_CERT=operator-<name>.crt EVERSERVE_API_KEY=operator-<name>.key
```

The coordinator's certificate names the chart's coordinator name (`coordinator` by default;
`--coordinator-name` or `EVERSERVE_COORDINATOR_NAME` otherwise), which the command verifies
whatever address reaches it. A call without a credential is refused 401; one the caller's role
does not allow, 403. The coordinator logs the identity, the verb and the resource of each. A
command exits 0 when done, 1 when the API refused (its reason printed), 2 when the coordinator
could not be reached.

The reads an operator needs most:

| command | the API's route | what it says |
|---|---|---|
| `everserve-controller get engines [<engine>]` | `GET /v1/engines[/<engine>]` | each engine's instance state, epoch, members, spares and whether a node's loss is covered; with `-o json`, its whole `status`: change in flight, conditions (`Serving`, `Progressing`, `Covered`, `Degraded`, `Down`), and for each node holding a member whether it can be reclaimed and how |
| `everserve-controller get nodes [<node>]` | `GET /v1/nodes[/<node>]` | every node: linked, cordoned, where its drain stands, the engines on it |
| `everserve-controller get pools [<pool>]` | `GET /v1/pools[/<pool>]` | each spare pool, what it wants and what stands |
| `everserve-controller watch nodes` | `GET /v1/nodes?watch=true` | each change as it happens, until stopped |
| | `GET /v1/{engines,nodes,pools}/<name>/events` | what happened to it, oldest first, each with a reason |

Every object and route is in `docs/api/openapi.json`. Without the binary, any HTTP client calls
the same routes with the same credential, as `curl --cacert ca.crt -H "Authorization: Bearer
$(cat token)" https://coordinator:7443/v1/engines --resolve coordinator:7443:127.0.0.1`.

## Moving work off a node

A node's spec says what is asked of it. A command reads the node, changes its spec and writes it
back at the version it read: if the node changed in between, the write is refused, 409
`Conflict`, and running the command again reads it afresh.

- **Cordon** keeps new spares off a node; what it holds stays:
  `everserve-controller cordon <node> --reason firmware`; `uncordon <node>` undoes it.
- **Drain** moves every rank on a node to spares by its deadline and releases the node; a
  drained node is cordoned: `everserve-controller drain <node> --in 60 --reason maintenance`.
  It prints where the drain stands: `Scheduled`, or `Waiting` when other drains already move as
  many positions as the policy allows at once. Read `get engines -o json` first: a node whose
  `reclaimable` is false would be rebuilt, not replaced.
- **Undrain** withdraws a drain that waits its turn or whose change was refused:
  `everserve-controller undrain <node>`. A drain whose notices went out is underway, and the write
  is refused 409 `DrainUnderway`. The node stays cordoned until it is uncordoned.

Each write records the caller's authenticated identity as its writer.

## When a process hangs

`POST /v1/engines/<engine>/stacks` with `{"rank":<rank>}` asks the process serving `rank` (or the
spare taking it) for its Python stacks; `{"head":true}` asks the head, and
`{"node":"<node>","device":<device>}` a spare standing for any position. The answer carries the
stacks, at most 1 MiB, within 5 s.

## Alerts

Each alert's `runbook_url` points at its section here.

### EverserveEngineDown

The engine is down (decision 0032): its head stopped, failed or was lost, or a change failed.
Everserve does not restart an engine; the framework that launched it does (Grove under Dynamo).
The head's loss always takes its engine down: it is a known gap (decision 0045), and under Dynamo
the frontend migrates the broken streams to another instance. A coordinator that has just started
or taken the lease claims no change of an engine until it has heard every rank's holder from the
daemons, or for at most 10 s.

1. `everserve-controller get engines <engine> -o json`: `status.down` says why and since when; the `Down` condition's reason is
   `HeadStopped`, `HeadFailed`, `HeadLost` or `ChangeFailed`.
2. `GET /v1/engines/<engine>/events` for the change or the loss before it.
3. Check the framework restarted the engine's pods. Once every process of the old incarnation
   has gone and one of the new asks, the engine starts anew (`EngineRestarted` in its events) and
   the alert clears.

### EverserveChangeStuck

A change has been in flight for 2 minutes; a change commits in seconds. The instance reports
recovering meanwhile, so the framework holds its own recovery off.

1. `everserve-controller get engines <engine> -o json`: `status.change` names the epoch and the ranks it takes.
2. Take the stacks of the ranks it waits on (above) and of the head. A process inside a
   collective that never completes is the usual cause.
3. `GET /v1/nodes/<node>/events` for a node of the change: a daemon that lost its link holds
   its acknowledgements.

### EverserveChangeFailed

A change ended other than committed or folded into the next; the `result` label says how.

| result | meaning |
|---|---|
| `refused_rebuild`, `refused_uncovered` | no spare or formed set could answer the loss; the policy refused to rebuild |
| `survivor_error`, `survivor_departed` | a surviving rank failed or left during the change |
| `no_delivery` | a node's daemon could not be reached with the change |
| `epoch_mismatch` | a process answered for another epoch |
| `engine_down` | the engine went down during the change |
| `domain` | the spare is in another network domain than the change needs |
| `unknown_rank` | the loss named a rank the engine does not have |
| `deposed`, `lease_lost` | the coordinator lost its lease during the change; the next holder finishes it |
| `unlogged` | the change could not be written to the spec store before its delivery |

`GET /v1/engines/<engine>/events` has the change's event with the detail. A refused change
leaves the engine as it was; a failed delivery leaves it down (EverserveEngineDown).

### EverserveEngineDegraded

The engine serves short of its policy for 5 minutes. The `Degraded` condition's reason says how:

- `Shrunk`: ranks lost under SHRINK still count against the policy's concurrent losses. An
  instance does not grow back; it serves smaller until its framework restarts it.
- `SecondHead`: a second head registered; the message names its node and pid. Stop the one the
  framework did not mean to run.
- `SpareRefused`: a spare was built differently from the position it stands for: its engine or
  adapter commit, NCCL or CUDA version, checkpoint, dtype, KV format or step config. The message
  names the field and both values. Run the spare from the engine's image and configuration.

### EverserveEngineUncovered

For 10 minutes, a loss of some node or position would be rebuilt, not replaced. The `Covered`
condition's message lists the losses; its reason is `NodeUncovered` (no spare stands for the
node's ranks), `LossUncovered` (no planned set answers the loss) or `Forming` (the sets are
planned and their groups are still forming). A loss of the head's node, or of any rank on it, is
named in the message and never counted: it restarts the engine (decision 0045), and no spare
answers it, so it never holds this alert up.

1. `everserve-controller get engines <engine> -o json`: each node's `why`, and the cover plan's `uncovered`.
2. `everserve-controller get pools`: a pool short of its spares raises EverservePoolShort too; raise the pool's
   `wants`, or uncordon nodes spares may stand on.

### EverserveReleaseLeaked

A refused communicator's release (`ncclCommAbort`, or torch's abort of its groups) is still
running past its bound in a serving process (decision 0025, amended 2026-10-15). Serving is
unaffected: the refused communicator is never called again. But the release holds that
communicator's memory and a thread until the process ends. So the process reports itself
degraded and asks to leave its position by notice. A spare takes the position with no gap, and
the daemon restarts the process.

1. `GET /v1/engines/<engine>/events`: `ReleaseLeaked` names the release, its process and its run
   time. A later departure of that process's position, with the change committed, means it has
   resolved, and the alert clears once the counter stops rising.
2. If the process stays degraded, `everserve-controller get engines <engine> -o json` gives the
   position's cover. A process waits for a spare that can take its position: none stands, or none
   is covered for it (EverserveEngineUncovered and EverservePoolShort say which). Either:
   - add spare capacity (raise the pool's `wants`, or uncordon a node), and the departure goes
     through at the next plan that covers the position; or
   - restart the process by hand (delete its pod), accepting the loss: its position is then
     taken as an unplanned loss, rebuilt or refused as the policy says.
3. Take the process's stacks (above) before you restart it. A release that never returns is
   recorded with NCCL's native stack, which says where it waits.

### EverservePoolShort

A pool has stood short of the spares it wants for 15 minutes.

1. `GET /v1/pools/<pool>/events`: a spare that failed to start, or one refused.
2. A spare stands only on an uncordoned node of the pool's device class and domain; check enough
   such nodes exist.

### EverserveDaemonUnlinked

A node's daemon has had no coordinator for a minute. A loss on its node waits until it links.

1. The coordinator: is a holder ready (`kubectl -n everserve get pods`)? Only the holder serves.
2. The daemon's log: a refused handshake means its certificate expired or was not issued; the
   `node-cert` container's log says why (on Kubernetes, the API server must state the pod's node
   in its tokens, decision 0037).
3. NetworkPolicies: the coordinator's link port (7000) admits the node daemons' pods only.

### EverserveCoordinatorDeposed

A coordinator learned of a newer term and should have exited, and is still running after a
minute. It acts no more; its supervisor did not restart it. Delete its pod (or restart its
service); it contends for the lease again as a standby.

### EverserveCoordinatorUnreachable

A coordinator started or took the lease, engines served before it, and no daemon has reached it
within a minute. It is the one cut off, not every node at once, and it changes nothing until a
daemon reaches it. The daemons dial it through its Service: check the Service's endpoints name
its pod (`kubectl -n everserve get endpoints`), that the pod is Ready, and that the daemons' logs
show them connecting (`coordinator link ended` with the reason). Once one daemon is heard, every
rank of an engine that served is either heard within the allowance (about 8 s) or lost, as a node
gone is: `RanksUnheard` in the engine's events and `everserve_unheard_ranks_lost_total` count them,
and `everserve_coordinator_relink_seconds` shows how long the daemons took.

### EverserveSpecWritesFenced

The spec store refused a coordinator's write as another's: two coordinators acted. The fence
held, and the deposed one stops at its refused write. Check one coordinator holds the lease
(`kubectl -n everserve get lease`) and that the other is standing by; if EverserveCoordinatorDeposed
fires too, follow it.

### EverserveReplicationDropped

A store dropped steps for an engine's ranks to a replica target that could not take them: the
replica of those ranks is not exact, and a change that restores from it would miss rows.

1. The target store's log and `everserve_store_replica_lag_steps`: a target that falls behind,
   or one refusing (EverserveStoreRefusing).
2. The link between the two nodes (port 7100).

### EverserveStoreCorrupt

A store refused rows whose checksums do not match, or that came without them (decision 0030).
It keeps none of them, so the ranks they belong to have no exact replica. The source or the
network altered rows.

1. The store's log names the layer and the slot.
2. Drain the source's node if it repeats; a node whose memory or NIC corrupts rows should not
   serve.

### EverserveStoreRefusing

A store refused a prepare or a commit; the `reason` label says why, and the rank it was for is
not kept:

| reason | meaning |
|---|---|
| `budget` | the tables would exceed the store's budget |
| `undersized` | the store holds fewer slots than the engine writes |
| `shape`, `size`, `slot`, `dtype`, `stage_name` | the engine's tables differ from what was prepared |
| `io` | the store could not read or write its files |

`budget` and `undersized` are sizing: the chart's `node.store.slots` and
`node.store.budgetBytes` must cover the engine's KV slots and bytes. The others mean an engine
built differently from the one prepared; the store's log has the detail.
