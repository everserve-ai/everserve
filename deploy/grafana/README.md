# The control-plane dashboard

`everserve.json` is a Grafana dashboard over the metrics the coordinator, the node daemons and
the stores export (`/metrics` on each process's metrics address). Import it and pick the
Prometheus datasource; `prometheus.yml` shows the three scrape jobs it expects
(`everserve-coord`, `everserve-node`, `everserve-store`), with `instance` as the node.

Two rows:

- **Membership:** daemons connected, changes in flight, each daemon's link, workers per engine,
  the spare pool by state, changes by outcome and result, and change seconds (p50 and p95 by
  cause) from a lost rank or a notice's deadline to the committed membership.
- **Store and replication:** commits per second and refusals per node, table bytes in use,
  replica lag in steps per source and target (the fence is three deep; a lag past it at a change
  means a restore from the replica would miss rows), replicated and applied rates, replication
  drops (must stay zero), and restore rows during a change.

The operator API (`GET /v1/engines` and `/v1/nodes` on the coordinator) is the per-node view
the dashboard does not show: whether a node can be reclaimed and with which outcome. A JSON panel
over it is a later addition.

## Alerts

`alerts.yml` holds Prometheus alerting rules over the same series. Load it as a rule file (or
wrap its groups in a `PrometheusRule` for the Prometheus operator).

| alert | when | severity |
|---|---|---|
| `EverserveEngineDown` | an engine is down (0032): its head lost or stopped, or a change failed | critical |
| `EverserveChangeStuck` | a change in flight for 2 minutes: the engine holds its framework's recovery off meanwhile | critical |
| `EverserveChangeFailed` | a change ended other than committed or folded into the next | warning |
| `EverserveEngineDegraded` | ranks lost under SHRINK, a second head, or a refused spare, for 5 minutes | warning |
| `EverserveEngineUncovered` | a loss no pre-formed set answers, for 10 minutes | warning |
| `EverservePoolShort` | a pool short of the spares it wants, for 15 minutes | warning |
| `EverserveDaemonUnlinked` | a node's daemon without a coordinator for a minute | critical |
| `EverserveReleaseLeaked` | a communicator's release ran past its bound in a serving process (0025): it holds memory until the process ends, and the coordinator retires the process once a spare can take its rank | warning |
| `EverserveCoordinatorDeposed` | a coordinator that learned a newer term still running after a minute | warning |
| `EverserveCoordinatorUnreachable` | engines served before this coordinator and no daemon has reached it for a minute: it is the one cut off, and changes nothing until one does | critical |
| `EverserveSpecWritesFenced` | the spec store refused a coordinator's write as another's | warning |
| `EverserveReplicationDropped` | a store dropped steps a replica target could not take | critical |
| `EverserveStoreCorrupt` | a store refused rows whose checksums do not match, or came without them | critical |
| `EverserveStoreRefusing` | a store refused a prepare or a commit for a reason other than a stale epoch | warning |

The engine's events (`GET /v1/engines/{engine}/events`) and its status say why. Each alert's
`runbook_url` points at its section of `docs/RUNBOOK.md`: what it means and what to do.

The gate checks the dashboard is valid JSON (`scripts/preflight.sh`), and that the dashboard and
the alerts name only series the code exports, match only label values it writes, and point at
runbook sections that exist (`controller/tests/dashboards.rs`). CI loads the rules with `promtool check rules` and
checks what they raise over given series with `promtool test rules` (`alerts.test.yml`).
