# Review: the control-plane landings of 2026-10-06 against their neighbours

*A review note from 2026-10-06: a snapshot, kept for the record and not maintained. The current design is docs/ARCHITECTURE.md and docs/decisions/.*

Read against the sources on 2026-10-06, not from memory, after the landings were on main:
CockroachDB `pkg/cli/cert.go` and `pkg/security/certs.go`; `aws/aws-node-termination-handler`
`pkg/ec2metadata/ec2metadata.go`, `pkg/monitor/spotitn/spot-itn-monitor.go`; Microsoft Learn
"Scheduled Events for Linux VMs" and "Azure Instance Metadata Service"; Google Cloud "Spot VMs",
"Create and use Spot VMs", "Getting live migration notice", "Default metadata values";
`ai-dynamo/dynamo` `deploy/observability/`, `lib/runtime/src/system_status_server.rs`,
`lib/runtime/src/tls_utils.rs`; `prometheus-community/ansible`
`roles/node_exporter/templates/node_exporter.service.j2`. The cluster autoscaler's taint source
(`ToBeDeletedByClusterAutoscaler`) could not be located in the repository's current layout; its
semantics stand on the autoscaler's documented behaviour and are marked unverified below.

Each row: what they do, what we did, the ruling. Rulings marked **changed** landed in the commit
that carries this file.

## `everserve-controller certs` against CockroachDB `cert`

| they | we | ruling |
|---|---|---|
| keys `0600`, certificates `0644` | the same | same |
| never overwrite an existing file without `--overwrite` | reissued every named certificate on every run | **changed**: an existing pair is kept and reported; `--overwrite` replaces it. A rerun that adds a node now touches nothing a running process holds |
| CA key reuse only with `--allow-ca-key-reuse` | the CA in the directory is reused | differ, deliberately: reuse is the command's purpose (adding nodes under one CA) and the README says so; a new CA is a new directory |
| `create-node <host>...` takes the node's hosts and IPs as SANs | the node's name and the store peer name only | differ, by design: daemons verify the coordinator by `--coordinator-name` and stores verify `--peer-name`, never by host, so the SANs are those names |
| `cert list` | none | not now |

## Notice sources against the clouds' own handlers and documents

| they | we | ruling |
|---|---|---|
| NTH: `PUT /latest/api/token` with the TTL header, `GET spot/instance-action`, 404 is nothing, RFC 3339 `time` | the same | same |
| NTH: HTTP timeout 2 s with retries; the monitor runs every 2 s | one 10 s request timeout, retried at the poll interval; poll was 5 s | **changed**: poll default 1 s (Azure recommends once a second; the endpoints are link-local) |
| NTH also consumes the rebalance recommendation | not read | not now: advisory, no deadline, AWS-specific; a neocloud's own reclaim is our case |
| GCP: the preemption notice is 0 s by default and 120 s when the VM asks for it; the flag flips, then ACPI soft-off, then up to 30 s of shutdown | "about 30 s" in the docs, a 30 s lead | **changed**: the docs state GCP's figures; the lead is documented as the VM's preemption notice; the 30 s default stays as the only sensible bet when the operator sets nothing |
| GCP: `maintenance-event` flips 60 minutes before a terminating host maintenance; GPU VMs cannot live-migrate | not read | **changed**: the GCP source polls it; `TERMINATE_ON_HOST_MAINTENANCE` is a notice 60 minutes out |
| Azure: events are delivered to every VM in the availability set or placement group; `Resources` says which | took every leaving event | **changed** (a real defect): the source learns this VM's name from `instance/compute/name` once and keeps only events naming it; a VM that cannot learn its name takes every event and says so |
| Azure: a hardware failure skips `Scheduled` and the event arrives `Started` with an empty `NotBefore` | ignored `Started` | **changed**: a started leaving event is due now |
| Azure: poll once a second | 5 s | **changed**, as above |
| Azure: `Freeze` pauses the VM for seconds | excluded | same; a long freeze would stall collectives, which no notice source can prevent. Open |
| Azure: approving an event by `POST` can shorten the wait | not done | not now |
| cluster autoscaler: `ToBeDeletedByClusterAutoscaler` with the deletion time as its value, `node.kubernetes.io/unschedulable` for a cordon | the same | **unverified against source at the time**; the file moved in the repository. Stays until a cluster run checks it (item 6b) |

## Readiness and metrics against Dynamo

| they | we | ruling |
|---|---|---|
| one HTTP port serving `/metrics`, `/health` (JSON `status` and `endpoints`), `/live`, and the operator routes | one port serving `/metrics`, `/healthz`, `/v1/readiness`, `/v1/topology`, `/v1/drain` (since replaced: the operator API is on its own authenticated listener, decisions 0039 and 0042) | same shape; a `/live` beside `/healthz` is not needed while `/healthz` is the liveness |
| Grafana dashboards and a Prometheus setup in `deploy/observability/`, as Kubernetes ConfigMaps | `deploy/grafana/` as a dashboard file and a scrape config | same practice; ours becomes a ConfigMap in the chart |
| `tls_utils`: rustls with `rustls-pemfile`, a handshake timeout, self-signed material in tests only | the same loaders; no handshake timeout; a certificate tool for operators | **changed**: `transport::HANDSHAKE_TIMEOUT` bounds a connect and a handshake on both sides. The tool is ours to keep (CockroachDB is the precedent) |

The listener still performs the TLS handshake inside `accept`, so a slow handshake holds the
next accept for up to the timeout; moving the handshake into the per-connection task is a
follow-up.

## systemd units against the node exporter's

| they | we | ruling |
|---|---|---|
| `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `ProtectControlGroups`, `ProtectKernelModules`, `ProtectKernelTunables`, `SyslogIdentifier` | the first three | **changed**: the rest added |
| `Restart=always`, `StartLimitInterval=0` | `Restart=on-failure` | differ, deliberately: an operator's stop must stick, and a crash loop should be visible, not infinite |

## What this review did not cover

The Kubernetes chart, lease and discovery (not built); the harness; the adapter.
