# 0039: the operator API knows who asks

2026-10-13. Status: current; built.

**Context.** The coordinator's operator API (`/v1/drain`, `/v1/nodes/{node}/cordon`,
`/uncordon`, `/undrain`, `/v1/pools/{pool}`, and the reads beside them) is served over plain HTTP
on the metrics port, to anyone who can reach it. A drain moves an engine's ranks off a node; on
Kubernetes anything in the cluster network could ask for one. A spec write records who wrote it
(`written_by`, decision 0036), but the name is one the caller chose. A due-diligence review fails
this on sight.

**How others do it.** A Kubernetes component that serves its own API puts it behind the cluster's
identity: the caller presents its credential, the component asks the API server who it is
(`TokenReview`) and whether it may (`SubjectAccessReview`, with the request's verb and resource),
and caches both answers briefly. kube-rbac-proxy and every aggregated API server work this way,
so an operator grants access with the RBAC they already use. Outside Kubernetes, services
authenticate a client certificate from their own CA and map its name to a role (etcd, CockroachDB,
Consul).

**Decision.**

- The operator API leaves the metrics port for its own TLS listener (`--api-addr`), served with
  the coordinator's certificate. `/metrics` and `/healthz` stay plain on the metrics port: a
  scraper and a probe read nothing an attacker can use, and they cannot carry credentials.
- Every request is authenticated, reads included: readiness and topology describe a cluster's
  GPUs and engines. A caller presents a client certificate chained to the deployment's CA (its
  name is its identity; `everserve-controller certs --operator <name>` issues one), or, on
  Kubernetes, a bearer token the API server's `TokenReview` authenticates (its user and groups are
  its identity). Neither: 401.
- Every request is authorized as a verb on a resource of the `everserve.ai` group: `get` or `list`
  on `readiness`, `topology`, `nodes`, `events` and `pools`; `update` on `nodes` (drain, cordon,
  uncordon, undrain) and `pools`, with the node's or pool's name; `create` on `stacks`, which
  signals live processes to dump theirs. On Kubernetes, a
  `SubjectAccessReview` decides, so an operator is granted `everserve.ai` verbs by RBAC; the chart
  ships `everserve-viewer` and `everserve-operator` ClusterRoles. Outside Kubernetes, the
  coordinator's settings name the operators and the viewers by certificate name. Not allowed: 403.
- Answers are cached for ten seconds, as kube-rbac-proxy does, so a polling scheduler costs the API
  server one review per caller per ten seconds, and a revoked grant ends within ten seconds.
- Every decision is logged with the identity, the verb and the resource; every write records the
  authenticated identity as its writer, never a name the caller supplied.

**Consequences.**

- The operator's tools (`everserve-controller check`, the bare-metal smoke, the drills' drivers)
  present credentials: an operator certificate, or a token.
- The coordinator's service account may create `TokenReview`s and `SubjectAccessReview`s
  (`system:auth-delegator`), which 0037's issuer needs as well.
- A daemon's or a store's certificate is not an operator's: its name is a node's, and the role
  settings name operators, so a compromised node cannot drain its neighbours.

**Built** (2026-10-13). `crate::api`: the TLS listener (a client certificate optional, verified when
presented), the guard that maps each route to its ask before the route runs, the certificate gate
(operators and viewers named under `operators.everserve`, a domain no node's certificate can be
valid for) and the Kubernetes gate (`TokenReview`, `SubjectAccessReview`, ten-second answers, an
unanswering API server never cached). `certs --operator`, `check --coordinator-api` with an
operator's certificate or a token, the bare-metal smoke over the API, and the chart's delegation
binding and `everserve-viewer` and `everserve-operator` roles. Tested: no credential is refused, a
viewer reads and writes nothing, an operator's write is recorded as theirs, a node's certificate
is no operator's even when the settings name the node, and on a real API server a token's rights
are the RBAC its account was bound to; letting a viewer write, dropping the operators' domain,
skipping the access review, letting no credential through or dropping the delegation binding each
fails a test.
