# 0037: a node proves its name to be issued its certificate

2026-10-13. Status: current; built (the chart is tested against a real API server and has run on a
managed cluster without GPUs, every node issued its certificate).

**Context.** Decision 0026 makes a node's certificate its identity on the fabric: the coordinator
believes a daemon's hello, and a store believes a replication link, only for the node name its
certificate is valid for. On bare metal the operator issues each node's certificate, with
`everserve-controller certs` or their own CA. On Kubernetes the node daemon and its store run as
a DaemonSet: one pod template for every node, so no pod can be handed its own node's certificate
the way an operator hands it to a host. Shipping one certificate valid for every node name would
undo 0026; naming a Secret per node by hand undoes the DaemonSet.

**How others solve it.** Every widely deployed answer separates the identity rule from how a
workload proves itself to be issued an identity. Istio's agent proves its pod with the pod's
projected service account token, which istiod checks with the API server (TokenReview) before it
signs the pod's certificate; Linkerd's identity controller does the same. SPIRE issues one kind of
identity behind pluggable attestors: `k8s_psat` (projected tokens) on Kubernetes, `aws_iid` on EC2,
`x509pop` for a machine already holding a certificate, a join token as the fallback. The kubelet
itself gets its serving certificate through a signing request the control plane approves.

**Decision.** 0026's rule stands unchanged on every platform: a peer is who its certificate says.
What a platform decides is how a node comes to hold that certificate.

- *Bare metal*: as today. The operator issues a long-lived certificate per node.
- *Kubernetes*: the coordinator issues it, as istiod does. A node daemon starts with no
  certificate. It asks the coordinator's issuing endpoint (TLS, authenticated by the coordinator's
  certificate only) for one, presenting its pod's projected service account token with the
  audience `everserve` and a public key it generated. The coordinator sends the token to the API
  server's TokenReview and accepts it only when it is authenticated for that audience, belongs to
  the node daemons' service account in the deployment's namespace, and is bound to a pod: the
  API server then states the pod's node (`authentication.kubernetes.io/node-name`). The coordinator
  signs, with the deployment's CA, a certificate for that node name, and the name stores verify on
  a peer, valid for hours. Nothing the daemon says chooses the name: a token from a pod on node-a
  is issued node-a's certificate or none. The daemon and its store take the certificate, renew it
  at half its life, and use the renewal without a restart.

**How a node holds it.** One process in the node pod, `everserve-controller node-cert`, is the
issuer's client; the daemon and the store only read files. It makes the key, asks, and writes
`node.key`, `node.crt` and `ca.crt` into one directory shared with the daemon and the store, each
file replaced in one step (written beside it, then renamed), the key first. It runs twice: as an
init container with `--once`, which waits out a coordinator not yet serving, so the daemon and the
store start on a certificate; and beside them, renewing at about half the life left (a quarter of
that either way, so a fleet issued together does not renew together), retrying a failure from
5 s, doubling to 1 min (a fresh install starts its coordinator and its nodes together, and a
node should not wait minutes for an issuer already up), while the files in place stay in use. It verifies the coordinator by the
deployment's CA certificate, which is public and given to the node pods as such; the CA key never
leaves the coordinator.

Every process with TLS material (the coordinator, the daemon, the store) reads its files again
every 30 s and uses what changed for every handshake after: a listener already bound presents the
renewed certificate, a connector already made presents its renewed client certificate, and a
connection already open keeps the certificate it was made with. A key that is not its
certificate's (the two read between their writes), or a file that does not parse, replaces
nothing: the material in use stays, and the next read tries again. The same reload serves a
Secret that cert-manager rotates, on any platform.

This rests on the API server binding a pod's node into the tokens it issues for that pod
(KEP-4193's pod and node claims), which the tests check against the `kube-apiserver` they run
(1.37): TokenReview of such a token states `node-name`. A cluster whose API server does not state
it is refused, with that reason, rather than given a weaker identity.

**Consequences.**

- The coordinator holds the deployment's CA key on Kubernetes, as istiod holds the mesh's. Its
  Secret is readable by the coordinator's service account alone, and the chart creates the CA if
  the operator brings none.
- The coordinator's service account may create TokenReviews (`system:auth-delegator`); the node
  daemons' may read the lease it follows and its own node.
- A certificate lives for hours, not years: a node removed from the cluster stops being renewed,
  and no long-lived node key exists to steal.
- A pod that is not the node daemon, in any namespace, is issued nothing, whatever audience its
  token carries.
- *Later*, not built: bare metal on the same issuing path, attested by `x509pop` (the operator's
  long-lived certificate proves the node; the coordinator issues the short-lived one), which
  gives bare metal rotation without rerunning `certs`.

**Landings.** (1) This decision, the container image, and the chart's coordinator side: its
Deployment with a standby under a `kubernetes://` lease, and RBAC that lets it take, renew and
release its `Lease` and never delete it, tested against a real API server. (2) The issuing
endpoint and its TokenReview check, tested with tokens bound to pods on nodes of a real API
server, a token from node-a asking for node-b's name included. (3) `node-cert`, the issuer's
client, and TLS material every process reloads in place: tested by a node issued its certificate
through the operator API's listener and renewing it under a process that reads the files, a
listener already bound presenting the renewal, and a certificate and key from two renewals
refused while the pair in use stays. (4) The node DaemonSet: `node-cert` as an init container and
beside the daemon and the store; the pod contract's host directories given to the node group by
an init container that owns them; RBAC that lets the daemons read the Lease and Nodes and write
neither, and keeps the CA's key from them; tested by the chart applied to a real API server, its
paths and socket read against `core/everserve/podcontract.py`.
