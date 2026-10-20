# The control plane on Kubernetes

The Helm chart in `everserve/` deploys the coordinator, and the node daemons with their stores
as a DaemonSet, each node issued its certificate by the coordinator (decision 0037). The chart is
applied to a real API server in the tests, and has run on managed Kubernetes without GPUs
(`bench/runs/2026-10-13-k8s-e2e`) and under the Dynamo proofs with them
(`bench/runs/README.md`).

| what | how | built |
|---|---|---|
| the image | `docker build -t <registry>/everserve:<tag> .` at the repository's root: `everserve-controller` and `everserve-store`, distroless, non-root; CI builds it and runs both | yes |
| the coordinator | a Deployment of two: one holds the `kubernetes://` Lease and acts, the other stands by on another node; a holder that loses the Lease exits 75 and is restarted to stand by (0036). Every pod is ready while it stands, so `helm --wait`, a rollout and the disruption budget see both; the holder labels its own pod `everserve.ai/lease-holder=true`, clears it at its start and when its hold ends, and the `<release>-coordinator` Service selects that pod alone | yes |
| its RBAC | create, get and update its one Lease; never delete it, and no other Lease; get and patch pods of its namespace, for its label; tested against a real API server | yes |
| the operator API | TLS on port 7443 of the `everserve-coordinator` Service; a caller's bearer token is reviewed by the cluster (`TokenReview`, `SubjectAccessReview`, 0039); grant with a RoleBinding in the release's namespace to `everserve-viewer` (reads and watches), `everserve-operator` (also drains, cordons, pools, stacks) or `everserve-launcher` (a pool's spare launcher, the command every node of the pool runs) | yes |
| the policy | `.Values.policy`, protobuf JSON as in `docs/examples`, into a ConfigMap; a new policy rolls the coordinators | yes |
| node daemons and stores | a DaemonSet on every node (`node.nodeSelector`, `node.tolerations`): the daemon and the store, and `node-cert`, which is issued this node's certificate from the pod's projected token (audience `everserve`) by the coordinator after `TokenReview`, before they start, and renews it at about half its 12 h life; every process uses a renewal without a restart (0037). The two host directories of the pod contract are the node group's (`node.group`, 64000, `podcontract.NODE_GID`), mode 2770; the daemon, the store and every engine process on the node are its members | yes |
| network | NetworkPolicies (`networkPolicy`): the coordinator's link (7000) and the stores' replication port (7100), both mutual TLS, reached from the node daemons' pods alone; the operator API (7443) from them and `networkPolicy.apiFrom` (every namespace by default, each caller authenticated by the cluster's RBAC); `/metrics`, which serves without authentication, from `networkPolicy.metricsFrom` (the `monitoring` namespace by default). The node certificate endpoint, outside the API's gate, bounds its body, refuses a bearer that is no token unread, and reviews at most 16 tokens at once | yes |
| node RBAC | the daemons read the coordinator's Lease and Nodes, and write neither; the CA's key is in the coordinator's Secret, which they cannot read; tested against a real API server | yes |
| desired state across a failover | the coordinator keeps its specs (cordons, drains, pools) and each engine's change in flight, assignments and plan generation in `EverserveState` objects (`crds/`), one root and one per engine, named for `coordinator.specStore`; a standby that takes the lease takes every object before it acts, and a deposed holder's next write is refused and stops it (0036, amended) | yes |

## Install

1. The image: a release's, `ghcr.io/everserve-ai/everserve` at the chart's `appVersion`, which
   the chart names unless `image.tag` is set; or build and push your own:
   `docker build -t <registry>/everserve:<tag> .`
2. TLS: `everserve-controller certs --out certs --operator <name>`, then
   `kubectl -n everserve create secret generic everserve-tls --from-file=certs/coordinator.crt --from-file=certs/coordinator.key --from-file=certs/ca.crt --from-file=certs/ca.key`.
   The coordinator signs the nodes' certificates with `ca.key`; no node certificate is made by hand.
3. The namespace: the DaemonSet mounts two host directories, which Pod Security's `baseline` and
   `restricted` profiles forbid, so the namespace enforces neither:
   `kubectl create namespace everserve && kubectl label namespace everserve pod-security.kubernetes.io/enforce=privileged`.
4. `helm install everserve deploy/kubernetes/everserve -n everserve --set-json "policy=$(cat policy.json)" --set-file tls.caCertificate=certs/ca.crt --set node.store.slots=<slots> --set node.store.budgetBytes=<bytes>`
   (add `--set image.repository=<registry>/everserve --set image.tag=<tag>` for an image of your
   own). The store refuses to start on a budget the node's `/dev/shm` has no room for. Every
   object is named from the release (`<release>-coordinator`, `<release>-node`, the ClusterRoles
   too), so two releases in one cluster need two names.

The API server must state a pod's node in the tokens it issues for the pod
(`ServiceAccountTokenPodNodeInfo`, on by default from Kubernetes 1.30): an older one issues no
node a certificate, and the coordinator logs why.

Every coordinator pod serves `/healthz` while it stands; the holder is the pod labelled
`everserve.ai/lease-holder=true`, which the `everserve-coordinator` Service selects for
`/metrics` and the operator API.

## Releases

A tag `vX.Y.Z` publishes the release (`.github/workflows/release.yml`): the image as
`ghcr.io/everserve-ai/everserve:X.Y.Z` and the chart as `oci://ghcr.io/everserve-ai/charts/everserve`
at the same version, each signed by its digest with the release workflow's own identity
(Sigstore keyless) and the image's SBOM attested. The tag, the workspace's version, the chart's
version and its appVersion must agree, or nothing is published; a test keeps the chart's equal to
the workspace's at every commit. Verify an image before it runs:

```sh
cosign verify ghcr.io/everserve-ai/everserve:X.Y.Z \
  --certificate-identity-regexp '^https://github.com/everserve-ai/everserve/\.github/workflows/release\.yml@refs/tags/v' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
cosign verify-attestation --type spdxjson ghcr.io/everserve-ai/everserve:X.Y.Z \
  --certificate-identity-regexp '^https://github.com/everserve-ai/everserve/\.github/workflows/release\.yml@refs/tags/v' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Run by hand (`gh workflow run release.yml`), the workflow is a dry run: every check and build,
nothing pushed or signed. Not yet run on a tag.

## Upgrading

`helm upgrade` rolls the coordinator first; the standby takes the Lease while the holder
restarts. A release serves the protocol of the one before it unless its notes say otherwise
(decision 0040), so the node daemons and the engines move after it, a node at a time, the engines'
processes by planned drains onto spares.
