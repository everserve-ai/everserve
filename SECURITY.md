# Security

Everserve runs under a model serving process and holds its sessions' KV state, so a flaw in it
is a flaw in the serving it protects. Please report one privately.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository (the Security tab, "Report a
vulnerability"). Say what you found, how to reproduce it, and which component (the control
plane, the store, the vLLM adapter, the chart). You will have an acknowledgement within three
working days and a fix or a stated plan within thirty. Please do not open a public issue for a
vulnerability before a fix is released.

## What is in scope

- The control plane's protocols: the daemon to coordinator link, the mirror service, the
  operator API, node certificate issuance (decisions 0026, 0037, 0039).
- The store's shared-memory data plane and its replication (decision 0011).
- The chart's RBAC, NetworkPolicies and pod security settings.
- The adapter's trust assumptions, stated in `adapters/vllm/README.md` and
  `integrations/dynamo/README.md`: the engine's own queues and NCCL carry no authentication and
  rely on the network between an engine's nodes being trusted, as vLLM's multi-node mode does.

## What the design assumes

A node daemon and the processes on its node are one trust domain (the node group of the pod
contract); a node's certificate names the node and nothing else; the coordinator is the one
writer of every engine's epoch and every write it makes is fenced by its lease term. A report
that one of these assumptions does not hold in a deployment the chart produces is a
vulnerability.

## Supported versions

The latest release. There is no long-term support branch yet.
