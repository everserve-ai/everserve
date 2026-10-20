# 0017: discovery, the lease and topology come from Kubernetes first; files stay for drills

2026-10-06. Status: current. A ruling on `docs/MULTINODE.md` M4.

**Decision.** The pilot's backend is Kubernetes: daemons register through the cluster API, the
coordinator's lease is a Lease object, engines are labelled pods, topology comes from node labels
and the device plugin. The file backend stays, behind the same `Discovery` interface, for drills,
CI and bare metal. The two-node run installs the way the pilot will.
