# 0014: the coordinator is a separate leased process; node daemons are its agents

2026-10-06. Status: current; the lease's term as a fence is 0036. A ruling on `docs/MULTINODE.md` M1.

**Decision.** One coordinator per cluster, a Rust process with no GPU, holds a lease in the
cluster's API (a Kubernetes Lease; a file lock on bare metal) and is the only writer of every
engine's epoch. Node daemons keep the worker links, the local store, local health and local
notice intake, deliver the coordinator's changes and return the acknowledgements. The coordinator
holds no state that cannot be rebuilt: on start or on taking the lease it asks every daemon which
engines it serves, at what epoch, with which members, and resumes.

**What the operator provisions.** A Deployment with one replica, restarted by Kubernetes if it
dies; a second replica only shortens the gap. No GPU, no pre-provisioned spare is involved. While
no coordinator holds the lease, engines keep serving; a fault in that window leaves its engine
paused at the step guard until the lease is held again. Sessions wait; they are not lost.

**Why not election among daemons.** Fewer processes, but the election, the handoff and the
assignment of engines to daemons would live inside the component that runs on every node and
touches workers. A leased process is one sentence to operate and maps to what a neocloud runs.
