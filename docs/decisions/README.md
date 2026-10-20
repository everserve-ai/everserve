# Decisions

Dated records of choices that are not derivable from the code. One file per decision.
Format: context, decision, consequences. A reversed decision gets a new file that links the old.

A record that names a commit by hash names one in the private archive the repository was
published from (`README.md`, History), unless the hash is dated after 2026-10-20.

**Status.** The first line under a record's title is its date and its status, so a reader knows
whether the record describes the system. The vocabulary:

- `current`: the decision stands and the system is built to it, or the record says what of it is
  built.
- `amended by NNNN (date); current as amended`: a later record changed part of it, and says which;
  the rest stands. `as amended below` means the record carries its own dated amendments.
- `reversed by NNNN (date)`: a later record undid it; the text stays as the record of the design or
  of the mechanism's proof.
- `superseded by NNNN (date)`: a later record carries the same decision in a wider form.
- `design, not built` (or `design, partly built`): the record says what is built and what is not.

Short rulings are allowed when they link the option text they ruled on (0014 to 0018 rule on
`docs/MULTINODE.md` M1 to M5).

## Index

| # | Title | Status | Decides |
|---|---|---|---|
| [0001](0001-monorepo.md) | one monorepo, splittable per package | amended by 0046; current as amended | one repository; every package has its own manifest and imports others through their published API |
| [0002](0002-languages.md) | Python where the engines force it, native everywhere else | current | Python for the core API and adapters, C++/CUDA for per-byte paths, Rust for the controller and store |
| [0003](0003-name.md) | the company is Everserve | current | the name and the code namespace |
| [0004](0004-replace-is-the-product.md) | REPLACE on stock model code is the product; SHRINK is a tier; REBUILD is the floor | amended by 0033; current as amended | REPLACE (a spare takes the dead rank, sessions kept) ships for every model; SHRINK is per architecture; REBUILD is the floor |
| [0005](0005-shadow-groups-primary.md) | pre-formed shadow groups are the primary shrink mechanism | current; planned by 0035 | every allowed member set is a communicator formed while healthy; a change is a pointer swap |
| [0006](0006-slot-kernel-p2p-primary.md) | the peer-to-peer slot kernel is the production kernel; host-memory is the fallback | current | two variants of the slot collective, chosen by topology at start |
| [0007](0007-spare-pool.md) | the spare is a controller-managed pool resource; per-engine pre-loaded spares ship first | amended by 0030 and 0031; current as amended | spares are a pool the controller owns, with a loading state per spare |
| [0008](0008-daemon-out-of-process.md) | the controller runs out of process; in-process bindings only for kernels | current | the node daemon is its own process; workers link to it over a Unix socket |
| [0009](0009-async-commit-one-step-fence.md) | the mirror commit is asynchronous with a one-step fence | amended by 0010 and 0019; current as amended | commit first, fence before the agreement, counts with the rows |
| [0010](0010-step-thread-never-reads-the-device.md) | the step thread never reads device memory | current | no device read on the step thread while rows are saved or committed |
| [0011](0011-shared-memory-data-plane.md) | the mirror's data plane is shared memory; the socket carries control | current | rows never cross the socket on a node; the store's tables are files on shared memory |
| [0012](0012-slot-order-and-device-control-words.md) | the slot collective sums in rank order, and reads its membership from device memory | amended by 0022; current as amended | summation order is the model's, not the region's; member order lives on the device |
| [0013](0013-exact-replay.md) | the replay is exact: from the committed position, one step at a time, in the same shape | current | a committed position is never recomputed; the tail replays as decode steps on captured graphs |
| [0014](0014-coordinator-is-a-leased-process.md) | the coordinator is a separate leased process; node daemons are its agents | current | one leased coordinator per cluster, holding nothing it cannot rebuild from the daemons |
| [0015](0015-fabric-collectives.md) | fabric collectives run outside the graphs first; the fabric slot kernel ships in phase 2 | current; fabric slot kernel not built | cross-node collectives run outside captured graphs through a handle that reads the member set |
| [0016](0016-replicate-to-spare-nodes.md) | a rank's rows replicate to the nodes holding its eligible spares, and nowhere else | reversed by 0030 | replication targets were the spare nodes only |
| [0017](0017-kubernetes-discovery-first.md) | discovery, the lease and topology come from Kubernetes first; files stay for drills | current | the Kubernetes backend is the pilot's; the file backend stays for drills, CI and bare metal |
| [0018](0018-two-node-run-shape.md) | the two-node run is three 4x H100 nodes, a 70B model at TP4xPP2, and a spare node | reversed by 0030 as the rental's shape | the shape of the first multi-node measurement |
| [0019](0019-commit-descriptor-and-fence-depth.md) | a step's commit is one descriptor, and the fence runs two steps behind | current | one message per step whatever the layer count; two steps in flight |
| [0020](0020-prepared-tables.md) | a rank's tables are created and touched at registration, and a commit copies runs | current | page faults paid once before serving; consecutive slots copied as one run |
| [0021](0021-detection-at-the-collective.md) | a lost rank is detected at the collective, not at its process's exit | current, as amended | survivors detect a lost rank in the slot kernel and by the step-stall rule; a worker leaving its daemon is a loss |
| [0022](0022-replacements-stay-in-the-domain.md) | a replacement stays in its captured group's domain, or the whole group moves | current | a tensor-parallel group moves whole when its spare is outside the domain |
| [0023](0023-node-loss-at-the-coordinator.md) | a node is lost when its daemon's link ends without a goodbye, or misses its heartbeats | current | node loss is decided by the coordinator from the daemon link |
| [0024](0024-daemon-owned-spares-and-the-opaque-join.md) | the daemon owns the spare processes on its node, and the join is an opaque exchange | current | daemons launch and supervise spares; the join payload is bytes the control plane never reads |
| [0025](0025-refuse-now-release-beside-the-change.md) | a change refuses its dead groups now and releases them beside the change, bounded | current, as amended | refusal is the change; freeing communicators runs beside it; a leaked release retires its process by notice |
| [0026](0026-a-peer-is-who-its-certificate-says.md) | a peer is who its certificate says | current, as amended | a certified peer may claim only the name its certificate is valid for, on both links |
| [0027](0027-every-first-message-carries-the-protocol.md) | every first message carries the protocol, and another protocol is refused by name | amended by 0040; current as amended | one protocol version in `signals.proto`, carried on every first message |
| [0028](0028-spares-resident-in-host-memory.md) | a spare that holds every shard in host memory and loads the one it takes | design; superseded by 0030 | a resident spare copies its shard from host memory at the take |
| [0029](0029-the-compile-cache-is-part-of-a-position.md) | the compile cache is part of a position | current | every process is held to its position's compile key and digest; timing never chooses numerics |
| [0030](0030-a-persistent-logical-engine-over-a-mutable-placement.md) | a persistent logical engine over a mutable placement | current, as amended | the engine is a logical object over five primitives; guarantees are stated per transition class |
| [0031](0031-a-spare-pool-belongs-to-the-fleet.md) | a spare pool belongs to the fleet | current, as amended; landings 1 to 3 built | pools serve an engine shape across engines; a node's role is derived from what it holds |
| [0032](0032-an-engine-is-alive-while-its-head-is.md) | an engine is alive while its head is | current, as amended | the head registers; its loss is the engine's; a failed change puts the engine down; a downed engine starts anew |
| [0033](0033-one-transition-machine-resize-is-a-membership-change.md) | one transition machine; resizing an engine is a membership change | design, partly built | every placement change is one machine with one set of phases; the kinds differ in their plan |
| [0034](0034-layers-and-seams.md) | layers and seams | current, as amended | seven layers, five contracts, five identities kept apart; capability checked before movement |
| [0035](0035-cover-plans.md) | cover plans | current, as amended | the coordinator plans the sets an engine holds; coverage is reported and gates REPLACE; groups are shared across sets |
| [0036](0036-the-engine-controller.md) | the engine controller: desired state, observed state, reconciliation | current, as amended | spec and status per resource, a reconcile loop, a fenced single writer, the Kubernetes spec store |
| [0037](0037-a-node-proves-its-name-to-be-issued-its-certificate.md) | a node proves its name to be issued its certificate | current; built | on Kubernetes the coordinator issues a node's certificate against a TokenReview of its pod's token |
| [0038](0038-a-gpu-node-is-disposable-and-the-head-fails-over.md) | a GPU node is disposable, and the engine's head fails over | reversed by 0045 (the head's failover); the change log stands | the coordinator's change log survives it (stands); the head's failover (reversed) |
| [0039](0039-the-operator-api-knows-who-asks.md) | the operator API knows who asks | current; built | TLS, client certificates or TokenReview, verbs on resources, RBAC on Kubernetes |
| [0040](0040-builds-serve-a-range-so-upgrades-roll.md) | builds serve a range of protocols, so upgrades roll | current; built | each build declares the oldest protocol it serves; a link speaks the lower version |
| [0041](0041-inside-the-instance-and-beneath-the-router.md) | inside the instance, beneath the router | amended by 0045; current as amended | the framework owns decisions between instances, Everserve within one; Dynamo first and only |
| [0042](0042-the-operator-apis-resources.md) | the operator API's resources | current; built | every route is a resource with metadata, spec and status; list, watch, optimistic concurrency, one error shape |
| [0043](0043-a-spares-node-fetches-its-positions-compile-cache.md) | a spare's node fetches its position's compile cache | current; built | the store moves the manifest's file set between nodes; the spare checks the digest before it loads |
| [0044](0044-a-spare-is-replayed-what-its-rank-was-sent.md) | a spare is replayed what its rank was sent | current; built | a spare runs the core's start-up calls, replayed over a queue, never derived |
| [0045](0045-the-heads-loss-is-a-known-gap.md) | the head's loss is a known gap | current; reverses 0038 | the head's failover is removed; its loss restarts the engine and the router migrates its streams |
| [0046](0046-apache-2.0.md) | the repository is public under Apache-2.0 | current; amends 0001 | Apache-2.0 for everything, SPDX on every file, DCO sign-off |

## Proposing a record

A new decision is a new numbered file, `NNNN-short-title.md`, with context, decision and
consequences, and the date-and-status line as the first line under its title (`current` at
first; `design, not built` when it is written before the code). A record that amends or reverses
an earlier one says so in its own status line and links the earlier file, and the earlier record's
status line gains `amended by NNNN` or `reversed by NNNN` with a link back. Cite a measurement by
its run directory under `bench/runs/`, not by a number or an attempt alone. Add the record to the
index above.
