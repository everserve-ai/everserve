# 0043: a spare's node fetches its position's compile cache

2026-10-14. Status: current; built, both halves: the control plane's (below) and the worker's (the
serving worker writes its manifest, `everserve_vllm.worker`; a spare asks `CacheWanted` through
`everserve.contract.daemon` from `everserve_vllm.node_main` and checks what landed before it loads,
`everserve_vllm.staging`), with the pod contract's cache directory (`core/everserve/podcontract.py`,
mounted by the Dynamo integration).

**Context.** Decision 0029 makes an engine's compile cache part of each position and has the
deploy stage it to every node that may host one, so no node autotunes by itself; a spare whose
pair (key, digest) differs from its rank's is not admitted. Under Kubernetes nothing stages it.
The seventh GPU proof under Dynamo (2026-10-14) read the consequence: the spare for rank 1
compiled for itself, its key the engine's and its digest its own, and was refused, so rank 1 had
no cover and its node's loss stalled the engine. A spare must run on its rank's kernels, and the
deploy cannot be what puts them there: a spare's pod is created by the framework when and where
its scheduler decides.

**Options.** A volume every node mounts read-write (ReadWriteMany) at vLLM's cache directory makes
a matching cache likely, not certain: the first warm-up and every later one write into it,
autotune entries land from whichever process writes last, an artifact outlives the engine it was
built for, and every customer must provide the storage class. Compiling on the spare with
autotuning off is not enough: Inductor still benchmarks pointwise kernels, and 0029 found a launch
on a loaded artifact re-deciding a configuration by timing. Carrying the files makes the pair
equal by construction.

**Decision.**

- *Where a node keeps it.* The pod contract gains a host directory for compile caches,
  `/var/lib/everserve/vllm-cache`, the node group's (2770), beside the daemon's and the shared
  memory's. An engine's pods mount it and point `VLLM_CACHE_ROOT` at it (the integration's
  renderer sets both); the node's store mounts it to serve and to write what it fetches.
- *What moves is what the digest covers.* vLLM loads rank r of an engine with key K from
  `torch_compile_cache/torch_aot_compile/<K>/rank_<r>_0/model` and reads the kernels that graph
  names from `<K>/inductor_cache` (vLLM 0.31, which sets `TORCHINDUCTOR_CACHE_DIR` there). The
  file set is the one `compile_cache.rank_digest` hashes: r's subtree, the kernel sources its
  graph names, and their autotune results. Nothing else under `<K>` is read. Triton's built
  kernels, outside the digest (0029) and outside `<K>`, are designed for and not carried: a cold
  Triton cache costs a spare compile time, not exactness. Moving exactly that set means a spare that lands it reports the
  rank's digest, and one that does not is refused as today.
- *One list of it.* The serving worker writes `<K>/rank_<r>_0.manifest.json` from the same pass
  that computes its pair: the key, the rank, the digest, and every file the digest hashes, its
  path relative to `<K>`, its size and its sha256. It is written under a temporary name and renamed, the node group's
  (0660), after the pair is computed and before the process registers, so a node named as a
  holder has its manifest. The file set has one implementation, the digest's; the stores move
  what the manifest lists and never derive it.
- *Who holds it.* A serving rank's node holds its rank's set once the rank registered its pair
  after its warm-up; a spare admitted for the rank holds it too. The coordinator names each rank's
  pair and its holders, the serving rank's node first, in the placement it already pushes to every
  daemon (0041), so a spare's node never searches.
- *How it moves.* The spare's node daemon asks its own store, on the node's socket, to fetch the
  rank's set from the holders in order; the store fetches from a holder's store over the
  replication link, store to store under the stores' certificates as rows move, so no new kind of
  caller reaches a store. The holder serves only a manifest whose digest is the pair asked for,
  and checks each file's sha256 as it reads it; the fetching store checks each again as it lands,
  writes into a sibling temporary directory under the node's cache directory, and renames it into
  place with the manifest last, so vLLM never reads a partial artifact. A holder that fails is
  followed by the next. Bulk bytes between nodes stay the store's alone.
- *What the spare does.* It waits on its link for its rank's pair, then for its node's daemon to
  have the set, then recomputes the digest over what landed, then loads, compiling nothing. On a
  mismatch it refuses (`STOP_CAUSE_REFUSED`, both digests) and never compiles instead. A spare that
  waits past its bound is refused; it leaves with a refused goodbye naming why, which is the
  engine's event. The received manifest is written
  last; a spare that fetched writes nothing of its own, and its recompute must equal the
  manifest's digest. An admitted spare is then a holder like the serving rank's node.
- *What bounds it.* A node keeps at most `compileCacheBytes` of compile caches beside the store's
  row budget; a fetch past it is refused, and the spare with it.

**On the wire.**

- `RankPlacement` carries the rank's pair (`compile_cache`) and its holders (`cache_holders`, the
  stores' replication addresses, the serving rank's node first).
- A spare told to stand sends `CacheWanted { engine, rank, dp }` (`Signal` 22) on its undecided
  connection at once, for its rank or any rank of its stage, and
  waits for the answer only before its compile step, so the fetch overlaps its weight load. The
  daemon answers `CacheReady { engine, rank, key, digest }` (23) when its node holds the manifest
  with the rank's digest, fetching it first if it does not, or `CacheRefused { engine, rank,
  reason }` (24) past its bound (10 minutes, as a launched spare's warming), each ask in turn. The
  daemon sends these only in answer to an ask, so a process of an older build never gets one. `dp` is the
  data-parallel index of the rank's directory (`rank_<r>_<dp>`), 0 for every engine placed today,
  so the key never needs a second meaning.
- The daemon asks its store `CacheFetch { key, rank, dp, digest, holders }` on the node's socket,
  answered `Done` once the set is in place, or the link's error reply.
- Between stores, pulled one reply per request as the link answers everything:
  `CacheManifestQuery { key, rank, dp, digest }` answered by the manifest, then
  `CacheRead { key, rank, dp, path, offset }` answered by `CacheChunk { data }` of at most 4 MiB,
  under the 16 MiB frame cap. A refusal (no such manifest, another digest, a path the manifest
  does not list) is the link's error reply. The fetching store paces itself.
- `<K>` and `<K>/inductor_cache` are the node group's and group-writable (2770, files 0660),
  since vLLM writes autotune and recompiled kernels there after a load; the fetched manifest and
  `rank_<r>_<dp>/` stay as they landed. A later write that rewrites a kernel the rank's graph
  names changes the rank's digest, and the next registration reports it.

**Consequences.**

- The store reads and writes the node's cache directory: the chart mounts it in the store's
  container, beside the shared memory it already mounts.
- A host directory outlives a pod: a pod restarted on its node finds its own cache, and loads
  rather than compiles, whether or not a spare ever fetched there.
- Both new `Signal` kinds are decoded on every Python link, with a test that an unknown one is
  skipped, before any process sends them.
- A spare standing for any position (0030) holds no shard and no cache until a change names its
  rank; fetching at the take is a cost of that change. Designed for, not built: the any-rank spare
  fetches at its take.
- The coordinator's checks are unchanged: the engine's key, each rank's pair, and
  `compileCacheConsistent` read what processes report, and a fetched set is reported like a staged
  one.

**Built** (2026-10-14), the control plane's half. The store (`store/src/cache.rs`): a holder serves
a manifest only for the pair asked and only the files it lists, each checked against its sha256
before its end is served; a fetching store pulls one chunk per request, past a holder that fails
to the next, checks each file's size and sha256 as it lands, renames each into place and writes
the manifest last, within its budget; a fetch is asked only on the node's own socket. The
coordinator's placement carries each rank's pair and its holders. The daemon answers a spare's
ask from the placement through its store. The chart mounts the node's cache directory in the
store's container. Tested (`store/tests/compile_cache.rs`, `controller/tests/roles.rs`): a set
fetched past a dead holder lands whole and its node serves it next; another digest, a key or path
leaving the directory and an unlisted path are refused; a file altered on the holder is never
served whole, and bytes a holder serves that are not what it listed never land; a fetch past the
budget writes nothing; a peer may not ask a store to fetch; the placement names a rank's pair and
its holders in order; a spare told to stand is given its rank's set from the serving rank's node,
and a node with no store refuses. The worker's half (the manifest, the ask, the
recompute) and the pod contract's cache directory followed, by their owner: the status line names
where each lives.
