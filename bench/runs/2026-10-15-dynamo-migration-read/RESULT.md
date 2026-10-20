# Dynamo 1.5.1's request migration and routing, read at the pin (for 0038's E5)

Read on 2026-10-15 at the tag `v1.5.1` of `ai-dynamo/dynamo`, the release the integration pins:
- `lib/llm/src/migration.rs`
- `lib/runtime/src/pipeline/network/egress/push_router.rs`

Decision 0038's strategy C (Cort, 2026-10-15) leaves the stream to the router. This record is what
the router does, so that E5 relies on nothing unread. Dynamo's `main` was read first; 1.5.1 agrees
with it on every point below.

## Migration (`migration.rs`)

- **A broken stream is re-sent.** When a stream fails mid-generation, the frontend re-sends the
  request to another instance, and the request's `token_ids` are extended by every token already
  generated (`self.request.token_ids.push(*token_id)` for each). The retry carries the whole
  context, and the client is not sent again what it already has.
- **The failures that are migrated:** `CannotConnect`, `Disconnected`, `ConnectionTimeout`,
  `ResponseTimeout`, `Backend(EngineShutdown)`, `Backend(StreamIncomplete)` and
  `WorkerOverloaded`. `Cancelled` and `ResourceExhausted` are not.
- **The limit:** `migration_limit` counts the retries after the first attempt. The constructor
  takes it from configuration; this file sets no default. When the retries are spent, the request
  fails ("Migration limit exhausted").
- **No delay between retries.** No sleep, backoff or wait for an instance: a retry follows the
  last failure at once.
- **No instance to send to** gives `CannotConnect` ("no responders") for a new stream, or
  `Disconnected` for one under way. Both are migratable, so each consumes a retry at once.

## Routing (`push_router.rs`)

- **The candidates are the free instances** (`routing_instances.free_ids()`), picked by the
  routing mode: round robin, random, power of two choices, least loaded, or device-aware weighted.
- **A failed instance leaves routing.** An instance that answers `CannotConnect`, `Disconnected`,
  `ConnectionTimeout`, `ResponseTimeout` or `StreamIncomplete` is reported down
  (`report_instance_down`) and kept out. An instance gone from discovery is removed
  (`spawn_instance_removal_watcher`). A `WorkerOverloaded` answer marks it overloaded for a bounded
  lease.
- **With no free instance, a request fails at once:**
  - "All workers are busy, please retry later" (`ResourceExhausted`, not migrated) when instances
    exist but none is free;
  - "No workers available for endpoint" (`Unavailable`) when none is routable.
  There is no `wait_for_instances` before dispatch.
- **No state is registered but held:** there is no "unready", "draining" or zero-weight state in
  which an instance is known and requests wait for it. An overloaded or busy instance is excluded,
  and a request that finds none free fails.

## What E5 takes from this

1. **With two or more engine replicas behind the router**, the production shape, a lost head's
   broken streams migrate at once to another replica and continue there, with that replica
   prefilling the whole context. The client sees one continuous stream. Head failover then gives
   the lost engine its capacity back, with its KV, in seconds rather than a restart.
2. **With a single engine**, no instance is free between the head's loss and the standby's
   registration. Migration spends its retries at once, and the request fails, so clients see
   errors for that window. Nothing in 1.5.1 lets a standby be known but held. The integration's
   documentation says so, and proof-24's single-engine case is not run on 1.5.1.
3. **`migration_limit` must be set** on the frontend's model, since it has no default here. The
   renderer sets it, and the integration's README states it.
