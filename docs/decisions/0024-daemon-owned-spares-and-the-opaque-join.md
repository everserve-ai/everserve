# 0024: the daemon owns the spare processes on its node, and the join is an opaque exchange

2026-10-07. Status: current.

**Context.** Today's spare is the executor's: launched as a child process over host-local pipes,
bootstrapped over them, and joined to the executor's rings by handing shared-memory queue handles
over the same pipes. None of that reaches another host. The two-node smoke found the head holds no
handle at all to a remote node's processes, so a spare on the spare node cannot be the executor's
in any form. A host-local four-spare join built now would be rewritten when the remote path
existed, and a remote path that shipped vLLM's handles as something the control plane reads would
need a second path for the next engine.

**Decision.** The daemon owns the spare processes on its node. The policy's engine entry places
spares statically, as it places members: `spares` lists node, device and the rank each stands in
for, and `spare_launch` says how to start one, a command with `{engine}`, `{rank}`, `{device}`,
`{node}` and `{socket}` filled in and the environment it needs. When a daemon links, the
coordinator tells it which spares the policy places on its node (`LaunchSpare`), and the daemon
launches, watches and restarts them, with a backoff, and stops them when it stops. A launched
spare registers with the daemon as any spare does, and the pool sees it as before. An engine
whose policy names no launch keeps launching its own spares, as the single-node drill does today.

The join is an opaque exchange. When a change gives spares to an engine, the adapter on the
executor's side produces a `Join` payload for each position and sends it through its daemon; the
coordinator routes it to the daemon holding that rank, which hands it to the process in that
position; the process's adapter consumes it and answers with a `Join` of its own, routed back to
the engine's clients. The payload is bytes the coordinator and daemons never read. For vLLM the
bytes are its queue handles; for another engine, whatever its scheduler needs. The executor's
half of the join, a new ring with several new readers and the install of several spares at one
boundary, is the same whether the spares are local or remote, and is built once against this
carrier.

**Consequences.** One control plane and one wire for every engine, with one engine-specific
thing per engine: the two halves of the join. The single-node spare moves under the same
mechanism, so there is one. The daemon becomes a process supervisor, which it was not; it reads
no engine object and runs any command the policy names. The coordinator's spare pool is fed by
registrations, as before, so a spare that fails to start is simply absent from the pool and the
readiness view says so. The cross-host join itself needs two hosts to prove; the routing and the
supervisor are proven on one.
