# 0023: a node is lost when its daemon's link ends without a goodbye, or misses its heartbeats

2026-10-07. Status: current.

**Context.** Decision 0021 detects a lost rank on its own node: the slot kernel, the stalled
step and `/proc` all see the node's processes and nothing else. A lost node is seen by nobody on
it. The coordinator saw its daemon's link close and marked the ranks departed, driving no change;
the two-node pre-registration's node-fault arm had no detection path at all, and its band was a
guess. The two-node smoke found that the surviving node's worker sees a remote death after about
two seconds, when the dead rank's sockets close and the pipeline send fails, and that the head's
executor never sees it: it waits on a queue that reports nothing when its writer dies.

**Decision.** Every daemon sends a heartbeat on its coordinator link at an interval it declares
in its hello (`--heartbeat-ms`, 500 by default; 0 sends none). The coordinator takes a node as
lost when the link ends without a `Bye`, which is immediate for a process exit, or when the
daemon misses its heartbeats a policy-set number of times (`--node-loss-misses`, 3 by default:
1.5 s), which covers a crashed host. A daemon stopped on purpose sends `Bye` first, and its
ranks are departed, not lost, until it is back. A lost node's ranks are one change per engine
for the set, driven by the coordinator as a fault. The daemon forwards every `Change` the
coordinator claims to the engine's clients as well as its workers, so an executor learns a
remote rank's loss from the coordinator's word, the one authority that decides it, and aborts
its step as it would for a local loss. The worker's own failed pipeline send stays as a second,
slower signal through the existing failed-collective report.

**Consequences.** A node whose processes are killed is detected within the link's close, tens
of milliseconds; a crashed host within the miss count times the interval. The miss count is a
setting because a false loss costs a whole node's handover: three is a defensible default for a
dedicated link between processes without a garbage collector, and the value for a given fabric
is measured, not chosen. The two-node pre-registration's node-fault band follows from this. The
executor's consumption of the forwarded change is adapter work.
