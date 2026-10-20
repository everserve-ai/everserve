# Glossary

The words the code, the records and the operator API use, each with one meaning.

| word | meaning |
|---|---|
| **engine** | one model served by one vLLM engine core over a set of GPUs; named in the policy; the unit a change applies to |
| **head** | the engine's core process: the scheduler, the request queue, the executor. It runs on rank 0's node and has no spare (decision 0045) |
| **rank** | a position in the engine's parallel layout (tensor, pipeline, data, expert), numbered from 0; the engine's view of a GPU's work |
| **position** | a rank as a place to be held: a process holds a position, loses it, or takes it |
| **member** | a process holding a position at the current epoch: (node, device) in the membership |
| **membership** | every rank's holder at an epoch; committed by the coordinator and never changed except by a change |
| **epoch** | the membership's version, one per committed change; every change's fence |
| **change** | one membership transition: the ranks lost, the outcome, the takers, the departing ranks; claimed, delivered, acknowledged, committed or abandoned |
| **outcome** | what a change does: REPLACE (a spare takes the lost rank), SHRINK (the survivors serve at N-1, the reference tier), REBUILD (the floor); decision 0004 |
| **spare** | a process built ahead of a loss and standing for a position: *pre-loaded* for one rank, *any-position* for whichever rank its node is given (decision 0030), *resident* holding every shard in host memory (0028); a spare *takes* a position at a change |
| **take** | a spare standing for any position being given one rank at a change |
| **join** | the exchange by which a taker enters the engine's own queues and ring, carried by the control plane unread (decision 0024) |
| **start-up replay** | the engine core's start-up broadcasts to a rank, recorded and replayed to the spare that builds for it (decision 0044) |
| **notice** | a rank or a node will leave at a deadline: a cloud's preemption signal, an operator's drain, a process leaving; the change runs before the loss |
| **drain** | an operator's or a notice source's instruction to move every position off a node by a deadline; kept in the spec store |
| **cordon** | a node that takes no new position |
| **cover** | the groups a process forms ahead of a loss so the change at that loss forms nothing; a cover *plan* names them, a *set* is one membership the plan answers, and an engine is *covered* when every loss it may suffer has a formed set (decision 0035) |
| **group** | a communicator over some members (a NCCL group); the *live* group serves, *shadow* and *cover* groups stand ready |
| **collective** | the object a model's traffic goes through; it holds the live group and swaps it at a step boundary |
| **slot collective** | the all-reduce through a shared region of slots, one row per member, that sums in rank order and detects a dead member in the kernel (decisions 0006, 0012, 0021) |
| **slot** | a row of KV cache: one block position of one layer and head; also a row of the slot collective's region |
| **committed** | a request's tokens and KV rows the store holds, through the last fenced step; a restore resumes from there (decision 0013) |
| **mirror**, **store** | the process on every node holding committed rows off the GPU in shared memory, replicating them to the nodes that may restore them |
| **fence** | the step the store's rows are complete through; a restore reads to it |
| **replay** | re-running steps from the committed position after a change, one step at a time in the same shape (decision 0013) |
| **daemon** | the coordinator's agent on a node (`everserve-controller node`): holds every process's link, delivers changes, reports |
| **coordinator** | the one process that decides and commits changes for a cluster, under a lease (decisions 0014, 0036) |
| **lease**, **term**, **hold** | the coordinator's exclusive right to act, numbered by term; a coordinator acts only while it holds it, and every message it sends carries the term |
| **spec store** | where the coordinator keeps desired state and each engine's change in flight, fenced by term (decision 0036): a file, or the cluster's `EverserveState` objects |
| **policy** | the deployment's document: engines, outcomes, spares, limits; data the coordinator reads, not code |
| **identity** | what a process runs, reported when it registers: engine, commits, torch, CUDA, NCCL, checkpoint, dtypes, layout, step config (decision 0034); a destination whose identity differs is refused |
| **step config** | the engine's step-shaping settings every process is held to |
| **compile cache** | the compiled graphs a position runs with, part of the position (decisions 0029, 0043) |
| **undecided** | a process that has asked its node what its GPUs are for and has not yet been told (decision 0041) |
| **instance** | Dynamo's unit: one engine as Dynamo deploys and routes to it (decision 0041) |
| **node group** | the one trust domain on a node: the daemon, the store and every engine process, members of one Unix group sharing the pod contract's directories |
| **pod contract** | the host directories and sockets an engine's pod and the node's daemon share |
| **pre-registration** | a run's predictions and pass condition, committed before it runs; its record is the only source of a number in prose |
