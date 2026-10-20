# 0019: a step's commit is one descriptor, and the fence runs two steps behind

2026-10-06. Status: current.

**Context.** The single-node frontier run (`bench/runs/2026-10-06-frontier-1node/`, Qwen2.5 72B
at TP4 on 5x H100) kept every session exact through three REPLACEs, two notices and a 16K-context
fault, and paused 1.5–1.8 s, but the adapter ran 42–47 % behind stock. The decode step itself was
stock's (16.6 ms median on both); the step thread waited on the mirror's one-step fence for 8 s
of a 20 s run. The fence at the commit of step N waits for the acknowledgement of N-1, which
arrives once the GPU has finished forward N-1 (the step thread runs about a step ahead of the GPU
under async scheduling, so that is roughly when it calls the fence), plus the gather, plus the
store's round trip. On the 30B MoE at TP4 that tail was under a millisecond and the fence waited
nothing. On this shape the round trip was 2.5–3.4 ms: a step is described to the store as one
`StagedRows` per (layer, head slot) table, 160 messages at 80 layers and 2 KV heads per rank
against 48 before, each parsed, validated and copied one by one. Every step paid that tail, and
the tail scales with the shape.

**Decision.**

1. *One descriptor per step.* The commit names the staging file once with its layout: the slot
   ids, the layer indices saved, the held head slots, and the byte strides of the ``[layers,
   heads, capacity, 2, d]`` buffer. The store derives every table from it and copies them in one
   pass. The message size and the Python work per step no longer grow with the layer or head
   count.
2. *Two steps in flight.* The fence at the commit of step N waits for N-2 (`depth` 3 in the
   transport's terms). The acknowledgement of N-2 has long arrived when N commits, so the step
   thread pays nothing in steady state. A step the survivors accept has every rank's rows durable
   through N-2; the replay after a fault is at most three positions, each a one-token decode step
   on the captured graph (decision 0013), so the price of the second step of slack is one more
   replayed step on a fault. `docs/CONTRACT.md` item 3 states the fence as two behind.

**Consequences.** The shape stops setting the overhead: a step's commit is one message and one
copy pass whatever the layer count, and the fence is off the step thread's path. The transport
reports the same spans, so the re-run judges each term: `sender` under 1 ms, `fence_waits_s` near
zero, overhead within a point of mirror-off. The 16K-context restore (3.1 s for about 8 GB per
rank) is the pause's largest term at long context and is the incremental restore's and the
device-side gather's work, not this decision's. Logprobs are not bit-reproducible between runs on
H100 even where tokens are, so the logits check of decision 0013 is judged on tokens on that
hardware until the engine's own run-to-run noise is understood; the exactness claim stands on
tokens and on the committed rows, which were exact in every run.
