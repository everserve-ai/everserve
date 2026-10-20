# Battery dry run, pass 15 (2026-10-09 19:01–20:11 UTC): the cold node fixed, 7–10 s

The resident shape on three fresh single-GPU L40S VMs on Nebius eu-north1 (node-a rank 0,
node-b rank 1, Qwen3-1.7B at TP1 x PP2; node-c holding the whole model pinned with one spare
standing for any position, nothing on its GPU). The question: whether a take on a node that
never served reads like a take on a warm one, now that the install builds Everserve's CUDA
kernels, the deploy stages the serving node's Triton cache onto the spare's node, and a spare
loads the kernels when it stands. The fault arms ran on tree `d82de7a` (with the store's row
checksums at protocol 2 and the spare's KV sized from the engine's block count); the stock arm on
the same VMs before them. Within the user's $20.

## Result

| arm | sessions identical | restored rows mismatched | backup after the take | kill→resume | gap |
|---|---|---|---|---|---|
| stock restart | 0/8 | n/a | n/a | n/a | 393.1 s to the first token (312.5 s to the relaunch, the engine ready in 29.8 s) |
| node fault (node-c cold) | 8/8 | 0 (coverage 1.0, 1.45M rows) | 1.0 / 1.0 | 10.12 s | 10.21 s |
| single death | 8/8 | 0 (coverage 1.0, 1.55M rows) | 1.0 / 1.0 | 6.93 s | 7.06 s |

Exact on both arms, every restored row matching, the backup after the take scored 1.0 for both
ranks (every head slot's table, every 16th slot, holder and replica at one step), the join answer
0.67 / 0.72 s, the first steps after the resume 11–21 ms. Clocks: node-b and node-c against
node-a within 0.33 ms, each ± 0.12 ms or better, at the start and the end.

## The take on a cold node

| term (spare's stamps) | pass 14 node fault (cold) | pass 15 node fault (cold) | pass 15 single death |
|---|---|---|---|
| the take | 72.58 | 9.21 | 5.95 |
| identity and slot (the kernels) | 47.95 | 0.00 | 0.00 |
| the worker built as the rank | 0.23 | 0.23 | 0.22 |
| the modules | 4.31 | 4.36 | 1.77 |
| the weights from the host copy | 0.09 | 0.09 | 0.09 |
| the KV configuration | 3.99 | 1.44 | 1.72 |
| the warm-up and graph capture | 15.89 | 2.96 | 2.02 |

The kernels built at install, and the take never builds them; the staged Triton cache took the
cold warm-up from 15.9 s to 3.0 s; vLLM, given the KV size the engine's block count fixes
(`kv_cache_memory_bytes`), logged "skipped memory profiling", the KV configuration 1.4–1.7 s
against 3–4 s (what remains is the forward pass that loads the compiled model). What remains of a
take: the modules' construction (1.8–4.4 s, the first build in a process the slower) and the
warm-up (2.0–3.0 s), the pre-attachment trade's two terms.

## The pass's own detours (about 34 minutes)

1. **The install's kernel build failed on every node.** It ran before the system packages that
   bring the Python headers torch's build includes, without the environment's own `ninja` on the
   PATH, and the setup's wait matched error text that a `RuntimeError` does not contain, so it
   waited 28 minutes. Fixed on the live nodes, then in the harness (`d82de7a`): the build after
   the packages with the environment's bin on the PATH, and a failed setup leaving its exit status.
2. **Protocol 2 Python over protocol 1 binaries.** The battery syncs the working tree when it
   starts; a rebase onto the store's checksums seconds before put protocol-2 Python on nodes
   whose store and controller were built at protocol 1, and the node-fault control failed on
   "protocol 2 refused: this build speaks protocol 1". The fault arms reran on the same VMs with
   both rebuilt at the synced tree. The working tree is not touched while a run deploys from it.

## Cost

Deployed 19:01:38Z, released 20:11:01Z: 1.16 h x 3 x $1.55/h = $5.38. The user's $20: passes 13
to 15, $14.15.
