# Battery dry run, pass 7 (2026-10-09 00:19–02:03 UTC): deterministic mode, the staged cache, the register check's first read

Three fresh single-GPU L40S VMs on Nebius eu-north1 (`gpu-l40s-a 1gpu-8vcpu-32gb`; node-a
10.0.0.17 rank 0 and the head, node-b 10.0.0.44 rank 1, node-c 10.0.0.41 the spare's node),
Qwen3-1.7B at TP1 x PP2, 8 sessions x 512 prompt, 1792 tokens, blocks 1400, store slots 22400.
Deployed on the user's word at 00:19Z ("ok the money is there lets do it"), driver
`battery-dry4.sh` (scratch), tree main `0a53c55` (every member registering its compile cache,
the battery refusing an inconsistent engine, deterministic mode in every engine the adapter
launches). Released by hand at 02:03Z: 1.78 h x 3 x $1.55/h = $8.27.

## Attempt 1 (00:19–00:43Z): deterministic mode failed the compile

Stock ran (1208 tok/s). Every adapter engine then raised at its first compile:
`InductorError: RuntimeError: In the deterministic mode of Inductor, we will avoid those
benchmarkings that would cause non-deterministic results`, from `speedup_by_combo_kernel` →
`benchmark_fused_nodes` (torch 2.13). vLLM turns combo kernels on with their benchmark whenever
neither is set (`config/compilation.py` 982–990; Inductor's own defaults have both off), and
that benchmark is one deterministic mode bans. Fix `90014bf`: `collective_op.INDUCTOR` sets
`combo_kernels` and `benchmark_combo_kernel` false beside `deterministic`. The driver was
stopped after the first failed trial; the VMs held.

## Attempt 2 (00:45–01:30Z, `90014bf`; `0ce9abc` from 01:15Z): the arms

| arm | route | tok/s |
|---|---|---|
| stock | vLLM's own | 1208 |
| adapter, deterministic mode | packed, depth 2 | 1199, 1190 |
| mirror off, deterministic mode | packed, depth 2 | 1208 |

Deterministic mode's cost is inside the noise of pass 6's 1189–1204 at the same depth. The
engine's first compile under it saved a new AOT artifact (`35fd1f19…`).

| trial | resumed at | coverage | sessions identical | first differing position vs the control | detection | kill→resume | first step after the resume | gap |
|---|---|---|---|---|---|---|---|---|
| node_fault 1 | 1526 | 1.0 | 8/8 | none on both ranks | 11 ms | 0.645 s | 0.633 s | 1.33 s |
| node_fault 2 | 1377 | 1.0 | 8/8 | none on both ranks | 11 ms | 0.547 s | 0.067 s | 0.67 s |
| single_death 1 | | | refused: readiness read `compile_cache_consistent: false` | | | | | |
| single_death rerun (`sd_rerun2`, check bypassed) | 1680 | 1.0 | 8/8 | none on both ranks | 7 ms | 0.545 s | 0.059 s | 0.67 s |

Every fault trial that ran was row-exact on both ranks against its control, the spare on the
serving node's cache staged by `nodes.sh cache_sync` before each (both nodes' digests
identical for every key in the trial records), the resumed step a decode on both ranks, every
request at its committed count. Node fault twice and single death once (the rerun): the rows
claim under deterministic mode and the staged cache.

**The register check's first reads.** Node fault trials 1 and 2 ran before `0ce9abc` reached the
nodes: readiness showed no pair for any member (vLLM's AOT path sets only
`compilation_config.local_cache_dir`, never `cache_dir`, so `compiled_with` returned None) and
`compile_cache_consistent` true by absence. The single-death trial, on `0ce9abc`, read the
serving ranks' pairs, one key (`35fd1f19…`) and two digests (rank 0 `f2adf7d0…`, rank 1
`cba2173a…`), and was refused: on a pipeline engine each node compiles its own stage into its
copy of the artifact, so the whole-artifact digest differs per rank by construction and the rule
"every member reports the first member's pair" can never read true. Decision 0029 became per
position (the peer's `eff8118`: one key per engine, a spare held to the pair of the rank it
stands for, serving ranks' digests free to differ), and the digest became the rank's share of
the artifact (`ea2848d`: `rank_<r>_<dp>/model`, the `inductor_cache` files whose paths carry a
kernel name the graph names, and the `best_config` files beside them; a node staged with every
rank's share reports a rank's digest as that rank's own node does). The rerun on `ea2848d`
read the serving ranks' rank-scoped pairs (rank 0 `415831d5…`, rank 1 `2327de61…`) and the
spare's still empty: on the spare the artifact lives on the compiled backbone module, not on the
model or the configuration it registers from; `compiled_with` reads it off the module tree from
`01578aa`'s successor (in the pass-8 tree). The rerun ran with the refusal bypassed
(`CACHE_CHECK=0`), recorded as such.

**The gap's parts.** Kill to resume: detection 0.011, acknowledgements 0.21–0.23, the join's
path 0.015–0.35 (the executor's carrier path between the commit and the survivors' boundary;
the survivors' own boundary and ring swap 1.2 and 0.7 ms; stamped per leg from `01578aa`),
restore 0.21–0.22 for 1.37–1.51M rows, resume RPC 0.01–0.11. Then the first engine step after
the resume: 0.633, 0.067 and 0.059 s against 8 ms steady, NCCL's lazy transport setup on the
new ring's first operation of each kind; `01578aa` warms every pre-formed group at formation
(an all-reduce, a broadcast, on a pair a send and a receive each way) and the exchange times
each communicator's first transfer, the figure that shows the term gone on pass 8.

## Harness lessons of the pass

The tree the nodes receive had grown to 120 MB with the committed run records (116 MB under
`bench/runs`); `nodes.sh update` took seven minutes a node and ships the tree without them now.
A fresh `RUN_DIR` had no `setup/` directory, so the fabric stage wrote no file and the battery
refused every arm; the battery makes it. Rewriting a running script through a new inode drops
its execute bit, and `battery.sh` invokes `nodes.sh` directly: three relaunches failed on
"update failed" with no output until `chmod +x`. The staging's own log line never prints (its
summary python under `nodes.sh`'s `set -e`); the digest files are the record.
