# Battery dry run, pass 8 (2026-10-09 02:06–02:34 UTC): the register check's first real read refused the spare

Three fresh single-GPU L40S VMs on Nebius eu-north1 (`gpu-l40s-a 1gpu-8vcpu-32gb`; node-a
10.0.0.51 rank 0 and the head, node-b 10.0.0.80 rank 1, node-c 10.0.0.33 the spare's node),
Qwen3-1.7B at TP1 x PP2, 8 sessions x 512 prompt, 1792 tokens, blocks 1400, store slots 22400.
Deployed on the user's word at 02:06Z ("lets ride on 8"), driver `battery-dry5.sh` (scratch),
tree main `d37ad7f` (the per-position compile cache rule, the rank-scoped digest, the pair read
off the module tree, every pre-formed group warmed at formation, the join's legs stamped).
One arm: node fault, one trial. The driver released the VMs at its end (02:30–02:34Z):
0.42 h x 3 x $1.55/h = $1.94.

## What ran

The control (02:16–02:19Z) ran to its end. The trial's engine launched at 02:19:50Z on the
cache staged from node-b onto node-c (`cache_sync`: both nodes `f964ea4030:18b63f14a43e`, 379
files). Rank 1 registered at 02:20:29Z with the pair `35fd1f19…/c6c0c508…` (the AOT artifact's
hash as the key, the rank's share as the digest), rank 0 at 02:20:41Z per node-a's daemon. The
spare for rank 1 registered at 02:20:41Z with `35fd1f19…/6b066b25…` and the coordinator did not
admit it: "it runs on compile cache 35fd1f19…/6b066b25…, rank 1's is 35fd1f19…/c6c0c508…".
Readiness at 02:27:22Z, after the battery's wait for a spare had timed out, read
`compile_key 35fd1f19…`, `compile_cache_consistent true`, one member (rank 1) and no spare; the
battery killed node-b anyway at 02:27:28Z, the generation long finished (1792 steps by
02:21Z), and the coordinator wrote "policy chose REBUILD for rank 1; the executor must rebuild".
Verdict: `loss arm INCOMPLETE`, changes 0, 8/8 sessions identical to the control at every
token (the control and the trial were two launches of the serving engine, the kill after both
had finished).

## The four findings

1. **The engine's key outlived its processes** (the controller's; the peer's fix in their gate).
   In the control, the coordinator warned at 02:18:36Z that rank 0 "runs on compile cache
   4154aea857/…, the engine's key is f964ea4030" (two ranks of a pipeline engine compile under
   two key directories; the artifact they save is one, `35fd1f19…`). In the trial, readiness
   listed rank 1 alone among the members though both ranks registered. The rule now: the key
   belongs to the engine's processes, forgotten when the last serving rank or spare departs,
   set by the next process to register; a spare that outlives the engine keeps it alive, since
   a new world registering with another key beside an old spare is a real inconsistency.
2. **The key depended on the compile path** (mine, `_artifact_of` in `worker.compiled_with`).
   A rank that compiled fresh had `compilation_config.cache_dir` set and reported its key
   directory's name (`f964ea4030`, `4154aea857`: one per rank on a pipeline engine); a rank
   that loaded the artifact directly had only `local_cache_dir` and reported the artifact's
   hash. The pair's key is the artifact's hash on every path; the key directory names the
   pair only when no artifact exists.
3. **The spare's digest differed by a built file** (mine, `compile_cache.rank_digest`). The
   serving rank's digest `c6c0c508…` against the spare's `6b066b25…` on the same staged
   artifact: one `cuda_utils…so` under `inductor_cache` that rank 1's launch had built after
   the staging, a Triton runtime helper, nothing numeric. The digest covers the kernel sources
   (`.py`) the graph names and the `best_config` autotune results beside them, and nothing a
   launch builds from those.
4. **The battery's wait for a spare fell through to the kill** (mine, `battery.sh ready_spare`).
   A trial that cannot replace anybody is not a trial: no kill unless readiness lists every
   wanted spare in the pool and every rank among the members ("the spare pool or the membership
   is short").

The check did what it is for: the per-position rule refused a spare whose digest differed by
one file and reported ranks holding different keys, both true statements about pairs that were
measured wrong.

## Measured beside it

The warm at formation holds: the spare's pre-formed pair group warmed in 0.009 s (0.012 s in
the control's spare); the exchange's first transfer on the serving ring read 0.046 s on rank 0
and 0.219 s on rank 1 (NCCL's lazy transport setup on the serving ring's first operation, paid
once at the engine's start). The spare's shadow formation took 33.8 s in the trial and 42.8 s
in the control (`shadow_s`; the summary's `shadow_formation_s` 5.5 is the daemon's figure), a
term for the after-loop list, not for this pass. The first step after a resume and the join's
legs were not read: no change happened.

## Next

Pass 8's rerun on both landings (the peer's key lifetime, the three above): a fresh deploy on
the user's word, readiness before the kill with every rank's pair, the spare admitted and
`compile_cache_consistent` true, one node-fault trial on packed depth 2 under deterministic
mode and the staged cache, the rows read against the control, the first step after the resume
beside the warm's figure, the join's legs stamped.
