# Result: the two-loss proof under the coordinator's audiences (S1)

**Verdict: pass.** On 2026-10-14, `bench/harness/k8s/dynamo_proof.sh --losses 2` ran on images
built from `31789ed`. That commit is main with S1 (4ce8921): an engine's policy, placement and
state go only to the nodes that hold or reached a process of it, and a node's first ask or
registration for an engine is answered behind that engine's facts.

The run repeats [proof-18](../2026-10-14-dynamo-proof-pipe-leave/RESULT.md), which ran without
S1, and its record matches proof-18's:
- Both losses of rank 1 reached the coordinator as notices. The second came from the spare that
  took rank 1 in the first.
- Each committed a REPLACE through a covered set.
- All 16 streams finished with every token and no error.
- Grove's replacement pod asked, stood and was covered at epoch 1, 42 s after the first change.
  Its first ask was its node's first contact with the engine under S1.

The run's output is `run/`, laid out as proof-18's. The cluster was asked for at 16:10:01Z,
served at 16:27:29Z, and was taken down at 16:43:21Z.

| | Loss 1 | Loss 2 |
|---|---|---|
| covering set, at the take | plan 2 at epoch 0 | plan 4 at epoch 1 |
| notice heard by the coordinator | 16:27:56.061 | 16:29:22.910 |
| committed | +367.0 ms | +1,141.0 ms |
| acknowledgements | rank 0 +22.8 ms, rank 1 (the spare) +365.7 ms | rank 0 +11.4 ms, rank 1 (the spare) +1,140.5 ms |
| restore onto the spare's node | 420,054 rows | 1,572,480 rows (rank 1's whole KV capacity) |
| streams | 8 of 8, no error | 8 of 8, no error |
| worst gap between tokens | 0.651–0.660 s, at the change | 1.447–1.463 s, at the change |

The second loss's longer restore is proof-18's finding, reproduced: a restore sends every row
the rank has written, which grows toward the KV capacity. The trades that would shorten it are
open for Cort in proof-18's record.
