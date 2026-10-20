# Battery dry run, pass 18 (2026-10-10 00:44–01:22 UTC): the fixes from pass 17 on a GPU

Pass 17's shape and checkpoint (RedHatAI/Qwen3-1.7B-FP8-dynamic, the rental checkpoint's FP8
scheme; three single-GPU L40S VMs on Nebius eu-north1, node-c the shared spare, the whole model
pinned) at `1503a1f`, the checkpoint pinned at `e776dcc`, on Cort's word. The question: do pass
17's fixes hold on a GPU. A spare stops clean on its daemon's SIGTERM, standing or after a take
(`5b63905`); the install compiles the environment's bytecode, so a first take writes none
(`364b35a`); setup takes the checkpoint at its pinned revision and checks what it resolved.

## Result

| arm | sessions identical | restored rows mismatched | backup after the take | processes killed at their stop | gap |
|---|---|---|---|---|---|
| repeat (two runs, no fault) | 8/8 agree | n/a | n/a | 0 | n/a |
| node fault (node-c's first take) | 8/8 | 0 | 1.0 / 1.0 | 0 | 6.88 s |
| single death (node-c's second take) | 8/8 | 0 | 1.0 / 1.0 | 0 | 6.13 s |

Every stop in the pass was clean or had nothing left to stop: the spare standing on node-c, stopped
through its daemon, and the spare that had taken rank 1, on both fault arms (pass 17: killed after
20 s on both). No store refusal; the repeat agrees token for token.

## The cold take now reads as a warm one

| term | pass 17 first take | pass 18 first take | pass 18 second take |
|---|---|---|---|
| the take | 9.58 | 5.74 | 5.75 |
| the modules | 4.37 | 1.77 | 1.78 |
| the KV configuration | 1.66 | 1.41 | 1.44 |
| the warm-up and graph capture | 3.00 | 2.03 | 2.00 |
| written by the take's start | 37.9 MB (1,578 bytecode files) | 4.4 MB | 4.2 MB |

The 4.4 MB is the node's observability agent, in every read. The first take on a node that never
served writes nothing of its own and costs what the second does: the cold case is the warm case on
this shape. The bytecode was most of the module term and part of the warm-up.

Setup: every node downloaded `e776dcc126bb25d3d5af2b619acaad8a94332a86`, the revision read back
from what hf wrote matched the pin, and the record names it.

## Found on this pass

node-c's first spare launch of the node-fault trial read node-b's discovery record while the
harness was still copying it (an empty file), refused it as "not a daemon record" and exited; its
daemon launched it again a second later, and the trial ran on that one. The harness copied each
record with a plain `cat >`; it now writes beside the record and renames it into place, as the
daemon writes its own.

Pass 16's 13 s (the "fp8" method through Marlin on Ada) stays open; it is not on this path.

## Cost

VMs 00:44:50 to 01:22:01Z, 106.4 VM-minutes at $1.55/h: $2.75. The user's $20: passes 13 to 18,
$21.66, $1.66 past it on the user's "lets run it for sure", with a reload to come before the H100s.
