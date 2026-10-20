# 0018: the two-node run is three 4x H100 nodes, a 70B model at TP4xPP2, and a spare node

2026-10-06. Status: reversed by 0030 (2026-10-09) as the rental's shape; kept as the record of the
dry-run shape (`bench/runs/PLANNED/p2-two-node`). A ruling on `docs/MULTINODE.md` M5.

**Decision.** Roadmap step 5 runs on three 4x H100 nodes: a 70B-class dense model at TP4 inside
each of two serving nodes and PP2 across them, the third node as the spare node. A notice on one
serving node; the spare node takes its four positions in one change; sessions identical. Twelve
H100s, about two hours, cap $100. Every cross-node piece is exercised: coordinator over daemons,
replication, a set change of four, fabric collectives outside the graphs, a spare node. Two 8x
H100 nodes would cost three times as much and exercise nothing more; that shape is for the
multi-node frontier result after this one.
