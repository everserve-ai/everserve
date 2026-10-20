# 0012: the slot collective sums in rank order, and reads its membership from device memory

2026-10-06. Status: amended by 0022 (2026-10-07); current as amended.

**Context.** The kernels re-run (`bench/runs/2026-10-05-kernels-4gpu/`) failed two predictions
for two causes in the slot collective. First, one session in eight produced different tokens from
the no-fault run from 13 tokens after a REPLACE, with every restored row exact: the kernel sums
members in region-row order, the spare's row is not the row of the rank it replaced, and
floating-point addition is not associative, so the sum after the swap rounds differently and a
token flips. Through NCCL, which sums in rank order, all eight sessions matched in every earlier
run. Second, the adapter runs 7.8 % behind stock vLLM with its custom all-reduce on; the mirror is
0.5 % of that and the slot kernel the rest. Every call reads its member mask from host-mapped
memory, one PCIe round trip per block, which the build session had already found to be where the
first version lost its time.

**Decision.**

1. *Order is a property of the model, not of the region.* The control row carries an ordered list
   of member rows in tensor-parallel rank position; the executor writes it at every change with the
   spare in the position of the rank it replaces; the kernel accumulates in that order. The sum is
   therefore the same function of the ranks' inputs before and after any membership change. Pinned
   by bit equality against the reference sum (`everserve.reference.arith`) across a change with the spare in
   the replaced position, on both variants.
2. *Slow-changing state lives on the device; the one asynchronous signal stays on the host.* Each
   worker keeps its member order in its own device memory and copies it there from the region at a
   step boundary, when it is quiescent (`SlotKernel.set_members`). The kernel reads device memory
   on every call. The abort word, which the executor must raise at any instant without the worker's
   cooperation, stays in the mapped region and is polled rarely, as it is now.
3. *The kernel's target is stock vLLM's custom all-reduce*, the one-shot NVLink reduction, not
   NCCL. `bench/harness/slot_bench.py` reports both. If the flag-in-word algorithm cannot reach it
   after the control-word move, the algorithm changes inside the same kernel contract; the gap is
   not accepted.

*Amended 2026-10-07 (the node role, decision 0022): each member writes its own row's order into
the region at the boundary, from the change its group applies, and copies it to its device
memory; the executor writes only the initial order, when it creates the region.*

**Consequences.** The region's control row changes shape (an ordered list replaces the mask; the
abort and published words stay); the region's magic number changes with it so an old file is
refused. The host-memory variant gets the same order and the same device-resident control. The
harness gains a reference run of notice length so a notice's token equality is checked like a
REPLACE's. The claim the product makes, that output after a fault is identical to output without
one, rests on item 1 and is tested as such from this decision on.

**Outcome 2026-10-06** (`bench/runs/2026-10-05-kernels-order-4gpu/`). Items 1 and 2 are built and
item 3's target is met: with the order in device memory and one word per thread the slot kernel
takes 4.2 µs at decode size against 5.8 µs for stock's custom all-reduce, and the adapter runs 1.3 %
ahead of stock with that all-reduce on. But the context paragraph's diagnosis of the token
mismatch was wrong. Every member applied the correct order, and the three notice swaps, same spare
in the same position, matched 8 of 8 sessions; the REPLACE runs still diverged in one session from
the same token in all six. The summation order was worth fixing, since without it the sum did
depend on the region's layout, but it was not the cause. The cause is in the fault path and not in
the restore (rows exact, counts matched): the aborted step or the replay of the tail beyond the
committed count. Decision 0013 records what it turns out to be.
