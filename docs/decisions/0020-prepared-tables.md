# 0020: a rank's tables are created and touched at registration, and a commit copies runs

2026-10-06. Status: current.

**Context.** After decision 0019 the commit-path session on the 72B shape (`bench/runs/PLANNED/
commit-path-session/NOTE.md`) measured the overhead at about 4 % (from 42–48 %), the store's
round trip at about 1 ms and one message per step, but still 4 s of fence waits per rank per run.
Reproduced locally at that shape (80 layers, 2 KV heads, 16,384 prefill slots, 1.34 GB per rank):
the first commit into fresh tables took 424 ms in the store against 22 ms once the pages were
warm; the decode commit took 0.12 ms. The cost is the page faults of a table's whole file at its
first write, paid once by the prefill's commit, which the fence two steps later waits out, and on
a busy host it is seconds. The store also copied rows one at a time, 2.6 million small copies for
that prefill.

**Decision.** At registration the worker sends the store the tables it will commit (`Prepare`:
one id per layer and held head slot, the row shape and dtype). The store creates them within the
budget and touches every page before answering, so the first commit costs what every later one
does; a table committed without preparation is still created on first use. A commit copies
consecutive slots at consecutive rows as one run, so a prefill of whole blocks is a copy per block,
not per row.

**Consequences.** Registration takes the page-fault cost once, before serving, and reports it in
the store's log; the prefill's fence no longer stalls the step thread. The budget is claimed at
registration rather than at the first commit, which is the honest moment: a rank that cannot be
mirrored is refused before it serves.
