# Result: inconclusive by its own rule, everything else consistent with the child running like vLLM's (NCCL probe 5)

**Inconclusive: vLLM's own eager spread at 7,168 was 14.0 %, over the pre-registered 10 %. So bar
2 decides nothing, and (C) is neither built nor out.** On 2026-10-15 the probe ran on 3 ×
`gpu-h100-sxm` in eu-north1, Cort's choice after four attempts found no L40S capacity
(`attempt-1` to `attempt-4`). It used images `dev-51e2763` and the probe at `ae4795a` (the H100
amendment) and `c519bfd` (`--platform`). The answers are in `run-h100/`, and `medians.txt` holds
the table below for both ranks.

## The figures

Rank 0's medians across 10 rounds, in µs per all-reduce. Rank 1's are within 0.2 µs of them.

| Size, mode | V | V's IQR | P (polled) | C (split child) | W (guarded V) |
|---|---|---|---|---|---|
| 896, eager | 149.58 | 11.46 (7.7 %) | 152.50 (+2.0 %) | 145.43 (−2.8 %) | 149.40 (−0.1 %) |
| 896, replay | 147.77 | 4.00 (2.7 %) | 149.92 (+1.5 %) | 149.06 (+0.9 %) | n/a |
| 7,168, eager | 214.35 | 29.93 (14.0 %) | 227.12 (+6.0 %) | 218.94 (+2.1 %) | 212.87 (−0.7 %) |
| 7,168, replay | 218.86 | 13.52 (6.2 %) | 218.92 (0.0 %) | 217.96 (−0.4 %) | n/a |

Every call on every arm returned 0 (`all_success`). The split completed in 0.008 s, and both aborts
returned in 0.51 s after the graphs were freed.

The GPU was an NVIDIA H100 80GB HBM3 on driver 580.173.02. Its SM clock read 1,980 MHz, the
maximum, before the rounds and after them, with clocks not locked.

## Against the bars

| Bar | Read | Verdict |
|---|---|---|
| Inconclusive row: V's eager IQR over 10 % at either size | 7.7 % at 896, **14.0 % at 7,168** | **applies**: bar 2 decides nothing |
| 1: C's replay within 5 % of V's (recorded only, after the amendment) | +0.9 %, −0.4 % | holds |
| 2: C's eager within max(5 %, V's IQR) of V's | 896: −2.8 % against a 7.7 % bound; 7,168: undecided | none, by the inconclusive row |
| 3: probe 3's (A) replay figure | decides nothing on H100 | n/a |
| W, the guard's cost | −0.1 % and −0.7 % | in the noise: the guard costs nothing measurable |

## What it means, and what comes next

The figures that can be read all point one way:
- the split's child runs like vLLM's own communicator, to within a few percent on every reading;
- the guard costs nothing measurable;
- the clocks sat at their maximum throughout, so locking them would not have helped.

The noise is the eager method's own: at 7,168, 5 batches of 200 calls each leave V's rounds 14 %
apart. So the method is fixed, as the table says, by more calls per batch and more rounds, and the
run repeated.

## The second H100 run, with the method tightened: conclusive, and (C) fails bar 2

The run used 1,000 calls a batch, 7 batches and 12 rounds (`18a2d6e`), with a 600 s bound per
process (`931750a`), on H100 as before. The answers are in `run-h100-2/`. Rank 0's medians are
below; rank 1's are within 0.1 µs of them.

| Size, mode | V | V's IQR | P (polled) | C (split child) | W (guarded V) |
|---|---|---|---|---|---|
| 896, eager | 147.72 | 5.36 (3.6 %) | 164.73 (+11.5 %) | 164.67 (+11.5 %) | 148.48 (+0.5 %) |
| 896, replay | 150.95 | 3.83 (2.5 %) | 167.09 (+10.7 %) | 162.84 (+7.9 %) | n/a |
| 7,168, eager | 217.76 | 9.97 (4.6 %) | 244.23 (+12.2 %) | 238.21 (+9.4 %) | 224.06 (+2.9 %) |
| 7,168, replay | 220.05 | 5.63 (2.6 %) | 243.89 (+10.8 %) | 238.68 (+8.5 %) | n/a |

How the bars read:
- **The inconclusive row does not apply:** V's eager IQR was 3.6 % and 4.6 %, both under 10 %.
- **Bar 2 fails:** C's eager is +11.5 % and +9.4 %, both outside max(5 %, V's IQR). So **(C) is
  out**, as the table says. (A) is no better: P is +11.5 % and +12.2 %.
- **W, the guard,** is +0.5 % and +2.9 %. It goes in whichever join is built.

**A contradiction, recorded and not acted on.** In the first run, at 896, where V's spread was
readable (7.7 %), C was −2.8 %. In this run, P and C, the two communicators born nonblocking, are
both 8 to 12 % slower than V, and evenly so, while W is not. The rule reads this run. The
disagreement stands in the record.

## The decision (Cort, 2026-10-15)

NCCL 2.29.7's blocking initialisation does not bound the case that matters. Read at `v2.29.7-1`,
`src/bootstrap.cc` and `src/misc/socket.cc`:
- the root's accept loop (`bootstrapRoot`) waits for every rank, with no timeout;
- `ncclSocketAccept` ends only on a connection or an abort flag, and a blocking initialisation
  gives no handle to set the flag.

So a member that dies after the go and before it connects to the root hangs the others'
formation. Cort chose the blocking join, with that window bounded:
- the members gate on the control plane's links before joining (`GroupReady`; the group's id is
  relayed only once every member is ready);
- the coordinator's formation deadline fails the group and retries it fresh (`6054ff1`);
- a formation thread past the deadline is reported as a leak, and the coordinator retires its
  process by one planned departure with no gap (0025, as amended; `87d01dd`).

The alternative was (C) at about 10 % per collective.
