# S3 result: concurrent changes of different engines partly wait on each other

Pre-registration: `PREREGISTRATION.md`. Numbers: `results.json`. Same host as S1, at 82a9a3d.

| | one at a time, p50 | burst of 100: wall | burst p50 / max | wall over one-at-a-time p50 |
|---|---|---|---|---|
| 1,250 nodes | 0.59 ms | 13.7 ms | 12.9 / 13.5 ms | 23x |
| 2,500 nodes | 0.59 ms | 14.9 ms | 14.0 / 14.7 ms | 25x |
| 5,000 nodes | 0.49 ms | 9.4 ms | 8.8 / 9.3 ms | 19x |

## Against the predictions

1. **Held:** every change of every burst committed.
2. **Failed:** a burst took 19–25x one change's time, against a bound of 10x. Fully serialized
   would be about 100x and fully independent about 1x, so the changes partly wait on each other.
3. **Held:** the burst's wall time did not grow with the cell (13.7, 14.9, 9.4 ms).

## What it means

The changes of different engines are not independent, though they are not serialized either.
This run does not separate the one engines lock from the CPU the burst itself costs. Each change
sends to and hears from two fake daemons in the same process, on a 14-core host. The absolute cost
is small: 100 simultaneous failures in a 40,000-GPU cell all commit within 15 ms, against about
360 ms for one real change on GPUs (proof-19). The bound set before the run failed all the same.

## What changes the plan

As pre-registered, S3 comes next. First, a profile of the burst to find where the changes wait;
then a lock per engine, if that is where; then this run again, with its bound.

## Addendum: where the burst waits (not pre-registered)

Taken after the result, to find what prediction 2's failure is made of. The same 1,250-node cell,
once with the in-memory store, once with every store write taking 10 ms first, as a real store's
round trips do (`cell_1250_store_10ms`).

| 1,250 nodes | in-memory store | each store write 10 ms |
|---|---|---|
| one change at a time, p50 | 3.5 ms | 26.3 ms |
| burst of 100: wall | 10.3 ms | 37.5 ms |
| burst p50 | 9.7 ms | 36.8 ms |

If a change held a lock other engines need across its store writes, a burst of 100 at 10 ms a
write would take on the order of 100 one-at-a-time changes, about 2.6 s. It took 37.5 ms: one
change's writes, plus the same 10 ms the in-memory burst spends. So the store writes of different
engines run concurrently. What the engines share is about 0.1 ms per change of in-memory work
under the one engines lock.

**The plan, against what was pre-registered.** The pre-registration said a failed prediction 2
makes a lock per engine the next build. This addendum shows that lock would save at most about
0.1 ms per simultaneous change, about 100 ms in a burst of 1,000, against changes that take
hundreds of milliseconds on GPUs. So S3 is recorded as measured and deferred, not built. It is
built when a burst's shared time matters next to a real change's. The harness's burst phase
stays, and it shows when that happens.
