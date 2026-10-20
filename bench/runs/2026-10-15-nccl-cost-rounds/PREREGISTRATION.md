# Pre-registration: the split's child against vLLM's communicator, timed in interleaved rounds (NCCL probe 5)

Written before the run. The run is `bench/harness/k8s/dynamo_proof.sh --losses 0 --nccl-probe
--nccl-probe-arms rounds` on images `dev-51e2763`: 3 × L40S in eu-north1 or the capacity
fallback, a 1.5 h watchdog, and a cap of about $4.

## Why

Probe 4 ([RESULT](../2026-10-15-nccl-split-probe-2/RESULT.md)) showed that a split on a
nonblocking parent can report success without its peer. The peer's answer is (C) plus a
confirmation over the links: a group counts as formed only once every member's report of its child
has arrived before the deadline, and otherwise every member aborts its child and its parent. That
makes the formation bounded end to end. One question is left: does the child serve like a fresh
blocking communicator?

Probes 3 and 4 timed the arms one after another, minutes apart. Two blocking communicators on the
same pods came out 23 % apart per eager call. That is above any 5 % bar. This run times them
interleaved in rounds, so that the same moment's conditions apply to every arm.

## The method

One process pair, the head's pod and a node pod on two nodes, holds three communicators at once:
- **V:** vLLM's blocking communicator (`PyNcclCommunicator.from_unique_id_bytes`), the baseline.
- **P:** a nonblocking parent, used polled on every call, as (A) would be.
- **C:** P's blocking child, split with the completion rule probe 4 confirmed, as (C) would serve.

The run:
- **Graphs:** each arm's graph of 50 all-reduces is captured once per size, before the rounds,
  after 20 warm calls.
- **Rounds:** 10 of them. Each round runs every arm in an order rotated so that each leads in turn
  (V,P,C, then P,C,V, then C,V,P). Both ranks compute the same order. A test checks, against
  stubs, that both issue the same calls.
- **Per arm, per size (896 and 7,168 bf16 elements), per round:**
  - eager: one discarded batch, then 5 batches of 200 calls, each synchronised at its end;
  - replay: one discarded batch, then 5 batches of 20 replays of the graph.
  - The round's value is the median of its 5 batches, in µs per call.
- **Reported:** each arm's 10 per-round values, their median, interquartile range, minimum and
  maximum.
- **Recorded beside the figures:** the GPU and driver, from `nvidia-smi`, before and after; NCCL's
  version; and that clocks were not locked. The probe has no root to lock them.

The engine is idle: no load runs during the probe, as in every probe before.

## The bars

1. **C's replay median is within 5 % of V's**, at both sizes.
2. **C's eager median is within max(5 %, V's eager IQR as a fraction of V's median) of V's**, at
   both sizes.
3. **P's replay at 896 is slower than V's** only if P's median exceeds V's by more than V's replay
   IQR.

## What each outcome decides

| Outcome | Decision |
|---|---|
| C passes bars 1 and 2 | **(C) plus confirmation is the bounded join.** The peer builds the confirmation round on the coordinator. I build the formation (a nonblocking parent, a blocking child, both under the deadline) and the aborts. |
| C fails bar 1 at either size | (C) is out. The gated join and (A) are looked at again, with these figures. |
| C passes bar 1 and fails bar 2 | The decision goes to the peer and me, with the figures. Replays are the decode path; eager calls are prefill's and oversized batches'. |
| Bar 3, either way | Decides only what probe 3's record says about (A)'s replay cost. Nothing about the build. |

The decision is the peer's and mine, from this table, not after reading the results.

## Amendment, before any figure was read (after the first run hung in its cleanup)

The first run measured its rounds and then hung in the child's `ncclCommAbort`, which waits for
every CUDA graph captured on the communicator: a capture variable kept the last one alive. No
figure was printed or read. The probe now frees its graphs before the aborts, bounds each abort
at 5 s on a thread as vLLM's `destroy` does, and prints each round's figures as the round ends.

**Which bar decides, corrected.** The table above calls replays "the decode path". For this
engine that is wrong. No serving communicator is ever captured in a graph: the TP all-reduce goes
through the graph-safe op, a PIECEWISE split point, so every NCCL call runs eagerly between graph
pieces (`adapters/vllm/everserve_vllm/collective_op.py`, decision 0029). PP's sends and receives
run between stages, outside graphs too. A child under (C) would serve through the same op. So:

| Outcome | Decision |
|---|---|
| C passes bar 2 (eager) | **(C) plus confirmation is the bounded join**, built as the table above says |
| C fails bar 2 at either size | (C) is out; the gated join and (A) are looked at again with these figures |
| Bar 1 (replay), either way | recorded; decides nothing about the build |
| Bar 3, either way | decides only what probe 3's record says about (A) |

The bars themselves are unchanged.

**Inconclusive, added before booking.** A bar set by V's own spread is loose exactly when the
method is noisy, and probe 4 saw 23 % between two blocking communicators. So, read first: if V's
eager IQR across rounds is over 10 % of V's eager median at either size, bar 2 decides nothing.
(C) is then neither built nor out, and the method is fixed first (more calls per batch, or locked
clocks on a rental with root) before another run.

**A fourth arm, W, added before booking: the capture guard's cost.** The (C) design puts a guard
on vLLM's communicator handles: a wrapper that delegates every attribute and refuses a collective
while a stream is capturing. W is V called through that per-call work (a delegated attribute and
`torch.cuda.is_current_stream_capturing()`), interleaved with the other arms, eager only, since
inside a capture the guard refuses. **Reported:** W's eager median against V's at both sizes, and
the difference in µs. **It decides nothing about (C)**: it is the guard's measured cost, which the
(C) design cites.

## Amendment, before booking on H100 (Cort's word, 2026-10-15)

Four attempts on `gpu-l40s-a` found no capacity (`attempt-1` to `attempt-4`, 117 minutes of
control plane, no GPU). Cort chose H100 in eu-north1: `gpu-h100-sxm`, `1gpu-16vcpu-200gb`, through
`dynamo_proof.sh --platform/--preset` (`c519bfd`). That option is checked by `bash -n` and the
harness's shell tests only; its argument handling cannot run without a rental.

What this changes:
- **Bars 1 and 2** compare arms within the same run, interleaved, so they stand unchanged on H100.
- **Bar 3** (whether probe 3's +28 % replay for (A) at 896 was real) decides nothing on H100. It
  compared with L40S figures.
- **Absolute figures** are not comparable with probes 2 to 4, because H100's NCCL paths differ
  from L40S's.
- **The inconclusive row** (V's eager IQR over 10 % of its median) stands.

**Cap.** About $4.45 per GPU-hour all-in, from our one Nebius bill (the billing page is not
readable from the harness). Three single-GPU nodes come to about $13.35 an hour. A run is about 30
minutes, so about $7. The watchdog is set to 1 hour, so the worst case is about $13.

## Amendment, before the second H100 run: the method tightened, and no third run

The first H100 run was inconclusive by the 10 % row: V's eager IQR at 7,168 was 14.0 %
([RESULT](RESULT.md)). As the table says, the method is fixed and the run repeated:
- each eager batch is 1,000 calls (it was 200);
- each arm, size and mode runs 7 batches after its discarded one (it was 5);
- there are 12 rounds (it was 10);
- the per-process bound is 600 s (it was 300).

W stays in: it adds about 40 s to a run of about 4 minutes and gives a second reading of the
guard's cost. The bars, the 10 % row, the decisions and the platform (H100) are unchanged.

**If this run is also inconclusive** (V's eager IQR over 10 % of its median at either size), there
is no third run. The record goes to Cort with both runs' figures, and the decision is his.
