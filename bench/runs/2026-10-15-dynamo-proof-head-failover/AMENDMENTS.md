# proof-24: amendments to the run's conditions

`PREREGISTRATION.md` is unchanged (sha256
`ba6982918a78a784e2585510a6af4135233bb8f774017f840f52c041d59bf2c8`). Each amendment below was
committed before the attempt it governs. None moves a bar or a prediction.

## 1. Attempt 1 aborted at bring-up; the images move to the commit that fixes it (2026-10-15)

- Attempt 1, at images `739a62a`, began at about 16:31Z. The rig session stopped it at about
  16:48Z, before any loss. Engine A never served, and no prediction was read. Engine B served.
  The control plane told A's core HEAD at epoch 0, and the pods were placed as rendered. Its
  logs are kept in `run-1/`.
- Cause: both carrier starts (decision 0038, E1) skipped vLLM's `MultiprocExecutor._init_executor`,
  and with it `_get_parallel_sizes()`, the only place `world_size` and `local_world_size` are set.
  The head's core failed on `world_size` and restarted in a loop. The GPU nodes failed on
  `local_world_size` and went into CrashLoopBackOff, and a finalizer then read `_stop` before it
  was set. The tests had missed it because their fakes preset the sizes.
- Fix: both starts call `_get_parallel_sizes()` first, and `NodeExecutor` sets `_stop` before
  `super().__init__`. A fake now has sizes only through that call (mutation-checked), and a new
  vLLM-gated test runs `_init_carrier` from a config-only executor. On main at `4145637`.
- So the clause "no engine or controller change between" is replaced for attempt 2 by: images
  built from main at `4145637` or later, with no engine or controller change after `4145637`. The
  record names the image commit.
- Attempt 1 cost about 18 minutes of the 6 H100s. Attempt 2 runs under the same 1.5 h watchdog.
