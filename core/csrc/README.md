# csrc

The C++/CUDA torch extension `everserve_kernels`. `everserve.kernels` builds it from these
sources against the torch and CUDA toolkit installed beside the engine (the engine's wheel pins
its own torch build, and an extension must match it exactly). torch caches the build by source
hash and toolchain, so an install runs `python -m everserve.kernels` once and every worker process
only loads the result. The wheel carries these sources as `everserve/_csrc`.

Building needs a CUDA toolkit whose major version matches torch's (`CUDA_HOME`), and `ninja` on
the `PATH` (a dependency of the core package, in its virtual environment's `bin`). Without them,
`everserve.kernels.available()` is False and callers take their torch paths.

| file | what |
|---|---|
| `kernels.h` | the C++ interface the bindings expose |
| `kernels.cpp` | the Python bindings |
| `gather.cu` | the mirror's step gather: one launch moves every layer's held rows into the staging file, mapped into the device address space (decisions 0010, 0011) |
| `slot.cu` | the slot collective, both variants of decision 0006 in one kernel: peer-to-peer pushes each member's LL words into every member's device buffer over CUDA IPC and polls locally; host memory has each member write its row of a shared region and read every row. One thread per block reads the control words, which live in host memory. `everserve.comm.slotkernel` drives it; `bench/harness/slot_bench.py` times it against NCCL |
