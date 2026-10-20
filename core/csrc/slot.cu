// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//
// The slot collective: a decode-size bf16 all-reduce a FULL CUDA graph can keep across a
// membership change (decisions 0005, 0006).
//
// Words are NCCL's LL protocol: each 8-byte word carries two bf16 values and the call's sequence
// number in one store, so a reader that sees the sequence number sees the data. A member waits until
// every member's words carry the call's sequence number, then sums the members in fp32 in the order
// the control row lists them: tensor-parallel rank position, with a spare in the position of the
// rank it replaced (decision 0012). Floating-point addition is not associative, so that order is
// what makes the sum after a change the same function of the ranks' inputs as before it.
// The variant decides where words go (decision 0006):
//
//   peer-to-peer (push): each member's buffer, [2 parities][8 senders][maxn/2], is in its own device
//                 memory, mapped by the others over CUDA IPC; a member stores its words into every
//                 member's buffer at its sender slot (posted writes over NVLink) and polls only its
//                 own buffer, so every poll is a local read;
//   host memory (pull): each member's buffer, [2 parities][maxn/2], is a row of a shared host region
//                 mapped into every device; a member writes its own row and reads every member's.
//
// The kernel is the same for both: it reads the members' buffer addresses from a small device
// table and the member order from this process's device memory on every replay, and polls the
// abort word in the member's host-mapped control row. A captured graph therefore follows a change
// of members, of buffers, or an abort, without recapture. The order is copied to the device at a
// step boundary, when the member is quiescent; only the abort, which the executor raises at any
// instant, stays in host memory, where a call reads it once per 1,024 polls (decision 0012).
//
// A member that waits too long names who it waits for (decision 0021). Thread 0 of every block
// reads the global timer when its block starts waiting. At each abort check, once the wait has
// passed the bound the executor wrote into this member's control row (nanoseconds; 0 is off), it
// writes into the row's late word a mask of the rows whose words its block has not received: for
// the batch it waits on, the words it is waiting for; for members of batches it has not reached,
// its first word. Then it keeps waiting. Every block names for itself, because a killed member's
// push can stop part way, with some blocks' words delivered and others not; a block that has its
// words finishes and names nobody. A node's loss is several members at once, and the mask carries
// the set. A device counter holds the blocks that have named; the last one to finish clears the
// word, so a block that finishes early does not erase another's mask. The worker's watchdog reads the word: a late
// member whose process is exiting is dead, and is reported without waiting for its process to be
// torn down. A call that completes before its first abort check never reads the timer's bound.
//
// The sequence number is the low 32 bits of a 64-bit counter and wraps after 2^32 calls, about
// twelve hours at decode rates. A stale word is mistaken for a fresh one only if its position was
// last written exactly 2^32 calls earlier and by no call since, which needs a call of that size to
// have been absent for the whole interval. NCCL's LL protocol carries the same 32-bit flag and the
// same exposure. The counters are zeroed at every membership change (SlotKernel.resume).

#include "kernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace everserve {
namespace {

constexpr int kThreads = 256;
constexpr int kBatch = 4;      // members whose loads one thread keeps in flight together
// kMaxRows, kCtrlWords, kAbort, kLate and kBound are the region's layout, in kernels.h: the
// abort word is [1] of each row's control words; a late mask of 0 names no row; a bound of 0 is
// off

__device__ __forceinline__ long long global_ns() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <int kWords, bool kPush>
__global__ void slot_all_reduce(const unsigned* __restrict__ x, unsigned* __restrict__ out, int n2,
                                const int64_t* __restrict__ bufs, int64_t* ctrl_base,
                                const int* __restrict__ order, int64_t* seqs, int my, int maxn2) {
  volatile long long* ctrl = reinterpret_cast<volatile long long*>(ctrl_base);
  // order: [count, rows...] in device memory; one thread per block stages it in shared memory
  __shared__ unsigned s_seq;
  __shared__ int s_n;
  __shared__ int s_order[kMaxRows];
  volatile long long* c = ctrl + my * kCtrlWords;
  if (threadIdx.x == 0) {
    const long long s = seqs[blockIdx.x] + 1;
    seqs[blockIdx.x] = s;
    s_seq = static_cast<unsigned>(s);
    const int n = order[0];
    s_n = n < 0 ? 0 : (n > kMaxRows ? kMaxRows : n);
    for (int q = 0; q < s_n; q++) s_order[q] = order[1 + q];
  }
  __syncthreads();
  const unsigned seq = s_seq;
  const size_t parity = (seq & 1u) * static_cast<size_t>(maxn2);
  const int nmem = s_n;
  const int b0 = blockIdx.x * kWords * blockDim.x + threadIdx.x;

  // Pull (host memory): each member writes its own row and reads every member's row. Push
  // (peer-to-peer): each member writes its words into every member's buffer, at its own sender slot,
  // over NVLink, and then polls only its own buffer, in local memory.
  if (kPush) {
    const size_t at = parity * kMaxRows + static_cast<size_t>(my) * maxn2;
    for (int q = 0; q < nmem; q++) {
      unsigned long long* dst = reinterpret_cast<unsigned long long*>(bufs[s_order[q]]) + at;
#pragma unroll
      for (int j = 0; j < kWords; j++) {
        const int i = b0 + j * blockDim.x;
        if (i < n2) __stcg(dst + i, (static_cast<unsigned long long>(seq) << 32) | x[i]);
      }
    }
  } else {
    unsigned long long* mine = reinterpret_cast<unsigned long long*>(bufs[my]) + parity;
#pragma unroll
    for (int j = 0; j < kWords; j++) {
      const int i = b0 + j * blockDim.x;
      if (i < n2) __stcg(mine + i, (static_cast<unsigned long long>(seq) << 32) | x[i]);
    }
  }

  float lo[kWords], hi[kWords];
#pragma unroll
  for (int j = 0; j < kWords; j++) {
    lo[j] = 0.f;
    hi[j] = 0.f;
  }
  unsigned valid = 0;
#pragma unroll
  for (int j = 0; j < kWords; j++)
    if (b0 + j * blockDim.x < n2) valid |= 1u << j;

  // thread 0 of each block times its block's wait and names late members (decision 0021)
  const bool watch = threadIdx.x == 0;
  const long long t0 = watch ? global_ns() : 0;
  bool counted = false;
  // after every block's counter at one word per thread, slot_blocks(maxn, 1)
  unsigned long long* late_blocks =
      reinterpret_cast<unsigned long long*>(seqs + (maxn2 + kThreads - 1) / kThreads);

  // members in the listed order, kBatch at a time; within a batch the sums also go in that order
  for (int b = 0; b < nmem; b += kBatch) {
    int ms[kBatch];
    const int nm = nmem - b < kBatch ? nmem - b : kBatch;
    for (int q = 0; q < nm; q++) ms[q] = s_order[b + q];
    unsigned long long w[kBatch][kWords];
    unsigned todo[kBatch];
#pragma unroll
    for (int q = 0; q < kBatch; q++) {
      todo[q] = q < nm ? valid : 0u;
#pragma unroll
      for (int j = 0; j < kWords; j++) w[q][j] = 0ull;  // an aborted read sums zeros, not garbage
    }
    unsigned k = 0;
    while (true) {
      unsigned any = 0;
#pragma unroll
      for (int q = 0; q < kBatch; q++) {
        if (!todo[q]) continue;
        const unsigned long long* p =
            kPush ? reinterpret_cast<const unsigned long long*>(bufs[my]) + parity * kMaxRows +
                        static_cast<size_t>(ms[q]) * maxn2 + b0
                  : reinterpret_cast<const unsigned long long*>(bufs[ms[q]]) + parity + b0;
#pragma unroll
        for (int j = 0; j < kWords; j++)
          if (todo[q] >> j & 1) w[q][j] = __ldcv(p + j * blockDim.x);
      }
#pragma unroll
      for (int q = 0; q < kBatch; q++) {
#pragma unroll
        for (int j = 0; j < kWords; j++)
          if ((todo[q] >> j & 1) && static_cast<unsigned>(w[q][j] >> 32) == seq)
            todo[q] &= ~(1u << j);
        any |= todo[q];
      }
      if (!any) break;
      // released by the abort word; read rarely, since it is host memory and every poll is not
      if ((++k & 1023u) == 0) {
        if (c[kAbort]) break;
        if (watch) {
          const long long bound = c[kBound];
          if (bound > 0 && global_ns() - t0 > bound) {
            // this batch: exactly the members whose words this thread still waits for
            long long late = 0;
#pragma unroll
            for (int q = 0; q < kBatch; q++)
              if (todo[q]) late |= 1ll << ms[q];
            // batches not reached yet: this thread's first word of each member
            for (int q = b + nm; q < nmem; q++) {
              const int m = s_order[q];
              const unsigned long long* p =
                  kPush ? reinterpret_cast<const unsigned long long*>(bufs[my]) +
                              parity * kMaxRows + static_cast<size_t>(m) * maxn2 + b0
                        : reinterpret_cast<const unsigned long long*>(bufs[m]) + parity + b0;
              if (static_cast<unsigned>(__ldcv(p) >> 32) != seq) late |= 1ll << m;
            }
            if (!counted) {
              atomicAdd(late_blocks, 1ull);
              counted = true;
            }
            c[kLate] = late;  // every check: another block's last clear may have raced it
          }
        }
      }
    }
#pragma unroll
    for (int q = 0; q < kBatch; q++) {
      if (q >= nm) continue;
#pragma unroll
      for (int j = 0; j < kWords; j++) {
        const unsigned v = static_cast<unsigned>(w[q][j]);
        lo[j] += __uint_as_float((v & 0xffffu) << 16);
        hi[j] += __uint_as_float(v & 0xffff0000u);
      }
    }
  }
  // the wait is over (arrived, or released by the abort); the last block that named clears
  if (counted && atomicAdd(late_blocks, ~0ull) == 1ull) c[kLate] = 0;
#pragma unroll
  for (int j = 0; j < kWords; j++) {
    const int i = b0 + j * blockDim.x;
    if (i >= n2) continue;
    unsigned ul = __float_as_uint(lo[j]);
    unsigned uh = __float_as_uint(hi[j]);
    ul = (ul + 0x7fffu + ((ul >> 16) & 1u)) >> 16;  // round to nearest even, as torch does
    uh = (uh + 0x7fffu + ((uh >> 16) & 1u)) >> 16;
    out[i] = (ul & 0xffffu) | (uh << 16);
  }
}

}  // namespace

int64_t slot_blocks(int64_t maxn, int64_t words) {
  const int64_t per_block = static_cast<int64_t>(kThreads) * words;
  return (maxn / 2 + per_block - 1) / per_block;
}

void slot_all_reduce(const torch::Tensor& inp, const torch::Tensor& out, const torch::Tensor& bufs,
                     int64_t ctrl_ptr, const torch::Tensor& order, const torch::Tensor& seqs,
                     int64_t row, int64_t maxn, int64_t words, bool push) {
  TORCH_CHECK(inp.is_cuda() && out.is_cuda() && bufs.is_cuda() && seqs.is_cuda() && order.is_cuda(),
              "slot_all_reduce: tensors must be on the device");
  TORCH_CHECK(order.scalar_type() == torch::kInt32 && order.numel() == 1 + kMaxRows,
              "slot_all_reduce: the order is int32 [count, rows...] of ", 1 + kMaxRows);
  TORCH_CHECK(inp.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16,
              "slot_all_reduce: bf16 only");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "slot_all_reduce: contiguous only");
  TORCH_CHECK(inp.numel() % 2 == 0 && inp.numel() <= maxn && out.numel() == inp.numel(),
              "slot_all_reduce: an even element count up to ", maxn);
  TORCH_CHECK(row >= 0 && row < kMaxRows, "slot_all_reduce: row out of range");
  const int64_t n2 = inp.numel() / 2;
  if (n2 == 0) return;
  const c10::cuda::CUDAGuard guard(inp.device());
  // a counter per block at one word per thread (the most blocks), then the late-block count
  TORCH_CHECK(seqs.numel() > slot_blocks(maxn, 1), "slot_all_reduce: too few counters");
  const int64_t per_block = static_cast<int64_t>(kThreads) * words;
  const unsigned blocks = static_cast<unsigned>((n2 + per_block - 1) / per_block);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const auto* x = reinterpret_cast<const unsigned*>(inp.data_ptr());
  auto* y = reinterpret_cast<unsigned*>(out.data_ptr());
  auto* ctrl = reinterpret_cast<int64_t*>(ctrl_ptr);
  const int r = static_cast<int>(row);
  const int m2 = static_cast<int>(maxn / 2);
  const int* ord = order.data_ptr<int>();
#define EVERSERVE_SLOT_LAUNCH(W)                                                                  \
  case W:                                                                                         \
    if (push)                                                                                     \
      slot_all_reduce<W, true><<<blocks, kThreads, 0, stream>>>(x, y, static_cast<int>(n2),      \
          bufs.data_ptr<int64_t>(), ctrl, ord, seqs.data_ptr<int64_t>(), r, m2);                \
    else                                                                                          \
      slot_all_reduce<W, false><<<blocks, kThreads, 0, stream>>>(x, y, static_cast<int>(n2),     \
          bufs.data_ptr<int64_t>(), ctrl, ord, seqs.data_ptr<int64_t>(), r, m2);                \
    break;
  switch (words) {
    EVERSERVE_SLOT_LAUNCH(16)
    EVERSERVE_SLOT_LAUNCH(8)
    EVERSERVE_SLOT_LAUNCH(4)
    EVERSERVE_SLOT_LAUNCH(2)
    EVERSERVE_SLOT_LAUNCH(1)
    default:
      TORCH_CHECK(false, "slot_all_reduce: words per thread must be 1, 2, 4, 8 or 16");
  }
#undef EVERSERVE_SLOT_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t device_alloc(int64_t bytes) {
  void* p = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(bytes)));
  C10_CUDA_CHECK(cudaMemset(p, 0, static_cast<size_t>(bytes)));
  return reinterpret_cast<int64_t>(p);
}

void device_free(int64_t ptr) { C10_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr))); }

void memset_async(int64_t ptr, int64_t bytes) {
  C10_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(ptr), 0, static_cast<size_t>(bytes),
                                 at::cuda::getCurrentCUDAStream().stream()));
}

pybind11::bytes ipc_handle(int64_t ptr) {
  cudaIpcMemHandle_t h;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&h, reinterpret_cast<void*>(ptr)));
  return pybind11::bytes(reinterpret_cast<const char*>(&h), sizeof(h));
}

int64_t ipc_open(const std::string& handle) {
  TORCH_CHECK(handle.size() == sizeof(cudaIpcMemHandle_t), "ipc_open: not an IPC handle");
  cudaIpcMemHandle_t h;
  std::memcpy(&h, handle.data(), sizeof(h));
  void* p = nullptr;
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

void ipc_close(int64_t ptr) { C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr))); }

}  // namespace everserve
