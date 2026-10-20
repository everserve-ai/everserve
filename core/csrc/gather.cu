// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//
// One step's committed KV rows, every layer and every held head slot, in one launch.
//
// The mirror stages a step as [layers, heads, capacity, 2, d] in a pinned file mapped into the
// device address space (decision 0011). This kernel reads each (slot, head, layer) row of the
// paged cache and writes it straight there, so a step costs one launch on the side stream instead
// of a gather, a transpose and a copy per layer. A cache layout is five element strides: block,
// offset in block, head slot, K-or-V, and element. Padding slot ids (negative) read row 0, as the
// torch path does; the store skips them.

#include "kernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace everserve {
namespace {

struct Layout {
  int64_t block, offset, head, kv, elem;
};

template <typename T>
__global__ void gather_elements(const int64_t* __restrict__ layer_ptrs,
                                const int64_t* __restrict__ slots,
                                const int64_t* __restrict__ heads, T* __restrict__ out,
                                int heads_n, int64_t capacity, int d, int64_t block_size,
                                Layout s) {
  const int64_t t = blockIdx.x;
  const int h = blockIdx.y;
  const int l = blockIdx.z;
  int64_t slot = slots[t];
  slot = slot < 0 ? 0 : slot;
  const T* src = reinterpret_cast<const T*>(layer_ptrs[l]) + (slot / block_size) * s.block +
                 (slot % block_size) * s.offset + heads[h] * s.head;
  T* dst = out + ((static_cast<int64_t>(l) * heads_n + h) * capacity + t) * 2 * d;
  for (int i = threadIdx.x; i < 2 * d; i += blockDim.x) {
    const int k = i / d;
    const int j = i - k * d;
    dst[i] = src[k * s.kv + j * s.elem];
  }
}

// Each half-row (the K or the V of one head) is d contiguous elements in both places: move it in
// 16-byte words.
template <typename T>
__global__ void gather_vectors(const int64_t* __restrict__ layer_ptrs,
                               const int64_t* __restrict__ slots,
                               const int64_t* __restrict__ heads, T* __restrict__ out,
                               int heads_n, int64_t capacity, int d, int64_t block_size,
                               Layout s) {
  const int64_t t = blockIdx.x;
  const int h = blockIdx.y;
  const int l = blockIdx.z;
  int64_t slot = slots[t];
  slot = slot < 0 ? 0 : slot;
  const T* src = reinterpret_cast<const T*>(layer_ptrs[l]) + (slot / block_size) * s.block +
                 (slot % block_size) * s.offset + heads[h] * s.head;
  T* dst = out + ((static_cast<int64_t>(l) * heads_n + h) * capacity + t) * 2 * d;
  const int words = d * static_cast<int>(sizeof(T)) / 16;
  for (int i = threadIdx.x; i < 2 * words; i += blockDim.x) {
    const int k = i / words;
    const int j = i - k * words;
    reinterpret_cast<uint4*>(dst + k * d)[j] = reinterpret_cast<const uint4*>(src + k * s.kv)[j];
  }
}

template <typename T>
void launch(const torch::Tensor& layer_ptrs, const torch::Tensor& slots,
            const torch::Tensor& heads, int64_t out_ptr, int64_t capacity, int64_t d,
            int64_t block_size, const Layout& s, bool vectors, cudaStream_t stream) {
  const int64_t n = slots.size(0);
  const dim3 grid(static_cast<unsigned>(n), static_cast<unsigned>(heads.size(0)),
                  static_cast<unsigned>(layer_ptrs.size(0)));
  T* out = reinterpret_cast<T*>(out_ptr);
  if (vectors) {
    const int words = static_cast<int>(2 * d * sizeof(T) / 16);
    const int threads = words < 256 ? ((words + 31) / 32) * 32 : 256;
    gather_vectors<T><<<grid, threads, 0, stream>>>(
        layer_ptrs.data_ptr<int64_t>(), slots.data_ptr<int64_t>(), heads.data_ptr<int64_t>(),
        out, static_cast<int>(heads.size(0)), capacity, static_cast<int>(d), block_size, s);
  } else {
    const int threads = 2 * d < 256 ? static_cast<int>(((2 * d + 31) / 32) * 32) : 256;
    gather_elements<T><<<grid, threads, 0, stream>>>(
        layer_ptrs.data_ptr<int64_t>(), slots.data_ptr<int64_t>(), heads.data_ptr<int64_t>(),
        out, static_cast<int>(heads.size(0)), capacity, static_cast<int>(d), block_size, s);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool aligned16(int64_t v) { return (v & 15) == 0; }

}  // namespace

void gather_rows(const torch::Tensor& layer_ptrs, const torch::Tensor& slots,
                 const torch::Tensor& heads, int64_t out_ptr, int64_t capacity, int64_t d,
                 int64_t block_size, const std::vector<int64_t>& strides, int64_t elem_size) {
  TORCH_CHECK(layer_ptrs.is_cuda() && slots.is_cuda() && heads.is_cuda(),
              "gather_rows: index tensors must be on the device");
  TORCH_CHECK(layer_ptrs.scalar_type() == torch::kLong && slots.scalar_type() == torch::kLong &&
                  heads.scalar_type() == torch::kLong,
              "gather_rows: index tensors must be int64");
  TORCH_CHECK(strides.size() == 5, "gather_rows: a layout is five strides");
  TORCH_CHECK(slots.size(0) <= capacity, "gather_rows: more slots than the staging capacity");
  if (slots.size(0) == 0 || heads.size(0) == 0 || layer_ptrs.size(0) == 0) return;
  const Layout s{strides[0], strides[1], strides[2], strides[3], strides[4]};
  const c10::cuda::CUDAGuard guard(slots.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  // 16-byte words when every half-row is contiguous and 16-byte aligned at both ends; the layer
  // base pointers are checked by the caller, which holds them on the host.
  const bool vectors = s.elem == 1 && aligned16(d * elem_size) && aligned16(out_ptr) &&
                       aligned16(s.block * elem_size) && aligned16(s.offset * elem_size) &&
                       aligned16(s.head * elem_size) && aligned16(s.kv * elem_size);
  switch (elem_size) {
    case 1:
      launch<uint8_t>(layer_ptrs, slots, heads, out_ptr, capacity, d, block_size, s, vectors,
                      stream);
      break;
    case 2:
      launch<uint16_t>(layer_ptrs, slots, heads, out_ptr, capacity, d, block_size, s, vectors,
                       stream);
      break;
    case 4:
      launch<uint32_t>(layer_ptrs, slots, heads, out_ptr, capacity, d, block_size, s, vectors,
                       stream);
      break;
    default:
      TORCH_CHECK(false, "gather_rows: unsupported element size ", elem_size);
  }
}

int64_t host_register_mapped(int64_t ptr, int64_t bytes) {
  void* host = reinterpret_cast<void*>(ptr);
  C10_CUDA_CHECK(cudaHostRegister(host, static_cast<size_t>(bytes),
                                  cudaHostRegisterMapped | cudaHostRegisterPortable));
  void* device = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
  return reinterpret_cast<int64_t>(device);
}

void host_unregister(int64_t ptr) {
  C10_CUDA_CHECK(cudaHostUnregister(reinterpret_cast<void*>(ptr)));
}

}  // namespace everserve
