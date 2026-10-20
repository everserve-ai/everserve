// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//
// The Everserve CUDA kernels, as the torch extension `everserve_kernels` exposes them.

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace everserve {

// The slot region's layout the kernels and `everserve.comm.slotkernel` share. The extension
// exports them (MAX_ROWS, CTRL_WORDS, ABORT, LATE, BOUND), and the Python side refuses to map
// a region when its own copy differs: a build from other sources would read other words.
constexpr int kMaxRows = 8;     // rows in a region: up to seven ranks and the spare
constexpr int kCtrlWords = 16;  // control words per row
constexpr int kAbort = 1;       // control word: abort
constexpr int kLate = 11;       // control word: a mask of the rows the member waits too long for
constexpr int kBound = 12;      // control word: the wait in nanoseconds after which it names them

// Copy one step's rows of every layer and held head slot into a staging buffer laid out
// [layers, heads, capacity, 2, d], on the current stream. `layer_ptrs` holds each layer's cache
// base address; `strides` are the layout's element strides (block, offset, head, kv, element).
void gather_rows(const torch::Tensor& layer_ptrs, const torch::Tensor& slots,
                 const torch::Tensor& heads, int64_t out_ptr, int64_t capacity, int64_t d,
                 int64_t block_size, const std::vector<int64_t>& strides, int64_t elem_size);

// Pin `bytes` of host memory at `ptr`, mapped into the device address space; returns the device
// pointer that kernels write through.
int64_t host_register_mapped(int64_t ptr, int64_t bytes);

// Undo host_register_mapped.
void host_unregister(int64_t ptr);

// The slot collective (slot.cu): sum bf16 `inp` over the members this row's control words list,
// in the listed order, into `out`, on the current stream. `bufs` is the device table of every
// row's word buffer, `ctrl_ptr` the device address of the region's control rows (for the abort
// word), `order` this process's device copy of its member order, int32 [count, rows...], and
// `seqs` its counters.
void slot_all_reduce(const torch::Tensor& inp, const torch::Tensor& out, const torch::Tensor& bufs,
                     int64_t ctrl_ptr, const torch::Tensor& order, const torch::Tensor& seqs,
                     int64_t row, int64_t maxn, int64_t words, bool push);

// Sequence counters the slot collective needs for tensors of up to `maxn` elements when each
// thread moves `words` eight-byte words.
int64_t slot_blocks(int64_t maxn, int64_t words);

// Device memory outside torch's caching allocator, so a CUDA IPC handle names it exactly.
int64_t device_alloc(int64_t bytes);
void device_free(int64_t ptr);
void memset_async(int64_t ptr, int64_t bytes);

// CUDA IPC: a handle for this process's buffer, and a mapping of another process's.
pybind11::bytes ipc_handle(int64_t ptr);
int64_t ipc_open(const std::string& handle);
void ipc_close(int64_t ptr);

}  // namespace everserve
