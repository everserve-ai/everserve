// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings of the Everserve CUDA kernels (`everserve.kernels` builds and loads them).

#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // the slot region's layout this build was made with (kernels.h); the Python side checks its
  // own copy against these before it maps a region
  m.attr("MAX_ROWS") = everserve::kMaxRows;
  m.attr("CTRL_WORDS") = everserve::kCtrlWords;
  m.attr("ABORT") = everserve::kAbort;
  m.attr("LATE") = everserve::kLate;
  m.attr("BOUND") = everserve::kBound;
  m.doc() = "Everserve CUDA kernels: the step's KV gather and the slot collective.";
  m.def("gather_rows", &everserve::gather_rows,
        "Copy a step's rows of every layer and held head into a [L, H, cap, 2, d] buffer.");
  m.def("host_register_mapped", &everserve::host_register_mapped,
        "Pin host memory mapped into the device address space; returns the device pointer.");
  m.def("host_unregister", &everserve::host_unregister, "Unpin host memory.");
  m.def("slot_all_reduce", &everserve::slot_all_reduce,
        "The slot collective: sum bf16 over the row's members, graph-capturable.");
  m.def("slot_blocks", &everserve::slot_blocks, "Sequence counters for tensors up to maxn at a words-per-thread setting.");
  m.def("device_alloc", &everserve::device_alloc, "cudaMalloc, zeroed.");
  m.def("device_free", &everserve::device_free, "cudaFree.");
  m.def("memset_async", &everserve::memset_async, "Zero device memory on the current stream.");
  m.def("ipc_handle", &everserve::ipc_handle, "A CUDA IPC handle for a device_alloc buffer.");
  m.def("ipc_open", &everserve::ipc_open, "Map another process's buffer from its IPC handle.");
  m.def("ipc_close", &everserve::ipc_close, "Unmap an ipc_open mapping.");
}
