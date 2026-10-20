# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Communicators and the per-process membership runtime.

The product path, what the vLLM adapter drives:

* :mod:`everserve.comm.collective`: the collective a model's traffic goes through, with the
  replacement sets pre-formed beside the live group and the fabric family a pipeline spans.
* :mod:`everserve.comm.cover`: cover plans (decision 0035): the groups a process forms ahead of
  a loss and reports to its daemon.
* :mod:`everserve.comm.slotkernel`, :mod:`everserve.comm.slot`, :mod:`everserve.comm.slotwatch`:
  the slot collective (decisions 0006, 0012, 0021): a shared region, its kernel, and the
  watchdog that names a dead row.
* :mod:`everserve.comm.exchange`: named tensors between pipeline stages over any communicator.
* :mod:`everserve.comm.guard`: the shared and exclusive guard an abort takes against calls.
* :mod:`everserve.comm.runtime`: the step guard: fault flag, epoch, the change at a boundary.
* :mod:`everserve.comm.handle`, :mod:`everserve.comm.layout`, :mod:`everserve.comm.p2pprobe`:
  the indirected group handle captured graphs call through, the parallel layout, and the
  peer-to-peer probe the variant choice reads.

The group and set machinery the collective is built on:

* :mod:`everserve.comm.group`: a process group that can lose a member (``ncclCommShrink`` or
  recreate over a store); the adapter's collective op is one.
* :mod:`everserve.comm.shadow`: member sets, their keys and the rendezvous a group is pre-formed
  at; the survivor and replacement sets beside the live group are these.

The reference tier (decision 0004) is :mod:`everserve.reference`: the executor-to-worker fault
channel and the world re-form the CPU tests run SHRINK under live there, and the step guard they
run under is :meth:`everserve.comm.runtime.Runtime.run_step`, beside the product path's
:meth:`~everserve.comm.runtime.Runtime.at_boundary`.
"""
