# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The reference tier (decision 0004): what the CPU tests run SHRINK against.

Decision 0004 splits the runtime in two tiers. The product tier runs REPLACE through an engine's
adapter: a lost rank's position taken by a spare, the engine's own step loop resumed at a step
boundary (:meth:`everserve.comm.runtime.Runtime.at_boundary`). The reference tier is the
mechanism in Python, complete and exact, that the CPU spec in ``core/tests`` runs under SHRINK on
four gloo processes: a rank lost between steps or mid-step leaves every token and logit unchanged.

Nothing here is on a product path: no adapter imports this package, and no served model is ported
to its cores. It ships in the wheel so the tests run against an installed package, and so a reader
of the mechanism has it beside the product modules it is the reference for.

* :mod:`everserve.reference.control`: the executor-to-worker fault channel the tests' executor
  flags a loss over.
* :mod:`everserve.reference.world`: leave a broken default ``torch.distributed`` world and join a
  new one at the same size.
* :mod:`everserve.reference.store`, :mod:`everserve.reference.kvcopy`,
  :mod:`everserve.reference.committed`: the Python KV store process and its client, the loopback
  copy of a rank's step-committed rows, and the ``CommittedKV`` interface over them, synchronous
  and exact. The production transport is :mod:`everserve.state.mirror`.
* :mod:`everserve.reference.plan`: placement plans: even splits, expert chunks, the greedy plan
  after a loss, and the capacity a rank reserves so absorbing its share needs no reallocation.
* :mod:`everserve.reference.staging`: which KV blocks to copy to a replacement during a notice.
* :mod:`everserve.reference.recapture`: the order CUDA graphs come back in after a change.
* :mod:`everserve.reference.arith`: placement-independent arithmetic and the reference model
  cores (Qwen3-MoE, Llama) whose decode output is bit-identical across a membership change.

The step guard these run under, :meth:`everserve.comm.runtime.Runtime.run_step` and the agreement
all-reduce it ends with, stays beside the product path in :mod:`everserve.comm.runtime`, whose
docstring says which members are which.
"""
