# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Committed state: KV that outlives the rank that produced it.

* :mod:`everserve.state.committed`: the ``CommittedKV`` interface an engine's connector drives.
* :mod:`everserve.state.mirror`: the production transport: each step's rows into the node's
  store over shared memory, the commit asynchronous behind a one-step fence (decisions 0009,
  0011, 0019).
* :mod:`everserve.state.kvcopy`: the paged cache layouts and the rows of one head slot in them,
  which the transports and the adapter read and write through.

The loopback transport and the Python store the CPU tests run the same interface against are
:mod:`everserve.reference.kvcopy`, :mod:`everserve.reference.committed` and
:mod:`everserve.reference.store` (the reference tier, decision 0004).
"""
