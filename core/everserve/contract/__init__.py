# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The engine contract's process side (decision 0034, seam 1).

What a process of an engine says to its node's daemon and hears from it, independent of the
engine: the signals (:mod:`~everserve.contract.signals`), the connection that carries them
(:mod:`~everserve.contract.daemon`), the head's registration and end
(:mod:`~everserve.contract.head`), and the compile cache a process names in its registration
(:mod:`~everserve.contract.compile_cache`). An engine is supported when its adapter implements
this; nothing here names an engine.
"""
