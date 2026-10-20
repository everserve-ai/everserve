# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Placement-independent arithmetic, and the reference cores built on it (decision 0004).

* :mod:`everserve.reference.arith.ops`: RMS norm and rotary embeddings in the Hugging Face
  conventions.
* :mod:`everserve.reference.arith.sources`: where a rank reads the slice of each weight its units
  need.
* :mod:`everserve.reference.arith.cores`: reference model cores (Qwen3-MoE, Llama) whose decode
  output is bit-identical before and after a membership change. The production kernels are
  checked against these; the CPU tests run them. No adapter imports them, and no served model is
  ported to them.
"""
