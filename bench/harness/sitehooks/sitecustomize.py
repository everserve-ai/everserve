# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Drill-only: measure the shadow sets' device memory where they are formed, in every process.

The drill puts this directory on ``PYTHONPATH`` for the engine's workers and the spare, so Python
imports this module at start-up. With ``EVERSERVE_MEASURE_SHADOW=1`` it wraps
``everserve_vllm.groups.form_shadow_groups`` when that module is first imported, and records the
device memory in use (the driver's count, which includes NCCL's buffers, and torch's allocator)
just before and after the call. The drill reads the record by RPC (``shadow_memory`` in the run's
JSON). Nothing in the product tree changes; a run without the variable is untouched.
"""

import os
import sys

if os.environ.get("EVERSERVE_MEASURE_SHADOW") == "1":
    _TARGET = "everserve_vllm.groups"

    def _wrap(module):  # noqa: ANN001, ANN202
        form = module.form_shadow_groups

        def measured(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            import torch

            torch.cuda.synchronize()
            free0, total = torch.cuda.mem_get_info()
            alloc0 = torch.cuda.memory_allocated()
            out = form(*args, **kwargs)
            torch.cuda.synchronize()
            free1, _ = torch.cuda.mem_get_info()
            module._es_shadow_memory = {
                "device_used_before_gb": round((total - free0) / 1e9, 3),
                "device_used_after_gb": round((total - free1) / 1e9, 3),
                "device_delta_gb": round((free0 - free1) / 1e9, 3),
                "torch_allocated_delta_gb": round(
                    (torch.cuda.memory_allocated() - alloc0) / 1e9, 3
                ),
            }
            return out

        module.form_shadow_groups = measured

    class _Hook:
        def find_spec(self, name, path=None, target=None):  # noqa: ANN001, ANN201
            if name != _TARGET:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            run = spec.loader.exec_module

            def exec_module(module):  # noqa: ANN001, ANN202
                run(module)
                _wrap(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())

if os.environ.get("EVERSERVE_MIRROR_DEBUG") == "1":
    # drill-only: the worker's mirror logger at DEBUG, its own handler so vLLM's level cannot hide
    # it; one line per request and per reply, to match against the store's debug log
    import logging

    _mirror_log = logging.getLogger("everserve.state.mirror")
    _mirror_log.setLevel(logging.DEBUG)
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s.%(msecs)03d pid={os.getpid()} %(name)s %(message)s", "%H:%M:%S"
        )
    )
    _mirror_log.addHandler(_handler)
    _mirror_log.propagate = False

if os.environ.get("EVERSERVE_STACKS_DIR"):
    # drill-only: SIGUSR1 writes every thread's Python stack to stacks_<pid>.txt, even while the
    # main thread is blocked in a CUDA or NCCL call (faulthandler needs no GIL and no ptrace); the
    # drill signals the survivors after a kill to see where they wait (decision 0021)
    import faulthandler
    import signal

    _stacks = open(  # noqa: SIM115 - held for the process's life, faulthandler writes to its fd
        os.path.join(os.environ["EVERSERVE_STACKS_DIR"], f"stacks_{os.getpid()}.txt"), "w"
    )
    faulthandler.register(signal.SIGUSR1, file=_stacks, all_threads=True, chain=False)
