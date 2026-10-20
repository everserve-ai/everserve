# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The engine arguments every process of an engine builds its configuration from.

The head (``vllm serve``, or NVIDIA Dynamo's ``dynamo.vllm``, decision 0041), the executor of a
node without the engine (:mod:`everserve_vllm.node_main`) and a spare
(:mod:`everserve_vllm.spare_main`) must agree on every buffer size, split point and numeric
setting, so each takes the same vLLM engine arguments, in vLLM's own spelling
(``--tensor-parallel-size``, ``--max-model-len`` and so on). A deployment passes one argument list
to every process of the engine. A process parses its own options and hands the rest to vLLM's
parser untouched (:func:`config`); it reads the model and the widths it needs for itself from the
same list without vLLM (:func:`identity`).

What this adapter requires of every process's engine, whatever the deployment passed, is
:func:`required` (the KV connector, the compile settings, vLLM's custom all-reduce off), applied
over the parsed arguments, and printed as vLLM flags by :mod:`everserve_vllm.flags` for a head
launched by a command line this adapter does not build.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Any

log = logging.getLogger(__name__)

CONNECTOR = {
    "kv_connector": "EverserveConnector",
    "kv_connector_module_path": "everserve_vllm.connector",
    "kv_role": "kv_both",
}
"""The KV connector every process of an engine runs: committed rows to the mirror each step."""

USAGE_CONTEXT = "OPENAI_API_SERVER"
"""The vLLM usage context every process of an engine builds its configuration under: the head's.

vLLM sizes defaults by it (``EngineArgs.get_batch_defaults``: ``max_num_seqs`` and
``max_num_batched_tokens``, and through them the warm-up's batch and the graphs' sizes), so a
node or a spare that built its configuration under another context, or none, would step with
other buffers than its head: the stages' first sampled-token broadcast differs in shape and the
engine hangs in its warm-up (``max_num_seqs`` 256 on the head against 128 on a node). Both heads
this adapter serves name it: ``vllm serve`` and NVIDIA Dynamo's ``dynamo.vllm``
(``main.py``)."""


@dataclasses.dataclass(frozen=True)
class Identity:
    """What a process reads from the engine arguments for itself: the model and the widths."""

    model: str
    tp: int
    pp: int


def identity(argv: list[str]) -> Identity:
    """The model and the parallel widths in ``argv``, vLLM's engine arguments, read without vLLM.

    Raises:
        SystemExit: when ``argv`` names no model.
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model")
    ap.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    ap.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
    a, _ = ap.parse_known_args(argv)
    if not a.model:
        raise SystemExit("the engine arguments name no model (--model)")
    return Identity(a.model, a.tensor_parallel_size, a.pipeline_parallel_size)


def required(guarantee: Any = None) -> dict[str, Any]:
    """The engine settings this adapter requires of every process, as ``EngineArgs`` fields.

    ``guarantee`` is the engine's guarantee class (the policy's ``Engine.guarantee``), which
    decides the compile settings (:func:`everserve_vllm.collective_op.compilation_config`).
    """
    from everserve_vllm.collective_op import compilation_config

    return {
        "compilation_config": compilation_config(guarantee),
        "kv_transfer_config": dict(CONNECTOR),
        "disable_custom_all_reduce": True,
    }


def config(argv: list[str], headless: bool = False, guarantee: Any = None, **parallel: Any) -> Any:
    """The engine configuration from ``argv`` (vLLM's engine arguments) with :func:`required` over.

    ``parallel`` adds the multi-node placement (``nnodes``, ``node_rank``, ``master_addr``,
    ``master_port``). ``headless`` builds the configuration of a node without the engine, as
    ``vllm serve --headless`` does. The backend is vLLM's multiprocess one: a node's executor is
    constructed by name (:mod:`everserve_vllm.node_main`) and a spare constructs none, and named,
    vLLM does not count this node's GPUs against the engine's world. The usage context is the
    head's (:data:`USAGE_CONTEXT`).
    """
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser(prog="everserve engine arguments"))
    args = EngineArgs.from_cli_args(parser.parse_args(argv))
    ours = required(guarantee)
    for name in ours:
        flag = "--" + name.replace("_", "-")
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            log.warning("engine argument %s is the adapter's; the value given is replaced", flag)
    args = dataclasses.replace(args, distributed_executor_backend="mp", **ours, **parallel)
    return args.create_engine_config(usage_context=UsageContext[USAGE_CONTEXT], headless=headless)
