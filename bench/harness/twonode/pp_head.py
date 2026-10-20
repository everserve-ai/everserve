# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The head of a TP1 x PP2 engine over two pods, stock vLLM's multi-node mp mode, in-process.

Usage: python pp_head.py MODEL HEAD_IP OUT_JSON
Phases, each gated by a file the laptop touches: /workspace/GO_KILLED after node-b's worker is
killed, /workspace/GO_REJOINED after node-b's headless process was restarted.
"""

import faulthandler
import json
import os
import signal
import sys
import threading
import time

model, head_ip, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["VLLM_HOST_IP"] = head_ip
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
faulthandler.register(signal.SIGUSR1, all_threads=True)
from vllm import EngineArgs, LLMEngine, SamplingParams  # noqa: E402 - after the env is set
from vllm.inputs import TokensPrompt  # noqa: E402

rec = {"phases": {}}


def save():
    """Write the record so far."""
    json.dump(rec, open(out_path, "w"), indent=1, default=str)


t0 = time.time()
engine = LLMEngine.from_engine_args(
    EngineArgs(
        model=model,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        nnodes=2,
        node_rank=0,
        master_addr=head_ip,
        master_port=29501,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.6,
        seed=0,
        enforce_eager=False,
    )
)
core = engine.engine_core.engine_core
ex = core.model_executor
rec["engine_ready_s"] = round(time.time() - t0, 1)
rec["executor"] = {
    "class": type(ex).__name__,
    "world_size": ex.world_size,
    "local_world_size": ex.local_world_size,
    "local_workers": len(ex.workers),
    "response_mqs": [
        type(m).__name__ + ("(remote)" if getattr(m, "_is_remote_reader", False) else "(local)")
        for m in ex.response_mqs
    ],
    "broadcast_handle": str(ex.rpc_broadcast_mq.handle),
}
save()


def run(name, tokens=64, prompts=4, watchdog_s=60.0):
    """Generate; a watchdog marks the phase hung and dumps stacks if it exceeds watchdog_s."""
    hung = {"v": False}

    def wd():
        time.sleep(watchdog_s)
        if name in rec["phases"] and "tps" in rec["phases"][name]:
            return
        hung["v"] = True
        faulthandler.dump_traceback(
            all_threads=True, file=open(f"/workspace/stacks_{name}.txt", "w")
        )
        rec["phases"][name] = {"hung_s": watchdog_s}
        save()
        os._exit(3)

    threading.Thread(target=wd, daemon=True).start()
    sp = SamplingParams(max_tokens=tokens, temperature=0, ignore_eos=True)
    for i in range(prompts):
        engine.add_request(f"{name}-{i}", TokensPrompt(prompt_token_ids=[1000 + i] * 64), sp)
    t, n, steps, first = time.time(), 0, 0, None
    try:
        while engine.has_unfinished_requests():
            outs = engine.step()
            steps += 1
            for o in outs:
                n = max(n, len(o.outputs[0].token_ids))
            if first is None:
                first = time.time() - t
    except Exception as e:
        rec["phases"][name] = {
            "error": f"{type(e).__name__}: {e}",
            "after_s": round(time.time() - t, 2),
            "steps": steps,
        }
        save()
        return False
    dt = time.time() - t
    rec["phases"][name] = {
        "tokens": n,
        "steps": steps,
        "s": round(dt, 2),
        "tps": round(prompts * n / dt, 1),
        "first_token_s": round(first or 0, 3),
    }
    save()
    return True


run("baseline")
run("baseline2")
open("/workspace/HEAD_READY", "w").close()
while not os.path.exists("/workspace/GO_KILLED"):
    time.sleep(0.5)
rec["phases"]["after_kill_started_at"] = time.time()
save()
ok = run("after_node_b_worker_killed", watchdog_s=60.0)
open("/workspace/HEAD_AFTER_KILL", "w").close()
while not os.path.exists("/workspace/GO_REJOINED"):
    time.sleep(0.5)
run("after_node_b_restarted", watchdog_s=60.0)
open("/workspace/HEAD_DONE", "w").close()
