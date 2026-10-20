# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Drill driver for the adapter port: one engine, eight sessions, one arm per invocation.

    python bench/harness/adapter_drill.py --arm replace --kill between --model M --out run.json

Arms (``bench/runs/*adapter-port-4gpu/PREREGISTRATION.md``):

* ``stock``    stock vLLM at the pin: CUDA graphs, async scheduling, serving GPUs 0-3;
* ``adapter``  the adapter with everything resident and no fault (the overhead arm);
* ``adapter_nomirror``  as ``adapter`` with the mirror off (``null_connector.py``), to attribute
  the overhead between the slot kernel and the commit;
* ``replace``  the adapter; SIGKILL of ``--kill-rank`` after step ``--kill-step``, ``--kill
  between`` (at a step boundary) or ``--kill mid`` (``--mid-frac`` of a median step into the
  next step);
* ``follow``   the adapter; every change the operator sets in motion elsewhere (a drain through
  the coordinator) is completed as it commits, any number of them, and nothing is sent or
  killed here: with ``--kill external`` the harness acts once ``<out>.armed`` appears;
* ``notice``   the adapter; a ``Notice`` for ``--kill-rank`` after step ``--kill-step`` with a
  deadline ``--notice-steps`` decode steps out (seconds from the measured step time), and the
  SIGKILL performed once the daemon's replacement has been completed at the deadline.

The engine core runs in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so the driver can
reach it: this script plays the engine core's hook and calls
``EverserveExecutor.complete_change`` when a step raises ``RankLostError`` or the daemon commits a
change. It talks to the node daemon over its socket as a client: the notice goes out on it and
every ``MembershipCommitted`` comes back on it, timestamped.

Every session's token emission times are recorded; the pause is the largest per-session
inter-token gap.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import threading
import time
from pathlib import Path
from typing import Any

STAMPS: dict[str, float] = {"process": time.time()}
"""When the run reached each phase, this host's clock: the process (before vLLM's import), vLLM
imported, the engine's arguments, the engine ready, the first request added. A stock restart's
relaunch-to-first-step split at them (stock_restart.py): most of a stock restart's time (59 s of
85 on H100) lies before vLLM's own load timer, and the stamps name it."""

TEXT = (
    "The history of the lighthouse begins with fires lit on hilltops to guide ships at night. "
    "Over centuries these became towers with oil lamps, then lenses, then electric lights. "
)
ADAPTER = ("adapter", "adapter_nomirror", "replace", "notice", "follow")
ROWS_CHECK_DELAY_STEPS = 20
"""Steps after the first one back before the post-resume row check (off the measured gap)."""
REPO = Path(__file__).resolve().parents[2]


def prompts(model: str, length: int, vocab: int, n: int) -> list[list[int]]:
    """Seeded random prompts; the first is natural text so output can be read by eye."""
    import torch

    g = torch.Generator().manual_seed(1234)
    p = torch.randint(100, min(vocab, 100000), (n, length), generator=g).tolist()
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
        ids: list[int] = []
        while len(ids) < length:
            ids += tok.encode(TEXT, add_special_tokens=False)
        p[0] = ids[-length:]
    except Exception as e:  # noqa: BLE001 - a missing tokenizer only changes prompt 0
        print("tokenizer unavailable:", e, flush=True)
    return p


def _device_memory(worker: Any) -> dict[str, Any]:
    """RPC target: this rank's device memory in use, by the driver's count and torch's."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "rank": worker.worker.rank,
        "used_gb": round((total - free) / 1e9, 3),
        "torch_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
    }


def _shadow_memory(worker: Any) -> dict[str, Any] | None:
    """RPC target: the shadow sets' memory as measured at their formation (sitehooks)."""
    import sys

    mod = sys.modules.get("everserve_vllm.groups")
    rec = getattr(mod, "_es_shadow_memory", None)
    return {"rank": worker.worker.rank, **rec} if rec else None


def post_resume(batches: list[dict], t_resumed: float | None) -> dict[str, dict[str, dict]]:
    """Per rank, the first batch after the resume: each request's tokens, count and prefill flag.

    The first step after a resume is a one-token chunked prefill in vLLM's terms (is_prefilling,
    prefill_len the committed count plus one): the resume working as designed, not a recompute.
    A request run from below its committed count, or with more than one token, was recomputed on
    the spare, and an exact read would then say less than it seems to (:func:`re_prefilled`).
    """
    out: dict[str, dict[str, dict]] = {}
    if t_resumed is None:
        return out
    for rk in batches or []:
        first = next((b for b in rk.get("batches") or [] if b["t"] >= t_resumed), None)
        if first is None:
            continue
        ids = first.get("req_ids") or []
        none = [None] * len(ids)
        computed = first.get("num_computed_prefill_tokens") or none
        prefilling = first.get("is_prefilling") or none
        out[str(rk["rank"])] = {
            rid: {
                "scheduled": first["num_scheduled_tokens"][i],
                "computed": computed[i],
                "prefilling": prefilling[i],
            }
            for i, rid in enumerate(ids)
        }
    return out


def resume_cause(committed: int | None, resume_from: int, num_tokens: int) -> dict:
    """Where a request resumed against its committed count, and the cause when they differ.

    ``at_committed``: the two are equal. ``last_token_died``: the committed count reached the
    token count, so the last sampled token died with the rank before the engine received it,
    and the resume is one below to recompute that position and sample it again
    (:func:`everserve_vllm.resume.resume_point`). ``none_committed``: no committed count (no
    store held the request's rows). ``below_committed``: any other shortfall, a resume below
    the committed state, which the summary scores as a kill.
    """
    if committed is None:
        cause = "none_committed"
    elif resume_from == committed:
        cause = "at_committed"
    elif committed >= num_tokens and resume_from == num_tokens - 1:
        cause = "last_token_died"
    else:
        cause = "below_committed"
    return {
        "committed": committed,
        "resume_from": resume_from,
        "tokens": num_tokens,
        "cause": cause,
    }


def re_prefilled(post: dict[str, dict[str, dict]], committed: dict[str, int]) -> list[str]:
    """Requests some rank ran from below their committed count, or with more than one token."""
    bad = set()
    for per in post.values():
        for rid, b in per.items():
            c = committed.get(rid)
            below = c is not None and b["computed"] is not None and b["computed"] < c
            if b["scheduled"] > 1 or below:
                bad.add(rid)
    return sorted(bad)


def restore_coverage(change: dict) -> dict[str, dict]:
    """Per spare, the rows its restores copied against the rows its committed counts name.

    The expected rows are the position's tables (its restore check's count) times the committed
    positions of every request. The ratio is 1 when the restore covered every committed position;
    a replica that never held the prompt rows reads under 1 (831/1343) with the output still
    exact in most sessions, which is why the ratio is reported and not inferred from exactness.
    """
    out: dict[str, dict] = {}
    rc = change.get("restore_check") or {}
    for rank, s in (change.get("spares") or {}).items():
        info = (s or {}).get("restore") or {}
        counts = info.get("counts") or {}
        check = rc.get(str(rank), rc.get(rank)) if isinstance(rc, dict) else None
        tables = (check or {}).get("tables") if isinstance(check, dict) else None
        rows = (info.get("rows") or 0) + ((info.get("catch_up") or {}).get("rows") or 0)
        expected = tables * sum(counts.values()) if tables and counts else None
        out[str(rank)] = {
            "rows": rows,
            "expected": expected,
            "ratio": round(rows / expected, 4) if expected else None,
        }
    return out


def session_gap(stamps: list[float], t_kill: float | None, t_resumed: float | None = None) -> dict:
    """A session's largest gap between consecutive tokens at the client, and the one at the change.

    ``stamps`` are the arrival times of the session's tokens; the gap at index ``i`` is between
    token ``i - 1`` and token ``i``. ``around_change_s`` is the gap that spans the change: from
    the last token before the kill (or the notice) to the first token after the resume, which
    is the pause a client of that session saw for it. The run's largest gap is reported beside
    it, since the drill's own probes after a change (the row digests) pause the engine for
    seconds and must not be read as the change (3.6 s probe pauses against a 0.7 s change).
    """
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    if not gaps:
        return {"max_s": None, "at": None, "around_change_s": None}
    i = max(range(len(gaps)), key=lambda k: gaps[k])
    out = {"max_s": round(gaps[i], 4), "at": i + 1, "around_change_s": None}
    if t_kill is not None:
        after = [(g, k + 1) for k, g in enumerate(gaps) if stamps[k + 1] >= t_kill]
        if after:
            if t_resumed is not None:
                spanning = [(g, at) for g, at in after if stamps[at] >= t_resumed]
                g, at = spanning[0] if spanning else after[0]
            else:
                g, at = after[0]
            out["around_change_s"] = round(g, 4)
            out["around_change_at"] = at
    return out


def main() -> None:  # noqa: C901, PLR0912, PLR0915 - a drill script reads top to bottom
    """Run one arm and write its result file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["stock", *ADAPTER], required=True)
    ap.add_argument(
        "--kill",
        choices=["between", "mid", "external"],
        default="between",
        help="external: nothing is killed here; at --kill-step the run writes <out>.armed and "
        "steps on, the harness kills elsewhere (a node) and writes the kill's time to <out>.kill "
        "as JSON, which is merged into the run's kill record; give --max-tokens room for the wait",
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--clock-offsets",
        default="",
        help="a JSON file of the harness's clock probes (clockoffset.py), copied into the record",
    )
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--pp", type=int, default=1, help="pipeline stages; over 1 spans nodes")
    ap.add_argument("--nnodes", type=int, default=1, help="nodes the engine spans (vLLM's mp mode)")
    ap.add_argument("--master-addr", default="127.0.0.1", help="this head's reachable address")
    ap.add_argument("--master-port", type=int, default=29501)
    ap.add_argument("--serve-gpus", default="0,1,2,3")
    ap.add_argument("--spare-gpu", default="4")
    ap.add_argument("--kill-step", type=int, default=100)
    ap.add_argument("--kill-rank", type=int, default=2)
    ap.add_argument("--mid-frac", type=float, default=0.5)
    ap.add_argument(
        "--stacks-after",
        type=float,
        default=0.0,
        help="seconds after the kill to make every surviving worker write its Python stacks "
        "(sitehooks, SIGUSR1); 0 is off",
    )
    ap.add_argument(
        "--notice-steps",
        type=int,
        default=300,
        help="the notice deadline, in decode steps from the notice; the seconds follow from the "
        "median step time measured so far, so the sessions span the deadline on any host",
    )
    ap.add_argument("--max-tokens", type=int, default=201)
    ap.add_argument(
        "--start-after",
        default="",
        help="a file the sessions wait for once the engine is ready: the harness prepares the "
        "engine's cover and cordons before any session runs; empty, they start at once",
    )
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=2304)
    ap.add_argument("--blocks", type=int, default=1300)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--engine", default="rig")
    ap.add_argument("--node", default="rig")
    ap.add_argument("--discovery-dir", default="/tmp/es-discovery")
    ap.add_argument("--policy", default=str(REPO / "docs/examples/policy-rig-4gpu.json"))
    ap.add_argument("--event-dir", default="")
    ap.add_argument("--custom-all-reduce", choices=["off", "default"], default="off")
    # the REPLACE-divergence diagnostic (bench/harness/divergence_probe.py); off in every
    # pre-registered arm
    ap.add_argument("--diag", action="store_true")
    # every request's K/V rows digested per position on every rank at the end of the run, and
    # the batch and scheduler logs on an arm without a change (the fault arms and their
    # controls): a run's rows against its control's, position by position, offline (summary.py)
    ap.add_argument("--kv-digest", action="store_true")
    ap.add_argument("--logprobs", type=int, default=0)
    ap.add_argument("--enforce-eager", action="store_true")
    a = ap.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # its JIT flags fail on the pod image
    os.environ["CUDA_VISIBLE_DEVICES"] = a.serve_gpus
    adapter = a.arm in ADAPTER
    if adapter:
        os.environ.update(
            {
                "EVERSERVE_GRAPH_SAFE": "1",
                "EVERSERVE_ENGINE": a.engine,
                "EVERSERVE_NODE": a.node,
                "EVERSERVE_DISCOVERY_DIR": a.discovery_dir,
                "EVERSERVE_POLICY": a.policy,
                "EVERSERVE_SPARE_GPU": a.spare_gpu,
                "EVERSERVE_SPARE_RANK": str(a.kill_rank),
                "EVERSERVE_EVENT_DIR": a.event_dir,
            }
        )
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt

    STAMPS["vllm_imported"] = time.time()

    cfg = json.loads((Path(a.model) / "config.json").read_text())
    if a.nnodes > 1:
        os.environ.setdefault("VLLM_HOST_IP", a.master_addr)  # the queues bind this address
    kw = dict(
        model=a.model,
        tensor_parallel_size=a.tp,
        pipeline_parallel_size=a.pp,
        nnodes=a.nnodes,
        node_rank=0,
        master_addr=a.master_addr,
        master_port=a.master_port,
        dtype="bfloat16",
        max_model_len=a.max_model_len,
        max_num_seqs=a.n,
        max_num_batched_tokens=a.n * a.prompt_len,
        enable_prefix_caching=True,  # the replay finds the kept blocks through it
        gpu_memory_utilization=a.gpu_util,
        num_gpu_blocks_override=a.blocks,
        seed=0,
        # experts in parallel for an MoE checkpoint, as the four-GPU runs ran; a dense one has none
        enable_expert_parallel=any(
            cfg.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")
        ),
        # vLLM's custom all-reduce is off for the adapter arms (the graph-safe op replaces the TP
        # all-reduce) and for stock unless --custom-all-reduce default asks for vLLM's own choice
        disable_custom_all_reduce=not (a.arm == "stock" and a.custom_all_reduce == "default"),
    )
    if adapter:
        # vLLM's multiprocess backend by name: its pipeline-parallel check reads supports_pp
        # off the backend before a path is resolved, and a multi-node world (nnodes > 1) is
        # allowed only under "mp"; our executor class goes on the configuration once it is
        # built, where the engine takes the class as it is
        kw["distributed_executor_backend"] = "mp"
        from everserve.policy import load as load_policy
        from everserve_vllm import engine_args

        # the adapter's requirements, as every process of the engine takes them (the compile
        # settings by the engine's guarantee class, decision 0030; the connector; custom
        # all-reduce off)
        kw.update(engine_args.required(load_policy(a.policy).engine(a.engine).guarantee))
        if a.arm == "adapter_nomirror":  # drill-only connector; workers import it by path
            here = str(Path(__file__).parent)
            os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")
            import sys

            sys.path.insert(0, here)  # the engine core runs in this process too
            kw["kv_transfer_config"]["kv_connector"] = "NullConnector"
            kw["kv_transfer_config"]["kv_connector_module_path"] = "null_connector"
    if a.enforce_eager:
        kw["enforce_eager"] = True
    # decision 0013's checks need the probe; the digest's comparison needs both arms' logs
    probe = a.diag or a.kv_digest or a.arm in ("replace", "notice")
    if a.stacks_after > 0:  # where the survivors wait after a kill (sitehooks)
        os.environ["EVERSERVE_STACKS_DIR"] = f"{Path(a.out).with_suffix('')}_stacks"
        Path(os.environ["EVERSERVE_STACKS_DIR"]).mkdir(parents=True, exist_ok=True)
    if adapter:  # the shadow sets' memory, measured where they form (bench/harness/sitehooks)
        hooks = str(Path(__file__).parent / "sitehooks")
        os.environ["PYTHONPATH"] = hooks + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.environ["EVERSERVE_MEASURE_SHADOW"] = "1"
    if probe:  # workers import the probe by path, as the null connector above
        here = str(Path(__file__).parent)
        os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")
    t0 = STAMPS["engine_args"] = time.time()
    vllm_config = EngineArgs(**kw).create_engine_config()
    if adapter:
        from everserve_vllm.executor import EverserveExecutor

        vllm_config.parallel_config.distributed_executor_backend = EverserveExecutor
    engine = LLMEngine.from_vllm_config(vllm_config)
    load_s = time.time() - t0
    STAMPS["engine_ready"] = t0 + load_s
    core = engine.engine_core.engine_core
    ex = core.model_executor
    print(f"engine ready in {load_s:.0f}s, executor {type(ex).__name__}", flush=True)
    # every rank's device memory in use once ready: the adapter's against stock's bounds what the
    # shadow sets and slot buffers cost (``bench/runs/2026-10-06-frontier-1node``, prediction 8)
    device_memory = ex.collective_rpc(_device_memory)

    client = mirror = None
    snap: dict = {}
    if adapter:
        # Drill instrumentation for prediction 5, kept out of the product tree on purpose. The
        # workers cannot import bench/, so its RPC targets travel to them by value.
        import sys

        import cloudpickle

        sys.path.insert(0, str(Path(__file__).parent))
        import restore_check

        cloudpickle.register_pickle_by_value(restore_check)
        from everserve.contract.daemon import DaemonClient
        from everserve.contract.signals import Member
        from everserve.discovery import FileDiscovery
        from everserve.state.mirror import MirrorClient

        rec = FileDiscovery(a.discovery_dir).daemon(a.node)
        client = DaemonClient(rec.address, a.engine, Member(a.node, 0))
        mirror = MirrorClient(rec.mirror_address)

    sched_log: list = []
    blocks: dict[str, list[int]] = {}
    blocks_at_kill: dict[str, list[int]] = {}
    if probe:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        import divergence_probe

        divergence_probe.log_scheduler(core, sched_log)
        print("diag:", ex.collective_rpc(divergence_probe.install_batch_log), flush=True)
    ps = prompts(a.model, a.prompt_len, cfg["vocab_size"], a.n)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=a.max_tokens,
        ignore_eos=True,
        logprobs=a.logprobs or None,
    )
    logprobs: dict[str, list] = {}
    if a.start_after:
        print(f"sessions wait for {a.start_after}", flush=True)
        while not Path(a.start_after).exists():
            time.sleep(0.2)
        STAMPS["started_after"] = time.time()
    for i in range(a.n):
        STAMPS.setdefault("first_request", time.time())
        engine.add_request(str(i), TokensPrompt(prompt_token_ids=ps[i]), sp)

    tokens: dict[str, list[int]] = {str(i): [] for i in range(a.n)}
    times: dict[str, list[float]] = {str(i): [] for i in range(a.n)}
    steps, changes = [], []
    kill_info: dict = {}
    notice_info: dict = {}
    step = -1
    t_run = time.time()

    def check_restore(r: dict) -> None:
        """After the swap, before a step touches the cache: the spare's rows vs the snapshot."""
        if snap:
            checks = {}
            for rank in r.get("ranks") or [r["rank"]]:  # the carrier's record names every position
                after = ex.collective_rpc(
                    restore_check.digests_at,
                    args=(rank, snap["slots"]),
                    unique_reply_rank=rank,
                )
                checks[rank] = restore_check.compare(snap, after)
            r["restore_check"] = checks if "ranks" in r else checks[r["rank"]]

    exact: dict = {}
    n_steps = [0]  # engine.step() calls that returned

    def after_change(r: dict) -> None:
        """The spare's runner joins the batch log; the next step's rows are checked after it."""
        if probe:
            ex.collective_rpc(divergence_probe.install_batch_log)
            exact["t_resumed"] = r["t_resumed"]
            exact["committed"] = dict(r.get("safe_tokens", {}))
            exact["pending"] = n_steps[0]  # checked once a step has been enqueued after this
            # positions each request recomputes from its committed count (decision 0013)
            # where each request resumed: its committed count, capped at tokens - 1 (resume_point)
            exact["resume_from"] = {
                q.request_id: q.num_computed_tokens for q in core.scheduler.running
            }
            exact["replay_positions"] = {
                q.request_id: q.num_tokens_with_spec - q.num_computed_tokens
                for q in core.scheduler.running
            }
            # per request, where it resumed against what was committed, and why they differ:
            # the committed count can reach the token count when the last sampled token died
            # with the rank before the engine received it (resume_point caps the resume one
            # below the token count so that position is recomputed to sample it again); any
            # other shortfall is a resume below the committed state (summary.py scores it)
            exact["resume_cause"] = {
                q.request_id: resume_cause(
                    exact["committed"].get(q.request_id),
                    q.num_computed_tokens,
                    q.num_tokens_with_spec,
                )
                for q in core.scheduler.running
            }

    def check_rows_after() -> None:
        """Steps after the first one back: every rank's committed rows against the snapshot.

        The rows checked sit below the committed count, which no later step rewrites, so the
        check reads the same rows :data:`ROWS_CHECK_DELAY_STEPS` later as it would at the first
        step back, and its pause (the digests stall the engine for seconds) lands after the
        first tokens of the resume have reached the client instead of inside the measured gap.
        """
        exact["pending"] = False
        t0 = time.time()
        exact["rows_checked"] = {"step": n_steps[0], "s": None}
        if not snap.get("all"):
            return
        after = ex.collective_rpc(restore_check.digests_all, args=(snap["slots"],))
        rows = {}
        for got in after:
            before = snap["all"][got["rank"]]
            bad = sorted(k for k in before if before[k] != got["sums"].get(k))
            rows[got["rank"]] = {"tables": len(before), "mismatched": bad}
        exact["kv_below_committed"] = rows
        if snap.get("per_slot"):
            # the spare in the dead rank's position: which slots of which request differ
            after_slots = ex.collective_rpc(
                restore_check.checksums_at,
                args=(a.kill_rank, snap["slots"]),
                unique_reply_rank=a.kill_rank,
            )
            diff = restore_check.compare_slots(snap["per_slot"], after_slots, snap["by_request"])
            exact["slot_mismatch"] = {
                rid: {"count": len(bad), "first": bad[:40]} for rid, bad in diff.items()
            }
            sample = [(key, slot) for bad in diff.values() for key, slot, _ in bad[:3]][:12]
            if sample:
                exact["kv_identity"] = ex.collective_rpc(
                    restore_check.kv_identity,
                    args=(a.kill_rank, sample[0][0], sample[0][1]),
                    unique_reply_rank=a.kill_rank,
                )
            if sample:
                exact["spare_rows"] = ex.collective_rpc(
                    restore_check.rows_at,
                    args=(a.kill_rank, sample),
                    unique_reply_rank=a.kill_rank,
                )
        exact["rows_checked"]["s"] = round(time.time() - t0, 4)  # the instrument's own pause

    def exited(pid: int) -> None:
        """Record when the killed process has finished exiting (a zombie, or gone).

        Apart from detection, which decision 0021 makes independent of it.
        """
        stat = f"/proc/{pid}/stat"
        while True:
            try:
                with open(stat) as f:
                    text = f.read()
                if text[text.rfind(")") + 2] in "ZXx":
                    break
            except OSError:
                break
            time.sleep(0.001)
        kill_info["t_exit"] = time.time()

    def stacks(dead: int) -> None:
        """After ``--stacks-after`` seconds, every surviving worker writes its Python stacks."""
        time.sleep(a.stacks_after)
        sent = []
        for w in ex.workers:
            if w.proc.pid != dead and w.proc.is_alive():
                os.kill(w.proc.pid, signal.SIGUSR1)
                sent.append(w.proc.pid)
        kill_info.update(t_stacks=time.time(), stacks_pids=sent)

    armed: dict[str, float] = {}

    def kill(pid: int, **info: object) -> None:
        kill_info.update(t_kill=time.time(), pid=pid, **info)
        os.kill(pid, 9)
        threading.Thread(target=exited, args=(pid,), daemon=True).start()
        if a.stacks_after > 0:
            threading.Thread(target=stacks, args=(pid,), daemon=True).start()

    while engine.has_unfinished_requests():
        if (
            a.arm == "replace"
            and a.kill == "mid"
            and step == a.kill_step
            and "armed" not in kill_info
        ):
            med = statistics.median(s["dt"] for s in steps[10:])
            delay = a.mid_frac * med
            kill_info.update(armed=True, delay_s=delay, median_step_s=med, step=step)
            pid = ex.workers[a.kill_rank].proc.pid
            threading.Timer(delay, kill, args=(pid,), kwargs={"mode": "mid"}).start()
        ts, tp = time.time(), time.perf_counter()
        try:
            outs = engine.step()
        except Exception as e:
            from everserve_vllm.executor import RankLostError

            if not isinstance(e, RankLostError):
                raise
            print("rank lost:", e, flush=True)
            r = ex.complete_change(core)
            check_restore(r)
            after_change(r)
            changes.append(r)
            print(
                "replaced:",
                {k: v for k, v in r.items() if k not in ("boundaries", "spare")},
                flush=True,
            )
            continue
        dt = time.perf_counter() - tp
        n_steps[0] += 1
        te = time.time()
        new = 0
        for o in outs:
            if a.logprobs and o.outputs[0].logprobs is not None:
                logprobs[o.request_id] = o.outputs[0].logprobs
            ids = list(o.outputs[0].token_ids)
            k = len(ids) - len(tokens[o.request_id])
            new += k
            times[o.request_id] += [te] * k
            tokens[o.request_id] = ids
        if a.arm in ("notice", "follow"):
            heard = ex.pending_change()
            if heard is not None:  # a commit without a death: the notice path
                r = ex.complete_change(core, heard)
                check_restore(r)
                after_change(r)
                changes.append(r)
                print(
                    "swapped:",
                    {k: v for k, v in r.items() if k not in ("boundaries", "spare")},
                    flush=True,
                )
                if a.arm == "notice" and "t_kill" not in kill_info:
                    kill(r["leaving_pid"], mode="after-notice-swap", step=step)
        if (
            exact.get("pending") is not None
            and exact["pending"] is not False
            and n_steps[0] > exact["pending"] + ROWS_CHECK_DELAY_STEPS
        ):
            check_rows_after()
        if new == 0 and not outs:
            continue
        step += 1
        steps.append({"step": step, "dt": dt, "t_start": ts, "t_end": te, "new": new})
        if a.kv_digest:  # each running request's blocks, as of the latest step it ran
            for r in core.scheduler.running:
                blocks[r.request_id] = list(
                    core.scheduler.kv_cache_manager.get_block_ids(r.request_id)[0]
                )
            if step == a.kill_step:
                blocks_at_kill.update({rid: list(b) for rid, b in blocks.items()})
        if a.diag and step == a.kill_step + 30 and not blocks:  # after any change and replay
            for r in core.scheduler.running:
                blocks[r.request_id] = list(
                    core.scheduler.kv_cache_manager.get_block_ids(r.request_id)[0]
                )
        if a.arm in ("replace", "notice") and step == a.kill_step and not snap:
            snap["slots"] = restore_check.committed_slots(core, mirror, a.engine)
            got = ex.collective_rpc(
                restore_check.digests_at,
                args=(a.kill_rank, snap["slots"]),
                unique_reply_rank=a.kill_rank,
            )
            snap["sums"] = got["sums"]
            kill_info["snapshot"] = {"slots": len(snap["slots"]), "s": got["s"], "t": time.time()}
            snap["all"] = {
                g["rank"]: g["sums"]
                for g in ex.collective_rpc(restore_check.digests_all, args=(snap["slots"],))
            }
            # the localisation of a differing row (a session that diverges a few tokens after
            # the resume): per-slot checksums of
            # the dying rank's rows, each request's slots, and the raw rows kept on its node
            snap["per_slot"] = ex.collective_rpc(
                restore_check.checksums_at,
                args=(a.kill_rank, snap["slots"]),
                unique_reply_rank=a.kill_rank,
            )
            snap["by_request"] = restore_check.request_slots(
                core, {q.request_id: q.num_computed_tokens for q in core.scheduler.running}
            )
            rows_path = f"/workspace/out/{Path(a.out).stem}.rows_rank{a.kill_rank}.pt"
            kill_info["rows_at_kill"] = ex.collective_rpc(
                restore_check.save_rows,
                args=(a.kill_rank, snap["slots"], rows_path),
                unique_reply_rank=a.kill_rank,
            )
        if (
            a.arm == "replace"
            and a.kill == "between"
            and step == a.kill_step
            and "t_kill" not in kill_info
        ):
            kill(ex.workers[a.kill_rank].proc.pid, mode="between", step=step)
        if a.kill == "external" and step == a.kill_step and not armed:
            # the harness kills on another host once this file says the run is at its kill
            # step; the steps go on meanwhile, so the kill lands with the engine serving
            armed = {"t_armed": time.time(), "step": step}
            Path(a.out + ".armed").write_text(json.dumps(armed))
        if a.kill == "external" and armed and "t_kill" not in kill_info:
            marker = Path(a.out + ".kill")
            if marker.exists():  # the harness's kill has landed: its record, and the stacks timer
                kill_info.update(json.loads(marker.read_text()), step=step)
                if a.stacks_after > 0:
                    threading.Thread(target=stacks, args=(-1,), daemon=True).start()
        if a.arm == "notice" and step == a.kill_step and not notice_info:
            recent = sorted(st["dt"] for st in steps[-20:])
            step_s = recent[len(recent) // 2]
            notice_s = a.notice_steps * step_s
            deadline_ms = int((time.time() + notice_s) * 1000)
            client.notice(a.kill_rank, deadline_ms)
            notice_info.update(
                t_notice=time.time(),
                deadline_ms=deadline_ms,
                step=step,
                notice_steps=a.notice_steps,
                step_s=step_s,
                notice_s=notice_s,
            )
    total = time.time() - t_run

    heard = []
    if client is not None:
        while True:
            try:
                h = client.committed.get_nowait()
            except Exception:  # noqa: BLE001 - queue.Empty
                break
            heard.append(
                {"epoch": h.committed.epoch, "active": list(h.committed.active_ranks), "at": h.at}
            )
    if probe and adapter:
        exact["batches"] = ex.collective_rpc(divergence_probe.batch_log)
        exact["scheduler"] = sched_log
    diag: dict = {}
    kv_digest: dict = {}
    if a.kv_digest and blocks:
        # every position written in the run, on every rank: rows below a position never change
        # once written, so the end-of-run digest of position p is the digest of the row the
        # step wrote there; a run's digests against its control's name the first position
        # whose row differs (summary.py), unless a preemption or a moved block makes the
        # positions incomparable, which the record says
        bs = core.vllm_config.cache_config.block_size
        hi = a.prompt_len + a.max_tokens - 1
        kv_digest = {
            "first_position": 0,
            "last_position": hi,
            "blocks": blocks,
            "blocks_at_kill": blocks_at_kill,
            "preempted": sorted({x for s in sched_log for x in s.get("preempted", [])}),
            "rows": ex.collective_rpc(
                divergence_probe.kv_rows_by_position,
                args=(
                    {
                        rid: [b[t // bs] * bs + t % bs for t in range(min(hi, len(b) * bs))]
                        for rid, b in blocks.items()
                    },
                ),
            ),
        }
    if a.diag:
        bs = core.vllm_config.cache_config.block_size
        lo = a.prompt_len - 8
        slots = {
            rid: [
                b[t // bs] * bs + t % bs
                for t in range(lo, min(len(b) * bs, a.prompt_len + a.max_tokens - 1))
            ]
            for rid, b in blocks.items()
        }
        diag = {
            "kv_first_position": lo,
            "blocks": blocks,
            "kv_rows": ex.collective_rpc(divergence_probe.kv_rows_by_position, args=(slots,)),
            "batches": exact.get("batches"),
            "scheduler": sched_log,
        }
    lp_out = {
        rid: [
            sorted([[int(t), float(lp.logprob)] for t, lp in d.items()], key=lambda x: -x[1])
            for d in seq
        ]
        for rid, seq in logprobs.items()
    }
    stats = ex.worker_stats() if adapter else None
    res = {
        "arm": a.arm,
        "kill": a.kill if a.arm == "replace" else None,
        "model": a.model,
        "load_s": load_s,
        "stamps": dict(STAMPS),
        "run_s": total,
        "steps": steps,
        "tokens": [tokens[str(i)] for i in range(a.n)],
        "times": [times[str(i)] for i in range(a.n)],
        "kill_info": kill_info,
        "notice": notice_info,
        "changes": changes,
        "driver_heard": heard,
        "executor_events": getattr(ex, "events", []),
        "started": getattr(ex, "started", None),
        "worker_stats": stats,
        # the executor's step RPCs still counted in flight once the run is idle: 0, change or not
        "steps_in_flight_at_end": getattr(ex, "_steps_out", None) if adapter else None,
        "async_scheduling": core.vllm_config.scheduler_config.async_scheduling,
        "cudagraph_mode": str(core.vllm_config.compilation_config.cudagraph_mode),
        "custom_all_reduce": not core.vllm_config.parallel_config.disable_custom_all_reduce,
        "max_tokens": a.max_tokens,
        "prompt_len": a.prompt_len,
        "enforce_eager": a.enforce_eager,
        "logprobs": [lp_out.get(str(i)) for i in range(a.n)] if a.logprobs else None,
        "diag": diag or None,
        "kv_digest": kv_digest or None,
        "exact": exact or None,
        "device_memory": device_memory,
        "shadow_memory": ex.collective_rpc(_shadow_memory) if adapter else None,
    }
    if a.kill == "external":  # the harness's kill, on another host: its time comes from a file
        marker = Path(a.out + ".kill")
        if marker.exists() and "t_kill" not in kill_info:
            kill_info.update(json.loads(marker.read_text()))
        kill_info.update(armed)
    sel = [s for s in steps if 10 <= s["step"] <= 200]
    tps = sum(s["new"] for s in sel) / max(sum(s["dt"] for s in sel), 1e-9)
    res["tok_s_steps_10_200"] = round(tps, 2)
    # the site's figure: each session's largest gap between two tokens at the client, and where
    t_kill = kill_info.get("t_kill") or notice_info.get("t_notice")
    t_resumed = changes[-1].get("t_resumed") if changes else None
    res["gaps"] = [session_gap(times[str(i)], t_kill, t_resumed) for i in range(a.n)]
    if changes and exact.get("batches"):
        exact["post_resume"] = post_resume(exact["batches"], t_resumed)
        # against the count each request resumed from (resume_point caps the committed count at
        # one below the token count, so the last position is always recomputed by design)
        exact["re_prefilled"] = re_prefilled(exact["post_resume"], exact.get("resume_from") or {})
    for c in changes:
        c["restore_coverage"] = restore_coverage(c)
    lost_events = [e for e in getattr(ex, "events", []) if e.get("kind") == "rank_lost"]
    res["detection_s"] = round(lost_events[0]["at"] - t_kill, 4) if lost_events and t_kill else None
    if a.clock_offsets:
        try:  # one probe per line (nodes.sh clocks appends them)
            res["clock_offsets"] = [
                json.loads(line)
                for line in Path(a.clock_offsets).read_text().splitlines()
                if line.strip()
            ]
        except (OSError, ValueError) as e:
            res["clock_offsets"] = {"error": str(e)}
    Path(a.out).write_text(json.dumps(res, default=str))
    print(
        json.dumps(
            {
                "arm": a.arm,
                "steps": len(steps),
                "tps_10_200": round(tps, 2),
                "changes": len(changes),
            }
        ),
        flush=True,
    )
    try:
        ex.shutdown()
    except Exception as e:  # noqa: BLE001 - the run is written; teardown errors are noise here
        print("shutdown:", e, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
