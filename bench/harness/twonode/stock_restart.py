# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The stock-restart baseline's summary: from node-b's loss to stock vLLM serving again.

Stock vLLM dies with a lost rank and keeps nothing: the harness relaunches the engine (both
halves) and runs the same sessions from their prompts. The figures, on node-a's clock with
node-b's kill moved by the probed offset where the probe answered:

- ``stock_restart_first_token_s``: the kill to the relaunched engine's first sampled token, the
  baseline the headline is against (decision 0030: recovery in seconds against minutes);
- ``stock_restart_engine_ready_s``: the relaunched engine's own load, warm-up and capture;
- ``stock_restart_relaunch_latency_s``: the kill to the relaunch, the supervisor's share (the
  harness's: the dead engine's exit, the control plane, ssh), stated so the engine's own time is
  read apart from it;
- the first token's time split at its stamps: ``stock_restart_detection_s``, the kill to the
  dead engine's exit (vLLM's worker RPC timeout on the survivor, 300 s at the pin);
  ``stock_restart_supervisor_s``, the exit to the relaunch; ``stock_restart_relaunch_to_ready_s``,
  the relaunch to the relaunched engine's first step (both halves launched, the checkpoint
  loaded, the engine warmed); ``stock_restart_ready_to_first_token_s``, that step, the sessions'
  prompts prefilled again. A trial recorded before the exit was stamped has no detection or
  supervisor figure;
- ``sessions_kept_identical``: ``0/N`` by construction, every request prefilled again.

    stock_restart.py --trial 1 --kill KILL.json [--exited T] --relaunched T --result RESTART.json
        [--clocks clocks.json] --predictions predictions.json --out summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import summary
from summary import figure


def offset_of(clocks: Path | None, node: str) -> tuple[float, dict]:
    """``node``'s clock against node-a's from the probes in ``clocks``, and how it was read."""
    if clocks is None or not clocks.exists():
        return 0.0, {"source": "no clocks file", "error_s": None}
    probes = []
    for line in clocks.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        p = rec.get("probe") or {}
        if rec.get("node") == node and "offset_s" in p:
            probes.append(rec)
    if not probes:
        return 0.0, {"source": "no probe answered: the offset is taken as zero", "error_s": None}
    last = probes[-1]["probe"]
    return float(last["offset_s"]), {
        "source": f"probe at {probes[-1].get('when')}",
        "error_s": last.get("error_s"),
        "rtt_s": last.get("rtt_s"),
    }


def summarize(a: argparse.Namespace) -> dict:
    """The trial's summary from the kill record, the relaunch stamp and the relaunched run."""
    kill = json.loads(Path(a.kill).read_text())
    r = json.loads(Path(a.result).read_text())
    offset, clock = offset_of(Path(a.clocks) if a.clocks else None, "node-b")
    t_kill = float(kill["t_kill"]) - offset  # node-b's stamp, on node-a's clock
    steps = r.get("steps") or []
    n = len(r.get("tokens") or [])
    out: dict = {
        "arm": "stock_restart",
        "trial": a.trial,
        "record": str(a.result),
        "model": r.get("model"),
        "sessions": n,
        "prompt_len": r.get("prompt_len"),
        "kill": kill,
        "clock": {"offset_s": offset, **clock},
        "figures": {},
    }
    f = out["figures"]
    if steps:
        first = steps[0]
        f["stock_restart_first_token_s"] = figure(
            "stock_restart_first_token_s",
            round(first["t_end"] - t_kill, 3),
            t_kill_node_a=t_kill,
            t_first_token=first["t_end"],
        )
        relaunched = float(a.relaunched)
        f["stock_restart_relaunch_latency_s"] = figure(
            "stock_restart_relaunch_latency_s", round(relaunched - t_kill, 3)
        )
        if a.exited is not None:
            exited = float(a.exited)
            f["stock_restart_detection_s"] = figure(
                "stock_restart_detection_s", round(exited - t_kill, 3), t_exit_node_a=exited
            )
            f["stock_restart_supervisor_s"] = figure(
                "stock_restart_supervisor_s", round(relaunched - exited, 3)
            )
        f["stock_restart_relaunch_to_ready_s"] = figure(
            "stock_restart_relaunch_to_ready_s",
            round(first["t_start"] - relaunched, 3),
            t_relaunch_node_a=relaunched,
        )
        f["stock_restart_ready_to_first_token_s"] = figure(
            "stock_restart_ready_to_first_token_s", round(first["t_end"] - first["t_start"], 3)
        )
        # the relaunch to the first step, at the relaunched drill's own stamps (node-a's clock):
        # the control plane, node-b's half and the launch before the process; its imports; the
        # engine's load; the time from ready to the first step
        stamps = r.get("stamps") or {}
        edges = [
            ("relaunch_to_process_s", relaunched, stamps.get("process")),
            ("imports_s", stamps.get("process"), stamps.get("vllm_imported")),
            ("to_engine_args_s", stamps.get("vllm_imported"), stamps.get("engine_args")),
            ("engine_load_s", stamps.get("engine_args"), stamps.get("engine_ready")),
            ("ready_to_first_step_s", stamps.get("engine_ready"), first["t_start"]),
        ]
        split = {k: round(b - a, 3) for k, a, b in edges if a is not None and b is not None}
        if split:
            f["stock_restart_relaunch_to_ready_s"]["parts"] = split
    if r.get("load_s") is not None:
        f["stock_restart_engine_ready_s"] = figure(
            "stock_restart_engine_ready_s", round(float(r["load_s"]), 3)
        )
    f["sessions_kept_identical"] = figure(
        "sessions_kept_identical",
        f"0/{n}",
        verdict="stock keeps no session across a lost rank: every request prefilled again",
        green=False,
    )
    f["requests_re_prefilled"] = figure("requests_re_prefilled", n)
    out["verdict"] = {"green": True, "line": "the baseline: measured, not scored"}
    return out


def main(argv: list[str]) -> int:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--kill", required=True, help="the harness's kill record (t_kill on node-b)")
    ap.add_argument("--exited", default=None, help="when the dead engine exited, node-a's clock")
    ap.add_argument("--relaunched", required=True, help="when the relaunch began, node-a's clock")
    ap.add_argument("--result", required=True, help="the relaunched run's record")
    ap.add_argument("--clocks", default=None)
    ap.add_argument(
        "--predictions", required=True, help="the predictions.json beside the pre-registration"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv[1:])
    summary.use_predictions(a.predictions)
    out = {**summarize(a), "preregistration": summary.PREREGISTRATION}
    Path(a.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(
        f"stock_restart trial {a.trial}: "
        + ", ".join(f"{k}={v['value']}" for k, v in out["figures"].items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
