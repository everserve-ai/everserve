# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Every pre-registered figure of the adapter-port run, per run, from the driver's result files.

    python bench/harness/summarize_adapter_port.py OUT_DIR results.json

File names: ``stock_<i>.json``, ``adapter_<i>.json``, ``replace_between_<i>.json``,
``replace_mid_<i>.json``, ``notice_<i>.json``. Definitions, fixed before the run:

* **throughput**: tokens per wall-clock second over steps 10-200 (``dt`` of ``engine.step``);
  overhead = 1 - adapter / mean(stock). Prediction 3.
* **pause**: the largest inter-token gap of any session. Prediction 1 (REPLACE), 6 (notice).
* **breakdown** (REPLACE): detect = executor's death detection - kill; change = daemon commit
  heard - detection - restore; restore = the spare's restore; resume = engine handed back -
  commit; first token = first token after resume - resume.
* **sessions kept**: every session reaches ``max_tokens``; none re-prefilled = every session's
  replay point is past its prompt. Prediction 2 with the token check: the 20 tokens after the
  fault equal the reference no-fault run's (``adapter_1``, same seed).
* **commit**: the daemon's ``MembershipCommitted`` heard by the executor within 100 ms of the
  engine resuming, and every survivor and the spare applied that epoch. Prediction 4.
* **restore check**: no layer's restored rows differ from the dead rank's (digests taken at the
  last committed step before the kill). Prediction 5.
* **no fault on notice**: no ``rank_lost`` in the executor's events. Prediction 6.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

AFTER = 20


METRICS = (
    "everserve_change_seconds",
    "everserve_changes_total",
    "everserve_store_commits_total",
    "everserve_store_bytes_in_use",
)


def parse_metrics(text: str) -> dict[str, Any]:
    """The run's named series from Prometheus text: samples by name and labels, histograms whole."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, _, value = line.rpartition(" ")
        name = name_labels.split("{", 1)[0]
        base = next((m for m in METRICS if name == m or name.startswith(m + "_")), None)
        if base is None:
            continue
        out.setdefault(base, {})[name_labels[len(base) :] or "value"] = float(value)
    return out


def metrics_for(directory: Path, name: str) -> dict[str, Any] | None:
    """Both daemons' series for one run, when they were captured."""
    got = {}
    for who in ("daemon", "store"):
        f = directory / f"{name}.{who}.metrics.txt"
        if f.exists():
            got[who] = f.read_text()
    if not got:
        return None
    return {k: v for text in got.values() for k, v in parse_metrics(text).items()}


def transport(r: dict[str, Any]) -> dict[str, Any] | None:
    """Per rank: the commit pipeline's figures and back-pressure per step (prediction 5)."""
    if not r.get("worker_stats"):
        return None
    out = {}
    for s in r["worker_stats"]:
        t = (s.get("commit") or {}).get("transport") or {}
        if not t:
            continue
        keep = (
            "steps",
            "backpressure_waits",
            "fence_waits_s",
            "bytes_sent",
            "max_in_flight",
            "messages",
        )
        row = {k: t.get(k) for k in keep}
        row["backpressure_per_step"] = t.get("backpressure_waits", 0) / max(t.get("steps", 0), 1)
        row["step_thread_p50_ms"] = (s.get("commit") or {}).get("p50_ms")
        spans = ((s.get("commit") or {}).get("timeline") or {}).get("spans") or {}
        for name in ("sender", "fence_wait", "event_lag"):  # sender: the store span, ready -> ack
            if name in spans:
                row[f"{name}_p50_ms"] = spans[name].get("p50_ms")
                row[f"{name}_p90_ms"] = spans[name].get("p90_ms")
        out[s["rank"]] = row
    return out


def tps(r: dict[str, Any]) -> float:
    """Tokens per second over steps 10-200."""
    sel = [s for s in r["steps"] if 10 <= s["step"] <= 200]
    return sum(s["new"] for s in sel) / max(sum(s["dt"] for s in sel), 1e-9)


def pause(r: dict[str, Any]) -> tuple[float, int]:
    """The largest inter-token gap over every session after the first 5 tokens, and its session."""
    best, who = 0.0, -1
    for i, ts in enumerate(r["times"]):
        for a, b in zip(ts[5:], ts[6:], strict=False):
            if b - a > best:
                best, who = b - a, i
    return best, who


def first_token_after(r: dict[str, Any], t: float) -> float | None:
    """When the last session got its first token after ``t``."""
    firsts = []
    for ts in r["times"]:
        later = [x for x in ts if x > t]
        if not later:
            return None
        firsts.append(later[0])
    return max(firsts)


def after_fault_match(r: dict[str, Any], ref: dict[str, Any], t: float) -> list[bool]:
    """Per session: the ``AFTER`` tokens emitted after ``t`` equal the reference's."""
    out = []
    for toks, ts, rtoks in zip(r["tokens"], r["times"], ref["tokens"], strict=True):
        k = sum(1 for x in ts if x <= t)
        out.append(toks[k : k + AFTER] == rtoks[k : k + AFTER] and len(toks) >= k + AFTER)
    return out


def reference_for(runs: dict[str, dict[str, Any]], r: dict[str, Any]) -> str | None:
    """The no-fault adapter run a fault run's tokens are compared with.

    The first by name of the same length and prompt: a REPLACE run of 201 tokens takes
    ``adapter_1``, a notice run of 601 takes ``notice_ref``, the arm's reference of notice length.
    None when there is none, and the run's token equality is then not tested rather than false.
    """
    for name, cand in sorted(runs.items()):
        if (
            cand["arm"] == "adapter"
            and not cand["changes"]
            and cand["max_tokens"] == r["max_tokens"]
            and cand["prompt_len"] == r["prompt_len"]
        ):
            return name
    return None


def determinism(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The control for every token-equality check: no-fault runs of one arm and length agree.

    Per group (arm, tokens), each run's sessions against the group's first run: the first token
    where they differ, or None. A fault run's mismatch counts against the adapter only where its
    group's controls all agree.
    """
    groups: dict[tuple[str, int, int], list[str]] = {}
    for name, r in sorted(runs.items()):
        if not r["changes"]:
            groups.setdefault((r["arm"], r["max_tokens"], r["prompt_len"]), []).append(name)
    out: dict[str, Any] = {}
    for (arm, n, prompt), names in groups.items():
        first = runs[names[0]]["tokens"]
        diverge = {
            name: [
                next((i for i, (p, q) in enumerate(zip(a, b, strict=False)) if p != q), None)
                for a, b in zip(runs[name]["tokens"], first, strict=True)
            ]
            for name in names[1:]
        }
        out[f"{arm}/{n}" + ("" if prompt == 2048 else f"/prompt{prompt}")] = {
            "against": names[0],
            "first_difference": diverge,
            "identical": all(i is None for d in diverge.values() for i in d) if diverge else None,
        }
    return out


def exactness(r: dict[str, Any], ref: dict[str, Any] | None) -> dict[str, Any] | None:
    """Decision 0013's three checks on a fault run, each with its pass.

    * ``kv_below_committed``: every rank's K and V rows at the committed positions are
      byte-identical before the change and after the first step back (the spare's against the
      rank it replaced).
    * ``first_step_back``: the scheduler's first steps after resuming give each request one token
      at a time from exactly its committed count, and every rank runs the first one on the FULL
      (decode) graph.
    * ``logits``: the run's top-20 logprobs equal the no-fault reference's for every token.
    """
    ex = r.get("exact")
    if not ex:
        return None
    out: dict[str, Any] = {}
    rows = ex.get("kv_below_committed")
    out["kv_below_committed"] = {
        "per_rank": rows,
        "pass": bool(rows) and all(not v["mismatched"] for v in rows.values()),
    }
    t_r = ex.get("t_resumed", 0)
    # the first step back starts where each request resumed (its committed count, at most
    # tokens - 1); runs before the resume point was recorded resumed at the committed count
    committed = ex.get("resume_from") or ex.get("committed", {})
    after = [s for s in ex.get("scheduler", []) if s["t"] > t_r]
    seq: dict[str, list[tuple[int, int]]] = {}
    for s in after[:4]:
        for rid, c in zip(s["cached"], s["cached_num_computed"], strict=False):
            seq.setdefault(rid, []).append((c, s["num_scheduled_tokens"].get(rid, 0)))
    replay_ok = bool(committed) and all(
        rid in seq
        and seq[rid][0] == (committed[rid], 1)
        and all(n == 1 for _, n in seq[rid])
        and all(b[0] == a[0] + 1 for a, b in zip(seq[rid], seq[rid][1:], strict=False))
        for rid in committed
    )
    re_added = [x for s in after[:4] for x in s.get("new", [])]
    firsts = {}
    for b in ex.get("batches") or []:
        first = next((x for x in b["batches"] if x["t"] > t_r), None)
        firsts[b["rank"]] = (
            {k: first.get(k) for k in ("cg_mode", "num_scheduled_tokens", "has_prefill")}
            if first
            else None
        )
    graph_ok = bool(firsts) and all(
        f is not None
        and str(f["cg_mode"]).endswith("FULL")
        and all(n == 1 for n in f["num_scheduled_tokens"])
        for f in firsts.values()
    )
    out["first_step_back"] = {
        "scheduler": {rid: v[:3] for rid, v in seq.items()},
        "committed": committed,
        "re_added": re_added,
        "per_rank": firsts,
        "pass": replay_ok and graph_ok and not re_added,
    }
    lp, lref = r.get("logprobs"), (ref or {}).get("logprobs")
    if lp and lref:
        diff = [
            next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
            for a, b in zip(lp, lref, strict=True)
        ]
        out["logits"] = {"first_difference": diff, "pass": all(d is None for d in diff)}
    else:
        out["logits"] = {"first_difference": None, "pass": None}
    restore = ((r["changes"][0].get("spare") or {}).get("restore") or {}) if r["changes"] else {}
    cu = restore.get("catch_up")
    replay = max((ex.get("replay_positions") or {}).values(), default=None)
    if cu is not None:
        before, after = cu["counts_before"], restore.get("counts", {})
        first = sum(before.values())
        delta = sum(after.get(k, 0) - v for k, v in before.items())
        # the catch-up copies rows for the positions committed after the watermark only: the same
        # rows per position as the full restore copied for everything before it
        rows_ok = (cu["rows"] == 0 and delta == 0) or (
            first > 0 and cu["rows"] * first == restore.get("rows", 0) * delta
        )
        # the ordering rule: the catch-up (synchronize included) ended before any rank's first
        # step back began
        firsts = [
            next((x["t"] for x in b["batches"] if x["t"] > ex.get("t_resumed", 0)), None)
            for b in ex.get("batches") or []
        ]
        firsts = [t for t in firsts if t is not None]
        ordered = cu.get("t_end") is not None and bool(firsts) and cu["t_end"] < min(firsts)
        out["catch_up"] = {
            "ended_before_first_step_back": ordered,
            "rows": cu["rows"],
            "positions_after_watermark": delta,
            "rows_first_restore": restore.get("rows"),
            "positions_first_restore": first,
            "replay_positions_max": replay,
            "s": cu.get("s"),
            "pass": rows_ok
            and ordered
            and replay is not None
            and replay <= (2 if r["arm"] == "notice" else 3),
        }
    # tokens and committed rows carry the exactness claim; logits are reported beside them
    # (decision 0019: logprobs are not bit-reproducible between runs on H100)
    out["pass"] = all(
        v.get("pass") for k, v in out.items() if isinstance(v, dict) and k != "logits"
    )
    return out


def change_figures(r: dict[str, Any], ref: dict[str, Any] | None) -> dict[str, Any]:  # noqa: PLR0914
    """Figures of a run with a membership change."""
    out: dict[str, Any] = {}
    ch = r["changes"][0] if r["changes"] else None
    if ch is None:
        return {"error": "no change completed"}
    kill = r["kill_info"].get("t_kill")
    lost = [e for e in r["executor_events"] if e.get("kind") == "rank_lost"]
    restore = (ch.get("spare") or {}).get("restore") or {}
    restore_s = restore.get("restore_s")
    t_c, t_r = ch["t_committed"], ch["t_resumed"]
    ft = first_token_after(r, t_r)
    out["rank_lost_events"] = len(lost)
    if lost and kill:
        t_d = lost[0]["at"]
        out["detect_s"] = t_d - kill
        out["detected_by"] = lost[0].get("source", "sentinel")
        out["change_s"] = t_c - t_d - (restore_s or 0.0)
    t_exit = r["kill_info"].get("t_exit")
    if t_exit and kill:  # the dead process's own exit, apart from detection (decision 0021)
        out["exit_s"] = t_exit - kill
    sentinel = [e for e in r["executor_events"] if e.get("kind") == "exited"]
    if sentinel and kill:
        out["sentinel_s"] = sentinel[0]["at"] - kill
    out["detect_bound_s"] = ((r.get("started") or {}).get("detect") or {}).get("bound_s")
    out["restore_s"] = restore_s
    out["resume_s"] = t_r - t_c
    out["first_token_s"] = (ft - t_r) if ft else None
    out["commit_to_resume_ms"] = 1e3 * (t_r - t_c)
    applied = [e for e in ch["epochs_applied"].values()] + [ch.get("spare_epoch")]
    out["epoch_committed"] = ch["epoch_committed"]
    out["epochs_applied"] = applied
    out["epoch_match"] = all(e == ch["epoch_committed"] for e in applied)
    out["restore_check"] = ch.get("restore_check")
    safe = ch.get("safe_tokens", {})
    out["replay_from"] = safe
    out["re_prefilled"] = [rid for rid, n in safe.items() if n < r["prompt_len"]]
    out["n_replayed"] = ch.get("n_replayed")
    out["sessions_kept"] = sum(1 for t in r["tokens"] if len(t) >= r["max_tokens"])
    t_fault = kill if kill else t_c
    if ref is not None:
        m = after_fault_match(r, ref, min(t_fault, t_c))
        out["after_fault_match"] = m
    out["exactness"] = exactness(r, ref)
    return out


def summarize(directory: Path) -> dict[str, Any]:
    """Every run's figures and the per-arm aggregates."""
    runs = {
        p.stem: json.loads(p.read_text())
        for p in sorted(directory.glob("*.json"))
        if p.stem != "results"
    }
    stock = [tps(r) for k, r in runs.items() if k.startswith("stock_")]
    base = statistics.mean(stock) if stock else None
    per_run: dict[str, Any] = {}
    for name, r in runs.items():
        fig: dict[str, Any] = {"arm": r["arm"], "kill": r.get("kill")}
        fig["tps_10_200"] = tps(r)
        if base:
            fig["overhead_pct"] = 100 * (1 - fig["tps_10_200"] / base)
        p, who = pause(r)
        fig["pause_s"], fig["pause_session"] = p, who
        fig["sessions"] = sum(1 for t in r["tokens"] if len(t) >= r["max_tokens"])
        if r.get("shadow_memory"):
            fig["shadow_memory"] = {m["rank"]: m for m in r["shadow_memory"] if m}
        if r["worker_stats"]:
            fig["commit_ms"] = {s["rank"]: s["commit"] for s in r["worker_stats"]}
            # the kernels each worker ran: the slot variant and the gather path
            fig["paths"] = {s["rank"]: s.get("paths") for s in r["worker_stats"]}
            # decision 0021: every report the slot watch made, and every late member found alive
            fig["slot_watch"] = {s["rank"]: s.get("slot_watch") for s in r["worker_stats"]}
        fig["rank_lost_events"] = sum(
            1 for e in r.get("executor_events") or [] if e.get("kind") == "rank_lost"
        )
        fig["detect"] = (r.get("started") or {}).get("detect")
        if r["changes"]:
            ref = reference_for(runs, r)
            fig["reference"] = ref
            fig.update(change_figures(r, runs[ref] if ref else None))
        if r.get("kill_info", {}).get("snapshot"):
            fig["snapshot"] = r["kill_info"]["snapshot"]
        fig["transport"] = transport(r)
        fig["custom_all_reduce"] = r.get("custom_all_reduce")
        # the slot collective's variant and how the engine chose it (override, nvml, cache, probe)
        fig["slot_choice"] = (r.get("started") or {}).get("slot")
        if r["changes"] and r["worker_stats"]:
            # what each worker's first steps after the change did (everserve_vllm.firststep)
            fig["first_steps"] = {s["rank"]: s.get("first_steps") for s in r["worker_stats"]}
        if r["changes"]:
            fig["releases"] = {
                rk: b.get("released") for rk, b in r["changes"][0]["boundaries"].items()
            }
        m = metrics_for(directory, name)
        if m is not None:
            fig["metrics"] = m
            sent = [(s.get("commit") or {}).get("transport", {}) for s in r["worker_stats"] or []]
            steps = sum(t.get("steps", 0) for t in sent)
            if steps and all("messages" in t for t in sent):
                # decision 0019: one commit message per rank per step, as the transport counted
                fig["commit_messages_per_step"] = round(sum(t["messages"] for t in sent) / steps, 3)
            cs = m.get("everserve_change_seconds", {})
            n = sum(v for k, v in cs.items() if k.startswith("_count"))
            if n:
                # the daemon's span: loss report to every acknowledgement (driver: kill to token)
                total = sum(v for k, v in cs.items() if k.startswith("_sum"))
                fig["daemon_change_s"] = total / n
        per_run[name] = fig
    return {
        "determinism": determinism(runs),
        "stock_tps_mean": base,
        "reconcile_on_recover": {
            "value": False,
            "source": "adapters/vllm/everserve_vllm/worker.py, reconcile_on_recover=False",
        },
        "runs": per_run,
    }


def main() -> None:
    """Write ``results.json``."""
    out = summarize(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str))
    for name, f in out["runs"].items():
        print(
            name,
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in f.items()
                if k
                in (
                    "tps_10_200",
                    "overhead_pct",
                    "pause_s",
                    "sessions",
                    "detect_s",
                    "detected_by",
                    "exit_s",
                    "sentinel_s",
                    "detect_bound_s",
                    "change_s",
                    "restore_s",
                    "resume_s",
                    "first_token_s",
                    "epoch_match",
                    "restore_check",
                    "re_prefilled",
                    "rank_lost_events",
                )
            },
        )


if __name__ == "__main__":
    main()
