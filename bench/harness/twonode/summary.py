# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""One file per trial: the record's figures, the verdict, the clocks, each beside its prediction.

The site and the pre-registration read this file and nothing else; nothing is computed from
logs. Every number the run's pre-registration scores is a field here with the prediction's
number next to it, from the ``predictions.json`` beside that pre-registration (``--predictions``),
so a reader sees the figure, the prediction and the kill condition on one line, and the summary
names the pre-registration it was scored against.

    summary.py --arm node_fault --trial 2 --result r.json --predictions predictions.json
        [--control c.json] [--stock s.json] [--store-metrics m.json] [--engine-gpus N]
        --out summary.json

``derived`` holds what a whole-replica restore would have discarded at the trial's fault and
stock's time to redo it: reported, never scored, apart from ``figures``.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from accept import agreement, verdict

PREDICTIONS: dict[str, dict] = {}
"""The pre-registration's prediction for each figure, by name: ``n`` (its number in the text),
``predicted`` and ``kill``. Loaded from the ``predictions.json`` beside the pre-registration the
battery runs under (:func:`use_predictions`); a figure the file does not name is reported with
none. Numbers held in this module would belong to one pre-registration and be reported for every
other; the file is the only source."""

PREREGISTRATION: str | None = None
"""The pre-registration the loaded predictions come from, recorded in every summary."""


def use_predictions(path: str | Path) -> None:
    """Score against the predictions file at ``path`` from here on.

    Raises:
        ValueError: a file without ``preregistration`` or ``figures``.
    """
    global PREREGISTRATION
    d = json.loads(Path(path).read_text())
    if not d.get("preregistration") or not isinstance(d.get("figures"), dict):
        raise ValueError(f"{path}: a predictions file names its preregistration and figures")
    PREDICTIONS.clear()
    PREDICTIONS.update(d["figures"])
    PREREGISTRATION = d["preregistration"]


# what a whole-replica restore (a snapshot of the engine, reloaded on a fault) would have discarded
# at the same fault, derived from the record and reported under "derived", never scored: every
# session in the engine at the kill, its prompt and the tokens it had committed, and the time
# stock would take to compute them again at its own rates on the same nodes (the stock arm's first
# trial). Where the rates are approximate they understate that time: linear prefill at longer
# prefixes, decode at steps 10-200 for later positions, stock's batch step for a lone long session.
# Neither time includes the replica's own restore, which is not measured.
DERIVED = (
    "replica_lost_sessions",
    "replica_lost_prompt_tokens",
    "replica_lost_generated_tokens",
    "replica_redo_retry_s",
    "replica_redo_continuation_s",
    "replica_restore_gpus",
    "take_gpus",
)

ARM_GAP_FIELD = {
    "node_drain": "drain_gap_s",
    "node_fault": "node_fault_gap_s",
    "single_death": "single_death_gap_s",
    "long_context": "long_context_gap_s",
}
STOCK_ARMS = ("stock", "stock_restart")
"""The arms that serve under stock vLLM."""
STOCK_NODES = ("a", "b")
"""The nodes whose processes are stock vLLM's under those arms."""

ARM_DETECTION_FIELD = {
    "node_fault": "node_fault_detection_s",
    "single_death": "single_death_detection_s",
}


def figure(name: str, value, **extra) -> dict:
    """One scored figure: the value, the prediction's number and kill beside it."""
    p = PREDICTIONS.get(name, {})
    out = {"value": value, "prediction": p.get("n"), "predicted": p.get("predicted")}
    out["kill"] = p.get("kill")
    out.update(extra)
    return out


def stock_rates(stock: dict, path: str) -> dict | None:
    """Stock's prefill rate and per-session decode step from its record, or None.

    The prefill: every prompt over the time from the first step's start to the last session's
    first token (one step when the batch budget holds every prompt, several when chunked). The
    decode: the mean step over steps 10 to 200, each step one token per session.
    """
    steps, times = stock.get("steps") or [], stock.get("times") or []
    firsts = [ts[0] for ts in times if ts]
    sel = [s for s in steps if 10 <= s["step"] <= 200]
    if not steps or not firsts or len(firsts) != len(times) or not sel:
        return None
    prefill_s = max(firsts) - steps[0]["t_start"]
    prompt_tokens = len(times) * stock["prompt_len"]
    return {
        "stock_prefill_tok_s": round(prompt_tokens / prefill_s, 1),
        "stock_prefill_s": round(prefill_s, 4),
        "stock_prefill_tokens": prompt_tokens,
        "stock_decode_step_s": round(sum(s["dt"] for s in sel) / len(sel), 6),
        "stock_record": path,
    }


def replica_loss(r: dict, rates: dict | None, engine_gpus: int | None) -> dict:
    """What a whole-replica restore discards at this trial's fault, and stock's time to redo it.

    The sessions in the engine at the kill are those not finished by then; each one's generated
    count is its committed count (the change record's ``safe_tokens``) less its prompt, beside the
    tokens the client had received before the kill, an independent read of the same count.
    """
    c = (r.get("changes") or [{}])[-1]
    t_kill = (r.get("kill_info") or {}).get("t_kill")
    safe = c.get("safe_tokens") or {}
    if not t_kill or not safe:
        return {}
    prompt_len = r["prompt_len"]
    generated = {rid: n - prompt_len for rid, n in sorted(safe.items())}
    live = {rid: g for rid, g in generated.items() if g < r["max_tokens"]}
    had = [sum(1 for t in ts if t < t_kill) for ts in r.get("times") or []]
    prompt_tokens = len(live) * prompt_len
    gen_tokens = sum(live.values())
    out = {
        "replica_lost_sessions": {"value": len(live)},
        "replica_lost_prompt_tokens": {"value": prompt_tokens},
        "replica_lost_generated_tokens": {
            "value": gen_tokens,
            "per_session": live,
            "client_had_at_kill": had,
        },
        "take_gpus": {"value": len(c.get("ranks") or [c.get("rank")])},
    }
    if engine_gpus:
        out["replica_restore_gpus"] = {"value": engine_gpus}
    if rates and live:
        bound = "derived; understated where approximate; the replica's restore not included"
        out["replica_redo_retry_s"] = {
            "value": round(
                prompt_tokens / rates["stock_prefill_tok_s"]
                + max(live.values()) * rates["stock_decode_step_s"],
                3,
            ),
            "how": "the prompts prefilled again, then the longest session's tokens decoded again",
            "bound": bound,
            **rates,
        }
        out["replica_redo_continuation_s"] = {
            "value": round((prompt_tokens + gen_tokens) / rates["stock_prefill_tok_s"], 3),
            "how": "every prompt with its generated tokens prefilled again, nothing decoded again",
            "bound": bound,
            **rates,
        }
    return {k: out[k] for k in DERIVED if k in out}


def first_differing_positions(mine: dict, theirs: dict) -> dict:
    """A run's per-position K/V digests against its control's: the first differing position.

    Per request and rank, and the worst (lowest) over them; None when every position matches.
    Refused (``comparable`` False, ``worst`` None, the reason named) when either run preempted a
    request or the run's block table at the kill is not a prefix of its table at the end: a
    moved block makes the positions mean different rows.
    """
    out: dict = {"comparable": True, "reason": None, "per_request": {}, "worst": None}
    pre = sorted(set(mine.get("preempted") or []) | set(theirs.get("preempted") or []))
    if pre:
        out.update(comparable=False, reason=f"preempted: {pre}")
        return out
    for rid, at_kill in (mine.get("blocks_at_kill") or {}).items():
        end = (mine.get("blocks") or {}).get(rid) or []
        if end[: len(at_kill)] != at_kill:
            out.update(comparable=False, reason=f"{rid}: blocks moved after the kill")
            return out
    # request ids carry a per-run suffix ("4-ad4a42cb"); the session index before it is the
    # same prompt in both runs
    session = lambda rid: str(rid).split("-")[0]  # noqa: E731 - a one-line key
    rows_mine = {str(x["rank"]): x["rows"] for x in mine.get("rows") or []}
    rows_theirs = {
        str(x["rank"]): {session(rid): d for rid, d in x["rows"].items()}
        for x in theirs.get("rows") or []
    }
    firsts = []
    for rank, by_req in rows_mine.items():
        for rid, digests in by_req.items():
            ref = (rows_theirs.get(rank) or {}).get(session(rid))
            if ref is None:
                continue
            n = min(len(digests), len(ref))
            first = next((p for p in range(n) if digests[p] != ref[p]), None)
            out["per_request"][f"{rank}:{session(rid)}"] = first
            if first is not None:
                firsts.append(first)
    out["worst"] = min(firsts) if firsts else None
    return out


def kill_in_step(steps: list[dict], t_kill: float | None) -> dict | None:
    """Where in node-a's step the kill landed: the step, the seconds into it, the fraction.

    A step the kill aborted is not among the recorded steps, so a kill that lands after the last
    completed step reads the time since that step ended against its length: how far into the
    next, aborted, step it fell (``fraction`` over 1: past a step's length). The kill's stamp is
    node-b's clock, read against node-a's steps as is: the probed offsets between the nodes are
    under 0.06 ms. A kill that loses the engine leaves its place in the step unrecorded, and
    this returns None for it.
    """
    if t_kill is None or not steps:
        return None
    prev = None
    for s in steps:
        if s["t_start"] <= t_kill <= s["t_end"]:
            span = s["t_end"] - s["t_start"]
            into = t_kill - s["t_start"]
            return {
                "step": s["step"],
                "into_s": round(into, 4),
                "of_s": round(span, 4),
                "fraction": round(into / span, 3) if span else None,
            }
        if t_kill < s["t_start"]:
            break
        prev = s
    if prev is None:
        return None
    span = prev["t_end"] - prev["t_start"]
    since = t_kill - prev["t_end"]
    return {
        "after_step": prev["step"],
        "since_end_s": round(since, 4),
        "of_s": round(span, 4),
        "fraction": round(since / span, 3) if span else None,
    }


def load(path: str | None) -> dict | None:
    """The JSON at ``path``, or None for no path, no file or a file that does not parse."""
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except ValueError:
        return None


def engine_status(path: str) -> dict | None:
    """An engine's status, from a saved read of ``GET /v1/engines/<e>`` (decision 0042)."""
    engine = load(path)
    return engine.get("status") if isinstance(engine, dict) else None


def member_pair(readiness: dict | None, rank: int) -> tuple[str, str] | None:
    """The (key, digest) ``rank``'s registered process reported, as a readiness read shows it."""
    for m in (readiness or {}).get("members") or []:
        if m.get("rank") == rank and m.get("compileCache"):
            return m["compileCache"].get("key"), m["compileCache"].get("digest")
    return None


def compile_cache(
    before: dict | None,
    committed: dict | None,
    ranks: list[int],
    written: dict | None,
    position: dict | None = None,
    spare: dict | None = None,
) -> dict | None:
    """Whether the compile cache held over a trial (decision 0029), the failing check named.

    Three checks, each over what the record holds:

    - the readiness read before the trial says the engine's caches agree (``before``);
    - each taken position's pair as the taking spare registered it (``committed``, read once the
      change committed) equals the position's before the trial: the per-rank pair the coordinator
      enforces (``PRIVATE-INTERNALS.md`` A40), which a node-local build helper is no part of;
    - the take's start wrote no autotune result (``written``'s ``autotune``).

    The whole-node digests of the position's node and the spare's after staging (``position``,
    ``spare``, ``cache_digest.py``) are reported beside it, not scored: they differ by the Triton
    helper ``cache_sync`` builds in place, and scored as check (b) they would kill the prediction
    while the coordinator's comparison holds.
    None when no check can be read.
    """
    checks: dict[str, bool | None] = {
        "readiness_consistent": (
            before.get("compileCacheConsistent") if before is not None else None
        ),
        "spare_holds_position_s": None,
        "no_autotune_written": None,
    }
    pairs = {r: (member_pair(before, r), member_pair(committed, r)) for r in ranks}
    if ranks and all(a is not None and b is not None for a, b in pairs.values()):
        checks["spare_holds_position_s"] = all(a == b for a, b in pairs.values())
    autotune = (written or {}).get("autotune")
    if autotune is not None:
        checks["no_autotune_written"] = not autotune
    read = {k: v for k, v in checks.items() if v is not None}
    if not read:
        return None
    staging = None
    if position is not None and spare is not None:
        node_pairs = lambda d: sorted((c["key"], c["digest"]) for c in d.get("caches") or [])  # noqa: E731
        staging = node_pairs(position) == node_pairs(spare)
    return figure(
        "compile_cache_consistent",
        all(read.values()),
        checks=checks,
        failing=[k for k, v in read.items() if not v],
        pairs={str(r): {"position": a, "spare": b} for r, (a, b) in pairs.items()},
        autotune_files=autotune or [],
        staging_digests_equal=staging,
    )


def engine_down(before: dict | None, during: dict | None, after: dict | None) -> dict | None:
    """Whether the engine went down over a trial (decision 0032), each down read named.

    Down before the trial or while it served counts whatever its cause; down after it counts
    unless its cause is ``stopped``, the head's orderly goodbye at the drill's end. None when no
    read was taken.
    """
    reads = {"before": before, "during": during, "after": after}
    downs = {k: r["down"] for k, r in reads.items() if r is not None and r.get("down")}
    counted = {k: d for k, d in downs.items() if k != "after" or d.get("cause") != "stopped"}
    if all(r is None for r in reads.values()):
        return None
    return figure("engine_down", bool(counted), downs=downs, counted=sorted(counted))


def summarize(a: argparse.Namespace) -> dict:
    """The trial's summary from its record and the others it is scored against."""
    r = json.loads(Path(a.result).read_text())
    control = json.loads(Path(a.control).read_text()) if a.control else None
    stock = json.loads(Path(a.stock).read_text()) if a.stock else None
    metrics = json.loads(Path(a.store_metrics).read_text()) if a.store_metrics else {}
    backup = None
    if getattr(a, "backup", None) and Path(a.backup).exists():
        try:
            backup = json.loads(Path(a.backup).read_text())
        except ValueError:
            backup = {"error": "unreadable"}
    written = None
    if getattr(a, "written", None) and Path(a.written).exists():
        try:
            written = json.loads(Path(a.written).read_text())
        except ValueError:
            written = None
    # the take's index on its node (1: the node's first take, cold; the reads of every start say
    # whether a later one is): each take figure carries it, so a summary over trials can split them
    take_index = (written or {}).get("takes")
    take_tags = {"take_index": take_index, "cold": take_index == 1} if take_index else {}
    out: dict = {
        "arm": a.arm,
        "trial": a.trial,
        "record": str(a.result),
        "model": r.get("model"),
        "sessions": len(r.get("tokens") or []),
        "prompt_len": r.get("prompt_len"),
        "tok_s_steps_10_200": r.get("tok_s_steps_10_200"),
        "clock_offsets": r.get("clock_offsets"),
        "figures": {},
    }
    f = out["figures"]
    tps = r.get("tok_s_steps_10_200")
    # the overhead is the adapter's own arms against stock on the same sessions and prompts
    if a.arm in ("adapter", "mirror_off") and stock and tps and stock.get("tok_s_steps_10_200"):
        f["adapter_overhead_pct"] = figure(
            "adapter_overhead_pct",
            round(100.0 * (1 - tps / stock["tok_s_steps_10_200"]), 2),
            stock_tok_s=stock["tok_s_steps_10_200"],
        )
    gaps = r.get("gaps") or []
    change_gaps = [g.get("around_change_s") for g in gaps if g.get("around_change_s") is not None]
    if a.arm in ARM_GAP_FIELD and change_gaps:
        f[ARM_GAP_FIELD[a.arm]] = figure(
            ARM_GAP_FIELD[a.arm],
            round(max(change_gaps), 4),
            per_session=change_gaps,
            median_over_sessions=round(statistics.median(change_gaps), 4),
        )
    if a.arm in ARM_DETECTION_FIELD and r.get("detection_s") is not None:
        f[ARM_DETECTION_FIELD[a.arm]] = figure(ARM_DETECTION_FIELD[a.arm], r["detection_s"])
    if control is not None and a.arm == "repeat":
        green, line = agreement(r, control)
        mine, theirs = r.get("tokens") or [], control.get("tokens") or []
        f["runs_agree"] = figure(
            "runs_agree",
            f"{sum(1 for x, y in zip(mine, theirs, strict=False) if x == y)}/{len(theirs)}",
            verdict=line,
            green=green,
        )
        out["verdict"] = {"green": green, "line": line}
    elif control is not None:
        green, line = verdict(r, control)
        mine, theirs = r.get("tokens") or [], control.get("tokens") or []
        f["sessions_kept_identical"] = figure(
            "sessions_kept_identical",
            f"{sum(1 for x, y in zip(mine, theirs, strict=False) if x == y)}/{len(theirs)}",
            verdict=line,
            green=green,
        )
        out["verdict"] = {"green": green, "line": line}
    changes = r.get("changes") or []
    if changes:
        c = changes[-1]
        mism = 0
        rc = c.get("restore_check") or {}
        for v in rc.values() if isinstance(rc, dict) else []:
            mism += len((v or {}).get("mismatched") or []) if isinstance(v, dict) else 0
        f["restored_rows_mismatched"] = figure("restored_rows_mismatched", mism)
        ratios = [v["ratio"] for v in (c.get("restore_coverage") or {}).values() if v.get("ratio")]
        if ratios:  # the restore covered every committed position only at 1
            f["restore_coverage"] = figure("restore_coverage", min(ratios))
        rp = (r.get("exact") or {}).get("re_prefilled")
        if rp is not None:  # an exact read means what it says only at 0
            f["requests_re_prefilled"] = figure("requests_re_prefilled", len(rp), requests=rp)
        causes = (r.get("exact") or {}).get("resume_cause")
        if causes:  # every request at its committed count, or one below when the last token died
            below = sorted(k for k, v in causes.items() if v.get("cause") == "below_committed")
            f["resume_below_committed"] = figure(
                "resume_below_committed",
                len(below),
                requests=below,
                causes={k: v.get("cause") for k, v in causes.items()},
            )
        if control is not None and r.get("kv_digest") and control.get("kv_digest"):
            diff = first_differing_positions(r["kv_digest"], control["kv_digest"])
            f["first_differing_position"] = figure(
                "first_differing_position",
                diff["worst"],
                comparable=diff["comparable"],
                reason=diff["reason"],
                per_request=diff["per_request"],
            )
        checked = (r.get("exact") or {}).get("rows_checked")
        if checked:  # the drill's own digests: where they ran and how long they stalled the engine
            f["instrument_pause_s"] = figure(
                "instrument_pause_s", checked.get("s"), step=checked.get("step")
            )
        out["change"] = {
            k: c.get(k)
            for k in (
                "ranks",
                "faulted",
                "epoch_committed",
                "wait_commit_s",
                "boundary_s",
                "resume_s",
                "n_replayed",
                "claim_to_commit",
                "join_legs_s",
                "spares",
                "boundaries",
            )
        }
        for rank, sp in sorted((c.get("spares") or {}).items(), key=lambda kv: str(kv[0])):
            att = (sp or {}).get("attachment") or {}
            staged = ((sp or {}).get("restore") or {}).get("staged") or {}
            terms = {
                "take_s": att.get("take_s"),
                "take_weights_s": att.get("weights_s"),
                "take_modules_s": (
                    round(att["shard_copy_s"] - att.get("weights_s", 0.0), 3)
                    if att.get("shard_copy_s") is not None
                    else None
                ),
                "take_kv_config_s": att.get("kv_config_s"),
                "take_kv_alloc_s": att.get("kv_alloc_s"),
                "take_sizing_s": att.get("sizing_s"),
                "take_warm_s": att.get("warm_s"),
                "take_identity_s": att.get("identity_s"),
                "take_restore_s": ((sp or {}).get("restore") or {}).get("restore_s"),
                "staged_wait_s": staged.get("s"),
            }
            for name, value in terms.items():
                if value is not None:
                    f[name] = figure(name, value, rank=rank, **take_tags)
            if "take_kv_config_s" in f and att.get("kv_config_parts"):
                f["take_kv_config_s"]["parts"] = att["kv_config_parts"]
            if "take_warm_s" in f and att.get("warm_parts"):
                f["take_warm_s"]["parts"] = att["warm_parts"]
        out["change"]["replaced_after_kill_s"] = next(
            (
                round(e["at"] - r["kill_info"]["t_kill"], 4)
                for e in r.get("executor_events") or []
                if e.get("kind") == "replaced" and (r.get("kill_info") or {}).get("t_kill")
            ),
            None,
        )
    where = kill_in_step(r.get("steps") or [], (r.get("kill_info") or {}).get("t_kill"))
    if where is not None:
        out.setdefault("change", {})["kill_in_step"] = where
    if metrics:
        if "replica_lag_max_steps" in metrics:
            f["replica_lag_max_steps"] = figure(
                "replica_lag_max_steps", metrics["replica_lag_max_steps"]
            )
        if metrics.get("refusals_total") is not None:
            refused = metrics["refusals_total"]
            f["store_refusals_total"] = figure(
                "store_refusals_total",
                sum(refused.values()),
                by_reason=refused,
                stores_read=metrics.get("stores_read"),
            )
        if "replication_dropped_total" in metrics:
            f["replication_dropped_total"] = figure(
                "replication_dropped_total", metrics["replication_dropped_total"]
            )
    if written is not None:
        f["take_written_bytes"] = figure(
            "take_written_bytes",
            written.get("bytes"),
            files=written.get("files"),
            start=written.get("start"),
            roots=written.get("roots"),
            dirs=(written.get("dirs") or [])[:10],
            **take_tags,
        )
    stopped = {}
    for path in getattr(a, "stopped", None) or []:
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        stopped[Path(path).stem.rsplit("_", 1)[-1]] = d.get("exit")
    if stopped:
        # under the stock arms the serving nodes run stock vLLM's own processes, whose stop is
        # stock's to answer for: reported apart, so the scored figure counts ours alone (the
        # spare's node runs ours in every arm)
        theirs = STOCK_NODES if a.arm in STOCK_ARMS else ()
        ours = {n: e for n, e in stopped.items() if n not in theirs}
        stock = {n: e for n, e in stopped.items() if n in theirs}
        killed = lambda d: sum(1 for e in d.values() if e == "killed")  # noqa: E731
        f["stops_killed"] = figure("stops_killed", killed(ours), by_node=ours)
        if stock:
            f["stock_stops_killed"] = figure("stock_stops_killed", killed(stock), by_node=stock)
    baseline = None
    if written is not None and getattr(a, "baseline", None) and Path(a.baseline).exists():
        try:
            baseline = json.loads(Path(a.baseline).read_text())
        except ValueError:
            baseline = None
    if baseline is not None:
        idle_roots = set(baseline.get("roots") or {})
        beyond = {r: v for r, v in (written.get("roots") or {}).items() if r not in idle_roots}
        f["take_written_beyond_baseline_bytes"] = figure(
            "take_written_beyond_baseline_bytes",
            sum(v.get("bytes", 0) for v in beyond.values()),
            files=sum(v.get("files", 0) for v in beyond.values()),
            roots=beyond,
            baseline_roots=sorted(idle_roots),
            **take_tags,
        )
    if backup is not None:
        per = {k: (v or {}).get("coverage") for k, v in backup.items() if k != "error"}
        vals = [v for v in per.values() if v is not None]
        f["backup_coverage"] = figure(
            "backup_coverage",
            min(vals) if len(vals) == len(per) and vals else None,
            per_rank=per,
            detail=backup,
        )
    taken = sorted(int(k) for k in ((r.get("changes") or [{}])[-1].get("spares") or {}))
    cc = compile_cache(
        engine_status(getattr(a, "readiness_before", "")),
        engine_status(getattr(a, "readiness_committed", "")),
        taken,
        written,
        load(getattr(a, "cache_position", "")),
        load(getattr(a, "cache_spare", "")),
    )
    if cc is not None:
        f["compile_cache_consistent"] = cc
    down = engine_down(
        engine_status(getattr(a, "readiness_before", "")),
        engine_status(getattr(a, "readiness_committed", "")),
        engine_status(getattr(a, "readiness_after", "")),
    )
    if down is not None:
        f["engine_down"] = down
    started = r.get("started") or {}
    formed = [t for w in started.get("workers") or [] for t in (w.get("formed") or {}).values()]
    if formed:
        f["shadow_formation_s"] = figure("shadow_formation_s", round(max(formed), 3))
    # the coordinator's record of the trial's change (its stamps, each acknowledgement and each
    # join payload in and out, on its clock but the acknowledgements'), as readiness kept it
    after = engine_status(getattr(a, "readiness_after", ""))
    last = (after or {}).get("lastChange")
    if last is not None and r.get("changes"):
        out.setdefault("change", {})["control_plane"] = last
    if r.get("changes") and (r.get("kill_info") or {}).get("t_kill"):
        rates = stock_rates(stock, a.stock) if stock else None
        derived = replica_loss(r, rates, getattr(a, "engine_gpus", None))
        if derived:
            out["derived"] = derived
    shadow = r.get("shadow_memory") or []
    gbs = [m.get("used_gb") for m in shadow if isinstance(m, dict) and m.get("used_gb")]
    if gbs:
        f["shadow_memory_gb_max"] = figure("shadow_memory_gb_max", max(gbs))
    return out


def main(argv: list[str]) -> int:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--control", default="")
    ap.add_argument("--stock", default="")
    ap.add_argument("--store-metrics", default="")
    ap.add_argument("--backup", default="", help="backup_coverage.json from the battery's check")
    ap.add_argument(
        "--written", default="", help="the take's start's written_since read on the spare's node"
    )
    ap.add_argument(
        "--baseline", default="", help="the spare's node's written_since read with no start"
    )
    ap.add_argument(
        "--stopped", nargs="*", default=[], help="each node's written_since read after the stop"
    )
    ap.add_argument(
        "--engine-gpus", type=int, help="TP x PP: the GPUs a whole-replica restore reloads"
    )
    ap.add_argument("--readiness-before", default="", help="the readiness read before the trial")
    ap.add_argument(
        "--readiness-committed", default="", help="the readiness read once the change committed"
    )
    ap.add_argument("--readiness-after", default="", help="the readiness read after the trial")
    ap.add_argument("--cache-position", default="", help="cache_digest.py on the position's node")
    ap.add_argument("--cache-spare", default="", help="cache_digest.py on the spare's node")
    ap.add_argument(
        "--predictions", required=True, help="the predictions.json beside the pre-registration"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv[1:])
    use_predictions(a.predictions)
    out = {**summarize(a), "preregistration": PREREGISTRATION}
    Path(a.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    v = out.get("verdict")
    print(f"{a.arm} trial {a.trial}: " + (v["line"] if v else "no control; figures only"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
