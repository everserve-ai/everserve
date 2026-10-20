# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The drains smoke's verdict: each pre-registered prediction scored from the run's record.

    drains_verdict.py RUN_DIR [--max-tokens N] > RUN_DIR/verdict.json

``RUN_DIR`` is what ``drains.sh`` wrote: ``reads/`` (every API answer, by phase and action),
``actions.jsonl``, ``setup/pin_guards_a.txt`` (the node-a guards, from setup) and ``collect/<node>/
out/`` (every node's logs and the drill's result). Each prediction of
``bench/runs/2026-10-13-run3-drains-smoke/PREREGISTRATION.md`` is a function of these records;
nothing is read from memory or judged by hand. A cross-node order is read on the nodes' own
clocks with :data:`CLOCK_TOLERANCE_S` of slack, which the record's clock offsets bound.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CLOCK_TOLERANCE_S = 1.0
"""Slack for an order read across two nodes' clocks; the run's clock offsets are far inside it."""
RELEASE_BOUND_S = 2.0
"""A released worker is killed within this: the kill and the reaping of its process."""
COMMIT_AFTER_DEADLINE_S = 20.0
"""A drained node's change commits within this of its deadline: the change's own time."""
GUARD_TESTS = (
    "test_only_the_head_registers_as_the_engine_s_head_and_says_goodbye[EverserveExecutor]",
    "test_only_the_head_registers_as_the_engine_s_head_and_says_goodbye[NodeExecutor]",
    "test_a_change_moves_the_spare_s_row_in_and_clears_the_rows_that_left",
    "test_a_position_moved_to_another_node_has_left_not_been_taken_here",
)
DRAINED = {"b": 1, "c": 2}
"""Each drained node and the rank it serves."""
SPARE_NODE = {1: "node-d", 2: "node-e"}
"""Each drained rank's spare's node."""
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ISO = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z")
PY_TIME = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) ")


class Record:
    """A drains smoke's run directory, read lazily."""

    def __init__(self, root: Path, max_tokens: int) -> None:
        """Read the record under ``root``; ``max_tokens`` is the sessions' length."""
        self.root = root
        self.max_tokens = max_tokens

    def text(self, rel: str) -> str:
        """A file's text, empty when absent."""
        p = self.root / rel
        return p.read_text(errors="replace") if p.exists() else ""

    def json(self, rel: str) -> Any:
        """A file's JSON, None when absent or unreadable."""
        try:
            return json.loads(self.text(rel))
        except ValueError:
            return None

    def log(self, node: str, name: str) -> list[str]:
        """A node's log lines, colour codes removed."""
        return [ANSI.sub("", line) for line in self.text(f"collect/{node}/out/{name}").splitlines()]

    def answer(self, action: str) -> tuple[int | None, Any]:
        """An API action's HTTP status and its JSON body (or its text when not JSON)."""
        body, status = [], None
        for line in self.text(f"reads/{action}.txt").splitlines():
            m = re.fullmatch(r"HTTP (\d{3})", line.strip())
            if m:
                status = int(m.group(1))
            else:
                body.append(line)
        text = "\n".join(body).strip()
        try:
            return status, json.loads(text)
        except ValueError:
            return status, text

    def actions(self) -> list[dict]:
        """Every action the driver took, in order."""
        return [json.loads(x) for x in self.text("actions.jsonl").splitlines() if x.strip()]


def when(line: str) -> float | None:
    """A log line's time, Unix seconds: a tracing line's ISO stamp or a Python logging stamp."""
    m = ISO.search(line)
    if m:
        return datetime.fromisoformat(m.group(1)).replace(tzinfo=UTC).timestamp()
    m = PY_TIME.match(line)
    if m:
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        return t.timestamp() + int(m.group(2)) / 1000
    return None


def last_launch(lines: list[str]) -> list[str]:
    """The lines after the last launch nodes.sh marked in a log: the run's own start.

    Not the warm-up's, nor an earlier start's.
    """
    marks = [i for i, x in enumerate(lines) if x.startswith("==== ") and x.endswith(" launch")]
    return lines[marks[-1] + 1 :] if marks else lines


def first(lines: list[str], pattern: str) -> tuple[float | None, str | None]:
    """The first line matching ``pattern`` and its time."""
    rx = re.compile(pattern)
    for line in lines:
        if rx.search(line):
            return when(line), line
    return None, None


def metric(text: str, name: str, **labels: str) -> float:
    """A metric's value with exactly these labels among its own; 0 when it is absent."""
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        m = re.fullmatch(rf"{re.escape(name)}(?:\{{(.*)\}})?\s+(\S+)", line.strip())
        if not m:
            continue
        have = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1) or ""))
        if all(have.get(k) == v for k, v in labels.items()):
            return float(m.group(2))
    return 0.0


def status(r: Record, phase: str) -> dict:
    """The engine's readiness at ``phase``: the ``.status`` of its ``/v1/engines/rig`` read."""
    return (r.json(f"reads/{phase}_readiness.json") or {}).get("status") or {}


def condition(obj_status: Any, kind: str) -> dict | None:
    """The condition of ``kind`` in an object's status (an engine's or a node's)."""
    for c in (obj_status or {}).get("conditions") or []:
        if c.get("type") == kind:
            return c
    return None


def node_entry(nodes: Any, node: str) -> dict:
    """A node's object in a ``/v1/nodes`` read (a list: its ``items``)."""
    return next(
        (
            n
            for n in (nodes or {}).get("items") or []
            if (n.get("metadata") or {}).get("name") == node
        ),
        {},
    )


def draining(node: Any) -> str | None:
    """A node's drain as its ``Draining`` condition says it; None when it is not draining.

    ``Scheduled`` (its notices went out), ``Waiting`` (its turn has not come) or ``Blocked``.
    """
    c = condition((node or {}).get("status"), "Draining") or {}
    return c.get("reason") if c.get("status") == "True" else None


def deadline_of(node: Any) -> int | None:
    """The drain's deadline a node's spec holds, Unix milliseconds; None without a drain."""
    return (((node or {}).get("spec") or {}).get("drain") or {}).get("deadlineMs")


def unschedulable(node: Any) -> bool | None:
    """Whether a node's spec cordons it."""
    return ((node or {}).get("spec") or {}).get("unschedulable")


def result(ok: bool, why: str) -> dict:
    """One prediction's score."""
    return {"pass": bool(ok), "why": why}


# ------------------------------------------------------------------------- the head fix


def h1(r: Record) -> dict:
    """The node-a guards ran every head and release test, none skipped or failed."""
    text = r.text("setup/pin_guards_a.txt")
    missing = [
        t for t in GUARD_TESTS if f"PASSED adapters/vllm/tests/test_node.py::{t}" not in text
    ]
    bad = re.findall(r"\b(\d+) (skipped|failed|errors?)\b", text)
    return result(
        bool(text) and not missing and not bad, f"missing {missing}, skipped/failed {bad}"
    )


def h2(r: Record) -> dict:
    """Heads only on node-a, one per start; node-b and node-c register as clients."""
    heads = {n: sum("head of rig registered" in x for x in r.log(n, "daemon.log")) for n in "abc"}
    clients = {
        n: sum("client of rig registered" in x for x in r.log(n, "daemon.log")) for n in "bc"
    }
    ok = heads["a"] >= 1 and heads["b"] == heads["c"] == 0 and all(clients.values())
    return result(ok, f"head registrations {heads}, client registrations {clients}")


def h3(r: Record) -> dict:
    """No second head: none in readiness, none counted."""
    conflicts = {p: status(r, p).get("headConflict") for p in ("registered", "final")}
    counted = metric(r.text("reads/final_metrics.txt"), "everserve_head_conflicts_total")
    return result(
        all(v is None for v in conflicts.values()) and counted == 0,
        f"headConflict {conflicts}, everserve_head_conflicts_total {counted}",
    )


# ------------------------------------------------------------------------- released workers


def released(r: Record, node: str) -> dict | None:
    """The node role's record of its released worker, from its log."""
    rank = DRAINED[node]
    _, line = first(last_launch(r.log(node, "headless.log")), rf"released worker {rank} ended: ")
    if line is None:
        return None
    return ast.literal_eval(line.split("ended: ", 1)[1])


def r1(r: Record) -> dict:
    """Each drained node released its worker at its boundary, not ended it as overdue."""
    got = {n: released(r, n) for n in DRAINED}
    ok = all(v is not None and not v.get("overdue") for v in got.values())
    return result(ok, f"released {got}")


def r2(r: Record) -> dict:
    """Each released worker was killed, in time, and no engine process remains on its node."""
    got = {n: released(r, n) or {} for n in DRAINED}
    procs = {n: r.text(f"reads/final_procs_{n}.txt").strip() for n in DRAINED}
    ok = all(
        v.get("how") == "killed" and (v.get("end_s") or 1e9) < RELEASE_BOUND_S for v in got.values()
    ) and not any(procs.values())
    return result(ok, f"ended {got}, remaining {procs}")


def r3(r: Record) -> dict:
    """A released worker's exit is no loss, and its position is held on its spare's node."""
    reported = {
        n: [x for x in last_launch(r.log(n, "headless.log")) if f"worker {k} is exiting" in x]
        for n, k in DRAINED.items()
    }
    members = {m.get("rank"): m.get("node") for m in status(r, "final").get("members") or []}
    held = {k: members.get(k) for k in DRAINED.values()}
    ok = not any(reported.values()) and held == SPARE_NODE
    return result(ok, f"exits reported lost {reported}, held at the end {held}")


def r4(r: Record) -> dict:
    """Each drained node's process ended on its own, holding nothing, with status 0."""
    got = {}
    for n in DRAINED:
        lines = last_launch(r.log(n, "headless.log"))
        got[n] = (
            any("this node holds nothing of rig" in x for x in lines),
            next((x.rsplit(" ", 1)[1] for x in reversed(lines) if "node_main exited" in x), None),
        )
    return result(all(v == (True, "0") for v in got.values()), f"(holds nothing, exit) {got}")


def r5(r: Record) -> dict:
    """Every session ran to its length across both changes, in a drill that ended cleanly."""
    drill = r.json("collect/a/out/drains.json") or {}
    lengths = [len(t) for t in drill.get("tokens") or []]
    end = next((a for a in r.actions() if a["action"] == "drill_end"), {})
    ok = (
        len(lengths) > 0
        and all(n == r.max_tokens for n in lengths)
        and len(drill.get("changes") or []) == 2
        and end.get("rc") == 0
    )
    return result(
        ok,
        f"session lengths {lengths} of {r.max_tokens}, changes {len(drill.get('changes') or [])}, "
        f"drill exit {end.get('rc')}",
    )


# ------------------------------------------------------------------------- the Node resource


def n1(r: Record) -> dict:
    """The spare nodes' cordons show in every read and are undone by the uncordons."""
    cordons = [r.answer(f"cordon_{n}") for n in "de"]
    uncordons = [r.answer(f"uncordon_{n}") for n in "de"]
    # each answer is the Node written (decision 0042)
    bodies = [b if isinstance(b, dict) else {} for _, b in cordons]
    ubodies = [b if isinstance(b, dict) else {} for _, b in uncordons]
    spare_nodes = set(SPARE_NODE.values())
    ready = status(r, "cordoned")
    after = status(r, "uncordoned")
    d_spares = [s.get("state") for s in ready.get("spares") or [] if s.get("node") in spare_nodes]
    d_after = [s.get("state") for s in after.get("spares") or [] if s.get("node") in spare_nodes]
    out_of_reach = [
        (n.get("outcome"), n.get("reclaimable"))
        for n in ready.get("nodes") or []
        if n.get("node") in ("node-b", "node-c")
    ]
    warm = [
        metric(r.text(f"reads/{p}_metrics.txt"), "everserve_spares", engine="rig", state="warm")
        for p in ("registered", "cordoned", "uncordoned")
    ]
    covered = condition(ready, "Covered") or {}
    covered_after = condition(status(r, "recovered"), "Covered") or {}
    checks = {
        "cordons 200": [c for c, _ in cordons] == [200, 200],
        "unschedulable": all(unschedulable(b) is True for b in bodies),
        # the API records who wrote a spec as "operator api: <the caller's name>"
        "written by": all(
            ((b.get("metadata") or {}).get("writtenBy") or "").startswith("operator api")
            for b in bodies
        ),
        "reason": all((b.get("spec") or {}).get("reason") for b in bodies),
        "/v1/nodes cordoned": all(
            unschedulable(node_entry(r.json("reads/cordoned_nodes.json"), n)) is True
            for n in spare_nodes
        ),
        "spares cordoned": d_spares == ["cordoned", "cordoned"],
        "rebuild, not reclaimable": out_of_reach == [("rebuild", False), ("rebuild", False)],
        "warm 2, 0, 2": warm == [2.0, 0.0, 2.0],
        "Covered lists ranks 1 and 2 while cordoned": covered.get("status") == "False"
        and covered.get("reason") == "LossUncovered"
        and "[1]" in (covered.get("message") or "")
        and "[2]" in (covered.get("message") or ""),
        "uncordons 200, generation 2": [c for c, _ in uncordons] == [200, 200]
        and all(
            unschedulable(b) is False and (b.get("metadata") or {}).get("generation") == 2
            for b in ubodies
        ),
        "spares back": len(d_after) == 2 and "cordoned" not in d_after,
        "/v1/nodes uncordoned": all(
            unschedulable(node_entry(r.json("reads/uncordoned_nodes.json"), n)) is False
            for n in spare_nodes
        ),
        "Covered lists only rank 0 once formed again": covered_after.get("reason")
        == "LossUncovered"
        and (covered_after.get("message") or "").endswith("[[0]]"),
    }
    return result(all(checks.values()), f"failed: {[k for k, v in checks.items() if not v]}")


def commit_time(r: Record, rank: int, after: float | None = None) -> float | None:
    """When the coordinator committed the change that took ``rank`` (after ``after``, if given)."""
    for line in r.log("a", "coord.log"):
        if re.search(rf"rig: membership committed at epoch \d+ .*lost \[{rank}\]", line):
            t = when(line)
            if t is not None and (after is None or t >= after):
                return t
    return None


def n2(r: Record) -> dict:
    """Node-c's drain waits its turn, and its notices follow node-b's commit."""
    code, body = r.answer("drain_c")
    body = body if isinstance(body, dict) else {}
    coord = r.log("a", "coord.log")
    nodes = r.json("reads/draining_nodes.json")
    b, c = node_entry(nodes, "node-b"), node_entry(nodes, "node-c")
    committed_b = commit_time(r, 1)
    t_release, _ = first(coord, r"rig: drain of node-c releases ranks \[2\]")
    t_notice, _ = first(coord, r"rig: notice for rank 2 at")
    waits = "2 positions would move at once, the policy allows 1"
    checks = {
        # the PUT answers the Node written, its drain waiting its turn (decision 0042)
        "200 waiting": code == 200
        and draining(body) == "Waiting"
        and waits in ((condition(body.get("status"), "Draining") or {}).get("message") or ""),
        "waits line": any(f"node-c: its drain waits: {waits}" in x for x in coord),
        "b scheduled, c waiting, both cordoned": (draining(b), draining(c))
        == ("Scheduled", "Waiting")
        and unschedulable(b) is True
        and unschedulable(c) is True,
        "c released after b's commit": committed_b is not None
        and t_release is not None
        and t_release >= committed_b,
        "c's notice after b's commit": committed_b is not None
        and t_notice is not None
        and t_notice >= committed_b,
    }
    return result(all(checks.values()), f"failed: {[k for k, v in checks.items() if not v]}")


def n3(r: Record) -> dict:
    """A second drain of node-b, with another deadline, is refused and moves nothing."""
    code, body = r.answer("drain_b_again")
    _, first_body = r.answer("drain_b")
    deadline = deadline_of(first_body)
    held = deadline_of(node_entry(r.json("reads/draining_nodes.json"), "node-b"))
    # a stale resourceVersion is a 409 too (Conflict): the refusal must be the drain's own
    ok = (
        code == 409
        and isinstance(body, dict)
        and body.get("reason") == "DrainUnderway"
        and deadline is not None
        and held == deadline
    )
    return result(ok, f"status {code}, body {str(body)[:160]!r}, deadline {deadline}, held {held}")


def n4(r: Record) -> dict:
    """Both stored drains survive the coordinator's restart, node-b's first and on time."""
    coord = r.log("a", "coord.log")
    # nodes.sh's spawn marks every launch in the log: the warm-up's, the run's, then the restart
    starts = [when(x) for x in coord if x.startswith("==== ") and x.endswith(" launch")]
    restart = starts[-1] if len(starts) > 1 else None
    after = [x for x in coord if restart is not None and (when(x) or 0) >= restart]
    _, first_body = r.answer("drain_b")
    deadline = deadline_of(first_body)
    t_b = commit_time(r, 1, restart)
    t_c = commit_time(r, 2, restart)
    t_c_start, _ = first(after, r"rig: drain of node-c releases ranks \[2\]")
    drains = [a["action"] for a in r.actions() if a["action"].startswith("drain")]
    checks = {
        "restarted": restart is not None,
        "store read": any("spec store: file" in x for x in after),
        "b committed after the restart, at its deadline": t_b is not None
        and deadline is not None
        and t_b <= deadline / 1000 + COMMIT_AFTER_DEADLINE_S,
        "c committed after b": t_b is not None and t_c is not None and t_c > t_b,
        "c started after b's commit": t_b is not None
        and t_c_start is not None
        and t_c_start >= t_b,
        "no drain asked again": drains == ["drain_b", "drain_c", "drain_b_again"],
    }
    return result(all(checks.values()), f"failed: {[k for k, v in checks.items() if not v]}")


def n5(r: Record) -> dict:
    """The engine was never down: in no read, and not in the coordinator's log once it served.

    A read can miss it: a restarted coordinator rebuilds the engine from the hellos, and an engine
    whose head is gone is not in them. A head that stopped in order (its goodbye's cause
    ``stopped``) is the run's own end, when the sessions are done, not the engine down.
    """
    phases = sorted(
        p.name.removesuffix("_readiness.json") for p in (r.root / "reads").glob("*_readiness.json")
    )
    serving = next((a["t"] for a in r.actions() if a["action"] == "read_registered"), None)
    logged = [
        x
        for x in r.log("a", "coord.log")
        if ("rig: down:" in x or "the engine is down" in x)
        and "its head stopped" not in x
        and "the engine is down (stopped)" not in x
        and serving is not None
        and (when(x) or 0) >= serving
    ]
    down = {}
    for p in phases:
        d = status(r, p).get("down")
        flag = metric(r.text(f"reads/{p}_metrics.txt"), "everserve_engine_down", engine="rig")
        ended = isinstance(d, dict) and d.get("cause") == "stopped"
        down[p] = (None, 0.0) if ended else (d, flag)
    ok = bool(phases) and all(v == (None, 0.0) for v in down.values()) and not logged
    return result(
        ok, f"(down, everserve_engine_down) by read {down}; logged {[x[:160] for x in logged]}"
    )


def set_covered(readiness: Any, rank: int, epoch: int = 0) -> bool:
    """Whether ``rank``'s loss has a covered set in an engine's status, at ``epoch`` or later."""
    cover = (readiness or {}).get("cover") or {}
    return cover.get("epoch", -1) >= epoch and any(
        s.get("purpose") == f"REPLACE of ranks [{rank}]" and s.get("covered")
        for s in cover.get("sets") or []
    )


def c1(r: Record) -> dict:
    """Both losses covered before any drain, and again once the uncordon's plan formed."""
    reads = {p: status(r, p) for p in ("registered", "recovered")}
    got = {p: (set_covered(v, 1), set_covered(v, 2)) for p, v in reads.items()}
    return result(all(v == (True, True) for v in got.values()), f"(rank 1, rank 2) covered {got}")


def c2(r: Record) -> dict:
    """After node-b's commit, rank 2's set is covered at epoch 1 before node-c's deadline."""
    at = next((a for a in r.actions() if a["action"] == "covered_after_b"), {})
    _, body = r.answer("drain_c")
    deadline = deadline_of(body)
    read = set_covered(status(r, "covered_after_b"), 2, 1)
    ok = (
        at.get("rc") == 0
        and read
        and deadline is not None
        and at.get("t", float("inf")) <= deadline / 1000
    )
    return result(ok, f"covered {at}, read {read}, node-c's deadline {deadline}")


PREDICTIONS: dict[str, Callable[[Record], dict]] = {
    "H1": h1,
    "H2": h2,
    "H3": h3,
    "R1": r1,
    "R2": r2,
    "R3": r3,
    "R4": r4,
    "R5": r5,
    "N1": n1,
    "N2": n2,
    "N3": n3,
    "N4": n4,
    "N5": n5,
    "C1": c1,
    "C2": c2,
}


def verdict(r: Record) -> dict:
    """Every prediction scored, and the run's verdict: pass only when every one passes."""
    scored = {}
    for name, f in PREDICTIONS.items():
        try:
            scored[name] = f(r)
        except Exception as e:  # noqa: BLE001 - a record that cannot be read fails its prediction
            scored[name] = result(False, f"unreadable: {type(e).__name__}: {e}")
    return {
        "verdict": "pass" if all(p["pass"] for p in scored.values()) else "fail",
        "predictions": scored,
    }


def main(argv: list[str]) -> int:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--max-tokens", type=int, default=7000)
    a = ap.parse_args(argv[1:])
    json.dump(verdict(Record(a.run_dir, a.max_tokens)), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
