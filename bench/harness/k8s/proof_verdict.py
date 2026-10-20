# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What the coordinator's record of the proof's engine says, read from its Engine object (0042).

``covered <engine.json> <rank>``: exit 0 when a covered set answers the loss of ``rank`` alone
(its REPLACE set formed); the driver takes the victim only then (taken after the spare's set is
planned and before it forms, the change is refused). The Covered condition is
not the test: it is False while any loss is uncovered, the head's included, which has no spare.

``replaced <engine.json> <rank> [<epoch>]``: exit 0 when the last committed change was a REPLACE
that took ``rank``, at ``epoch`` where one is given. Streams that finish are not the proof on
their own: they finish while a taken pod serves through its grace period and the coordinator
refuses the change.

``engine-covered <engine.json> <rank>``: exit 0 when the engine's Covered condition is True. A
loss on the head's node restarts the engine and leaves Covered's list, so an engine whose other
ranks all have formed sets reads Covered True (``rank`` is unused, for the common form).

``holder <engine.json> <rank>``: print the node the record places ``rank`` on; exit 1 if none.
The driver takes the pod on that node: after a REPLACE the rank is the spare's, whatever pod logs
say.

``serving <engine.json>``: exit 0 when the engine serves by its record: Serving True, every
position held, a head registered, and its members running the same software.

Each prints the line it decided on.
"""

from __future__ import annotations

import json
import sys


def _status(path: str) -> dict:
    with open(path) as f:
        return json.load(f).get("status") or {}


def covered(status: dict, rank: int) -> tuple[bool, str]:
    """Whether a formed set answers the loss of ``rank`` alone, and each such set's progress."""
    purpose = f"REPLACE of ranks [{rank}]"
    cover = status.get("cover") or {}
    # the plan the sets stand in: a loss after a change goes through a set of the change's epoch
    plan = f"plan {cover.get('generation')} at epoch {cover.get('epoch')}"
    sets = [s for s in cover.get("sets") or [] if s.get("purpose") == purpose]
    if not sets:
        return False, f"{plan}: no set planned for the loss of rank {rank}"
    line = "; ".join(
        f"{s.get('set')}: {s.get('formed')} of {s.get('groups')} groups formed" for s in sets
    )
    return any(s.get("covered") for s in sets), f"{plan}: {line}"


def replaced(status: dict, rank: int, epoch: int | None = None) -> tuple[bool, str]:
    """Whether the last committed change replaced ``rank`` (at ``epoch``), and what it was."""
    last = status.get("lastChange")
    if not last:
        return False, "no change committed"
    line = (
        f"epoch {last.get('epoch')}: {last.get('outcome')} of ranks {last.get('lost')}"
        # the coordinator's own words: "through covered set <set>", or how it formed in the change
        f" {last.get('cover') or 'with cover not planned'}"
    )
    ok = last.get("outcome") == "replace" and rank in (last.get("lost") or [])
    if epoch is not None and last.get("epoch") != epoch:
        return False, f"{line}; epoch {epoch} expected"
    return ok, line


def engine_covered(status: dict, rank: int) -> tuple[bool, str]:
    """Whether the engine's Covered condition is True, and the condition's own words."""
    for c in status.get("conditions") or []:
        if c.get("type") == "Covered":
            return c.get("status") == "True", f"Covered {c.get('status')} ({c.get('reason')}): " + (
                c.get("message") or ""
            )
    return False, "no Covered condition"


def holder(status: dict, rank: int) -> tuple[bool, str]:
    """The node the record's members place ``rank`` on."""
    for m in status.get("members") or []:
        if m.get("rank") == rank and m.get("node"):
            return True, m["node"]
    return False, f"no member holds rank {rank}"


def serving(status: dict) -> tuple[bool, str]:
    """Whether the engine serves, its members alike.

    Serving True, every position held, a head registered, and no Degraded MembersDiffer (the
    members running different software).

    A head's registration alone, or a frontend answering a request, does not say the engine
    serves: a record read 1 s after a warm-up holds no member yet.
    """
    conditions = status.get("conditions") or []
    cond = next((c for c in conditions if c.get("type") == "Serving"), {})
    differ = next(
        (
            c
            for c in conditions
            if c.get("type") == "Degraded"
            and c.get("status") == "True"
            and c.get("reason") == "MembersDiffer"
        ),
        None,
    )
    members = status.get("members") or []
    world = status.get("worldSize") or 0
    head = (status.get("head") or {}).get("node")
    line = (
        f"Serving {cond.get('status')} ({cond.get('reason')}); {len(members)} of {world} "
        f"positions held; head on {head}"
    )
    if differ is not None:
        # members that differ serve, and no spare can then be held to them all: the run stops at
        # bring-up, naming the field (decision 0034's check across members)
        return False, f"{line}; MembersDiffer: {differ.get('message')}"
    return cond.get("status") == "True" and world > 0 and len(members) == world and bool(head), line


def inside(load: dict, taken_at: float) -> tuple[bool, str]:
    """Whether every stream was under way when the node was taken at ``taken_at``.

    A stream's first token came before the loss and its last after it, so the loss fell inside
    it; a stream that had finished before, or started after, shows nothing about the loss. Also
    whether every stream's worst gap fell at or after the loss, which a REPLACE under load
    leaves as its one mark.
    """
    streams = load.get("per_stream") or {}
    if not streams:
        return False, "no streams in the load record"
    outside = []
    gaps_before = []
    for i, r in sorted(streams.items(), key=lambda kv: int(kv[0])):
        first, last = r.get("first"), r.get("last")
        if first is None or last is None or not (first <= taken_at <= last):
            outside.append(f"{i}: first {first}, last {last}")
        elif r.get("gap_at") is not None and r["gap_at"] + r.get("gap_s", 0) < taken_at:
            gaps_before.append(f"{i}: gap {r.get('gap_s')} s at {r['gap_at']}")
    within = len(streams) - len(outside)
    line = f"the node was taken at {taken_at:.3f}, inside {within} of {len(streams)} streams"
    if outside:
        return False, f"{line}; outside: {'; '.join(outside)}"
    if gaps_before:
        return False, f"{line}; a worst gap before the loss: {'; '.join(gaps_before)}"
    return True, line


def main(argv: list[str]) -> int:
    """Decide a check on a saved Engine object; 0 when it holds.

    The checks: ``covered``, ``replaced``, ``holder``, ``engine-covered``, ``serving``, and
    ``inside <load.json> <taken-at>`` on a saved load record.
    """
    if argv[1:2] == ["inside"] and len(argv) == 4:
        try:
            with open(argv[2]) as f:
                load = json.load(f)
            taken_at = float(argv[3])
        except (OSError, ValueError) as e:
            print(f"no load record: {e}")
            return 1
        ok, line = inside(load, taken_at)
        print(line)
        return 0 if ok else 1
    if argv[1:2] == ["serving"] and len(argv) == 3:
        try:
            status = _status(argv[2])
        except (OSError, ValueError) as e:
            print(f"no record: {e}")
            return 1
        ok, line = serving(status)
        print(line)
        return 0 if ok else 1
    if not (
        (len(argv) == 4 and argv[1] in ("covered", "replaced", "holder", "engine-covered"))
        or (len(argv) == 5 and argv[1] == "replaced")
    ):
        print(
            "usage: proof_verdict.py covered|holder|engine-covered <engine.json> <rank>\n"
            "       proof_verdict.py replaced <engine.json> <rank> [<epoch>]\n"
            "       proof_verdict.py serving <engine.json>\n"
            "       proof_verdict.py inside <load.json> <taken-at>",
            file=sys.stderr,
        )
        return 2
    try:
        status = _status(argv[2])
    except (OSError, ValueError) as e:
        print(f"no record: {e}")
        return 1
    rank = int(argv[3])
    if argv[1] == "covered":
        ok, line = covered(status, rank)
    elif argv[1] == "holder":
        ok, line = holder(status, rank)
    elif argv[1] == "engine-covered":
        ok, line = engine_covered(status, rank)
    else:
        ok, line = replaced(status, rank, int(argv[4]) if len(argv) == 5 else None)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
